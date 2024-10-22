//emacs settings:-*- mode: c++; c-basic-offset: 2; indent-tabs-mode: nil -*-

/***************************************************
 * TODO:  check Matacq                             *
 *        add DTT                                  *
 *        completion of partial output file        * 
 ***************************************************/

#include "CalibCalorimetry/EcalLaserSorting/interface/LaserSorter.h"
#include "CalibCalorimetry/EcalLaserSorting/src/Majority.h"

#include <iostream>
#include <iomanip>
#include <algorithm>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <cerrno>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Provenance/interface/EventID.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/Provenance/interface/LuminosityBlockID.h"
#include "DataFormats/Provenance/interface/Timestamp.h"
#include "EventFilter/EcalRawToDigi/interface/MatacqRawEvent.h"

using namespace std;

const int LaserSorter::ecalDccFedIdMin_ = 601;
const int LaserSorter::ecalDccFedIdMax_ = 654;

const size_t LaserSorter::OutStreamRecord::indexReserve_ = 2000;

static const struct timeval nullTime = {0, 0};

static const char* const detailedTrigNames[] = {
    "Inv0",  //000
    "Inv1",  //001
    "Inv2",  //010
    "Inv3",  //011
    "Las",   //100
    "Led",   //101
    "TP",    //110
    "Ped"    //111
};

static const char* const colorNames[] = {"Blue", "Green", "Red", "IR"};

const LaserSorter::stats_t LaserSorter::stats_init = {0, 0, 0, 0, 0};
const int LaserSorter::indexOffset32_ = 1;

static std::string now() {
  struct timeval t;
  gettimeofday(&t, nullptr);

  char buf[256];
  strftime(buf, sizeof(buf), "%F %R %S s", localtime(&t.tv_sec));
  buf[sizeof(buf) - 1] = 0;

  stringstream buf2;
  buf2 << buf << " " << ((t.tv_usec + 500) / 1000) << " ms";

  return buf2.str();
}

LaserSorter::LaserSorter(const edm::ParameterSet& pset)
    : lumiBlock_(0),
      lumiBlockPrev_(0),
      formatVersion_(5),
      outputDir_(pset.getParameter<std::string>("outputDir")),
      fedSubDirs_(pset.getParameter<std::vector<std::string> >("fedSubDirs")),
      timeLogFile_(pset.getUntrackedParameter<std::string>("timeLogFile", "")),
      disableOutput_(pset.getUntrackedParameter<bool>("disableOutput", false)),
      runNumber_(0),
      outputListFile_(pset.getUntrackedParameter<string>("outputListFile", "")),
      doOutputList_(false),
      verbosity_(pset.getUntrackedParameter<int>("verbosity", 0)),
      iNoFullReadoutDccError_(0),
      maxFullReadoutDccError_(pset.getParameter<int>("maxFullReadoutDccError")),
      iNoEcalDataMess_(0),
      maxNoEcalDataMess_(pset.getParameter<int>("maxNoEcalDataMess")),
      lumiBlockSpan_(pset.getParameter<int>("lumiBlockSpan")),
      fedRawDataCollectionTag_(pset.getParameter<edm::InputTag>("fedRawDataCollectionTag")),
      stats_(stats_init),
      overWriteLumiBlockId_(pset.getParameter<bool>("overWriteLumiBlockId")),
      orbitCountInALumiBlock_(pset.getParameter<int>("orbitCountInALumiBlock")),
      orbit_(-1),
      orbitZeroTime_(nullTime) {
  gettimeofday(&timer_, nullptr);
  logFile_.open("eventSelect.log", ios::app | ios::out);

  const unsigned nEcalFeds = 54;
  if (fedSubDirs_.size() != nEcalFeds + 1) {
    throw cms::Exception("LaserSorter") << "Configuration error: "
                                        << "fedSubDirs parameter must be a vector "
                                        << " of " << nEcalFeds << " strings"
                                        << " (subdirectory for unknown triggered FED followed by "
                                           "subdirectories for FED ID 601 "
                                           "to FED ID 654 in increasing FED ID order)";
  }

  fedRawDataCollectionToken_ = consumes<FEDRawDataCollection>(fedRawDataCollectionTag_);

  if (!outputListFile_.empty()) {
    outputList_.open(outputListFile_.c_str(), ios::app);
    if (outputList_.bad()) {
      throw cms::Exception("FileOpen") << "Failed to open file '" << outputListFile_
                                       << "' for logging of output file path list.";
    }
    doOutputList_ = true;
  }

  if (!timeLogFile_.empty()) {
    timeLog_.open(timeLogFile_.c_str());
    if (timeLog_.fail()) {
      cout << "[LaserSorter " << now() << "] "
           << "Failed to open file " << timeLogFile_ << " to log timing.\n";
      timing_ = false;
    } else {
      timing_ = true;
    }
  }

  struct stat fileStat;
  if (0 == stat(outputDir_.c_str(), &fileStat)) {
    if (!S_ISDIR(fileStat.st_mode)) {
      throw cms::Exception("[LaserSorter]") << "File " << outputDir_ << " exists but is not a directory "
                                            << " as expected.";
    }
  } else {  //directory does not exists, let's try to create it
    if (0 != mkdir(outputDir_.c_str(), 0755)) {
      throw cms::Exception("[LaserSorter]") << "Failed to create directory " << outputDir_ << " for writing data.";
    }
  }

  logFile_ << "# "
              "run\t"
              "LB\t"
              "event\t"
              "trigType\t"
              "FED\t"
              "side\t"
              "LB out\t"
              "Written\t"
              "ECAL data\n";
}

LaserSorter::~LaserSorter() {
  logFile_ << "Summary. Event count: " << stats_.nRead << " processed, " << stats_.nWritten << " written, "
           << stats_.nInvalidDccStrict << " with errors in DCC ID values, " << stats_.nInvalidDccWeak
           << " with unusable DCC ID values, " << stats_.nRestoredDcc << " restored DCC ID based on DCC block size\n";
}

// ------------ method called to analyze the data  ------------
void LaserSorter::analyze(const edm::Event& event, const edm::EventSetup& es) {
  if (timing_) {
    timeval t;
    gettimeofday(&t, nullptr);
    timeLog_ << t.tv_sec << "." << setfill('0') << setw(3) << (t.tv_usec + 500) / 1000 << setfill(' ') << "\t"
             << (t.tv_usec - timer_.tv_usec) * 1. + (t.tv_sec - timer_.tv_sec) * 1.e6 << "\t";
    timer_ = t;
  }

  ++stats_.nRead;

  if (event.id().run() != runNumber_) {  //run changed or first event
    //for a new run, starts with a new output stream set.
    closeAllStreams();
    runNumber_ = event.id().run();
    iNoFullReadoutDccError_ = 0;
    iNoEcalDataMess_ = 0;
    lumiBlockPrev_ = 0;
    lumiBlock_ = 0;
  }

  edm::Handle<FEDRawDataCollection> rawdata;
  event.getByToken(fedRawDataCollectionToken_, rawdata);

  //orbit number
  //FIXME: orbit from edm Event is currently wrong. Forcing to use of CMS orbit until
  //it is fixed. See https://hypernews.cern.ch/HyperNews/CMS/get/commissioning/5343/2.html
#if 0
  orbit_ = event.orbitNumber();
#else
  orbit_ = -1;
#endif

  //  std::cerr << "Orbit ID CMS, ECAL: " << orbit_ << "\t" << getOrbitFromDcc(rawdata) << "\n";

  if (orbit_ < 0) {  //For local run CMSSW failed to find the orbit number
    //    cout << "Look for orbit from DCC headers....\n";
    orbit_ = getOrbitFromDcc(rawdata);
  }

  //The "detailed trigger type DCC field content:
  double dttProba = 0;
  int dtt = getDetailedTriggerType(rawdata, &dttProba);

  if (overWriteLumiBlockId_) {
    edm::LuminosityBlockNumber_t lb = lumiBlock_;
    lumiBlock_ = orbit_ / orbitCountInALumiBlock_;
    if (lb != lumiBlock_) {
      std::cout << "[LaserSorter " << now() << "] Overwrite LB mode. LB number changed from: " << lb << " to "
                << lumiBlock_ << "\n";
    }
  } else {
    edm::LuminosityBlockNumber_t lb = lumiBlock_;
    lumiBlock_ = event.luminosityBlock();
    if (lb != lumiBlock_) {
      std::cout << "[LaserSorter " << now() << "] Standard LB mode. LB number changed from: " << lb << " to "
                << lumiBlock_ << "\n";
    }
  }

  detailedTrigType_ = dtt;
  const int trigType = (detailedTrigType_ >> 8) & 0x7;
  const int color = (detailedTrigType_ >> 6) & 0x3;
  const int dccId = (detailedTrigType_ >> 0) & 0x3F;
  int triggeredFedId = (detailedTrigType_ == -2) ? -1 : (600 + dccId);
  const int side = (detailedTrigType_ >> 11) & 0x1;
  //monitoring region extended id:
  //  const int lme = dcc2Lme(dccId, side);

  if (detailedTrigType_ > -2) {
    if (dttProba < 1. || triggeredFedId < ecalDccFedIdMin_ || triggeredFedId > ecalDccFedIdMax_) {
      ++stats_.nInvalidDccStrict;
    }

    if (triggeredFedId < ecalDccFedIdMin_ || triggeredFedId > ecalDccFedIdMax_) {
      if (verbosity_)
        cout << "[LaserSorter " << now() << "] "
             << "DCC ID (" << dccId << ") found in trigger type is out of range.";
      ++stats_.nInvalidDccWeak;
      vector<int> ids = getFullyReadoutDccs(*rawdata);
      if (ids.empty()) {
        if (verbosity_ && iNoFullReadoutDccError_ < maxFullReadoutDccError_) {
          cout << " No fully read-out DCC found\n";
          ++iNoFullReadoutDccError_;
        }
      } else if (ids.size() == 1) {
        triggeredFedId = ids[0];
        if (verbosity_)
          cout << " ID guessed from DCC payloads\n";
        ++stats_.nRestoredDcc;
      } else {  //ids.size()>1
        if (verbosity_) {
          cout << " Several fully read-out Dccs:";
          for (unsigned i = 0; i < ids.size(); ++i)
            cout << " " << ids[i];
          cout << "\n";
        }
      }
    }

    if (verbosity_ > 1)
      cout << "\n----------------------------------------------------------------------\n"
           << "Event id: "
           << " " << event.id() << "\n"
           << "Lumin block: " << lumiBlock_ << "\n"
           << "TrigType: " << detailedTrigNames[trigType & 0x7] << " Color: " << colorNames[color & 0x3]
           << " FED: " << triggeredFedId << " side:" << side << "\n"
           << "\n----------------------------------------------------------------------\n";

  } else {  //NO ECAL DATA
    if (verbosity_ > 1)
      cout << "\n----------------------------------------------------------------------\n"
           << "Event id: "
           << " " << event.id() << "\n"
           << "Lumin block: " << lumiBlock_ << "\n"
           << "No ECAL data\n"
           << "\n----------------------------------------------------------------------\n";
  }

  logFile_ << event.id().run() << "\t" << lumiBlock_ << "\t" << event.id().event() << "\t" << trigType << "\t"
           << triggeredFedId << "\t" << side;

  bool written = false;
  int assignedLB = -1;

  if (lumiBlock_ != lumiBlockPrev_) {
    //lumi block change => need for stream garbage collection
    const int lb = lumiBlock_;
    closeOldStreams(lb);
    int minLumi = lumiBlock_ - lumiBlockSpan_;
    int maxLumi = lumiBlock_ + lumiBlockSpan_;
    for (int lb1 = minLumi; lb1 <= maxLumi; ++lb1) {
      restoreStreamsOfLumiBlock(lb1);
    }
  }

  //     if(lumiBlock_ < lumiBlockPrev_){
  //       throw cms::Exception("LaserSorter")
  //         << "Process event has a lumi block (" << lumiBlock_ << ")"
  //         << "older than previous one (" << lumiBlockPrev_ << "). "
  //         << "This can be due by wrong input file ordering or bad luminosity "
  //         << "block indication is the event header. "
  //         << "Event cannot be processed";
  //     }

  if (disableOutput_) {
    /* NO OP*/
  } else {
    const auto& out = getStream(triggeredFedId, lumiBlock_);

    if (out != nullptr) {
      assignedLB = out->startingLumiBlock();
      if (out->excludedOrbit().find(orbit_) == out->excludedOrbit().end()) {
        if (verbosity_ > 1)
          cout << "[LaserSorter " << now() << "] "
               << "Writing out event from FED " << triggeredFedId << " LB " << lumiBlock_ << " orbit " << orbit_
               << "\n";
        int dtt = (detailedTrigType_ >= 0) ? detailedTrigType_ : -1;  //shall we use -1 or 0 for undefined value?
        written = written || writeEvent(*out, event, dtt, *rawdata);
        ++stats_.nWritten;
      } else {
        if (verbosity_)
          cout << "[LaserSorter " << now() << "] "
               << "File " << out->finalFileName() << " "
               << "already contains calibration event from FED " << triggeredFedId << ", LB = " << lumiBlock_
               << " with orbit ID " << orbit_ << ". Event skipped.\n";
      }
    }
  }
  lumiBlockPrev_ = lumiBlock_;

  logFile_ << "\t";
  if (assignedLB >= 0)
    logFile_ << assignedLB;
  else
    logFile_ << "-";
  logFile_ << "\t" << (written ? "Y" : "N") << "\n";
  logFile_ << "\t" << (detailedTrigType_ == -2 ? "N" : "Y") << "\n";

  if (timing_) {
    timeval t;
    gettimeofday(&t, nullptr);
    timeLog_ << (t.tv_usec - timer_.tv_usec) * 1. + (t.tv_sec - timer_.tv_sec) * 1.e6 << "\n";
    timer_ = t;
  }
}

int LaserSorter::dcc2Lme(int dcc, int side) {
  int fedid = (dcc % 600) + 600;  //to handle both FED and DCC id.
  vector<int> lmes;
  // EE -
  if (fedid <= 609) {
    if (fedid <= 607) {
      lmes.push_back(fedid - 601 + 83);
    } else if (fedid == 608) {
      lmes.push_back(90);
      lmes.push_back(91);
    } else if (fedid == 609) {
      lmes.push_back(92);
    }
  }  //EB
  else if (fedid >= 610 && fedid <= 645) {
    lmes.push_back(2 * (fedid - 610) + 1);
    lmes.push_back(lmes[0] + 1);
  }  // EE+
  else if (fedid >= 646) {
    if (fedid <= 652) {
      lmes.push_back(fedid - 646 + 73);
    } else if (fedid == 653) {
      lmes.push_back(80);
      lmes.push_back(81);
    } else if (fedid == 654) {
      lmes.push_back(82);
    }
  }
  return lmes.empty() ? -1 : lmes[min(lmes.size(), (size_t)side)];
}

int LaserSorter::getOrbitFromDcc(const edm::Handle<FEDRawDataCollection>& rawdata) const {
  const int orbit32 = 6;
  for (int id = ecalDccFedIdMin_; id <= ecalDccFedIdMax_; ++id) {
    if (!FEDNumbering::inRange(id))
      continue;
    const FEDRawData& data = rawdata->FEDData(id);
    if (data.size() >= 4 * (orbit32 + 1)) {
      const uint32_t* pData32 = (const uint32_t*)data.data();
      //      cout << "Found a DCC header: "
      //           << pData32[0] << " "
      //           << pData32[1] << " "
      //           << pData32[2] << " "
      //           << pData32[3] << " "
      //           << pData32[4] << " "
      //           << pData32[5] << " "
      //           << pData32[6] << " "
      //           << "\n";
      return pData32[orbit32];
    }
  }
  return -1;
}

int LaserSorter::getDetailedTriggerType(const edm::Handle<FEDRawDataCollection>& rawdata, double* proba) {
  Majority<int> stat;
  bool ecalData = false;
  for (int id = ecalDccFedIdMin_; id <= ecalDccFedIdMax_; ++id) {
    if (!FEDNumbering::inRange(id))
      continue;
    const FEDRawData& data = rawdata->FEDData(id);
    const int detailedTrigger32 = 5;
    if (verbosity_ > 3)
      cout << "[LaserSorter " << now() << "] "
           << "FED " << id << " data size: " << data.size() << "\n";
    if (data.size() >= 4 * (detailedTrigger32 + 1)) {
      ecalData = true;
      const uint32_t* pData32 = (const uint32_t*)data.data();
      int tType = pData32[detailedTrigger32] & 0xFFF;
      if (verbosity_ > 3)
        cout << "[LaserSorter " << now() << "] "
             << "Trigger type " << tType << "\n";
      stat.add(tType);
    }
  }
  if (!ecalData)
    return -2;
  double p;
  int tType = stat.result(&p);
  if (p < 0) {
    //throw cms::Exception("NotFound") << "No ECAL DCC data found\n";
    if (iNoEcalDataMess_ < maxNoEcalDataMess_) {
      edm::LogWarning("NotFound") << "No ECAL DCC data found. "
                                     "(This warning will be disabled for the current run after "
                                  << maxNoEcalDataMess_ << " occurences.)";
      ++iNoEcalDataMess_;
    }
    tType = -1;
  } else if (p < .8) {
    //throw cms::Exception("EventCorruption") << "Inconsitency in detailed trigger type indicated in ECAL DCC data headers\n";
    edm::LogWarning("EventCorruption") << "Inconsitency in detailed trigger type indicated in ECAL DCC data headers\n";
    tType = -1;
  }
  if (proba)
    *proba = p;
  return tType;
}

void LaserSorter::closeAllStreams() {
  for (OutStreamList::iterator it = outStreamList_.begin(); it != outStreamList_.end(); /*NOOP*/) {
    it = closeOutStream(it);
  }
}

void LaserSorter::closeOldStreams(edm::LuminosityBlockNumber_t lumiBlock) {
  const edm::LuminosityBlockNumber_t minLumiBlock = lumiBlock - lumiBlockSpan_;
  const edm::LuminosityBlockNumber_t maxLumiBlock = lumiBlock + lumiBlockSpan_;
  //If container type is ever changed, beware that
  //closeOutStream call in the loop removes it from outStreamList
  for (OutStreamList::iterator it = outStreamList_.begin(); it != outStreamList_.end();
       /*NOOP*/) {
    if ((*it)->startingLumiBlock() < minLumiBlock || (*it)->startingLumiBlock() > maxLumiBlock) {
      //event older than 2 lumi block => stream can be closed
      if (verbosity_)
        cout << "[LaserSorter " << now() << "] "
             << "Closing file for "
             << "FED " << (*it)->fedId() << " LB " << (*it)->startingLumiBlock() << "\n";
      it = closeOutStream(it);
    } else {
      ++it;
    }
  }
}

const std::unique_ptr<LaserSorter::OutStreamRecord>& LaserSorter::getStream(int fedId,
                                                                            edm::LuminosityBlockNumber_t lumiBlock) {
  const static std::unique_ptr<LaserSorter::OutStreamRecord> streamNotFound(nullptr);

  if ((fedId != -1) && (fedId < ecalDccFedIdMin_ || fedId > ecalDccFedIdMax_))
    fedId = -1;

  if (verbosity_ > 1)
    cout << "[LaserSorter " << now() << "] "
         << "Looking for an opened output file for FED " << fedId << " LB " << lumiBlock << "\n";

  //first look if stream is already open:
  for (OutStreamList::iterator it = outStreamList_.begin(); it != outStreamList_.end(); ++it) {
    if ((*it)->fedId() == fedId && (abs((int)(*it)->startingLumiBlock() - (int)lumiBlock) <= lumiBlockSpan_)) {
      //stream found!
      return (*it);
    }
  }
  //stream was not found. Let's create one

  if (verbosity_)
    cout << "[LaserSorter " << now() << "] "
         << "File not yet opened. Opening it.\n";

  OutStreamList::iterator streamRecord = createOutStream(fedId, lumiBlock);
  return streamRecord != outStreamList_.end() ? (*streamRecord) : streamNotFound;
}

bool LaserSorter::writeEvent(OutStreamRecord& outRcd,
                             const edm::Event& event,
                             int dtt,
                             const FEDRawDataCollection& data) {
  ofstream& out = *outRcd.out();
  bool rc = true;
  vector<unsigned> fedIds;
  getOutputFedList(event, data, fedIds);

  out.clear();
  uint32_t evtStart = out.tellp();
  if (out.bad())
    evtStart = 0;
  rc &= writeEventHeader(out, event, dtt, fedIds.size());

  if (orbitZeroTime_.tv_sec == 0 && data.FEDData(matacqFedId_).size() != 0) {
    struct timeval ts = {0, 0};
    MatacqRawEvent mre(data.FEDData(matacqFedId_).data(), data.FEDData(matacqFedId_).size());
    mre.getTimeStamp(ts);
    uint32_t orb = mre.getOrbitId();
    if (ts.tv_sec != 0) {
      div_t dt = div(orb * 89.1, 1000 * 1000);  //an orbit lasts 89.1 microseconds
      orbitZeroTime_.tv_sec = ts.tv_sec - dt.quot;
      orbitZeroTime_.tv_usec = ts.tv_usec - dt.rem;
      if (orbitZeroTime_.tv_usec < 0) {
        orbitZeroTime_.tv_usec += 1000 * 1000;
        orbitZeroTime_.tv_sec -= 1;
      }
    }
  }

  for (unsigned iFed = 0; iFed < fedIds.size() && rc; ++iFed) {
    if (verbosity_ > 3)
      cout << "[LaserSorter " << now() << "] "
           << "Writing data block of FED " << fedIds[iFed] << ". Data size: " << data.FEDData(fedIds[iFed]).size()
           << "\n";
    rc &= writeFedBlock(out, data.FEDData(fedIds[iFed]));
  }

  if (rc) {
    //update index table for this file:
    vector<IndexRecord>& indices = *outRcd.indices();
    if (verbosity_ > 2) {
      std::cout << "Event "
                << " written successfully. "
                << "Orbit: " << orbit_ << "\tFile index: " << evtStart << "\n";
    }
    IndexRecord indexRcd = {(uint32_t)orbit_, evtStart};
    indices.push_back(indexRcd);
  }
  return rc;
}

bool LaserSorter::writeFedBlock(std::ofstream& out, const FEDRawData& data) {
  bool rc = false;
  if (data.size() > 4) {
    const uint32_t* pData = reinterpret_cast<const uint32_t*>(data.data());

    uint32_t dccLen64 = pData[2] & 0x00FFFFFF;  //in 32-byte unit

    if (data.size() != dccLen64 * sizeof(uint64_t)) {
      //       throw cms::Exception("Bug") << "Bug found in "
      //                                   << __FILE__ << ":" << __LINE__ << ".";
      throw cms::Exception("LaserSorter") << "Mismatch between FED fragment size indicated in header "
                                          << "(" << dccLen64 << "*8 Byte) "
                                          << "and actual size (" << data.size() << " Byte) "
                                          << "for FED ID " << ((pData[0] >> 8) & 0xFFF) << "!\n";
    }

    if (verbosity_ > 3)
      cout << "[LaserSorter " << now() << "] "
           << "Event fragment size: " << data.size() << " Byte"
           << "\t From Dcc header: " << dccLen64 * 8 << " Byte\n";

    const size_t nBytes = data.size();
    //       cout << "[LaserSorter " << now() << "] "
    //            << "Writing " << nBytes << " byte from adress "
    //            << (void*) data.data() << " to file.\n";
    if (out.fail())
      cout << "[LaserSorter " << now() << "] "
           << "Problem with stream!\n";
    out.write((const char*)data.data(), nBytes);
    rc = true;
  } else {
    throw cms::Exception("Bug") << "Bug found in " << __FILE__ << ":" << __LINE__ << ".\n";
  }
  return rc;
}

bool LaserSorter::renameAsBackup(const std::string& fileName, std::string& newFileName) {
  int i = 0;
  int err;
  //  static int maxTries = 100;
  int maxTries = 20;
  stringstream newFileName_;
  do {
    newFileName_.str("");
    newFileName_ << fileName << "~";
    if (i > 0)
      newFileName_ << i;
    err = link(fileName.c_str(), newFileName_.str().c_str());
    if (err == 0) {
      newFileName = newFileName_.str();
      err = unlink(fileName.c_str());
    }
    ++i;
  } while ((err != 0) && (errno == EEXIST) && (i < maxTries));
  return err == 0;
}

LaserSorter::OutStreamList::iterator LaserSorter::createOutStream(int fedId, edm::LuminosityBlockNumber_t lumiBlock) {
  if (verbosity_)
    cout << "[LaserSorter " << now() << "] "
         << "Creating a stream for FED " << fedId << " lumi block " << lumiBlock << ".\n";
  std::string tmpName;
  std::string finalName;

  streamFileName(fedId, lumiBlock, tmpName, finalName);

  errno = 0;

  //checks if a file with tmpName name already exists:
  ofstream* out = new ofstream(tmpName.c_str(), ios::out | ios::in);
  if (out->is_open()) {  //temporary file already exists. Making a backup:
    string newName;
    if (!renameAsBackup(tmpName, newName)) {
      throw cms::Exception("LaserSorter") << "Failed to rename file " << tmpName << "  to " << newName << "\n";
    }
    if (verbosity_)
      cout << "[LaserSorter " << now() << "] "
           << "Already existing File " << tmpName << " renamed to " << newName << "\n";
    out->close();
  }

  out->clear();
  out->open(tmpName.c_str(), ios::out | ios::trunc);

  if (out->fail()) {  //failed to create file
    delete out;
    throw cms::Exception("LaserSorter") << "Failed to create file " << tmpName << " for writing event from FED "
                                        << fedId << " lumi block " << lumiBlock << ": " << strerror(errno) << "\n.";
  }

  ifstream in(finalName.c_str());
  bool newFile = true;
  if (in.good()) {  //file already exists with final name.
    if (verbosity_)
      cout << "[LaserSorter " << now() << "] "
           << "File " << finalName << " already exists. It will be updated if needed.\n";
    //Copying its contents:
    char buffer[256];
    streamsize nread = -1;
    int vers = readFormatVersion(in, finalName);
    if (vers == -1) {
      edm::LogWarning("LaserSorter") << "File " << tmpName.c_str() << " is not an LMF file despite its extension or "
                                     << "it is corrupted.\n";
    } else if (vers != formatVersion_) {
      edm::LogWarning("LaserSorter") << "Cannot include events already in file " << tmpName.c_str()
                                     << " because of version "
                                     << "mismatch (found version " << (int)vers << " while "
                                     << "only version " << (int)formatVersion_ << " is supported).\n";
    } else {
      newFile = false;
      //read index table offset value:
      const int indexTableOffsetPos8 = 1 * sizeof(uint32_t);
      uint32_t indexTableOffsetValue = 0;
      in.clear();
      in.seekg(indexTableOffsetPos8, ios::beg);
      in.read((char*)&indexTableOffsetValue, sizeof(indexTableOffsetValue));
      if (in.fail()) {
        cout << "[LaserSorter " << now() << "] "
             << "Failed to read offset of index table "
                " in the existing file "
             << finalName << "\n";
      } else {
        if (verbosity_ > 2)
          cout << "[LaserSorter " << now() << "] "
               << "Index table offset of "
                  "original file "
               << finalName << ": 0x" << hex << setfill('0') << setw(8) << indexTableOffsetValue << dec << setfill(' ')
               << "\n";
      }
      in.clear();
      in.seekg(0, ios::beg);

      //copy legacy file contents except the index table
      uint32_t toRead = indexTableOffsetValue;
      cout << "[LaserSorter " << now() << "] "
           << "Copying " << finalName << " to " << tmpName << endl;
      while (!in.eof() && (toRead > 0) && (nread = in.readsome(buffer, min(toRead, (uint32_t)sizeof(buffer)))) != 0) {
        //         cout << "Writing " << nread << " bytes to file "
        //              << tmpName.c_str() << "\n";
        toRead -= nread;
        // out->seekp(0, ios::end);
        out->write(buffer, nread);
        if (out->bad()) {
          throw cms::Exception("LaserSorter")
              << "Error while writing to file " << tmpName << ". Check if there is enough "
              << "space on the device.\n";
        }
      }

      //resets index table offset field:
      indexTableOffsetValue = 0;
      out->clear();
      out->seekp(indexTableOffsetPos8, ios::beg);
      out->write((char*)&indexTableOffsetValue, sizeof(uint32_t));
      out->clear();
      out->seekp(0, ios::end);
    }
  }

#if 0
  out->flush();
  cout << "Press enter... file name was " << tmpName << endl;
  char c;
  cin >> c;
#endif

  OutStreamRecord* outRcd = new OutStreamRecord(fedId, lumiBlock, out, tmpName, finalName);

  if (newFile) {
    writeFileHeader(*out);
  } else {
    std::string errMsg;
    if (!readIndexTable(in, finalName, *outRcd, &errMsg)) {
      throw cms::Exception("LaserSorter") << errMsg << "\n";
    }
  }

  return outStreamList_.emplace(outStreamList_.end(), outRcd);
}

void LaserSorter::writeFileHeader(std::ofstream& out) {
  out.clear();

  uint32_t id = 'L' | ('M' << 8) | ('F' << 16) | (formatVersion_ << 24);

  out.write((char*)&id, sizeof(uint32_t));

  //index position (to be filled at end of writing)
  uint32_t zero = 0;
  out.write((char*)&zero, sizeof(uint32_t));

  if (out.fail()) {
    throw cms::Exception("LaserSorter") << "Failed to write file header.\n";
  }
}

bool LaserSorter::writeEventHeader(std::ofstream& out, const edm::Event& evt, int dtt, unsigned nFeds) {
  uint32_t data[10];
  timeval tt = {0, 0};
  if ((evt.time().value() >> 32)) {
    tt.tv_usec = evt.time().value() & 0xFFFFFFFF;
    tt.tv_sec = evt.time().value() >> 32;
  } else if (orbitZeroTime_.tv_sec) {
    div_t dt = div(orbit_ * 89.1, 1000 * 1000);  //one orbit lasts 89.1 microseconds
    tt.tv_sec = orbitZeroTime_.tv_sec + dt.quot;
    tt.tv_usec = orbitZeroTime_.tv_usec + dt.rem;
    if (tt.tv_usec > 1000 * 1000) {
      tt.tv_usec -= 1000 * 1000;
      tt.tv_sec += 1;
    }
  }

  data[0] = tt.tv_usec;
  data[1] = tt.tv_sec;
  data[2] = evt.luminosityBlock();
  data[3] = evt.run();
  data[4] = orbit_;
  data[5] = evt.bunchCrossing();
  data[6] = evt.id().event();
  data[7] = dtt;
  data[8] = nFeds;
  data[9] = 0;  //reserved (to be aligned on 64-bits)

  if (verbosity_ > 1) {
    cout << "[LaserSorter " << now() << "] "
         << "Write header of event: "
         << "Time: " << toString(evt.time().value()) << ", LB: " << evt.luminosityBlock() << ", Run: " << evt.run()
         << ", Bx: " << evt.bunchCrossing() << ", Event ID: " << evt.id().event() << ", Detailed trigger type: 0x"
         << hex << dtt << dec << " (" << detailedTrigNames[(dtt >> 8) & 0x7] << ", " << colorNames[(dtt >> 6) & 0x3]
         << ", DCC " << (dtt & 0x3f) << ", side " << ((dtt >> 10) & 0x1) << ")"
         << ", number of FEDs: "
         << "\n";
  }

  out.clear();
  out.write((char*)data, sizeof(data));
  return !out.bad();
}

void LaserSorter::streamFileName(int fedId,
                                 edm::LuminosityBlockNumber_t lumiBlock,
                                 std::string& tmpName,
                                 std::string& finalName) {
  int iFed;
  if (fedId >= ecalDccFedIdMin_ && fedId <= ecalDccFedIdMax_) {
    iFed = fedId - ecalDccFedIdMin_ + 1;
  } else if (fedId < 0) {
    iFed = -1;  //event w/o ECAL data
  } else {
    iFed = 0;
  }
  if (iFed < -1 || iFed >= (int)fedSubDirs_.size()) {
    throw cms::Exception("LaserSorter") << "Bug found at " << __FILE__ << ":" << __LINE__
                                        << ". FED ID is out of index!";
  }

  struct stat fileStat;

  stringstream buf;
  buf << outputDir_ << "/" << (iFed < 0 ? "Empty" : fedSubDirs_[iFed]);

  string dir = buf.str();
  if (0 == stat(dir.c_str(), &fileStat)) {
    if (!S_ISDIR(fileStat.st_mode)) {
      throw cms::Exception("[LaserSorter]") << "File " << dir << " exists but is not a directory "
                                            << " as expected.";
    }
  } else {  //directory does not exists, let's try to create it
    if (0 != mkdir(dir.c_str(), 0755)) {
      throw cms::Exception("[LaserSorter]") << "Failed to create directory " << dir << " for writing data.";
    }
  }

  buf.str("");
  buf << "Run" << runNumber_ << "_LB" << setfill('0') << setw(4) << lumiBlock << ".lmf";
  string fileName = buf.str();
  string tmpFileName = fileName + ".part";

  finalName = dir + "/" + fileName;
  tmpName = dir + "/" + tmpFileName;

  if (verbosity_ > 3)
    cout << "[LaserSorter " << now() << "] "
         << "File path: " << finalName << "\n";
}

LaserSorter::OutStreamList::iterator LaserSorter::closeOutStream(LaserSorter::OutStreamList::iterator streamRecord) {
  if (streamRecord == outStreamList_.end())
    return outStreamList_.end();

  if (verbosity_)
    cout << "[LaserSorter " << now() << "] "
         << "Writing Index table of file " << (*streamRecord)->finalFileName() << "\n";
  ofstream& out = *(*streamRecord)->out();
  out.clear();
  if (!writeIndexTable(out, *(*streamRecord)->indices())) {
    cout << "Error while writing index table for file " << (*streamRecord)->finalFileName() << ". "
         << "Resulting file might be corrupted. "
         << "The error can be due to a lack of disk space.";
  }

  if (verbosity_)
    cout << "[LaserSorter " << now() << "] "
         << "Closing file " << (*streamRecord)->finalFileName() << ".\n";
  out.close();

  const std::string& tmpFileName = (*streamRecord)->tmpFileName();
  const std::string& finalFileName = (*streamRecord)->finalFileName();

  if (verbosity_)
    cout << "[LaserSorter " << now() << "] "
         << "Renaming " << tmpFileName << " to " << finalFileName << ".\n";

  if (0 != rename(tmpFileName.c_str(), finalFileName.c_str())) {
    cout << "[LaserSorter " << now() << "] "
         << " Failed to rename output file from " << tmpFileName << " to " << finalFileName << ". " << strerror(errno)
         << "\n";
  }

  if (doOutputList_) {
    char buf[256];
    time_t t = time(nullptr);
    strftime(buf, sizeof(buf), "%F %R:%S", localtime(&t));

    ifstream f(".watcherfile");
    string inputFile;
    f >> inputFile;
    outputList_ << finalFileName << "\t" << buf << "\t" << inputFile << endl;
  }

  return outStreamList_.erase(streamRecord);
}

void LaserSorter::endJob() {
  //TODO: better treatement of last files:
  //they might be imcomplete...
  closeAllStreams();
}

void LaserSorter::beginJob() {}

bool LaserSorter::isDccEventEmpty(const FEDRawData& data, size_t* dccLen, int* nTowerBlocks) const {
  if (nTowerBlocks)
    *nTowerBlocks = 0;
  //DCC event is considered empty if it does not contains any Tower block
  //( = FE data)
  bool rc = true;
  if (dccLen)
    *dccLen = 0;
  const unsigned nWord32 = data.size() / sizeof(uint32_t);
  if (nWord32 == 0) {
    //cout << "[LaserSorter " << now() << "] " << "FED block completly empty\n";
    return true;
  }
  for (unsigned iWord32 = 0; iWord32 < nWord32; iWord32 += 2) {
    const uint32_t* data32 = ((const uint32_t*)(data.data())) + iWord32;
    int dataType = (data32[1] >> 28) & 0xF;
    //     cout << hex << "0x" << setfill('0')
    //          << setw(8) << data32[1] << "'" << setw(8) << data32[0]
    //          << " dataType: 0x" << dataType
    //          << dec << setfill(' ') << "\n";
    if (0 == (dataType >> 2)) {  //in DCC header
      const int dccHeaderId = (data32[1] >> 24) & 0x3F;
      if (dccHeaderId == 1) {
        if (dccLen)
          *dccLen = ((data32[0] >> 0) & 0xFFFFFF);
      }
    }
    if ((dataType >> 2) == 3) {  //Tower block
      rc = false;
      if (nTowerBlocks) {  //number of tower block must be counted
        ++(*nTowerBlocks);
      } else {
        break;
      }
    }
  }
  //   cout << "[LaserSorter " << now() << "] " << "DCC Len: ";

  //   if(dccLen){
  //     cout << (*dccLen) << " event ";
  //   }
  //   cout << (rc?"":"non") << " empty"
  //        << endl;
  return rc;
}

void LaserSorter::getOutputFedList(const edm::Event& event,
                                   const FEDRawDataCollection& data,
                                   std::vector<unsigned>& fedIds) const {
  fedIds.erase(fedIds.begin(), fedIds.end());
  for (int id = ecalDccFedIdMin_; id <= ecalDccFedIdMax_; ++id) {
    size_t dccLen;
    const FEDRawData& dccEvent = data.FEDData(id);
    if (!isDccEventEmpty(dccEvent, &dccLen)) {
      fedIds.push_back(id);
    }
    if (dccLen * sizeof(uint64_t) != dccEvent.size()) {
      edm::LogWarning("LaserSorter") << "Length error in data of FED " << id << " in event " << event.id()
                                     << ", Data of this FED dropped.";
    }
  }
  //   cout << __FILE__ << ":" << __LINE__ << ": "
  //        <<  "data.FEDData(" << matacqFedId_ << ").size() = "
  //        <<  data.FEDData(matacqFedId_).size() << "\n";
  if (data.FEDData(matacqFedId_).size() > 4) {  //matacq block present
    //    cout << __FILE__ << ":" << __LINE__ << ": "
    //     <<  "Adding matacq to list of FEDs\n";
    fedIds.push_back(matacqFedId_);
  }
}

std::vector<int> LaserSorter::getFullyReadoutDccs(const FEDRawDataCollection& data) const {
  int nTowers;
  vector<int> result;
  for (int fed = ecalDccFedIdMin_; fed <= ecalDccFedIdMax_; ++fed) {
    const FEDRawData& fedData = data.FEDData(fed);
    isDccEventEmpty(fedData, nullptr, &nTowers);
    if (nTowers >= 68)
      result.push_back(fed);
  }
  return result;
}

bool LaserSorter::writeIndexTable(std::ofstream& out, std::vector<IndexRecord>& indices) {
  uint32_t indexTablePos = out.tellp();
  uint32_t nevts = indices.size();

  out.clear();
  out.write((char*)&nevts, sizeof(nevts));
  const uint32_t reserved = 0;
  out.write((const char*)&reserved, sizeof(reserved));

  if (out.bad())
    return false;

  sort(indices.begin(), indices.end());

  for (unsigned i = 0; i < indices.size(); ++i) {
    uint32_t data[2];
    data[0] = indices[i].orbit;
    data[1] = indices[i].filePos;
    out.write((char*)data, sizeof(data));
  }

  if (out.bad())
    return false;  //intial 0 valur for index table position
  //                            is left to indicate corrupted table.

  //writes index table position:x
  out.clear();
  out.seekp(indexOffset32_ * sizeof(uint32_t));
  //   cout << "[LaserSorter] Index table position: 0x" << hex << indexTablePos
  //        << dec << "\n";
  if (!out.bad())
    out.write((char*)&indexTablePos, sizeof(uint32_t));

  bool rc = !out.bad();

  //reposition pointer to eof:
  out.seekp(0, ios::end);

  return rc;
}

//beware this method change the pointer position in the ifstream in
bool LaserSorter::readIndexTable(std::ifstream& in, std::string& inName, OutStreamRecord& outRcd, std::string* err) {
  stringstream errMsg;

  ifstream* s = &in;

  //streampos pos = s->tellg();
  s->clear();
  s->seekg(0);

  uint32_t fileHeader[2];
  s->read((char*)&fileHeader[0], sizeof(fileHeader));
  uint32_t indexTablePos = fileHeader[1];

  if (s->eof()) {
    s->clear();
    s->seekg(0);
    errMsg << "Failed to read header of file " << inName << ".";
    if (err)
      *err = errMsg.str();
    return false;
  }

  s->seekg(indexTablePos);

  uint32_t nevts = 0;
  s->read((char*)&nevts, sizeof(nevts));
  s->ignore(4);
  if (s->bad()) {
    errMsg << "Failed to read index table from file " << inName << ".";
    if (err)
      *err = errMsg.str();
    return false;
  }
  if (nevts > maxEvents_) {
    errMsg << "Number of events indicated in event index of file " << inName << " (" << nevts << ") "
           << "is unexpectively large.";
    if (err)
      *err = errMsg.str();
    return false;
  }
  outRcd.indices()->resize(nevts);
  s->read((char*)&(*outRcd.indices())[0], nevts * sizeof(IndexRecord));
  if (s->bad()) {
    outRcd.indices()->clear();
    errMsg << "Failed to read index table from file " << inName << ".";
    if (err)
      *err = errMsg.str();
    return false;
  }
  if (nevts > maxEvents_) {
    errMsg << "Number of events indicated in event index of file " << inName << " is unexpectively large.";
    if (err)
      *err = errMsg.str();
    outRcd.indices()->clear();
    return false;
  }

  if (verbosity_ > 1)
    cout << "[LaserSorter " << now() << "] "
         << "Orbit IDs of events "
         << "already contained in the file " << inName << ":";
  for (unsigned i = 0; i < outRcd.indices()->size(); ++i) {
    if (verbosity_ > 1) {
      cout << " " << setw(9) << (*outRcd.indices())[i].orbit;
    }
    outRcd.excludedOrbit().insert((*outRcd.indices())[i].orbit);
  }
  if (verbosity_ > 1)
    cout << "\n";

  return true;
}

int LaserSorter::readFormatVersion(std::ifstream& in, const std::string& fileName) {
  int vers = -1;
  streampos p = in.tellg();

  uint32_t data;

  in.read((char*)&data, sizeof(data));

  char magic[4];

  magic[0] = data & 0xFF;
  magic[1] = (data >> 8) & 0xFF;
  magic[2] = (data >> 16) & 0xFF;
  magic[3] = 0;

  const string lmf = string("LMF");

  if (in.good() && lmf == magic) {
    vers = (data >> 24) & 0xFF;
  }

  if (lmf != magic) {
    edm::LogWarning("LaserSorter") << "File " << fileName << "is not an LMF file.\n";
  }

  in.clear();
  in.seekg(p);
  return vers;
}

std::string LaserSorter::toString(uint64_t t) {
  char buf[256];

  time_t tsec = t >> 32;

  uint32_t tusec = t & 0xFFFFFFFF;
  strftime(buf, sizeof(buf), "%F %R %S s", localtime(&tsec));
  buf[sizeof(buf) - 1] = 0;

  stringstream buf2;
  buf2 << (tusec + 500) / 1000;

  return string(buf) + " " + buf2.str() + " ms";
}

void LaserSorter::restoreStreamsOfLumiBlock(int lumiBlock) {
  string dummy;
  string fileName;

  for (int fedId = ecalDccFedIdMin_ - 2; fedId <= ecalDccFedIdMax_; ++fedId) {
    int fedId_;
    if (fedId == ecalDccFedIdMin_ - 2)
      fedId_ = -1;  //stream for event w/o ECAL data
    else
      fedId_ = fedId;
    streamFileName(fedId_, lumiBlock, dummy, fileName);
    struct stat s;
    //TODO: could be optimized by adding an option to get stream
    //to open only existing file: would avoid double call to streamFileName.
    if (stat(fileName.c_str(), &s) == 0) {  //file exists
      getStream(fedId_, lumiBlock);
    }
  }
}

void LaserSorter::beginRun(edm::Run const& run, edm::EventSetup const& es) {}

void LaserSorter::endRun(edm::Run const& run, edm::EventSetup const& es) {}
