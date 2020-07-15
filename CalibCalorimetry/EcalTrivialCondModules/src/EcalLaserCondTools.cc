//emacs settings:-*- mode: c++; c-basic-offset: 2; indent-tabs-mode: nil -*-
/*
 * $Id: EcalLaserCondTools.cc,v 1.2 2010/06/14 10:45:17 pgras Exp $
 *
 * author: Ph Gras. June, 2010
 */

#include "CalibCalorimetry/EcalTrivialCondModules/interface/EcalLaserCondTools.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "CondFormats/DataRecord/interface/EcalLaserAPDPNRatiosRcd.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"

#include <string>
#include <fstream>
#include <algorithm>

using namespace std;

EcalLaserCondTools::EcalLaserCondTools(const edm::ParameterSet& ps)
    : fout_(nullptr),
      eventList_(nullptr),
      eventListFileName_(ps.getParameter<string>("eventListFile")),
      verb_(ps.getParameter<int>("verbosity")),
      mode_(ps.getParameter<string>("mode")),
      fnames_(ps.getParameter<vector<string> >("laserCorrectionAsciiFiles")),
      skipIov_(ps.getParameter<int>("skipIov")),
      nIovs_(ps.getParameter<int>("nIovs")),
      fromTime_(ps.getParameter<int>("fromTime")),
      toTime_(ps.getParameter<int>("toTime")),
      minP_(ps.getParameter<double>("transparencyMin")),
      maxP_(ps.getParameter<double>("transparencyMax")) {
  ferr_ = fopen("corr_errors.txt", "w");
  fprintf(ferr_, "#t1\tdetid\tp1\tp2\tp3");

  if (!eventListFileName_.empty()) {
    eventList_ = fopen(eventListFileName_.c_str(), "r");
    if (eventList_ == nullptr)
      throw cms::Exception("User") << "Failed to open file " << eventListFileName_ << "\n";
  }
}

EcalLaserCondTools::~EcalLaserCondTools() {
  if (ferr_)
    fclose(ferr_);
  if (fout_)
    fclose(fout_);
}

void EcalLaserCondTools::analyze(const edm::Event& event, const edm::EventSetup& es) {
  static bool done = false;

  if (mode_ == "ascii_file_to_db") {
    if (done)
      return;
    if (verb_ > 2)
      cout << "ascii_file_to_db mode" << endl;
    if (!db_.isAvailable()) {
      throw cms::Exception("CondDBAccess") << "Failed to connect to PoolDBOutputService\n";
    }
    FileReader corrReader(fnames_);
    corrReader.setVerbosity(verb_);
    fillDb(corrReader);
  } else if (mode_ == "db_to_ascii_file") {
    dbToAscii(es);
  } else {
    cms::Exception("InvalidParam") << "Value of parameter mode is not valid. Expecting ascii_file_to_db or read";
  }
}

void EcalLaserCondTools::fillDb(CorrReader& r) {
  int iIov = 0;
  int processedIovs = 0;
  if (verb_ > 2)
    cout << "Starting filling DB...\n";
  int t1 = 0;
  int t3 = 0;
  int t2[nLmes];

  int prevT1 = 0;
  int prevT3 = 0;
  int prevT = 0;
  int t = 0;
  if (eventList_) {
    int iline = 0;
    while (!feof(eventList_)) {
      //skips comment lines:
      char c;
      while (fscanf(eventList_, " %1[#]%*[^\n]\n", &c) == 1)
        ++iline;

      int n = fscanf(eventList_, "%*d %*d %*d %d%*[^\n]\n", &t);
      if (verb_ > 1)
        cout << "Event time: t = " << t << ", " << timeToString(t) << "\n";
      ++iline;
      if (n != 1)
        throw cms::Exception("User") << "Syntax error in event list file " << eventListFileName_ << " at line " << iline
                                     << " " << n << " "
                                     << ".\n";
      if (t < prevT)
        throw cms::Exception("User") << "Events in the event list file " << eventListFileName_
                                     << " are not ordered in increased time as required! See line " << iline << "\n";
      if (t == 0)
        throw cms::Exception("User") << "Found an unexpected t = 0 time value "
                                        "in the event list file"
                                     << eventListFileName_ << " at line " << iline << "!\n";
      //Look for IOV:
      bool iovFound = true;
      if (t <= t3) {  //IOV already inserted for previous event.
        if (verb_ > 1)
          cout << "Event in same IOV than previous one.\n";
        continue;
      }

      while ((iovFound = r.readTime(t1, t2, t3)) && t3 < t) /*NOP*/
        ;

      if (iovFound) {
        if (t1 < prevT1 && t3 < prevT3)
          throw cms::Exception("User")
              << "IOVs in the correction ascii file are not ordered in increased time as required!\n";
        else if (t1 < prevT1 || t3 < prevT3)
          throw cms::Exception("User") << "Found interleaved IOVs in the correction ascii file!\n";
        processIov(r, t1, t2, t3);
      } else {
        cout << "Warning: event beyond last IOV t3. Event time: " << timeToString(t)
             << ". Last IOV t3: " << timeToString(t3) << "\n";
      }
    }
  } else
    while (r.readTime(t1, t2, t3)) {
      ++iIov;
      if (iIov <= skipIov_) {
        std::cout << "Skipping IOV " << iIov << "\n";
        continue;
      } else if (processedIovs >= nIovs_) {
        std::cout << "Requested number of IOVs, " << nIovs_ << ", processed.\n";
        return;
      } else {
        processIov(r, t1, t2, t3);
        ++processedIovs;
      }
    }
}

void EcalLaserCondTools::processIov(CorrReader& r, int t1, int t2[EcalLaserCondTools::nLmes], int t3) {
  static int iIov = 0;
  ++iIov;

  //   FILE* fdebug = 0;
  //   if(i==) fdebug = fopen("debug.txt", "w");

  //  if(iIov <= skipIov_) { std::cout << "Skipping IOV " << iIov << "\n"; return; }

  cond::Time_t iovStart = 0;

  if (verb_ > 1) {
    cout << "t1:" << t1 << "(" << timeToString(t1) << ") \n"
         << "t3: " << t3 << "(" << timeToString(t3) << ")\nt2-t1: ";
    for (int i = 0; i < EcalLaserCondTools::nLmes; ++i)
      cout << t2[i] - t1 << "\t";
    cout << "\n";
  }
  if (t1 < fromTime_) {
    std::cout << "Skipping IOV " << iIov << ", "
              << ", which is before 'fromTime'," << timeToString(fromTime_) << "(" << fromTime_ << ").\n";
    return;
  }

  //   if(toTime_!=-1 && t2 < toTime_) { std::cout << "Skipping IOV " << iIov << ", " << ", which is beyond 'toTime',"
  //                                               << timeToString(toTime_) << "(" << toTime_ << ").\n"; return; }

  if (t1 == 0) {
    std::cout << "Skipping IOV with t1 = 0"
              << "\n";
    return;
  }

  EcalLaserAPDPNRatios* corrSet = new EcalLaserAPDPNRatios;

  EcalLaserAPDPNRatios::EcalLaserTimeStamp t;
  iovStart = uint64_t(t1) << 32;
  for (size_t i = 0; i < EcalLaserCondTools::nLmes; ++i) {
    t.t1 = edm::Timestamp(uint64_t(t1) << 32);
    t.t2 = edm::Timestamp(uint64_t(t2[i]) << 32);
    t.t3 = edm::Timestamp(uint64_t(t3) << 32);
    corrSet->setTime(i, t);
  }

  constexpr int ncrystals = 75848;
  std::set<int> detidList;
  for (int i = 0; i < ncrystals; ++i) {
    DetId detid;
    //EcalLaserAPDPNRatios::EcalLaserAPDPNpair corr = {0, 0, 0};
    EcalLaserAPDPNRatios::EcalLaserAPDPNpair corr = EcalLaserAPDPNRatios::EcalLaserAPDPNpair();
    if (verb_ > 2)
      cout << "Reading " << toNth(i + 1) << " crystal\n";
    if (!r.readPs(detid, corr)) {
      throw cms::Exception("LasCor") << "Failed to read " << toNth(i + 1) << " crystal correction.\n";
    }

    pair<std::set<int>::iterator, bool> res = detidList.insert(int(detid));

    if (!res.second) {  //detid already processed
      std::cout << "Duplicate det id, for IOV " << iIov << " t1 = " << t1 << " detid = " << int(detid) << endl;
    }

    if (!isfinite(corr.p1) || !isfinite(corr.p2) || !isfinite(corr.p3) || corr.p1 < minP_ || corr.p1 > maxP_ ||
        corr.p2 < minP_ || corr.p2 > maxP_ || corr.p3 < minP_ || corr.p3 > maxP_) {
      fprintf(ferr_, "%d %d %f %f %f\n", t1, (int)detid, corr.p1, corr.p2, corr.p3);
      corr.p1 = corr.p2 = corr.p3 = 1;
    }

    if (verb_ > 2) {
      if (detid.subdetId() == EcalBarrel) {
        cout << EBDetId(detid);
      } else if (detid.subdetId() == EcalEndcap) {
        cout << EEDetId(detid);
      } else {
        cout << (int)detid;
      }
      cout << ": "
           << "p1 = " << corr.p1 << "\t"
           << "p2 = " << corr.p2 << "\t"
           << "p3 = " << corr.p3 << "\n";
    }

    corrSet->setValue((int)detid, corr);
  }

  try {
    //Write correction set in DB (one IOV):
    if (db_->isNewTagRequest("EcalLaserAPDPNRatiosRcd")) {
      if (verb_)
        cout << "First IOV, extending starting time." << endl;
      iovStart = db_->beginOfTime();
    }
    timeval t;
    gettimeofday(&t, nullptr);
    if (verb_ > 1)
      cout << "[" << timeToString(t.tv_sec) << "] "
           << "Write IOV " << iIov << " starting from " << timeToString(iovStart >> 32) << "... ";
    db_->writeOne(corrSet, iovStart, "EcalLaserAPDPNRatiosRcd");
  } catch (const cms::Exception& e) {
    std::cout << "Failed.\nException cathed while writting to cond DB" << e.what() << "\n";
  }
  std::cout << "Suceeded.\n";
}

bool EcalLaserCondTools::FileReader::nextFile() {
  for (;;) {
    ++ifile_;
    if (ifile_ >= fnames_.size()) {
      if (verb_ > 1)
        cout << "No more correction files.\n";
      return false;
    }
    if (verb_ > 1)
      cout << "Opening file " << fnames_[ifile_] << "\n";
    f_ = fopen(fnames_[ifile_].c_str(), "r");
    iline_ = 0;
    if (f_ == nullptr) {
      std::cerr << "Failed to open file " << fnames_[ifile_] << ". File skipped!\n";
    } else {
      return true;
    }
  }
}

bool EcalLaserCondTools::FileReader::readTime(int& t1, int t2[EcalLaserCondTools::nLmes], int& t3) {
  trim();
  if ((f_ == nullptr || feof(f_)) && !nextFile()) {
    if (verb_ > 1)
      cout << "No more record\n";
    return false;
  }
  int i;
  char* buf = nullptr;
  size_t s = 0;
  while ((i = fgetc(f_)) != 'T' && i != 'L' && i >= 0)
    getline(&buf, &s, f_);
  if (buf)
    free(buf);
  buf = nullptr;

  if (i == 'L') {  //last record put 3 consecutive times starting from end of prev. IOV
    t1 = t3;
    for (int i = 0; i < EcalLaserCondTools::nLmes; ++i)
      t2[i] = t1 + 1;
    t3 = t1 + 2;
    return true;
  }

  if (i != 'T') {
    if (verb_ > 1)
      cout << "No more record or bad line type/marker (getc returned " << i << ")\n";
    return false;
  }

  EcalLaserAPDPNRatios::EcalLaserTimeStamp t;
  int n = fscanf(f_, "%d %d", &t1, &t3);
  for (int i = 0; i < EcalLaserCondTools::nLmes; ++i) {
    int nn = fscanf(f_, "%d", &t2[i]);
    if (nn != 1)
      break;
    n += nn;
  }

  int nnn = fscanf(f_, " ");

  if (n != (2 + EcalLaserCondTools::nLmes) || nnn != 0)
    throw cms::Exception("LasCorFile") << "File " << fnames_[ifile_] << " line " << iline_
                                       << ": syntax error. Expecting 'T' marker followed by 94 values: "
                                       << "t1 t2 t3(lme 1) t3(lme 2) ... t3(lme " << EcalLaserCondTools::nLmes << ")\n";

  return true;
}

bool EcalLaserCondTools::FileReader::readPs(DetId& detid, EcalLaserAPDPNRatios::EcalLaserAPDPNpair& corr) {
  if (f_ == nullptr) {
    if (verb_)
      cout << "Requested to read p1..p3 parameter line while no file is closed.\n";
    return false;
  }

  trim();
  int i = fgetc(f_);

  if (i != 'P') {
    if (verb_ && i >= 0)
      cout << "File " << fnames_[ifile_] << " line " << iline_ << ": unexpected line type, '" << (char)i
           << "' while expecting 'P'\n";
    if (verb_ && i < 0)
      cout << "Failed to read p1..p3 parameter line\n";
    return false;
  }

  int rawdetid;
  int n = fscanf(f_, "%d %f %f %f\n", &rawdetid, &corr.p1, &corr.p2, &corr.p3);
  ++iline_;

  if (n != 4) {
    //    corr.p2=corr.p1;
    //    corr.p3=corr.p1;
    throw cms::Exception("I/O") << "Syntax error at line " << iline_ << "of file " << fnames_[ifile_] << " read " << n
                                << " values,"
                                << " raw id" << rawdetid << ": " << corr.p1 << ", " << corr.p2;
  }
  detid = rawdetid;
  constexpr int ECALID = 3;
  if (detid.det() != ECALID)
    throw cms::Exception("InvalidValue") << "Line " << iline_ << "of file " << fnames_[ifile_]
                                         << " contains an invalid det ID (detector code is not ECAL!)\n";
  if (detid.subdetId() == EcalBarrel) {
    EBDetId ebDetId(detid);
    if (!EBDetId::validDetId(ebDetId.ietaAbs(), ebDetId.iphi()))
      throw cms::Exception("InvalidValue") << "Line " << iline_ << "of file " << fnames_[ifile_]
                                           << " contains an invalid det ID (detector code is not ECAL!)\n";
  }
  if (detid.subdetId() == EcalEndcap) {
    EEDetId eeDetId(detid);
    if (!EEDetId::validDetId(eeDetId.ix(), eeDetId.iy(), eeDetId.zside()))
      throw cms::Exception("InvalidValue") << "Line " << iline_ << "of file " << fnames_[ifile_]
                                           << " contains an invalid det ID (detector code is not ECAL!)\n";
  }
  ++iline_;
  return true;
}

void EcalLaserCondTools::FileReader::trim() {
  if (f_ == nullptr)
    return;
  bool skipLine = false;
  int c;
  while ((c = fgetc(f_)) >= 0 && (c == ' ' || c == '\t' || c == '\n' || c == '#' || skipLine)) {
    if (c == '#')
      skipLine = true;
    if (c == '\n') {
      ++iline_;
      skipLine = false;
    }
  }
  ungetc(c, f_);
}

string EcalLaserCondTools::toNth(int n) {
  stringstream s;
  s << n;
  if (n % 100 < 10 || n % 100 > 20) {
    switch (n % 10) {
      case 1:
        s << "st";
        break;
      case 2:
        s << "nd";
        break;
      case 3:
        s << "rd";
        break;
      default:
        s << "th";
    }
  } else {
    s << "th";
  }
  return s.str();
}

std::string EcalLaserCondTools::timeToString(time_t t) {
  char buf[256];
  struct tm lt;
  localtime_r(&t, &lt);
  strftime(buf, sizeof(buf), "%F %R:%S", &lt);
  buf[sizeof(buf) - 1] = 0;
  return string(buf);
}

void EcalLaserCondTools::dbToAscii(const edm::EventSetup& es) {
  edm::ESHandle<EcalLaserAPDPNRatios> hCorr;
  es.get<EcalLaserAPDPNRatiosRcd>().get(hCorr);

  const EcalLaserAPDPNRatios* corr = hCorr.product();

  const EcalLaserAPDPNRatios::EcalLaserAPDPNRatiosMap& p = corr->getLaserMap();
  const EcalLaserAPDPNRatios::EcalLaserTimeStampMap& t = corr->getTimeMap();

  if (t.size() != EcalLaserCondTools::nLmes)
    throw cms::Exception("LasCor") << "Unexpected number time parameter triplets\n";

  if (fout_ == nullptr) {
    fout_ = fopen("corr_dump.txt", "w");
    if (fout_ == nullptr)
      throw cms::Exception("LasCor") << "Failed to create file corr_dump.txt\n";
  }

  unsigned t1 = t[0].t1.unixTime();
  unsigned t3 = t[0].t3.unixTime();
  fprintf(fout_, "T %d\t%d", t1, t3);

  if (verb_)
    cout << "Processing IOV " << t1 << " - " << t3 << "(" << timeToString(t1) << " - " << timeToString(t3) << "\n";

  for (unsigned i = 0; i < t.size(); ++i) {
    if (t[i].t1.unixTime() != t1 || t[i].t3.unixTime() != t3) {
      throw cms::Exception("LasCor") << "Inconsitency in t1, t3: "
                                     << "t1(lme 1) =" << t1 << ", t1(lme " << (i + 1) << ") = " << t[i].t1.unixTime()
                                     << ", t3(lme 1) =" << t3 << ", t3(lme " << (i + 1) << ") = " << t[i].t3.unixTime()
                                     << "\n";
    }
    fprintf(fout_, "\t%d", t[i].t2.unixTime());
  }
  fputc('\n', fout_);
  fflush(fout_);

  for (int ieta = -EBDetId::MAX_IETA; ieta <= EBDetId::MAX_IETA; ++ieta) {
    if (ieta == 0)
      continue;
    for (int iphi = EBDetId::MIN_IPHI; iphi <= EBDetId::MAX_IPHI; ++iphi) {
      if (EBDetId::validDetId(ieta, iphi)) {
        EBDetId detId(ieta, iphi);
        EcalLaserAPDPNRatios::EcalLaserAPDPNpair corr = p.barrel(detId.hashedIndex());
        fprintf(fout_, "P %d\t%f\t%f\t%f\n", (int)detId, corr.p1, corr.p2, corr.p3);
      }
    }
  }

  for (int iZ = 1; iZ >= -1; --iZ) {
    for (int iX = EEDetId::IX_MIN; iX <= EEDetId::IX_MAX; ++iX) {
      for (int iY = EEDetId::IY_MIN; iY <= EEDetId::IY_MAX; ++iY) {
        if (EEDetId::validDetId(iX, iY, iZ)) {
          EEDetId detId(iX, iY, iZ);
          EcalLaserAPDPNRatios::EcalLaserAPDPNpair corr = p.endcap(detId.hashedIndex());
          fprintf(fout_, "P %d\t%f\t%f\t%f\n", (int)detId, corr.p1, corr.p2, corr.p3);
        }
      }
    }
  }
}
//DEFINE_FWK_MODULE(EcalLaserCondTools);
