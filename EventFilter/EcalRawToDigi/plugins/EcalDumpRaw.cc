//emacs settings:-*- mode: c++; c-basic-offset: 2; indent-tabs-mode: nil -*-
/*
 *
 * Author: Ph Gras. CEA/IRFU - Saclay
 *
 */

#include "EventFilter/EcalRawToDigi/interface/EcalDumpRaw.h"

#include <iostream>
#include <fstream>
#include <iomanip>
#include <limits>
#include <algorithm>
#include <sys/time.h>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/Scalers/interface/L1AcceptBunchCrossing.h"

// FE BX counter starts at 0, while OD BX starts at 1.
// For some reason, I do not understand myself,
// Bx offset is often set such that:
//     BX_FE = BX_OD for BX_OD < 3564
// and BX_FE = BX_OD - 3564 for BX_OD = 3564
// set feBxOffset to 1 if this FE BX shift is operated, 0 otherwise
//Ph. Gras.
const int feBxOffset = 1;

const int EcalDumpRaw::ttId_[nTccTypes_][maxTpgsPerTcc_] = {
    //EB-
    {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
     24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46,
     47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68},

    //EB+
    {4,  3,  2,  1,  8,  7,  6,  5,  12, 11, 10, 9,  16, 15, 14, 13, 20, 19, 18, 17, 24, 23, 22,
     21, 28, 27, 26, 25, 32, 31, 30, 29, 36, 35, 34, 33, 40, 39, 38, 37, 44, 43, 42, 41, 48, 47,
     46, 45, 52, 51, 50, 49, 56, 55, 54, 53, 60, 59, 58, 57, 64, 63, 62, 61, 68, 67, 66, 65},

    //inner EE
    {1,  2,  3,  4,  5,  6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
     24, 25, 26, 27, 28, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
     0,  0,  0,  0,  0,  0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0},

    //outer EE
    {1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0, 0, 0, 0, 0, 0, 9, 10, 11, 12, 13, 14, 15, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0}};

using namespace std;

static const char* const trigNames[] = {"Unknown", "Phys", "Calib", "Test", "Ext", "Simu", "Trace", "Err"};

static const char* const detailedTrigNames[] = {
    "?",    //000
    "?",    //001
    "?",    //010
    "?",    //011
    "Las",  //100
    "Led",  //101
    "TP",   //110
    "Ped"   //111
};

static const char* const colorNames[] = {"Blue", "Green", "Red", "IR"};

static const char* const ttsNames[] = {
    "Discon'd",  //0000
    "OvFWarn",   //0001
    "OoS",       //0010
    "Forb",      //0011
    "Busy",      //0100
    "Forb",      //0101
    "Forb",      //0110
    "Forb",      //0111
    "Ready",     //1000
    "Forb",      //1001
    "Idle",      //1010
    "Forb",      //1011
    "Err",       //1100
    "Forb",      //1101
    "Forb",      //1110
    "Discon'd"   //1111
};

//double mgpaGainFactors[] = {12., 1., 12./6., 12.}; //index 0->saturation
//          gain setting:     sat  12       6    1  //index 0->saturation
//          gain setting:  1(sat)     12      6        1
//index 0->saturation
static const double mgpaGainFactors[] = {10.63, 1., 10.63 / 5.43, 10.63};

EcalDumpRaw::EcalDumpRaw(const edm::ParameterSet& ps)
    : iEvent_(0),
      adc_(nSamples, 0.),
      amplCut_(ps.getUntrackedParameter<double>("amplCut", 5.)),
      dump_(ps.getUntrackedParameter<bool>("dump", true)),
      dumpAdc_(ps.getUntrackedParameter<bool>("dumpAdc", true)),
      l1aHistory_(ps.getUntrackedParameter<bool>("l1aHistory", true)),
      //  doHisto_(ps.getUntrackedParameter<bool>("doHisto", true)),
      maxEvt_(ps.getUntrackedParameter<int>("maxEvt", 10000)),
      profileFedId_(ps.getUntrackedParameter<int>("profileFedId", 0)),
      profileRuId_(ps.getUntrackedParameter<int>("profileRuId", 1)),
      l1aMinX_(ps.getUntrackedParameter<int>("l1aMinX", 1)),
      l1aMaxX_(ps.getUntrackedParameter<int>("l1aMaxX", 601)),
      lastOrbit_(nDccs_, numeric_limits<uint32_t>::max()),
      eventId_(numeric_limits<unsigned>::max()),
      eventList_(ps.getUntrackedParameter<vector<unsigned> >("eventList", vector<unsigned>())),
      minEventId_(999999),
      maxEventId_(0),
      orbit0_(0),
      orbit0Set_(false),
      bx_(-1),
      l1a_(-1),
      simpleTrigType_(-1),
      detailedTrigType_(-1),
      //  histo_("hist.root", "RECREATE"),
      l1as_(36 + 2),
      orbits_(36 + 2),
      tpg_(maxTccsPerDcc_, std::vector<int>(maxTpgsPerTcc_)),
      nTpgs_(maxTccsPerDcc_, 0),
      dccChStatus_(70, 0),
      srpL1a_(-1),
      tccL1a_(-1),
      nTts_(-1),
      tccBlockLen64_(19),
      feL1a_(nRu_, -1),
      srpBx_(-1),
      tccBx_(-1),
      tccType_(0),
      feBx_(nRu_, -1),
      feRuId_(nRu_, -1),
      iTow_(0),
      pulsePerRu_(ps.getUntrackedParameter<bool>("pulsePerRu", true)),
      pulsePerLmod_(ps.getUntrackedParameter<bool>("pulsePerLmod", true)),
      pulsePerLme_(ps.getUntrackedParameter<bool>("pulsePerLme", true)),
      tccId_(0),
      fedRawDataCollectionTag_(ps.getParameter<edm::InputTag>("fedRawDataCollectionTag")),
      l1AcceptBunchCrossingCollectionTag_(ps.getParameter<edm::InputTag>("l1AcceptBunchCrossingCollectionTag")) {
  verbosity_ = ps.getUntrackedParameter<int>("verbosity", 1);

  beg_fed_id_ = ps.getUntrackedParameter<int>("beg_fed_id", 601);
  end_fed_id_ = ps.getUntrackedParameter<int>("end_fed_id", 654);

  first_event_ = ps.getUntrackedParameter<int>("first_event", 1);
  last_event_ = ps.getUntrackedParameter<int>("last_event", numeric_limits<int>::max());

  writeDcc_ = ps.getUntrackedParameter<bool>("writeDCC", false);
  filename_ = ps.getUntrackedParameter<string>("filename", "dump.bin");

  fedRawDataCollectionToken_ = consumes<FEDRawDataCollection>(fedRawDataCollectionTag_);
  l1AcceptBunchCrossingCollectionToken_ =
      consumes<L1AcceptBunchCrossingCollection>(l1AcceptBunchCrossingCollectionTag_);

  if (writeDcc_) {
    dumpFile_.open(filename_.c_str());
    if (dumpFile_.bad()) {
      /*edm::LogError("EcalDumpRaw")*/ std::cout << "Failed to open file '" << filename_.c_str() << "' specified by "
                                                 << "parameter filename for writing. DCC data "
                                                    " dump will be disabled.";
      writeDcc_ = false;
    }
  }
}

void EcalDumpRaw::endJob() {}

EcalDumpRaw::~EcalDumpRaw() {}

// ------------ method called to analyze the data  ------------
void EcalDumpRaw::analyze(const edm::Event& event, const edm::EventSetup& es) {
  ++iEvent_;
  eventId_ = event.id().event();

  if (!eventList_.empty() && find(eventList_.begin(), eventList_.end(), eventId_) == eventList_.end()) {
    cout << "Skipping event " << eventId_ << ".\n";
    return;
  }

  if ((first_event_ > 0 && iEvent_ < first_event_) || (last_event_ > 0 && last_event_ < iEvent_))
    return;
  timeval start;
  timeval stop;
  gettimeofday(&start, nullptr);

  edm::Handle<FEDRawDataCollection> rawdata;
  event.getByToken(fedRawDataCollectionToken_, rawdata);

  if (dump_ || l1aHistory_)
    cout << "\n======================================================================\n"
         << toNth(iEvent_) << " read event. "
         << "Event id: "
         << " " << eventId_ << "\n----------------------------------------------------------------------\n";

  if (l1aHistory_) {
    edm::Handle<L1AcceptBunchCrossingCollection> l1aHist;
    event.getByToken(l1AcceptBunchCrossingCollectionToken_, l1aHist);
    if (!l1aHist.isValid()) {
      cout << "L1A history not found.\n";
    } else if (l1aHist->empty()) {
      cout << "L1A history is empty.\n";
    } else {
      cout << "L1A history: \n";
      for (L1AcceptBunchCrossingCollection::const_iterator it = l1aHist->begin(); it != l1aHist->end(); ++it) {
        cout << "L1A offset: " << it->l1AcceptOffset() << "\t"
             << "BX: " << it->bunchCrossing() << "\t"
             << "Orbit ID: " << it->orbitNumber() << "\t"
             << "Trigger type: " << it->eventType() << " (" << trigNames[it->eventType() & 0xF] << ")\n";
      }
    }
    cout << "----------------------------------------------------------------------\n";
  }

  if (eventId_ < minEventId_)
    minEventId_ = eventId_;
  if (eventId_ > maxEventId_)
    maxEventId_ = eventId_;

#if 1

  bool dccIdErr = false;
  unsigned iFed = 0;
  unsigned refDccId = 0;
  //  static bool recordNextPhys = false;
  //static int bxCalib = -1;
  //x static int nCalib = 0;

  for (int id = 0; id <= FEDNumbering::lastFEDId(); ++id) {
    if (id < beg_fed_id_ || end_fed_id_ < id)
      continue;

    const FEDRawData& data = rawdata->FEDData(id);

    if (data.size() > 4) {
      ++iFed;
      if ((data.size() % 8) != 0) {
        cout << "***********************************************\n";
        cout << " Fed size in bits not multiple of 64, strange.\n";
        cout << "***********************************************\n";
      }

      size_t nWord32 = data.size() / 4;
      const uint32_t* pData = (reinterpret_cast<uint32_t*>(const_cast<unsigned char*>(data.data())));
      stringstream s;
      srpL1a_ = -1;
      tccL1a_ = -1;
      srpBx_ = -1;
      tccBx_ = -1;
      iTow_ = 0;
      iRu_ = 0;
      nTts_ = -1;
      iTcc_ = 0;
      tccType_ = 0;

      for (int i = 0; i < nRu_; ++i) {
        feL1a_[i] = -1;
        feBx_[i] = -1;
        feRuId_[i] = -1;
      }

      fill(nTpgs_.begin(), nTpgs_.end(), 0);

      fill(dccChStatus_.begin(), dccChStatus_.end(), 0);

      bool rc;
      for (size_t iWord32 = 0; iWord32 < nWord32; iWord32 += 2) {
        s.str("");
        if (id >= 601 && id <= 654) {  // ECAL DCC data
          rc = decode(pData + iWord32, iWord32 / 2, s);
        } else {
          rc = true;
        }
        if (rc && dump_) {
          cout << setfill('0') << hex << "[" << setw(8) << iWord32 * 4 << "] " << setw(4)
               << (pData[iWord32 + 1] >> 16 & 0xFFFF) << " " << setw(4) << (pData[iWord32 + 1] >> 0 & 0xFFFF) << " "
               << setw(4) << (pData[iWord32] >> 16 & 0xFFFF) << " " << setw(4) << (pData[iWord32] >> 0 & 0xFFFF) << " "
               << setfill(' ') << dec << s.str() << "\n";
        }
      }

      if (iFed == 1) {
        refDccId = dccId_;
      } else {
        if (dccId_ != refDccId) {
          dccIdErr = true;
        }
      }

      if (dump_)
        cout << flush;  //flushing cout before writing to cerr

      if (srpBx_ != -1 && srpBx_ != bx_) {
        cerr << "Bx discrepancy between SRP and DCC, Bx(SRP) = " << srpBx_ << ", Bx(DCC) = " << bx_ << " in "
             << toNth(iEvent_) << " event, FED " << id << endl;
      }

      if (tccBx_ != -1 && tccBx_ != bx_) {
        cerr << "Bx discrepancy between TCC and DCC, Bx(TCC) = " << srpBx_ << ", Bx(DCC) = " << bx_ << " in "
             << toNth(iEvent_) << " event, FED " << id << endl;
      }

      bool feBxErr = false;
      for (int i = 0; i < nRu_; ++i) {
        int expectedFeBx;
        if (feBxOffset == 0) {
          expectedFeBx = bx_ - 1;
        } else {
          expectedFeBx = (bx_ == 3564) ? 0 : bx_;
        }
        if (feBx_[i] != -1 && feBx_[i] != expectedFeBx) {
          cerr << "BX error for " << toNth(i + 1) << " RU, RU ID " << feRuId_[i];
          if ((unsigned)feRuId_[i] <= dccChStatus_.size()) {
            bool detected = (dccChStatus_[feRuId_[i] - 1] == 10 || dccChStatus_[feRuId_[i] - 1] == 11);
            cerr << (detected ? " " : " not ") << "detected by DCC (ch status: " << dccChStatus_[feRuId_[i] - 1] << ")";
          }
          cerr << " in " << toNth(iEvent_) << " event, FED " << id << "." << endl;

          feBxErr = true;
        }
      }
      if (feBxErr)
        cerr << "Bx discrepancy between DCC and at least one FE"
             << " in " << toNth(iEvent_) << " event, FED " << id << "\n";

      int localL1a = l1a_ & 0xFFF;
      if (srpL1a_ != -1 && srpL1a_ != localL1a) {
        cerr << "Discrepancy between SRP and DCC L1a counter, L1a(SRP) = " << srpL1a_
             << ", L1a(DCC) & 0xFFF = " << localL1a << " in " << toNth(iEvent_) << " event, FED " << id << endl;
      }

      if (tccL1a_ != -1 && tccL1a_ != localL1a) {
        cerr << "Discrepancy between TCC and DCC L1a counter, L1a(TCC) = " << srpL1a_
             << ", L1a(DCC) & 0xFFF = " << localL1a << " in " << toNth(iEvent_) << " event, FED " << id << endl;
      }

      bool feL1aErr = false;
      for (int i = 0; i < nRu_; ++i) {
        if (feL1a_[i] != -1 && feL1a_[i] != ((localL1a - 1) & 0xFFF)) {
          cerr << "FE L1A error for " << toNth(i + 1) << " RU, RU ID " << feRuId_[i];
          if ((unsigned)feRuId_[i] <= dccChStatus_.size()) {
            bool detected = (dccChStatus_[feRuId_[i] - 1] == 9 || dccChStatus_[feRuId_[i] - 1] == 11);
            cerr << (detected ? " " : " not ") << "detected by DCC (ch status: " << dccChStatus_[feRuId_[i] - 1] << ")";
          }
          cerr << " in " << toNth(iEvent_) << " event, FED " << id << "." << endl;
          feL1aErr = true;
        }
      }
      if (feL1aErr)
        cerr << "Discrepancy in L1a counter between DCC "
                "and at least one FE (L1A(DCC) & 0xFFF = "
             << localL1a << ")"
             << " in " << toNth(iEvent_) << " event, FED " << id << "\n";

      if (iTow_ > 0 && iTow_ < nRu_ && feRuId_[iTow_] < feRuId_[iTow_ - 1]) {
        cerr << "Error in RU ID (TT/SC ID)"
             << " in " << toNth(iEvent_) << " event, FED " << id << endl;
      }

      if (beg_fed_id_ <= id && id <= end_fed_id_ && writeDcc_) {
        dumpFile_.write(reinterpret_cast<const char*>(pData), nWord32 * 4);
      }

      if (dump_)
        cout << "\n";
    } else {
      //      cout << "No data for FED " <<  id << ". Size = "
      //     << data.size() << " byte(s).\n";
    }
  }  //next fed

  if (dump_)
    cout << "Number of selected FEDs with a data block: " << iFed << "\n";

  if (dccIdErr) {
    cout << flush;
    cerr << "DCC ID discrepancy in detailed trigger type "
         << " of " << toNth(iEvent_) << " event." << endl;
  }

#endif

  gettimeofday(&stop, nullptr);
  //  double dt  = (stop.tv_sec-start.tv_sec)*1.e3
  //  + (stop.tv_usec-start.tv_usec)*1.e-3;
  //  histo_.fillD("hCodeTime", "Code execution time;Duration (ms);Event count",
  //             PGXAxis(100, 0, 100),
  //             dt);
}

string EcalDumpRaw::toNth(int n) {
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

bool EcalDumpRaw::decode(const uint32_t* data, int iWord64, ostream& out) {
  bool rc = true;
  const bool d = dump_;
  if (iWord64 == 0) {  //start of event
    iSrWord64_ = 0;
    iTccWord64_ = 0;
    iTowerWord64_ = 0;
  }
  int dataType = (data[1] >> 28) & 0xF;
  const int boe = 5;
  const int eoe = 10;
  if (dataType == boe) {  //Begin of Event header
    /**********************************************************************
     *  DAQ header
     *
     **********************************************************************/
    simpleTrigType_ = (data[1] >> 24) & 0xF;
    l1a_ = (data[1] >> 0) & 0xFFffFF;
    bx_ = (data[0] >> 20) & 0xFFF;
    fedId_ = (data[0] >> 8) & 0xFFF;
    if (d)
      out << "Trigger type: " << simpleTrigType_ << "(" << trigNames[(data[1] >> 24) & 0xF] << ")"
          << " L1A: " << l1a_ << " BX: " << bx_ << " FED ID: " << fedId_ << " FOV: " << ((data[0] >> 4) & 0xF)
          << " H: " << ((data[0] >> 3) & 0x1);
  } else if ((dataType >> 2) == 0) {  //DCC header
    /**********************************************************************
     * ECAL DCC header
     *
     **********************************************************************/
    int dccHeaderId = (data[1] >> 24) & 0x3F;
    switch (dccHeaderId) {
      case 1:
        if (d)
          out << "Run #: " << ((data[1] >> 0) & 0xFFFFFF) << " DCC Err: " << ((data[0] >> 24) & 0xFF)
              << " Evt Len:  " << ((data[0] >> 0) & 0xFFFFFF);
        break;
      case 2:
        side_ = (data[1] >> 11) & 0x1;
        detailedTrigType_ = (data[1] >> 8) & 0x7;
        dccId_ = (data[1] >> 0) & 0x3F;
        if (d)
          out << "DCC FOV: " << ((data[1] >> 16) & 0xF) << " Side: " << side_ << " Trig.: " << detailedTrigType_ << " ("
              << detailedTrigNames[(data[1] >> 8) & 0x7] << ")"
              << " Color: " << ((data[1] >> 6) & 0x3) << " (" << colorNames[(data[1] >> 6) & 0x3] << ")"
              << " DCC ID: " << dccId_;
        break;
      case 3: {
        if (d)
          out << "TCC Status ch<4..1>: 0x" << hex << ((data[1] >> 8) & 0xFFFF) << dec
              << " SR status: " << ((data[1] >> 4) & 0xF) << " TZS: " << ((data[1] >> 2) & 0x1)
              << " ZS: " << ((data[1] >> 1) & 0x1) << " SR: " << ((data[1] >> 0) & 0x1);
        orbit_ = data[0];
        if (d)
          out << " Orbit: " << orbit_;
        if (!orbit0Set_) {
          orbit0_ = orbit_;
          orbit0Set_ = true;
        }
        int iDcc0 = fedId_ - fedStart_;
        if ((unsigned)iDcc0 < nDccs_) {
          if (lastOrbit_[iDcc0] != numeric_limits<uint32_t>::max()) {
            if (d)
              out << " (+" << (int)orbit_ - (int)lastOrbit_[iDcc0] << ")";
          }
          lastOrbit_[iDcc0] = orbit_;
        }
      } break;
      case 4:
      case 5:
      case 6:
      case 7:
      case 8: {
        int chOffset = (dccHeaderId - 4) * 14;
        dccChStatus_[13 + chOffset] = ((data[1] >> 20) & 0xF);
        dccChStatus_[12 + chOffset] = ((data[1] >> 16) & 0xF);
        dccChStatus_[11 + chOffset] = ((data[1] >> 12) & 0xF);
        dccChStatus_[10 + chOffset] = ((data[1] >> 8) & 0xF);
        dccChStatus_[9 + chOffset] = ((data[1] >> 4) & 0xF);
        dccChStatus_[8 + chOffset] = ((data[1] >> 0) & 0xF);
        dccChStatus_[7 + chOffset] = ((data[0] >> 28) & 0xF);
        dccChStatus_[6 + chOffset] = ((data[0] >> 24) & 0xF);
        dccChStatus_[5 + chOffset] = ((data[0] >> 20) & 0xF);
        dccChStatus_[4 + chOffset] = ((data[0] >> 16) & 0xF);
        dccChStatus_[3 + chOffset] = ((data[0] >> 12) & 0xF);
        dccChStatus_[2 + chOffset] = ((data[0] >> 8) & 0xF);
        dccChStatus_[1 + chOffset] = ((data[0] >> 4) & 0xF);
        dccChStatus_[0 + chOffset] = ((data[0] >> 0) & 0xF);

        if (d) {
          out << "FE CH status:";
          for (int i = chOffset; i < chOffset + 14; ++i) {
            out << " #" << (i + 1) << ":" << dccChStatus_[i];
          }
        }
      } break;
      default:
        if (d)
          out << " bits<63..62>=0 (DCC header) bits<61..56>=" << dccHeaderId << "(unknown=>ERROR?)";
    }
  } else if ((dataType >> 1) == 3) {  //TCC block
    /**********************************************************************
     * TCC block
     *
     **********************************************************************/
    if (iTccWord64_ == 0) {
      //header
      tccL1a_ = (data[1] >> 0) & 0xFFF;
      tccId_ = ((data[0] >> 0) & 0xFF);
      nTts_ = ((data[1] >> 16) & 0x7F);
      if (iTcc_ < maxTccsPerDcc_)
        nTpgs_[iTcc_] = nTts_;
      ++iTcc_;
      if (d)
        out << "LE1: " << ((data[1] >> 28) & 0x1) << " LE0: " << ((data[1] >> 27) & 0x1)
            << " N_samples: " << ((data[1] >> 23) & 0x1F) << " N_TTs: " << nTts_ << " E1: " << ((data[1] >> 12) & 0x1)
            << " L1A: " << tccL1a_ << " '3': " << ((data[0] >> 29) & 0x7) << " E0: " << ((data[0] >> 28) & 0x1)
            << " Bx: " << ((data[0] >> 16) & 0xFFF) << " TTC ID: " << tccId_;
      if (nTts_ == 68) {  //EB TCC (TCC68)
        if (fedId_ < 628)
          tccType_ = ebmTcc_;
        else
          tccType_ = ebpTcc_;
      } else if (nTts_ == 16) {  //Inner EE TCC (TCC48)
        tccType_ = eeOuterTcc_;
      } else if (nTts_ == 28) {  //Outer EE TCC (TCC48)
        tccType_ = eeInnerTcc_;
      } else {
        cout << flush;
        cerr << "Error in #TT field of TCC block."
                "This field is normally used to determine type of TCC "
                "(TCC48 or TCC68). Type of TCC will be deduced from the TCC ID.\n";
        if (tccId_ < 19)
          tccType_ = eeInnerTcc_;
        else if (tccId_ < 37)
          tccType_ = eeOuterTcc_;
        else if (tccId_ < 55)
          tccType_ = ebmTcc_;
        else if (tccId_ < 73)
          tccType_ = ebpTcc_;
        else if (tccId_ < 91)
          tccType_ = eeOuterTcc_;
        else if (tccId_ < 109)
          tccType_ = eeInnerTcc_;
        else {
          cerr << "TCC ID is also invalid. EB- TCC type will be assumed.\n";
          tccType_ = ebmTcc_;
        }
        cerr << flush;
      }
      tccBlockLen64_ = (tccType_ == ebmTcc_ || tccType_ == ebpTcc_) ? 18 : 9;
    } else {  // if(iTccWord64_<18){
      int tpgOffset = (iTccWord64_ - 1) * 4;
      if (iTcc_ > maxTccsPerDcc_) {
        out << "Too many TCC blocks";
      } else if (tpgOffset > (maxTpgsPerTcc_ - 4)) {
        out << "Too many TPG in one TCC block";
      } else {
        tpg_[iTcc_ - 1][3 + tpgOffset] = (data[1] >> 16) & 0x1FF;
        tpg_[iTcc_ - 1][2 + tpgOffset] = (data[1] >> 0) & 0x1FF;
        tpg_[iTcc_ - 1][1 + tpgOffset] = (data[0] >> 16) & 0x1FF;
        tpg_[iTcc_ - 1][0 + tpgOffset] = (data[0] >> 0) & 0x1FF;
        //int n[2][4] = {{1,2,3,4},
        //             {4,3,2,1}};
        //int iorder = (628<=fedId_ && fedId_<=645)?1:0;
        if (d)
          out << ttfTag(tccType_, 3 + tpgOffset) << ":"  //"TTF# " << setw(2) << ttId_[3 + tpgOffset] << ":"
              << ((data[1] >> 25) & 0x7) << " " << tpgTag(tccType_, 3 + tpgOffset)
              << ":"  //" TPG# "<< setw(2) << ttId_[3 + tpgOffset] << ":"
              << setw(3) << tpg_[iTcc_ - 1][3 + tpgOffset] << " " << ttfTag(tccType_, 2 + tpgOffset)
              << ":"  //" TTF# "<< setw(2) << ttId_[2 + tpgOffset] << ":"
              << ((data[1] >> 9) & 0x7) << " " << tpgTag(tccType_, 2 + tpgOffset)
              << ":"  //" TPG# "<< setw(2) << ttId_[2 + tpgOffset] << ":"
              << setw(3) << tpg_[iTcc_ - 1][2 + tpgOffset] << " "
              << " '3': " << ((data[0] >> 29) & 0x7) << " " << ttfTag(tccType_, 1 + tpgOffset)
              << ":"  //" TTF# "<< setw(2) << ttId_[1 + tpgOffset] << ":"
              << ((data[0] >> 25) & 0x7) << " " << setw(3) << tpgTag(tccType_, 1 + tpgOffset)
              << ": "  //" TPG# "<< setw(2) << ttId_[1 + tpgOffset] << ":"
              << tpg_[iTcc_ - 1][1 + tpgOffset] << " " << ttfTag(tccType_, 0 + tpgOffset)
              << ":"  //" TTF# "<< setw(2) << ttId_[0 + tpgOffset] << ":"
              << ((data[0] >> 9) & 0x7) << " " << setw(3) << tpgTag(tccType_, 0 + tpgOffset)
              << ":"  //" TPG# "<< setw(2) << ttId_[0 + tpgOffset] << ":"
              << tpg_[iTcc_ - 1][0 + tpgOffset];
      }
    }  // else{
       // if(d) out << "ERROR";
    //}
    ++iTccWord64_;
    if (iTccWord64_ >= (unsigned)tccBlockLen64_)
      iTccWord64_ = 0;
  } else if ((dataType >> 1) == 4) {  //SRP block
    /**********************************************************************
     * SRP block
     *
     **********************************************************************/
    if (iSrWord64_ == 0) {  //header
      srpL1a_ = (data[1] >> 0) & 0xFFF;
      srpBx_ = (data[0] >> 16) & 0xFFF;
      if (d)
        out << "LE1: " << ((data[1] >> 28) & 0x1) << " LE0: " << ((data[1] >> 27) & 0x1)
            << " N_SRFs: " << ((data[1] >> 16) & 0x7F) << " E1: " << ((data[1] >> 12) & 0x1) << " L1A: " << srpL1a_
            << " '4': " << ((data[0] >> 29) & 0x7) << " E0: " << ((data[0] >> 28) & 0x1) << " Bx: " << srpBx_
            << " SRP ID: " << ((data[0] >> 0) & 0xFF);
    } else if (iSrWord64_ < 6) {
      int ttfOffset = (iSrWord64_ - 1) * 16;
      if (d) {
        if (iSrWord64_ < 5) {
          out << "SRF# " << setw(6) << right
              << srRange(12 + ttfOffset) /*16+ttfOffset << "..#" << 13+ttfOffset*/ << ": " << oct
              << ((data[1] >> 16) & 0xFFF) << dec << " SRF# "
              << srRange(8 + ttfOffset) /*12+ttfOffset << "..#" << 9+ttfOffset*/ << ": " << oct
              << ((data[1] >> 0) & 0xFFF) << dec << " '4':" << ((data[0] >> 29) & 0x7) << " SRF# "
              << srRange(4 + ttfOffset) /*8+ttfOffset << "..#" << 5+ttfOffset*/ << ": " << oct
              << ((data[0] >> 16) & 0xFFF) << dec;
        } else {  //last 64-bit word has only 4 SRFs.
          out << "                                                           ";
        }
        out << " SRF# " << srRange(ttfOffset) /*4+ttfOffset << "..#" << 1+ttfOffset*/ << ": " << oct
            << ((data[0] >> 0) & 0xFFF) << dec;
      }
    } else {
      if (d)
        out << "ERROR";
    }
    ++iSrWord64_;
  } else if ((dataType >> 2) == 3) {  //Tower block
    /**********************************************************************
     * "Tower" block (crystal channel data from a RU (=1 FE cards))
     *
     **********************************************************************/
    if (iTowerWord64_ == 0) {  //header
      towerBlockLength_ = (data[1] >> 16) & 0x1FF;
      int l1a;
      int bx;
      l1a = (data[1] >> 0) & 0xFFF;
      bx = (data[0] >> 16) & 0xFFF;
      dccCh_ = (data[0] >> 0) & 0xFF;
      if (d)
        out << "Block Len: " << towerBlockLength_ << " E1: " << ((data[1] >> 12) & 0x1) << " L1A: " << l1a
            << " '3': " << ((data[0] >> 30) & 0x3) << " E0: " << ((data[0] >> 28) & 0x1) << " Bx: " << bx
            << " N_samples: " << ((data[0] >> 8) & 0x7F) << " RU ID: " << dccCh_;
      if (iRu_ < nRu_) {
        feL1a_[iRu_] = l1a;
        feBx_[iRu_] = bx;
        feRuId_[iRu_] = dccCh_;
        ++iRu_;
      }
    } else if ((unsigned)iTowerWord64_ < towerBlockLength_) {
      if (!dumpAdc_) {
        //no output.
        rc = false;
      }
      const bool da = dumpAdc_ && dump_;
      switch ((iTowerWord64_ - 1) % 3) {
        int s[4];
        int g[4];
        case 0:
          s[0] = (data[0] >> 16) & 0xFFF;
          g[0] = (data[0] >> 28) & 0x3;
          s[1] = (data[1] >> 0) & 0xFFF;
          g[1] = (data[1] >> 12) & 0x3;
          s[2] = (data[1] >> 16) & 0xFFF;
          g[2] = (data[1] >> 28) & 0x3;
          fill(adc_.begin(), adc_.end(), 0.);
          if (da)
            out << "GMF: " << ((data[0] >> 11) & 0x1) << " SMF: " << ((data[0] >> 9) & 0x1)
                << " M: " << ((data[0] >> 8) & 0x1) << " XTAL: " << ((data[0] >> 4) & 0x7)
                << " STRIP: " << ((data[0] >> 0) & 0x7) << " " << setw(4) << s[0] << "G" << g[0] << " " << setw(4)
                << s[1] << "G" << g[1] << " " << setw(4) << s[2] << "G" << g[2];
          for (int i = 0; i < 3; ++i)
            adc_[i] = s[i] * mgpaGainFactors[g[i]];
          break;
        case 1:
          s[0] = (data[0] >> 0) & 0xFFF;
          g[0] = (data[0] >> 12) & 0x3;
          s[1] = (data[0] >> 16) & 0xFFF;
          g[1] = (data[0] >> 28) & 0x3;
          s[2] = (data[1] >> 0) & 0xFFF;
          g[2] = (data[1] >> 12) & 0x3;
          s[3] = (data[1] >> 16) & 0xFFF;
          g[3] = (data[1] >> 28) & 0x3;
          if (da)
            out << "                                   "
                << " " << setw(4) << s[0] << "G" << g[0] << " " << setw(4) << s[1] << "G" << g[1] << " " << setw(4)
                << s[2] << "G" << g[2] << " " << setw(4) << s[3] << "G" << g[3];
          for (int i = 0; i < 4; ++i)
            adc_[i + 3] = s[i] * mgpaGainFactors[g[i]];
          break;
        case 2:
          if (da)
            out << "TZS: " << ((data[1] >> 14) & 0x1);

          s[0] = (data[0] >> 0) & 0xFFF;
          g[0] = (data[0] >> 12) & 0x3;
          s[1] = (data[0] >> 16) & 0xFFF;
          g[1] = (data[0] >> 28) & 0x3;
          s[2] = (data[1] >> 0) & 0xFFF;
          g[2] = (data[1] >> 12) & 0x3;

          for (int i = 0; i < 3; ++i)
            adc_[i + 7] = s[i] * mgpaGainFactors[g[i]];
          if (dccCh_ <= 68) {
            unsigned bom0;  //Bin of Maximum, starting counting from 0
            double ampl = max(adc_, bom0) - min(adc_);
            if (da)
              out << " Ampl: " << setw(4) << ampl << (ampl > amplCut_ ? "*" : " ") << " BoM:" << setw(2) << (bom0 + 1)
                  << "          ";
            if (fedId_ == dccId_ + 600  //block of the read-out SM
                //if laser, only one side:
                && (detailedTrigType_ != 4 || sideOfRu(dccCh_) == (int)side_)) {
            }
          } else {
            if (da)
              out << setw(29) << "";
          }
          if (da)
            out << " " << setw(4) << s[0] << "G" << g[0] << " " << setw(4) << s[1] << "G" << g[1] << " " << setw(4)
                << s[2] << "G" << g[2];
          break;
        default:
          assert(false);
      }
    } else {
      if (d)
        out << "ERROR";
    }
    ++iTowerWord64_;
    if (iTowerWord64_ >= towerBlockLength_) {
      iTowerWord64_ -= towerBlockLength_;
      ++dccCh_;
    }
  } else if (dataType == eoe) {  //End of event trailer
    /**********************************************************************
     * Event DAQ trailer
     *
     **********************************************************************/
    int tts = (data[0] >> 4) & 0xF;
    if (d)
      out << "Evt Len.: " << ((data[1] >> 0) & 0xFFFFFF) << " CRC16: " << ((data[0] >> 16) & 0xFFFF)
          << " Evt Status: " << ((data[0] >> 8) & 0xF) << " TTS: " << tts << " (" << ttsNames[tts] << ")"
          << " T:" << ((data[0] >> 3) & 0x1);
  } else {
    if (d)
      out << " incorrect 64-bit word type marker (see MSBs)";
  }
  return rc;
}

// The following method was not removed due to package maintainer
// (Philippe Gras <philippe.gras@cern.ch>) request.

//int EcalDumpRaw::lme(int dcc1, int side){
//  int fedid = ((dcc1-1)%600) + 600; //to handle both FED and DCC id.
//   vector<int> lmes;
//   // EE -
//   if( fedid <= 609 ) {
//     if ( fedid <= 607 ) {
//       lmes.push_back(fedid-601+83);
//     } else if ( fedid == 608 ) {
//       lmes.push_back(90);
//       lmes.push_back(91);
//     } else if ( fedid == 609 ) {
//       lmes.push_back(92);
//     }
//   } //EB
//   else if ( fedid >= 610  && fedid <= 645 ) {
//     lmes.push_back(2*(fedid-610)+1);
//     lmes.push_back(lmes[0]+1);
//   } // EE+
//   else if ( fedid >= 646 ) {
//     if ( fedid <= 652 ) {
//       lmes.push_back(fedid-646+73);
//     } else if ( fedid == 653 ) {
//       lmes.push_back(80);
//       lmes.push_back(81);
//     } else if ( fedid == 654 ) {
//       lmes.push_back(82);
//     }
//   }
//   return lmes.size()==0?-1:lmes[std::min(lmes.size(), (size_t)side)];
//}

int EcalDumpRaw::sideOfRu(int ru1) {
  if (ru1 < 5 || (ru1 - 5) % 4 >= 2) {
    return 0;
  } else {
    return 1;
  }
}

int EcalDumpRaw::modOfRu(int ru1) {
  int iEta0 = (ru1 - 1) / 4;
  if (iEta0 < 5) {
    return 1;
  } else {
    return 2 + (iEta0 - 5) / 4;
  }
}

int EcalDumpRaw::lmodOfRu(int ru1) {
  int iEta0 = (ru1 - 1) / 4;
  int iPhi0 = (ru1 - 1) % 4;
  int rs;
  if (iEta0 == 0) {
    rs = 1;
  } else {
    rs = 2 + ((iEta0 - 1) / 4) * 2 + (iPhi0 % 4) / 2;
  }
  //  cout << "ru1 = " << ru1 << " -> lmod = " << rs << "\n";
  return rs;
}

std::string EcalDumpRaw::srRange(int offset) const {
  int min = offset + 1;
  int max = offset + 4;
  stringstream buf;
  if (628 <= fedId_ && fedId_ <= 646) {  //EB+
    buf << right << min << ".." << left << max;
  } else {
    buf << right << max << ".." << left << min;
  }
  string s = buf.str();
  buf.str("");
  buf << setw(6) << right << s;
  return buf.str();
}

std::string EcalDumpRaw::ttfTag(int tccType, unsigned iSeq) const {
  if ((unsigned)iSeq > sizeof(ttId_))
    throw cms::Exception("OutOfRange") << __FILE__ << ":" << __LINE__ << ": "
                                       << "parameter out of range\n";

  const int ttId = ttId_[tccType][iSeq];
  stringstream buf;
  buf.str("");
  if (ttId == 0) {
    buf << "    '0'";
  } else {
    buf << "TTF# " << setw(2) << ttId;
  }
  return buf.str();
}

std::string EcalDumpRaw::tpgTag(int tccType, unsigned iSeq) const {
  if ((unsigned)iSeq > sizeof(ttId_))
    throw cms::Exception("OutOfRange") << __FILE__ << ":" << __LINE__ << ": "
                                       << "parameter out of range\n";

  const int ttId = ttId_[tccType][iSeq];
  stringstream buf;
  buf.str("");
  if (ttId == 0) {
    buf << "    '0'";
  } else {
    buf << "TPG# " << setw(2) << ttId;
  }
  return buf.str();
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(EcalDumpRaw);
