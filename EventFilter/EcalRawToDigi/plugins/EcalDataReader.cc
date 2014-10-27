//Emacs settings:-*- mode: c++; c-basic-offset: 2; indent-tabs-mode: nil -*-
/*
 * Original author: Ph. Gras CEA/Saclay
 */

#include "EventFilter/EcalRawToDigi/interface/EcalDataReader.h"

#include <iostream>
#include <fstream>
#include <iomanip>
#include <limits>
#include <algorithm>
#include <sys/time.h>

#include "TGraph.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"

#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDetId/interface/EcalScDetId.h"
#include "DataFormats/EcalDetId/interface/EcalTrigTowerDetId.h"

#include "Geometry/EcalMapping/interface/EcalElectronicsMapping.h"
#include "Geometry/EcalMapping/interface/EcalMappingRcd.h"

#include "EventFilter/EcalRawToDigi/interface/EcalDCCHeaderRuntypeDecoder.h"

//#define TIMING_TEST
//#define TIME_HIST //to produce an histogram of event processing time.
//                  BEWARE: it is not thread-safe and this part must evently
//                  be dropped.
#define FAST_SC_RETRIEVAL

#ifdef FAST_SC_RETRIEVAL
#  include "EventFilter/EcalRawToDigi/plugins/dccRu2scDetId.h"
#endif

#ifdef TIME_HIST
#  include "EventFilter/EcalRawToDigi/interface/PGHisto.h"
PGHisto histo("hist.root", "RECREATE"),
#endif //TIME_HIST defined


// FE BX counter starts at 0, while OD BX starts at 1.
// For some reason, I do not understand myself,
// Bx offset is often set such that:
//     BX_FE = BX_OD for BX_OD < 3564
// and BX_FE = BX_OD - 3564 for BX_OD = 3564
// set feBxOffset to 1 if this FE BX shift is operated, 0 otherwise
//Ph. Gras.
const int feBxOffset = 1;

const int EcalDataReader::ttId_[nTccTypes_][maxTpsPerTcc_] = {
  //EB-
  { 1,   2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
    17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
    33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
    49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64,
    65, 66, 67, 68
  },

  //EB+
  { 4,   3,  2,  1,  8,  7,  6,  5, 12, 11, 10,  9, 16, 15, 14, 13,
    20, 19, 18, 17, 24, 23, 22, 21, 28, 27, 26, 25, 32, 31, 30, 29,
    36, 35, 34, 33, 40, 39, 38, 37, 44, 43, 42, 41, 48, 47, 46, 45,
    52, 51, 50, 49, 56, 55, 54, 53, 60, 59, 58, 57, 64, 63, 62, 61,
    68, 67, 66, 65
  },

  //inner EE
  { 1,   2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
    17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,  0,  0,  0,  0,
    0,   0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
    0,   0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
    0,   0,  0,  0
  },

  //outer EE
  { 1,   2,  3,  4,  5,  6,  7,  8,
    0,   0,  0,  0,  0,  0,  0,  0,
    9,  10, 11, 12, 13, 14, 15, 16,  0,  0,
    0,   0,  0,  0,  0,  0,  0,  0,  0,
    0,   0,  0,  0,  0,  0,  0,  0,  0,
    0,   0,  0,  0,  0,  0,  0,  0,  0,
    0,   0,  0,  0,  0,  0,  0,  0,  0,
    0,   0,  0,  0,  0, 0
  }
};


const int EcalDataReader::seq2iTt0_[nTccTypes_][maxTpsPerTcc_] = {
  //EB-
  { 0,  1,   2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
    16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
    32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
    48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
    64, 65, 66, 67
  },

  //EB+
  { 3,  2,   1,  0,  7,  6,  5,  4, 11, 10,  9,  8, 15, 14, 13, 12,
    19, 18, 17, 16, 23, 22, 21, 20, 27, 26, 25, 24, 31, 30, 29, 28,
    35, 34, 33, 32, 39, 38, 37, 36, 43, 42, 41, 40, 47, 46, 45, 44,
    51, 50, 49, 48, 55, 54, 53, 52, 59, 58, 57, 56, 63, 62, 61, 60,
    67, 66, 65, 64
  },

  //inner EE
  { 0,   1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
    16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, -1, -1, -1, -1,
    -1,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1,  -1, -1, -1
  },

  //outer EE
  { 0,   1,  2,  3,  4,  5,  6,  7,  8,
    -1,  -1, -1, -1, -1, -1, -1, -1, -1,
    9,  10, 11, 12, 13, 14, 15, -1, -1,
    -1,  -1, -1, -1, -1, -1, -1, -1, -1,
    -1,  -1, -1, -1, -1, -1, -1, -1, -1,
    -1,  -1, -1, -1, -1, -1, -1, -1, -1,
    -1,  -1, -1, -1, -1, -1, -1, -1, -1,
    -1,  -1, -1, -1, -1
  }
};


//input indices start from 0
//output eta index runs from -28 to -1 and 1 to 28 (zerophobic index)
//output phi index starts at -1 at phi = -1.5 degree. It is equal to the trigger
//tower zerophobic index modulo 72.
//Phi numbering starts at one in the middle of SMs with FED ID 637 and 655
const int EcalDataReader::tccEtaPhi_[108][2] = {
  //Inner EE- TCC 1 to 18, 28 TTs per TCC
  {-22, -1}, {-22,  3}, {-22,  7}, {-22, 11}, {-22, 15}, {-22, 19}, {-22, 23}, {-22, 27}, {-22, 31},
  {-22, 35}, {-22, 39}, {-22, 43}, {-22, 47}, {-22, 51}, {-22, 55}, {-22, 59}, {-22, 63}, {-22, 67},

  //Outer EE- TCC 19 to 36, 16 TTs per TCC
  {-18, -1}, {-18,  3}, {-18,  7}, {-18, 11}, {-18, 15}, {-18, 19}, {-18, 23}, {-18, 27}, {-18, 31},
  {-18, 35}, {-18, 39}, {-18, 43}, {-18, 47}, {-18, 51}, {-18, 55}, {-18, 59}, {-18, 63}, {-18, 67},

  //EB- TCC 37 to 54, 68 TTs per TCC
  { -1, -1}, { -1,  3}, { -1,  7}, { -1, 11}, { -1, 15}, { -1, 19}, { -1, 23}, { -1, 27}, { -1, 31},
  { -1, 35}, { -1, 39}, { -1, 43}, { -1, 47}, { -1, 51}, { -1, 55}, { -1, 59}, { -1, 63}, { -1, 67},

  //EB+ TCC 55 to 72, 68 TTs per TCC
  {  1, -1}, {  1,  3}, {  1,  7}, {  1, 11}, {  1, 15}, {  1, 19}, {  1, 23}, {  1, 27}, {  1, 31},
  {  1, 35}, {  1, 39}, {  1, 43}, {  1, 47}, {  1, 51}, {  1, 55}, {  1, 59}, {  1, 63}, {  1, 67},

  //Outer EE+, TCC 73 to 90, 16 TTs per TCC
  { 18, -1}, { 18,  3}, { 18,  7}, { 18, 11}, { 18, 15}, { 18, 19}, { 18, 23}, { 18, 27}, { 18, 31},
  { 18, 35}, { 18, 39}, { 18, 43}, { 18, 47}, { 18, 51}, { 18, 55}, { 18, 59}, { 18, 63}, { 18, 67},

  //Inner EE+, TCC 91 to 108, 28 TTs per TCC
  { 22, -1}, { 22,  3}, { 22,  7}, { 22, 11}, { 22, 15}, { 22, 19}, { 22, 23}, { 22, 27}, { 22, 31},
  { 22, 35}, { 22, 39}, { 22, 43}, { 22, 47}, { 22, 51}, { 22, 55}, { 22, 59}, { 22, 63}, { 22, 67},
};

//const int EcalDataReader::tccTtOffset_[108] = {
//  //Inner EE- TCC 1 to 18, 28 TTs per TCC
//  0,     28,  56,  84, 112, 140, 168, 196, 224, 252, 280, 308, 336, 364, 392, 420, 448, 476,
//
//  //Outer EE- TCC 19 to 36, 16 TTs per TCC
//  504,  520, 536, 552, 568, 584, 600, 616, 632, 648, 664, 680, 696, 712, 728, 744, 760, 776,
//
//  //EB- TCC 37 to 54, 68 TTs per TCC
//  792,  860, 928, 996,1064,1132,1200,1268,1336,1404,1472,1540,1608,1676,1744,1812,1880,1948,
//
//  //EB+ TCC 55 to 72, 68 TTs per TCC
//  2016,2084,2152,2220,2288,2356,2424,2492,2560,2628,2696,2764,2832,2900,2968,3036,3104,3172,
//
//  //Outer EE+, TCC 73 to 90, 16 TTs per TCC
//  3240,3256,3272,3288,3304,3320,3336,3352,3368,3384,3400,3416,3432,3448,3464,3480,3496,3512,
//
//  //Inner EE+, TCC 91 to 108, 28 TTs per TCC
//  3528,3556,3584,3612,3640,3668,3696,3724,3752,3780,3808,3836,3864,3892,3920,3948,3976,4004
//};

using namespace std;

static const char* const trigNames[] = {
  "Unknown",
  "Phys",
  "Calib",
  "Test",
  "Ext",
  "Simu",
  "Trace",
  "Err"
};

static const char* const detailedTrigNames[] = {
  "?",   //000
  "?",   //001
  "?",   //010
  "?",   //011
  "Las", //100
  "Led", //101
  "TP",  //110
  "Ped"  //111
};

static const char* const colorNames[] = {
  "Blue",
  "Green",
  "Red",
  "IR"
};

static const char* const ttsNames[] = {
  "Discon'd", //0000
  "OvFWarn",  //0001
  "OoS",      //0010
  "Forb",     //0011
  "Busy",     //0100
  "Forb",     //0101
  "Forb",     //0110
  "Forb",     //0111
  "Ready",    //1000
  "Forb",     //1001
  "Idle",     //1010
  "Forb",     //1011
  "Err",      //1100
  "Forb",     //1101
  "Forb",     //1110
  "Discon'd"  //1111
};

//double mgpaGainFactors[] = {12., 1., 12./6., 12.}; //index 0->saturation
//          gain setting:     sat  12       6    1  //index 0->saturation
//          gain setting:  1(sat)     12      6        1
//index 0->saturation
static const double mgpaGainFactors[] = {10.63, 1., 10.63/5.43, 10.63};
static const double fppaGainFactors[] = {0, 1., 16./1.,  0.};

void EcalDataReader::reset(){
  simpleTrigType_   = -1;
  detailedTrigType_ = -1;
  color_ = 0;
  bx_ =  -1;
  l1a_ = -1;
  srpL1a_ = -1;
  tccL1a_[0] = tccL1a_[1] = tccL1a_[2] = tccL1a_[3] = -1;
  srpBx_ = -1;
  tccBx_[0] = tccBx_[1] = tccBx_[2] = tccBx_[3] = -1;
  thisTccId_ = -1;
  tccId_[0] = tccId_[1] =   tccId_[2] =   tccId_[3] = -1;
  iTow_ = 0;
  iRu_ = 0;
  iTcc_ = 0;
  nSamples_ = 0;
  tccType_ = 0;
  runNumber_ = -1;
  dccErrors_ = 0;
  eventLengthFromHeader_ = 0;
  runType_ = 0;
  tccStatus_ = 0;
  srStatus_ = 0;
  mf_  = 0;
  tzs_ = 0;
  zs_  = 0;
  sr_  = 0;
  strip_ = 0;
  xtalInStrip_ = 0;

  for(int i = 0; i < nRu_; ++i){
    feL1a_[i]  = -1;
    feBx_[i]   = -1;
    feRuId_[i] = -1;
    feStatus_[i] = 0;
  }
  fill(nTps_.begin(), nTps_.end(), 0);
  //  fill(l1as_.begin(), l1as_.end(), 0);
  //  fill(orbits_.begin(), orbits_.end(), 0);
  fill(tpg_.begin(), tpg_.end(), std::vector<int>(maxTpsPerTcc_));
}

EcalDataReader::EcalDataReader(const edm::ParameterSet& ps):
  iEvent_(0),
  adc_(nSamples, 0.),
  amplCut_(ps.getUntrackedParameter<double>("amplCut", 5.)),
  dump_(ps.getUntrackedParameter<bool>("dump", true)),
  dumpAdc_(ps.getUntrackedParameter<bool>("dumpAdc", true)),
  maxEvt_(ps.getUntrackedParameter<int>("maxEvt", 10000)),
  profileFedId_(ps.getUntrackedParameter<int>("profileFedId", 0)),
  profileRuId_(ps.getUntrackedParameter<int>("profileRuId", 1)),
  l1aMinX_(ps.getUntrackedParameter<int>("l1aMinX", 1)),
  l1aMaxX_(ps.getUntrackedParameter<int>("l1aMaxX", 601)),
  ecalRawDataCollection_(ps.getParameter<edm::InputTag>("ecalRawDataCollection")),
  ebDigiCollection_(ps.getParameter<std::string>("ebDigiCollection")),
  eeDigiCollection_(ps.getParameter<std::string>("eeDigiCollection")),
  ebSrFlagCollection_(ps.getParameter<std::string>("ebSrFlagCollection")),
  eeSrFlagCollection_(ps.getParameter<std::string>("eeSrFlagCollection")),
  tpgCollection_(ps.getParameter<std::string>("tpgCollection")),
  dccHeaderCollection_(ps.getParameter<std::string>("dccHeaderCollection")),
  produceDigis_(ps.getParameter<bool>("produceDigis")),
  produceSrfs_(ps.getParameter<bool>("produceSrfs")),
  produceTps_(ps.getParameter<bool>("produceTpgs")),
  produceDccHeaders_(ps.getParameter<bool>("produceDccHeaders")),
  producePnDiodeDigis_(ps.getParameter<bool>("producePnDiodeDigis")),
  producePseudoStripInputs_(ps.getParameter<bool>("producePseudoStripInputs")),
  produceBadChannelList_(ps.getParameter<bool>("produceBadChannelList")),
  lastOrbit_(nDccs_, std::numeric_limits<uint32_t>::max()),
  eventId_(std::numeric_limits<unsigned>::max()),
  minEventId_(999999),
  maxEventId_(0),
  orbit0_(0),
  orbit0Set_(false),
  l1amin_(std::numeric_limits<int>::max()),
  l1amax_(-std::numeric_limits<int>::min()),
  tpg_(maxTccsPerDcc_, std::vector<int>(maxTpsPerTcc_)),
  nTps_(maxTccsPerDcc_, 0),
  tccL1a_(maxTccsPerDcc_,-1),
  tccBlockLen64_(19),
  feL1a_(nRu_,-1),
  tccBx_(maxTccsPerDcc_, -1),
  feBx_(nRu_,-1),
  feRuId_(nRu_,-1),
  pulsePerRu_(ps.getUntrackedParameter<bool>("pulsePerRu", true)),
  pulsePerLmod_(ps.getUntrackedParameter<bool>("pulsePerLmod", true)),
  pulsePerLme_(ps.getUntrackedParameter<bool>("pulsePerLme", true)),
  feStatus_(nRu_, 0),
  elecMap_(0)
{
  reset();

  verbosity_= ps.getUntrackedParameter<int>("verbosity",1);
  beg_fed_id_= ps.getUntrackedParameter<int>("beg_fed_id",601);
  end_fed_id_= ps.getUntrackedParameter<int>("end_fed_id",654);
  first_event_ = ps.getUntrackedParameter<int>("first_event",1);
  last_event_  = ps.getUntrackedParameter<int>("last_event",
                                               numeric_limits<int>::max());
  writeDcc_ = ps.getUntrackedParameter<bool>("writeDCC",false);
  filename_  = ps.getUntrackedParameter<string>("dccBinDumpFileName","dump.bin");
  if(writeDcc_){
    dumpFile_.open(filename_.c_str());
    if(dumpFile_.bad()){
      edm::LogError("EcalDataReader") << "Failed to open file '"
                                      << filename_.c_str() << "' specified by "
                                      << "parameter filename for writing. DCC data "
        " dump will be disabled.";
      writeDcc_ = false;
    }
  }

#ifndef TIMING_TEST
  if(produceTps_) produces<EcalTrigPrimDigiCollection>(tpgCollection_);
  if(produceSrfs_){
    produces<EBSrFlagCollection>(ebSrFlagCollection_);
    produces<EESrFlagCollection>(eeSrFlagCollection_);
  }
  if(produceDigis_){
    produces<EBDigiCollection>(ebDigiCollection_);
    produces<EEDigiCollection>(eeDigiCollection_);
  }
  if(produceDccHeaders_){
    produces<EcalRawDataCollection>(dccHeaderCollection_);
  }

  if(producePnDiodeDigis_){
    produces<EcalPnDiodeDigiCollection>();
  }

  if(producePseudoStripInputs_){
    produces<EcalPSInputDigiCollection>("EcalPseudoStripInputs");
  }

  if(produceBadChannelList_){
    // Integrity for xtal data
    produces<EBDetIdCollection>("EcalIntegrityGainErrors");
    produces<EBDetIdCollection>("EcalIntegrityGainSwitchErrors");
    produces<EBDetIdCollection>("EcalIntegrityChIdErrors");

    // Integrity for xtal data - EE specific (to be rivisited towards EB+EE common collection)
    produces<EEDetIdCollection>("EcalIntegrityGainErrors");
    produces<EEDetIdCollection>("EcalIntegrityGainSwitchErrors");
    produces<EEDetIdCollection>("EcalIntegrityChIdErrors");

    // Integrity Errors
    produces<EcalElectronicsIdCollection>("EcalIntegrityTTIdErrors");
    produces<EcalElectronicsIdCollection>("EcalIntegrityZSXtalIdErrors");
    produces<EcalElectronicsIdCollection>("EcalIntegrityBlockSizeErrors");

    // Mem channels' integrity
    produces<EcalElectronicsIdCollection>("EcalIntegrityMemTtIdErrors");
    produces<EcalElectronicsIdCollection>("EcalIntegrityMemBlockSizeErrors");
    produces<EcalElectronicsIdCollection>("EcalIntegrityMemChIdErrors");
    produces<EcalElectronicsIdCollection>("EcalIntegrityMemGainErrors");
  }

#endif //TIMING_TEST
}

void EcalDataReader::beginJob(){
}

void EcalDataReader::endJob(){
}

EcalDataReader::~EcalDataReader(){
}

// ------------ method called to analyze the data  ------------
void
EcalDataReader::produce(edm::Event& event, const edm::EventSetup& es){

  if(!elecMap_ && (produceSrfs_ || produceDigis_)){
    edm::ESHandle<EcalElectronicsMapping> ecalMapping;
    es.get<EcalMappingRcd>().get(ecalMapping);
    elecMap_ = ecalMapping.product();
  }

  ++iEvent_;
  eventId_ = event.id().event();

  if ((first_event_ > 0 && iEvent_ < first_event_)
      || (last_event_ > 0 && last_event_ < iEvent_)) return;

  timeval start;
  timeval stop;
  gettimeofday(&start, 0);

  edm::Handle<FEDRawDataCollection> rawdata;
  event.getByLabel(ecalRawDataCollection_, rawdata);


  if(dump_) cout << "\n----------------------------------------------------------------------\n"
                 << toNth(iEvent_)
                 << " read event. "
                 << "Event id: "
                 << " " << eventId_
                 << "\n----------------------------------------------------------------------\n";

  if(eventId_ < minEventId_) minEventId_ = eventId_;
  if(eventId_ > maxEventId_) maxEventId_ = eventId_;

#if 1

  bool dccIdErr = false;
  //  bool outOfSync = false;
  unsigned iFed = 0;
  unsigned refDccId = 0;
  //  int refSimpleTrigType = -1;
  //  int refBx = -1;
  initCollections();

  for (int id = 0; id<=FEDNumbering::lastFEDId(); ++id){

    if (id < beg_fed_id_ || end_fed_id_ < id) continue;

    //    const FEDRawData& data = rawdata->FEDData(id);
    const FEDRawData& data = rawdata->FEDData(id);

    if (data.size()>4){
      ++iFed;
      if ((data.size() % 8) !=0){
        cout << "***********************************************\n";
        cout << " Fed size in bits not multiple of 64, strange.\n";
        cout << "***********************************************n";
      }


      size_t nWord32 = data.size()/4;
      const uint32_t * pData = ( reinterpret_cast<uint32_t*>(const_cast<unsigned char*> ( data.data())));
      stringstream s;

      reset();

      bool rc;
      for(size_t iWord32=0; iWord32 < nWord32; iWord32+=2){
        s.str("");
        rc = decode(pData+iWord32, iWord32/2, s);
        if(rc && dump_){
          cout << setfill('0') << hex
               << "[" << setw(8) << iWord32*4 << "] "
               <<        setw(4) << (pData[iWord32+1]>>16 & 0xFFFF) << " "
               <<        setw(4) << (pData[iWord32+1]>>0  & 0xFFFF) << " "
               <<        setw(4) << (pData[iWord32]>>16 & 0xFFFF) << " "
               <<        setw(4) << (pData[iWord32]>>0  & 0xFFFF) << " "
               << setfill(' ') << dec
               << s.str()
               << "\n";
        }
      }
      if(dump_) cout << "\n";

      if(iFed==1){
        refDccId = dccId_;
        //        refSimpleTrigType = simpleTrigType_;
        //        refBx = bx_;
      } else{
        if(dccId_!=refDccId){
          dccIdErr = true;
        }
        //        if(refBx!=bx_){
        //          outOfSync = true;
        //}
      }

      if(dump_) cout << flush; //flushing cout before writing to cerr

      if(srpBx_!=-1 && srpBx_!=bx_){
        cerr << "Bx discrepancy between SRP and DCC, Bx(SRP) = "
             << srpBx_ << ", Bx(DCC) = " << bx_
             << " in " << toNth(iEvent_) << " event, FED "
             << id << "\n";
      }

      for(unsigned i = 0; i < nTccs_ && i < maxTccsPerDcc_; ++i){
        if(tccBx_[i]!=-1 && tccBx_[i]!=bx_){
          cerr << "Bx discrepancy between TCC and DCC, Bx(TCC) = "
               << srpBx_ << ", Bx(DCC) = " << bx_
               << " in " << toNth(iEvent_) << " event, FED "
               << id << "\n";
        }
      }

      bool feBxErr = false;
      for(int i=0; i < nRu_; ++i){
        int expectedFeBx;
        if(feBxOffset==0){
          expectedFeBx = bx_ - 1;
        } else{
          expectedFeBx = (bx_==3564) ? 0 : bx_;
        }
        if(feBx_[i]!=-1 && feBx_[i]!=expectedFeBx) feBxErr = true;
      }
      if(feBxErr) cerr << "Bx discrepancy between DCC and at least one FE"
                       << " in " << toNth(iEvent_) << " event, FED "
                       << id << "\n";


      int localL1a = l1a_ & 0xFFF;
      if(srpL1a_!=-1 && srpL1a_!=localL1a){
        cerr << "Discrepancy between SRP and DCC L1a counter, L1a(SRP) = "
             << srpL1a_ << ", L1a(DCC) & 0xFFF = " << localL1a
             << " in " << toNth(iEvent_) << " event, FED "
             << id << "\n";

      }

      for(unsigned i = 0; i < nTccs_ && i < maxTccsPerDcc_; ++i){
        if(tccL1a_[i] !=  -1 && tccL1a_[i] != localL1a){
          cerr << "Discrepancy between TCC and DCC L1a counter, L1a(TCC) = "
               << srpL1a_ << ", L1a(DCC) & 0xFFF = " << localL1a
               << " in " << toNth(iEvent_) << " event, FED "
               << id << "\n";

        }
      }

      bool feL1aErr = false;
      for(int i=0; i < nRu_; ++i){
        if(feL1a_[i]!=-1 && feL1a_[i]!=localL1a-1){
          cerr << "FE L1A error for RU " << (i+1) << endl;
          feL1aErr = true;
        }
      }
      if(feL1aErr) cerr << "Discrepancy in L1a counter between DCC "
                     "and at least one FE (L1A(DCC) & 0xFFF = " << localL1a << ")"
                        << " in " << toNth(iEvent_) << " event, FED "
                        << id << "\n";


      if(iTow_>0 && iTow_< nRu_ && feRuId_[iTow_] < feRuId_[iTow_-1]){
        cerr << "Error in RU ID (TT/SC ID)"
             << " in " << toNth(iEvent_) << " event, FED "
             << id << "\n";
      }

      if (beg_fed_id_ <= id && id <= end_fed_id_ && writeDcc_){
        dumpFile_.write( reinterpret_cast <const char *> (pData), nWord32*4);
      }
    } else{
      //      cout << "No data for FED " <<  id << ". Size = "
      //     << data.size() << " byte(s).\n";
    }

    //Fills DCC header collection:
    if(produceDccHeaders_){
#ifndef TIMING_TEST
      setDccHeader();
      dccHeaderColl_->push_back(dccHeader_);
#endif
    }

  } //next fed

  putCollections(event);

  if(dump_) cout << "Number of selected FEDs with a data block: "
                 << iFed << "\n";

  if(dccIdErr){
    cerr << "DCC ID discrepancy in detailed trigger type "
         << " of " << toNth(iEvent_) << " event.\n";
  }

#endif

  gettimeofday(&stop, 0);
  //  double dt  = (stop.tv_sec-start.tv_sec)*1.e3
  //  + (stop.tv_usec-start.tv_usec)*1.e-3;
  //  cerr << "dt = " << dt << "\n";
#ifdef TIME_HIST
  histo.fillD("hCodeTime", "Code execution time;Duration (ms);Event count",
               PGXAxis(100, 0, 15),
               dt);
#endif //TIMEHIST
}

string EcalDataReader::toNth(int n){
  stringstream s;
  s << n;
  if(n%100<10 || n%100>20){
    switch(n%10){
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
  } else{
    s << "th";
  }
  return s.str();
}


bool EcalDataReader::decode(const uint32_t* data, int iWord64, ostream& out){
  bool rc = true;
  const bool d  = dump_;
  if(iWord64==0){//start of event
    iSrWord64_ = 0;
    iTccWord64_ = 0;
    iTowerWord64_ = 0;
  }
  int dataType = (data[1] >>28) & 0xF;
  const int boe = 5;
  const int eoe = 10;
  if(dataType==boe){//Begin of Event header
    /**********************************************************************
     *  DAQ header
     *
     **********************************************************************/
    simpleTrigType_ = (data[1] >>24) & 0xF;
    l1a_ = (data[1]>>0 )  & 0xFFffFF;
    bx_ = (data[0] >>20) & 0xFFF;
    fedId_ = (data[0] >>8 ) & 0xFFF;
    if(d) out << "Trigger type: " << simpleTrigType_
              << "(" << trigNames[(data[1]>>24) & 0xF] << ")"
              << " L1A: "         << l1a_
              << " BX: "          << bx_
              << " FED ID: "      << fedId_
              << " FOV: "         << ((data[0] >>4 ) & 0xF)
              << " H: "           << ((data[0] >>3 ) & 0x1);
  } else if((dataType>>2)==0){//DCC header
    /**********************************************************************
     * ECAL DCC header
     *
     **********************************************************************/
    int dccHeaderId = (data[1] >>24) & 0x3F;
    switch(dccHeaderId){
    case 1:
      runNumber_ = (data[1] >>0 ) & 0xFFFFFF;
      dccErrors_ = ((data[0] >>24) & 0xFF);
      eventLengthFromHeader_ = ((data[0] >>0 ) & 0xFFFFFF);
      if(d) out << "Run #: "     << runNumber_
                << " DCC Err: "  << dccErrors_
                << " Evt Len:  " << eventLengthFromHeader_ << "\n";
      break;
    case 2:
      runType_ = data[0];
      side_ = (data[1] >>11) & 0x1;
      detailedTrigType_ = (data[1] >>8 ) & 0x7;
      dccId_ = (data[1] >>0 ) & 0x3F;
      color_ = (data[1] >>6 ) & 0x3;
      if(d) out << "DCC FOV: " << ((data[1] >>12) & 0xF)
                << " Side: "   << side_
                << " Trig.: "   << detailedTrigType_
                << " (" << detailedTrigNames[(data[1]>>8)&0x7] << ")"
                << " Color: "  << color_
                << " (" << colorNames[(data[1]>>6)&0x3] << ")"
                << " DCC ID: " << dccId_;
      break;
    case 3:
      {
        tccStatus_ = (data[1]>>8) & 0xFFFF;
        srStatus_ = (data[1] >>4 )& 0xF;
        mf_ =  (data[1] >>3 ) & 0x1;
        tzs_ = (data[1] >>2 ) & 0x1;
        zs_  = (data[1] >>1 ) & 0x1;
        sr_  = (data[1] >>0 ) & 0x1;
        if(d) out << "TCC Status ch<4..1>: 0x"
                  << hex << tccStatus_ << dec
                  << " SR status: " << srStatus_
                  << " MF: "        << mf_
                  << " TZS: "       << tzs_
                  << " ZS: "        << zs_
                  << " SR: "        << sr_;
        orbit_ = data[0];
        if(d) out << " Orbit: "     << orbit_;
        if(!orbit0Set_){
          orbit0_ = orbit_;
          orbit0Set_ = true;
        }
        int iDcc0 = fedId_-fedStart_;
        if((unsigned)iDcc0<nDccs_){
          if(lastOrbit_[iDcc0]!=numeric_limits<uint32_t>::max()){
            if(d) out << " (+" << (int)orbit_-(int)lastOrbit_[iDcc0] <<")";
          }
          lastOrbit_[iDcc0] = orbit_;
        }
      }
      break;
    case 4:
    case 5:
    case 6:
    case 7:
    case 8:
      {
        int chOffset = (dccHeaderId-4)*14;

        feStatus_[13+chOffset] = ((data[1] >>20) & 0xF);
        feStatus_[12+chOffset] = ((data[1] >>16) & 0xF);
        feStatus_[11+chOffset] = ((data[1] >>12) & 0xF);
        feStatus_[10+chOffset] = ((data[1] >>8 ) & 0xF);
        feStatus_[ 9+chOffset] = ((data[1] >>4 ) & 0xF);
        feStatus_[ 8+chOffset] = ((data[1] >>0)  & 0xF);
        feStatus_[ 7+chOffset] = ((data[0] >>28) & 0xF);
        feStatus_[ 6+chOffset] = ((data[0] >>24) & 0xF);
        feStatus_[ 5+chOffset] = ((data[0] >>20) & 0xF);
        feStatus_[ 4+chOffset] = ((data[0] >>16) & 0xF);
        feStatus_[ 3+chOffset] = ((data[0] >>12) & 0xF);
        feStatus_[ 2+chOffset] = ((data[0] >>8 ) & 0xF);
        feStatus_[ 1+chOffset] = ((data[0] >>4 ) & 0xF);
        feStatus_[ 0+chOffset] = ((data[0] >>0 ) & 0xF);

        if(d) out << "FE CH status:"
                  << " #" << 14+chOffset << ":" << feStatus_[13+chOffset]
                  << " #" << 13+chOffset << ":" << feStatus_[12+chOffset]
                  << " #" << 12+chOffset << ":" << feStatus_[11+chOffset]
                  << " #" << 11+chOffset << ":" << feStatus_[10+chOffset]
                  << " #" << 10+chOffset << ":" << feStatus_[ 9+chOffset]
                  << " #" <<  9+chOffset << ":" << feStatus_[ 8+chOffset]
                  << " #" <<  8+chOffset << ":" << feStatus_[ 7+chOffset]
                  << " #" <<  7+chOffset << ":" << feStatus_[ 6+chOffset]
                  << " #" <<  6+chOffset << ":" << feStatus_[ 5+chOffset]
                  << " #" <<  5+chOffset << ":" << feStatus_[ 4+chOffset]
                  << " #" <<  4+chOffset << ":" << feStatus_[ 3+chOffset]
                  << " #" <<  3+chOffset << ":" << feStatus_[ 2+chOffset]
                  << " #" <<  2+chOffset << ":" << feStatus_[ 1+chOffset]
                  << " #" <<  1+chOffset << ":" << feStatus_[ 0+chOffset];
      }
      break;
    default:
      if(d) out << " bits<63..62>=0 (DCC header) bits<61..56>=" << dccHeaderId
                << "(unknown=>ERROR?)";
    }
  } else if((dataType>>1)==3){//TCC block
    /**********************************************************************
     * TCC block
     *
     **********************************************************************/
    if(iTccWord64_==0){
      //header
      int tccL1a = (data[1] >>0 ) & 0xFFF;
      thisTccId_ = (data[0] >>0 ) & 0xFF;
      int nTps =  (data[1] >>16) & 0x7F;
      int thisTccBx = (data[0] >>16) & 0xFFF;

      if(iTcc_ < maxTccsPerDcc_){
        tccL1a_[iTcc_] = tccL1a;
        tccId_[iTcc_]  = thisTccId_;
        nTps_[iTcc_]   = nTps;
        tccBx_[iTcc_]   = thisTccBx;
      }
      ++iTcc_;
      if(d) out << "LE1: "         << ((data[1] >>28) & 0x1)
                << " LE0: "        << ((data[1] >>27) & 0x1)
                << " N_samples: "  << setw(2) << ((data[1] >>23) & 0x1F)
                << " N_TTs: "      << nTps
                << " E1: "         << ((data[1] >>12) & 0x1)
                << " L1A: "        << setw(4) << tccL1a
                << " '3': "        << ((data[0] >>29) & 0x7)
                << " E0: "         << ((data[0] >>28) & 0x1)
                << " Bx: "         << setw(4) << thisTccBx
                << " TTC ID: "     << setw(3) << thisTccId_;
      if(nTps==68){ //EB TCC (TCC68)
        if(fedId_ < 628) tccType_ = ebmTcc_;
        else tccType_ = ebpTcc_;
      } else if(nTps == 16){//Inner EE TCC (TCC48)
        tccType_ = eeOuterTcc_;
      } else if(nTps == 28){//Outer EE TCC (TCC48)
        tccType_ = eeInnerTcc_;
      } else {
        cerr << "Error in #TT field of TCC block."
          "This field is normally used to determine type of TCC "
          "(TCC48 or TCC68). Type of TCC will be deduced from the TCC ID.\n";
        if(thisTccId_< 19) tccType_ = eeInnerTcc_;
        else if(thisTccId_<  37) tccType_ = eeOuterTcc_;
        else if(thisTccId_<  55) tccType_ = ebmTcc_;
        else if(thisTccId_<  73) tccType_ = ebpTcc_;
        else if(thisTccId_<  91) tccType_ = eeOuterTcc_;
        else if(thisTccId_< 109) tccType_ = eeInnerTcc_;
        else{
          cerr << "TCC ID is also invalid. EB- TCC type will be assumed.\n";
          tccType_ = ebmTcc_;
        }
      }
      tccBlockLen64_ = (tccType_==ebmTcc_ || tccType_==ebpTcc_) ? 18 : 9;
    } else{// if(iTccWord64_<18){
      int tpgOffset = (iTccWord64_-1)*4;
      if(iTcc_ > maxTccsPerDcc_){
        out << "Too many TCC blocks";
      } else if(tpgOffset > (int(maxTpsPerTcc_) - 4)){
        out << "Too many TPG in one TCC block";
      } else{
        newTpg(iTcc_, 3 + tpgOffset, (data[1] >>16) & 0x1FFF);
        newTpg(iTcc_, 2 + tpgOffset, (data[1] >>0 ) & 0x1FFF);
        newTpg(iTcc_, 1 + tpgOffset, (data[0] >>16) & 0x1FFF);
        newTpg(iTcc_, 0 + tpgOffset, (data[0] >>0 ) & 0x1FFF);
        if(d) out << tpgTag(tccType_, 3+tpgOffset) << ":"
                  << prettyTp(tpg_[iTcc_-1][3+tpgOffset]) << " "

                  << tpgTag(tccType_, 2+tpgOffset) << ":"
                  << prettyTp(tpg_[iTcc_-1][2+tpgOffset])  << " "

                  << " '3': "                     << ((data[0] >>29) & 0x7) << " "

                  << tpgTag(tccType_, 1+tpgOffset) << ": "
                  << prettyTp(tpg_[iTcc_-1][1+tpgOffset]) << " "

                  << tpgTag(tccType_, 0+tpgOffset) << ":"
                  << prettyTp(tpg_[iTcc_-1][0+tpgOffset]);

      }
    }// else{
     // if(d) out << "ERROR";
    //}
    ++iTccWord64_;
    if(iTccWord64_ >= (int64_t) tccBlockLen64_) iTccWord64_ = 0;
  } else if((dataType>>1)==4){//SRP block
    /**********************************************************************
     * SRP block
     *
     **********************************************************************/
    if(iSrWord64_==0){//header
      srpL1a_ = (data[1] >>0 ) & 0xFFF;
      srpBx_ = (data[0] >>16) & 0xFFF;
      if(d) out << "LE1: "     << ((data[1] >>28) & 0x1)
                << " LE0: "    << ((data[1] >>27) & 0x1)
                << " N_SRFs: " << ((data[1] >>16) & 0x7F)
                << " E1: "     << ((data[1] >>12) & 0x1)
                << " L1A: "    << srpL1a_
                << " '4': "    << ((data[0] >>29) & 0x7)
                << " E0: "     << ((data[0] >>28) & 0x1)
                << " Bx: "     << srpBx_
                << " SRP ID: " << ((data[0] >>0 ) & 0xFF);
    } else if(iSrWord64_<6){
      int srfOffset = (iSrWord64_-1)*16;
      if(iSrWord64_<5){
        newSrf(srfOffset + 15, (data[1] >>25) & 0x7);
        newSrf(srfOffset + 14, (data[1] >>22) & 0x7);
        newSrf(srfOffset + 13, (data[1] >>19) & 0x7);
        newSrf(srfOffset + 12, (data[1] >>16) & 0x7);
        newSrf(srfOffset + 11, (data[1] >>9)  & 0x7);
        newSrf(srfOffset + 10, (data[1] >>6)  & 0x7);
        newSrf(srfOffset +  9, (data[1] >>3)  & 0x7);
        newSrf(srfOffset +  8, (data[1] >>0)  & 0x7);
        newSrf(srfOffset +  7, (data[0] >>25) & 0x7);
        newSrf(srfOffset +  6, (data[0] >>22) & 0x7);
        newSrf(srfOffset +  5, (data[0] >>19) & 0x7);
        newSrf(srfOffset +  4, (data[0] >>16) & 0x7);
        if(d){
          out << "SRF# " << setw(6) << right << srRange(12+srfOffset) << ": "
              << setfill('0') << oct << setw(4)
              << ((data[1] >>16) & 0xFFF)
              << setfill(' ') << dec
              << " SRF# " << srRange(8+srfOffset) << ": "
              << setfill('0') << oct << setw(4)
              << ((data[1] >>0 ) & 0xFFF)
              << setfill(' ') << dec
              << " '4':" << ((data[0] >>29) & 0x7)
              << " SRF# " << srRange(4+srfOffset) << ": "
              << setfill('0') << oct << setw(4)
              << ((data[0] >>16) & 0xFFF)
              << setfill(' ') << dec;
        }
      } else{//last 64-bit word has only 4 SRFs.
        if(d) out << "                                                           ";
      }
      newSrf(srfOffset + 3, (data[0] >>9) & 0x7);
      newSrf(srfOffset + 2, (data[0] >>6) & 0x7);
      newSrf(srfOffset + 1, (data[0] >>3) & 0x7);
      newSrf(srfOffset,     (data[0] >>0) & 0x7);

      if(d) out << " SRF# " << srRange(srfOffset) << ": "
                << setfill('0') << oct << setw(4)
                << ((data[0] >>0 ) & 0xFFF)
                << setfill(' ') << dec;
    } else{
      if(d) out << "ERROR";
    }
    ++iSrWord64_;
  } else if((dataType>>2)==3){//Tower block
    /**********************************************************************
     * "Tower" block (crystal channel data from a RU (=1 FE cards))
     *
     **********************************************************************/
    if(iTowerWord64_==0){//header
      towerBlockLength_ = (data[1]>>16) & 0x1FF;
      int l1a;
      int bx;
      l1a = (data[1] >>0 ) & 0xFFF;
      bx  = (data[0] >>16) & 0xFFF;
      dccCh_ = (data[0] >>0 ) & 0xFF;
      nSamples_ = ((data[0] >>8 ) & 0x7F);
      if(d) out << "Block Len: "  << towerBlockLength_
                << " E1: "        << ((data[1] >>12) & 0x1)
                << " L1A: "       << l1a
                << " '3': "       << ((data[0] >>30) & 0x3)
                << " E0: "        << ((data[0] >>28) & 0x1)
                << " Bx: "        << bx
                << " N_samples: " << nSamples_
                << " RU ID: "     << dccCh_;
      if(iRu_ < nRu_){
        feL1a_[iRu_] = l1a;
        feBx_[iRu_] = bx;
        feRuId_[iRu_] = dccCh_;
        ++iRu_;
      }
    } else if((unsigned)iTowerWord64_<towerBlockLength_){
      if(!dumpAdc_){
        //no output.
        rc = false;
      }
      const bool da = dumpAdc_ && dump_;
      switch((iTowerWord64_-1)%3){
        int s[4];
        int g[4];
      case 0:
        pDataFrame_ = (uint16_t*) data + 1;
        s[0]=(data[0] >>16) & 0xFFF;
        g[0]=(data[0] >>28) & 0x3;
        s[1]=(data[1] >>0 ) & 0xFFF;
        g[1]=(data[1] >>12) & 0x3;
        s[2]=(data[1] >>16) & 0xFFF;
        g[2]=(data[1] >>28) & 0x3;
        strip_ = (data[0] >>0 ) & 0x7;
        xtalInStrip_ = (data[0] >>4 ) & 0x7;
        fill(adc_.begin(), adc_.end(), 0.);
        if(da) out << "GMF: "    << ((data[0] >>11) & 0x1)
                   << " SMF: "   << ((data[0] >>9 ) & 0x1)
                   << " M: "     << ((data[0] >>8 ) & 0x1)
                   << " XTAL: "  << xtalInStrip_
                   << " STRIP: " << strip_
                   << " " << setw(4) << s[0]
                   << "G" << g[0]
                   << " " << setw(4) << s[1]
                   << "G" << g[1]
                   << " " << setw(4) << s[2]
                   << "G" << g[2];
        for(int i=0; i<3; ++i) adc_[i] = s[i]*mgpaGainFactors[g[i]];
        break;
      case 1:
        s[0]=(data[0] >>0 ) & 0xFFF;
        g[0]=(data[0] >>12) & 0x3;
        s[1]=(data[0] >>16) & 0xFFF;
        g[1]=(data[0] >>28) & 0x3;
        s[2]=(data[1] >>0 ) & 0xFFF;
        g[2]=(data[1] >>12) & 0x3;
        s[3]=(data[1] >>16) & 0xFFF;
        g[3]=(data[1] >>28) & 0x3;
        if(da) out << "                                   "
                   << " " << setw(4) << s[0]
                   << "G" << g[0]
                   << " " << setw(4) << s[1]
                   << "G" << g[1]
                   << " " << setw(4) << s[2]
                   << "G" << g[2]
                   << " " << setw(4) << s[3]
                   << "G" << g[3];
        for(int i=0; i<4; ++i) adc_[i+3] = s[i]*mgpaGainFactors[g[i]];
        break;
      case 2:
        if(da) out << "TZS: " << ((data[1] >>14) & 0x1);

        s[0]=(data[0] >>0 ) & 0xFFF;
        g[0]=(data[0] >>12) & 0x3;
        s[1]=(data[0] >>16) & 0xFFF;
        g[1]=(data[0] >>28) & 0x3;
        s[2]=(data[1] >>0 ) & 0xFFF;
        g[2]=(data[1] >>12) & 0x3  ;

        for(int i=0; i<3; ++i) adc_[i+7] = s[i]*mgpaGainFactors[g[i]];
        if(dccCh_<=68){
          unsigned bom0; //Bin of Maximum, starting counting from 0
          double ampl = max(adc_, bom0)-min(adc_);
          if(da) out << " Ampl: " << setw(4) << ampl
                     << (ampl>amplCut_?"*":" ")
                     << " BoM:" << setw(2) << (bom0+1)
                     << "          ";
        } else{
          if(da) out << setw(29) << "";
        }
        if(da) out << " " << setw(4) << s[0]
                   << "G" << g[0]
                   << " " << setw(4) << s[1]
                   << "G" << g[1]
                   << " " << setw(4) << s[2]
                   << "G" << g[2];
        newDataFrame();
        break;
      default:
        assert(false);
      }
    } else {
      if(d) out << "ERROR";
    }
    ++iTowerWord64_;
    if(iTowerWord64_>=towerBlockLength_){
      iTowerWord64_-=towerBlockLength_;
      ++dccCh_;
    }
  } else if(dataType==eoe){//End of event trailer
    /**********************************************************************
     * Event DAQ trailer
     *
     **********************************************************************/
    int tts = (data[0] >>4)  & 0xF;
    if(d) out << "Evt Len.: "    << ((data[1] >>0 ) & 0xFFFFFF)
              << " CRC16: "       << ((data[0] >>16) & 0xFFFF)
              << " Evt Status: "  << ((data[0] >>8 ) & 0xF)
              << " TTS: "         << tts
              << " (" << ttsNames[tts] << ")"
              << " T:"            << ((data[0] >>3)  & 0x1);
  } else{
    if(d) out << " incorrect 64-bit word type marker (see MSBs)";
  }
  return rc;
}

std::string EcalDataReader::prettyTp(int tp) const {
  int spike = (tp >>12) & 0x1;
  int ttf   = (tp >>9)  & 0x7;
  int et    = (tp >>0)  & 0x1FF;
  std::stringstream buf;
  buf << spike << " " << setw(3) << ttf << " " << setw(3) << et;
  return buf.str();
}

int EcalDataReader::lme(int dcc1, int side){
  int fedid = ((dcc1-1)%600) + 600; //to handle both FED and DCC id.
  vector<int> lmes;
  // EE -
  if( fedid <= 609 ) {
    if ( fedid <= 607 ) {
      lmes.push_back(fedid-601+83);
    } else if ( fedid == 608 ) {
      lmes.push_back(90);
      lmes.push_back(91);
    } else if ( fedid == 609 ) {
      lmes.push_back(92);
    }
  } //EB
  else if ( fedid >= 610  && fedid <= 645 ) {
    lmes.push_back(2*(fedid-610)+1);
    lmes.push_back(lmes[0]+1);
  } // EE+
  else if ( fedid >= 646 ) {
    if ( fedid <= 652 ) {
      lmes.push_back(fedid-646+73);
    } else if ( fedid == 653 ) {
      lmes.push_back(80);
      lmes.push_back(81);
    } else if ( fedid == 654 ) {
      lmes.push_back(82);
    }
  }
  return lmes.size()==0?-1:lmes[std::min(lmes.size(), (size_t)side)];
}


int EcalDataReader::sideOfRu(int ru1){
  if(ru1 < 5 || (ru1-5)%4 >= 2){
    return 0;
  } else{
    return 1;
  }
}


int EcalDataReader::modOfRu(int ru1){
  int iEta0 = (ru1-1)/4;
  if(iEta0<5){
    return 1;
  } else{
    return 2 + (iEta0-5)/4;
  }
}

int EcalDataReader::lmodOfRu(int ru1){
  int iEta0 = (ru1-1)/4;
  int iPhi0 = (ru1-1)%4;
  int rs;
  if(iEta0==0){
    rs =  1;
  } else{
    rs = 2 + ((iEta0-1)/4)*2 + (iPhi0%4)/2;
  }
  //  cout << "ru1 = " << ru1 << " -> lmod = " << rs << "\n";
  return rs;
}

std::string EcalDataReader::srRange(int offset) const{
  int min = offset+1;
  int max = offset+4;
  stringstream buf;
  if(628 <= fedId_ && fedId_ <= 646){//EB+
    buf << right << min << ".."
        << left  << max;
  } else{
    buf << right << max << ".."
        << left  << min;
  }
  string s = buf.str();
  buf.str("");
  buf << setw(6) << right << s;
  return buf.str();
}

std::string EcalDataReader::ttfTag(int tccType, unsigned iSeq) const{
  if((unsigned)iSeq > sizeof(ttId_))
    throw cms::Exception("OutOfRange")
      << __FILE__ << ":"  << __LINE__ << ": "
      << "parameter out of range\n";

  const int ttId = ttId_[tccType][iSeq];
  stringstream buf;
  buf.str("");
  if(ttId==0){
    buf << "   '0'";
  } else{
    buf << "TTF# " << setw(2) << ttId;
  }
  return buf.str();
}

std::string EcalDataReader::tpgTag(int tccType, unsigned iSeq) const{
  if((unsigned)iSeq > sizeof(ttId_))
    throw cms::Exception("OutOfRange")
      << __FILE__ << ":"  << __LINE__ << ": "
      << "parameter out of range\n";

  const int ttId = ttId_[tccType][iSeq];
  stringstream buf;
  buf.str("");
  if(ttId==0){
    buf << "   '0'";
  } else{
    //    buf << "TPG# " << setw(2) << ttId;
    buf << "TP# " << setw(2) << ttId;
  }
  return buf.str();
}

// EcalTrigTowerDetId EcalDataReader::ttSeq2ttDetId(int iSeq) const{
//   if(tccId_ < 1 || tccId_ > 108
//      || iSeq < 0  || iSeq > 68) return EcalTrigTowerDetId();
//   const int iTtInTcc0 = seq2iTt0_[tccType_][iSeq];
//   if(iTtInTcc0<0) return EcalTrigTowerDetId();
//   const int iTt0 = tccTtOffset_[iTcc_] + iTtInTcc0;
//   div_t d = div(iTt0, 4);
//   const int zside = (d.quot >= 28) ? +1 : -1;
//   const int iTtEtaAbs = abs(d.quot-28) + (zside>=0?1:0);
//   const int iTtPhi = 1 + d.rem;
//   std::cout << "---> " << iTtEtaAbs << "\t" << iTtPhi << std::endl;
//   return EcalTrigTowerDetId(zside, (iTtEtaAbs<18?EcalBarrel:EcalEndcap), iTtEtaAbs, iTtPhi);
//}

EcalTrigTowerDetId EcalDataReader::ttSeq2ttDetId(int iSeq) const{
  if(thisTccId_< 1 || thisTccId_> 108
     || iSeq < 0  || iSeq > 68) return EcalTrigTowerDetId();

  int iTtInTcc0;
  if(tccType_ == ebmTcc_ || tccType_ == ebpTcc_){
    iTtInTcc0 = iSeq;
  } else{
    iTtInTcc0 = ttId_[tccType_][iSeq] - 1;
  }
  if(iTtInTcc0 < 0) return EcalTrigTowerDetId();
  //  const int iTt0 = tccTtOffset_[iTcc_] + iTtInTcc0;
  div_t d = div(std::abs(iTtInTcc0), 4);
  const int zside     = (tccEtaPhi_[thisTccId_ - 1][0] > 0 ) ? +1 : -1;
  const int iTtEtaAbs = std::abs(tccEtaPhi_[thisTccId_ - 1][0]) + d.quot;
  int iTtPhi = tccEtaPhi_[thisTccId_ - 1][1] + d.rem;
  if(iTtPhi < 1) iTtPhi += 72;

  // if(fedId_ == 610){
  //   std::cout << "---> iSeq = " << iSeq << std::endl;
  //   std::cout << "---> tccId_ = " << tccId_ << std::endl;
  //   std::cout << "---> tccType_ = " << tccType_ << std::endl;
  //   std::cout << "---> iTtInTcc0 = " << iTtInTcc0 << std::endl;
  //   std::cout << "---> iTtEtaAbs = " << iTtEtaAbs << std::endl;
  //   std::cout << "---> iTtPhi = " << iTtPhi << std::endl;
  //   std::cout << "---> d.rem = " << d.rem << std::endl;
  //   std::cout << "---> d.quot = " << d.quot << std::endl;
  //   std::cout << "--> " << EcalTrigTowerDetId(zside, (iTtEtaAbs < 18 ? EcalBarrel : EcalEndcap), iTtEtaAbs, iTtPhi) << std::endl;
  //}
  return EcalTrigTowerDetId(zside, (iTtEtaAbs < 18 ? EcalBarrel : EcalEndcap), iTtEtaAbs, iTtPhi);
}

void EcalDataReader::newTpg(int iTcc, int iSeq, int val){
  tpg_[iTcc-1][iSeq] = val;
  if(produceTps_){
    try{
      const EcalTrigTowerDetId& ttId = ttSeq2ttDetId(iSeq);
      if(ttId==EcalTrigTowerDetId()) return;
#ifndef TIMING_TEST
      tpgColl_->push_back(EcalTriggerPrimitiveDigi(ttId));
      tpgColl_->back().setSize(1);
      tpgColl_->back().setSample(0, val);
#endif
    } catch(cms::Exception){
    }
  }
}

EcalTrigTowerDetId EcalDataReader::ebSrfSeq2ttDetId(int iSeq) const{
  const int iEbmDcc0 = (fedId_ - 610) % 18;
  const int zside = (fedId_ < 628) ? -1 : 1;

  const int iEbSmTt0 = iSeq;
  div_t d = div(iEbSmTt0, 4);
  const int iTtEtaAbs1 = 1 + d.quot;
  int iTtSmPhi0 = d.rem;
  if(zside > 0) iTtSmPhi0 = 3 - iTtSmPhi0;

  int iTtPhi1 = -1 + iEbmDcc0 * 4 + iTtSmPhi0;
  if(iTtPhi1 < 1) iTtPhi1 += 72;

  return EcalTrigTowerDetId(zside, EcalBarrel, iTtEtaAbs1, iTtPhi1);
}

void EcalDataReader::newSrf(int iSeq, int val){
  if(produceSrfs_){
    if((610 <= fedId_) && (fedId_ <= 645)){//barrel
      try{
        EcalTrigTowerDetId ttId = ebSrfSeq2ttDetId(iSeq);
#ifndef TIMING_TEST
        ebSrfColl_->push_back(EBSrFlag(ttId, val));
#endif
      } catch(cms::Exception){
        // PRINTVALN(fedId_);
        // PRINTVALN(iSeq);
      }
    } else if((601 <= fedId_) && (fedId_ <= 654)){//endcap
      if(iSeq<0 || iSeq > 68) return;
      try{
#ifndef TIMING_TEST
#  ifdef FAST_SC_RETRIEVAL
        div_t d = div(fedId_ - 601, 45);
        for(int isc = 0; isc < 3; ++isc){
          const int v = EcalScDetId(dccRu2scDetid[d.quot][d.rem][iSeq][isc]);
          if(v > 0) eeSrfColl_->push_back(EESrFlag(v, val));
        }
#  else
        vector<EcalScDetId> scIds = elecMap_->getEcalScDetId(fedId_-600, iSeq+1);
        for(size_t i = 0; i < scIds.size(); ++i){
          eeSrfColl_->push_back(EESrFlag(scIds[i], val));
        }
#  endif
#endif
      } catch(cms::Exception){
        /*NO-OP*/
      }
    } else{
      std::cerr << __FILE__ << ":" << __LINE__ << ": "
                << "Invalid ECAL FED ID, " << fedId_ << "\n";
    }
  }
}

void EcalDataReader::newDataFrame(){
  if(produceDigis_){
    DetId xtalId = elecMap_->getDetId(EcalElectronicsId(fedId_-600, dccCh_, strip_, xtalInStrip_));
    if(xtalId==DetId()) return;
#ifndef TIMING_TEST
    if(610 <= fedId_ && fedId_ <= 645){//barrel
      ebDigiColl_->push_back(xtalId.rawId(), pDataFrame_);
    } else{//endcap
      eeDigiColl_->push_back(xtalId.rawId(), pDataFrame_);
    }
#endif
  }
}

void EcalDataReader::initCollections(){
#ifndef TIMING_TEST
  if(produceDigis_){
    ebDigiColl_ = auto_ptr<EBDigiCollection>(new EBDigiCollection);
    eeDigiColl_ = auto_ptr<EEDigiCollection>(new EEDigiCollection);
  }

  if(produceSrfs_){
    ebSrfColl_ = auto_ptr<EBSrFlagCollection>(new EBSrFlagCollection);
    eeSrfColl_ = auto_ptr<EESrFlagCollection>(new EESrFlagCollection);
  }

  if(produceTps_){
    tpgColl_ = auto_ptr<EcalTrigPrimDigiCollection>(new EcalTrigPrimDigiCollection);
  }

  if(produceDccHeaders_){
    dccHeaderColl_ = auto_ptr<EcalRawDataCollection>(new EcalRawDataCollection);
  }
#endif
}

void EcalDataReader::putCollections(edm::Event& event){
#ifndef TIMING_TEST
  if(produceDigis_){
    ebDigiColl_->sort();
    event.put(ebDigiColl_, ebDigiCollection_);
    eeDigiColl_->sort();
    event.put(eeDigiColl_, eeDigiCollection_);
  }

  if(produceSrfs_){
    ebSrfColl_->sort();
    event.put(ebSrfColl_, ebSrFlagCollection_);
    eeSrfColl_->sort();
    event.put(eeSrfColl_, eeSrFlagCollection_);
  }

  if(produceTps_){
    tpgColl_->sort();
    event.put(tpgColl_, tpgCollection_);
  }

  if(produceDccHeaders_){
    dccHeaderColl_->sort();
    event.put(dccHeaderColl_, dccHeaderCollection_);
  }


  if(producePnDiodeDigis_){
    event.put(auto_ptr<EcalPnDiodeDigiCollection>(new EcalPnDiodeDigiCollection()), "");
  }

  if(producePseudoStripInputs_){
    event.put(auto_ptr<EcalPSInputDigiCollection>(new EcalPSInputDigiCollection()), "EcalPseudoStripInputs");
  }

  if(produceBadChannelList_){
    // Integrity for xtal data
    event.put(auto_ptr<EBDetIdCollection>(new EBDetIdCollection()), "EcalIntegrityGainErrors");
    event.put(auto_ptr<EBDetIdCollection>(new EBDetIdCollection()), "EcalIntegrityGainSwitchErrors");
    event.put(auto_ptr<EBDetIdCollection>(new EBDetIdCollection()), "EcalIntegrityChIdErrors");

    // Integrity for xtal data - EE specific (to be rivisited towards EB+EE common collection)
    event.put(auto_ptr<EEDetIdCollection>(new EEDetIdCollection()), "EcalIntegrityGainErrors");
    event.put(auto_ptr<EEDetIdCollection>(new EEDetIdCollection()), "EcalIntegrityGainSwitchErrors");
    event.put(auto_ptr<EEDetIdCollection>(new EEDetIdCollection()), "EcalIntegrityChIdErrors");

    // Integrity Errors
    event.put(auto_ptr<EcalElectronicsIdCollection>(new EcalElectronicsIdCollection()), "EcalIntegrityTTIdErrors");
    event.put(auto_ptr<EcalElectronicsIdCollection>(new EcalElectronicsIdCollection()), "EcalIntegrityZSXtalIdErrors");
    event.put(auto_ptr<EcalElectronicsIdCollection>(new EcalElectronicsIdCollection()), "EcalIntegrityBlockSizeErrors");

    // Mem channels' integrity
    event.put(auto_ptr<EcalElectronicsIdCollection>(new EcalElectronicsIdCollection()), "EcalIntegrityMemTtIdErrors");
    event.put(auto_ptr<EcalElectronicsIdCollection>(new EcalElectronicsIdCollection()), "EcalIntegrityMemBlockSizeErrors");
    event.put(auto_ptr<EcalElectronicsIdCollection>(new EcalElectronicsIdCollection()), "EcalIntegrityMemChIdErrors");
    event.put(auto_ptr<EcalElectronicsIdCollection>(new EcalElectronicsIdCollection()), "EcalIntegrityMemGainErrors");

  }



#endif
}

void EcalDataReader::setDccHeader(){
  EcalDCCHeaderRuntypeDecoder decoder;
  decoder.Decode(simpleTrigType_, detailedTrigType_, runType_, &dccHeader_);
  dccHeader_.setId(fedId_-600);
  dccHeader_.setFedId(fedId_);
  dccHeader_.setErrors(dccErrors_);
  dccHeader_.setDccInTTCCommand(dccId_);
  dccHeader_.setRunNumber(runNumber_);
  dccHeader_.setLV1(l1a_);
  dccHeader_.setBX(bx_);
  dccHeader_.setOrbit(orbit_);
  dccHeader_.setSelectiveReadout(sr_);
  dccHeader_.setZeroSuppression(zs_);
  dccHeader_.setTestZeroSuppression(tzs_);
  dccHeader_.setSrpStatus(srStatus_);
  std::vector<short> tccStatus(4);
  tccStatus[0] = tccStatus_ & 0xF;
  tccStatus[1] = (tccStatus_ >>4 )& 0xF;
  tccStatus[2] = (tccStatus_ >>8 )& 0xF;
  tccStatus[3] = (tccStatus_ >>12)& 0xF;
  dccHeader_.setTccStatus(tccStatus) ;
  dccHeader_.setFEStatus(feStatus_) ;
  dccHeader_.setFEBx(feBx_);
  dccHeader_.setTCCBx(tccBx_);
  dccHeader_.setSRPBx(srpBx_);
  dccHeader_.setFELv1(feL1a_);
  dccHeader_.setTCCLv1(tccL1a_);
  dccHeader_.setSRPLv1(srpL1a_);
}
