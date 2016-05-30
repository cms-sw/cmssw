//Emacs settings:-*- mode: c++; c-basic-offset: 2; indent-tabs-mode: nil -*-
/*
 * Original author: Ph. Gras CEA/Saclay
 */

#include "EventFilter/EcalRawToDigi/interface/EcalRawDataRecovery.h"

#include <iostream>
#include <fstream>
#include <iomanip>
#include <limits>
#include <algorithm>
#include <sys/time.h>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"

using namespace std;

EcalRawDataRecovery::EcalRawDataRecovery(const edm::ParameterSet& ps):
  ecalRawDataIn_(consumes<FEDRawDataCollection>(ps.getParameter<edm::InputTag>("inputCollection"))),
  ecalRawDataOut_(ps.getParameter<std::string>("outputCollectionLabel")),
  iEvent_(0),
  nFixedEvents_(0),
  nOkEvents_(0)
{
  produces<FEDRawDataCollection>(ecalRawDataOut_);
}

void EcalRawDataRecovery::beginJob(){
}

void EcalRawDataRecovery::endJob(){
  /*edm::LogInfo("EcalRawDataRecovery")*/ std::cout << "Number of processed events: " << iEvent_
                                      << "\nNumber of fixed run274157 events: " << nFixedEvents_
                                      << "\nNumber of run274157 events alredy ok: " << nOkEvents_;
}

EcalRawDataRecovery::~EcalRawDataRecovery(){
}

void EcalRawDataRecovery::fixRun274157(){
  //Data of run 274157 are missing the SRP data block header
  //for FED 620

  //Example of corrupted data:
  // in [], is indicated the address offset in the FEDdata expressed in bytes;
  // on left is how the data are interpreted;
  //  [000000d8] 8249 0249 8249 0249 LE1: 0 LE0: 0 N_SRFs: 73 E1: 0 L1A: 585 '4': 4 E0: 0 Bx: 585 SRP ID: 9 SRP CH: 4
  //  [000000e0] 8249 0249 8249 0249 SRF# 16..13: 1111 SRF#  12..9: 1111 '4':4 SRF#   8..5: 1111 SRF#   4..1: 1111
  //  [000000e8] 8249 0249 8249 0249 SRF# 32..29: 1111 SRF# 28..25: 1111 '4':4 SRF# 24..21: 1111 SRF# 20..17: 1111
  //  [000000f0] 8249 0a49 8249 0249 SRF# 48..45: 1111 SRF# 44..41: 5111 '4':4 SRF# 40..37: 1111 SRF# 36..33: 1111
  //  [000000f8] 8000 0000 8563 0249 SRF# 64..61: 0 SRF# 60..57: 0 '4':4 SRF# 56..53: 2543 SRF# 52..49: 1111
  //  [00000100] 8044 0d04 8257 0055                                                             SRF# 68..65: 125
  // It whould be fixed to:
  //  [000000d8] 8044 0d04 8257 0055 LE1: 0 LE0: 0 N_SRFs: 68 E1: 0 L1A: 3332 '4': 4 E0: 0 Bx: 599 SRP ID: 5 SRP CH: 5
  //  [000000e0] 8249 0249 8249 0249 SRF# 16..13: 1111 SRF#  12..9: 1111 '4':4 SRF#   8..5: 1111 SRF#   4..1: 1111
  //  [000000e8] 8249 0249 8249 0249 SRF# 32..29: 1111 SRF# 28..25: 1111 '4':4 SRF# 24..21: 1111 SRF# 20..17: 1111
  //  [000000f0] 8249 0249 8249 0249 SRF# 48..45: 1111 SRF# 44..41: 5111 '4':4 SRF# 40..37: 1111 SRF# 36..33: 1111
  //  [000000f8] 8249 0a49 8249 0249 SRF# 64..61: 0 SRF# 60..57: 0 '4':4 SRF# 56..53: 2543 SRF# 52..49: 1111
  //  [00000100] 8000 0000 8563 0249                                                       SRF# 68..65: 1111


  uint64_t fixedSrpHeader(0);
  const int srpBlockOffset = 0xd8;
  const int srpBlockSize = 49;

  size_t minDataSize = srpBlockOffset + srpBlockSize;
  
  //sanity checks:
  if(rawData_->FEDData(619).size() < minDataSize) return;
  if(rawData_->FEDData(620).size() < minDataSize) return;

  const uint64_t * pSrpDataFed619 = reinterpret_cast<const uint64_t*>(rawData_->FEDData(619).data() + srpBlockOffset);
  const uint64_t * pSrpDataFed620 = reinterpret_cast<const uint64_t*>(rawData_->FEDData(620).data() + srpBlockOffset);
  
  fixedSrpHeader = pSrpDataFed619[0];
  //update the SRP-DCC channel value for FED 620 (SRP CH: 5):
  fixedSrpHeader = (fixedSrpHeader & 0xFFFFFFFFFFFFFF0FLL) | 0x50;

  uint64_t srpHeader = pSrpDataFed620[0];
  if(srpHeader != fixedSrpHeader){//Data are corrupted
    //copy data of FED 620
    newRawData_->FEDData(620) = rawData_->FEDData(620);
    uint64_t * pSrpData = reinterpret_cast<uint64_t*>(newRawData_->FEDData(620).data() + srpBlockOffset);
    
    //Fixes SRP data block
    //The fix assumes the SRP header is missing, it shifts data and inserts the fixed header:
    pSrpData[5] = pSrpData[4];
    pSrpData[4] = pSrpData[3];
    pSrpData[3] = pSrpData[2];
    pSrpData[2] = pSrpData[1];
    pSrpData[1] = pSrpData[0];
    pSrpData[0] = fixedSrpHeader;

    //    std::cout << __FILE__ << ":" << __LINE__ << ": 0x" << hex << pSrpDataFed620[0] << " -> "
    //          << pSrpData[0] << dec << "\n";
    ++nFixedEvents_;
  } else{
    ++nOkEvents_;
  }
}

// ------------ method called to analyze the data  ------------
void
EcalRawDataRecovery::produce(edm::Event& event, const edm::EventSetup& es){
  edm::Handle<FEDRawDataCollection> rawData;
  event.getByToken(ecalRawDataIn_, rawData);
  rawData_ = rawData.product();
  newRawData_ = std::auto_ptr<FEDRawDataCollection>(new FEDRawDataCollection);

  //see http://cmsonline.cern.ch/cms-elog/923609
  if(event.run() == 274157){
    fixRun274157();
  }
  event.put(newRawData_, ecalRawDataOut_);
  ++iEvent_;
}

