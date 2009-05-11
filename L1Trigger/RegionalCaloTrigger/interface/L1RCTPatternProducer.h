// -*- C++ -*-
//
// Package:    L1RCTPatternProducer
// Class:      L1RCTPatternProducer
//
/**\class L1RCTPatternProducer L1RCTPatternProducer.cc src/L1RCTPatternProducer/src/L1RCTPatternProducer.cc

 Description:Creates patterns to input into the RCT

 Implementation: Kind of kludgy -- should think of a better way in future

*/
//
// Original Author:  Jonathan Efron
//         Created:  Jan 2008
// $Id: L1RCTSaveInput.h,
//
//

#include <memory>
#include <iostream>
#include <fstream>

#include "DataFormats/Common/interface/Handle.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/L1CaloTrigger/interface/L1CaloCollections.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"

// default scales
#include "CondFormats/L1TObjects/interface/L1CaloEcalScale.h"
#include "CondFormats/DataRecord/interface/L1CaloEcalScaleRcd.h"
#include "CondFormats/L1TObjects/interface/L1CaloHcalScale.h"
#include "CondFormats/DataRecord/interface/L1CaloHcalScaleRcd.h"

#include "CalibCalorimetry/EcalTPGTools/interface/EcalTPGScale.h"
#include "CalibFormats/CaloTPG/interface/CaloTPGTranscoder.h"
#include "CalibFormats/CaloTPG/interface/CaloTPGRecord.h"

#include "CondFormats/L1TObjects/interface/L1CaloEtScale.h"
#include "CondFormats/DataRecord/interface/L1EmEtScaleRcd.h"

#include "CondFormats/L1TObjects/interface/L1RCTParameters.h"
#include "CondFormats/DataRecord/interface/L1RCTParametersRcd.h"
#include "CondFormats/L1TObjects/interface/L1RCTChannelMask.h"
#include "CondFormats/DataRecord/interface/L1RCTChannelMaskRcd.h"

#include "L1Trigger/RegionalCaloTrigger/interface/L1RCT.h"
#include "L1Trigger/RegionalCaloTrigger/interface/L1RCTLookupTables.h" 

#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "CalibFormats/HcalObjects/interface/HcalDbRecord.h"

class L1RCTLookupTables;
class L1RCT;

class L1RCTPatternProducer : public edm::EDProducer {
public:
  explicit L1RCTPatternProducer(const edm::ParameterSet&);
  ~L1RCTPatternProducer();

private:
  virtual void beginJob(const edm::EventSetup&) {}
  virtual void produce(edm::Event&, const edm::EventSetup&);
  virtual void endJob() {}

  void firstEvent(unsigned short iCrate, unsigned short iCard,unsigned short iTower);
  void walkHCAL(int nEvents, unsigned short iCard,unsigned short iTower);
  void walkZeroHCAL(int nEvents, unsigned short iCard,unsigned short iTower);
  void jetSumPins(int nEvents, unsigned short iCard, unsigned short iTower, int num);
  // void writeEcalFiles(int TCC, int iEta, int iPhi);

  std::string fileName;
  //std::string ecalFileName;
  L1RCTLookupTables* rctLookupTables;
  std::ofstream ofs;
  std::ofstream ecalOfs;
  int fgEcalE;
  //int testType;
  std::string testName;
  int randomPercent;
  int randomSeed;
  bool regionSums;
  unsigned short ecal, fgbit, hcal, mubit, Etot, hf;

};
