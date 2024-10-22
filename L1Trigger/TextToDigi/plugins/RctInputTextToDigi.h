#ifndef RCTINPUTTEXTTODIGI_H
#define RCTINPUTTEXTTODIGI_H

// -*- C++ -*-
//
// Class:      RctInputTextToDigi
//
/**\class RctInputTextToDigi L1Trigger/TextToDigi/plugins/RctInputTextToDigi.cc
 L1Trigger/TextToDigi/plugins/RctInputTextToDigi.h

 Description: Creates an EcalTrigPrimDigiCollection and an
 HcalTrigPrimDigiCollection from a text file (formatted as
 read out from saveRCTInput module), for input to the
 L1Trigger/RegionalCaloTrigger module.

*/

//
// Original Author:  jleonard
//         Created:  Fri Sep 21 16:16:27 CEST 2007
//
//

// system include files
#include <fstream>
#include <iostream>
#include <memory>
#include <string>

// user include files
#include "FWCore/Framework/interface/one/EDProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CondFormats/DataRecord/interface/L1RCTParametersRcd.h"
#include "CondFormats/L1TObjects/interface/L1RCTParameters.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "L1Trigger/RegionalCaloTrigger/interface/L1RCTLookupTables.h"

//
// class declaration
//

class RctInputTextToDigi : public edm::one::EDProducer<> {
public:
  explicit RctInputTextToDigi(const edm::ParameterSet &);
  ~RctInputTextToDigi() override;

private:
  void beginJob() override;
  void produce(edm::Event &, const edm::EventSetup &) override;
  void endJob() override;

  // ----------member data ---------------------------

  edm::FileInPath inputFile_;
  std::ifstream inputStream_;
  L1RCTLookupTables *lookupTables_;
  edm::ESGetToken<L1RCTParameters, L1RCTParametersRcd> paramsToken_;
  int nEvent_;
  bool oldVersion_;
};

#endif
