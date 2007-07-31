#include <memory>
#include <string>
#include <iostream>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "L1Trigger/RegionalCaloTrigger/interface/L1RCTSaveInput.h"

#include "DataFormats/L1CaloTrigger/interface/L1CaloCollections.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"

#include "CalibFormats/CaloTPG/interface/CaloTPGTranscoder.h"
#include "CalibFormats/CaloTPG/interface/CaloTPGRecord.h"

#include "CondFormats/L1TObjects/interface/L1CaloEtScale.h"
#include "CondFormats/DataRecord/interface/L1EmEtScaleRcd.h"

#include "CondFormats/L1TObjects/interface/L1RCTParameters.h"
#include "CondFormats/DataRecord/interface/L1RCTParametersRcd.h"

#include "L1Trigger/RegionalCaloTrigger/interface/L1RCT.h"
#include "L1Trigger/RegionalCaloTrigger/interface/L1RCTLookupTables.h" 

L1RCTSaveInput::L1RCTSaveInput(const edm::ParameterSet& iConfig) :
  fileName(iConfig.getUntrackedParameter<std::string>("rctTestInputFile")),
  rctLookupTables(new L1RCTLookupTables),
  rct(new L1RCT(rctLookupTables))
{
  ofs.open(fileName.c_str(), std::ios::app);
  if(!ofs)
    {
      std::cerr << "Could not create " << fileName << endl;
      exit(1);
    }
}

L1RCTSaveInput::~L1RCTSaveInput()
{
  if(rct != 0) delete rct;
  if(rctLookupTables != 0) delete rctLookupTables;
}

void
L1RCTSaveInput::analyze(const edm::Event& iEvent,
			const edm::EventSetup& iSetup)
{
  edm::ESHandle<L1RCTParameters> rctParameters;
  eventSetup.get<L1RCTParametersRcd>().get(rctParameters);
  const L1RCTParameters* r = rctParameters.product();
  edm::ESHandle<CaloTPGTranscoder> transcoder;
  eventSetup.get<CaloTPGRecord>().get(transcoder);
  const CaloTPGTranscoder* t = transcoder.product();
  edm::ESHandle<L1CaloEtScale> emScale;
  eventSetup.get<L1EmEtScaleRcd>().get(emScale);
  const L1CaloEtScale* s = emScale.product();
  rctLookupTables->setRCTParameters(r);
  rctLookupTables->setTranscoder(t);
  rctLookupTables->setL1CaloEtScale(s);
  edm::Handle<EcalTrigPrimDigiCollection> ecal;
  edm::Handle<HcalTrigPrimDigiCollection> hcal;
  event.getByLabel(ecalDigisLabel, ecal); 
  event.getByLabel(hcalDigisLabel, hcal); 
  EcalTrigPrimDigiCollection ecalColl;
  HcalTrigPrimDigiCollection hcalColl;
  if (ecal.isValid()) { ecalColl = *ecal; }
  if (hcal.isValid()) { hcalColl = *hcal; }
  rct->digiInput(ecalColl, hcalColl, &ofs);
}
