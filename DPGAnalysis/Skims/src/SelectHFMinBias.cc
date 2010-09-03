// -*- C++ -*-
//
// Package:   BeamSplash
// Class:     BeamSPlash
//
//
// Original Author:  Luca Malgeri

#include <memory>
#include <vector>
#include <map>
#include <set>

// user include files

#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DPGAnalysis/Skims/interface/SelectHFMinBias.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"

using namespace edm;
using namespace std;

SelectHFMinBias::SelectHFMinBias(const edm::ParameterSet& iConfig)
{
}

SelectHFMinBias::~SelectHFMinBias()
{
}

bool SelectHFMinBias::filter( edm::Event& iEvent, const edm::EventSetup& iSetup)
{

edm::Handle<CaloTowerCollection> towers;
iEvent.getByLabel("towerMaker",towers);


int negTowers = 0;
int posTowers = 0;
for(CaloTowerCollection::const_iterator cal = towers->begin(); cal != towers->end(); ++cal) {
   for(unsigned int i = 0; i < cal->constituentsSize(); i++) {
      const DetId id = cal->constituent(i);
      if(id.det() == DetId::Hcal) {
        HcalSubdetector subdet=(HcalSubdetector(id.subdetId()));
        if(subdet == HcalForward) {
          if(cal->energy()>3. && cal->eta()<-3.)
            negTowers++;
          if(cal->energy()>3. && cal->eta()>3.)
            posTowers++;
        }
     }
   }
}
if(negTowers>0 && posTowers>0)
  return true;

  return false;

}

//define this as a plug-in
DEFINE_FWK_MODULE(SelectHFMinBias);
