// -*- C++ -*-
//
// Class:      HLTHcalMETNoiseFilter
// 
/**\class HLTHcalMETNoiseFilter

 Description: HLT filter module for rejecting MET events due to noise in the HCAL

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Leonard Apanasevich
//         Created:  Wed Mar 25 16:01:27 CDT 2009
// $Id$
//
//

#include "HLTrigger/JetMET/interface/HLTHcalMETNoiseFilter.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/METReco/interface/HcalNoiseRBXArray.h"
#include "DataFormats/METReco/interface/HcalNoiseSummary.h"


HLTHcalMETNoiseFilter::HLTHcalMETNoiseFilter(const edm::ParameterSet& iConfig)
  : HcalNoiseRBXCollectionTag (iConfig.getParameter <edm::InputTag> ("HcalNoiseRBXCollection")),
    HcalNoiseSummaryTag (iConfig.getParameter <edm::InputTag> ("HcalNoiseSummary")),
    severity (iConfig.getParameter <int> ("severity")),
    EMFractionMin(iConfig.getParameter <double> ("EMFractionMin")),
    nRBXhitsMax(iConfig.getParameter <int> ("nRBXhitsMax")),
    RBXhitThresh(iConfig.getParameter <double> ("RBXhitThresh"))
{
}


HLTHcalMETNoiseFilter::~HLTHcalMETNoiseFilter(){}


//
// member functions
//

bool HLTHcalMETNoiseFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace reco;

  bool accept=true;  // assume good event
  if (severity == 0 ) return accept; // do not filter anything

  edm::Handle<HcalNoiseRBXCollection> RBXCollection;
  iEvent.getByLabel(HcalNoiseRBXCollectionTag,RBXCollection);
  if (!RBXCollection.isValid()) {
    LogDebug("") << "HLTHcalMETNoiseFilter: Could not find HcalNoiseRBX Collection" << std::endl;
    return accept;
  }

  edm::Handle<HcalNoiseSummary> NoiseSummary;
  iEvent.getByLabel(HcalNoiseSummaryTag,NoiseSummary);
  if (!NoiseSummary.isValid()) {
    LogDebug("") << "HLTHcalMETNoiseFilter: Could not find Hcal NoiseSummary product" << std::endl;
    return accept;
  }

  //std::cout << "NoiseSummary: " << NoiseSummary->eventEMFraction() << std::endl;
  if (NoiseSummary->eventEMFraction() < EMFractionMin) return false;
 

  //std::cout << "Size of NoiseRBX collection:  " << RBXCollection->size() << std::endl;
  for (HcalNoiseRBXCollection::const_iterator rbx = RBXCollection->begin();
         rbx!=(RBXCollection->end()); rbx++) {
    int nRBXhits=rbx->numRecHits(RBXhitThresh);
    //std::cout << "\tNumber of RBX hits:     " << nRBXhits << std::endl;
    if (nRBXhits > nRBXhitsMax ) return false;
  }
  return accept;
}
