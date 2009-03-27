//
// Package:    MuonTimingProducer
// Class:      MuonTimingProducer
// 
/**\class MuonTimingProducer MuonTimingProducer.cc RecoMuon/MuonIdentification/src/MuonTimingProducer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Piotr Traczyk, CERN
//         Created:  Mon Mar 16 12:27:22 CET 2009
// $Id: MuonTimingProducer.cc,v 1.2 2009/03/26 23:56:44 ptraczyk Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/MuonReco/interface/Muon.h" 
#include "DataFormats/MuonReco/interface/MuonFwd.h" 
#include "DataFormats/MuonReco/interface/MuonTimeExtra.h"
#include "DataFormats/MuonReco/interface/MuonTimeExtraMap.h"

#include "RecoMuon/MuonIdentification/plugins/MuonTimingProducer.h"
#include "RecoMuon/MuonIdentification/interface/TimeMeasurementSequence.h"


//
// constructors and destructor
//
MuonTimingProducer::MuonTimingProducer(const edm::ParameterSet& iConfig)
{
   produces<reco::MuonTimeExtraMap>();

   m_muonCollection = iConfig.getParameter<edm::InputTag>("MuonCollection");

   // Load parameters for the TimingFiller
   edm::ParameterSet fillerParameters = iConfig.getParameter<edm::ParameterSet>("TimingFillerParameters");
   theTimingFiller_ = new MuonTimingFiller(fillerParameters);
}


MuonTimingProducer::~MuonTimingProducer()
{
   if (theTimingFiller_) delete theTimingFiller_;
}


//
// member functions
//

// ------------ method called once each job just before starting event loop  ------------
void 
MuonTimingProducer::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
MuonTimingProducer::endJob() {
}

// ------------ method called to produce the data  ------------
void
MuonTimingProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {

  std::auto_ptr<reco::MuonTimeExtraMap> muonTimeMap(new reco::MuonTimeExtraMap());
  reco::MuonTimeExtraMap::Filler filler(*muonTimeMap);
  
  edm::Handle<reco::MuonCollection> muons; 
  iEvent.getByLabel(m_muonCollection, muons);

  unsigned int nMuons = muons->size();
  if (!nMuons) return;
  
  vector<reco::MuonTimeExtra> dtTimeColl(nMuons);
  vector<reco::MuonTimeExtra> cscTimeColl(nMuons);
  vector<reco::MuonTimeExtra> combinedTimeColl(nMuons);

  for ( unsigned int i=0; i<nMuons; ++i ) {

    reco::MuonTimeExtra dtTime;
    reco::MuonTimeExtra cscTime;
    reco::MuonTimeExtra combinedTime;

    reco::MuonRef muonr(muons,i);
    
    theTimingFiller_->fillTiming(*muonr, dtTime, cscTime, combinedTime, iEvent, iSetup);
    
    dtTimeColl[i] = dtTime;
    cscTimeColl[i] = cscTime;
    combinedTimeColl[i] = combinedTime;
     
  }
  
  filler.insert(muons, combinedTimeColl.begin(), combinedTimeColl.end());
  
  filler.fill();
  
  iEvent.put(muonTimeMap);

}


//define this as a plug-in
//DEFINE_FWK_MODULE(MuonTimingProducer);
