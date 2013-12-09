#include "RecoEgamma/EgammaElectronProducers/plugins/PFIsolationFiller.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"

#include <iostream>
#include <string>

using namespace reco;

PFIsolationFiller::PFIsolationFiller( const edm::ParameterSet & cfg )
 {   
   previousGsfElectrons_ = consumes<reco::GsfElectronCollection>(cfg.getParameter<edm::InputTag>("previousGsfElectronsTag"));
   outputCollectionLabel_ = cfg.getParameter<std::string>("outputCollectionLabel");
   edm::ParameterSet pfIsoVals(cfg.getParameter<edm::ParameterSet> ("pfIsolationValues"));
   
   tokenElectronIsoVals_.push_back(consumes<edm::ValueMap<double> >(pfIsoVals.getParameter<edm::InputTag>("pfSumChargedHadronPt")));
   tokenElectronIsoVals_.push_back(consumes<edm::ValueMap<double> >(pfIsoVals.getParameter<edm::InputTag>("pfSumPhotonEt")));
   tokenElectronIsoVals_.push_back(consumes<edm::ValueMap<double> >(pfIsoVals.getParameter<edm::InputTag>("pfSumNeutralHadronEt")));
   tokenElectronIsoVals_.push_back(consumes<edm::ValueMap<double> >(pfIsoVals.getParameter<edm::InputTag>("pfSumPUPt")));
//   std::vector<std::string> isoNames = pfIsoVals.getParameterNamesForType<edm::InputTag>();
//   for(const std::string& name : isoNames) {
//     edm::InputTag tag = 
//       pfIsoVals.getParameter<edm::InputTag>(name);
//     tokenElectronIsoVals_.push_back(consumes<edm::ValueMap<double> >(tag));   
//   }

   nDeps_ =  tokenElectronIsoVals_.size();

   produces<reco::GsfElectronCollection> (outputCollectionLabel_);
}

PFIsolationFiller::~PFIsolationFiller()
 {}

// ------------ method called to produce the data  ------------
void PFIsolationFiller::produce( edm::Event & event, const edm::EventSetup & setup )
 {
   
   // Output collection
   std::auto_ptr<reco::GsfElectronCollection> outputElectrons_p(new reco::GsfElectronCollection);
   
   // read input collections
   // electrons
   edm::Handle<reco::GsfElectronCollection> gedElectronHandle;
   event.getByToken(previousGsfElectrons_,gedElectronHandle);

   // value maps

   std::vector< edm::Handle< edm::ValueMap<double> > > isolationValueMaps(nDeps_);
   
   for(unsigned i=0; i < nDeps_ ; ++i) {
     event.getByToken(tokenElectronIsoVals_[i],isolationValueMaps[i]);
   }
   
   // Now loop on the electrons
   unsigned nele=gedElectronHandle->size();
   for(unsigned iele=0; iele<nele;++iele) {
     reco::GsfElectronRef myElectronRef(gedElectronHandle,iele);
     
     reco::GsfElectron newElectron(*myElectronRef);
     reco::GsfElectron::PflowIsolationVariables isoVariables;
     isoVariables.sumChargedHadronPt = (*(isolationValueMaps)[0])[myElectronRef];
     isoVariables.sumPhotonEt = (*(isolationValueMaps)[1])[myElectronRef];
     isoVariables.sumNeutralHadronEt = (*(isolationValueMaps)[2])[myElectronRef];
     isoVariables.sumPUPt = (*(isolationValueMaps)[3])[myElectronRef];
     newElectron.setPfIsolationVariables(isoVariables);

     outputElectrons_p->push_back(newElectron);
   }
   
   event.put(outputElectrons_p,outputCollectionLabel_);
 }


