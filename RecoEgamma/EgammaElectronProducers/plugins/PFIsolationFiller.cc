#include "RecoEgamma/EgammaElectronProducers/plugins/PFIsolationFiller.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "DataFormats/Common/interface/ValueMap.h"

#include <iostream>
#include <string>

using namespace reco;

PFIsolationFiller::PFIsolationFiller( const edm::ParameterSet & cfg )
 {   
   previousGsfElectrons_ = consumes<reco::GsfElectronCollection>(cfg.getParameter<edm::InputTag>("previousGedGsfElectronsTag"));
   outputCollectionLabel_ = cfg.getParameter<std::string>("OutputCollectionLabel");
   pfIsoVals = 
     cfg.getParameter<edm::ParameterSet> ("pfIsolationValues");

   std::vector<std::string> isoNames = pfIsoVals.getParameterNamesForType<edm::InputTag>();
   for(const std::string& name : isoNames) {
     edm::InputTag tag = 
       pfIsoVals.getParameter<edm::InputTag>(name);
     consumes<edm::ValueMap<double> >(tag);   
     inputTagElectronIsoDeposits_.push_back(tag);
   }

   produces<GsfElectronCollection> >(outputCollectionLabel_);
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
   unsigned nDeps =  inputTagElectronIsoDeposits_.size();

   typedef std::vector< edm::Handle< edm::ValueMap<double> > > IsolationValueMaps;
   for(unsigned i=0; i < nDeps ; ++i) {
     event.getByToken(inputTagElectronISoDeposits_[i],electronIsoDep[i]);
   }
   
   // Now loop on the electrons
   unsigned nele=gedElectronH->size();
   for(unsigned iele=0; iele<nele;++iele) {
     reco::GsfElectronRef myElectronRef(gedElectronH,iele);
     
     
   }
   
   event.put(outputElectrons_p,outputCollectionLabel_);
 }


