#ifndef RecoMuon_MuonIsolation_MuPFIsoHelper_H
#define RecoMuon_MuonIsolation_MuPFIsoHelper_H

//MuPFIsoHelper 
//Class to embed PF2PAT style Isodeposits  
//To reco::Muon 
//
//Author: Michalis Bachtis(U.Wisconsin)
//bachtis@cern.ch


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/Common/interface/ValueMap.h"



class MuPFIsoHelper {
   public:
  typedef edm::ValueMap<double> CandDoubleMap;
  
  MuPFIsoHelper(const edm::ParameterSet& iConfig);
  void beginEvent(const edm::Event& iEvent);

  int  embedPFIsolation(reco::Muon&,reco::MuonRef& );

  ~MuPFIsoHelper(); 


   private:
  edm::ParameterSet isoCfg03_;
  edm::ParameterSet isoCfg04_;


  edm::Handle<CandDoubleMap> chargedParticle03_;
  edm::Handle<CandDoubleMap> chargedHadron03_;
  edm::Handle<CandDoubleMap> neutralHadron03_;
  edm::Handle<CandDoubleMap> photon03_;
  edm::Handle<CandDoubleMap> pu03_;
  ///////////////////////////////////////////////
  edm::Handle<CandDoubleMap> chargedParticle04_;
  edm::Handle<CandDoubleMap> chargedHadron04_;
  edm::Handle<CandDoubleMap> neutralHadron04_;
  edm::Handle<CandDoubleMap> photon04_;
  edm::Handle<CandDoubleMap> pu04_;
  

};
#endif
