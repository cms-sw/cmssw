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
  
  MuPFIsoHelper(const std::map<std::string,edm::ParameterSet>&);

  void beginEvent(const edm::Event& iEvent);

  int  embedPFIsolation(reco::Muon&,reco::MuonRef& );
  reco::MuonPFIsolation makeIsoDeposit(reco::MuonRef&, 
				       const edm::Handle<CandDoubleMap>&,
				       const edm::Handle<CandDoubleMap>&,
				       const edm::Handle<CandDoubleMap>&,
				       const edm::Handle<CandDoubleMap>&,
				       const edm::Handle<CandDoubleMap>&,
				       const edm::Handle<CandDoubleMap>&,
				       const edm::Handle<CandDoubleMap>&);


  ~MuPFIsoHelper(); 


   private:

  std::map<std::string,edm::ParameterSet> labelMap_;

  std::vector<edm::Handle<CandDoubleMap> > chargedParticle_;
  std::vector<edm::Handle<CandDoubleMap> > chargedHadron_;
  std::vector<edm::Handle<CandDoubleMap> > neutralHadron_;
  std::vector<edm::Handle<CandDoubleMap> > neutralHadronHighThreshold_;
  std::vector<edm::Handle<CandDoubleMap> > photon_;
  std::vector<edm::Handle<CandDoubleMap> > photonHighThreshold_;
  std::vector<edm::Handle<CandDoubleMap> > pu_;

};
#endif
