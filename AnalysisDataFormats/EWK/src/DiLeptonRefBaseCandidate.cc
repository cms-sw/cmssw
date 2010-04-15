#include "AnalysisDataFormats/EWK/interface/DiLeptonRefBaseCandidate.h"

#include "FWCore/Utilities/interface/EDMException.h"

namespace ewk
{

  DiLeptonRefBaseCandidate::DiLeptonRefBaseCandidate(const reco::CandidateBaseRef& d1,
				       const reco::CandidateBaseRef& d2)
  {
    addDaughter(d1);
    addDaughter(d2);
    determineType();
  }
  
  DiLeptonRefBaseCandidate::DiLeptonRefBaseCandidate(const edm::Ref<reco::MuonCollection>& d1,
				       const edm::Ref<reco::MuonCollection>& d2)
  {
    addDaughter(reco::CandidateBaseRef(d1));
    addDaughter(reco::CandidateBaseRef(d2));
    determineType();
  }

  DiLeptonRefBaseCandidate::DiLeptonRefBaseCandidate(const edm::Ref<reco::GsfElectronCollection>& d1,
				       const edm::Ref<reco::GsfElectronCollection>& d2)
  {
    addDaughter(reco::CandidateBaseRef(d1));
    addDaughter(reco::CandidateBaseRef(d2));
    determineType();
  }
  
  const reco::Muon DiLeptonRefBaseCandidate::muDaughter1() const
  {
    edm::Ref<reco::MuonCollection> d = daughterRef(0).castTo<edm::Ref<reco::MuonCollection> >();
    
    if(d.isNonnull()) return *d;
    return reco::Muon();
  }
  
  const reco::Muon DiLeptonRefBaseCandidate::muDaughter2() const
  {
    edm::Ref<reco::MuonCollection> d = daughterRef(1).castTo<edm::Ref<reco::MuonCollection> >();
    
    if(d.isNonnull()) return *d;
    return reco::Muon();
  }
  
  const reco::GsfElectron DiLeptonRefBaseCandidate::eDaughter1() const
  {
    edm::Ref<reco::GsfElectronCollection> d = daughterRef(0).castTo<edm::Ref<reco::GsfElectronCollection> >();
    
    if(d.isNonnull()) return *d;
    return reco::GsfElectron();
  }
  
  const reco::GsfElectron DiLeptonRefBaseCandidate::eDaughter2() const
  {
    edm::Ref<reco::GsfElectronCollection> d = daughterRef(1).castTo<edm::Ref<reco::GsfElectronCollection> >();
    
    if(d.isNonnull()) return *d;
    return reco::GsfElectron();
  }
  
  void DiLeptonRefBaseCandidate::determineType()
  {
    ZTYPE d1, d2;
    
    if(daughter(0)->isMuon()) d1 = ZMUMU;
    if(daughter(1)->isMuon()) d2 = ZMUMU;
    
    if(daughter(0)->isElectron()) d1 = ZEE;
    if(daughter(1)->isElectron()) d2 = ZEE;
    
    if(d1 == d2) ztype_ = d1;
    else
      edm::Exception::throwThis(edm::errors::LogicError,"Cannot have Z with different lepton types!");
  }
}

