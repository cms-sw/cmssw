#include "AnalysisDataFormats/EWK/interface/LeptonMETRefBaseCandidate.h"

#include "FWCore/Utilities/interface/EDMException.h"

namespace ewk
{

  LeptonMETRefBaseCandidate::LeptonMETRefBaseCandidate(const reco::CandidateBaseRef& lepton,
				       const reco::CandidateBaseRef& caloMET,
				       const reco::CandidateBaseRef& pfMET)
  {
    addDaughter(lepton);
    addDaughter(caloMET);
    addDaughter(pfMET);
    determineType();
  }
  
  LeptonMETRefBaseCandidate::LeptonMETRefBaseCandidate(const edm::Ref<reco::MuonCollection>& lepton,
				       const edm::Ref<reco::CaloMETCollection>& caloMET,
				       const edm::Ref<reco::PFMETCollection>& pfMET)
    
  {
    addDaughter(reco::CandidateBaseRef(lepton));
    addDaughter(reco::CandidateBaseRef(caloMET));
    addDaughter(reco::CandidateBaseRef(pfMET));
    determineType();
  }

  LeptonMETRefBaseCandidate::LeptonMETRefBaseCandidate(const edm::Ref<reco::GsfElectronCollection>& lepton,
				       const edm::Ref<reco::CaloMETCollection>& caloMET,
				       const edm::Ref<reco::PFMETCollection>& pfMET)
  {
    addDaughter(reco::CandidateBaseRef(lepton));
    addDaughter(reco::CandidateBaseRef(caloMET));
    addDaughter(reco::CandidateBaseRef(pfMET));
    determineType();
  }
  
  const edm::Ref<reco::MuonCollection> LeptonMETRefBaseCandidate::muDaughterptr() const
  {
    edm::Ref<reco::MuonCollection> d;
    for(unsigned i = 0; i < numberOfDaughters(); ++i)
      {
	try
	  {
	    d = daughterRef(i).castTo<edm::Ref<reco::MuonCollection> >();
	  }
	catch(edm::Exception& e)
	  {
	    continue;
	  }
      }
    return d;
  }

  const reco::Muon LeptonMETRefBaseCandidate::muDaughter() const
  {
    edm::Ref<reco::MuonCollection> d = muDaughterptr();
    if(d.isNonnull()) return *d;
    return reco::Muon();
  }
    
  const edm::Ref<reco::GsfElectronCollection> LeptonMETRefBaseCandidate::eDaughterptr() const
  {
    edm::Ref<reco::GsfElectronCollection> d;
    for(unsigned i = 0; i < numberOfDaughters(); ++i)
      {
	try
	  {
	    d = daughterRef(i).castTo<edm::Ref<reco::GsfElectronCollection> >();
	  }
	catch(edm::Exception& e)
	  {
	    continue;
	  }
      }
    return d;
  }
  
  const reco::GsfElectron LeptonMETRefBaseCandidate::eDaughter() const
  {
    edm::Ref<reco::GsfElectronCollection> d = eDaughterptr();    
    if(d.isNonnull()) return *d;
    return reco::GsfElectron();
  }

  const edm::Ref<reco::CaloMETCollection> LeptonMETRefBaseCandidate::caloMETptr() const
  {
    edm::Ref<reco::CaloMETCollection> d;
    for(unsigned i = 0; i < numberOfDaughters(); ++i)
      {
	try
	  {
	    d = daughterRef(i).castTo<edm::Ref<reco::CaloMETCollection> >();
	  }
	catch(edm::Exception& e)
	  {
	    continue;
	  }
      }
    return d;
  }
  
  const reco::CaloMET LeptonMETRefBaseCandidate::caloMET() const
  {
    edm::Ref<reco::CaloMETCollection> d = caloMETptr();    
    if(d.isNonnull()) return *d;
    return reco::CaloMET();
  }

  const edm::Ref<reco::PFMETCollection> LeptonMETRefBaseCandidate::pfMETptr() const
  {
    edm::Ref<reco::PFMETCollection> d;
    for(unsigned i = 0; i < numberOfDaughters(); ++i)
      {
	try
	  {
	    d = daughterRef(i).castTo<edm::Ref<reco::PFMETCollection> >();
	  }
	catch(edm::Exception& e)
	  {
	    continue;
	  }
      }
    return d;
  }
  
  const reco::PFMET LeptonMETRefBaseCandidate::pfMET() const
  {
    edm::Ref<reco::PFMETCollection> d = pfMETptr();    
    if(d.isNonnull()) return *d;
    return reco::PFMET();
  }
  
  void LeptonMETRefBaseCandidate::determineType()
  {    
    edm::Ref<reco::MuonCollection> m = muDaughterptr();
    edm::Ref<reco::GsfElectronCollection> e = eDaughterptr(); 

    if(m.isNonnull() && e.isNonnull())
      edm::Exception::throwThis(edm::errors::LogicError,"Cannot have W with two leptonic legs!");

    if(m.isNull() && e.isNull())
      edm::Exception::throwThis(edm::errors::LogicError,"Cannot have W with no leptonic legs!");

    if(m.isNonnull()) wtype_ = WMUNU;
    if(e.isNonnull()) wtype_ = WENU;
  }
}

