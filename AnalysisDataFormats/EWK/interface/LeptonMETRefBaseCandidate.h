#ifndef AnalysisDataFormats_EWK_LeptonMETRefBaseCandidate_h
#define AnalysisDataFormats_EWK_LeptonMETRefBaseCandidate_h

#include <map>
#include <memory>

#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"

#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"

#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/METReco/interface/CaloMETCollection.h"

#include "DataFormats/METReco/interface/PFMET.h"
#include "DataFormats/METReco/interface/PFMETCollection.h"

#include "DataFormats/Candidate/interface/CompositeRefBaseCandidate.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/Common/interface/Ptr.h"
#include "DataFormats/Common/interface/Ref.h"

namespace ewk{
  
  class LeptonMETRefBaseCandidate : public reco::CompositeRefBaseCandidate
  {
  public:
    enum WTYPE {WENU, WMUNU};
    
    LeptonMETRefBaseCandidate() {};
    LeptonMETRefBaseCandidate(const reco::CandidateBaseRef&, const reco::CandidateBaseRef&, const reco::CandidateBaseRef&);
    LeptonMETRefBaseCandidate(const edm::Ref<reco::MuonCollection>&, 
		      const edm::Ref<reco::CaloMETCollection>&, 
		      const edm::Ref<reco::PFMETCollection>&);
    LeptonMETRefBaseCandidate(const edm::Ref<reco::GsfElectronCollection>&, 
		      const edm::Ref<reco::CaloMETCollection>&, 
		      const edm::Ref<reco::PFMETCollection>&);
    
    virtual ~LeptonMETRefBaseCandidate() {};
    
    const edm::Ref<reco::MuonCollection> muDaughterptr() const;    
    const reco::Muon muDaughter() const;
    
    const edm::Ref<reco::GsfElectronCollection> eDaughterptr() const;    
    const reco::GsfElectron eDaughter() const;

    const edm::Ref<reco::PFMETCollection> pfMETptr() const;
    const reco::PFMET pfMET() const;

    const edm::Ref<reco::CaloMETCollection> caloMETptr() const;
    const reco::CaloMET caloMET() const;
        
    bool Wenu() const {return (wtype_ == WENU);}
    bool Wmunu() const {return (wtype_ == WMUNU);}

    
    
    void determineType();
    
  private:    
    WTYPE wtype_;

    
  };

  typedef std::vector<LeptonMETRefBaseCandidate> LeptonMETRefBaseCandidateCollection;

}
#endif
