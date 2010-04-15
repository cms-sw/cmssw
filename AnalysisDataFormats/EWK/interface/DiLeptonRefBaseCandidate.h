#ifndef AnalysisDataFormats_EWK_DiLeptonRefBaseCandidate_h
#define AnalysisDataFormats_EWK_DiLeptonRefBaseCandidate_h

#include <map>
#include <memory>

#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"

#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"

#include "DataFormats/Candidate/interface/CompositeRefBaseCandidate.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/Common/interface/Ptr.h"
#include "DataFormats/Common/interface/Ref.h"

namespace ewk{
  
  class DiLeptonRefBaseCandidate : public reco::CompositeRefBaseCandidate
  {
  public:
    enum ZTYPE {ZEE, ZMUMU};
    
    DiLeptonRefBaseCandidate() {};
    DiLeptonRefBaseCandidate(const reco::CandidateBaseRef&, const reco::CandidateBaseRef&);
    DiLeptonRefBaseCandidate(const edm::Ref<reco::MuonCollection>&, const edm::Ref<reco::MuonCollection>&);
    DiLeptonRefBaseCandidate(const edm::Ref<reco::GsfElectronCollection>&, const edm::Ref<reco::GsfElectronCollection>&);
    
    virtual ~DiLeptonRefBaseCandidate() {};
    
    const edm::Ref<reco::MuonCollection> 
      muDaughter1ptr() const { return daughterRef(0).castTo<edm::Ref<reco::MuonCollection> >();}
    const edm::Ref<reco::MuonCollection> 
      muDaughter2ptr() const { return daughterRef(1).castTo<edm::Ref<reco::MuonCollection> >();}
    
    const reco::Muon muDaughter1() const;
    const reco::Muon muDaughter2() const;
    
    const edm::Ref<reco::GsfElectronCollection> 
      eDaughter1ptr() const { return daughterRef(0).castTo<edm::Ref<reco::GsfElectronCollection> >();}
    const edm::Ref<reco::GsfElectronCollection> 
      eDaughter2ptr() const { return daughterRef(1).castTo<edm::Ref<reco::GsfElectronCollection> >();}
    
    const reco::GsfElectron eDaughter1() const;
    const reco::GsfElectron eDaughter2() const;
    
    bool Zee() const {return (ztype_ == ZEE);}
    bool Zmumu() const {return (ztype_ == ZMUMU);}
    
    void determineType();
    
  private:
    
    ZTYPE ztype_;
  };

  typedef std::vector<DiLeptonRefBaseCandidate> DiLeptonRefBaseCandidateCollection;

}
#endif
