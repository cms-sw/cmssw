#ifndef DataFormats_BTauReco_SoftLeptonTagInfo_h
#define DataFormats_BTauReco_SoftLeptonTagInfo_h

#include "DataFormats/BTauReco/interface/RefMacros.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h" 
#include "DataFormats/BTauReco/interface/JetTagInfo.h"
#include "DataFormats/BTauReco/interface/TaggingVariable.h"

namespace reco {
 
class SoftLeptonProperties {
public:
    float quality;                          // lepton quality
   
    float sip2d;                            // 2D signed impact parameter
    float sip3d;                            // 3D signed impact parameter
    float ptRel;                            // transverse momentum wrt. the jet axis
    float p0Par;                            // momentum along the jet direction, in the jet rest frame

    float etaRel;                           // (pseudo)rapidity along jet axis
    float deltaR;                           // (pseudo)angular distance to jet axis
    float ratio;                            // momentum over jet energy
    float ratioRel;                         // momentum paraller to jet axis over jet energy
};

class SoftLeptonTagInfo : public JetTagInfo {
public:
    typedef std::vector< std::pair< edm::RefToBase<reco::Track>, SoftLeptonProperties > > LeptonMap;
    
    SoftLeptonTagInfo(void) : m_leptons() {}

    virtual ~SoftLeptonTagInfo(void) {}
  
    virtual SoftLeptonTagInfo* clone(void) const { return new SoftLeptonTagInfo(*this); }

    unsigned int leptons(void) const { 
        return m_leptons.size(); 
    } 

    const edm::RefToBase<reco::Track> & lepton(size_t i) const {
        return m_leptons[i].first;
    }
    
    const SoftLeptonProperties & properties(size_t i) const {
        return m_leptons[i].second;
    }

    void insert(const edm::RefToBase<reco::Track> & lepton, const SoftLeptonProperties & properties) {
        m_leptons.push_back( std::pair< edm::RefToBase<reco::Track>, SoftLeptonProperties > (lepton, properties) );
    }

    /// returns a description of the extended informations in a TaggingVariableList
    virtual TaggingVariableList taggingVariables(void) const;

private:
    LeptonMap m_leptons;

};

DECLARE_EDM_REFS( SoftLeptonTagInfo )

}

#endif // DataFormats_BTauReco_SoftLeptonTagInfo_h
