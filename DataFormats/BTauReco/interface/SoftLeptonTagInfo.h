#ifndef BTauReco_SoftLeptonTagInfo_h
#define BTauReco_SoftLeptonTagInfo_h

#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/BTauReco/interface/JetTagInfo.h"
#include "DataFormats/BTauReco/interface/TaggingVariable.h"

namespace reco {
 
class SoftLeptonProperties {
public:
    enum {
        AXIS_CALORIMETRIC             = 0,  // use the calorimietric jet axis
        AXIS_CHARGED_AVERAGE          = 1,  // refine jet axis using charged tracks: use a pT-weighted average of (eta, phi)
        AXIS_CHARGED_AVERAGE_NOLEPTON = 2,  // as above, without the tagging lepton track
        AXIS_CHARGED_SUM              = 3,  // refine jet axis using charged tracks: use the sum of tracks momentum
        AXIS_CHARGED_SUM_NOLEPTON     = 4   // as above, without the tagging lepton track
    };

    unsigned int axisRefinement;            // if and how the jet axis is refined
    float quality;                          // lepton quality
    float sip2d;                            // 2D signed impact parameter
    float sip3d;                            // 3D signed impact parameter
    float ptRel;                            // transverse momentum wrt. jet axis
    float etaRel;                           // (pseudo)rapidity along jet axis
    float deltaR;                           // pseudoangular distance to jet axis
    float ratio;                            // momentum over jet energy
    float ratioRel;                         // momentum parallet to jet axis over jet energy
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

}

#endif // BTauReco_SoftLeptonTagInfo_h
