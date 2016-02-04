#ifndef DataFormats_BTauReco_SoftLeptonTagInfo_h
#define DataFormats_BTauReco_SoftLeptonTagInfo_h

#include <vector>
#include <limits>

#include "DataFormats/BTauReco/interface/RefMacros.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h" 
#include "DataFormats/BTauReco/interface/JetTagInfo.h"
#include "DataFormats/BTauReco/interface/TaggingVariable.h"

namespace reco {
 
class SoftLeptonProperties {
public:
    SoftLeptonProperties() :
        sip2d(    std::numeric_limits<float>::quiet_NaN() ),
        sip3d(    std::numeric_limits<float>::quiet_NaN() ),
        ptRel(    std::numeric_limits<float>::quiet_NaN() ),
        p0Par(    std::numeric_limits<float>::quiet_NaN() ),
        etaRel(   std::numeric_limits<float>::quiet_NaN() ),
        deltaR(   std::numeric_limits<float>::quiet_NaN() ),
        ratio(    std::numeric_limits<float>::quiet_NaN() ),
        ratioRel( std::numeric_limits<float>::quiet_NaN() )
    { }

    float sip2d;                            // 2D signed impact parameter
    float sip3d;                            // 3D signed impact parameter
    float ptRel;                            // transverse momentum wrt. the jet axis
    float p0Par;                            // momentum along the jet direction, in the jet rest frame

    float etaRel;                           // (pseudo)rapidity along jet axis
    float deltaR;                           // (pseudo)angular distance to jet axis
    float ratio;                            // momentum over jet energy
    float ratioRel;                         // momentum parallel to jet axis over jet energy

    struct quality {
        static const float undef;

	// these first two entries work for both electrons and muons,
	// entries afterwards can be specific to either one
	// electrons & muons shared the same indicies to avoid waste of space
        enum Generic {
            leptonId = 0,
            btagLeptonCands
        };

        enum Electron {
            pfElectronId = 0,
            btagElectronCands,
	    egammaElectronId
        };

        enum Muon {
            muonId = 0,
	    btagMuonCands
        };

        template<typename T> static inline T byName(const char *name)
        { return static_cast<T>(internalByName(name)); }

      private:
        static unsigned int internalByName(const char *name);
    };

    // check to see if quality has been set

    inline float hasQuality() const
    { return quality() != quality::undef; }
    inline float hasQuality(quality::Generic qual) const
    { return quality((unsigned int)qual, false) != quality::undef; }
    inline float hasQuality(quality::Muon qual) const
    { return quality((unsigned int)qual, false) != quality::undef; }
    inline float hasQuality(quality::Electron qual) const
    { return quality((unsigned int)qual, false) != quality::undef; }

    // retrieve lepton quality

    inline float quality(quality::Generic qual, bool throwIfUndefined = true) const
    { return quality((unsigned int)qual, throwIfUndefined); }
    inline float quality(quality::Muon qual, bool throwIfUndefined = true) const
    { return quality((unsigned int)qual, throwIfUndefined); }
    inline float quality(quality::Electron qual, bool throwIfUndefined = true) const
    { return quality((unsigned int)qual, throwIfUndefined); }

    // default value
    inline float quality() const { return quality(0, false); }

    // set lepton quality

    inline void setQuality(quality::Generic qual, float value)
    { setQuality((unsigned int)qual, value); }
    inline void setQuality(quality::Muon qual, float value)
    { setQuality((unsigned int)qual, value); }
    inline void setQuality(quality::Electron qual, float value)
    { setQuality((unsigned int)qual, value); }

private:
    float quality(unsigned int index, bool throwIfUndefined) const;
    void setQuality(unsigned int index, float value);

    std::vector<float> qualities_;          // lepton qualities
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
