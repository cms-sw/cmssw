#ifndef DataFormats_BTauReco_TemplatedSoftLeptonTagInfo_h
#define DataFormats_BTauReco_TemplatedSoftLeptonTagInfo_h

#include <vector>
#include <limits>

#include "DataFormats/Common/interface/CMS_CLASS_VERSION.h"
#include "DataFormats/BTauReco/interface/RefMacros.h"
#include "DataFormats/BTauReco/interface/JetTagInfo.h"
#include "DataFormats/BTauReco/interface/IPTagInfo.h"
#include "DataFormats/BTauReco/interface/TaggingVariable.h"

namespace reco {

class SoftLeptonProperties {
public:
    SoftLeptonProperties() :
        sip2dsig(    std::numeric_limits<float>::quiet_NaN() ),
        sip3dsig(    std::numeric_limits<float>::quiet_NaN() ),
        sip2d(    std::numeric_limits<float>::quiet_NaN() ),
        sip3d(    std::numeric_limits<float>::quiet_NaN() ),
        ptRel(    std::numeric_limits<float>::quiet_NaN() ),
        p0Par(    std::numeric_limits<float>::quiet_NaN() ),
        etaRel(   std::numeric_limits<float>::quiet_NaN() ),
        deltaR(   std::numeric_limits<float>::quiet_NaN() ),
        ratio(    std::numeric_limits<float>::quiet_NaN() ),
        ratioRel( std::numeric_limits<float>::quiet_NaN() ),
        elec_mva( std::numeric_limits<float>::quiet_NaN() )
    { }

    float sip2dsig;                            // 2D signed impact parameter significance
    float sip3dsig;                            // 3D signed impact parameter significance
    float sip2d;                            // 2D signed impact parameter
    float sip3d;                            // 3D signed impact parameter
    float ptRel;                            // transverse momentum wrt. the jet axis
    float p0Par;                            // momentum along the jet direction, in the jet rest frame

    float etaRel;                           // (pseudo)rapidity along jet axis
    float deltaR;                           // (pseudo)angular distance to jet axis
    float ratio;                            // momentum over jet energy
    float ratioRel;                         // momentum parallel to jet axis over jet energy

    float elec_mva;

    struct Quality {
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
    { return quality() != Quality::undef; }
    inline float hasQuality(Quality::Generic qual) const
    { return quality((unsigned int)qual, false) != Quality::undef; }
    inline float hasQuality(Quality::Muon qual) const
    { return quality((unsigned int)qual, false) != Quality::undef; }
    inline float hasQuality(Quality::Electron qual) const
    { return quality((unsigned int)qual, false) != Quality::undef; }

    // retrieve lepton quality

    inline float quality(Quality::Generic qual, bool throwIfUndefined = true) const
    { return quality((unsigned int)qual, throwIfUndefined); }
    inline float quality(Quality::Muon qual, bool throwIfUndefined = true) const
    { return quality((unsigned int)qual, throwIfUndefined); }
    inline float quality(Quality::Electron qual, bool throwIfUndefined = true) const
    { return quality((unsigned int)qual, throwIfUndefined); }

    // default value
    inline float quality() const { return quality(0, false); }

    // set lepton quality

    inline void setQuality(Quality::Generic qual, float value)
    { setQuality((unsigned int)qual, value); }
    inline void setQuality(Quality::Muon qual, float value)
    { setQuality((unsigned int)qual, value); }
    inline void setQuality(Quality::Electron qual, float value)
    { setQuality((unsigned int)qual, value); }

private:
    float quality(unsigned int index, bool throwIfUndefined) const;
    void setQuality(unsigned int index, float value);

    std::vector<float> qualities_;          // lepton qualities
};

template<class REF>
class TemplatedSoftLeptonTagInfo : public JetTagInfo {
public:
    typedef std::vector< std::pair< REF, SoftLeptonProperties > > LeptonMap;
    
    TemplatedSoftLeptonTagInfo(void) : m_leptons() {}

    virtual ~TemplatedSoftLeptonTagInfo(void) {}
  
    virtual TemplatedSoftLeptonTagInfo* clone(void) const { return new TemplatedSoftLeptonTagInfo(*this); }

    unsigned int leptons(void) const { 
        return m_leptons.size(); 
    } 

    const REF & lepton(size_t i) const {
        return m_leptons[i].first;
    }
    
    const SoftLeptonProperties & properties(size_t i) const {
        return m_leptons[i].second;
    }

    void insert(const REF & lepton, const SoftLeptonProperties & properties) {
        m_leptons.push_back( std::pair< REF, SoftLeptonProperties > (lepton, properties) );
    }

    /// returns a description of the extended informations in a TaggingVariableList
    virtual TaggingVariableList taggingVariables(void) const;

    // Used by ROOT storage
    CMS_CLASS_VERSION(2)

private:
    LeptonMap m_leptons;

};

template<class REF>
TaggingVariableList TemplatedSoftLeptonTagInfo<REF>::taggingVariables(void) const {
  TaggingVariableList list;

  const Jet & jet = *( this->jet() );
  list.insert( TaggingVariable(btau::jetEnergy, jet.energy()), true );
  list.insert( TaggingVariable(btau::jetPt,     jet.et()),     true );
  list.insert( TaggingVariable(btau::jetEta,    jet.eta()),    true );
  list.insert( TaggingVariable(btau::jetPhi,    jet.phi()),    true );

  for (unsigned int i = 0; i < m_leptons.size(); ++i) {
    const REF & trackRef = m_leptons[i].first;
    list.insert( TaggingVariable(btau::trackMomentum,  reco::btag::toTrack(trackRef)->p()),     true );
    list.insert( TaggingVariable(btau::trackEta,       reco::btag::toTrack(trackRef)->eta()),   true );
    list.insert( TaggingVariable(btau::trackPhi,       reco::btag::toTrack(trackRef)->phi()),   true );
    list.insert( TaggingVariable(btau::trackChi2,      reco::btag::toTrack(trackRef)->normalizedChi2()), true );
    const SoftLeptonProperties & data = m_leptons[i].second;
    list.insert( TaggingVariable(btau::leptonQuality,  data.quality(SoftLeptonProperties::Quality::leptonId, false)), true );
    list.insert( TaggingVariable(btau::leptonQuality2, data.quality(SoftLeptonProperties::Quality::btagLeptonCands, false)), true );
    list.insert( TaggingVariable(btau::trackSip2dVal,  data.sip2d),    true );
    list.insert( TaggingVariable(btau::trackSip3dVal,  data.sip3d),    true );
    list.insert( TaggingVariable(btau::trackSip2dSig,  data.sip2dsig),    true );
    list.insert( TaggingVariable(btau::trackSip3dSig,  data.sip3dsig),    true );
    list.insert( TaggingVariable(btau::trackPtRel,     data.ptRel),    true );
    list.insert( TaggingVariable(btau::trackP0Par,     data.p0Par),    true );
    list.insert( TaggingVariable(btau::trackEtaRel,    data.etaRel),   true );
    list.insert( TaggingVariable(btau::trackDeltaR,    data.deltaR),   true );
    list.insert( TaggingVariable(btau::trackPParRatio, data.ratioRel), true );
    list.insert( TaggingVariable(btau::electronMVA,    data.elec_mva), true );
  }

  list.finalize();
  return list;
}

}

#endif // DataFormats_BTauReco_TemplatedSoftLeptonTagInfo_h
