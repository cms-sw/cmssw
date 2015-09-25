#ifndef __FSA_DATAALGOS_PHOTONPARENTAGE_H__
#define __FSA_DATAALGOS_PHOTONPARENTAGE_H__

#include <DataFormats/Common/interface/Ref.h>
#include <DataFormats/HepMCCandidate/interface/GenParticle.h>

// This class gets a GenLevel particle and spits out the information
// it recurses down the information until 
// the photon's provenance is determined.

namespace genpartparentage {
  class GenParticleParentage {
  public:
    GenParticleParentage(reco::GenParticleRef& );
    
    reco::GenParticleRef match() const {return _match;}
    
    reco::GenParticleRef parent() const {return _realParent;}

    bool hasQCDParent() const { return _qcdParent.isNonnull(); }
    reco::GenParticleRef getQuarkParent() const { return _qcdParent; }

    bool hasLeptonParent() const { return _leptonParent.isNonnull(); }
    reco::GenParticleRef getLeptonParent() const { return _leptonParent; }

    bool hasBosonParent()  const { return _ewkBosonParent.isNonnull(); }
    reco::GenParticleRef getBosonParent() const { return _ewkBosonParent; }

    bool hasNonPromptParent()  const { return _nonPromptParent.isNonnull(); }
    reco::GenParticleRef getNonPromptParent() const { return _nonPromptParent;}

    bool hasExoticParent()  const { return _exoticParent.isNonnull(); }
    reco::GenParticleRef getExoticParent() const {return _exoticParent; }

    bool hasRealParent() const { return _realParent.isNonnull() && _realParent.isAvailable(); }

  private:    
    void getParentageRecursive(const reco::GenParticleRef&, int);    
    void resolveParentage();
    bool hasAsParent(const reco::GenParticleRef& daughter,
		     const reco::GenParticleRef& parent_check) const;

    reco::GenParticleRef _match;
    reco::GenParticleRef _realParent,_leptonParent,_qcdParent,
      _ewkBosonParent,_nonPromptParent,_exoticParent;
    std::vector<reco::GenParticleRef> _leptonParents,_qcdParents,
      _ewkBosonParents,_nonPromptParents,_exoticParents;
  };
}

#endif
