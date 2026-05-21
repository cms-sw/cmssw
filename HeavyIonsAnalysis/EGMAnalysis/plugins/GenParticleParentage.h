#ifndef HEAVYIONSANALYSIS_EGMANALYSIS_GENPARTICLEPARENTAGE_H
#define HEAVYIONSANALYSIS_EGMANALYSIS_GENPARTICLEPARENTAGE_H

#include <DataFormats/Common/interface/Ref.h>
#include <DataFormats/HepMCCandidate/interface/GenParticle.h>
#include <DataFormats/PatCandidates/interface/PackedGenParticle.h>

#include <vector>

class GenParticleParentage {
public:
  GenParticleParentage(const reco::GenParticleRef&);
  GenParticleParentage(const pat::PackedGenParticleRef& p) { GenParticleParentage(p->lastPrunedRef()); };

  reco::GenParticleRef match() const { return _match; }
  reco::GenParticleRef parent() const { return _realParent; }

  bool hasQCDParent() const { return _qcdParent.isNonnull(); }
  reco::GenParticleRef getQuarkParent() const { return _qcdParent; }

  bool hasLeptonParent() const { return _leptonParent.isNonnull(); }
  reco::GenParticleRef getLeptonParent() const { return _leptonParent; }

  bool hasBosonParent() const { return _ewkBosonParent.isNonnull(); }
  reco::GenParticleRef getBosonParent() const { return _ewkBosonParent; }

  bool hasNonPromptParent() const { return _nonPromptParent.isNonnull(); }
  reco::GenParticleRef getNonPromptParent() const { return _nonPromptParent; }

  bool hasExoticParent() const { return _exoticParent.isNonnull(); }
  reco::GenParticleRef getExoticParent() const { return _exoticParent; }

  bool hasRealParent() const { return _realParent.isNonnull() && _realParent.isAvailable(); }

  static reco::GenParticleRef findGenMother(const reco::GenParticleRef&, const int& pId=0);
  static reco::GenParticleRef findGenMother(const pat::PackedGenParticleRef& p) { return findGenMother(p->lastPrunedRef()); };

private:
  void getParentageRecursive(const reco::GenParticleRef&, int);
  void resolveParentage();
  bool hasAsParent(const reco::GenParticleRef& daughter, const reco::GenParticleRef& parent_check) const;

  reco::GenParticleRef _match;

  reco::GenParticleRef _realParent;
  reco::GenParticleRef _leptonParent;
  reco::GenParticleRef _qcdParent;
  reco::GenParticleRef _ewkBosonParent;
  reco::GenParticleRef _nonPromptParent;
  reco::GenParticleRef _exoticParent;

  std::vector<reco::GenParticleRef> _leptonParents;
  std::vector<reco::GenParticleRef> _qcdParents;
  std::vector<reco::GenParticleRef> _ewkBosonParents;
  std::vector<reco::GenParticleRef> _nonPromptParents;
  std::vector<reco::GenParticleRef> _exoticParents;
};

#endif /* HEAVYIONSANALYSIS_EGMANALYSIS_GENPARTICLEPARENTAGE_H */
