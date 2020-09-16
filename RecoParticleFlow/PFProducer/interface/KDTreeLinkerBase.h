#ifndef KDTreeLinkerBase_h
#define KDTreeLinkerBase_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitFraction.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElement.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"
#include "RecoParticleFlow/PFProducer/interface/PFTables.h"
#include "RecoParticleFlow/PFProducer/interface/PFMultiLinksIndex.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <vector>
#include <map>
#include <set>

using BlockEltSet = std::set<reco::PFBlockElement *>;
using RecHitSet = std::set<const reco::PFRecHit *>;

using RecHit2BlockEltMap = std::map<const reco::PFRecHit *, BlockEltSet>;
using BlockElt2BlockEltMap = std::map<reco::PFBlockElement *, BlockEltSet>;
using ElementList = std::vector<std::unique_ptr<reco::PFBlockElement>>;

class KDTreeLinkerBase {
public:
  KDTreeLinkerBase(const edm::ParameterSet &conf) {}
  virtual ~KDTreeLinkerBase() {}

  void setTargetType(const reco::PFBlockElement::Type &tgt) { _targetType = tgt; }

  void setFieldType(const reco::PFBlockElement::Type &fld) { _fieldType = fld; }

  const reco::PFBlockElement::Type &targetType() const { return _targetType; }

  const reco::PFBlockElement::Type &fieldType() const { return _fieldType; }

  // Get/Set of the maximal size of the cristal (ECAL, HCAL,...) in phi/eta and
  // X/Y. By default, thus value are set for the ECAL cristal.

  // Get/Set phi offset. See bellow in the description of phiOffset_ to understand
  // the application.

  // Debug flag.
  void setDebug(bool isDebug);

  // The KDTree building from rechits list.
  virtual void buildTree(const PFTables &pftables) = 0;

  // Here we will iterate over all target elements. For each one, we will search the closest
  // rechits in the KDTree, from rechits we will find the associated clusters and after that
  // we will check the links between the target and all closest clusters.
  virtual void searchLinks(const PFTables &pftables, reco::PFMultiLinksIndex &multilinks) = 0;

  // Here we free all allocated structures.
  virtual void clear() = 0;

  // This method calls is the good order buildTree(), searchLinks(),
  // updatePFBlockEltWithLinks() and clear()
  inline void process(const PFTables &pftables, const ElementList &elem_pointers, reco::PFMultiLinksIndex &multilinks) {
    buildTree(pftables);
    searchLinks(pftables, multilinks);
    clear();
  }

protected:
  // target and field
  reco::PFBlockElement::Type _targetType, _fieldType;
  // Cristal maximal size. By default, thus value are set for the ECAL cristal.
  float cristalPhiEtaMaxSize_ = 0.04;
  float cristalXYMaxSize_ = 3.;

  // Usually, phi is between -Pi and +Pi. But phi space is circular, that's why an element
  // with phi = 3.13 and another with phi = -3.14 are close. To solve this problem, during
  // the kdtree building step, we duplicate some elements close enough to +Pi (resp -Pi) by
  // substracting (adding) 2Pi. This field define the threshold of this operation.
  float phiOffset_ = 0.25;

  // Debug boolean. Not used until now.
  bool debug_ = false;

  // Sorted indexes
  template <typename T>
  static std::vector<size_t> sort_indexes(const std::vector<T> &v) {
    std::vector<size_t> idx(v.size());
    for (size_t i = 0; i != idx.size(); ++i)
      idx[i] = i;
    std::sort(idx.begin(), idx.end(), [&v](size_t i1, size_t i2) { return v[i1] < v[i2]; });
    return idx;
  }
};

#include "FWCore/PluginManager/interface/PluginFactory.h"
typedef edmplugin::PluginFactory<KDTreeLinkerBase *(const edm::ParameterSet &)> KDTreeLinkerFactory;

#endif /* !KDTreeLinkerBase_h */
