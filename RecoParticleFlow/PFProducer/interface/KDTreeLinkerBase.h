#ifndef KDTreeLinkerBase_h
#define KDTreeLinkerBase_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitFraction.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElement.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"

#include <vector>
#include <map>
#include <set>

using BlockEltSet = std::set<reco::PFBlockElement *>;
using RecHitSet = std::set<const reco::PFRecHit *>;

using RecHit2BlockEltMap = std::map<const reco::PFRecHit *, BlockEltSet>;
using BlockElt2BlockEltMap = std::map<reco::PFBlockElement *, BlockEltSet>;

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

  // With this method, we create the list of elements that we want to link.
  virtual void insertTargetElt(reco::PFBlockElement *target) = 0;

  // Here, we create the list of cluster that we want to link. From cluster
  // and fraction, we will create a second list of rechits that will be used to
  // build the KDTree.
  virtual void insertFieldClusterElt(reco::PFBlockElement *cluster) = 0;

  // The KDTree building from rechits list.
  virtual void buildTree() = 0;

  // Here we will iterate over all target elements. For each one, we will search the closest
  // rechits in the KDTree, from rechits we will find the associated clusters and after that
  // we will check the links between the target and all closest clusters.
  virtual void searchLinks() = 0;

  // Here, we will store all target/cluster founded links in the PFBlockElement class
  // of each target in the PFmultilinks field.
  virtual void updatePFBlockEltWithLinks() = 0;

  // Here we free all allocated structures.
  virtual void clear() = 0;

  // This method calls is the good order buildTree(), searchLinks(),
  // updatePFBlockEltWithLinks() and clear()
  inline void process() {
    buildTree();
    searchLinks();
    updatePFBlockEltWithLinks();
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

  // rechit with fraction this value will be ignored in KDTreeLinker
  const float cutOffFrac = 1E-4;

  // Debug boolean. Not used until now.
  bool debug_ = false;
};

#include "FWCore/PluginManager/interface/PluginFactory.h"
typedef edmplugin::PluginFactory<KDTreeLinkerBase *(const edm::ParameterSet &)> KDTreeLinkerFactory;

#endif /* !KDTreeLinkerBase_h */
