#ifndef __BlockElementLinkerBase_H__
#define __BlockElementLinkerBase_H__

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElement.h"
#include "RecoParticleFlow/PFProducer/interface/PFTables.h"
#include "RecoParticleFlow/PFProducer/interface/PFMultiLinksIndex.h"

#include <string>

using ElementListConst = std::vector<const reco::PFBlockElement*>;

class BlockElementLinkerBase {
public:
  BlockElementLinkerBase(const edm::ParameterSet& conf) : _linkerName(conf.getParameter<std::string>("linkerName")) {}
  BlockElementLinkerBase(const BlockElementLinkerBase&) = delete;
  virtual ~BlockElementLinkerBase() = default;
  BlockElementLinkerBase& operator=(const BlockElementLinkerBase&) = delete;

  virtual bool linkPrefilter(size_t ielem1,
                             size_t ielem2,
                             reco::PFBlockElement::Type type1,
                             reco::PFBlockElement::Type type2,
                             const PFTables& tables,
                             const reco::PFMultiLinksIndex& multilinks) const {
    return true;
  }

  virtual double testLink(size_t ielem1,
                          size_t ielem2,
                          reco::PFBlockElement::Type type1,
                          reco::PFBlockElement::Type type2,
                          const ElementListConst& elements,
                          const PFTables& tables,
                          const reco::PFMultiLinksIndex& multilinks) const = 0;

  const std::string& name() const { return _linkerName; }

private:
  const std::string _linkerName;
};

#include "FWCore/PluginManager/interface/PluginFactory.h"
typedef edmplugin::PluginFactory<BlockElementLinkerBase*(const edm::ParameterSet&)> BlockElementLinkerFactory;

#endif
