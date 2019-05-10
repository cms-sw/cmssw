
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "CommonTools/UtilAlgos/interface/ObjectSelector.h"
#include "Alignment/CommonAlignmentProducer/interface/AlignmentSeedSelector.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"

struct SeedConfigSelector {
  typedef std::vector<const TrajectorySeed *> container;
  typedef container::const_iterator const_iterator;
  typedef TrajectorySeedCollection collection;

  SeedConfigSelector(const edm::ParameterSet &cfg, edm::ConsumesCollector &&iC) : theSelector(cfg) {}

  const_iterator begin() const { return selected_.begin(); }
  const_iterator end() const { return selected_.end(); }
  size_t size() const { return selected_.size(); }

  void select(const edm::Handle<TrajectorySeedCollection> c, const edm::Event &evt, const edm::EventSetup & /*dummy*/) {
    all_.clear();
    selected_.clear();
    for (collection::const_iterator i = c.product()->begin(), iE = c.product()->end(); i != iE; ++i) {
      all_.push_back(&*i);
    }
    selected_ = theSelector.select(all_, evt);  // might add dummy...
  }

private:
  container all_, selected_;
  AlignmentSeedSelector theSelector;
};

typedef ObjectSelector<SeedConfigSelector> AlignmentSeedSelectorModule;

DEFINE_FWK_MODULE(AlignmentSeedSelectorModule);
