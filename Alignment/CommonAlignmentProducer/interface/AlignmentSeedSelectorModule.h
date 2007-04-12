
#include "PhysicsTools/UtilAlgos/interface/ObjectSelector.h"
#include "Alignment/CommonAlignmentProducer/interface/AlignmentSeedSelector.h"

struct SeedConfigSelector {

  typedef std::vector<const TrajectorySeed*> container;
  typedef container::const_iterator const_iterator;
  typedef TrajectorySeedCollection collection; 

  SeedConfigSelector( const edm::ParameterSet & cfg ) :
    theSelector(cfg) {}

  const_iterator begin() const { return selected_.begin(); }
  const_iterator end() const { return selected_.end(); }
  size_t size() const { return selected_.size(); }

  void select( const edm::Handle<TrajectorySeedCollection> c,  const edm::Event & evt) {
    all_.clear();
    selected_.clear();
    for( TrajectorySeedCollection::const_iterator i=c.product()->begin();i!=c.product()->end();++i){
      all_.push_back( & * i );
    }
    selected_=theSelector.select(all_,evt);
  }

private:
  container all_,selected_;
  AlignmentSeedSelector theSelector;
};

typedef ObjectSelector<SeedConfigSelector>  AlignmentSeedSelectorModule;

