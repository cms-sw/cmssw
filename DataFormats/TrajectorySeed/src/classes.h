#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/CLHEP/interface/Migration.h" 
#include <boost/cstdint.hpp> 
#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h" 
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Common/interface/RefVectorIterator.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/Common/interface/AssociationMap.h"

namespace {
  struct dictionary {
    std::vector<TrajectorySeed> v1;
    TrajectorySeedCollection c1;
    edm::Wrapper<TrajectorySeedCollection> w1;

    edm::RefVectorIterator<std::vector<TrajectorySeed>,TrajectorySeed,edm::refhelper::FindUsingAdvance<std::vector<TrajectorySeed>,TrajectorySeed> > rfitr1;
    edm::Ref<TrajectorySeedCollection> s1;
    edm::RefProd<TrajectorySeedCollection> s2;
    edm::RefVector<TrajectorySeedCollection> s3;

    edm::RefToBase<TrajectorySeed> sr;  
    edm::reftobase::IndirectHolder<TrajectorySeed> ihs;
    edm::reftobase::Holder< TrajectorySeed, edm::Ref<TrajectorySeedCollection> > rbh;
    edm::reftobase::RefHolder< edm::Ref<TrajectorySeedCollection> > rbrh;
    edm::helpers::KeyVal<edm::RefProd<std::vector<reco::Track> >,edm::RefProd<std::vector<TrajectorySeed> > > x1;
    edm::AssociationMap<edm::OneToMany<std::vector<reco::Track>,std::vector<TrajectorySeed>,unsigned int> > x2;
    edm::Wrapper<edm::AssociationMap<edm::OneToMany<std::vector<reco::Track>,std::vector<TrajectorySeed>,unsigned int> > > x3;

  };
}
