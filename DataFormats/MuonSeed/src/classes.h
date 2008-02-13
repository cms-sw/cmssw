#include "DataFormats/MuonSeed/interface/L2MuonTrajectorySeedCollection.h"
#include "DataFormats/MuonSeed/interface/L3MuonTrajectorySeedCollection.h"
#include "DataFormats/CLHEP/interface/Migration.h" 
#include <boost/cstdint.hpp> 

#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Common/interface/RefVectorIterator.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefToBase.h"
    
namespace {
  namespace {
    std::vector<L2MuonTrajectorySeed> v1;
    L2MuonTrajectorySeedCollection c1;
    edm::Wrapper<L2MuonTrajectorySeedCollection> w1;
    
    edm::RefVectorIterator<std::vector<L2MuonTrajectorySeed>,L2MuonTrajectorySeed,edm::refhelper::FindUsingAdvance<std::vector<L2MuonTrajectorySeed>,L2MuonTrajectorySeed> > rfitr1;
    edm::Ref<L2MuonTrajectorySeedCollection> s1;
    edm::RefProd<L2MuonTrajectorySeedCollection> s2;
    edm::RefVector<L2MuonTrajectorySeedCollection> s3;
    
    edm::RefToBase<L2MuonTrajectorySeed> sr;  
    edm::reftobase::IndirectHolder<L2MuonTrajectorySeed> ihs;
    edm::reftobase::Holder< L2MuonTrajectorySeed, edm::Ref<L2MuonTrajectorySeedCollection> > rbh;
    edm::reftobase::RefHolder< edm::Ref<L2MuonTrajectorySeedCollection> > rbrh;
  }
  namespace {
    std::vector<L3MuonTrajectorySeed> v12;
    L3MuonTrajectorySeedCollection c12;
    edm::Wrapper<L3MuonTrajectorySeedCollection> w12;
    
    edm::RefVectorIterator<std::vector<L3MuonTrajectorySeed>,L3MuonTrajectorySeed,edm::refhelper::FindUsingAdvance<std::vector<L3MuonTrajectorySeed>,L3MuonTrajectorySeed> > rfitr12;
    edm::Ref<L3MuonTrajectorySeedCollection> s12;
    edm::RefProd<L3MuonTrajectorySeedCollection> s22;
    edm::RefVector<L3MuonTrajectorySeedCollection> s32;
    
    edm::RefToBase<L3MuonTrajectorySeed> sr2;  
    edm::reftobase::IndirectHolder<L3MuonTrajectorySeed> ihs2;
    edm::reftobase::Holder< L3MuonTrajectorySeed, edm::Ref<L3MuonTrajectorySeedCollection> > rbh2;
    edm::reftobase::RefHolder< edm::Ref<L3MuonTrajectorySeedCollection> > rbrh2;
  }
}
