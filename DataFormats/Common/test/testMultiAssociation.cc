#include "DataFormats/Common/interface/OwnVector.h"
#include "DataFormats/Common/interface/PtrVector.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/TestHandle.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "cppunit/extensions/HelperMacros.h"

#include "boost/lambda/bind.hpp"
#include "boost/lambda/lambda.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <memory>

#include "DataFormats/Common/interface/MultiAssociation.h"

namespace {
    struct DummyBase {
        virtual ~DummyBase(){}
        virtual int id() const { return 0; }
        virtual DummyBase* clone() const { return new DummyBase(*this); }
        void const* addr() const { return this; }
    };
    struct DummyDer1 : public DummyBase {     virtual int id() const { return 1; } virtual DummyDer1* clone() const { return new DummyDer1(*this); } };
    struct DummyDer2 : public DummyBase {     virtual int id() const { return 2; } virtual DummyDer2* clone() const { return new DummyDer2(*this); } };
}

using namespace edm;

class testMultiAssociation : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testMultiAssociation);
  CPPUNIT_TEST(checkAll);
  CPPUNIT_TEST(checkVals);
  CPPUNIT_TEST(checkTwoFillers);
  CPPUNIT_TEST(checkUnsortedKeys);
  CPPUNIT_TEST(checkBadFill);
  CPPUNIT_TEST(checkBadRead);
  CPPUNIT_TEST(checkWithPtr);
  CPPUNIT_TEST(checkWithOwn);
  CPPUNIT_TEST(checkWritableMap);
  CPPUNIT_TEST_SUITE_END();
  typedef std::vector<double> CVal;  // Values
  typedef std::vector<int>    CKey1; // Keys 1
  typedef std::vector<float>  CKey2; // Keys 2
  typedef std::vector<DummyDer1> CObj;
  typedef Ptr<DummyBase>   PObj;
  typedef PtrVector<DummyBase> PObjs;
  typedef OwnVector<DummyBase> OObj;
  typedef MultiAssociation<RefVector<CVal> > MultiRef;
  typedef MultiAssociation<CVal>             MultiVal;
  typedef MultiAssociation<PObjs>            MultiPtr;
  typedef MultiAssociation<OObj>             MultiOwn;
  /**
   * Map to each key, all values that are greater than key
   * Try both with MultiAssociation<RefVector<double>> and  MultiAssociation<vector<double> >
   */
public:
  testMultiAssociation();
  void setUp() {}
  void tearDown() {}
  void checkAll();
  void checkVals();
  void checkTwoFillers();
  void checkUnsortedKeys();
  void checkBadFill();
  void checkBadRead();
  template<typename Map> bool tryTwoFillers(bool lazy) ;
  template<typename Map> bool tryUnsortedKeys(bool lazy) ;
  bool tryBadFill(int i) ;
  bool tryBadRead(int i) ;
  void checkWithPtr();
  void checkWithOwn();
  void checkWritableMap();
  void test(MultiRef const&);
  void test(MultiVal const&);
  void test2(MultiRef const&);
#ifdef private
  void dump(MultiRef::Indices const&);
  template<typename T> void dump(RefVector<T> const&);
  template<typename T> void dump(std::vector<T> const&);
  void dump(MultiRef const&, char const* when);
  void dump(MultiVal const&, char const* when);
#else
  template<typename T> void dump(RefVector<T> const&) {}
  template<typename T> void dump(std::vector<T> const&) {}
  void dump(MultiRef const&, char const*/*when*/) {}
  void dump(MultiVal const&, char const*/*when*/) {}
#endif
  CVal k;
  CKey1 v1;
  CKey2 v2;
  CObj  der1s;
  PObjs ptrs;
  OObj  bases;
  edm::TestHandle<CVal>  handleV;
  edm::TestHandle<CKey1> handleK1;
  edm::TestHandle<CKey2> handleK2;
  std::vector<int> w1, w2;

    template<typename Key, typename UnaryFunc, typename BinaryFunc>
    void fastFillRefs(edm::TestHandle<Key> const& handle, MultiRef& map, UnaryFunc const& u, BinaryFunc const& f) {
      MultiRef::FastFiller filler = map.fastFiller(handle);
      for(typename Key::const_iterator it = handle->begin(), ed = handle->end(); it != ed; ++it) {
          if(!u(*it)) continue;
          RefVector<CVal> vals;
          for(std::vector<double>::const_iterator it2 = k.begin(), ed2 = k.end(); it2 != ed2; ++it2) {
              if(f(*it,*it2)) {
                  vals.push_back(Ref<CVal>(handleV, it2 - k.begin()));
              }
          }
          filler.setValues(Ref<Key>(handle, it - handle->begin()), vals);
      }
    }

    template<typename Key, typename UnaryFunc, typename BinaryFunc>
    void lazyFillRefs(edm::TestHandle<Key> const& handle, MultiRef& map, UnaryFunc const& u, BinaryFunc const& f, bool swap) {
      MultiRef::LazyFiller filler = map.lazyFiller(handle, true);
      for(typename Key::const_iterator it = handle->begin(), ed = handle->end(); it != ed; ++it) {
          if(!u(*it)) continue;
          RefVector<CVal> vals;
          for(std::vector<double>::const_iterator it2 = k.begin(), ed2 = k.end(); it2 != ed2; ++it2) {
              if(f(*it,*it2)) {
                  vals.push_back(Ref<CVal>(handleV, it2 - k.begin()));
              }
          }
          if(swap) {
              filler.swapValues(Ref<Key>(handle, it - handle->begin()), vals);
          } else {
              filler.setValues(Ref<Key>(handle, it - handle->begin()), vals);
          }
      }
    }

    template<typename Key, typename UnaryFunc, typename BinaryFunc>
    void fastFillVals(edm::TestHandle<Key> const& handle, MultiVal& map, UnaryFunc const& u, BinaryFunc const& f) {
      MultiVal::FastFiller filler = map.fastFiller(handle);
      for(typename Key::const_iterator it = handle->begin(), ed = handle->end(); it != ed; ++it) {
          if(!u(*it)) continue;
          CVal vals;
          for(std::vector<double>::const_iterator it2 = k.begin(), ed2 = k.end(); it2 != ed2; ++it2) {
              if(f(*it,*it2)) {
                  vals.push_back(*it2);
              }
          }
          filler.setValues(Ref<Key>(handle, it - handle->begin()), vals);
      }
    }

    template<typename Key, typename UnaryFunc, typename BinaryFunc>
    void lazyFillVals(edm::TestHandle<Key> const& handle, MultiVal& map, UnaryFunc const& u, BinaryFunc const& f, bool swap) {
      MultiVal::LazyFiller filler = map.lazyFiller(handle, true);
      for(typename Key::const_iterator it = handle->begin(), ed = handle->end(); it != ed; ++it) {
          if(!u(*it)) continue;
          CVal vals;
          for(std::vector<double>::const_iterator it2 = k.begin(), ed2 = k.end(); it2 != ed2; ++it2) {
              if(f(*it,*it2)) {
                  vals.push_back(*it2);
              }
          }
          if(swap) {
              filler.swapValues(Ref<Key>(handle, it - handle->begin()), vals);
          } else {
              filler.setValues(Ref<Key>(handle, it - handle->begin()), vals);
          }
      }
    }

    //template<typename Map, typename Filler> tryUnsortedKeys() ;

};

CPPUNIT_TEST_SUITE_REGISTRATION(testMultiAssociation);

testMultiAssociation::testMultiAssociation() {
  k.push_back(1.1);
  k.push_back(2.2);
  k.push_back(3.3);
  k.push_back(4.4);
  ProductID const pidV(1);
  handleV = edm::TestHandle<CVal>(&k, pidV);

  v1.push_back(1);
  v1.push_back(2);
  v1.push_back(3);
  v1.push_back(4);
  ProductID const pidK1(2);
  handleK1 = edm::TestHandle<CKey1>(&v1, pidK1);

  v2.push_back(1.);
  v2.push_back(2.);
  v2.push_back(3.);
  v2.push_back(4.);
  v2.push_back(5.);
  ProductID const pidK2(3);
  handleK2 = edm::TestHandle<CKey2>(&v2, pidK2);

  for(size_t j = 0; j < 10; ++j) der1s.push_back(DummyDer1());
  for(size_t j = 0; j < 10; ++j) {
        if(j % 3 == 0) bases.push_back(std::auto_ptr<DummyBase>(new DummyBase()));
        if(j % 3 == 1) bases.push_back(std::auto_ptr<DummyDer1>(new DummyDer1()));
        if(j % 3 == 2) bases.push_back(std::auto_ptr<DummyDer2>(new DummyDer2()));
        CPPUNIT_ASSERT(bases[j].id() == int(j % 3));
  }
  edm::TestHandle<CObj> handleObj(&der1s, ProductID(10));
  for(size_t j = 0; j < 7; ++j) {
    size_t k = (j * 37) % 10;
    ptrs.push_back(PObj(handleObj, k));
    CPPUNIT_ASSERT(ptrs[j]->id() == 1);
    CPPUNIT_ASSERT(ptrs[j]->addr() == &der1s[k]);
  }
}

void testMultiAssociation::checkAll() {
  using boost::lambda::_1;
  using boost::lambda::_2;
  {
    MultiRef try1;
    dump(try1, "empty");
    fastFillRefs(handleK1, try1, _1 > 0, _1 > _2);
    dump(try1, "fill 1");
    fastFillRefs(handleK2, try1, _1 > 0, _1 > _2);
    dump(try1, "fill 2");
    test(try1);
  }
  {
    MultiRef try2;
    fastFillRefs(handleK1, try2, _1 > 0, (_1 < _2) && (2*bind(floor,_1/2) != _1) ); // fill all, but leave empty the odds
    fastFillRefs(handleK2, try2, _1 > 0, (_1 < _2) && (2*bind(floor,_1/2) != _1) ); // fill all, but leave empty the odds
    dump(try2, "fill 2");
    test2(try2);
  }
  {
    MultiRef try2;
    fastFillRefs(handleK1, try2, (2*bind(floor,_1/2) != _1), (_1 < _2) && (2*bind(floor,_1/2) != _1) ); // don't fill the odds
    fastFillRefs(handleK2, try2, (2*bind(floor,_1/2) != _1), (_1 < _2) && (2*bind(floor,_1/2) != _1) ); // don't fill the odds
    dump(try2, "fill 2");
    test2(try2);
  }
  {
    MultiRef try3;
    fastFillRefs(handleK1, try3, _1 < 0, (_1 > _2)); // don't fill any of the first
    fastFillRefs(handleK2, try3, _1 > 0, (_1 > _2)); //
    dump(try3, "no first");
  }
  {
    MultiRef try3;
    fastFillRefs(handleK1, try3, _1 > 0, (_1 > _2)); //
    fastFillRefs(handleK2, try3, _1 < 0, (_1 > _2)); // don't fill the second
    dump(try3, "no second");
  }
  {
    MultiRef try3;
    fastFillRefs(handleK1, try3, _1 < 0, (_1 > _2)); // don't fill any of the first
    fastFillRefs(handleK2, try3, _1 < 0, (_1 > _2)); // nor the second
    dump(try3, "neither");
  }
  {
    MultiRef try1;
    lazyFillRefs(handleK1, try1, _1 > 0, _1 > _2, false);
    lazyFillRefs(handleK2, try1, _1 > 0, _1 > _2, false);
    dump(try1, "fill 2");
    test(try1);
  }
  {
    MultiRef try2;
    lazyFillRefs(handleK1, try2, (2*bind(floor,_1/2) != _1), (_1 < _2) && (2*bind(floor,_1/2) != _1), false ); // don't fill the odds
    lazyFillRefs(handleK2, try2, (2*bind(floor,_1/2) != _1), (_1 < _2) && (2*bind(floor,_1/2) != _1), false ); // don't fill the odds
    dump(try2, "fill 2");
    test2(try2);
  }
  {
    MultiRef try1;
    lazyFillRefs(handleK1, try1, _1 > 0, _1 > _2, true);
    lazyFillRefs(handleK2, try1, _1 > 0, _1 > _2, true);
    dump(try1, "fill 2");
    test(try1);
  }
  {
    MultiRef try2;
    lazyFillRefs(handleK1, try2, (2*bind(floor,_1/2) != _1), (_1 < _2) && (2*bind(floor,_1/2) != _1), true ); // don't fill the odds
    lazyFillRefs(handleK2, try2, (2*bind(floor,_1/2) != _1), (_1 < _2) && (2*bind(floor,_1/2) != _1), true ); // don't fill the odds
    dump(try2, "fill 2");
    test2(try2);
  }


}

void testMultiAssociation::checkVals() {
  using boost::lambda::_1;
  using boost::lambda::_2;
  {
    MultiVal try1;
    dump(try1, "empty");
    fastFillVals(handleK1, try1, _1 > 0, _1 > _2);
    dump(try1, "fill 1");
    fastFillVals(handleK2, try1, _1 > 0, _1 > _2);
    dump(try1, "fill 2");
    test(try1);
  }
  {
    MultiVal try1;
    lazyFillVals(handleK1, try1, _1 > 0, _1 > _2, false);
    lazyFillVals(handleK2, try1, _1 > 0, _1 > _2, false);
    dump(try1, "fill 2");
    test(try1);
  }
  {
    MultiVal try1;
    lazyFillVals(handleK1, try1, _1 > 0, _1 > _2, true);
    lazyFillVals(handleK2, try1, _1 > 0, _1 > _2, true);
    dump(try1, "fill 2");
    test(try1);
  }
}


#ifdef private
void testMultiAssociation::dump(MultiRef::Indices const& indices) {
    using namespace std;
    cerr <<  "    Dumping Index map at " << &indices << endl;
    cerr <<  "    id_offsets_ (size = " << indices.id_offsets_.size() << ")" << endl;
    for(size_t i = 0; i < indices.id_offsets_.size(); ++i) {
         cerr <<  "      [" << setw(3) << i << "]: (" << setw(3) << indices.id_offsets_[i].first
                                            << ", " << setw(3) << indices.id_offsets_[i].second << ")" << endl;
    }
    cerr <<  "    ref_offsets_ (size = " << indices.ref_offsets_.size() << ")" << endl;
    for(size_t i = 0; i < indices.ref_offsets_.size(); ++i) {
         cerr <<  "      [" << setw(3) << i << "]: " << setw(4) << indices.ref_offsets_[i] << ")" << endl;
    }
    cerr <<  "    isFilling_: " << indices.isFilling_ << endl;

}
template<typename T>
void testMultiAssociation::dump(edm::RefVector<T> const& data) {
    using namespace std;
    cerr <<  "  Dumping " << typeid(data).name() << " at " << &data << endl;
    cerr <<  "    ID: " << data.id() << endl;
    cerr <<  "    Values (size = " << data.size() << ")" << endl;
    for(size_t i = 0; i < data.size(); ++i) {
        cerr << "      [" << setw(3) << i << "]: key = " << setw(4) << data[i].key();
        if(data[i].isNull()) cerr << ", NULL" << endl;
        else cerr << ", value = " << *data[i] << endl;
    }
}
template<typename T>
void testMultiAssociation::dump(std::vector<T> const& data) {
    using namespace std;
    cerr <<  "  Dumping " << typeid(data).name() << " at " << &data << endl;
    cerr <<  "    Values (size = " << data.size() << ")" << endl;
    for(size_t i = 0; i < data.size(); ++i) {
        cerr << "      [" << setw(3) << i << "]: key = " << setw(4) << data[i] << endl;
    }
}

void testMultiAssociation::dump(MultiRef const& assoc, char const* what) {
    using namespace std;
    cerr << "\nDumping MultiRef at " << &assoc << " for " << what << endl;
    dump(assoc.indices_);
    dump(assoc.data_);
    cerr << endl;
}
void testMultiAssociation::dump(MultiVal const& assoc, char const* what) {
    using namespace std;
    cerr << "\nDumping MultiVal at " << &assoc << " for " << what << endl;
    dump(assoc.indices_);
    dump(assoc.data_);
    cerr << endl;
}

#endif

void testMultiAssociation::test(MultiRef const& assoc) {
  // TEST contains
  CPPUNIT_ASSERT(!assoc.contains(ProductID(1)));
  CPPUNIT_ASSERT(assoc.contains(ProductID(2)));
  CPPUNIT_ASSERT(assoc.contains(ProductID(3)));
  CPPUNIT_ASSERT(!assoc.contains(ProductID(4)));

  // TEST const_range access
  MultiRef::const_range br1,br2,br3,br4;
  br1 = assoc[edm::Ref<CKey1>(handleK1, 0)];
  br2 = assoc[edm::Ref<CKey1>(handleK1, 1)];
  br3 = assoc[edm::Ref<CKey1>(handleK1, 2)];
  br4 = assoc[edm::Ref<CKey1>(handleK1, 3)];
  CPPUNIT_ASSERT(br1.size() == 0);
  CPPUNIT_ASSERT(br2.size() == 1);
  CPPUNIT_ASSERT(br3.size() == 2);
  CPPUNIT_ASSERT(br4.size() == 3);
  CPPUNIT_ASSERT(br2.begin()->id()  == ProductID(1));
  CPPUNIT_ASSERT(br2.begin()->key() == 0);
  CPPUNIT_ASSERT(**br2.begin() == k.front());
  CPPUNIT_ASSERT(*br2.front() == k.front());
  CPPUNIT_ASSERT(*br2[0] == k[0]);
  CPPUNIT_ASSERT(br4.back().id()  == ProductID(1));
  CPPUNIT_ASSERT(br4.back().key() == 2);
  CPPUNIT_ASSERT(br4[2].key() == 2);

  // TEST const_ranges
  // Check that ranges are consecutive
  br1 = assoc[edm::Ref<CKey1>(handleK1, 0)];
  CPPUNIT_ASSERT(br1.end() == br1.begin());
  br2 = assoc[edm::Ref<CKey1>(handleK1, 1)];
  CPPUNIT_ASSERT(br2.end() == br2.begin() + 1);
  CPPUNIT_ASSERT(br2.begin()  == br1.end());
  br3 = assoc[edm::Ref<CKey1>(handleK1, 2)];
  CPPUNIT_ASSERT(br3.end() == br3.begin() + 2);
  CPPUNIT_ASSERT(br3.begin()  == br2.end());
  // Check that ranges are consecutive across collections
  br1 = assoc[edm::Ref<CKey1>(handleK1, 3)];
  br2 = assoc[edm::Ref<CKey2>(handleK2, 0)];
  br3 = assoc[edm::Ref<CKey2>(handleK2, 1)];
  CPPUNIT_ASSERT(br1.end() == br1.begin() + 3);
  CPPUNIT_ASSERT(br1.end() == br2.begin() + 0);
  CPPUNIT_ASSERT(br2.end() == br2.begin() + 0);
  CPPUNIT_ASSERT(br2.end() == br3.begin() + 0);
  CPPUNIT_ASSERT(br3.end() == br3.begin() + 1);

  // TEST RefVector access
  edm::RefVector<CVal> r1,r2,r3,r4,r5;
  r1 = assoc.getValues(edm::Ref<CKey1>(handleK1, 0));
  r2 = assoc.getValues(edm::Ref<CKey1>(handleK1, 1));
  r3 = assoc.getValues(edm::Ref<CKey1>(handleK1, 2));
  r4 = assoc.getValues(edm::Ref<CKey1>(handleK1, 3));
  CPPUNIT_ASSERT(r1.size() == 0);
  CPPUNIT_ASSERT(r2.size() == 1);
  CPPUNIT_ASSERT(r3.size() == 2);
  CPPUNIT_ASSERT(r4.size() == 3);
  CPPUNIT_ASSERT(r2.begin()->id()  == ProductID(1));
  CPPUNIT_ASSERT(r2.begin()->key() == 0);
  CPPUNIT_ASSERT(**r2.begin() == k.front());
  CPPUNIT_ASSERT(**r2.begin() == k.front());
  CPPUNIT_ASSERT((r4.end()-1)->id()  == ProductID(1));
  CPPUNIT_ASSERT((r4.end()-1)->key() == 2);
}

void testMultiAssociation::test2(MultiRef const& assoc) {
  // TEST contains
  CPPUNIT_ASSERT(!assoc.contains(ProductID(1)));
  CPPUNIT_ASSERT(assoc.contains(ProductID(2)));
  CPPUNIT_ASSERT(assoc.contains(ProductID(3)));
  CPPUNIT_ASSERT(!assoc.contains(ProductID(4)));

  // TEST const_range access
  MultiRef::const_range br1,br2,br3,br4;
  br1 = assoc[edm::Ref<CKey1>(handleK1, 0)];
  CPPUNIT_ASSERT(br1.end() == br1.begin() + 4);
  br2 = assoc[edm::Ref<CKey1>(handleK1, 1)];
  CPPUNIT_ASSERT(br2.end() == br2.begin() + 0);
  CPPUNIT_ASSERT(br2.begin()  == br1.end());
  br3 = assoc[edm::Ref<CKey1>(handleK1, 2)];
  CPPUNIT_ASSERT(br3.end() == br3.begin() + 2);
  CPPUNIT_ASSERT(br3.begin()  == br2.end());
  // Check that ranges are consecutive across collections
  br1 = assoc[edm::Ref<CKey1>(handleK1, 3)];
  br2 = assoc[edm::Ref<CKey2>(handleK2, 0)];
  br3 = assoc[edm::Ref<CKey2>(handleK2, 1)];
  CPPUNIT_ASSERT(br1.end() == br1.begin() + 0);
  CPPUNIT_ASSERT(br1.end() == br2.begin() + 0);
  CPPUNIT_ASSERT(br2.end() == br2.begin() + 4);
  CPPUNIT_ASSERT(br2.end() == br3.begin() + 0);
  CPPUNIT_ASSERT(br3.end() == br3.begin() + 0);
}

void testMultiAssociation::test(MultiVal const& assoc) {
#if 1
  // TEST Vector access
  MultiVal::const_range r1,r2,r3,r4,r5;
  r1 = assoc[edm::Ref<CKey1>(handleK1, 0)];
  r2 = assoc[edm::Ref<CKey1>(handleK1, 1)];
  r3 = assoc[edm::Ref<CKey1>(handleK1, 2)];
  r4 = assoc[edm::Ref<CKey1>(handleK1, 3)];
#else
  // TEST Vector access
  CVal r1,r2,r3,r4,r5;
  r1 = assoc.getValues(edm::Ref<CKey1>(handleK1, 0));
  r2 = assoc.getValues(edm::Ref<CKey1>(handleK1, 1));
  r3 = assoc.getValues(edm::Ref<CKey1>(handleK1, 2));
  r4 = assoc.getValues(edm::Ref<CKey1>(handleK1, 3));
#endif
  CPPUNIT_ASSERT(r1.size() == 0);
  CPPUNIT_ASSERT(r2.size() == 1);
  CPPUNIT_ASSERT(r3.size() == 2);
  CPPUNIT_ASSERT(r4.size() == 3);
  CPPUNIT_ASSERT(r2[0] == k[0]);
  CPPUNIT_ASSERT(r3[0] == k[0]);
  CPPUNIT_ASSERT(r3[1] == k[1]);
  CPPUNIT_ASSERT(r4[0] == k[0]);
  CPPUNIT_ASSERT(r4[1] == k[1]);
  CPPUNIT_ASSERT(r4[2] == k[2]);
}



template<typename Map>
bool testMultiAssociation::tryTwoFillers(bool lazyfiller) {
    Map map;
    if(lazyfiller) {
        typename Map::LazyFiller filler1(map, handleK1, true);
        typename Map::LazyFiller filler2(map, handleK2, true);
    } else {
        typename Map::FastFiller filler1(map, handleK1);
        typename Map::FastFiller filler2(map, handleK2);
    }
    return true;
}
void testMultiAssociation::checkTwoFillers() {
    CPPUNIT_ASSERT(tryTwoFillers<MultiRef>(true));
    CPPUNIT_ASSERT_THROW(tryTwoFillers<MultiRef>(false), cms::Exception);
    CPPUNIT_ASSERT(tryTwoFillers<MultiVal>(true));
    CPPUNIT_ASSERT_THROW(tryTwoFillers<MultiVal>(false), cms::Exception);
}
template<typename Map>
bool testMultiAssociation::tryUnsortedKeys(bool lazy) {
    Map map;
    typename Map::Collection coll1, coll2;
    if(lazy) {
        typename Map::LazyFiller filler(map, handleK1, true);
        filler.setValues(Ref<CKey1>(handleK1, 1), coll1);
        filler.setValues(Ref<CKey1>(handleK1, 0), coll2);
    } else {
        typename Map::FastFiller filler(map, handleK1);
        filler.setValues(Ref<CKey1>(handleK1, 1), coll1);
        filler.setValues(Ref<CKey1>(handleK1, 0), coll2);
    }
    return true;
}
void testMultiAssociation::checkUnsortedKeys() {
    CPPUNIT_ASSERT(tryUnsortedKeys<MultiRef>(true));
    CPPUNIT_ASSERT_THROW(tryUnsortedKeys<MultiRef>(false), cms::Exception);
    CPPUNIT_ASSERT(tryUnsortedKeys<MultiVal>(true));
    CPPUNIT_ASSERT_THROW(tryUnsortedKeys<MultiVal>(false), cms::Exception);
}

// i = even: succeed; i = odd: fail
bool testMultiAssociation::tryBadFill(int i) {
    MultiRef m;
    MultiRef::Collection coll1;
    coll1.push_back(Ref<CVal>(handleV,1));
    switch (i) {
        case 0:
            {// fill with right prod. id
                MultiRef::FastFiller filler = m.fastFiller(handleK1);
                filler.setValues(Ref<CKey1>(handleK1,0), coll1);
            }; break;
        case 1:
            {// fill with wrong prod. id
                MultiRef::FastFiller filler = m.fastFiller(handleK1);
                filler.setValues(Ref<CKey2>(handleK2,0), coll1);
            }; break;
        case 2:
            {// fill again with different id
                { MultiRef::FastFiller filler = m.fastFiller(handleK1); }
                { MultiRef::FastFiller filler = m.fastFiller(handleK2); }
            };
            break;
        case 3:
            {// fill again with the same id
                { MultiRef::FastFiller filler = m.fastFiller(handleK1); }
                { MultiRef::FastFiller filler = m.fastFiller(handleK1); }
            };
            break;
        case 4:
            {// Check lazyFiller doesn't fill if not requested
                { MultiRef::LazyFiller filler = m.lazyFiller(handleK1); }
                { MultiRef::LazyFiller filler = m.lazyFiller(handleK1); }
            };
            break;
        case 5:
            {// Check lazyFiller can't fill twice the same key if requested
                { MultiRef::LazyFiller filler = m.lazyFiller(handleK1,true); }
                { MultiRef::LazyFiller filler = m.lazyFiller(handleK1,true); }
            };
            break;
        case 6:
            {// Check lazyFiller doesn't fill twice by mistake
                MultiRef::LazyFiller filler = m.lazyFiller(handleK1,true);
                CPPUNIT_ASSERT(m.empty());
                filler.fill();
                CPPUNIT_ASSERT(!m.empty());
                filler.fill();
            };
            break;
        case 8:
            { // Check lazyFiller doesn't fill if not requested
              { MultiRef::LazyFiller filler = m.lazyFiller(handleK1,false); }
              CPPUNIT_ASSERT(m.empty());
            }
            break;
        case 9:
            { // Check index out of bounds
              MultiRef::FastFiller filler = m.fastFiller(handleK1);
              filler.setValues(Ref<CKey1>(handleK1,handleK1->size()+5,false), coll1);
            }
            break;
        case 10:
            { // Can copy a LazyFiller, if I don't fill twice
              MultiRef::LazyFiller filler = m.lazyFiller(handleK1,false);
              MultiRef::LazyFiller filler2 = filler;
              filler2.setValues(Ref<CKey1>(handleK1,0), coll1);
              filler2.fill();
            }
            break;
        case 11:
            { // Can copy a LazyFiller, but crash if I fill twice
              MultiRef::LazyFiller filler = m.lazyFiller(handleK1,true);
              MultiRef::LazyFiller filler2 = filler;
            }
            break;
        case 12:
            { // Can copy a FastFiller
              MultiRef::FastFiller filler = m.fastFiller(handleK1);
              MultiRef::FastFiller filler2 = filler;
            }
            break;
        default:
            if(i % 2 == 1) throw cms::Exception("Programmed failure");
            break;
    }
    return true;
}
void testMultiAssociation::checkBadFill() {
    for(int i = 0; i < 100; ++i) {
        if(i % 2 == 0) CPPUNIT_ASSERT(tryBadFill(i));
        else CPPUNIT_ASSERT_THROW(tryBadFill(i), cms::Exception);
    }
}

bool testMultiAssociation::tryBadRead(int i) {
    using boost::lambda::_1;
    using boost::lambda::_2;
    MultiRef m;
    fastFillRefs(handleK1, m, _1 > 0, _1 > _2);
    fastFillRefs(handleK2, m, _1 > 0, _1 > _2);
    switch (i) {
        case 0: // good id and key
            m[Ref<CKey1>(handleK1,0,false)] ;
            break;
        case 1: // wrong id
            m[Ref<CVal >(handleV ,0,false)] ;
            break;
        case 3: // wrong id & key, and outside bounds
                // this does crash
            m[Ref<CKey2>(handleK2,5,false)] ;
            break;
        case 5: // wrong id & key, but still within bounds
                // we check explicitly for this in Indexconst_rangeAssociation::get, even if it costs
                // extra time
            m[Ref<CKey1>(handleK1,5,false)] ;
            break;
        default:
            if(i % 2 == 1) throw cms::Exception("Programmed failure");
            break;
    }
    return true;
}
void testMultiAssociation::checkWritableMap() {
    using boost::lambda::_1;
    using boost::lambda::_2;
    MultiVal tryRW;
    fastFillVals(handleK1, tryRW, _1 > 0, _1 > _2);
    fastFillVals(handleK2, tryRW, _1 > 0, _1 > _2);
    test(tryRW);

    MultiVal::range r1, r1bis;
    r1 = tryRW[edm::Ref<CKey1>(handleK1, 1)];
    CPPUNIT_ASSERT(r1[0] == k[0]);
    r1[0] = k[1];
    // check that we modified the range
    CPPUNIT_ASSERT(r1[0] == k[1]);
    // check that even the real thing got modified
    r1bis = tryRW[edm::Ref<CKey1>(handleK1, 1)];
    CPPUNIT_ASSERT(r1bis[0] == k[1]);

}

void testMultiAssociation::checkBadRead() {
    for(int i = 0; i < 100; ++i) {
        if(i % 2 == 0) CPPUNIT_ASSERT(tryBadRead(i));
        else CPPUNIT_ASSERT_THROW(tryBadRead(i), cms::Exception);
    }
}

void testMultiAssociation::checkWithPtr() {
    MultiPtr map;
    { // Fill the map
      MultiPtr::FastFiller filler = map.fastFiller(handleK1);
      edm::TestHandle<CObj> handleObj(&der1s, ProductID(10));
      for(size_t i = 0; i < handleK1->size(); ++i) {
        PObjs vals;
        for(size_t j = 0; j < ((i+2) % 3); ++j) {
            vals.push_back( PObj(handleObj, (3*i+4*j)%10) );
        }
        if(!vals.empty()) filler.setValues(Ref<CKey1>(handleK1, i), vals);
      }
    }
    { // Read the map
      for(size_t i = 0; i < handleK1->size(); ++i) {
        MultiPtr::const_range r = map[Ref<CKey1>(handleK1, i)];
        CPPUNIT_ASSERT(static_cast<size_t>(r.size()) == ((i+2) % 3));
        for(size_t j = 0; j < ((i+2) % 3); ++j) {
            CPPUNIT_ASSERT(  r[j].key()   ==        (3*i+4*j)%10 );
            CPPUNIT_ASSERT(  r[j]->addr() == &der1s[(3*i+4*j)%10]);
            //CPPUNIT_ASSERT(  (r.begin()+j)->key()   ==        (3*i+4*j)%10 );
            //CPPUNIT_ASSERT(  (*(r.begin()+j))->addr() == &der1s[(3*i+4*j)%10]);
        }
      }
    }
}
void testMultiAssociation::checkWithOwn() {
    MultiOwn map;
    { // Fill the map
      MultiOwn::FastFiller filler = map.fastFiller(handleK1);
      for(size_t i = 0; i < handleK1->size(); ++i) {
        OObj vals;
        for(size_t j = 0; j < ((i+2) % 3); ++j) {
            vals.push_back( bases[(i+j)%3].clone() );
        }
        if(!vals.empty()) filler.setValues(Ref<CKey1>(handleK1, i), vals);
      }
    }
    { // Read the map
      for(size_t i = 0; i < handleK1->size(); ++i) {
        MultiOwn::const_range r = map[Ref<CKey1>(handleK1, i)];
        CPPUNIT_ASSERT(static_cast<size_t>(r.size()) == ((i+2) % 3));
        for(size_t j = 0; j < ((i+2) % 3); ++j) {
            CPPUNIT_ASSERT((r.begin()+j)->id() == bases[(i+j)%3].id());
        }
      }
    }

}
