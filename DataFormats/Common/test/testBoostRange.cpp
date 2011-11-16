#include "DataFormats/Common/interface/OwnVector.h"
#include "DataFormats/Common/interface/PtrVector.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "boost/range.hpp"

#include <algorithm>
#include <cassert>
#include <ext/algorithm>
#include <iostream>
#include <vector>

struct Dummy {
    Dummy() : id() {}
    Dummy(int i) : id(i) {}
    Dummy * clone() const { return new Dummy(*this); }
    int id;
    bool operator==(int const& i) const { return id == i; }
};
typedef std::vector<Dummy> Coll;
Coll dummies_;

template<typename T>
void testFill(T &r) {
    for(int i = 0; i < 12; ++i) { r.push_back(typename T::value_type(i)); }
}
template<>
void testFill<edm::RefVector<Coll> >(edm::RefVector<Coll> &r) {
    edm::TestHandle<Coll> h(&dummies_, edm::ProductID(1));
    for(int i = 0; i < 12; ++i) { r.push_back(edm::Ref<Coll>(h,i)); }
}
template<>
void testFill<edm::PtrVector<Dummy> >(edm::PtrVector<Dummy> &r) {
    edm::TestHandle<Coll> h(&dummies_, edm::ProductID(1));
    for(int i = 0; i < 12; ++i) { r.push_back(edm::Ptr<Dummy>(h,i)); }
}
//template<>
//void testFill<edm::OwnVector<Dummy> >(edm::OwnVector<Dummy> &r) {
//    for(int i = 0; i < 12; ++i) { r.push_back(std::auto_ptr<Dummy>(new Dummy(i))); }
//}

bool operator==(edm::Ref<Coll>  const& ref, int i) { return (int(ref.key()) == i) && (ref->id == i); }
bool operator==(edm::Ptr<Dummy> const& ref, int i) { return (int(ref.key()) == i) && (ref->id == i); }
std::ostream & operator<<(std::ostream &o, Dummy const& v) { o << "Dummy(" << v.id << ")"; return o; }
std::ostream & operator<<(std::ostream &o, edm::Ref<Coll> const& v) { o << "DummyRef(" << v.key() << " -> " << v->id << ")"; return o; }
std::ostream & operator<<(std::ostream &o, edm::Ptr<Dummy> const& v) { o << "DummyPtr(" << v.key() << " -> " << v->id << ")"; return o; }

template<typename R>
void testEquals(typename R::value_type const& v, int i, char const* name) {
    if(!(v == i)) { std::cout << "Error: " << v << " != " << i << " for " << typeid(R).name() << " at " << name << std::endl;  }
}

template<typename R>
void test_read_value_fb(R const& r) {
    typename R::value_type v1 = r.front();
    typename R::value_type v2 = r.back();
    testEquals<R>(v1,4,__func__);
    testEquals<R>(v2,7,__func__);
}

template<typename R>
void test_read_value_brackets(R const& r) {
    typename R::value_type v1 = r[0];
    typename R::value_type v2 = r[3];
    testEquals<R>(v1,4,__func__);
    testEquals<R>(v2,7,__func__);
}

template<typename R>
void test_read_iter(R const& r) {
    int check = 4;
    for(typename R::const_iterator it = r.begin(), ed = r.end(); it != ed; ++it, ++check) {
        typename R::value_type v = *it;
        testEquals<R>(v,check,__func__);
    }
}

template<typename R>
void test_size(R const& r) {
    size_t i = r.size();
    if(i != 4) std::cout << "Error: r.size() = " << i << " != 4 for " << typeid(R).name() << " at " << __func__ << std::endl;
}

template<typename R>
void test_empty(R const& r) {
    bool b = r.empty();
    if(b) std::cout << "Error: r.empty() is true for " << typeid(R).name() << " at " << __func__ << std::endl;
}

struct DummySorter {
    bool operator()(Dummy const& t1, Dummy const& t2) const { return t1.id > t2.id; }
    bool operator()(edm::Ref<Coll> const& t1, edm::Ref<Coll> const& t2) const { return t1->id > t2->id; }
    bool operator()(edm::Ptr<Dummy> const& t1, edm::Ptr<Dummy> const& t2) const { return t1->id > t2->id; }
};
template<typename R>
void test_sort(R r) {
    //std::cout << "Check sort for " << typeid(R).name() << std::endl;
    //std::cout << "Before sort: " << std::endl;
    assert(!__gnu_cxx::is_sorted(r.begin(), r.end(), DummySorter()));
    //for(typename R::const_iterator it = r.begin(), ed = r.end(); it != ed; ++it) { std::cout << " - " << *it << std::endl; }
    std::sort(r.begin(), r.end(), DummySorter());
    //std::cout << "After sort: " << std::endl;
    //for(typename R::const_iterator it = r.begin(), ed = r.end(); it != ed; ++it) { std::cout << " - " << *it << std::endl; }
    //std::cout << "End check " << std::endl;
    if(!__gnu_cxx::is_sorted(r.begin(), r.end(), DummySorter())) {
       std::cout << "Sort for " << typeid(R).name() << " compiles but doesn't sort!" << std::endl;
    }
}

template<typename T>
void test_subrange(T t) {
    typedef boost::sub_range<T const> R;
    testFill(t);
    R r(t.begin() + 4, t.begin() + 8);
    test_empty(r);
    test_size(r);
    test_read_iter(r);
    test_read_value_fb(r);
    test_read_value_brackets(r);
}

template<typename T>
void test_itrange(T t) {
    typedef boost::iterator_range<typename T::const_iterator> R;
    testFill(t);
    R r(t.begin() + 4, t.begin() + 8);
    test_empty(r);
    test_size(r);
    test_read_iter(r);
    test_read_value_fb(r);
    test_read_value_brackets(r);
    test_sort(r);
}

template<typename T>
void test_const_itr_is_const(T const& t) {
   typename T::const_iterator itr = t.begin();
   *itr = t[0];
}

template<typename T>
void test(T t) {
    testFill(t);
    test_const_itr_is_const(t);
    test_sort(t);
    test_itrange(t);
    test_subrange(t);
}


using namespace std;
using namespace edm;

#define DISABLE_SORT_IT(Y)   template<> void test_sort<boost::iterator_range<Y::const_iterator> >(boost::iterator_range<Y::const_iterator>) {}
#define DISABLE_SORT_SUB(Y)  template<> void test_sort<boost::sub_range<Y const> >(boost::sub_range<Y const> ) {}
#define DISABLE_SORT_BARE(Y) template<> void test_sort<Y>(Y) { }
#define DISABLE_SORT_ALL(Y) DISABLE_SORT_IT(Y) \
                            DISABLE_SORT_SUB(Y)
#define DISABLE_CONST_ITR_IS_CONST(Y) template<> void test_const_itr_is_const<Y>(Y const&) { }

// Comment out to check that std::sort does not compile for boost ranges of const X
DISABLE_SORT_ALL(std::vector<Dummy>)
DISABLE_SORT_ALL(edm::OwnVector<Dummy>)
DISABLE_SORT_ALL(edm::PtrVector<Dummy>)
DISABLE_SORT_ALL(edm::RefVector<Coll>)
// Comment out to check that you can't assign a value to the '*' of a const_iterator
DISABLE_CONST_ITR_IS_CONST(std::vector<Dummy>)
DISABLE_CONST_ITR_IS_CONST(edm::OwnVector<Dummy>)
DISABLE_CONST_ITR_IS_CONST(edm::PtrVector<Dummy>)
DISABLE_CONST_ITR_IS_CONST(edm::RefVector<Coll>)
// Comment out to check that std::sort doesn't compile or runs properly
DISABLE_SORT_BARE(edm::PtrVector<Dummy>)
DISABLE_SORT_BARE(edm::RefVector<Coll>)

int main(int, char**) try {
   dummies_.clear();
   for(int i = 0; i < 12; ++i) dummies_.push_back(Dummy(i));
   test(vector<Dummy>());
   test(RefVector<Coll>());
   test(PtrVector<Dummy>());
   test(OwnVector<Dummy>());
   return 0;
} catch(cms::Exception const& e) {
    std::cerr << e.explainSelf() << std::endl;
    return 1;
} catch(std::exception const& e) {
    std::cerr << e.what() << std::endl;
    return 1;
}
