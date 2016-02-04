#include <iostream>
#include <vector>
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/PtrVector.h"
#include "DataFormats/Common/interface/OwnVector.h"
#include <cassert>

#include <algorithm>
#include <ext/algorithm>
#include "boost/range.hpp"

struct Dummy { 
    Dummy() : id() {}
    Dummy(int i) : id(i) {}
    Dummy * clone() const { return new Dummy(*this); } 
    int id;
    bool operator==(const int &i) const { return id == i; }
};
typedef std::vector<Dummy> Coll;
Coll dummies_;

template<typename T>
void testFill(T &r) {
    for (int i = 0; i < 12; ++i) { r.push_back(typename T::value_type(i)); }
}
template<>
void testFill<edm::RefVector<Coll> >(edm::RefVector<Coll> &r) {
    edm::TestHandle<Coll> h(&dummies_, edm::ProductID(1));
    for (int i = 0; i < 12; ++i) { r.push_back(edm::Ref<Coll>(h,i)); }
}
template<>
void testFill<edm::PtrVector<Dummy> >(edm::PtrVector<Dummy> &r) {
    edm::TestHandle<Coll> h(&dummies_, edm::ProductID(1));
    for (int i = 0; i < 12; ++i) { r.push_back(edm::Ptr<Dummy>(h,i)); }
}
//template<>
//void testFill<edm::OwnVector<Dummy> >(edm::OwnVector<Dummy> &r) {
//    for (int i = 0; i < 12; ++i) { r.push_back(std::auto_ptr<Dummy>(new Dummy(i))); }
//}

bool operator==(const edm::Ref<Coll>  &ref, int i) { return (int(ref.key()) == i) && (ref->id == i); }
bool operator==(const edm::Ptr<Dummy> &ref, int i) { return (int(ref.key()) == i) && (ref->id == i); }
std::ostream & operator<<(std::ostream &o, const Dummy &v) { o << "Dummy(" << v.id << ")"; return o; }
std::ostream & operator<<(std::ostream &o, const edm::Ref<Coll>  &v) { o << "DummyRef(" << v.key() << " -> " << v->id << ")"; return o; }
std::ostream & operator<<(std::ostream &o, const edm::Ptr<Dummy> &v) { o << "DummyPtr(" << v.key() << " -> " << v->id << ")"; return o; }

template<typename R>
void testEquals(const typename R::value_type &v, int i, const char *name) {
    if (!(v == i)) { std::cout << "Error: " << v << " != " << i << " for " << typeid(R).name() << " at " << name <<std::endl;  }
}

template<typename R>
void test_read_value_fb(const R &r) {
    typename R::value_type v1 = r.front();
    typename R::value_type v2 = r.back();
    testEquals<R>(v1,4,__func__);
    testEquals<R>(v2,7,__func__);
}

template<typename R>
void test_read_value_brackets(const R &r) {
    typename R::value_type v1 = r[0];
    typename R::value_type v2 = r[3];
    testEquals<R>(v1,4,__func__);
    testEquals<R>(v2,7,__func__);
}

template<typename R>
void test_read_iter(const R &r) {
    int check = 4;
    for (typename R::const_iterator it = r.begin(), ed = r.end(); it != ed; ++it, ++check) {
        typename R::value_type v = *it;
        testEquals<R>(v,check,__func__);
    }
}

template<typename R>
void test_size(const R &r) {
    size_t i = r.size();
    if (i != 4) std::cout << "Error: r.size() = " << i << " != 4 for " << typeid(R).name() << " at " << __func__ <<std::endl; 
}

template<typename R>
void test_empty(const R &r) {
    bool b = r.empty();
    if (b) std::cout << "Error: r.empty() is true for " << typeid(R).name() << " at " << __func__ <<std::endl; 
}

struct DummySorter {
    bool operator()(const Dummy &t1, const Dummy &t2) const { return t1.id > t2.id; }
    bool operator()(const edm::Ref<Coll> &t1, const edm::Ref<Coll> &t2) const { return t1->id > t2->id; }
    bool operator()(const edm::Ptr<Dummy> &t1, const edm::Ptr<Dummy> &t2) const { return t1->id > t2->id; }
};
template<typename R>
void test_sort(R r) {
    //std::cout << "Check sort for " << typeid(R).name() << std::endl;
    //std::cout << "Before sort: " << std::endl;
    assert(!__gnu_cxx::is_sorted(r.begin(), r.end(), DummySorter()));
    //for (typename R::const_iterator it = r.begin(), ed = r.end(); it != ed; ++it) { std::cout << " - " << *it << std::endl; }
    std::sort(r.begin(), r.end(), DummySorter());
    //std::cout << "After sort: " << std::endl;
    //for (typename R::const_iterator it = r.begin(), ed = r.end(); it != ed; ++it) { std::cout << " - " << *it << std::endl; }
    //std::cout << "End check " << std::endl;
    if (!__gnu_cxx::is_sorted(r.begin(), r.end(), DummySorter())) {
        std::cout << "Sort for " << typeid(R).name() << " compiles but doesn't sort!" << std::endl;
    }
}

template<typename T>
void test_subrange(T t) {
    typedef boost::sub_range<const T> R;
    testFill(t);
    R r(t.begin()+4, t.begin()+8);
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
    R r(t.begin()+4, t.begin()+8);
    test_empty(r);    
    test_size(r);    
    test_read_iter(r);    
    test_read_value_fb(r);    
    test_read_value_brackets(r);    
    test_sort(r);
}

template<typename T>
void test_const_itr_is_const(const T &t) {
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

#define DISABLE_SORT_IT(Y)   template<> void test_sort<boost::iterator_range<Y::const_iterator> >( boost::iterator_range<Y::const_iterator> ) {}
#define DISABLE_SORT_SUB(Y)  template<> void test_sort<boost::sub_range<const Y> >(boost::sub_range<const Y> ) {} 
#define DISABLE_SORT_BARE(Y) template<> void test_sort<Y>(Y) { }
#define DISABLE_SORT_ALL(Y) DISABLE_SORT_IT(Y) \
                            DISABLE_SORT_SUB(Y)
#define DISABLE_CONST_ITR_IS_CONST(Y) template<> void test_const_itr_is_const<Y>(const Y &) { }
                            
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

int main(int argc, char **argv) {
   dummies_.clear();
   for (int i = 0; i < 12; ++i) dummies_.push_back(Dummy(i));
   test(vector<Dummy>());
   test(RefVector<Coll>());
   test(PtrVector<Dummy>());
   test(OwnVector<Dummy>());
}
