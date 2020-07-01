#include <algorithm>
#include <cassert>
#include <memory>

#include "DataFormats/Common/interface/OwnVector.h"
#include "FWCore/Utilities/interface/propagate_const.h"

namespace ownvector_test {
  struct Base {
    virtual ~Base();
    virtual Base* clone() const = 0;
  };

  Base::~Base() {}

  struct Derived : Base {
    explicit Derived(int n);
    Derived(Derived const& other);
    Derived& operator=(Derived const& other);
    ~Derived() override;
    void swap(Derived& other);
    Derived* clone() const override;

    edm::propagate_const<int*> pointer;
  };

  void Derived::swap(Derived& other) { std::swap(pointer, other.pointer); }
  Derived::Derived(int n) : pointer(new int(n)) {}

  Derived::Derived(Derived const& other) : pointer(new int(*other.pointer)) {}

  Derived& Derived::operator=(Derived const& other) {
    Derived temp(other);
    swap(temp);
    return *this;
  }

  Derived::~Derived() { delete pointer.get(); }

  Derived* Derived::clone() const { return new Derived(*this); }

  void swap(Derived& a, Derived& b) { a.swap(b); }

}  // namespace ownvector_test

using namespace ownvector_test;

namespace {

  /*
void same_guy_twice()
{
  edm::OwnVector<Base> vec;
  Base* p = new Derived(1);

  vec.push_back(p);
  vec.push_back(p);
}

void two_different_owners()
{
  edm::OwnVector<Base> v1,v2;
  Base* p = new Derived(1);
  v1.push_back(p);
  v2.push_back(p);
}
  */
  // void guy_on_stack()
  // {
  //   edm::OwnVector<Base> v;
  //   Derived d(10);
  //   v.push_back(&d);
  // }

  void copy_good_vec() {
    // v1 is perfectly fine...
    edm::OwnVector<Base> v1;
    Base* p = new Derived(100);
    v1.push_back(p);
    //v1.push_back(new Derived(100));

    // But what if we copy him?
    edm::OwnVector<Base> v2(v1);
  }

  void assign_to_other() {
    edm::OwnVector<Base> v1;
    Base* p = new Derived(100);
    v1.push_back(p);

    edm::OwnVector<Base> v2;
    v2 = v1;
  }

  void do_assign(edm::OwnVector<Base>& iLHS, edm::OwnVector<Base>& iRHS) { iLHS = iRHS; }

  void assign_to_self() {
    // Self-assignment happens, often by accident...
    edm::OwnVector<Base> v1;
    v1.push_back(new Derived(100));
    auto& v2 = v1;
    do_assign(v1, v2);
  }

  void pop_one() {
    edm::OwnVector<Base> v1;
    v1.push_back(new Derived(100));
    v1.pop_back();
  }

  void back_with_null_pointer() {
    edm::OwnVector<Base> v;
    Base* p = nullptr;
    v.push_back(p);
    try {
      v.back();
      assert("Failed to throw a required exception in OwnVector_t" == nullptr);
    } catch (edm::Exception& x) {
      // this is expected.
    } catch (...) {
      throw;
    }
  }

  void take_an_rvalue() {
    edm::OwnVector<Base> v;
    v.push_back(new Derived(101));
    Derived d(102);
    v.push_back(d.clone());
  }

  void take_an_lvalue() {
    edm::OwnVector<Base> v1;
    Base* p = new Derived(100);
    v1.push_back(p);

    assert(p == nullptr);
  }

  void take_an_auto_ptr() {
    edm::OwnVector<Base> v1;
    std::unique_ptr<Base> p = std::make_unique<Derived>(100);
    v1.push_back(std::move(p));
    assert(p.get() == nullptr);
  }

  void set_at_index() {
    edm::OwnVector<Base> v1;
    Base* p = new Derived(100);
    Base* backup = p;
    v1.push_back(p);
    assert(p == nullptr);
    assert(&v1[0] == backup);
    Base* p2 = new Derived(101);
    Base* backup2 = p2;
    assert(backup2 != backup);
    v1.set(0, p2);
    assert(p2 == nullptr);
    assert(&v1[0] == backup2);
  }

  void insert_with_iter() {
    edm::OwnVector<Base> v1;
    Base *p[3], *backup[3];
    for (int i = 0; i < 3; ++i) {
      backup[i] = p[i] = new Derived(100 + i);
    }
    v1.push_back(p[0]);
    v1.push_back(p[2]);
    v1.insert(v1.begin() + 1, p[1]);
    assert(p[0] == nullptr);
    assert(p[1] == nullptr);
    assert(p[2] == nullptr);
    assert(&v1[0] == backup[0]);
    assert(&v1[1] == backup[1]);
    assert(&v1[2] == backup[2]);
  }

}  // namespace

int main() {
  edm::OwnVector<Base> vec;
  vec.push_back(new Derived(100));
  edm::OwnVector<Base>* p = new edm::OwnVector<Base>;
  p->push_back(new Derived(2));
  delete p;
  //   same_guy_twice();
  //   two_different_owners();
  //   guy_on_stack();
  copy_good_vec();
  assign_to_other();
  assign_to_self();
  pop_one();
  back_with_null_pointer();

  take_an_rvalue();
  take_an_lvalue();
  take_an_auto_ptr();

  set_at_index();
  insert_with_iter();
}
