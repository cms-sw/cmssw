#include <algorithm>
#include <cassert>
#include <memory>

#include "DataFormats/Common/interface/OwnVector.h"


struct Base
{
  virtual ~Base();
  virtual Base* clone() const = 0;
};

Base::~Base() {}

struct Derived : Base
{
  explicit Derived(int n);
  Derived(Derived const& other);
  Derived& operator=(Derived const& other);
  virtual ~Derived();
  void swap(Derived& other);
  virtual Derived* clone() const;

  int*  pointer;
};

Derived::Derived(int n) : pointer(new int(n)) { }

Derived::Derived(Derived const& other) : pointer(new int(*other.pointer)) { }

Derived& Derived::operator=(Derived const& other) 
{
  Derived temp(other);
  swap(temp);
  return *this;
}

void Derived::swap(Derived& other)
{
  std::swap(pointer, other.pointer);
}

void swap(Derived& a, Derived& b)
{
  a.swap(b);
}

Derived::~Derived()
{
  delete pointer;
}

Derived*
Derived::clone() const
{
  return new Derived(*this); 
}



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

// void guy_on_stack()
// {
//   edm::OwnVector<Base> v;
//   Derived d(10);
//   v.push_back(&d);  
// }

void copy_good_vec()
{
  // v1 is perfectly fine...
  edm::OwnVector<Base> v1;
  Base* p = new Derived(100);
  v1.push_back(p);
  //v1.push_back(new Derived(100));

  // But what if we copy him?
  edm::OwnVector<Base> v2(v1);
}

void assign_to_other()
{
  edm::OwnVector<Base> v1;
  Base* p = new Derived(100);
  v1.push_back(p);

  edm::OwnVector<Base> v2;
  v2 = v1;
}

void assign_to_self()
{
  // Self-assignment happens, often by accident...
  edm::OwnVector<Base> v1;
  v1.push_back(new Derived(100));
  v1 = v1;
}

void pop_one()
{
  edm::OwnVector<Base> v1;
  v1.push_back(new Derived(100));
  v1.pop_back();
}

void back_with_null_pointer()
{
  edm::OwnVector<Base> v;
  Base* p = 0;
  v.push_back(p);
  try
    {
      v.back();
      assert("Failed to throw a required exception in OwnVector_t"==0);
    }
  catch (edm::Exception& x)
    {
      // this is expected.
    }
  catch (...)
    {
      throw;
    }
}

void take_an_rvalue()
{
  edm::OwnVector<Base> v;
  v.push_back(new Derived(101));
  Derived d(102);
  v.push_back(d.clone());
}

void take_an_lvalue()
{
  edm::OwnVector<Base> v1;
  Base* p = new Derived(100);
  v1.push_back(p);

  assert(p == 0);
}

void take_an_auto_ptr()
{
  edm::OwnVector<Base> v1;
  std::auto_ptr<Base> p(new Derived(100));
  v1.push_back(p);
  assert(p.get() == 0);
}

int main()
{
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
}
