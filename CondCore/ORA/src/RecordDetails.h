#ifndef CondCore_ORA_RecordDetails_H
#define CondCore_ORA_RecordDetails_H
#include"AnyData.h"

#include <typeinfo>
#include <vector>
#include <algorithm>
#include <boost/type_traits/is_pointer.hpp>

namespace ora {

  struct TypeHandler {
    std::type_info const * type;
    virtual bool isPointer() const=0; 
    virtual  ~TypeHandler(){}
    virtual void const * address(const AnyData & ad) const=0;

    virtual void set(AnyData & ad, void * p) const=0;
    virtual void const * get(const AnyData & ad) const=0;
    virtual void create(AnyData & ad) const=0;
    virtual void destroy(AnyData & ad) const=0;

  };

  struct NullTypeHandler : public TypeHandler{
    NullTypeHandler(std::type_info const& t) { type=&t;}
    virtual bool isPointer() const { return false;} 
    virtual void const * address(const AnyData & ad) const{return 0;}
    virtual void set(AnyData &, void*) const{};
    virtual void const * get(const AnyData &) const{return 0;};
    virtual void create(AnyData &) const{};
    virtual void destroy(AnyData &) const{};

  };

  template<typename T> struct SimpleTypeHandler : public TypeHandler {
    SimpleTypeHandler() { type = &typeid(T);}
    virtual bool isPointer() const { return false;}
    virtual void const * address(const AnyData & ad) const { return get(ad);}

    virtual void set(AnyData & ad, void * p) const { ad.data<T>() =  *reinterpret_cast<T*>(p);}
    virtual void const * get(const AnyData & ad) const { return &ad.data<T>();}
    virtual void create(AnyData & ad) const{}
    virtual void destroy(AnyData & ad) const{}

  };

  template<typename T> struct AnyTypeHandler : public TypeHandler {
    AnyTypeHandler() { type = &typeid(T);}
    inline static bool inplace() { return sizeof(T) < 9;}
    virtual bool isPointer() const { return boost::is_pointer<T>::value;} 

    // return pointer to the value (remove pointer if T is pointer)
    virtual void const * address(const AnyData & ad) const {
      if (isPointer()) 
	return ad.p;
      else
	return get(ad);
    }


    virtual void set(AnyData & ad, void * p) const {
      if (inplace()) 
	*reinterpret_cast<T*>(ad.address()) =  *reinterpret_cast<T*>(p);
      else 
	*reinterpret_cast<T*>(ad.p) =  *reinterpret_cast<T*>(p);
    }

    virtual void const * get(const AnyData & ad) const { 
      if (inplace()) 
        return ad.address();
      else
	return ad.p;
    }
    virtual void create(AnyData & ad) const{
      if (inplace())
	new(ad.address()) T();
      else 
	ad.p = new T();
    }
    virtual void destroy(AnyData & ad) const{ 
      if (inplace())
	reinterpret_cast<T*>(ad.address())->~T();
      else 
	delete reinterpret_cast<T*>(ad.p);
    }
  };

  struct AllKnowTypeHandlers {
    AllKnowTypeHandlers();

    AnyTypeHandler<bool> b;
    AnyTypeHandler<char> c;
    AnyTypeHandler<unsigned char> uc;
    AnyTypeHandler<short> s;
    AnyTypeHandler<unsigned short> us;
    AnyTypeHandler<int> i;
    AnyTypeHandler<unsigned int> ui;
    AnyTypeHandler<long long> l;
    AnyTypeHandler<unsigned long long> ul;
    AnyTypeHandler<float> f;
    AnyTypeHandler<double> d;

    AnyTypeHandler<long double> ld;
    AnyTypeHandler<std::string> ss;

    AnyTypeHandler<bool*> bp;
    AnyTypeHandler<char*> cp;
    AnyTypeHandler<unsigned char*> ucp;
    AnyTypeHandler<short*> sp;
    AnyTypeHandler<unsigned short*> usp;
    AnyTypeHandler<int*> ip;
    AnyTypeHandler<unsigned int*> uip;
    AnyTypeHandler<long long*> lp;
    AnyTypeHandler<unsigned long long*> ulp;
    AnyTypeHandler<float*> fp;
    AnyTypeHandler<double*> dp;
  
    AnyTypeHandler<long double*> ldp;
    AnyTypeHandler<std::string*> ssp;

    std::vector<TypeHandler const *> all;
    typedef std::vector<TypeHandler const *>::const_iterator CI;
    TypeHandler const * operator()(std::type_info const & type) const;
  };

  struct CompareTypeHandler {
    bool operator()(TypeHandler const * rh, TypeHandler const * lh) {
      return rh->type < lh->type;
    }
  };

  AllKnowTypeHandlers::AllKnowTypeHandlers() {
    all.push_back(&b);
    all.push_back(&c);
    all.push_back(&uc);
    all.push_back(&s);
    all.push_back(&us);
    all.push_back(&i);
    all.push_back(&ui);
    all.push_back(&l);
    all.push_back(&ul);
    all.push_back(&f);
    all.push_back(&d);
    all.push_back(&ld);
    all.push_back(&ss);

    all.push_back(&bp);
    all.push_back(&cp);
    all.push_back(&ucp);
    all.push_back(&sp);
    all.push_back(&usp);
    all.push_back(&ip);
    all.push_back(&uip);
    all.push_back(&lp);
    all.push_back(&ulp);
    all.push_back(&fp);
    all.push_back(&dp);
    all.push_back(&ldp);
    all.push_back(&ssp);
    std::sort(all.begin(),all.end(),CompareTypeHandler());
  }

  TypeHandler const * AllKnowTypeHandlers::operator()(std::type_info const & type) const {
    NullTypeHandler h(type);
    std::pair<CI,CI> range =  std::equal_range(all.begin(),all.end(),&h,CompareTypeHandler());
    return (range.first==range.second) ?  (TypeHandler const *)(0) : *range.first;
  }


}
#endif //  CondCore_ORA_RecordDEtails_H
