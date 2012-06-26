#include "CommonTools/Utils/src/ExpressionVar.h"
#include "Reflex/Object.h"
#include <Reflex/Builder/NewDelFunctions.h>
#include <map>
#include <assert.h>
using namespace reco::parser;
using namespace std;

ExpressionVar::ExpressionVar(const vector<MethodInvoker>& methods, method::TypeCode retType) : 
  methods_(methods), retType_(retType) { 
    initObjects_();
}

ExpressionVar::ExpressionVar(const ExpressionVar &other) :
  methods_(other.methods_), retType_(other.retType_) { 
    initObjects_();
}

ExpressionVar::~ExpressionVar() {
    for(std::vector<Reflex::Object>::iterator it = objects_.begin(); it != objects_.end(); ++it) {
        delStorage(*it);
    }
    objects_.clear();
}

void
ExpressionVar::delStorage(Reflex::Object &obj) {
    if (obj.Address() != 0) {
        if (obj.TypeOf().IsPointer() || obj.TypeOf().IsReference()) {
            // just delete a void *, as that's what it was
            void **p = static_cast<void **>(obj.Address());
            delete p;
        } else {
            //std::cout << "Calling Destruct on a " << obj.TypeOf().Name(QUALIFIED) << std::endl;
            obj.TypeOf().Deallocate(obj.Address());
        }
    }
}

void ExpressionVar::initObjects_() {
    objects_.resize(methods_.size());
    std::vector<MethodInvoker>::const_iterator it = methods_.begin(), ed = methods_.end();
    std::vector<Reflex::Object>::iterator itobj = objects_.begin();
    for (; it != ed; ++it, ++itobj) {
        needsDestructor_.push_back(makeStorage(*itobj, it->method()));
    }
}

bool
ExpressionVar::makeStorage(Reflex::Object &obj, const Reflex::Member &member) {
    bool ret = false;
    static Reflex::Type tVoid = Reflex::Type::ByName("void");
    if (member.IsFunctionMember()) {
        Reflex::Type retType = member.TypeOf().ReturnType();
        //remove any typedefs if any. If we do not do this it appears that we get a memory leak
        // because typedefs do not have 'destructors'
        retType = retType.FinalType();
        if (retType == tVoid) {
            obj = Reflex::Object(tVoid);
        } else if (retType.IsPointer() || retType.IsReference()) {
            // in this case, I have to allocate a void *, not an object!
            obj = Reflex::Object(retType, new void *);
        } else {
            obj = Reflex::Object(retType, retType.Allocate());
            ret = retType.IsClass();
            //std::cout << "ExpressionVar: reserved memory at "  << obj.Address() << " for a " << retType.Name(QUALIFIED) << " returned by " << member.Name() << std::endl;
        }
    } else { // no alloc, we don't need it
        obj = Reflex::Object();
    }
    return ret;
}

bool ExpressionVar::isValidReturnType(method::TypeCode retType)
{
   using namespace method;
   bool ret = false;
   switch(retType) {
      case(doubleType) : ret = true; break;
      case(floatType ) : ret = true; break;
      case(intType   ) : ret = true; break;
      case(uIntType  ) : ret = true; break;
      case(shortType ) : ret = true; break;
      case(uShortType) : ret = true; break;
      case(longType  ) : ret = true; break;
      case(uLongType ) : ret = true; break;
      case(charType  ) : ret = true; break;
      case(uCharType ) : ret = true; break;
      case(boolType  ) : ret = true; break;
      case(enumType  ) : ret = true; break;
      case(invalid):
      default:
        break;
   }
   return ret;
}

double ExpressionVar::value(const Reflex::Object & o) const {
  Reflex::Object ro = o;
  std::vector<MethodInvoker>::const_iterator itm, end = methods_.end();
  std::vector<Reflex::Object>::iterator      ito;
  for(itm = methods_.begin(), ito = objects_.begin(); itm != end; ++itm, ++ito) {
      ro = itm->invoke(ro, *ito);
  }
  double ret = objToDouble(ro, retType_);
  std::vector<Reflex::Object>::reverse_iterator rito, rend = objects_.rend();;
  std::vector<bool>::const_reverse_iterator ritb;
  for(rito = objects_.rbegin(), ritb = needsDestructor_.rbegin(); rito != rend; ++rito, ++ritb) {
      if (*ritb) rito->TypeOf().Destruct(rito->Address(), false);
  }
  return ret;
}

double
ExpressionVar::objToDouble(const Reflex::Object &obj, method::TypeCode type) {
  using namespace method;
  void * addr = obj.Address();
  double ret = 0;
  switch(type) {
  case(doubleType) : ret = * static_cast<double         *>(addr); break;
  case(floatType ) : ret = * static_cast<float          *>(addr); break;
  case(intType   ) : ret = * static_cast<int            *>(addr); break;
  case(uIntType  ) : ret = * static_cast<unsigned int   *>(addr); break;
  case(shortType ) : ret = * static_cast<short          *>(addr); break;
  case(uShortType) : ret = * static_cast<unsigned short *>(addr); break;
  case(longType  ) : ret = * static_cast<long           *>(addr); break;
  case(uLongType ) : ret = * static_cast<unsigned long  *>(addr); break;
  case(charType  ) : ret = * static_cast<char           *>(addr); break;
  case(uCharType ) : ret = * static_cast<unsigned char  *>(addr); break;
  case(boolType  ) : ret = * static_cast<bool           *>(addr); break;
  case(enumType  ) : ret = * static_cast<int            *>(addr); break;
  default:
  assert(false);
  };
  return ret;
}

void ExpressionVar::trueDelete(Reflex::Object & obj) {
     static std::map<void *, Reflex::NewDelFunctions *> deleters_;
     void * reflexTypeId = obj.TypeOf().Id();
     std::map<void *, Reflex::NewDelFunctions *>::iterator match = deleters_.find(reflexTypeId);
     if (match == deleters_.end()) {
         Reflex::NewDelFunctions *ptr;
         Reflex::Object newDel(Reflex::Type::ByTypeInfo(typeid(ptr)), &ptr);
         obj.Invoke("__getNewDelFunctions", &newDel);
         match = deleters_.insert(std::make_pair(reflexTypeId, ptr)).first;   
     }
     (*match->second->fDelete)(obj.Address());
}

ExpressionLazyVar::ExpressionLazyVar(const std::vector<LazyInvoker> & methods) :
    methods_(methods)
{
}

ExpressionLazyVar::~ExpressionLazyVar()
{
}

double
ExpressionLazyVar::value(const Reflex::Object & o) const {
    std::vector<LazyInvoker>::const_iterator it, ed = methods_.end()-1;
    Reflex::Object ro = o;
    for (it = methods_.begin(); it < ed; ++it) {
        ro = it->invoke(ro, objects_);
    }
    double ret = it->invokeLast(ro, objects_);
    std::vector<Reflex::Object>::reverse_iterator rit, red = objects_.rend();
    for (rit = objects_.rbegin(); rit != red; ++rit) {
        rit->TypeOf().Destruct(rit->Address(), false);
    }
    objects_.clear();
    return ret;
}

