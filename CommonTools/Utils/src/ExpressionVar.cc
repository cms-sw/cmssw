#include "CommonTools/Utils/src/ExpressionVar.h"
#include "FWCore/Utilities/interface/ObjectWithDict.h"
#include "FWCore/Utilities/interface/FunctionWithDict.h"
#include "FWCore/Utilities/interface/MemberWithDict.h"
#include "FWCore/Utilities/interface/TypeWithDict.h"
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
    for(std::vector<edm::ObjectWithDict>::iterator it = objects_.begin(); it != objects_.end(); ++it) {
        delStorage(*it);
    }
    objects_.clear();
}

void
ExpressionVar::delStorage(edm::ObjectWithDict &obj) {
    if (obj.address() != 0) {
        if (obj.typeOf().isPointer() || obj.typeOf().isReference()) {
            // just delete a void *, as that's what it was
            void **p = static_cast<void **>(obj.address());
            delete p;
        } else {
            //std::cout << "Calling Destruct on a " << obj.typeOf().qualifiedName() << std::endl;
            obj.typeOf().deallocate(obj.address());
        }
    }
}

void ExpressionVar::initObjects_() {
    objects_.resize(methods_.size());
    std::vector<MethodInvoker>::const_iterator it = methods_.begin(), ed = methods_.end();
    std::vector<edm::ObjectWithDict>::iterator itobj = objects_.begin();
    for (; it != ed; ++it, ++itobj) {
       if(it->isFunction()) {
          edm::TypeWithDict retType = it->method().finalReturnType();
          needsDestructor_.push_back(makeStorage(*itobj, retType));
       } else {
          *itobj = edm::ObjectWithDict();
          needsDestructor_.push_back(false);
       }
    }
}

bool
ExpressionVar::makeStorage(edm::ObjectWithDict &obj, const edm::TypeWithDict &retType) {
    bool ret = false;
    static edm::TypeWithDict tVoid(edm::TypeWithDict::byName("void"));
    if (retType == tVoid) {
        obj = edm::ObjectWithDict::byType(tVoid);
    } else if (retType.isPointer() || retType.isReference()) {
        // in this case, I have to allocate a void *, not an object!
        obj = edm::ObjectWithDict(retType, new void *);
    } else {
        obj = edm::ObjectWithDict(retType, retType.allocate());
        ret = retType.isClass();
        //std::cout << "ExpressionVar: reserved memory at "  << obj.address() << " for a " << retType.qualifiedName() << " returned by " << member.name() << std::endl;
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

double ExpressionVar::value(const edm::ObjectWithDict & o) const {
  edm::ObjectWithDict ro = o;
  std::vector<MethodInvoker>::const_iterator itm, end = methods_.end();
  std::vector<edm::ObjectWithDict>::iterator      ito;
  for(itm = methods_.begin(), ito = objects_.begin(); itm != end; ++itm, ++ito) {
      ro = itm->invoke(ro, *ito);
  }
  double ret = objToDouble(ro, retType_);
  std::vector<edm::ObjectWithDict>::reverse_iterator rito, rend = objects_.rend();;
  std::vector<bool>::const_reverse_iterator ritb;
  for(rito = objects_.rbegin(), ritb = needsDestructor_.rbegin(); rito != rend; ++rito, ++ritb) {
      if (*ritb) rito->typeOf().destruct(rito->address(), false);
  }
  return ret;
}

double
ExpressionVar::objToDouble(const edm::ObjectWithDict &obj, method::TypeCode type) {
  using namespace method;
  void * addr = obj.address();
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

ExpressionLazyVar::ExpressionLazyVar(const std::vector<LazyInvoker> & methods) :
    methods_(methods)
{
}

ExpressionLazyVar::~ExpressionLazyVar()
{
}

double
ExpressionLazyVar::value(const edm::ObjectWithDict & o) const {
    std::vector<LazyInvoker>::const_iterator it, ed = methods_.end()-1;
    edm::ObjectWithDict ro = o;
    for (it = methods_.begin(); it < ed; ++it) {
        ro = it->invoke(ro, objects_);
    }
    double ret = it->invokeLast(ro, objects_);
    std::vector<edm::ObjectWithDict>::reverse_iterator rit, red = objects_.rend();
    for (rit = objects_.rbegin(); rit != red; ++rit) {
        rit->typeOf().destruct(rit->address(), false);
    }
    objects_.clear();
    return ret;
}

