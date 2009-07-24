#include "CommonTools/Utils/src/ExpressionVar.h"
#include "Reflex/Object.h"
#include <Reflex/Builder/NewDelFunctions.h>
#include <map>
#include <assert.h>
using namespace reco::parser;
using namespace Reflex;
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
        if (it->Address() != 0) {
            if (it->TypeOf().IsPointer() || it->TypeOf().IsReference()) {
                // just delete a void *, as that's what it was
                void **p = static_cast<void **>(it->Address());
                delete p;
            } else {
	      //std::cout << "Calling Destruct on a " << it->TypeOf().Name(QUALIFIED) << std::endl;
	      it->Destruct(); 
            }
        }
    }
    objects_.clear();
}

void ExpressionVar::initObjects_() {
    objects_.reserve(methods_.size());
    static Type tVoid = Type::ByName("void");
    for (std::vector<MethodInvoker>::const_iterator it = methods_.begin(); it != methods_.end(); ++it) {
        if (it->method().IsFunctionMember()) {
            Reflex::Type retType = it->method().TypeOf().ReturnType();
	    //remove any typedefs if any. If we do not do this it appears that we get a memory leak
	    // because typedefs do not have 'destructors'
	    retType = retType.FinalType();
            if (retType == tVoid) {
                objects_.push_back(Reflex::Object(tVoid));
            } else if (retType.IsPointer() || retType.IsReference()) {
                // in this case, I have to allocate a void *, not an object!
                objects_.push_back(Reflex::Object(retType, new void *));
            } else {
                objects_.push_back(retType.Construct());
                //std::cout << "ExpressionVar: reserved memory at "  << objects_.back().Address() << " for a " << retType.Name(QUALIFIED) << " returned by " << it->method().Name() << std::endl;
            }
        } else { // no alloc, we don't need it
            objects_.push_back(Reflex::Object());
        }
    }
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
      case(invalid):
      default:
        break;
   }
   return ret;
}

double ExpressionVar::value(const Object & o) const {
  using namespace method;
  Object ro = o;
  std::vector<Object> toBeDeleted;
  std::vector<MethodInvoker>::const_iterator itm, end = methods_.end();
  std::vector<Reflex::Object>::iterator      ito;
  for(itm = methods_.begin(), ito = objects_.begin(); itm != end; ++itm, ++ito) {
      ro = itm->invoke(ro, *ito);
  }
  void * addr = ro.Address();
  double ret = 0;
  switch(retType_) {
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
