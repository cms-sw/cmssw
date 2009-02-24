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
  std::pair<Object,bool> ro(o,false);
  std::vector<Object> toBeDeleted;
  for(vector<MethodInvoker>::const_iterator m = methods_.begin();
      m != methods_.end(); ++m) {
      if (ro.second) { toBeDeleted.push_back(ro.first); }
      ro = m->value(ro.first);
  }
  for (std::vector<Object>::iterator it = toBeDeleted.begin(), ed = toBeDeleted.end(); it != ed; ++it) {
      //std::cout << "Should delete Object at " << it->Address() << ", type = " << it->TypeOf().Name() << std::endl;
      //it->Destruct(); // this is not ok, it uses "free" while we need "delete"
      trueDelete(*it);
  }
  void * addr = ro.first.Address();
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
         Reflex::Object newDel;
         obj.Invoke("__getNewDelFunctions", newDel);
         Reflex::NewDelFunctions *ptr = static_cast<Reflex::NewDelFunctions *>(newDel.Address());
         match = deleters_.insert(std::make_pair(reflexTypeId, ptr)).first;   
     }
     (*match->second->fDelete)(obj.Address());
}
