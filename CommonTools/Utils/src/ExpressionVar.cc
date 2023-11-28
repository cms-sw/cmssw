#include "CommonTools/Utils/src/ExpressionVar.h"
#include "CommonTools/Utils/interface/parser/MethodInvoker.h"

#include "FWCore/Reflection/interface/ObjectWithDict.h"
#include "FWCore/Reflection/interface/FunctionWithDict.h"
#include "FWCore/Reflection/interface/MemberWithDict.h"
#include "FWCore/Reflection/interface/TypeWithDict.h"

#include <cassert>
#include <map>

using namespace reco::parser;
using namespace std;

ExpressionVar::Objects ExpressionVar::initObjects_() const {
  Objects objects(methods_.size(), {edm::ObjectWithDict(), false});
  assert(objects.size() == methods_.size());
  auto IO = objects.begin();
  for (auto const& method : methods_) {
    if (method.isFunction()) {
      edm::TypeWithDict retType = method.method().finalReturnType();
      IO->second = makeStorage(IO->first, retType);
    } else {
      *IO = {edm::ObjectWithDict(), false};
    }
    ++IO;
  }
  return objects;
}

ExpressionVar::ExpressionVar(const vector<MethodInvoker>& methods, method::TypeCode retType)
    : methods_(methods), retType_(retType) {
  returnObjects(initObjects_());
}

ExpressionVar::ExpressionVar(const ExpressionVar& rhs) : methods_(rhs.methods_), retType_(rhs.retType_) {
  returnObjects(initObjects_());
}

ExpressionVar::Objects ExpressionVar::borrowObjects() const {
  Objects objects;
  if (objectsCache_.try_pop(objects)) {
    return objects;
  }
  return initObjects_();
}

void ExpressionVar::returnObjects(Objects&& iOb) const { objectsCache_.push(std::move(iOb)); }

ExpressionVar::~ExpressionVar() {
  Objects objects;
  while (objectsCache_.try_pop(objects)) {
    for (auto& o : objects) {
      delStorage(o.first);
    }
  }
}

void ExpressionVar::delStorage(edm::ObjectWithDict& obj) {
  if (!obj.address()) {
    return;
  }
  if (obj.typeOf().isPointer() || obj.typeOf().isReference()) {
    // just delete a void*, as that's what it was
    void** p = static_cast<void**>(obj.address());
    delete p;
  } else {
    //std::cout << "Calling Destruct on a " <<
    //  obj.typeOf().qualifiedName() << std::endl;
    obj.typeOf().deallocate(obj.address());
  }
}

bool ExpressionVar::makeStorage(edm::ObjectWithDict& obj, const edm::TypeWithDict& retType) {
  static const edm::TypeWithDict tVoid(edm::TypeWithDict::byName("void"));
  bool ret = false;
  if (retType == tVoid) {
    obj = edm::ObjectWithDict::byType(tVoid);
  } else if (retType.isPointer() || retType.isReference()) {
    // in this case, I have to allocate a void*, not an object!
    obj = edm::ObjectWithDict(retType, new void*);
  } else {
    obj = edm::ObjectWithDict(retType, retType.allocate());
    ret = retType.isClass();
    //std::cout << "ExpressionVar: reserved memory at "  << obj.address() <<
    //  " for a " << retType.qualifiedName() << " returned by " <<
    //  member.name() << std::endl;
  }
  return ret;
}

bool ExpressionVar::isValidReturnType(method::TypeCode retType) {
  using namespace method;
  bool ret = false;
  switch (retType) {
    case (doubleType):
      ret = true;
      break;
    case (floatType):
      ret = true;
      break;
    case (intType):
      ret = true;
      break;
    case (uIntType):
      ret = true;
      break;
    case (shortType):
      ret = true;
      break;
    case (uShortType):
      ret = true;
      break;
    case (longType):
      ret = true;
      break;
    case (uLongType):
      ret = true;
      break;
    case (charType):
      ret = true;
      break;
    case (uCharType):
      ret = true;
      break;
    case (boolType):
      ret = true;
      break;
    case (enumType):
      ret = true;
      break;
    case (invalid):
    default:
      break;
  }
  return ret;
}

double ExpressionVar::value(const edm::ObjectWithDict& obj) const {
  edm::ObjectWithDict val(obj);
  auto objects = borrowObjects();
  auto IO = objects.begin();
  for (auto& m : methods_) {
    val = m.invoke(val, IO->first);
    ++IO;
  }
  double ret = objToDouble(val, retType_);
  for (auto RI = objects.rbegin(), RE = objects.rend(); RI != RE; ++RI) {
    if (RI->second) {
      RI->first.destruct(false);
    }
  }
  returnObjects(std::move(objects));
  return ret;
}

double ExpressionVar::objToDouble(const edm::ObjectWithDict& obj, method::TypeCode type) {
  using namespace method;
  void* addr = obj.address();
  double ret = 0.0;
  switch (type) {
    case doubleType:
      ret = *static_cast<double*>(addr);
      break;
    case floatType:
      ret = *static_cast<float*>(addr);
      break;
    case intType:
      ret = *static_cast<int*>(addr);
      break;
    case uIntType:
      ret = *static_cast<unsigned int*>(addr);
      break;
    case shortType:
      ret = *static_cast<short*>(addr);
      break;
    case uShortType:
      ret = *static_cast<unsigned short*>(addr);
      break;
    case longType:
      ret = *static_cast<long*>(addr);
      break;
    case uLongType:
      ret = *static_cast<unsigned long*>(addr);
      break;
    case charType:
      ret = *static_cast<char*>(addr);
      break;
    case uCharType:
      ret = *static_cast<unsigned char*>(addr);
      break;
    case boolType:
      ret = *static_cast<bool*>(addr);
      break;
    case enumType:
      ret = *static_cast<int*>(addr);
      break;
    default:
      //FIXME: Error not caught in production build!
      assert(false && "objToDouble: invalid type!");
      break;
  };
  return ret;
}

ExpressionLazyVar::ExpressionLazyVar(const std::vector<LazyInvoker>& methods) : methods_(methods) {}

ExpressionLazyVar::~ExpressionLazyVar() {}

double ExpressionLazyVar::value(const edm::ObjectWithDict& o) const {
  edm::ObjectWithDict val = o;
  std::vector<StorageManager> storage;
  storage.reserve(methods_.size());

  std::vector<LazyInvoker>::const_iterator I = methods_.begin();
  std::vector<LazyInvoker>::const_iterator E = methods_.end() - 1;
  for (; I < E; ++I) {
    val = I->invoke(val, storage);
  }
  double ret = I->invokeLast(val, storage);
  while (not storage.empty()) {
    storage.pop_back();
  }
  return ret;
}
