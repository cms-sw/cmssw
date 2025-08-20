#include "CommonTools/Utils/interface/parser/MethodChain.h"
#include "CommonTools/Utils/interface/parser/MethodInvoker.h"

#include "FWCore/Reflection/interface/ObjectWithDict.h"
#include "FWCore/Reflection/interface/FunctionWithDict.h"
#include "FWCore/Reflection/interface/MemberWithDict.h"
#include "FWCore/Reflection/interface/TypeWithDict.h"

#include <cassert>
#include <map>

using namespace reco::parser;
using namespace std;

MethodChain::Objects MethodChain::initObjects_() const {
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

MethodChain::MethodChain(const vector<MethodInvoker>& methods) : methods_(methods) { returnObjects(initObjects_()); }

MethodChain::MethodChain(const MethodChain& rhs) : methods_(rhs.methods_) { returnObjects(initObjects_()); }

MethodChain::Objects MethodChain::borrowObjects() const {
  Objects objects;
  if (objectsCache_.try_pop(objects)) {
    return objects;
  }
  return initObjects_();
}

void MethodChain::returnObjects(Objects&& iOb) const { objectsCache_.push(std::move(iOb)); }

MethodChain::~MethodChain() {
  Objects objects;
  while (objectsCache_.try_pop(objects)) {
    for (auto& o : objects) {
      delStorage(o.first);
    }
  }
}

void MethodChain::delStorage(edm::ObjectWithDict& obj) {
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

bool MethodChain::makeStorage(edm::ObjectWithDict& obj, const edm::TypeWithDict& retType) {
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
    //std::cout << "MethodChain: reserved memory at "  << obj.address() <<
    //  " for a " << retType.qualifiedName() << " returned by " <<
    //  member.name() << std::endl;
  }
  return ret;
}

edm::ObjectWithDict MethodChain::value(const edm::ObjectWithDict& obj) const {
  edm::ObjectWithDict val(obj);
  auto objects = borrowObjects();
  auto IO = objects.begin();
  for (auto& m : methods_) {
    val = m.invoke(val, IO->first);
    ++IO;
  }
  for (auto RI = objects.rbegin(), RE = objects.rend(); RI != RE; ++RI) {
    if (RI->second) {
      RI->first.destruct(false);
    }
  }
  returnObjects(std::move(objects));
  return val;
}

LazyMethodChain::LazyMethodChain(const std::vector<LazyInvoker>& methods) : methods_(methods) {}

LazyMethodChain::~LazyMethodChain() {}

edm::ObjectWithDict LazyMethodChain::value(const edm::ObjectWithDict& o) const {
  edm::ObjectWithDict val = o;
  std::vector<StorageManager> storage;
  storage.reserve(methods_.size());

  std::vector<LazyInvoker>::const_iterator I = methods_.begin();
  std::vector<LazyInvoker>::const_iterator E = methods_.end();
  for (; I < E; ++I) {
    val = I->invoke(val, storage);
  }
  while (not storage.empty()) {
    storage.pop_back();
  }
  return val;
}
