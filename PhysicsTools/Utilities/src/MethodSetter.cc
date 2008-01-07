#include "PhysicsTools/Utilities/src/MethodSetter.h"
#include "PhysicsTools/Utilities/src/returnType.h"
#include "PhysicsTools/Utilities/src/findMethod.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include <string>
using namespace reco::parser;
using namespace std;
using namespace ROOT::Reflex;

void MethodSetter::operator()(const char * begin, const char * end) const {
  string methodName(begin, end);
  string::size_type endOfExpr = methodName.find_last_of(' ');
  if(endOfExpr != string::npos)
    methodName.erase(endOfExpr, methodName.size());
  push(methodName);
}

void MethodSetter::push(const string & methodName) const {
  Type type = typeStack_.back();
  pair<Member, bool> mem = reco::findMethod(type, methodName);
  methStack_.push_back(MethodInvoker(mem.first));
  Type retType = reco::returnType(mem.first);
  typeStack_.push_back(retType);
  // check for edm::Ref, edm::RefToBase, edm::Ptr
  if(mem.second) push(methodName);
}
    
