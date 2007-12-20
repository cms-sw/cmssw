#include "PhysicsTools/Utilities/src/MethodSetter.h"
#include "PhysicsTools/Utilities/src/returnType.h"
#include "PhysicsTools/Utilities/src/findMethod.h"
#include <string>
#include <iostream>
using namespace reco::parser;
using namespace std;
using namespace ROOT::Reflex;

void MethodSetter::operator()(const char * begin, const char * end) const {
  string methodName(begin, end);
  string::size_type endOfExpr = methodName.find_last_of(' ');
  if(endOfExpr != string::npos)
    methodName.erase(endOfExpr, methodName.size());

  Type type = typeStack_.back();
  ROOT::Reflex::Member mem = reco::findMethod(type, methodName);
  cerr << ">>> adding to stack member: " << mem.Name() << endl;
  methStack_.push_back(MethodInvoker(mem));
  Type retType = reco::returnType(mem);
  cerr << ">>> adding to stack member return type : " << retType.Name() << endl;
  typeStack_.push_back(retType);
}
