#include "PhysicsTools/Utilities/src/MethodSetter.h"
#include "PhysicsTools/Utilities/src/returnType.h"
#include "PhysicsTools/Utilities/src/findMethod.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include <string>
using namespace reco::parser;
using namespace std;
using namespace ROOT::Reflex;

void MethodSetter::operator()(const char * begin, const char * end) const {
  string name(begin, end);
  string::size_type parenthesis = name.find_first_of('(');
  std::vector<int> args;
  if(parenthesis != string::npos) {
    name.erase(parenthesis, name.size());
    if(intStack_.size()==0)
      throw edm::Exception(edm::errors::Configuration)
	<< "expected method argument, but integer stack is empty\n";    
    for(vector<int>::const_iterator i = intStack_.begin(); i != intStack_.end(); ++i)
      args.push_back(*i);
    intStack_.clear();
  }
  string::size_type endOfExpr = name.find_last_of(' ');
  if(endOfExpr != string::npos)
    name.erase(endOfExpr, name.size());
  push(name, args);
}

void MethodSetter::push(const string & name, const vector<int> & args) const {
  Type type = typeStack_.back();
  pair<Member, bool> mem = reco::findMethod(type, name, args.size());
  if(!mem.first)
    throw edm::Exception(edm::errors::Configuration)
      << "method \"" << name << "\" not found for type \"" 
      << type.Name() << "\"\n";
  Type retType = reco::returnType(mem.first);
  typeStack_.push_back(retType);
  // check for edm::Ref, edm::RefToBase, edm::Ptr
  if(mem.second) {
    methStack_.push_back(MethodInvoker(mem.first));
    push(name, args);
  } else
    methStack_.push_back(MethodInvoker(mem.first, args));
}
    
