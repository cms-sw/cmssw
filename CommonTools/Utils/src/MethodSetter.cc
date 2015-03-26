#include "CommonTools/Utils/src/MethodSetter.h"

#include "CommonTools/Utils/interface/Exception.h"
#include "CommonTools/Utils/src/ErrorCodes.h"
#include "CommonTools/Utils/src/MethodInvoker.h"
#include "CommonTools/Utils/src/findDataMember.h"
#include "CommonTools/Utils/src/findMethod.h"
#include "CommonTools/Utils/src/returnType.h"

#include <string>

using namespace reco::parser;
using namespace std;

void
MethodSetter::
operator()(const char* begin, const char* end) const
{
  string name(begin, end);
  string::size_type parenthesis = name.find_first_of('(');
  if ((*begin == '[') || (*begin == '(')) {
    name.insert(0, "operator..");    // operator..[arg];
    parenthesis = 10;                //           ^--- idx = 10
    name[8] = *begin;                // operator[.[arg];
    name[9] = name[name.size() - 1]; // operator[][arg];
    name[10] = '(';                  // operator[](arg];
    name[name.size() - 1] = ')';     // operator[](arg);
    // we don't actually need the last two, but just for extra care
    //std::cout << "Transformed {" << string(begin,end) << "} into
    //  {"<< name <<"}" << std::endl;
  }
  std::vector<AnyMethodArgument> args;
  if (parenthesis != string::npos) {
    name.erase(parenthesis, name.size());
    if (intStack_.size() == 0) {
      throw Exception(begin)
          << "expected method argument, but non given.";
    }
    for (vector<AnyMethodArgument>::const_iterator i = intStack_.begin();
         i != intStack_.end(); ++i) {
      args.push_back(*i);
    }
    intStack_.clear();
  }
  string::size_type endOfExpr = name.find_last_of(' ');
  if (endOfExpr != string::npos) {
    name.erase(endOfExpr, name.size());
  }
  //std::cerr << "Pushing [" << name << "] with " << args.size()
  //  << " args " << (lazy_ ? "(lazy)" : "(immediate)") << std::endl;
  if (lazy_) {
    // for lazy parsing we just push method name and arguments
    lazyMethStack_.push_back(LazyInvoker(name, args));
  }
  else {
    // otherwise we really have to resolve the method
    push(name, args, begin);
  }
  //std::cerr << "Pushed [" << name << "] with " << args.size() <<
  //  " args " << (lazy_ ? "(lazy)" : "(immediate)") << std::endl;
}

bool
MethodSetter::
push(const string& name, const vector<AnyMethodArgument>& args,
     const char* begin, bool deep) const
{
  edm::TypeWithDict type = typeStack_.back();
  vector<AnyMethodArgument> fixups;
  int error = 0;
  pair<edm::FunctionWithDict, bool> mem =
    reco::findMethod(type, name, args, fixups, begin, error);
  if (bool(mem.first)) {
    // We found the method.
    edm::TypeWithDict retType = reco::returnType(mem.first);
    if (!bool(retType)) {
      // Invalid return type, fatal error, throw.
      throw Exception(begin)
          << "member \"" << mem.first.name()
          << "\" return type is invalid:\n"
          << "  return type: \""
          << mem.first.typeName() << "\"\n";
    }
    typeStack_.push_back(retType);
    // check for edm::Ref, edm::RefToBase, edm::Ptr
    if (mem.second) {
      // Without fixups.
      //std::cout << "Mem.second, so book " << mem.first.name() <<
      //  " without fixups." << std::endl;
      methStack_.push_back(MethodInvoker(mem.first));
      if (!deep) {
        return false;
      }
      // note: we have not found the method, so we have not
      // fixupped the arguments
      push(name, args, begin);
    }
    else {
      // With fixups.
      //std::cout << "Not mem.second, so book " << mem.first.name()
      //  << " with #args = " << fixups.size() << std::endl;
      methStack_.push_back(MethodInvoker(mem.first, fixups));
    }
    return true;
  }
  if (error != reco::parser::kNameDoesNotExist) {
    // Fatal error, throw.
    switch (error) {
      case reco::parser::kIsNotPublic:
        throw Exception(begin)
            << "method named \"" << name << "\" for type \""
            << type.name() << "\" is not publically accessible.";
        break;
      case reco::parser::kIsStatic:
        throw Exception(begin)
            << "method named \"" << name << "\" for type \""
            << type.name() << "\" is static.";
        break;
      case reco::parser::kIsNotConst:
        throw Exception(begin)
            << "method named \"" << name << "\" for type \""
            << type.name() << "\" is not const.";
        break;
      case reco::parser::kWrongNumberOfArguments:
        throw Exception(begin)
            << "method named \"" << name << "\" for type \""
            << type.name() << "\" was passed the wrong number of arguments.";
        break;
      case reco::parser::kWrongArgumentType:
        throw Exception(begin)
            << "method named \"" << name << "\" for type \""
            << type.name() << "\" was passed the wrong types of arguments.";
        break;
      default:
        throw Exception(begin)
            << "method named \"" << name << "\" for type \""
            << type.name() << "\" is not usable in this context.";
    }
  }
  // Not a method, check for a data member.
  error = 0;
  edm::MemberWithDict member(reco::findDataMember(type, name, error));
  if (!bool(member)) {
    // Not a data member either, fatal error, throw.
    switch (error) {
      case reco::parser::kNameDoesNotExist:
        throw Exception(begin)
            << "no method or data member named \"" << name
            << "\" found for type \""
            << type.name() << "\"";
        break;
      case reco::parser::kIsNotPublic:
        throw Exception(begin)
            << "data member named \"" << name << "\" for type \""
            << type.name() << "\" is not publically accessible.";
        break;
      default:
        throw Exception(begin)
            << "data member named \"" << name << "\" for type \""
            << type.name() << "\" is not usable in this context.";
        break;
    }
  }
  // Ok, it was a data member.
  typeStack_.push_back(member.typeOf());
  methStack_.push_back(MethodInvoker(member));
  return true;
}

