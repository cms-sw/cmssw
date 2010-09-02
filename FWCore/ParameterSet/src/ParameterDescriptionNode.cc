
#include "FWCore/ParameterSet/interface/ParameterDescriptionNode.h"
#include "FWCore/ParameterSet/interface/ParameterDescriptionCases.h"
#include "FWCore/ParameterSet/interface/ANDGroupDescription.h"
#include "FWCore/ParameterSet/interface/ORGroupDescription.h"
#include "FWCore/ParameterSet/interface/XORGroupDescription.h"
#include "FWCore/ParameterSet/interface/DocFormatHelper.h"

#include <vector>
#include <cassert>
#include <ostream>

#define TYPE_TO_ENUM(type,e_val) template<> ParameterTypes ParameterTypeToEnum::toEnum<type >(){ return e_val; }
#define TYPE_TO_NAME(type) case k_ ## type: return #type

namespace edm {

  class MinimalEventID;
  class LuminosityBlockID;
  class LuminosityBlockRange;
  class EventRange;
  class InputTag;
  class FileInPath;

  TYPE_TO_ENUM(int,k_int32)
  TYPE_TO_ENUM(std::vector<int>,k_vint32)
  TYPE_TO_ENUM(unsigned,k_uint32)
  TYPE_TO_ENUM(std::vector<unsigned>,k_vuint32)
  TYPE_TO_ENUM(long long,k_int64)
  TYPE_TO_ENUM(std::vector<long long>,k_vint64)
  TYPE_TO_ENUM(unsigned long long,k_uint64)
  TYPE_TO_ENUM(std::vector<unsigned long long>,k_vuint64)
  TYPE_TO_ENUM(double,k_double)
  TYPE_TO_ENUM(std::vector<double>,k_vdouble)
  TYPE_TO_ENUM(bool,k_bool)
  TYPE_TO_ENUM(std::string,k_string)
  TYPE_TO_ENUM(std::vector<std::string>,k_vstring)
  TYPE_TO_ENUM(edm::MinimalEventID,k_EventID)
  TYPE_TO_ENUM(std::vector<edm::MinimalEventID>,k_VEventID)
  TYPE_TO_ENUM(edm::LuminosityBlockID,k_LuminosityBlockID)
  TYPE_TO_ENUM(std::vector<edm::LuminosityBlockID>,k_VLuminosityBlockID)
  TYPE_TO_ENUM(edm::InputTag,k_InputTag)
  TYPE_TO_ENUM(std::vector<edm::InputTag>,k_VInputTag)
  TYPE_TO_ENUM(edm::FileInPath,k_FileInPath)
  TYPE_TO_ENUM(edm::LuminosityBlockRange,k_LuminosityBlockRange)
  TYPE_TO_ENUM(std::vector<edm::LuminosityBlockRange>,k_VLuminosityBlockRange)
  TYPE_TO_ENUM(edm::EventRange,k_EventRange)
  TYPE_TO_ENUM(std::vector<edm::EventRange>,k_VEventRange)
  // These are intentionally not implemented to prevent one
  // from calling add<ParameterSet>.  One should call
  // add<ParameterSetDescription> instead.
  // TYPE_TO_ENUM(edm::ParameterSet,k_PSet)
  // TYPE_TO_ENUM(std::vector<edm::ParameterSet>,k_VPSet)

  std::string parameterTypeEnumToString(ParameterTypes iType) {
    switch(iType) {
      TYPE_TO_NAME(int32);
      TYPE_TO_NAME(vint32);
      TYPE_TO_NAME(uint32);
      TYPE_TO_NAME(vuint32);
      TYPE_TO_NAME(int64);
      TYPE_TO_NAME(vint64);
      TYPE_TO_NAME(uint64);
      TYPE_TO_NAME(vuint64);
      TYPE_TO_NAME(double);
      TYPE_TO_NAME(vdouble);
      TYPE_TO_NAME(bool);
      TYPE_TO_NAME(string);
      TYPE_TO_NAME(vstring);
      TYPE_TO_NAME(EventID);
      TYPE_TO_NAME(VEventID);
      TYPE_TO_NAME(LuminosityBlockID);
      TYPE_TO_NAME(VLuminosityBlockID);
      TYPE_TO_NAME(InputTag);
      TYPE_TO_NAME(VInputTag);
      TYPE_TO_NAME(FileInPath);
      TYPE_TO_NAME(PSet);
      TYPE_TO_NAME(VPSet);
      TYPE_TO_NAME(LuminosityBlockRange);
      TYPE_TO_NAME(VLuminosityBlockRange);
      TYPE_TO_NAME(EventRange);
      TYPE_TO_NAME(VEventRange);
    default:
      assert(false);
    }
    return "";
  }

  ParameterDescriptionNode::~ParameterDescriptionNode() {
  }

  void
  ParameterDescriptionNode::setComment(std::string const & value)
  { comment_ = value; }

  void
  ParameterDescriptionNode::setComment(char const* value)
  { comment_ = value; }

  void
  ParameterDescriptionNode::print(std::ostream & os,
                                  bool optional,
                                  bool writeToCfi,
                                  DocFormatHelper & dfh) {
    if (hasNestedContent()) {
      dfh.incrementCounter();
    }
    print_(os, optional, writeToCfi, dfh);
  }

  void
  ParameterDescriptionNode::printNestedContent(std::ostream & os,
                                               bool optional,
                                               DocFormatHelper & dfh) {
    if (hasNestedContent()) {
      dfh.incrementCounter();
      printNestedContent_(os, optional, dfh);
    }
  }

  void
  ParameterDescriptionNode::printSpaces(std::ostream & os, int n) {
    char oldFill = os.fill(' ');
    os.width(n);
    os << "";
    os.fill(oldFill);
  }

  // operator>> ---------------------------------------------

  std::auto_ptr<ParameterDescriptionCases<bool> >
  operator>>(bool caseValue,
             ParameterDescriptionNode const& node) {
    std::auto_ptr<ParameterDescriptionNode> clonedNode(node.clone());
    return caseValue >> clonedNode;
  }

  std::auto_ptr<ParameterDescriptionCases<int> >
  operator>>(int caseValue,
             ParameterDescriptionNode const& node) {
    std::auto_ptr<ParameterDescriptionNode> clonedNode(node.clone());
    return caseValue >> clonedNode;
  }

  std::auto_ptr<ParameterDescriptionCases<std::string> >
  operator>>(std::string const& caseValue,
             ParameterDescriptionNode const& node) {
    std::auto_ptr<ParameterDescriptionNode> clonedNode(node.clone());
    return caseValue >> clonedNode;
  }

  std::auto_ptr<ParameterDescriptionCases<std::string> >
  operator>>(char const* caseValue,
             ParameterDescriptionNode const& node) {
    std::auto_ptr<ParameterDescriptionNode> clonedNode(node.clone());
    return caseValue >> clonedNode;
  }

  std::auto_ptr<ParameterDescriptionCases<bool> >
  operator>>(bool caseValue,
             std::auto_ptr<ParameterDescriptionNode> node) {
    return std::auto_ptr<ParameterDescriptionCases<bool> >(
      new ParameterDescriptionCases<bool>(caseValue, node));
  }

  std::auto_ptr<ParameterDescriptionCases<int> >
  operator>>(int caseValue,
             std::auto_ptr<ParameterDescriptionNode> node) {
    return std::auto_ptr<ParameterDescriptionCases<int> >(
      new ParameterDescriptionCases<int>(caseValue, node));
  }

  std::auto_ptr<ParameterDescriptionCases<std::string> >
  operator>>(std::string const& caseValue,
             std::auto_ptr<ParameterDescriptionNode> node) {
    return std::auto_ptr<ParameterDescriptionCases<std::string> >(
      new ParameterDescriptionCases<std::string>(caseValue, node));
  }

  std::auto_ptr<ParameterDescriptionCases<std::string> >
  operator>>(char const* caseValue,
             std::auto_ptr<ParameterDescriptionNode> node) {
    std::string caseValueString(caseValue);
    return std::auto_ptr<ParameterDescriptionCases<std::string> >(
      new ParameterDescriptionCases<std::string>(caseValue, node));
  }

  // operator&& ---------------------------------------------

  std::auto_ptr<ParameterDescriptionNode>
  operator&&(ParameterDescriptionNode const& node_left,
             ParameterDescriptionNode const& node_right) {
    return std::auto_ptr<ParameterDescriptionNode>(new ANDGroupDescription(node_left, node_right));
  }

  std::auto_ptr<ParameterDescriptionNode>
  operator&&(std::auto_ptr<ParameterDescriptionNode> node_left,
             ParameterDescriptionNode const& node_right) {
    return std::auto_ptr<ParameterDescriptionNode>(new ANDGroupDescription(node_left, node_right));
  }

  std::auto_ptr<ParameterDescriptionNode>
  operator&&(ParameterDescriptionNode const& node_left,
             std::auto_ptr<ParameterDescriptionNode> node_right) {
    return std::auto_ptr<ParameterDescriptionNode>(new ANDGroupDescription(node_left, node_right));
  }

  std::auto_ptr<ParameterDescriptionNode>
  operator&&(std::auto_ptr<ParameterDescriptionNode> node_left,
             std::auto_ptr<ParameterDescriptionNode> node_right) {
    return std::auto_ptr<ParameterDescriptionNode>(new ANDGroupDescription(node_left, node_right));
  }

  // operator|| ---------------------------------------------

  std::auto_ptr<ParameterDescriptionNode>
  operator||(ParameterDescriptionNode const& node_left,
             ParameterDescriptionNode const& node_right) {
    return std::auto_ptr<ParameterDescriptionNode>(new ORGroupDescription(node_left, node_right));
  }

  std::auto_ptr<ParameterDescriptionNode>
  operator||(std::auto_ptr<ParameterDescriptionNode> node_left,
             ParameterDescriptionNode const& node_right) {
    return std::auto_ptr<ParameterDescriptionNode>(new ORGroupDescription(node_left, node_right));
  }

  std::auto_ptr<ParameterDescriptionNode>
  operator||(ParameterDescriptionNode const& node_left,
             std::auto_ptr<ParameterDescriptionNode> node_right) {
    return std::auto_ptr<ParameterDescriptionNode>(new ORGroupDescription(node_left, node_right));
  }

  std::auto_ptr<ParameterDescriptionNode>
  operator||(std::auto_ptr<ParameterDescriptionNode> node_left,
             std::auto_ptr<ParameterDescriptionNode> node_right) {
    return std::auto_ptr<ParameterDescriptionNode>(new ORGroupDescription(node_left, node_right));
  }

  // operator^  ---------------------------------------------

  std::auto_ptr<ParameterDescriptionNode>
  operator^(ParameterDescriptionNode const& node_left,
            ParameterDescriptionNode const& node_right) {
    return std::auto_ptr<ParameterDescriptionNode>(new XORGroupDescription(node_left, node_right));
  }

  std::auto_ptr<ParameterDescriptionNode>
  operator^(std::auto_ptr<ParameterDescriptionNode> node_left,
            ParameterDescriptionNode const& node_right) {
    return std::auto_ptr<ParameterDescriptionNode>(new XORGroupDescription(node_left, node_right));
  }

  std::auto_ptr<ParameterDescriptionNode>
  operator^(ParameterDescriptionNode const& node_left,
            std::auto_ptr<ParameterDescriptionNode> node_right) {
    return std::auto_ptr<ParameterDescriptionNode>(new XORGroupDescription(node_left, node_right));
  }

  std::auto_ptr<ParameterDescriptionNode>
  operator^(std::auto_ptr<ParameterDescriptionNode> node_left,
            std::auto_ptr<ParameterDescriptionNode> node_right) {
    return std::auto_ptr<ParameterDescriptionNode>(new XORGroupDescription(node_left, node_right));
  }
}
