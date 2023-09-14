
#include "FWCore/ParameterSet/interface/ParameterDescriptionNode.h"
#include "FWCore/ParameterSet/interface/ParameterDescriptionCases.h"
#include "FWCore/ParameterSet/src/ANDGroupDescription.h"
#include "FWCore/ParameterSet/src/ORGroupDescription.h"
#include "FWCore/ParameterSet/src/XORGroupDescription.h"
#include "FWCore/ParameterSet/interface/DocFormatHelper.h"

#include <vector>
#include <cassert>
#include <ostream>

#define TYPE_TO_ENUM(type, e_val)                      \
  template <>                                          \
  ParameterTypes ParameterTypeToEnum::toEnum<type>() { \
    return e_val;                                      \
  }
#define TYPE_TO_NAME(type) \
  case k_##type:           \
    return #type
#define TYPE_TO_NAME2(e_val, type) \
  case e_val:                      \
    return #type

namespace edm {

  class EventID;
  class LuminosityBlockID;
  class LuminosityBlockRange;
  class EventRange;
  class InputTag;
  class ESInputTag;
  class FileInPath;

  TYPE_TO_ENUM(int, k_int32)
  TYPE_TO_ENUM(std::vector<int>, k_vint32)
  TYPE_TO_ENUM(unsigned, k_uint32)
  TYPE_TO_ENUM(std::vector<unsigned>, k_vuint32)
  TYPE_TO_ENUM(long long, k_int64)
  TYPE_TO_ENUM(std::vector<long long>, k_vint64)
  TYPE_TO_ENUM(unsigned long long, k_uint64)
  TYPE_TO_ENUM(std::vector<unsigned long long>, k_vuint64)
  TYPE_TO_ENUM(double, k_double)
  TYPE_TO_ENUM(std::vector<double>, k_vdouble)
  TYPE_TO_ENUM(bool, k_bool)
  TYPE_TO_ENUM(std::string, k_stringRaw)
  TYPE_TO_ENUM(std::vector<std::string>, k_vstringRaw)
  TYPE_TO_ENUM(EventID, k_EventID)
  TYPE_TO_ENUM(std::vector<EventID>, k_VEventID)
  TYPE_TO_ENUM(LuminosityBlockID, k_LuminosityBlockID)
  TYPE_TO_ENUM(std::vector<LuminosityBlockID>, k_VLuminosityBlockID)
  TYPE_TO_ENUM(InputTag, k_InputTag)
  TYPE_TO_ENUM(std::vector<InputTag>, k_VInputTag)
  TYPE_TO_ENUM(ESInputTag, k_ESInputTag)
  TYPE_TO_ENUM(std::vector<ESInputTag>, k_VESInputTag)
  TYPE_TO_ENUM(FileInPath, k_FileInPath)
  TYPE_TO_ENUM(LuminosityBlockRange, k_LuminosityBlockRange)
  TYPE_TO_ENUM(std::vector<LuminosityBlockRange>, k_VLuminosityBlockRange)
  TYPE_TO_ENUM(EventRange, k_EventRange)
  TYPE_TO_ENUM(std::vector<EventRange>, k_VEventRange)
  // These are intentionally not implemented to prevent one
  // from calling add<ParameterSet>.  One should call
  // add<ParameterSetDescription> instead.
  // TYPE_TO_ENUM(ParameterSet,k_PSet)
  // TYPE_TO_ENUM(std::vector<ParameterSet>,k_VPSet)

  std::string parameterTypeEnumToString(ParameterTypes iType) {
    switch (iType) {
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
      TYPE_TO_NAME2(k_stringRaw, string);
      TYPE_TO_NAME2(k_vstringRaw, vstring);
      TYPE_TO_NAME(EventID);
      TYPE_TO_NAME(VEventID);
      TYPE_TO_NAME(LuminosityBlockID);
      TYPE_TO_NAME(VLuminosityBlockID);
      TYPE_TO_NAME(InputTag);
      TYPE_TO_NAME(VInputTag);
      TYPE_TO_NAME(ESInputTag);
      TYPE_TO_NAME(VESInputTag);
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

  Comment::Comment() {}
  Comment::Comment(std::string const& iComment) : comment_(iComment) {}
  Comment::Comment(char const* iComment) : comment_(iComment) {}

  ParameterDescriptionNode::~ParameterDescriptionNode() {}

  void ParameterDescriptionNode::setComment(std::string const& value) { comment_ = value; }

  void ParameterDescriptionNode::setComment(char const* value) { comment_ = value; }

  void ParameterDescriptionNode::print(std::ostream& os, bool optional, bool writeToCfi, DocFormatHelper& dfh) const {
    if (hasNestedContent()) {
      dfh.incrementCounter();
    }
    print_(os, optional, writeToCfi, dfh);
  }

  void ParameterDescriptionNode::printNestedContent(std::ostream& os, bool optional, DocFormatHelper& dfh) const {
    if (hasNestedContent()) {
      dfh.incrementCounter();
      printNestedContent_(os, optional, dfh);
    }
  }

  void ParameterDescriptionNode::printSpaces(std::ostream& os, int n) {
    char oldFill = os.fill(' ');
    os.width(n);
    os << "";
    os.fill(oldFill);
  }

  // operator>> ---------------------------------------------

  std::unique_ptr<ParameterDescriptionCases<bool>> operator>>(bool caseValue, ParameterDescriptionNode const& node) {
    std::unique_ptr<ParameterDescriptionNode> clonedNode(node.clone());
    return caseValue >> std::move(clonedNode);
  }

  std::unique_ptr<ParameterDescriptionCases<int>> operator>>(int caseValue, ParameterDescriptionNode const& node) {
    std::unique_ptr<ParameterDescriptionNode> clonedNode(node.clone());
    return caseValue >> std::move(clonedNode);
  }

  std::unique_ptr<ParameterDescriptionCases<std::string>> operator>>(std::string const& caseValue,
                                                                     ParameterDescriptionNode const& node) {
    std::unique_ptr<ParameterDescriptionNode> clonedNode(node.clone());
    return caseValue >> std::move(clonedNode);
  }

  std::unique_ptr<ParameterDescriptionCases<std::string>> operator>>(char const* caseValue,
                                                                     ParameterDescriptionNode const& node) {
    std::unique_ptr<ParameterDescriptionNode> clonedNode(node.clone());
    return caseValue >> std::move(clonedNode);
  }

  std::unique_ptr<ParameterDescriptionCases<bool>> operator>>(bool caseValue,
                                                              std::unique_ptr<ParameterDescriptionNode> node) {
    return std::unique_ptr<ParameterDescriptionCases<bool>>(
        new ParameterDescriptionCases<bool>(caseValue, std::move(node)));
  }

  std::unique_ptr<ParameterDescriptionCases<int>> operator>>(int caseValue,
                                                             std::unique_ptr<ParameterDescriptionNode> node) {
    return std::unique_ptr<ParameterDescriptionCases<int>>(
        new ParameterDescriptionCases<int>(caseValue, std::move(node)));
  }

  std::unique_ptr<ParameterDescriptionCases<std::string>> operator>>(std::string const& caseValue,
                                                                     std::unique_ptr<ParameterDescriptionNode> node) {
    return std::unique_ptr<ParameterDescriptionCases<std::string>>(
        new ParameterDescriptionCases<std::string>(caseValue, std::move(node)));
  }

  std::unique_ptr<ParameterDescriptionCases<std::string>> operator>>(char const* caseValue,
                                                                     std::unique_ptr<ParameterDescriptionNode> node) {
    std::string caseValueString(caseValue);
    return std::unique_ptr<ParameterDescriptionCases<std::string>>(
        new ParameterDescriptionCases<std::string>(caseValue, std::move(node)));
  }

  // operator&& ---------------------------------------------

  std::unique_ptr<ParameterDescriptionNode> operator&&(ParameterDescriptionNode const& node_left,
                                                       ParameterDescriptionNode const& node_right) {
    return std::make_unique<ANDGroupDescription>(node_left, node_right);
  }

  std::unique_ptr<ParameterDescriptionNode> operator&&(std::unique_ptr<ParameterDescriptionNode> node_left,
                                                       ParameterDescriptionNode const& node_right) {
    return std::make_unique<ANDGroupDescription>(std::move(node_left), node_right);
  }

  std::unique_ptr<ParameterDescriptionNode> operator&&(ParameterDescriptionNode const& node_left,
                                                       std::unique_ptr<ParameterDescriptionNode> node_right) {
    return std::make_unique<ANDGroupDescription>(node_left, std::move(node_right));
  }

  std::unique_ptr<ParameterDescriptionNode> operator&&(std::unique_ptr<ParameterDescriptionNode> node_left,
                                                       std::unique_ptr<ParameterDescriptionNode> node_right) {
    return std::make_unique<ANDGroupDescription>(std::move(node_left), std::move(node_right));
  }

  // operator|| ---------------------------------------------

  std::unique_ptr<ParameterDescriptionNode> operator||(ParameterDescriptionNode const& node_left,
                                                       ParameterDescriptionNode const& node_right) {
    return std::make_unique<ORGroupDescription>(node_left, node_right);
  }

  std::unique_ptr<ParameterDescriptionNode> operator||(std::unique_ptr<ParameterDescriptionNode> node_left,
                                                       ParameterDescriptionNode const& node_right) {
    return std::make_unique<ORGroupDescription>(std::move(node_left), node_right);
  }

  std::unique_ptr<ParameterDescriptionNode> operator||(ParameterDescriptionNode const& node_left,
                                                       std::unique_ptr<ParameterDescriptionNode> node_right) {
    return std::make_unique<ORGroupDescription>(node_left, std::move(node_right));
  }

  std::unique_ptr<ParameterDescriptionNode> operator||(std::unique_ptr<ParameterDescriptionNode> node_left,
                                                       std::unique_ptr<ParameterDescriptionNode> node_right) {
    return std::make_unique<ORGroupDescription>(std::move(node_left), std::move(node_right));
  }

  // operator^  ---------------------------------------------

  std::unique_ptr<ParameterDescriptionNode> operator^(ParameterDescriptionNode const& node_left,
                                                      ParameterDescriptionNode const& node_right) {
    return std::make_unique<XORGroupDescription>(node_left, node_right);
  }

  std::unique_ptr<ParameterDescriptionNode> operator^(std::unique_ptr<ParameterDescriptionNode> node_left,
                                                      ParameterDescriptionNode const& node_right) {
    return std::make_unique<XORGroupDescription>(std::move(node_left), node_right);
  }

  std::unique_ptr<ParameterDescriptionNode> operator^(ParameterDescriptionNode const& node_left,
                                                      std::unique_ptr<ParameterDescriptionNode> node_right) {
    return std::make_unique<XORGroupDescription>(node_left, std::move(node_right));
  }

  std::unique_ptr<ParameterDescriptionNode> operator^(std::unique_ptr<ParameterDescriptionNode> node_left,
                                                      std::unique_ptr<ParameterDescriptionNode> node_right) {
    return std::make_unique<XORGroupDescription>(std::move(node_left), std::move(node_right));
  }
}  // namespace edm
