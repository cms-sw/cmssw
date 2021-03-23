#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/Parse.h"
#include "FWCore/Utilities/interface/EDMException.h"

namespace edm {

  const std::string InputTag::kSkipCurrentProcess("@skipCurrentProcess");
  const std::string InputTag::kCurrentProcess("@currentProcess");
  static std::string const separator(":");

  InputTag::InputTag()
      : label_(),
        instance_(),
        process_(),
        typeID_(),
        productRegistry_(nullptr),
        index_(ProductResolverIndexInvalid),
        branchType_(NumBranchTypes),
        skipCurrentProcess_(false) {}

  InputTag::InputTag(std::string const& label, std::string const& instance, std::string const& processName)
      : label_(label),
        instance_(instance),
        process_(processName),
        typeID_(),
        productRegistry_(nullptr),
        index_(ProductResolverIndexInvalid),
        branchType_(NumBranchTypes),
        skipCurrentProcess_(calcSkipCurrentProcess()) {}

  InputTag::InputTag(char const* label, char const* instance, char const* processName)
      : label_(label),
        instance_(instance),
        process_(processName),
        typeID_(),
        productRegistry_(nullptr),
        index_(ProductResolverIndexInvalid),
        branchType_(NumBranchTypes),
        skipCurrentProcess_(calcSkipCurrentProcess()) {}

  InputTag::InputTag(std::string const& s)
      : label_(),
        instance_(),
        process_(),
        typeID_(),
        productRegistry_(nullptr),
        index_(ProductResolverIndexInvalid),
        branchType_(NumBranchTypes),
        skipCurrentProcess_(false) {
    // string is delimited by colons
    std::vector<std::string> tokens = tokenize(s, separator);
    size_t nwords = tokens.size();
    if (nwords > 3) {
      throw edm::Exception(errors::Configuration, "InputTag") << "Input tag " << s << " has " << nwords << " tokens";
    }
    if (nwords > 0)
      label_ = tokens[0];
    if (nwords > 1)
      instance_ = tokens[1];
    if (nwords > 2)
      process_ = tokens[2];
    skipCurrentProcess_ = calcSkipCurrentProcess();
  }

  InputTag::~InputTag() {}

  InputTag::InputTag(InputTag const& other)
      : label_(other.label()),
        instance_(other.instance()),
        process_(other.process()),
        typeID_(),
        productRegistry_(nullptr),
        index_(ProductResolverIndexInvalid),
        branchType_(NumBranchTypes),
        skipCurrentProcess_(other.willSkipCurrentProcess()) {
    ProductResolverIndex otherIndex = other.index_.load();
    if (otherIndex < ProductResolverIndexInitializing) {
      branchType_ = other.branchType_;
      typeID_ = other.typeID_;
      productRegistry_ = other.productRegistry_;
      index_.store(otherIndex);
    }
  }

  InputTag::InputTag(InputTag&& other)
      : label_(std::move(other.label_)),
        instance_(std::move(other.instance_)),
        process_(std::move(other.process_)),
        typeID_(),
        productRegistry_(nullptr),
        index_(ProductResolverIndexInvalid),
        branchType_(NumBranchTypes),
        skipCurrentProcess_(other.willSkipCurrentProcess()) {
    ProductResolverIndex otherIndex = other.index_.load();
    if (otherIndex < ProductResolverIndexInitializing) {
      branchType_ = other.branchType_;
      typeID_ = other.typeID_;
      productRegistry_ = other.productRegistry_;
      index_.store(otherIndex);
    }
  }

  InputTag& InputTag::operator=(InputTag const& other) {
    if (this != &other) {
      label_ = other.label_;
      instance_ = other.instance_;
      process_ = other.process_;
      skipCurrentProcess_ = other.skipCurrentProcess_;

      ProductResolverIndex otherIndex = other.index_.load();
      if (otherIndex < ProductResolverIndexInitializing) {
        branchType_ = other.branchType_;
        typeID_ = other.typeID_;
        productRegistry_ = other.productRegistry_;
        index_.store(otherIndex);
      } else {
        branchType_ = NumBranchTypes;
        typeID_ = TypeID();
        productRegistry_ = nullptr;
        index_.store(ProductResolverIndexInvalid);
      }
    }
    return *this;
  }

  InputTag& InputTag::operator=(InputTag&& other) {
    if (this != &other) {
      label_ = std::move(other.label_);
      instance_ = std::move(other.instance_);
      process_ = std::move(other.process_);
      skipCurrentProcess_ = other.skipCurrentProcess_;

      ProductResolverIndex otherIndex = other.index_.load();
      if (otherIndex < ProductResolverIndexInitializing) {
        branchType_ = other.branchType_;
        typeID_ = other.typeID_;
        productRegistry_ = other.productRegistry_;
        index_.store(otherIndex);
      } else {
        branchType_ = NumBranchTypes;
        typeID_ = TypeID();
        productRegistry_ = nullptr;
        index_.store(ProductResolverIndexInvalid);
      }
    }
    return *this;
  }

  bool InputTag::calcSkipCurrentProcess() const {
    char const* p1 = kSkipCurrentProcess.c_str();
    char const* p2 = process_.c_str();
    while (*p1 && (*p1 == *p2)) {
      ++p1;
      ++p2;
    }
    return *p1 == *p2;
  }

  std::string InputTag::encode() const {
    //NOTE: since the encoding gets used to form the configuration hash I did not want
    // to change it so that not specifying a process would cause two colons to appear in the
    // encoding and thus not being backwards compatible
    std::string result = label_;
    if (!instance_.empty() || !process_.empty()) {
      result += separator + instance_;
    }
    if (!process_.empty()) {
      result += separator + process_;
    }
    return result;
  }

  bool InputTag::operator==(InputTag const& tag) const {
    return (label_ == tag.label_) && (instance_ == tag.instance_) && (process_ == tag.process_);
  }

  ProductResolverIndex InputTag::indexFor(TypeID const& typeID,
                                          BranchType branchType,
                                          void const* productRegistry) const {
    ProductResolverIndex index = index_.load();

    if (index < ProductResolverIndexInitializing && typeID_ == typeID && branchType_ == branchType &&
        productRegistry_ == productRegistry) {
      return index;
    }
    return ProductResolverIndexInvalid;
  }

  void InputTag::tryToCacheIndex(ProductResolverIndex index,
                                 TypeID const& typeID,
                                 BranchType branchType,
                                 void const* productRegistry) const {
    unsigned int invalidValue = static_cast<unsigned int>(ProductResolverIndexInvalid);
    if (index_.compare_exchange_strong(invalidValue, static_cast<unsigned int>(ProductResolverIndexInitializing))) {
      typeID_ = typeID;
      branchType_ = branchType;
      productRegistry_ = productRegistry;
      index_.store(index);
    }
  }

  std::ostream& operator<<(std::ostream& ost, InputTag const& tag) {
    static std::string const process(", process = ");
    ost << "InputTag:  label = " << tag.label() << ", instance = " << tag.instance()
        << (tag.process().empty() ? std::string() : (process + tag.process()));
    return ost;
  }
}  // namespace edm
