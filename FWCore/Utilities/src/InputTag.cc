#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/Parse.h"
#include "FWCore/Utilities/interface/EDMException.h"

namespace edm {

  const std::string InputTag::kSkipCurrentProcess("@skipCurrentProcess");
  const std::string InputTag::kCurrentProcess("@currentProcess");
  static std::string const separator(":");

  InputTag::InputTag() : label_(), instance_(), process_(), token_(), skipCurrentProcess_(false) {}

  InputTag::InputTag(std::string const& label, std::string const& instance, std::string const& processName)
      : label_(label),
        instance_(instance),
        process_(processName),
        token_(),
        skipCurrentProcess_(calcSkipCurrentProcess()) {}

  InputTag::InputTag(char const* label, char const* instance, char const* processName)
      : label_(label),
        instance_(instance),
        process_(processName),
        token_(),
        skipCurrentProcess_(calcSkipCurrentProcess()) {}

  InputTag::InputTag(std::string const& s) : label_(), instance_(), process_(), token_(), skipCurrentProcess_(false) {
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
        token_(other.token_.load()),
        skipCurrentProcess_(other.willSkipCurrentProcess()) {}

  InputTag::InputTag(InputTag&& other)
      : label_(std::move(other.label_)),
        instance_(std::move(other.instance_)),
        process_(std::move(other.process_)),
        token_(other.token_.load()),
        skipCurrentProcess_(other.willSkipCurrentProcess()) {}

  InputTag& InputTag::operator=(InputTag const& other) {
    if (this != &other) {
      label_ = other.label_;
      instance_ = other.instance_;
      process_ = other.process_;
      skipCurrentProcess_ = other.skipCurrentProcess_;
      token_.store(other.token_.load());
    }
    return *this;
  }

  InputTag& InputTag::operator=(InputTag&& other) {
    if (this != &other) {
      label_ = std::move(other.label_);
      instance_ = std::move(other.instance_);
      process_ = std::move(other.process_);
      skipCurrentProcess_ = other.skipCurrentProcess_;
      token_.store(other.token_.load());
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

  void InputTag::cacheToken(EDGetToken token) const { token_.store(token); }

  bool InputTag::isUninitialized() const { return label_.empty(); }

  std::ostream& operator<<(std::ostream& ost, InputTag const& tag) {
    static std::string const process(", process = ");
    ost << "InputTag:  label = " << tag.label() << ", instance = " << tag.instance()
        << (tag.process().empty() ? std::string() : (process + tag.process()));
    return ost;
  }
}  // namespace edm
