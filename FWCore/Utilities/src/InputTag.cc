#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/Parse.h"
#include "FWCore/Utilities/interface/EDMException.h"

namespace edm {

  InputTag::InputTag()
  : label_(""),
    instance_(""),
    process_(),
    typeID_(),
    cachedOffset_(0U),
    fillCount_(0) {
  }


  InputTag::InputTag(std::string const& label, std::string const& instance, std::string const& processName)
  : label_(label),
    instance_(instance),
    process_(processName),
    typeID_(),
    cachedOffset_(0U),
    fillCount_(0) {
  }

  InputTag::InputTag(char const* label, char const* instance, char const* processName)
  : label_(label),
    instance_(instance),
    process_(processName),
    typeID_(),
    cachedOffset_(0U),
    fillCount_(0) {
  }


  InputTag::InputTag(std::string const& s) 
  : label_(""),
    instance_(""),
    process_(),
    cachedOffset_(0U),
    fillCount_(0) {

    // string is delimited by colons
    std::vector<std::string> tokens = tokenize(s, ":");
    size_t nwords = tokens.size();
    if(nwords > 3) {
      throw edm::Exception(errors::Configuration,"InputTag")
        << "Input tag " << s << " has " << nwords << " tokens";
    }
    if(nwords > 0) label_ = tokens[0];
    if(nwords > 1) instance_ = tokens[1];
    if(nwords > 2) process_=tokens[2];
  }

  InputTag::~InputTag() {}

  bool InputTag::operator==(InputTag const& tag) const {
    return (label_ == tag.label_)  
        && (instance_ == tag.instance_)
        && (process_ == tag.process_);
  }


  std::string InputTag::encode() const {
    //NOTE: since the encoding gets used to form the configuration hash I did not want
    // to change it so that not specifying a process would cause two colons to appear in the
    // encoding and thus not being backwards compatible
    static std::string const separator(":");
    std::string result = label_;
    if(!instance_.empty() || !process_.empty()) {
      result += separator + instance_;
    }
    if(!process_.empty()) {
      result += separator + process_;
    }
    return result;
  }

  std::ostream& operator<<(std::ostream& ost, InputTag const& tag) {
    static std::string const process(", process = ");
    ost << "InputTag:  label = " << tag.label() << ", instance = " << tag.instance()
    << (tag.process().empty() ? std::string() : (process + tag.process()));
    return ost;
  }
}

