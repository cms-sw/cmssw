#ifndef FWCore_Utilities_InputTag_h
#define FWCore_Utilities_InputTag_h

#include <atomic>
#include <iosfwd>
#include <string>

#include "FWCore/Utilities/interface/EDGetToken.h"

namespace edm {

  class InputTag {
  public:
    InputTag();
    InputTag(std::string const& label, std::string const& instance, std::string const& processName = "");
    InputTag(char const* label, char const* instance, char const* processName = "");
    /// the input string is of the form:
    /// label
    /// label:instance
    /// label:instance:process
    InputTag(std::string const& s);
    ~InputTag();

    InputTag(InputTag const& other);

    InputTag(InputTag&& other);
    InputTag& operator=(InputTag const& other);

    InputTag& operator=(InputTag&& other);

    std::string encode() const;

    std::string const& label() const { return label_; }
    std::string const& instance() const { return instance_; }
    ///an empty string means find the most recently produced
    ///product with the label and instance
    std::string const& process() const { return process_; }

    bool willSkipCurrentProcess() const { return skipCurrentProcess_; }

    bool operator==(InputTag const& tag) const;

    EDGetToken cachedToken() const { return token_; }

    void cacheToken(EDGetToken) const;

    bool isUninitialized() const;

    static const std::string kSkipCurrentProcess;
    static const std::string kCurrentProcess;

  private:
    bool calcSkipCurrentProcess() const;

    std::string label_;
    std::string instance_;
    std::string process_;

    mutable std::atomic<EDGetToken> token_;

    bool skipCurrentProcess_;
  };

  std::ostream& operator<<(std::ostream& ost, InputTag const& tag);
}  // namespace edm
#endif
