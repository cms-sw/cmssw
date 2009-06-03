#ifndef FWCore_Utilities_InputTag_h
#define FWCore_Utilities_InputTag_h

#include <string>
#include <iosfwd>

#include "FWCore/Utilities/interface/TypeID.h"
#include "FWCore/Utilities/interface/BranchType.h"

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
    std::string encode() const;

    std::string const& label() const {return label_;} 
    std::string const& instance() const {return instance_;}
    ///an empty string means find the most recently produced 
    ///product with the label and instance
    std::string const& process() const {return process_;} 
    
    bool operator==(InputTag const& tag) const;

    TypeID& typeID() const {return typeID_;}

    BranchType& branchType() const {return branchType_;}

    size_t& cachedOffset() const {return cachedOffset_;}

    int& fillCount() const {return fillCount_;}

  private:
    std::string label_;
    std::string instance_;
    std::string process_;
    mutable BranchType branchType_;
    mutable TypeID typeID_;
    mutable size_t cachedOffset_;
    mutable int fillCount_;
  };

  std::ostream& operator<<(std::ostream& ost, InputTag const& tag);

}

#endif

