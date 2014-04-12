#ifndef FWCore_Utilities_InputTag_h
#define FWCore_Utilities_InputTag_h

#ifndef __GCCXML__
#include <atomic>
#endif

#include <iosfwd>
#include <string>

#include "FWCore/Utilities/interface/TypeID.h"
#include "FWCore/Utilities/interface/BranchType.h"
#include "FWCore/Utilities/interface/ProductHolderIndex.h"

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

#ifndef __GCCXML__
    InputTag(InputTag&& other);
#endif
    InputTag& operator=(InputTag const& other);

#ifndef __GCCXML__
    InputTag& operator=(InputTag&& other);
#endif

    std::string encode() const;

    std::string const& label() const {return label_;} 
    std::string const& instance() const {return instance_;}
    ///an empty string means find the most recently produced 
    ///product with the label and instance
    std::string const& process() const {return process_;} 

    bool willSkipCurrentProcess() const { return skipCurrentProcess_; }
    
    bool operator==(InputTag const& tag) const;

    ProductHolderIndex indexFor(TypeID const& typeID, BranchType branchType, void const* productRegistry) const;

    void tryToCacheIndex(ProductHolderIndex index, TypeID const& typeID, BranchType branchType, void const* productRegistry) const;

    static const std::string kSkipCurrentProcess;

  private:

    bool calcSkipCurrentProcess() const;

    std::string label_;
    std::string instance_;
    std::string process_;

    mutable TypeID typeID_;
    mutable void const* productRegistry_;

#ifndef __GCCXML__
    mutable std::atomic<unsigned int> index_;
#else
    unsigned int index_;
#endif

    mutable char branchType_;

    bool skipCurrentProcess_;
  };

  std::ostream& operator<<(std::ostream& ost, InputTag const& tag);
}
#endif
