#ifndef ParameterSet_InputTag_h
#define ParameterSet_InputTag_h

#include <string>
#include <iosfwd>

namespace edm {

  class InputTag
  {
  public:
    InputTag();
    InputTag(const std::string & label, const std::string & instance, const std::string & processName = "");
    /// the input string is of the form:
    /// label
    /// label:instance
    /// label:instance:process
    InputTag(const std::string & s);
    std::string encode() const;

    const std::string & label()    const {return label_;} 
    const std::string & instance() const {return instance_;}
    ///an empty string means find the most recently produced 
    ///product with the label and instance
    const std::string & process() const {return process_;} 
    
    bool operator==(const InputTag & tag) const;

  private:
    std::string label_;
    std::string instance_;
    std::string process_;
  };

  std::ostream& operator<<(std::ostream& ost, const InputTag & tag);

}


#endif

