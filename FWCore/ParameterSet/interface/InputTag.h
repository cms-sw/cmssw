#ifndef ParameterSet_InputTag_h
#define ParameterSet_InputTag_h

#include <string>
#include <iosfwd>

namespace edm {

  class InputTag
  {
  public:
    InputTag();
    InputTag(const std::string & label, const std::string & instance);
    /// the input string is of the form:
    /// label
    /// label:instance
    InputTag(const std::string & s);
    std::string encode() const;

    std::string label()    const {return label_;} 
    std::string instance() const {return instance_;}

  private:
    std::string label_;
    std::string instance_;
  };

  std::ostream& operator<<(std::ostream& ost, const InputTag & tag);

}


#endif

