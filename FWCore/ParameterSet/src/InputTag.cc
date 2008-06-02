#include "FWCore/ParameterSet/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/parse.h"
#include "FWCore/Utilities/interface/EDMException.h"

namespace edm {

  InputTag::InputTag()
  : label_(""),
    instance_(""),
    process_()
  {
  }


  InputTag::InputTag(const std::string & label, const std::string & instance, const std::string & processName)
  : label_(label),
    instance_(instance),
    process_(processName)
  {
  }

  InputTag::InputTag(const std::string & s) 
  : label_(""),
    instance_(""),
    process_()
  {
    // string is delimited by colons
    std::vector<std::string> tokens = edm::pset::tokenize(s, ":");
    int nwords = tokens.size();
    if(nwords == 0 || nwords > 3)
    {
      throw edm::Exception(errors::Configuration,"InputTag")
        << "Input tag " << s << " has " << nwords << " tokens";
    }
    label_ = tokens[0];

    if(nwords > 1) instance_ = tokens[1];
    if(nwords > 2) process_=tokens[2];
  }


  std::string InputTag::encode() const
  {
    //NOTE: since the encoding gets used to form the configuration hash I did not want
    // to change it so that not specifying a process would cause two colons to appear in the
    // encoding and thus not being backwards compatible
    static const std::string separator(":");
    return label_ + separator + instance_+(process_.empty()?std::string():(separator+process_));
  }

  std::ostream& operator<<(std::ostream& ost, const edm::InputTag & tag)
{
    static const std::string process(", process = ");
    ost << "InputTag:  label = " << tag.label() << ", instance = " << tag.instance()
    <<(tag.process().empty()?std::string():(process+tag.process()));
    return ost;
}
}





