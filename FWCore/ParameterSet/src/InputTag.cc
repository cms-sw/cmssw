#include "FWCore/ParameterSet/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/parse.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include <iostream>

namespace edm {

  InputTag::InputTag()
  : label_(""),
    instance_("")
  {
  }


  InputTag::InputTag(const std::string & label, const std::string & instance)
  : label_(label),
    instance_(instance)
  {
  }

  InputTag::InputTag(const std::string & s) 
  : label_(""),
    instance_("")
  {
    // string is delimited by colons
    std::vector<std::string> tokens = edm::pset::tokenize(s, ":");
    int nwords = tokens.size();
    if(nwords == 0 || nwords > 2)
    {
      throw edm::Exception(errors::Configuration,"InputTag")
        << "Input tag " << s << " has " << nwords << " tokens";
    }
    label_ = tokens[0];

    if(nwords > 1) instance_ = tokens[1];
  }


  std::string InputTag::encode() const
  {
    return label_ + ":" + instance_;
  }
}


std::ostream& operator<<(std::ostream& ost, const edm::InputTag & tag)
{
  ost << "InputTag:  label = " << tag.label() << ", instance = " << tag.instance();
  return ost;
}



