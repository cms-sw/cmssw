#include "FWCore/ParameterSet/interface/ProductTag.h"
#include "FWCore/ParameterSet/interface/parse.h"
#include "FWCore/Utilities/interface/EDMException.h"

namespace edm {

  ProductTag::ProductTag()
  : label_(""),
    instance_(""),
    alias_("")
  {
  }

  void ProductTag::decode(const std::string & s) 
  {
    // string is delimited by colons
    std::vector<std::string> tokens = edm::pset::tokenize(s, ":");
    int nwords = tokens.size();
    if(nwords == 0)
    {
      throw edm::Exception(errors::Configuration,"ProductTag")
        << "Empty Product Tag  ";
    }
    label_ = tokens[0];

    if(nwords > 1) instance_ = tokens[1];
    if(nwords > 2) alias_ = tokens[2];
  }


  std::string ProductTag::encode() const
  {
    return label_ + ":" + instance_ + ":" + alias_;
  }
}



