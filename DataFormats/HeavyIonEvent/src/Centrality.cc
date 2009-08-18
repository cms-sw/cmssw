//
// $Id: Centrality.cc,v 1.2 2008/07/04 13:54:04 yilmaz Exp $
//

#include "DataFormats/HeavyIonEvent/interface/Centrality.h"

using namespace reco;

Centrality::Centrality(double d, std::string label)
  : 
value_(d),
label_(label)
{
  // default constructor
}


Centrality::~Centrality()
{
}


