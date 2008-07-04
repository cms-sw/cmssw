//
// $Id: Centrality.cc,v 1.1 2008/07/04 13:45:07 pyoungso Exp $
//

#include "DataFormats/HeavyIonEvent/interface/Centrality.h"

using namespace reco;

Centrality::Centrality(double eHF, double eCASTOR, double eZDC, int ZDCHits)
  : 
HFEnergy_(eHF),
CASTOREnergy_(eCASTOR),
ZDCEnergy_(eZDC),
ZDCHitCounts_(ZDCHits)
{
  // default constructor
}


Centrality::~Centrality()
{
}


