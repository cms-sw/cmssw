//
// $Id: Centrality.cc,v 1.3 2009/08/18 14:45:32 yilmaz Exp $
//

#include "DataFormats/HeavyIonEvent/interface/Centrality.h"

using namespace reco;

Centrality::Centrality(double d, std::string label)
  : 
value_(d),
label_(label),
etHFhitSumPlus_(0),
etHFtowerSumPlus_(0),
etHFtruncatedPlus_(0),
etHFhitSumMinus_(0),
etHFtowerSumMinus_(0),
etHFtruncatedMinus_(0),
etEESumPlus_(0),
etEEtruncatedPlus_(0),
etEESumMinus_(0),
etEEtruncatedMinus_(0),
etEBSum_(0),
etEBtruncated_(0),
pixelMultiplicity_(0),
zdcSumPlus_(0),
zdcSumMinus_(0)
{
}


Centrality::~Centrality()
{
}


