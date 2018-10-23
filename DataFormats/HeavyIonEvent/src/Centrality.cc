//
//

#include "DataFormats/HeavyIonEvent/interface/Centrality.h"

#include <iostream>
using namespace std;
using namespace reco;

Centrality::Centrality(double d, std::string label)
  : 
value_(d),
label_(label),
etHFhitSumPlus_(0),
etHFtowerSumPlus_(0),
etHFtowerSumECutPlus_(0),
etHFtruncatedPlus_(0),
etHFhitSumMinus_(0),
etHFtowerSumMinus_(0),
etHFtowerSumECutMinus_(0),
etHFtruncatedMinus_(0),
etEESumPlus_(0),
etEEtruncatedPlus_(0),
etEESumMinus_(0),
etEEtruncatedMinus_(0),
etEBSum_(0),
etEBtruncated_(0),
pixelMultiplicity_(0),
pixelMultiplicityPlus_(0),
pixelMultiplicityMinus_(0),
trackMultiplicity_(0),
zdcSumPlus_(0),
zdcSumMinus_(0),
etMidRapiditySum_(0),
ntracksPtCut_(0),
ntracksEtaCut_(0),
ntracksEtaPtCut_(0),
nPixelTracks_(0),
nPixelTracksPlus_(0),
nPixelTracksMinus_(0)
{
}


Centrality::~Centrality()
{
}

