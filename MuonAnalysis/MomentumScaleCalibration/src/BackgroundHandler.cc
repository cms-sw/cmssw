#ifndef BackgroundHandler_cc
#define BackgroundHandler_cc

#include "MuonAnalysis/MomentumScaleCalibration/interface/BackgroundHandler.h"
#include <algorithm>
#include <TF1.h>
#include <iostream>

using namespace std;
using namespace edm;

BackgroundHandler::BackgroundHandler( const vector<int> & identifiers,
                                      const vector<double> & leftWindowFactors,
                                      const vector<double> & rightWindowFactors,
                                      const double * ResMass ) :
  regionWindowEvents_(vector<double>(3, 0)),
  resonanceWindowEvents_(vector<double>(6,0)),
  leftWindowFactors_(leftWindowFactors),
  rightWindowFactors_(rightWindowFactors)
{
  // Compute the mass center of each region
  resMassForRegion_[0] = ResMass[0];
  resMassForRegion_[1] = (ResMass[1]+ResMass[2]+ResMass[3])/3;
  resMassForRegion_[2] = (ResMass[4]+ResMass[5])/2;

  // Store them internally for simplicity
  resMassForResonance_[0] = ResMass[0];
  resMassForResonance_[1] = ResMass[1];
  resMassForResonance_[2] = ResMass[2];
  resMassForResonance_[3] = ResMass[3];
  resMassForResonance_[4] = ResMass[4];
  resMassForResonance_[5] = ResMass[5];

  // Define the correspondence between regions and halfWidth to use
  // Defines also the function type to use (but they are checked to be consistent over a region)
  regToResHW_[0] = 0; // Region 0 use the one from Z
  regToResHW_[1] = 3; // Region 1 use the one from Upsilon1S
  regToResHW_[2] = 5; // Region 2 use the one from J/Psi

  // Define the correspondence between resonances and regions
  resToReg_[0] = 0; // Z
  resToReg_[1] = 1; // Upsilon3S
  resToReg_[2] = 1; // Upsilon2S
  resToReg_[3] = 1; // Upsilon1S
  resToReg_[4] = 2; // Psi2S
  resToReg_[5] = 2; // J/Psi

  // Throws cms::Exception("Configuration") in case the parameters are not what is expected
  consistencyCheck(identifiers, leftWindowFactors, rightWindowFactors);

  // Fill the backgroundFunctions for the regions using the backgroundFunctionService
  backgroundFunctionsForRegions_.resize(identifiers.size());
  transform(identifiers.begin(), identifiers.end(), backgroundFunctionsForRegions_.begin(), backgroundFunctionService);
  // Fill the backgroundFunctions for the resonances
  for( int iRes = 0; iRes < 6; ++iRes ) {
    backgroundFunctionsForResonances_.push_back(backgroundFunctionService(identifiers[resToReg_[iRes]]));
  }

  // Initialize the parNums to be used in the shifts of parval
  for( int i=0; i<3; ++i ) {
    parNumsRegions_[i] = 0;
    // Accumulate the parNums up to i-1
    for( vector<backgroundFunctionBase*>::const_iterator it = backgroundFunctionsForRegions_.begin();
         it != backgroundFunctionsForRegions_.begin()+i; ++it ) {
      parNumsRegions_[i] += (*it)->parNum();
    }
  }
  for( int i=0; i<6; ++i ) {
    // Start from the end of the parameters for regions
    parNumsResonances_[i] = parNumsRegions_[2]+backgroundFunctionsForRegions_[2]->parNum();
    for( vector<backgroundFunctionBase*>::const_iterator it = backgroundFunctionsForResonances_.begin();
         it != backgroundFunctionsForResonances_.begin()+i; ++it ) {
      parNumsResonances_[i] += (*it)->parNum();
    }
  }
}

BackgroundHandler::~BackgroundHandler()
{
  // cout << "Clearing the BackgroundHandler" << endl;
  vector<backgroundFunctionBase*>::iterator it = backgroundFunctionsForRegions_.begin();
  for( ; it != backgroundFunctionsForRegions_.end(); ++it ) {
    delete (*it);
  }
  it = backgroundFunctionsForResonances_.begin();
  for( ; it != backgroundFunctionsForResonances_.end(); ++it ) {
    delete (*it);
  }
}

void BackgroundHandler::setParameters(double* Start, double* Step, double* Mini, double* Maxi, int* ind, TString* parname, const vector<double> & parBgr, const vector<int> & parBgrOrder, const int muonType)
{
  vector<double>::const_iterator parBgrIt = parBgr.begin();
  vector<int>::const_iterator parBgrOrderIt = parBgrOrder.begin();
  // Set the parameters for the regions only if this is not a rescaling
  for( int iReg = 0; iReg < 3; ++iReg ) {
    int shift = parNumsRegions_[iReg];
    backgroundFunctionsForRegions_[iReg]->setParameters( &(Start[shift]), &(Step[shift]), &(Mini[shift]),
                                                         &(Maxi[shift]), &(ind[shift]), &(parname[shift]),
                                                         parBgrIt+shift, parBgrOrderIt+shift, muonType );
  }
  for( int iRes = 0; iRes < 6; ++iRes ) {
    // parNumsResonances is already shifted for the regions parameters
    int shift = parNumsResonances_[iRes];
    backgroundFunctionsForResonances_[iRes]->setParameters( &(Start[shift]), &(Step[shift]), &(Mini[shift]),
                                                            &(Maxi[shift]), &(ind[shift]), &(parname[shift]),
                                                            parBgrIt+shift, parBgrOrderIt+shift, muonType );
  }
}

bool BackgroundHandler::unlockParameter(const vector<int> & resfind, const unsigned int ipar)
{
  // parNumsRegions_ are shifted: [1] contains the number of parameters for 0 and so on.
  if( ipar < unsigned(parNumsRegions_[1]) && resfind[0] > 0 ) {
    return true;
  }
  if( ipar >= unsigned(parNumsRegions_[1]) && ipar < unsigned(parNumsRegions_[2]) && ( resfind[1] > 0 || resfind[2] > 0 || resfind[3] > 0 ) ) {
    return true;
  }
  // The first of parNumsResonances_ has the sum of parNums of the regions.
  if( ipar >= unsigned(parNumsRegions_[2]) && ipar < unsigned(parNumsResonances_[0]) && ( resfind[4] > 0 || resfind[5] > 0 ) ) {
    return true;
  }
  return false;
}

pair<double, double> BackgroundHandler::windowFactors( const bool doBackgroundFit, const int ires )
{
  if( doBackgroundFit ) {
    // Fitting the background: use the regions
    return make_pair(leftWindowFactors_[resToReg_[ires]], rightWindowFactors_[resToReg_[ires]]);
  }
  else {
    // Not fitting the background: use the resonances
    return make_pair(1.,1.);
  }
}

double BackgroundHandler::resMass( const bool doBackgroundFit, const int ires )
{
  if( doBackgroundFit ) {
    // Fitting the background: use the regions
    return resMassForRegion_[resToReg_[ires]];
  }
  else {
    // Not fitting the background: use the resonances
    return resMassForResonance_[ires];
  }
}

void BackgroundHandler::rescale( vector<double> & parBgr, const double * ResMass, const double massWindowHalfWidth[][3], const int muonType,
                                 const vector<std::pair<reco::Particle::LorentzVector,reco::Particle::LorentzVector> > & muonPairs, const double & weight )
{
  // Reset the counters
  fill( resonanceWindowEvents_.begin(), resonanceWindowEvents_.end(), 0. );
  // Compute the number of muons in each resonance window
  static std::vector<std::pair<reco::Particle::LorentzVector,reco::Particle::LorentzVector> >::const_iterator it = muonPairs.begin();
  for( ; it != muonPairs.end(); ++it ) {
    int iRes = 0;
    for( vector<double>::iterator resIt = resonanceWindowEvents_.begin();
         resIt != resonanceWindowEvents_.end(); ++resIt, ++iRes ) {
      if( MuScleFitUtils::checkMassWindow( (it->first + it->second).mass(), iRes, resMassForResonance_[iRes] ) ) {
        *resIt += weight;
      }
    }
  }

  // Compute the integration intervals for the regions
  vector<double> leftRegionWidth;
  vector<double> rightRegionWidth;
  for( int iRegion = 0; iRegion < 3; ++iRegion ) {
    // cout << "For Region = " << iRegion << endl;
    // cout << "leftWindowFactors_["<<iRegion<<"] = " << leftWindowFactors_[iRegion] << endl;
    // cout << "rightWindowFactors_["<<iRegion<<"] = " << rightWindowFactors_[iRegion] << endl;
    // cout << "massWindowHalfWidth["<<regToResHW_[iRegion]<<"]["<<muonType<<"] = " << massWindowHalfWidth[regToResHW_[iRegion]][muonType] << endl;
    // cout << "leftRegionWidth = " << resMassForRegion_[iRegion] - leftWindowFactors_[iRegion]*massWindowHalfWidth[regToResHW_[iRegion]][muonType] << endl;
    // cout << "rightRegionWidth = " << resMassForRegion_[iRegion] + rightWindowFactors_[iRegion]*massWindowHalfWidth[regToResHW_[iRegion]][muonType] << endl;
    // M - leftFactor*HalfWidth
    leftRegionWidth.push_back(resMassForRegion_[iRegion] - leftWindowFactors_[iRegion]*massWindowHalfWidth[regToResHW_[iRegion]][muonType]);
    // M + rightFactor*HalfWidth
    rightRegionWidth.push_back(resMassForRegion_[iRegion] + rightWindowFactors_[iRegion]*massWindowHalfWidth[regToResHW_[iRegion]][muonType]);
  }

  // First set all parameters of the resonances as those of the corresponding region
  for( int iRes = 0; iRes < 6; ++iRes ) {
    // parNumsResonances is already shifted for the regions parameters
    for( int iPar = 0; iPar < backgroundFunctionsForResonances_[iRes]->parNum(); ++iPar ) {
      parBgr[parNumsResonances_[iRes]+iPar] = parBgr[parNumsRegions_[resToReg_[iRes]]+iPar];
    }
  }

  // Now compute the new background fractions and apply them for each resonance
  for( int iRes = 0; iRes < 6; ++iRes ) {
    int iRegion = resToReg_[iRes];
    // The parameters are:
    // - TF1 of the function for computing the integrals
    // - number of events in the background and resonance windows
    // - integration interval for the region
    // - integration interval for the resonance
    cout << "Apply rescale for resonance = " << iRes << endl;
    parBgr[parNumsResonances_[iRes]] = applyRescale( backgroundFunctionsForRegions_[iRegion]->functionForIntegral(parBgr.begin()+parNumsRegions_[iRegion]),
                                                     regionWindowEvents_[iRegion], resonanceWindowEvents_[iRes],
                                                     leftRegionWidth[iRegion], rightRegionWidth[iRegion],
                                                     ResMass[iRes] - massWindowHalfWidth[iRes][muonType], ResMass[iRes] + massWindowHalfWidth[iRes][muonType]
                                                     );
  }
}

double BackgroundHandler::applyRescale( TF1* backgroundFunctionForIntegral, const double backgroundWindowEvents, const double resonanceWindowEvents,
                                        const double & leftRegionWidth, const double & rightRegionWidth,
                                        const double & leftResonanceWidth, const double & rightResonanceWidth ) const
{
  if( backgroundWindowEvents == 0 ) {
    cout << "Error: backgroundWindowEvents_ = " << backgroundWindowEvents << endl;
  }

  // WARNING: this expects the background fraction parameter to be parBgr[0] for all the background functions.
  double backgroundFraction = backgroundFunctionForIntegral->GetParameter(0);

  if( resonanceWindowEvents != 0 ) {

    cout << "number of events in the background window = " << backgroundWindowEvents << endl;
    cout << "number of events in the resonance window = " << resonanceWindowEvents << endl;

    double backgroundWindowIntegral = backgroundFunctionForIntegral->Integral(leftRegionWidth, rightRegionWidth);
    double resonanceWindowIntegral = backgroundFunctionForIntegral->Integral(leftResonanceWidth, rightResonanceWidth);

    cout << "Integral background region = " << backgroundWindowIntegral << endl;
    cout << "Integral resonance region = " << resonanceWindowIntegral << endl;

    // We compute the scaling of the background fraction based on the following assumptions:
    // 1. The number of signal events does not change
    // 2. The background function is a good estimate of the background shape
    // Given Ntot, Ntot' and k = Nb'/Nb (the ratio of the two integrals) we get:
    // Nb = parBackground[0]*Ntot
    // Nb' = k*Nb
    // parBackground[0]' = Nb'/Ntot' = k*parBackground[0]*Ntot/Ntot'

    double k = resonanceWindowIntegral/backgroundWindowIntegral;
    double Nb = backgroundFraction*backgroundWindowEvents;
    // double Nbp = k*Nb;
    cout << "old backgroundFraction = " << backgroundFraction << endl;
    backgroundFraction = k*Nb/resonanceWindowEvents;
    cout << "new backgroundFraction = " << backgroundFraction << endl;
  }
  else {
    cout << "WARNING: resonanceWindowEvents = " << resonanceWindowEvents << ", not rescaling for this resonance" << endl;
  }
  return backgroundFraction;
}

pair<double, double> BackgroundHandler::backgroundFunction( const bool doBackgroundFit,
                                                            const double * parval, const int resTotNum, const int ires,
                                                            const bool * resConsidered, const double * ResMass, const double ResHalfWidth[],
                                                            const int MuonType, const double & mass, const int nbins )
{
  if( doBackgroundFit ) {
    // Return the values for the region
    int iReg = resToReg_[ires];
    // cout << "Returning value for region["<<iReg<<"]"
    //      << ", with parval["<<parNumsRegions_[iReg]<<"] = "<< parval[parNumsRegions_[iReg]]
    //      << ", with parval["<<parNumsRegions_[iReg]+1<<"] = "<< parval[parNumsRegions_[iReg]+1] << endl;
    // cout << "and: resTotNum = " << resTotNum << ", ires = " << ires << ", resConsidered[ires] = " << resConsidered[ires] << ", ResMass[ires] = " << ResMass[ires]
    //      << ", ResHalfWidth[" << ires << "] = " << ResHalfWidth[ires] << ", MuonType = " << MuonType << ", mass = " << mass << ", nbins = " << nbins << endl;
    return make_pair( parval[parNumsRegions_[iReg]],
                      (*backgroundFunctionsForRegions_[iReg])( &(parval[parNumsRegions_[iReg]]), resTotNum, ires,
                                                               resConsidered, ResMass, ResHalfWidth, MuonType, mass, nbins ) );
  }
  // Return the values for the resonance
  // cout << "Returning value for resonance["<<ires<<"]" << endl;
  return make_pair( parval[parNumsResonances_[ires]],
                    (*backgroundFunctionsForResonances_[ires])( &(parval[parNumsResonances_[ires]]), resTotNum, ires,
                                                                resConsidered, ResMass, ResHalfWidth, MuonType, mass, nbins ) );
}

void BackgroundHandler::countEventsInBackgroundWindows(const vector<std::pair<reco::Particle::LorentzVector,reco::Particle::LorentzVector> > & muonPairs,
                                                       const double & weight)
{
  // Reset the counters
  // cout << "Counting background events" << endl;
  fill( regionWindowEvents_.begin(), regionWindowEvents_.end(), 0. );
  // Loop on all the muon pairs
  int muonPairNum = 0;
  vector<std::pair<lorentzVector,lorentzVector> >::const_iterator it = muonPairs.begin();
  for( ; it != muonPairs.end(); ++it, ++muonPairNum ) {
    // cout << "muonPair = " << muonPairNum << endl;
    // Loop on all the regions
    int iReg = 0;
    vector<double>::iterator regIt = regionWindowEvents_.begin();
    for( ; regIt != regionWindowEvents_.end(); ++regIt, ++iReg ) {
      // cout << "region = " << iReg << endl;
      if( MuScleFitUtils::checkMassWindow( (it->first + it->second).mass(), regToResHW_[iReg], leftWindowFactors_[iReg], rightWindowFactors_[iReg] ) ) {
        *regIt += weight;
        // cout << "background event counted" << endl;
      }
    }
  }
}

void BackgroundHandler::consistencyCheck(const vector<int> & identifiers,
                                         const vector<double> & leftWindowFactors,
                                         const vector<double> & rightWindowFactors) const throw(cms::Exception)
{
  if( leftWindowFactors_.size() != rightWindowFactors_.size() ) {
    throw cms::Exception("Configuration") << "BackgroundHandler::BackgroundHandler: leftWindowFactors_.size() = " << leftWindowFactors_.size()
                                          << " != rightWindowFactors_.size() = " << rightWindowFactors_.size() << std::endl;
  }
  if( leftWindowFactors_.size() != 3 ) {
    throw cms::Exception("Configuration") << "BackgroundHandler::BackgroundHandler: leftWindowFactors_.size() = rightWindowFactors_.size() = "
                                          << leftWindowFactors_.size() << " != 3" << std::endl;
  }
  if( identifiers.size() != 3 ) {
    throw cms::Exception("Configuration") << "BackgroundHandler::BackgroundHandler: identifiers must match the number of regions = 3" << std::endl;
  }
//   if( !(identifiers[1] == identifiers[2] && identifiers[2] == identifiers[3]) ) {
//     throw cms::Exception("Configuration") << "BackgroundHandler::BackgroundHandler: different identifiers for the Upsilons:"
//                                           << identifiers[1] << ", " << identifiers[2] << ", " << identifiers[3] << std::endl;
//   }
//   if( !(identifiers[4] == identifiers[5]) ) {
//     throw cms::Exception("Configuration") << "BackgroundHandler::BackgroundHandler: different identifiers for the J/Psi and Psi2S:"
//                                           << identifiers[4] << ", " << identifiers[5] << std::endl;
//   }
}

#endif // BackgroundHandler_cc
