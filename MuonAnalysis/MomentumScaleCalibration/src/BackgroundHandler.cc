#ifndef BackgroundHandler_cc
#define BackgroundHandler_cc

#include "MuonAnalysis/MomentumScaleCalibration/interface/BackgroundHandler.h"
#include <algorithm>
#include <TF1.h>
#include <iostream>
#include <boost/foreach.hpp>

using namespace std;
using namespace edm;

BackgroundHandler::BackgroundHandler( const vector<int> & identifiers,
                                      const vector<double> & leftWindowFactors,
                                      const vector<double> & rightWindowFactors,
                                      const double * ResMass,
                                      const double * massWindowHalfWidth ) :
  leftWindowFactors_(leftWindowFactors),
  rightWindowFactors_(rightWindowFactors)
{
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

  // Build the resonance windows
  for( unsigned int i=0; i<6; ++i ) {
    double mass = ResMass[i];
    resonanceWindow_.push_back( MassWindow( mass, mass - massWindowHalfWidth[i], mass + massWindowHalfWidth[i],
                                            vector<unsigned int>(1,i), backgroundFunctionService(identifiers[resToReg_[i]]) ) );
  }

  // Build the background windows
  // ----------------------------
  // Compute the mass center of each region
  double resMassForRegion[3];
  resMassForRegion[0] = ResMass[0];
  resMassForRegion[1] = (ResMass[1]+ResMass[2]+ResMass[3])/3;
  resMassForRegion[2] = (ResMass[4]+ResMass[5])/2;

  // Define which resonance is in which background window
  vector<vector<unsigned int> > indexes;
  indexes.push_back(vector<unsigned int>(1,0));
  indexes.push_back(vector<unsigned int>());
  for( int i=1; i<=3; ++i ) { indexes[1].push_back(i); }
  indexes.push_back(vector<unsigned int>());
  for( int i=4; i<=5; ++i ) { indexes[2].push_back(i); }

  unsigned int i=0;
  typedef vector<unsigned int> indexVec;
  BOOST_FOREACH(const indexVec & index, indexes) {
    backgroundWindow_.push_back( MassWindow( resMassForRegion[i],
        resMassForRegion[i] - massWindowHalfWidth[regToResHW_[i]]*leftWindowFactors[i],
        resMassForRegion[i] + massWindowHalfWidth[regToResHW_[i]]*rightWindowFactors[i],
        index, backgroundFunctionService(identifiers[i]) ) );
    ++i;
  }
  // Initialize the parNums to be used in the shifts of parval
  initializeParNums();
}

BackgroundHandler::~BackgroundHandler()
{
}

void BackgroundHandler::initializeParNums()
{
  // Initialize the parNums to be used in the shifts of parval
  parNumsRegions_[0] = 0;
  for( unsigned int i=1; i<backgroundWindow_.size() ; ++i ) {
    parNumsRegions_[i] = parNumsRegions_[i-1] + backgroundWindow_[i-1].backgroundFunction()->parNum();
  }
  parNumsResonances_[0] = parNumsRegions_[2]+backgroundWindow_[2].backgroundFunction()->parNum();
  for( unsigned int i=1; i<resonanceWindow_.size() ; ++i ) {
    parNumsResonances_[i] = parNumsResonances_[i-1] + resonanceWindow_[i-1].backgroundFunction()->parNum();
  }
}

void BackgroundHandler::setParameters(double* Start, double* Step, double* Mini, double* Maxi, int* ind, TString* parname, const vector<double> & parBgr, const vector<int> & parBgrOrder, const int muonType)
{
  vector<double>::const_iterator parBgrIt = parBgr.begin();
  vector<int>::const_iterator parBgrOrderIt = parBgrOrder.begin();
  // Set the parameters for the regions only if this is not a rescaling
  for( int iReg = 0; iReg < 3; ++iReg ) {
    int shift = parNumsRegions_[iReg];
    backgroundWindow_[iReg].backgroundFunction()->setParameters( &(Start[shift]), &(Step[shift]), &(Mini[shift]),
        &(Maxi[shift]), &(ind[shift]), &(parname[shift]),
        parBgrIt+shift, parBgrOrderIt+shift, muonType );
  }
  for( int iRes = 0; iRes < 6; ++iRes ) {
    // parNumsResonances is already shifted for the regions parameters
    int shift = parNumsResonances_[iRes];
    resonanceWindow_[iRes].backgroundFunction()->setParameters( &(Start[shift]), &(Step[shift]), &(Mini[shift]),
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
    return backgroundWindow_[resToReg_[ires]].mass();
  }
  else {
    // Not fitting the background: use the resonances
    return resonanceWindow_[ires].mass();
  }
}

void BackgroundHandler::rescale( vector<double> & parBgr, const double * ResMass, const double * massWindowHalfWidth,
                                 const vector<std::pair<reco::Particle::LorentzVector,reco::Particle::LorentzVector> > & muonPairs,
                                 const double & weight )
{
  countEventsInAllWindows(muonPairs, weight);

  // Loop on all regions and on all the resonances of each region and compute the background fraction
  // for each resonance window.
  unsigned int iRegion = 0;
  BOOST_FOREACH(MassWindow & backgroundWindow, backgroundWindow_)
  {
    TF1 * backgroundFunctionForIntegral = backgroundWindow.backgroundFunction()->functionForIntegral((parBgr.begin()+
                                                                                                      parNumsRegions_[iRegion]));
    // WARNING: this expects the background fraction parameter to be parBgr[0] for all the background functions.
    double eventsScaleFactor = backgroundFunctionForIntegral->GetParameter(0)*backgroundWindow.events();

    cout << "rescale: eventsScaleFactor = " << eventsScaleFactor << endl;

    // index is the index of the resonance in the background window
    BOOST_FOREACH( unsigned int index, *(backgroundWindow.indexes()) )
    {
      // First set all parameters of the resonance window background function to those of the corresponding region
      for( int iPar = 0; iPar < resonanceWindow_[index].backgroundFunction()->parNum(); ++iPar ) {
        parBgr[parNumsResonances_[index]+iPar] = parBgr[parNumsRegions_[resToReg_[index]]+iPar];
      }

      // Compute the integral of the background function in the resonance window.
      // This is the estimated number of background events in the resonance window.
      double backgroundIntegral = eventsScaleFactor*backgroundFunctionForIntegral->Integral(resonanceWindow_[index].lowerBound(),
          resonanceWindow_[index].upperBound());

      // Estimated fraction of events in the resonance window
      parBgr[parNumsResonances_[index]] = backgroundIntegral/(resonanceWindow_[index].events());
      cout << "rescale: backgroundIntegral = " << backgroundIntegral << endl;
      cout << "rescale: resonanceWindow_[index].events() = " << resonanceWindow_[index].events() << endl;
      cout << "rescale: parBgr["<<parNumsResonances_[index]<<"] = " << backgroundIntegral/(resonanceWindow_[index].events()) << endl;
    }
    ++iRegion;
    delete backgroundFunctionForIntegral;
  }
}

pair<double, double> BackgroundHandler::backgroundFunction( const bool doBackgroundFit,
                                                            const double * parval, const int resTotNum, const int ires,
                                                            const bool * resConsidered, const double * ResMass, const double ResHalfWidth[],
                                                            const int MuonType, const double & mass, const int nbins )
{
  if( doBackgroundFit ) {
    // Return the values for the region
    int iReg = resToReg_[ires];
    return make_pair( parval[parNumsRegions_[iReg]],
                      (*(backgroundWindow_[iReg].backgroundFunction()))( &(parval[parNumsRegions_[iReg]]), resTotNum, ires,
                                                                         resConsidered, ResMass, ResHalfWidth, MuonType, mass, nbins ) );
  }
  // Return the values for the resonance
  return make_pair( parval[parNumsResonances_[ires]],
                    (*(resonanceWindow_[ires].backgroundFunction()))( &(parval[parNumsResonances_[ires]]), resTotNum, ires,
                                                                      resConsidered, ResMass, ResHalfWidth, MuonType, mass, nbins ) );
}

void BackgroundHandler::countEventsInAllWindows(const vector<std::pair<reco::Particle::LorentzVector,reco::Particle::LorentzVector> > & muonPairs,
                                                const double & weight)
{
  std::pair<lorentzVector,lorentzVector> muonPair;
  BOOST_FOREACH(muonPair, muonPairs) {
    // Count events in resonance windows
    BOOST_FOREACH(MassWindow & resonanceWindow, resonanceWindow_) {
      resonanceWindow.count((muonPair.first + muonPair.second).mass(), weight);
    }
    // Count events in background windows
    BOOST_FOREACH(MassWindow & backgroundWindow, backgroundWindow_) {
      backgroundWindow.count((muonPair.first + muonPair.second).mass(), weight);
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
}

#endif // BackgroundHandler_cc
