#ifndef BackgroundHandler_cc
#define BackgroundHandler_cc

#include "MuonAnalysis/MomentumScaleCalibration/interface/BackgroundHandler.h"
#include <algorithm>
#include <TF1.h>
#include <iostream>
#include <boost/foreach.hpp>

typedef reco::Particle::LorentzVector lorentzVector;

BackgroundHandler::BackgroundHandler( const std::vector<int> & identifiers,
                                      const std::vector<double> & leftWindowBorders,
                                      const std::vector<double> & rightWindowBorders,
                                      const double * ResMass,
                                      const double * massWindowHalfWidth )
{
  // : leftWindowFactors_(leftWindowFactors),
  // rightWindowFactors_(rightWindowFactors)

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
  consistencyCheck(identifiers, leftWindowBorders, rightWindowBorders);

  // Build the resonance windows
  for( unsigned int i=0; i<6; ++i ) {
    double mass = ResMass[i];
    double lowerLimit = mass - massWindowHalfWidth[i];
    double upperLimit = mass + massWindowHalfWidth[i];
    resonanceWindow_.push_back( MassWindow( mass, lowerLimit, upperLimit,
                                            std::vector<unsigned int>(1,i),
                                            backgroundFunctionService(identifiers[resToReg_[i]],
                                                                      lowerLimit,
                                                                      upperLimit) ) );
  }

  // Build the background windows
  // ----------------------------
  // Compute the mass center of each region
  double resMassForRegion[3];
  resMassForRegion[0] = ResMass[0];
  resMassForRegion[1] = (ResMass[1]+ResMass[2]+ResMass[3])/3;
  resMassForRegion[2] = (ResMass[4]+ResMass[5])/2;

  // Define which resonance is in which background window
  std::vector<std::vector<unsigned int> > indexes;
  indexes.push_back(std::vector<unsigned int>(1,0));
  indexes.push_back(std::vector<unsigned int>());
  for( int i=1; i<=3; ++i ) { indexes[1].push_back(i); }
  indexes.push_back(std::vector<unsigned int>());
  for( int i=4; i<=5; ++i ) { indexes[2].push_back(i); }

  unsigned int i=0;
  typedef std::vector<unsigned int> indexVec;
  BOOST_FOREACH(const indexVec & index, indexes) {
    //     double lowerLimit = resMassForRegion[i] - massWindowHalfWidth[regToResHW_[i]]*leftWindowFactors[i];
    //     double upperLimit = resMassForRegion[i] + massWindowHalfWidth[regToResHW_[i]]*rightWindowFactors[i];
    //     backgroundWindow_.push_back( MassWindow( resMassForRegion[i], lowerLimit, upperLimit, index,
    //                                              backgroundFunctionService(identifiers[i], lowerLimit, upperLimit ) ) );
    backgroundWindow_.push_back( MassWindow( resMassForRegion[i], leftWindowBorders[i], rightWindowBorders[i], index,
                                             backgroundFunctionService(identifiers[i], leftWindowBorders[i], rightWindowBorders[i] ) ) );
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

void BackgroundHandler::setParameters(double* Start, double* Step, double* Mini, double* Maxi, int* ind, TString* parname, const std::vector<double> & parBgr, const std::vector<int> & parBgrOrder, const int muonType)
{
  std::vector<double>::const_iterator parBgrIt = parBgr.begin();
  std::vector<int>::const_iterator parBgrOrderIt = parBgrOrder.begin();
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

bool BackgroundHandler::unlockParameter(const std::vector<int> & resfind, const unsigned int ipar)
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

// std::pair<double, double> BackgroundHandler::windowFactors( const bool doBackgroundFit, const int ires )
// {
//   if( doBackgroundFit ) {
//     // Fitting the background: use the regions
//     return std::make_pair(leftWindowFactors_[resToReg_[ires]], rightWindowFactors_[resToReg_[ires]]);
//   }
//   else {
//     // Not fitting the background: use the resonances
//     return std::make_pair(1.,1.);
//   }
// }

std::pair<double, double> BackgroundHandler::windowBorders( const bool doBackgroundFit, const int ires )
{
  if( doBackgroundFit ) {
    // Fitting the background: use the regions
    return std::make_pair(backgroundWindow_[resToReg_[ires]].lowerBound(), backgroundWindow_[resToReg_[ires]].upperBound());
  }
  else {
    // Not fitting the background: use the resonances
    // return std::make_pair(1.,1.);
    return std::make_pair(resonanceWindow_[ires].lowerBound(), resonanceWindow_[ires].upperBound());
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

void BackgroundHandler::rescale( std::vector<double> & parBgr, const double * ResMass, const double * massWindowHalfWidth,
                                 const std::vector<std::pair<reco::Particle::LorentzVector,reco::Particle::LorentzVector> > & muonPairs,
                                 const double & weight )
{
  countEventsInAllWindows(muonPairs, weight);

  // Loop on all regions and on all the resonances of each region and compute the background fraction
  // for each resonance window.
  unsigned int iRegion = 0;
  BOOST_FOREACH(MassWindow & backgroundWindow, backgroundWindow_)
  {
    // Iterator pointing to the first parameter of this background function in the full set of parameters
    std::vector<double>::const_iterator parBgrIt = (parBgr.begin()+parNumsRegions_[iRegion]);
    TF1 * backgroundFunctionForIntegral = backgroundWindow.backgroundFunction()->functionForIntegral(parBgrIt);
    // WARNING: this expects the background fraction parameter to be parBgr[0] for all the background functions.
    double kOld = *parBgrIt;
    double Nbw = backgroundWindow.events();
    double Ibw = backgroundFunctionForIntegral->Integral(backgroundWindow.lowerBound(),
                                                         backgroundWindow.upperBound());

    // index is the index of the resonance in the background window
    BOOST_FOREACH( unsigned int index, *(backgroundWindow.indexes()) )
    {
      // First set all parameters of the resonance window background function to those of the corresponding region
      for( int iPar = 0; iPar < resonanceWindow_[index].backgroundFunction()->parNum(); ++iPar ) {
        parBgr[parNumsResonances_[index]+iPar] = parBgr[parNumsRegions_[resToReg_[index]]+iPar];
      }
      // Estimated fraction of events in the resonance window
      double Irw = backgroundFunctionForIntegral->Integral(resonanceWindow_[index].lowerBound(),
                                                           resonanceWindow_[index].upperBound());
      double Nrw = resonanceWindow_[index].events();

      // Ibw is 1 (to avoid effects from approximation errors we set it to 1 and do not include it in the computation).
      if( Nrw != 0 ) parBgr[parNumsResonances_[index]] = kOld*Nbw/Nrw*Irw;
      else parBgr[parNumsResonances_[index]] = 0.;

      // Protect against fluctuations of number of events which could cause the fraction to go above 1.
      if( parBgr[parNumsResonances_[index]] > 1. ) parBgr[parNumsResonances_[index]] = 1.;

      double kNew = parBgr[parNumsResonances_[index]];
      std::cout << "For resonance = " << index << std::endl;
      std::cout << "backgroundWindow.lowerBound = " << backgroundWindow.lowerBound() << std::endl;
      std::cout << "backgroundWindow.upperBound = " << backgroundWindow.upperBound() << std::endl;
      std::cout << "parNumsResonances_["<<index<<"] = " << parNumsResonances_[index] << std::endl;
      std::cout << "Nbw = " << Nbw << ", Ibw = " << Ibw << std::endl;
      std::cout << "Nrw = " << Nrw << ", Irw = " << Irw << std::endl;
      std::cout << "k = " << kOld << ", k' = " << parBgr[parNumsResonances_[index]] << std::endl;
      std::cout << "background fraction in background window = Nbw*k = " << Nbw*kOld << std::endl;
      std::cout << "background fraction in resonance window = Nrw*k' = " << Nrw*kNew << std::endl;
    }
    ++iRegion;
    delete backgroundFunctionForIntegral;
  }
}

std::pair<double, double> BackgroundHandler::backgroundFunction( const bool doBackgroundFit,
								 const double * parval, const int resTotNum, const int ires,
								 const bool * resConsidered, const double * ResMass, const double ResHalfWidth[],
								 const int MuonType, const double & mass,
								 const double & eta1, const double & eta2 )
{
  if( doBackgroundFit ) {
    // Return the values for the region
    int iReg = resToReg_[ires];
    // return std::make_pair( parval[parNumsRegions_[iReg]] * backgroundWindow_[iReg].backgroundFunction()->fracVsEta(&(parval[parNumsRegions_[iReg]]), resEta),
    return std::make_pair( parval[parNumsRegions_[iReg]] * backgroundWindow_[iReg].backgroundFunction()->fracVsEta(&(parval[parNumsRegions_[iReg]]), eta1, eta2 ),
    			   (*(backgroundWindow_[iReg].backgroundFunction()))( &(parval[parNumsRegions_[iReg]]), mass, eta1, eta2 ) );
    // return std::make_pair( backgroundWindow_[iReg].backgroundFunction()->fracVsEta(&(parval[parNumsRegions_[iReg]]), eta1, eta2),
    // 			   (*(backgroundWindow_[iReg].backgroundFunction()))( &(parval[parNumsRegions_[iReg]]), mass, eta1, eta2 ) );
  }
  // Return the values for the resonance
  // return std::make_pair( parval[parNumsResonances_[ires]] * resonanceWindow_[ires].backgroundFunction()->fracVsEta(&(parval[parNumsResonances_[ires]]), resEta),
  // 			 (*(resonanceWindow_[ires].backgroundFunction()))( &(parval[parNumsResonances_[ires]]), mass, resEta ) );
  return std::make_pair( parval[parNumsResonances_[ires]] * resonanceWindow_[ires].backgroundFunction()->fracVsEta(&(parval[parNumsResonances_[ires]]), eta1, eta2),
			 (*(resonanceWindow_[ires].backgroundFunction()))( &(parval[parNumsResonances_[ires]]), mass, eta1, eta2 ) );
}

void BackgroundHandler::countEventsInAllWindows(const std::vector<std::pair<reco::Particle::LorentzVector,reco::Particle::LorentzVector> > & muonPairs,
                                                const double & weight)
{
  // First reset all the counters
  BOOST_FOREACH(MassWindow & resonanceWindow, resonanceWindow_) {
    resonanceWindow.resetCounter();
  }
  // Count events in background windows
  BOOST_FOREACH(MassWindow & backgroundWindow, backgroundWindow_) {
    backgroundWindow.resetCounter();
  }

  // Now count the events in each window
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

void BackgroundHandler::consistencyCheck(const std::vector<int> & identifiers,
                                         const std::vector<double> & leftWindowBorders,
                                         const std::vector<double> & rightWindowBorders) const throw(cms::Exception)
{
  if( leftWindowBorders.size() != rightWindowBorders.size() ) {
    throw cms::Exception("Configuration") << "BackgroundHandler::BackgroundHandler: leftWindowBorders.size() = " << leftWindowBorders.size()
                                          << " != rightWindowBorders.size() = " << rightWindowBorders.size() << std::endl;
  }
  if( leftWindowBorders.size() != 3 ) {
    throw cms::Exception("Configuration") << "BackgroundHandler::BackgroundHandler: leftWindowBorders.size() = rightWindowBorders.size() = "
                                          << leftWindowBorders.size() << " != 3" << std::endl;
  }
  if( identifiers.size() != 3 ) {
    throw cms::Exception("Configuration") << "BackgroundHandler::BackgroundHandler: identifiers must match the number of regions = 3" << std::endl;
  }
}

#endif // BackgroundHandler_cc
