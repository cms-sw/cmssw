

#include "Alignment/LaserAlignment/src/LASProfileJudge.h"


// terminal colors
#define _R "\033[1;31m"
#define _B "\033[1;34m"
#define _G "\033[1;32m"
#define _N "\033[22;30m"


///
///
///
LASProfileJudge::LASProfileJudge() {

  // switch on the zero filter by default
  isZeroFilter = true;

}





///
/// Check if a LASModuleProfile indicates that the module has been hit,
/// i.e. contains a visible signal or is even distorted by too high laser amplitude.
/// This method doesn't care if the profile is usable for analysis.
///
bool LASProfileJudge::IsSignalIn( const LASModuleProfile& aProfile, int offset ) {

  profile = aProfile;
  
  double negativity = GetNegativity( offset );
  bool isPeaks = IsPeaksInProfile( offset );
  bool isNegativePeaks = IsNegativePeaksInProfile( offset );
  
  bool result = 
    ( negativity < -1000. ) ||  // if we see negativity, there was laser..
    ( isPeaks )             ||  // if we see a peak, " " "
    ( isNegativePeaks );    // same here
  
  return( result );


}





///
/// Check if a LASModuleProfile is usable for being stored,
/// i.e. contains a visible signal & no baseline distortions
///
bool LASProfileJudge::JudgeProfile( const LASModuleProfile& aProfile, int offset = 0 ) {

  profile = aProfile;
  
  // run the tests
  double negativity = GetNegativity( offset );

  bool isPeaks;
  if( isZeroFilter ) isPeaks = true; // disable this test if set in cfg
  else isPeaks = IsPeaksInProfile( offset );

  bool isNegativePeaks = IsNegativePeaksInProfile( offset );

  bool result = 
    ( negativity > -1000. ) &&  // < 1000. = distorted profile
    ( isPeaks )             &&  // want to see a peak (zero filter)
    !( isNegativePeaks ); // no negative peaks

  return( result );

}





///
/// toggle the zero filter (passed from cfg file)
///
void LASProfileJudge::EnableZeroFilter( bool zeroFilter ) {

  isZeroFilter = zeroFilter;

}





///
/// In case of too high laser intensities, the APV baselines tend
/// to drop down. here, the strip amplitudes in the area around the
/// signal region are summed to return a variable which can indicate this.
///
double LASProfileJudge::GetNegativity( int offset ) {

  // here we could later run the sum only on the affected (pair of) APV

  // expected beam position (in strips)
  const unsigned int meanPosition = 256 + offset;
  // backplane "alignment hole" (="signal region") approx. half size 
  const unsigned int halfWindowSize = 33;
  // half size of range over which is summed (must be > halfWindowSize)
  const unsigned int sumHalfRange = 128;

  double neg = 0;
  
  for( unsigned int i = meanPosition - sumHalfRange; i < meanPosition - halfWindowSize; ++i ) {
    neg += profile.GetValue( i );
  }

  for( unsigned int i = meanPosition + halfWindowSize; i < meanPosition + sumHalfRange; ++i ) {
    neg += profile.GetValue( i );
  }

  return( neg );

}




///
/// If the laser intensity is too small, there's no peak at all.
/// Here we look if any strip is well above noise level.
///
bool LASProfileJudge::IsPeaksInProfile( int offset ) {

  // expected beam position (in strips)
  const unsigned int meanPosition = 256 + offset;
  // backplane "alignment hole" approx. half size (in strips)
  const unsigned int halfWindowSize = 33;

  bool returnValue = false;
  
  // calculate average out-of-signal
  double average = 0., counterD = 0.;
  for( unsigned int strip = 0; strip < 512; ++strip ) {
    if( strip < meanPosition - halfWindowSize || strip > meanPosition + halfWindowSize ) {
      average += profile.GetValue( strip );
      counterD += 1.;
    }
  }
  average /= counterD;

  // find peaks well above noise level
  const double noiseLevel = 2.; // to be softcoded..
  for( unsigned int strip = meanPosition - halfWindowSize; strip < meanPosition + halfWindowSize; ++strip ) {
    if( profile.GetValue( strip ) > ( average + 10. * noiseLevel ) ) { 
      returnValue = true;
      thePeak.first = strip; thePeak.second = profile.GetValue( strip );
      break;
    }
  }

  return( returnValue );

}




///
/// sometimes when the laser intensity is too high the APVs get confused
/// and a negative peak (dip) shows up. this is filtered here.
///
bool LASProfileJudge::IsNegativePeaksInProfile( int offset ) {

  // expected beam position in middle of module (in strips)
  const unsigned int meanPosition = 256 + offset;
  // backplane "alignment hole" approx. half size (in strips)
  const unsigned int halfWindowSize = 33;

  bool returnValue = false;
  
  // calculate average out-of-signal
  double average = 0., counterD = 0.;
  for( unsigned int strip = 0; strip < 512; ++strip ) {
    if( strip < meanPosition - halfWindowSize || strip > meanPosition + halfWindowSize ) {
      average += profile.GetValue( strip );
      counterD += 1.;
    }
  }
  average /= counterD;
  
  // find strips with negative amplitude way above noise level
  const double noiseLevel = 2.;
  for( unsigned int strip = 0; strip < 512; ++strip ) {
    if( profile.GetValue( strip ) < ( average - 10. * noiseLevel ) ) { 
      returnValue = true;
      thePeak.first = strip; thePeak.second = profile.GetValue( strip );
      break;
    }
  }

  return( returnValue );

}
