

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
}




///
/// check if a LASModuleProfile is usable for being stored
///
bool LASProfileJudge::JudgeProfile( const LASModuleProfile& aProfile, int offset = 0 ) {

  profile = aProfile;
  
  // run the tests
  double negativity = GetNegativity( offset );
  bool isPeaks = IsPeaksInProfile( offset );
  bool isNegativePeaks = IsNegativePeaksInProfile( offset );

  bool result = 
    ( negativity > -1000. ) &&  //&
    ( isPeaks )             &&  //&
   !( isNegativePeaks );

  return( result );

}




///
/// in case of too high laser intensities, the APV baselines tend
/// to drop down. here, the strip amplitudes in the area around the
/// signal region are summed to compute a variable which can indicate this.
///
double LASProfileJudge::GetNegativity( int offset ) {

  // expected beam position (in strips)
  const unsigned int meanPosition = 256 + offset;
  // backplane "alignment hole" approx. half size (in strips)
  const unsigned int halfWindowSize = 33;

  double neg = 0;
  
  for( unsigned int i = 128; i < meanPosition - halfWindowSize; ++i ) {
    neg += profile.GetValue( i );
  }

  for( unsigned int i = meanPosition + halfWindowSize; i < 384; ++i ) {
    neg += profile.GetValue( i );
  }

  return( neg );

}




///
/// if the laser intensity is too small, there's no peak at all.
/// here we look if any strip is well above noise level.
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
