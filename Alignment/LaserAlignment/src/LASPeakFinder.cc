
#include "Alignment/LaserAlignment/src/LASPeakFinder.h"


///
///
///
LASPeakFinder::LASPeakFinder() {
}





///
/// set a profile to work on and start peak finding;
/// the pair<> will return mean/meanError (in strips);
/// offset is necessary for tob modules which are hit off-center
///
bool LASPeakFinder::FindPeakIn( const LASModuleProfile& aProfile, std::pair<double,double>& result, const int offset ) {

  TH1D* histogram = new TH1D( "bufferHistogram", "bufferHistogram", 512, 0, 512 );
  TF1* fitFunction = new TF1( "fitFunction", "gaus" );

  std::pair<int,double> largestAmplitude( 0, 0. ); // strip, amplitude
  double anAmplitude = 0.;

  // expected beam position (in strips)
  const unsigned int meanPosition = 256 + offset;
  // backplane "alignment hole" approx. half size (in strips)
  const unsigned int halfWindowSize = 33;

  // loop over the strips in the "alignment hole"
  // to fill the histogram
  // and determine fit parameter estimates
  for( unsigned int strip = meanPosition - halfWindowSize; strip < meanPosition + halfWindowSize; ++strip ) {
    anAmplitude = aProfile.GetValue( strip );
    histogram->SetBinContent( 1 + strip, anAmplitude );
    if( anAmplitude > largestAmplitude.second ) {
      largestAmplitude.first = strip; largestAmplitude.second = anAmplitude;
    }
  }

  // loop outside the "alignment hole"
  // to determine the noise level = sqrt(variance)
  double sum1 = 0., sum2 = 0.;
  int nStrips = 0;
  for( unsigned int strip = 0; strip < 512; ++strip ) {
    if( strip < meanPosition - halfWindowSize || strip > meanPosition + halfWindowSize ) {
      anAmplitude = aProfile.GetValue( strip );
      sum1 += anAmplitude;
      sum2 += pow( anAmplitude, 2 );
      nStrips++;
    }
  }

  // noise as sqrt of the amplitude variance
  const double noise = sqrt( 1. / ( nStrips - 1 ) * ( sum2 - pow( sum1, 2 ) / nStrips ) );

  // empty profile?
  if( fabs( sum1 ) < 1.e-3 ) {
    std:: cout << " [LASPeakFinder::FindPeakIn] ** WARNING: Empty profile." << std::endl;/////////////////////////////////
    return false;
  }

  // no reasonable peak?
  if( largestAmplitude.second < 10. * noise ) {
    std::cout << " [LASPeakFinder::FindPeakIn] ** WARNING: No reasonably large peak." << std::endl;/////////////////////////////////
    return false;
  }

  // prepare fit function: starting values..
  fitFunction->SetParameter( 0, largestAmplitude.second ); // amp
  fitFunction->SetParameter( 1, largestAmplitude.first ); // mean
  fitFunction->SetParameter( 2, 3. ); // width

  // ..and parameter limits
  fitFunction->SetParLimits( 0, largestAmplitude.second * 0.3, largestAmplitude.second * 1.8 ); // amp of the order of the peak height
  fitFunction->SetParLimits( 1, largestAmplitude.first - 12, largestAmplitude.first + 12 ); // mean around the peak maximum
  fitFunction->SetParLimits( 2, 0.5, 8. ); // reasonable width
  

  // and go
  histogram->Fit( fitFunction, "QWB", "", largestAmplitude.first - 12, largestAmplitude.first + 12 );
  //  std::cout << "MEAN: " << fitFunction->GetParameter( 1 ) << "±" << fitFunction->GetParError( 1 ) << ", AMP: " 
  //	    << fitFunction->GetParameter( 0 ) << ", WIDTH: " << fitFunction->GetParameter( 2 )
  //	    << ", NOISE: " << noise << std::endl; /////////////////////////////////

  // prepare output
  result.first = fitFunction->GetParameter( 1 );
  result.second = fitFunction->GetParError( 1 );

  return true;

}
