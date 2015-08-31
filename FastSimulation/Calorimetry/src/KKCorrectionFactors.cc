#include "FastSimulation/Calorimetry/interface/KKCorrectionFactors.h"

KKCorrectionFactors::KKCorrectionFactors( const edm::ParameterSet& pset )
{

  interpolate3D_ = pset.exists("interpolate3D") && pset.getUntrackedParameter<bool>("interpolate3D");

  // Get the filename and histogram name
  std::string fileName = pset.getUntrackedParameter<std::string>("fileName");
  std::string histogramName = pset.getUntrackedParameter<std::string>("histogramName");

  // Read histo
  edm::FileInPath myDataFile( fileName );
  TFile * myFile = TFile::Open( myDataFile.fullPath().c_str() );

  gROOT->cd(); // create histogram in memory
  auto obj = myFile->Get( histogramName.c_str() );

  // Check if histogram exists in file
  if(!obj) {
    throw cms::Exception( "FileReadError" )
      << "Histogram \"" << histogramName
      << "\" not found in file \"" << fileName
      << "\".\n";
  }
  h3_ = new TH3F( *((TH3F*)obj) );

  delete myFile;

}


float KKCorrectionFactors::getScale( float genE, float genEta, float simE ) const
{

  float r = simE / genE;
  float scale = 1.;

  if( interpolate3D_
      // TH3::Interpolate can only interpolate inside the bondaries of the histogram
      && genE > h3_->GetXaxis()->GetXmin()
      && genE < h3_->GetXaxis()->GetXmax()
      && genEta > h3_->GetYaxis()->GetXmin()
      && genEta < h3_->GetYaxis()->GetXmax()
      && r < h3_->GetZaxis()->GetXmax()
      && r > h3_->GetZaxis()->GetXmax() ) {

    scale = h3_->Interpolate( genE, genEta, r );

  } else { // interpolation in r is mandatory

    int binE   = h3_->GetXaxis()->FindFixBin( genE );
    int binEta = h3_->GetYaxis()->FindFixBin( genEta );

    // find the two bins which are closest to the actual value
    auto binWidthR = h3_->GetZaxis()->GetBinWidth(0);
    int binRup = h3_->GetZaxis()->FindFixBin( r + binWidthR/2. );
    int binRdn = h3_->GetZaxis()->FindFixBin( r - binWidthR/2. );

    auto scaleUp = h3_->GetBinContent( binE, binEta, binRup );
    auto scaleDn = h3_->GetBinContent( binE, binEta, binRdn );

    // make a linear extrapolation between neighbour bins if they are not zero
    auto Rdn = h3_->GetZaxis()->GetBinCenter( binRdn );
    scale = scaleDn + (scaleUp-scaleDn) * ( r - Rdn ) / binWidthR;

  }
  return scale;

}

