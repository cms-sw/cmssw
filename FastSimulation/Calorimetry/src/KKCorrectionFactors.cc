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
      << "\", used for correcting the response of the ECAL in FastSim.\n";
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
      &&  genE < h3_->GetXaxis()->GetXmax()
      && genEta > h3_->GetYaxis()->GetXmin()
      &&  genEta < h3_->GetYaxis()->GetXmax() ) {

    scale = h3_->Interpolate( genE, genEta, r );

  } else { // interpolation in r is mandatory

    int binE   = h3_->GetXaxis()->FindFixBin( genE );
    int binEta = h3_->GetYaxis()->FindFixBin( genEta );

    scale = h3_->ProjectionZ( "proZ", binE, binE, binEta, binEta )->Interpolate( r );

  }
  return scale;

}

