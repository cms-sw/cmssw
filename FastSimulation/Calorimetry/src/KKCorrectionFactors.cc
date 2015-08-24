#include "FastSimulation/Calorimetry/interface/KKCorrectionFactors.h"

KKCorrectionFactors::KKCorrectionFactors( const edm::ParameterSet& pset )
{

  interpolate_ = pset.exists("interpolate") && pset.getUntrackedParameter<bool>("interpolate");

  // Get the filename and histogram name
  std::string fileName = pset.getUntrackedParameter<std::string>("fileName");
  std::string histogramName = pset.getUntrackedParameter<std::string>("histogramName");

  // Read histo
  edm::FileInPath myDataFile( fileName ); // TODO: check if this is necessary
  TFile * myFile = TFile::Open( myDataFile.fullPath().c_str(), "READ" );

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

  // fill empty bins with 1.
  for( int x=0; x<h3_->GetNbinsX()+2; x++ ) {
    for( int y=0; y<h3_->GetNbinsY()+2; y++ ) {
      for( int z=0; z<h3_->GetNbinsZ()+2; z++ ) {
        if( h3_->GetBinContent(x,y,z) < 1e-5 ) {
          h3_->SetBinContent(x,y,z,1.);
        }
      }
    }
  }

  myFile->Close();
  delete myFile;

}


float KKCorrectionFactors::getScale( float genE, float genEta, float simE ) const
{

  float r = simE / genE;
  float scale = 1.;

  if( interpolate_
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

  // if the scale is unreasonable small, e.g. zero, do not scale
  if( scale < 1e-5 ) { scale = 1.; }

  return scale;

}

