#include "FastSimulation/Calorimetry/interface/CaloResponse.h"

CaloResponse::CaloResponse( const edm::ParameterSet& pset )
{

  // Get the filename and histogram name
  std::string ecalScalesFileName_ = pset.getUntrackedParameter<std::string>("fileName");
  std::string ecalScalesHistogramName_ = pset.getUntrackedParameter<std::string>("histogramName");

  // Read histo
  edm::FileInPath myDataFile( ecalScalesFileName_ ); // TODO: check if this is necessary
  TFile * myFile = new TFile( myDataFile.fullPath().c_str(), "READ" );

  gROOT->cd(); // create histogram in memory
  auto obj = myFile->Get( ecalScalesHistogramName_.c_str() );

  // Check if histogram exists in file
  if(!obj) {
    throw cms::Exception( "FileReadError" )
      << "Histogram \"" << ecalScalesHistogramName_
      << "\" not found in file \"" << ecalScalesFileName_
      << "\", used for correcting the response of the ECAL in FastSim.\n";
  }
  h3_ = new TH3F( *((TH3F*)obj) );

  myFile->Close();
  delete myFile;

}


float CaloResponse::getScale( const RawParticle& particleAtEcalEntrance,
    const std::map<CaloHitID,float>& hitMap ) const
{

  double simE = 0; // total simulated energy for this particle
  for( auto mapIterator : hitMap ) {
    simE += mapIterator.second;
  }

  float genEta = std::abs( particleAtEcalEntrance.eta() );
  float genE = particleAtEcalEntrance.e();

  return getScale( genE, genEta, simE );

}


float CaloResponse::getScale( float genE, float genEta, float simE ) const
{

  float r = simE / genE;

  int binE   = h3_->GetXaxis()->FindFixBin( genE );
  int binEta = h3_->GetYaxis()->FindFixBin( genEta );

  // find the two bins which are closest to the actual value
  auto binWidthR = h3_->GetZaxis()->GetBinWidth(0);
  int binRup = h3_->GetZaxis()->FindFixBin( r + binWidthR/2. );
  int binRdn = h3_->GetZaxis()->FindFixBin( r - binWidthR/2. );

  auto scaleUp = h3_->GetBinContent( binE, binEta, binRup );
  auto scaleDn = h3_->GetBinContent( binE, binEta, binRdn );

  auto scale = 1.;

  if( scaleUp > 1e-5 && scaleDn > 1e-5 ) {

    // make a linear extrapolation between neighbour bins if they are not zero
    auto Rup = h3_->GetZaxis()->GetBinCenter( binRup );
    auto Rdn = h3_->GetZaxis()->GetBinCenter( binRdn );
    scale = scaleDn + (scaleUp-scaleDn) * ( r - Rdn ) / ( Rup - Rdn );

  }

  // if the scale is unreasonable small, e.g. zero, do not scale
  if( scale < 0.0001 ) { scale = 1.; }

  return scale;

}

