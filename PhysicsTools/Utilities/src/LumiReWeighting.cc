#ifndef PhysicsTools_Utilities_interface_LumiReWeighting_cc
#define PhysicsTools_Utilities_interface_LumiReWeighting_cc


/**
  \class    LumiReWeighting LumiReWeighting.h "PhysicsTools/Utilities/interface/LumiReWeighting.h"
  \brief    Class to provide lumi weighting for analyzers to weight "flat-to-N" MC samples to data

  This class will trivially take two histograms:
  1. The generated "flat-to-N" distributions from a given processing (or any other generated input)
  2. A histogram generated from the "estimatePileup" macro here:

  https://twiki.cern.ch/twiki/bin/view/CMS/LumiCalc#How_to_use_script_estimatePileup

  and produce weights to convert the input distribution (1) to the latter (2).

  \author Salvatore Rappoccio, modified by Mike Hildreth
  
*/

#include "TH1.h"
#include "TFile.h"
#include <string>
#include <boost/shared_ptr.hpp>
#include "SimDataFormats/PileupSummaryInfo/interface/PileupSummaryInfo.h" 
#include "PhysicsTools/Utilities/interface/LumiReWeighting.h"

using namespace edm;

LumiReWeighting::LumiReWeighting( std::string generatedFile,
		   std::string dataFile,
		   std::string histName1 = "pileup",
		   std::string histName2 = "pileup" ) :
      generatedFileName_( generatedFile), 
      dataFileName_     ( dataFile ), 
      histName1_        ( histName1 ), 
      histName2_        ( histName2 )
      {
	generatedFile_ = boost::shared_ptr<TFile>( new TFile(generatedFileName_.c_str()) ); //MC distribution
	dataFile_      = boost::shared_ptr<TFile>( new TFile(dataFileName_.c_str()) );      //Data distribution

	weights_ = boost::shared_ptr<TH1F> ( new TH1F( *(static_cast<TH1F*>(dataFile_->Get( histName1_.c_str() )->Clone() ))));

	// MC * data/MC = data, so the weights are data/MC:

	// normalize both histograms first

	weights_->Scale( 1.0/ weights_->Integral() );
	weights_->SetName("lumiWeights");

	TH1F* den = dynamic_cast<TH1F*>(generatedFile_->Get( histName2_.c_str() ));

	den->Scale(1.0/ den->Integral());

	weights_->Divide( den );  // so now the average weight should be 1.0
      
}

LumiReWeighting::LumiReWeighting( std::vector< float > MC_distr, std::vector< float > Lumi_distr) {
  // no histograms for input: use vectors
  
  // now, make histograms out of them:

  // first, check they are the same size...

  if( MC_distr.size() != Lumi_distr.size() ){   

    std::cerr <<"ERROR: LumiReWeighting: input vectors have different sizes. Quitting... \n";
    return;

  }

  Int_t NBins = MC_distr.size();

  weights_ = boost::shared_ptr<TH1F> ( new TH1F("luminumer","luminumer",NBins,-0.5, float(NBins)-0.5) );
  TH1F* den = new TH1F("lumidenom","lumidenom",NBins,-0.5, float(NBins)-0.5) ;

  for(int ibin = 0; ibin<NBins; ++ibin ) {
    weights_->SetBinContent(ibin, Lumi_distr[ibin]);
    den->SetBinContent(ibin,MC_distr[ibin]);
  }

  // check integrals, make sure things are normalized

  float deltaH = weights_->Integral();
  if(fabs(1.0 - deltaH) > 0.02 ) { //*OOPS*...
    weights_->Scale( 1.0/ weights_->Integral() );
  }
  float deltaMC = den->Integral();
  if(fabs(1.0 - deltaMC) > 0.02 ) {
    den->Scale(1.0/ den->Integral());
  }

  weights_->Divide( den );  // so now the average weight should be 1.0    

}

double LumiReWeighting::weight( int npv ) const {
  int bin = weights_->GetXaxis()->FindBin( npv );
  return weights_->GetBinContent( bin );
}

// This version of weight does all of the work for you, assuming you want to re-weight
// using the true number of interactions in the in-time beam crossing.


double LumiReWeighting::weight( const edm::Event &e ) const {

  Handle<std::vector< PileupSummaryInfo > >  PupInfo;
  e.getByLabel(edm::InputTag("addPileupInfo"), PupInfo);

  std::vector<PileupSummaryInfo>::const_iterator PVI;

  int npv = -1;
  for(PVI = PupInfo->begin(); PVI != PupInfo->end(); ++PVI) {

    int BX = PVI->getBunchCrossing();

    if(BX == 0) { 
      npv = PVI->getPU_NumInteractions();
      continue;
    }
    //std::cout << " Pileup Information: bunchXing, nvtx: " << PVI->getBunchCrossing() << " " << PVI->getPU_NumInteractions() << std::endl;

 }

  if(npv < 0) std::cerr << " no in-time beam crossing found\n! " ;

  int bin = weights_->GetXaxis()->FindBin( npv );
  return weights_->GetBinContent( bin );
}



#endif
