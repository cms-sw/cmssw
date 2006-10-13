#include <memory>

// Framework
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
//

// Geometry
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "DataFormats/SiStripDetId/interface/TIBDetId.h" 
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h" 
#include "DataFormats/SiStripDetId/interface/TECDetId.h" 
//

// Hits
#include "DataFormats/TrackerRecHit2D/interface/SiTrackerGSRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiTrackerGSRecHit2DCollection.h"
#include "DataFormats/Common/interface/OwnVector.h" 
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"
//

// ROOT
#include <TROOT.h>
#include <TStyle.h>
#include <TGaxis.h>
#include <TFile.h>
#include <TTree.h>
#include <TVector3.h>
#include <TF1.h>
#include <TCanvas.h>
#include <TLegend.h>
//

// std
#include <iostream>
#include <string>
//

// itself
#include "FastSimulation/TrackingRecHitProducer/test/FamosRecHitAnalysis.h"
//

#define rrDEBUG

FamosRecHitAnalysis::FamosRecHitAnalysis(edm::ParameterSet const& pset) : 
  _pset(pset),
  theRecHits_( pset.getParameter<edm::InputTag>("RecHits") )
{
#ifdef rrDEBUG
  std::cout << "Start Famos RecHit Analysis" << std::endl;
#endif
}

void FamosRecHitAnalysis::beginJob(const edm::EventSetup& setup) {
  // Root File
  std::string rootFileName = _pset.getParameter<std::string>("RootFileName");
  theRootFile = new TFile ( rootFileName.c_str() , "RECREATE" );
  theRootFile->cd();
#ifdef rrDEBUG
  std::cout << "Root File " << rootFileName << " Created" << std::endl;
#endif
  //
  rootStyle();
  //
  book();
  //
}

void FamosRecHitAnalysis::book() {
  // Microstrips
  int    nbin   = 2000;
  double minmax = 1.0;
  // TIB
  bookValues( histos_TIB_x , histos_TIB_y , histos_TIB_z , nbin , minmax , "TIB" , nHist_TIB );
  bookErrors( histos_TIB_err_x , histos_TIB_err_y , histos_TIB_err_z , 500 , 0.0500 , "TIB" , nHist_TIB );
  bookNominals( histos_TIB_nom_x , nbin , minmax , "TIB" , nHist_TIB );
  // TID
  bookValues( histos_TID_x , histos_TID_y , histos_TID_z , nbin , minmax , "TID" , nHist_TID );
  bookErrors( histos_TID_err_x , histos_TID_err_y , histos_TID_err_z , 500 , 0.0500 , "TID" , nHist_TID );
  bookNominals( histos_TID_nom_x , nbin , minmax , "TID" , nHist_TID );
  // TOB
  bookValues( histos_TOB_x , histos_TOB_y , histos_TOB_z , nbin , minmax , "TOB" , nHist_TOB );
  bookErrors( histos_TOB_err_x , histos_TOB_err_y , histos_TOB_err_z , 500 , 0.0500 , "TOB" , nHist_TOB );
  bookNominals( histos_TOB_nom_x , nbin , minmax , "TOB" , nHist_TOB );
  // TEC
  bookValues( histos_TEC_x , histos_TEC_y , histos_TEC_z , nbin , minmax , "TEC" , nHist_TEC );
  bookErrors( histos_TEC_err_x , histos_TEC_err_y , histos_TEC_err_z , 500 , 0.0500 , "TEC" , nHist_TEC );
  bookNominals( histos_TEC_nom_x , nbin , minmax , "TEC" , nHist_TEC );
  //
#ifdef rrDEBUG
  std::cout << "Famos histograms " << theRootFile->GetName() << " booked" << std::endl;
#endif
}

void FamosRecHitAnalysis::bookValues(std::vector<TH1F*>& histos_x , std::vector<TH1F*>& histos_y , std::vector<TH1F*>& histos_z , int nBin, float range, char* det, unsigned int nHist) {
  //
  for(unsigned int iHist = 0; iHist < nHist; iHist++) {
    histos_x.push_back(            new TH1F(Form( "hist_%s_%u_deltaX" , det , iHist+1 ) ,
					    Form( "Hit Local Position #Deltax=x_{Rec}-x_{Sim} %s %u;#Deltax [cm];Entries/bin" , det , iHist+1 ) ,
					    nBin , -range*0.05 ,  range*0.05 ));
    histos_y.push_back(            new TH1F(Form( "hist_%s_%u_deltaY" , det , iHist+1 ) ,
					    Form( "Hit Local Position #Deltay=y_{Rec}-y_{Sim} %s %u;#Deltay [cm];Entries/bin" , det , iHist+1 ) ,
					    nBin , -range*10.0 ,  range*10.0 ));
    histos_z.push_back(            new TH1F(Form( "hist_%s_%u_deltaZ" , det , iHist+1 ) ,
					    Form( "Hit Local Position #Deltaz=z_{Rec}-z_{Sim} %s %u;#Deltaz [cm];Entries/bin" , det , iHist+1 ) ,
					    nBin , -range*0.5  ,  range*0.5  ));
  }
  //
}

void FamosRecHitAnalysis::bookErrors(std::vector<TH1F*>& histos_x , std::vector<TH1F*>& histos_y , std::vector<TH1F*>& histos_z , int nBin, float range, char* det, unsigned int nHist) {
  //
  for(unsigned int iHist = 0; iHist < nHist; iHist++) {
    histos_x.push_back(            new TH1F(Form( "hist_%s_%u_errX" , det , iHist+1 ) ,
					    Form( "Hit Local Error x %s %u;Resolution(x) [cm];Entries/bin" , det , iHist+1 ) ,
					    nBin , 0.0 ,  range      ));
    histos_y.push_back(            new TH1F(Form( "hist_%s_%u_errY" , det , iHist+1 ) ,
					    Form( "Hit Local Error y %s %u;Resolution(y) [cm];Entries/bin" , det , iHist+1 ) ,
					    nBin , 0.0 ,  100.*range ));
    histos_z.push_back(            new TH1F(Form( "hist_%s_%u_errZ" , det , iHist+1 ) ,
					    Form( "Hit Local Error z %s %u;Resolution(z) [cm];Entries/bin" , det , iHist+1 ) ,
					    nBin , 0.0 ,  range      ));
  }
  //
}

void FamosRecHitAnalysis::bookNominals(std::vector<TH1F*>& histos_x , int nBin, float range, char* det, unsigned int nHist) {
  //
  for(unsigned int iHist = 0; iHist < nHist; iHist++) {
    histos_x.push_back(            new TH1F(Form( "hist_%s_%u_nomX" , det , iHist+1 ) ,
					    Form( "Hit Local Position Nominal #Deltax=x_{Rec}-x_{Sim} %s %u;#Deltax [cm];Entries/bin" , det , iHist+1 ) ,
					    nBin , -range*0.05 ,  range*0.05 ));
  }
  //
}

//
void FamosRecHitAnalysis::write(std::vector<TH1F*> histos) {
  //
  for(std::vector<TH1F*>::iterator iHist = histos.begin(); iHist < histos.end(); iHist++) {
    (*iHist)->Write();
  }
  //
}
//

// Virtual destructor needed.
FamosRecHitAnalysis::~FamosRecHitAnalysis() { 
#ifdef rrDEBUG
  std::cout << "End Famos RecHit Analysis" << std::endl;
#endif
}  

// Functions that gets called by framework every event
void FamosRecHitAnalysis::analyze(const edm::Event& event, const edm::EventSetup& setup)
{
#ifdef rrDEBUG
  std::cout << "Famos analysis" << std::endl;
#endif
  // get event and run number
  int t_Run   = event.id().run();
  int t_Event = event.id().event();
#ifdef rrDEBUG
  std::cout
    << " #################################### Run " << t_Run 
    << " Event "                                    << t_Event << " #################################### " 
    << std::endl;
#endif
  //
  
  // RecHits
#ifdef rrDEBUG
  std::cout << "Famos RecHits" << std::endl;
#endif
  edm::Handle<SiTrackerGSRecHit2DCollection> theRecHits;
  event.getByLabel(theRecHits_, theRecHits);
  
  // histograms to fill
  TH1F* hist_x = 0;
  TH1F* hist_y = 0;
  TH1F* hist_z = 0;
  TH1F* hist_err_x = 0;
  TH1F* hist_err_y = 0;
  TH1F* hist_err_z = 0;
  //
  
  // loop on RecHits, no need to associate to PsimHits in Famos, because they have their PSimHit as member
  unsigned int iRecHit = 0;
  const std::vector<DetId> theDetIds = theRecHits->ids();
  // loop over detunits
  for ( std::vector<DetId>::const_iterator iDetId = theDetIds.begin(); iDetId != theDetIds.end(); iDetId++ ) {
    unsigned int detid = (*iDetId).rawId();
    if(detid!=999999999){ // valid detector
      SiTrackerGSRecHit2DCollection::range theRecHitRange = theRecHits->get((*iDetId));
      SiTrackerGSRecHit2DCollection::const_iterator theRecHitRangeIteratorBegin = theRecHitRange.first;
      SiTrackerGSRecHit2DCollection::const_iterator theRecHitRangeIteratorEnd   = theRecHitRange.second;
      SiTrackerGSRecHit2DCollection::const_iterator iterRecHit = theRecHitRangeIteratorBegin;
      // loop over RecHits of the same detector
      for(iterRecHit = theRecHitRangeIteratorBegin; iterRecHit != theRecHitRangeIteratorEnd; ++iterRecHit) {
	iRecHit++;
	float delta_x = (*iterRecHit).localPosition().x() - (*iterRecHit).simhit().localPosition().x();
	float delta_y = (*iterRecHit).localPosition().y() - (*iterRecHit).simhit().localPosition().y();
	float delta_z = (*iterRecHit).localPosition().z() - (*iterRecHit).simhit().localPosition().z();
	float err_x = sqrt((*iterRecHit).localPositionError().xx());
	float err_y = sqrt((*iterRecHit).localPositionError().yy());
	float err_z = 0.0;
#ifdef rrDEBUG
	std::cout << "\t" << iRecHit << std::endl;
	std::cout << "\tRecHit"
		  << "\t\tx = " << (*iterRecHit).localPosition().x() << " cm"
		  << "\t\ty = " << (*iterRecHit).localPosition().y() << " cm"
		  << "\t\tz = " << (*iterRecHit).localPosition().z() << " cm"
		  << std::endl;
	std::cout << "\tSimHit"
		  << "\t\tx = " << (*iterRecHit).simhit().localPosition().x() << " cm"
		  << "\t\ty = " << (*iterRecHit).simhit().localPosition().y() << " cm"
		  << "\t\tz = " << (*iterRecHit).simhit().localPosition().z() << " cm"
		  << std::endl;
	std::cout << "\tResiduals"
		  << "\t\tx = " << delta_x << " cm"
		  << "\t\ty = " << delta_y << " cm"
		  << "\t\tz = " << delta_z << " cm"
		  << std::endl;
	std::cout << "\tRecHit error (resolution)"
		  << "\t\tx = " << err_x << " cm"
		  << "\t\ty = " << err_y << " cm"
		  << "\t\tz = " << err_z << " cm"
		  << std::endl;
#endif
	// fill proper histograms
	chooseHist( detid , hist_x , hist_y , hist_z , hist_err_x , hist_err_y , hist_err_z );
	if(hist_x != 0) {
#ifdef rrDEBUG
	  std::cout << "\tFill histograms " << hist_x->GetName() << ", " << hist_y->GetName() << ", " << hist_z->GetName() << std::endl;
#endif	  
	  hist_x->Fill( delta_x );
	  hist_y->Fill( delta_y );
	  hist_z->Fill( delta_z );
	  hist_err_x->Fill( err_x );
	  hist_err_y->Fill( err_y );
	  hist_err_z->Fill( err_z );
	}
      } // loop over RecHits
      //
    } // valid detector
  } // loop over detunits
  
}

void FamosRecHitAnalysis::endJob() {
  //
  theRootFile->cd();
  // before closing file do root macro
  // TIB
  rootMacroStrip( histos_TIB_x , histos_TIB_y , histos_TIB_z , histos_TIB_err_x , histos_TIB_err_y , histos_TIB_err_z , histos_TIB_nom_x );
  // TID
  rootMacroStrip( histos_TID_x , histos_TID_y , histos_TID_z , histos_TID_err_x , histos_TID_err_y , histos_TID_err_z , histos_TID_nom_x );
  // TOB
  rootMacroStrip( histos_TOB_x , histos_TOB_y , histos_TOB_z , histos_TOB_err_x , histos_TOB_err_y , histos_TOB_err_z , histos_TOB_nom_x );
  // TEC
  rootMacroStrip( histos_TEC_x , histos_TEC_y , histos_TEC_z , histos_TEC_err_x , histos_TEC_err_y , histos_TEC_err_z , histos_TEC_nom_x );
  //
  // Write Histograms
  // TIB
  write(histos_TIB_x);
  write(histos_TIB_y);
  write(histos_TIB_z);
  write(histos_TIB_err_x);
  write(histos_TIB_err_y);
  write(histos_TIB_err_z);
  write(histos_TIB_nom_x);
  // TID
  write(histos_TID_x);
  write(histos_TID_y);
  write(histos_TID_z);
  write(histos_TID_err_x);
  write(histos_TID_err_y);
  write(histos_TID_err_z);
  write(histos_TID_nom_x);
  // TOB
  write(histos_TOB_x);
  write(histos_TOB_y);
  write(histos_TOB_z);
  write(histos_TOB_err_x);
  write(histos_TOB_err_y);
  write(histos_TOB_err_z);
  write(histos_TOB_nom_x);
  // TEC
  write(histos_TEC_x);
  write(histos_TEC_y);
  write(histos_TEC_z);
  write(histos_TEC_err_x);
  write(histos_TEC_err_y);
  write(histos_TEC_err_z);
  write(histos_TEC_nom_x);
  //
  //
  rootComparison( histos_TIB_x,histos_TIB_nom_x , 20 );
  rootComparison( histos_TID_x,histos_TID_nom_x , 20 );
  rootComparison( histos_TOB_x,histos_TOB_nom_x , 40 );
  rootComparison( histos_TEC_x,histos_TEC_nom_x , 40 );
  //
  theRootFile->Close();
  //
}

//
void FamosRecHitAnalysis::chooseHist(unsigned int rawid, TH1F*& hist_x , TH1F*& hist_y , TH1F*& hist_z, TH1F*& hist_err_x , TH1F*& hist_err_y , TH1F*& hist_err_z) {
  int subdetid = ((rawid>>25)&0x7);
  
  switch (subdetid) {
    // TIB
  case 3:
    {
      TIBDetId module(rawid);
      unsigned int theLayer = module.layer();
      hist_x = histos_TIB_x[theLayer-1];
      hist_y = histos_TIB_y[theLayer-1];
      hist_z = histos_TIB_z[theLayer-1];
      hist_err_x = histos_TIB_err_x[theLayer-1];
      hist_err_y = histos_TIB_err_y[theLayer-1];
      hist_err_z = histos_TIB_err_z[theLayer-1];
#ifdef rrDEBUG
      std::cout << "\tTracker subdetector " << subdetid << " TIB Layer " << theLayer << std::endl;
#endif
      break;
    } 
    //
    // TID
  case 4:
    {
      TIDDetId module(rawid);
      unsigned int theRing = module.ring();
      hist_x = histos_TID_x[theRing-1];
      hist_y = histos_TID_y[theRing-1];
      hist_z = histos_TID_z[theRing-1];
      hist_err_x = histos_TID_err_x[theRing-1];
      hist_err_y = histos_TID_err_y[theRing-1];
      hist_err_z = histos_TID_err_z[theRing-1];
#ifdef rrDEBUG
      std::cout << "\tTracker subdetector " << subdetid << " TID Ring " << theRing << std::endl;
#endif
      break; 
    }
    //
    // TOB
  case 5:
    {
      TOBDetId module(rawid);
      unsigned int theLayer = module.layer();
      hist_x = histos_TOB_x[theLayer-1];
      hist_y = histos_TOB_y[theLayer-1];
      hist_z = histos_TOB_z[theLayer-1];
      hist_err_x = histos_TOB_err_x[theLayer-1];
      hist_err_y = histos_TOB_err_y[theLayer-1];
      hist_err_z = histos_TOB_err_z[theLayer-1];
#ifdef rrDEBUG
      std::cout << "\tTracker subdetector " << subdetid << " TOB Layer " << theLayer << std::endl;
#endif
      break;
    }
    //
    // TEC
  case 6:
    {
      TECDetId module(rawid);
      unsigned int theRing = module.ring();
      hist_x = histos_TEC_x[theRing-1];
      hist_y = histos_TEC_y[theRing-1];
      hist_z = histos_TEC_z[theRing-1];
      hist_err_x = histos_TEC_err_x[theRing-1];
      hist_err_y = histos_TEC_err_y[theRing-1];
      hist_err_z = histos_TEC_err_z[theRing-1];
#ifdef rrDEBUG
      std::cout << "\tTracker subdetector " << subdetid << " TEC Ring " << theRing << std::endl;
#endif
      break;
    }
    //
  default: 
    {
#ifdef rrDEBUG
      std::cout << "\tTracker subdetector not valid " << subdetid << std::endl;
#endif
      break;
    }
    //
  } // switch
}
//

void FamosRecHitAnalysis::rootStyle() {
  // rrStyle
  TStyle* rrStyle = new TStyle("rrStyle","rrStyle");
  TGaxis::SetMaxDigits(3);          // to avoid too much decimal digits
  rrStyle->SetOptStat(2211);        // general statistics
  rrStyle->SetOptFit(1111);         // fit statistics
  rrStyle->SetOptLogy(1);           // logscale
  rrStyle->SetCanvasColor(kWhite);  // white canvas
  rrStyle->SetHistFillColor(34);    // histo: blue gray filling
  rrStyle->SetFuncColor(146);       // function: dark red line
  //
  // ROOT macro
  gROOT->SetBatch();
  gROOT->SetStyle("rrStyle");
}

//
void FamosRecHitAnalysis::rootMacroStrip( std::vector<TH1F*>& histos_x      , std::vector<TH1F*>& histos_y     , std::vector<TH1F*>& histos_z     ,
					  std::vector<TH1F*>& histos_err_x  , std::vector<TH1F*>& histos_err_y , std::vector<TH1F*>& histos_err_z ,
					  std::vector<TH1F*>& histos_nom_x   ) {
  // gaussian fits
  for(unsigned int iHist = 0; iHist < histos_x.size(); iHist++) {
    //
#ifdef rrDEBUG
    std::cout << "\tFit " << iHist << std::endl;
#endif
    TF1* gaussianFit_histos_x = new TF1("gaussianFit_histos_x","gaus",
					histos_x[iHist]->GetMean() - 5. * histos_x[iHist]->GetRMS() ,
					histos_x[iHist]->GetMean() + 5. * histos_x[iHist]->GetRMS() );
    histos_x[iHist]->Fit("gaussianFit_histos_x","R");
    //
    TF1* constantFit_histos_y = new TF1("constantFit_histos_y","pol1",
					histos_y[iHist]->GetMean() - 2. * histos_y[iHist]->GetRMS() ,
					histos_y[iHist]->GetMean() + 2. * histos_y[iHist]->GetRMS() );
    histos_y[iHist]->Fit("constantFit_histos_y","R");
    /*
      TF1* gaussianFit_histos_z = new TF1("gaussianFit_histos_z","gaus",
      histos_z[iHist]->GetMean() - 5. * histos_z[iHist]->GetRMS() ,
      histos_z[iHist]->GetMean() + 5. * histos_z[iHist]->GetRMS() );
      histos_z[iHist]->Fit("gaussianFit_histos_z","R");
    */
    
    // compatibility check for local x axis
    TF1* gaussianResolution = new TF1("gaussianResolution_histos_x","gaus");
    gaussianResolution->FixParameter( 0 , histos_x[iHist]->GetEntries() / ( sqrt(2*3.141592654) * histos_err_x[iHist]->GetMean()) ); // same normalization
    gaussianResolution->FixParameter( 1 , 0.0 ); // mean = 0.
    gaussianResolution->FixParameter( 2 , histos_err_x[iHist]->GetMean() ); // sigma set to RecHit error
    histos_nom_x[iHist]->FillRandom(gaussianResolution->GetName(),(int)histos_x[iHist]->GetEntries());
    histos_nom_x[iHist]->SetEntries(histos_x[iHist]->GetEntries());
    //
  }
}
//

//
void FamosRecHitAnalysis::rootComparison( std::vector<TH1F*> histos_value , std::vector<TH1F*> histos_nominal , int binFactor ) {
  //
  for(unsigned int iHist = 0; iHist < histos_value.size(); iHist++) {
    // canvas
    TCanvas can_comparison("can_comparison","can_comparison",800,800);
    can_comparison.Range(0,0,25,25);
    can_comparison.SetFillColor(kWhite);
    can_comparison.SetGridy(1);
    can_comparison.SetLogy(1);
    // settings
    gStyle->SetOptStat(0);
    gStyle->SetOptFit(0);
    //
    histos_value[iHist]->SetMarkerColor(104); // dark blue
    histos_value[iHist]->SetLineColor(4); // blue
    histos_value[iHist]->SetMarkerStyle(20); // cyrcles
    histos_value[iHist]->SetMarkerSize(1.0); // 
    histos_nominal[iHist]->SetLineColor(102); // dark red
    histos_nominal[iHist]->SetFillColor(0); // white
    //
    histos_value[iHist]->Rebin(binFactor);
    histos_nominal[iHist]->Rebin(binFactor);
    histos_value[iHist]->GetXaxis()->SetRangeUser( histos_value[iHist]->GetMean() - 3 * histos_value[iHist]->GetRMS() , 
						   histos_value[iHist]->GetMean() + 3 * histos_value[iHist]->GetRMS() );
    // Draw
    histos_value[iHist]->Draw("HIST P E1");
    histos_nominal[iHist]->Draw("HIST SAME");
    //
    // perform chi2 test between obtained and nominal histograms
    double compatibilityFactor = histos_value[iHist]->KolmogorovTest(histos_nominal[iHist],"");
    std::cout << " Compatibility of " << histos_value[iHist]->GetName()
	      << " with nominal distribution " << histos_nominal[iHist]->GetName() << " is " << compatibilityFactor << std::endl;
    // Legenda
    TLegend* theLegend = new TLegend(0.70, 0.70, 0.89, 0.89);
    theLegend->AddEntry( histos_value[iHist]   , "RecHits"          , "p" );
    theLegend->AddEntry( histos_nominal[iHist] , "Nominal Smearing" , "l" );
    theLegend->SetHeader( Form("Compatibility: %f",compatibilityFactor) );
    theLegend->Draw();
    //
    // Store
    can_comparison.Update();
    can_comparison.SaveAs( Form( "Images/Comparison_%s.eps"  , histos_value[iHist]->GetName() ) );
    can_comparison.SaveAs( Form( "Images/Comparison_%s.gif"  , histos_value[iHist]->GetName() ) );
    //
  }
  //
}
//

//
DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(FamosRecHitAnalysis);
