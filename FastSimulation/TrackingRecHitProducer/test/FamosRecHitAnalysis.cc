#include <memory>

// Framework
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
//

// Geometry
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"
#include "DataFormats/SiStripDetId/interface/TIBDetId.h" 
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h" 
#include "DataFormats/SiStripDetId/interface/TECDetId.h" 
//

// PSimHits
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
//

// RecHits
#include "DataFormats/TrackerRecHit2D/interface/SiTrackerGSRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiTrackerGSRecHit2DCollection.h"
#include "DataFormats/Common/interface/OwnVector.h" 
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

const double PI = 3.14159265358979323;

// #define rrDEBUG

FamosRecHitAnalysis::FamosRecHitAnalysis(edm::ParameterSet const& pset) : 
  _pset(pset),
  theRecHits_( pset.getParameter<edm::InputTag>("RecHits") )
{
#ifdef rrDEBUG
  std::cout << "Start Famos RecHit Analysis" << std::endl;
#endif
  //--- PSimHit Containers
  trackerContainers.clear();
  trackerContainers = pset.getParameter<std::vector<std::string> >("SimHitList");
  //
  thePixelMultiplicityFileName = pset.getUntrackedParameter<std::string>( "PixelMultiplicityFile" , "FastSimulation/TrackingRecHitProducer/data/PixelData.root" );
  nAlphaBarrel  = pset.getUntrackedParameter<int>("AlphaBarrelMultiplicity", 4);
  nBetaBarrel   = pset.getUntrackedParameter<int>("BetaBarrelMultiplicity",  6);
  nAlphaForward = pset.getUntrackedParameter<int>("AlphaForwardMultiplicity",3);
  nBetaForward  = pset.getUntrackedParameter<int>("BetaForwardMultiplicity", 3);
  // Resolution Barrel    
  thePixelBarrelResolutionFileName = pset.getUntrackedParameter<std::string>( "PixelBarrelResolutionFile" ,
									      "FastSimulation/TrackingRecHitProducer/data/PixelBarrelResolution.root" );
  resAlphaBarrel_binMin   = pset.getUntrackedParameter<double>("AlphaBarrel_BinMin"   ,  -0.2);
  resAlphaBarrel_binWidth = pset.getUntrackedParameter<double>("AlphaBarrel_BinWidth" ,   0.1);
  resAlphaBarrel_binN     = pset.getUntrackedParameter<int>(   "AlphaBarrel_BinN"     ,   4  );
  resBetaBarrel_binMin    = pset.getUntrackedParameter<double>("BetaBarrel_BinMin"    ,   0.0);
  resBetaBarrel_binWidth  = pset.getUntrackedParameter<double>("BetaBarrel_BinWidth"  ,   0.2);
  resBetaBarrel_binN      = pset.getUntrackedParameter<int>(   "BetaBarrel_BinN"      ,   7  );
  // Resolution Forward
  thePixelForwardResolutionFileName = pset.getUntrackedParameter<std::string>( "PixelForwardResolutionFile" ,
									       "FastSimulation/TrackingRecHitProducer/data/PixelForwardResolution.root" );
  resAlphaForward_binMin   = pset.getUntrackedParameter<double>("AlphaForward_BinMin"   ,  0.0);
  resAlphaForward_binWidth = pset.getUntrackedParameter<double>("AlphaForward_BinWidth" ,  0.0);
  resAlphaForward_binN     = pset.getUntrackedParameter<int>(   "AlphaBarrel_BinN"      ,  0  );
  resBetaForward_binMin    = pset.getUntrackedParameter<double>("BetaForward_BinMin"    ,  0.0);
  resBetaForward_binWidth  = pset.getUntrackedParameter<double>("BetaForward_BinWidth"  ,  0.0);
  resBetaForward_binN      = pset.getUntrackedParameter<int>(   "BetaForward_BinN"      ,  0);
  // root files
  thePixelMultiplicityFile      = new TFile ( edm::FileInPath( thePixelMultiplicityFileName      ).fullPath().c_str() , "READ" );
  thePixelBarrelResolutionFile  = new TFile ( edm::FileInPath( thePixelBarrelResolutionFileName  ).fullPath().c_str() , "READ" );
  thePixelForwardResolutionFile = new TFile ( edm::FileInPath( thePixelForwardResolutionFileName ).fullPath().c_str() , "READ" );
}

void FamosRecHitAnalysis::beginJob(const edm::EventSetup& setup) {
  // Initialize the Tracker Geometry
  edm::ESHandle<TrackerGeometry> theGeometry;
  setup.get<TrackerDigiGeometryRecord> ().get (theGeometry);
  geometry = &(*theGeometry);
  //
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
  bookEnergyLosses( histos_TIB_dedx, 200, 0.001, "TIB", nHist_TIB );
  // TID
  bookValues( histos_TID_x , histos_TID_y , histos_TID_z , nbin , minmax , "TID" , nHist_TID );
  bookErrors( histos_TID_err_x , histos_TID_err_y , histos_TID_err_z , 500 , 0.0500 , "TID" , nHist_TID );
  bookNominals( histos_TID_nom_x , nbin , minmax , "TID" , nHist_TID );
  bookEnergyLosses( histos_TID_dedx, 200, 0.001, "TID", nHist_TID );
  // TOB
  bookValues( histos_TOB_x , histos_TOB_y , histos_TOB_z , nbin , minmax , "TOB" , nHist_TOB );
  bookErrors( histos_TOB_err_x , histos_TOB_err_y , histos_TOB_err_z , 500 , 0.0500 , "TOB" , nHist_TOB );
  bookNominals( histos_TOB_nom_x , nbin , minmax , "TOB" , nHist_TOB );
  bookEnergyLosses( histos_TOB_dedx, 200, 0.002, "TOB", nHist_TOB );
  // TEC
  bookValues( histos_TEC_x , histos_TEC_y , histos_TEC_z , nbin , minmax , "TEC" , nHist_TEC );
  bookErrors( histos_TEC_err_x , histos_TEC_err_y , histos_TEC_err_z , 500 , 0.0500 , "TEC" , nHist_TEC );
  bookNominals( histos_TEC_nom_x , nbin , minmax , "TEC" , nHist_TEC );
  bookEnergyLosses( histos_TEC_dedx, 200, 0.002, "TEC", nHist_TEC );
  //
  
  // special Analysis of pixels
  loadPixelData(thePixelMultiplicityFile, thePixelBarrelResolutionFile, thePixelForwardResolutionFile);
  //
  bookPixel( histos_PXB_alpha , histos_PXB_beta , histos_PXB_nom_alpha , histos_PXB_nom_beta ,
             histos_PXB_dedx_alpha, histos_PXB_dedx_beta,
             "PXB" );
  bookPixel( histos_PXF_alpha , histos_PXF_beta , histos_PXF_nom_alpha , histos_PXF_nom_beta ,
             histos_PXF_dedx_alpha, histos_PXF_dedx_beta,
             "PXF" );
  bookPixel( histos_PXB_res_alpha , histos_PXB_res_beta , histos_PXB_nom_res_alpha , histos_PXB_nom_res_beta ,
             "PXB" ,
             nAlphaBarrel , resAlphaBarrel_binMin , resAlphaBarrel_binWidth , resAlphaBarrel_binN ,
             nBetaBarrel  , resBetaBarrel_binMin  , resBetaBarrel_binWidth  , resBetaBarrel_binN   );
  bookPixel( histos_PXF_res_alpha , histos_PXF_res_beta , histos_PXF_nom_res_alpha , histos_PXF_nom_res_beta ,
             "PXF" ,
             nAlphaForward , resAlphaForward_binMin , resAlphaForward_binWidth , resAlphaForward_binN ,
             nBetaForward  , resBetaForward_binMin  , resBetaForward_binWidth  , resBetaForward_binN   );
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

void FamosRecHitAnalysis::bookEnergyLosses(std::vector<TH1F*>& histos_x , int nBin, float range, char* det, unsigned int nHist) {
  //
  for(unsigned int iHist = 0; iHist < nHist; iHist++) {
    histos_x.push_back(            new TH1F(Form( "hist_%s_%u_dedx" , det , iHist+1 ) ,
                                            Form( "Sim Hit energy loss dE/dx %s %u;dE/dx;Entries/bin" , det , iHist+1 ) ,
                                            nBin , 0.0 ,  range ));
  }
  //
}

void FamosRecHitAnalysis::loadPixelData(TFile* pixelMultiplicityFile, TFile* pixelBarrelResolutionFile, TFile* pixelForwardResolutionFile) {
  // Special Pixel histos
  // alpha barrel
  loadPixelData( pixelMultiplicityFile, nAlphaBarrel  , std::string("hist_alpha_barrel")  , histos_PXB_nom_alpha );
  loadPixelData( pixelBarrelResolutionFile, nAlphaBarrel, resAlphaBarrel_binN, resAlphaBarrel_binWidth, histos_PXB_nom_res_alpha, true);
  // beta barrel
  loadPixelData( pixelMultiplicityFile, nBetaBarrel   , std::string("hist_beta_barrel")   , histos_PXB_nom_beta  );
  loadPixelData( pixelBarrelResolutionFile, nBetaBarrel, resBetaBarrel_binN, resBetaBarrel_binWidth, histos_PXB_nom_res_beta, false);
  // 
  // alpha forward
  loadPixelData( pixelMultiplicityFile, nAlphaForward , std::string("hist_alpha_forward") , histos_PXF_nom_alpha );
  loadPixelData( pixelForwardResolutionFile, nAlphaForward, resAlphaForward_binN, resAlphaForward_binWidth, histos_PXF_nom_res_alpha, true);
  // 
  // beta forward
  loadPixelData( pixelMultiplicityFile, nBetaForward  , std::string("hist_beta_forward")  , histos_PXF_nom_beta  );
  loadPixelData( pixelForwardResolutionFile, nBetaForward, resBetaForward_binN, resBetaForward_binWidth, histos_PXF_nom_res_beta, false);
  //
  //
}


void FamosRecHitAnalysis::loadPixelData( TFile* pixelDataFile, unsigned int nMultiplicity, std::string histName,
					 std::vector<TH1F*>& theMultiplicityProbabilities ) {
  std::string histName_i = histName + "_%u"; // needed to open histograms with a for
  theMultiplicityProbabilities.clear();
  //
  std::vector<double> mult; // vector with fixed multiplicity
  for(unsigned int i = 0; i<nMultiplicity; i++) {
    TH1F addHist = *((TH1F*) pixelDataFile->Get( Form( histName_i.c_str() ,i+1 )));
    theMultiplicityProbabilities.push_back( new TH1F(addHist) );
  }
  
#ifdef rrDEBUG
  std::cout << " Multiplicity probability " << histName << std::endl;
  for(unsigned int iMult = 0; iMult<theMultiplicityProbabilities.size(); iMult++) {
    for(int iBin = 1; iBin<=theMultiplicityProbabilities[iMult]->GetNbinsX(); iBin++) {
      std::cout << " Multiplicity " << iMult+1 << " bin " << iBin << " low edge = " << theMultiplicityProbabilities[iMult]->GetBinLowEdge(iBin)
		<< " prob = " << (theMultiplicityProbabilities[iMult])->GetBinContent(iBin) // remember in ROOT bin starts from 1 (0 underflow, nBin+1 overflow)
		<< std::endl;
    }
  }
#endif
  //
}

void FamosRecHitAnalysis::loadPixelData( TFile* pixelDataFile, unsigned int nMultiplicity, int nBins, double binWidth,
					 std::vector<TH1F*>& theResolutionHistograms, bool isAlpha) {
  //
  // resolutions
#ifdef rrDEBUG
      std::cout << " Load resolution histogram from file " << pixelDataFile->GetName()
		<< " multiplicities " << nMultiplicity << " bins " << nBins << std::endl;
#endif
  for(unsigned int iMult = 0; iMult<nMultiplicity; iMult++) {
    for(int iBin = -1; iBin<nBins; iBin++) {
      if(iBin<0) iBin++; // to avoid skip loop if nBins==0
      unsigned int histN = 0;
      if(isAlpha) {
	histN = ( binWidth != 0 ?
		  100 * (iBin+1)
		  + 10
		  + (iMult+1)
		  :
		  1110
		  + (iMult+1) );
      } else {
	histN = ( binWidth != 0 ?
		  100 * (iBin+1)
		  + (iMult+1)
		  :
		  1100 + (iMult+1) );
      }
      //
      TH1F hist = *(TH1F*) pixelDataFile->Get(  Form( "h%u" , histN ) );
      theResolutionHistograms.push_back( new TH1F(hist) );
#ifdef rrDEBUG
      std::cout << " Load resolution histogram " << hist.GetName() << " multiplicity " << iMult+1 << " bin " << iBin+1 << std::endl;
#endif
    }
  }
}



void FamosRecHitAnalysis::bookPixel( std::vector<TH1F*>& histos_alpha , std::vector<TH1F*>& histos_beta ,
				     std::vector<TH1F*>& histos_nom_alpha  , std::vector<TH1F*>& histos_nom_beta ,
                                     std::vector<TH1F*>& histos_dedx_alpha , std::vector<TH1F*>& histos_dedx_beta,
				     char* det ) {
  //
  for(unsigned int iHist = 0; iHist < histos_nom_alpha.size(); iHist++) {
    histos_alpha.push_back( new TH1F(Form( "hist_%s_%u_prob_alpha" , det , iHist+1 ) ,
				     Form( "Hit Local Position angle #alpha^{Rec} %s (multiplicity %u) probability;#alpha^{Rec} [rad];Probability" , det , iHist+1 ) ,
				     histos_nom_alpha[iHist]->GetNbinsX() , histos_nom_alpha[iHist]->GetXaxis()->GetXmin() , histos_nom_alpha[iHist]->GetXaxis()->GetXmax() ));
    histos_dedx_alpha.push_back( new TH1F(Form( "hist_%s_%u_dedx_alpha" , det , iHist+1 ) ,
                                    Form( "Sim Hit energy loss dE/dx %s %u;dE/dx;Entries/bin" , det , iHist+1 ) ,
                                    200, 0, 0.001 ));
    // change name to the nominal one
    histos_nom_alpha[iHist]->SetTitle( Form( "Hit Local Position angle #alpha^{Sim} %s (multiplicity %u) probability;#alpha^{Sim} [rad];Probability" , det , iHist+1 ) );
    histos_nom_alpha[iHist]->SetName(  Form( "hist_%s_%u_nom_prob_alpha" , det , iHist+1 ) );
#ifdef rrDEBUG
    for(int iBin = 1; iBin<=histos_nom_alpha[iHist]->GetNbinsX(); iBin++ )
      std::cout << "\tNominal Probability " << histos_nom_alpha[iHist]->GetName() << " bin " << iBin << " is " << histos_nom_alpha[iHist]->GetBinContent(iBin) << std::endl;
#endif
  }
  //
  //
  for(unsigned int iHist = 0; iHist < histos_nom_beta.size(); iHist++) {
    histos_beta.push_back( new TH1F(Form( "hist_%s_%u_prob_beta" , det , iHist+1 ) ,
				    Form( "Hit Local Position angle #beta^{Rec} %s (multiplicity %u) probability;#beta^{Rec} [rad];Probability" , det , iHist+1 ) ,
				    histos_nom_beta[iHist]->GetNbinsX() , histos_nom_beta[iHist]->GetXaxis()->GetXmin() , histos_nom_beta[iHist]->GetXaxis()->GetXmax() ));
    histos_dedx_beta.push_back( new TH1F(Form( "hist_%s_%u_dedx_beta" , det , iHist+1 ) ,
                                    Form( "Sim Hit energy loss dE/dx %s %u;dE/dx;Entries/bin" , det , iHist+1 ) ,
                                    200, 0, 0.001 ));
    // change name to the nominal one
    histos_nom_beta[iHist]->SetTitle( Form( "Hit Local Position angle #beta^{Sim} %s (multiplicity %u) probability;#beta^{Sim} [rad];Probability" , det , iHist+1 ) );
    histos_nom_beta[iHist]->SetName(  Form( "hist_%s_%u_nom_prob_beta" , det , iHist+1 ) );
#ifdef rrDEBUG
    for(int iBin = 1; iBin<=histos_nom_beta[iHist]->GetNbinsX(); iBin++ )
      std::cout << "\tNominal Probability " << histos_nom_beta[iHist]->GetName() << " bin " << iBin << " is " << histos_nom_beta[iHist]->GetBinContent(iBin) << std::endl;
#endif
  }
  //
}

void FamosRecHitAnalysis::bookPixel( std::vector<TH1F*>& histos_alpha , std::vector<TH1F*>& histos_beta ,
				     std::vector<TH1F*>& histos_nom_alpha  , std::vector<TH1F*>& histos_nom_beta ,
				     char* det ,
				     unsigned int nAlphaMultiplicity ,
				     double resAlpha_binMin , double resAlpha_binWidth , int resAlpha_binN , 
				     unsigned int nBetaMultiplicity ,
				     double resBeta_binMin  , double resBeta_binWidth  , int resBeta_binN    ) {
  // resolutions
#ifdef rrDEBUG
  std::cout << " Book resolution histogram "
	    << " alpha multiplicities " << nAlphaMultiplicity << " bins " << resAlpha_binN
	    << "\tbeta multiplicities " << nBetaMultiplicity  << " bins " << resBeta_binN
	    << std::endl;
#endif
  int iHist = -1; // count the histograms in the vector
  for(unsigned int iMult = 0; iMult<nAlphaMultiplicity; iMult++) {
    for(int iBin = -1; iBin<resAlpha_binN; iBin++) {
      if(iBin<0) iBin++; // to avoid skip loop if nBins==0
      iHist++;
      float binMin = ( resAlpha_binWidth!=0 ? resAlpha_binMin+(float)(iBin)*resAlpha_binWidth   : -PI );
      float binMax = ( resAlpha_binWidth!=0 ? resAlpha_binMin+(float)(iBin+1)*resAlpha_binWidth :  PI );
      histos_alpha.push_back(new TH1F(Form( "hist_%s_%u_%u_res_alpha" , det , iMult+1 , iBin+1 ) ,
				      Form( "Hit Local Position x^{Rec} [%f<#alpha^{Rec}<%f] %s (multiplicity %u);x [cm];Events/bin" ,
					    binMin , binMax ,
					    det , iMult+1 ) ,
				      histos_nom_alpha[iHist]->GetNbinsX() , histos_nom_alpha[iHist]->GetXaxis()->GetXmin() , histos_nom_alpha[iHist]->GetXaxis()->GetXmax() ) );
      // change name to the nominal one
      histos_nom_alpha[iHist]->SetName(  Form( "hist_%s_%u_%u_nom_res_alpha" , det , iMult+1 , iBin+1 ) );
      histos_nom_alpha[iHist]->SetTitle( Form( "Hit Local Position x^{Sim} [%f<#alpha^{Sim}<%f] %s (multiplicity %u);x [cm];Events/bin" ,
					       binMin , binMax ,
					       det , iMult+1 ) );
    }
  }
  
  iHist = -1; // count the histograms in the vector again
  for(unsigned int iMult = 0; iMult<nBetaMultiplicity; iMult++) {
    for(int iBin = -1; iBin<resBeta_binN; iBin++) {
      if(iBin<0) iBin++; // to avoid skip loop if nBins==0
      iHist++;
      histos_beta.push_back(new TH1F(Form( "hist_%s_%u_%u_res_beta" , det , iMult+1 , iBin+1 ) ,
				     Form( "Hit Local Position y^{Rec} [%f<#beta^{Rec}<%f] %s (multiplicity %u);y [cm];Events/bin" ,
					   resBeta_binMin+(float)(iBin)*resBeta_binWidth , resBeta_binMin+(float)(iBin+1)*resBeta_binWidth ,
					   det , iMult+1 ) ,
				     histos_nom_beta[iHist]->GetNbinsX() , histos_nom_beta[iHist]->GetXaxis()->GetXmin() , histos_nom_beta[iHist]->GetXaxis()->GetXmax() ) );
      // change name to the nominal one
      histos_nom_beta[iHist]->SetName(  Form( "hist_%s_%u_%u_nom_res_beta" , det , iMult+1 , iBin+1 ) );
      histos_nom_beta[iHist]->SetTitle( Form( "Hit Local Position y^{Sim} [%f<#beta^{Sim}<%f] %s (multiplicity %u);y [cm];Events/bin" ,
					      resBeta_binMin+(float)(iBin)*resBeta_binWidth , resBeta_binMin+(float)(iBin+1)*resBeta_binWidth ,
					      det , iMult+1 ) );
    }
  }
  
#ifdef rrDEBUG
  std::cout << " Resolution histogram booked "
	    << " alpha " << histos_alpha.size()
	    << "\tbeta " << histos_beta.size()
	    << std::endl;
#endif
  
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
#ifdef rrDEBUG
  int t_Run   = event.id().run();
  int t_Event = event.id().event();
  std::cout
    << " #################################### Run " << t_Run 
    << " Event "                                    << t_Event << " #################################### " 
    << std::endl;
#endif
  //

  // Get PSimHit's of the Event
  
  edm::Handle<CrossingFrame<PSimHit> > cf_simhit; 
  std::vector<const CrossingFrame<PSimHit> *> cf_simhitvec;
  for(uint32_t i=0; i<trackerContainers.size(); i++){
    event.getByLabel("mix",trackerContainers[i], cf_simhit);
    cf_simhitvec.push_back(cf_simhit.product());
  }
  std::auto_ptr<MixCollection<PSimHit> > allTrackerHits(new MixCollection<PSimHit>(cf_simhitvec));
  
  // RecHits
#ifdef rrDEBUG
  std::cout << "Famos RecHits" << std::endl;
#endif
  edm::Handle<SiTrackerFullGSRecHit2DCollection> theRecHits;
  event.getByLabel(theRecHits_, theRecHits);
  
  // histograms to fill
  TH1F* hist_x = 0;
  TH1F* hist_y = 0;
  TH1F* hist_z = 0;
  TH1F* hist_err_x = 0;
  TH1F* hist_err_y = 0;
  TH1F* hist_err_z = 0;
  TH1F* hist_alpha = 0;
  TH1F* hist_beta  = 0;
  TH1F* hist_res_alpha = 0;
  TH1F* hist_res_beta  = 0;
  TH1F* hist_dedx = 0;
  TH1F* hist_dedx_alpha = 0;
  TH1F* hist_dedx_beta = 0;
  //
  
  // loop on RecHits
  unsigned int iRecHit = 0;
  const std::vector<DetId> theDetIds = theRecHits->ids();
  // loop over detunits
  for ( std::vector<DetId>::const_iterator iDetId = theDetIds.begin(); iDetId != theDetIds.end(); iDetId++ ) {
    unsigned int detid = (*iDetId).rawId();
    if(detid!=999999999){ // valid detector
      SiTrackerFullGSRecHit2DCollection::range theRecHitRange = theRecHits->get((*iDetId));
      SiTrackerFullGSRecHit2DCollection::const_iterator theRecHitRangeIteratorBegin = theRecHitRange.first;
      SiTrackerFullGSRecHit2DCollection::const_iterator theRecHitRangeIteratorEnd   = theRecHitRange.second;
      SiTrackerFullGSRecHit2DCollection::const_iterator iterRecHit = theRecHitRangeIteratorBegin;
      // loop over RecHits of the same detector
      for(iterRecHit = theRecHitRangeIteratorBegin; iterRecHit != theRecHitRangeIteratorEnd; ++iterRecHit) {
	iRecHit++;
	
	// search the associated original PSimHit
	PSimHit* simHit = NULL;
	int simHitNumber = (*iterRecHit).simhitId();
	int simHitCounter = -1;
	for (MixCollection<PSimHit>::iterator isim=(*allTrackerHits).begin(); isim!= (*allTrackerHits).end(); isim++) {
	  simHitCounter++;
	  if(simHitCounter == simHitNumber) {
	    simHit = const_cast<PSimHit*>(&(*isim));
	    break;
	  }
	}    
	//
	float xRec = (*iterRecHit).localPosition().x();
	float yRec = (*iterRecHit).localPosition().y();
	float zRec = (*iterRecHit).localPosition().z();
	float xSim = simHit->localPosition().x();
	float ySim = simHit->localPosition().y();
	float zSim = simHit->localPosition().z();
	float delta_x = xRec - xSim;
	float delta_y = yRec - ySim;
	float delta_z = zRec - zSim;
	float err_x = sqrt((*iterRecHit).localPositionError().xx());
	float err_y = sqrt((*iterRecHit).localPositionError().yy());
	float err_z = 0.0;

        float dedx = simHit->energyLoss();
	
	// CALCULATE alpha AND beta
	// at the beginning the position is the Local Point in the local pixel module reference frame
	// same code as in PixelCPEBase
	LocalVector localDir = simHit->momentumAtEntry().unit();
	float locx = localDir.x();
	float locy = localDir.y();
	float locz = localDir.z();
	// alpha: angle with respect to local x axis in local (x,z) plane
	float alpha = acos(locx/sqrt(locx*locx+locz*locz));
	int subdetid = ((detid>>25)&0x7);
	unsigned int detUnitId = simHit->detUnitId();
	// do it only for Pixels: subdetid = 1 or 2
	if(subdetid == 1 || subdetid==2) {
	  if( isFlipped( dynamic_cast<const PixelGeomDetUnit*>
			 ( geometry->idToDetUnit( DetId( detUnitId ) ) ) ) ) { // &&& check for FPIX !!!
#ifdef rrDEBUG
	    std::cout << " isFlipped " << std::endl;
#endif
	    alpha = PI - alpha ;
	  }
	}
	// beta: angle with respect to local y axis in local (y,z) plane
	float beta = acos(locy/sqrt(locy*locy+locz*locz));
	// TO BE USED TO CHECK ROOTFILES
	// look old FAMOS: FamosGeneric/FamosTracker/src/FamosPixelErrorParametrization
	if(subdetid == 1 ) { // PXB
	  alpha = PI/2. - alpha;
	  beta  = fabs( PI/2. - beta );
	} else if(subdetid == 2 ) { // PXF
	  beta  = PI/2. - beta;
	  alpha = fabs( PI/2. - alpha );
	}
	//
	
	unsigned int mult_alpha = (*iterRecHit).simMultX();
	unsigned int mult_beta  = (*iterRecHit).simMultY();
#ifdef rrDEBUG
	std::cout << "\n\t" << iRecHit << std::endl;
	std::cout << "\tDet Unit id " << detUnitId << " in subdetector " << (*iDetId).subdetId() << std::endl;
	std::cout << "\tRecHit"
		  << "\t\tx = " << xRec << " cm"
		  << "\t\ty = " << yRec << " cm"
		  << "\t\tz = " << zRec << " cm"
		  << std::endl;
	std::cout << "\tSimHit"
		  << "\t\tx = " << xSim << " cm"
		  << "\t\ty = " << ySim << " cm"
		  << "\t\tz = " << zSim << " cm"
                  << "\t\tdedx = " << dedx << " ???"
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
	std::cout << "\tRecHit angles"
		  << "\t\talpha = " << alpha << " rad" << " multiplicity " << mult_alpha
		  << "\t\tbeta  = " << beta  << " rad" << " multiplicity " << mult_beta
		  << std::endl;
#endif
	// fill proper histograms
        chooseHist( detid , hist_x , hist_y , hist_z , hist_err_x , hist_err_y , hist_err_z ,
                    hist_alpha , hist_beta , hist_res_alpha , hist_res_beta , hist_dedx,
                    hist_dedx_alpha, hist_dedx_beta,
                    mult_alpha , mult_beta ,
                    alpha      , beta       );
	//
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
          hist_dedx->Fill( dedx );
	}
	//
	if(hist_alpha != 0 && mult_alpha != 0) {
#ifdef rrDEBUG
	  std::cout << "\tFill histograms " << hist_alpha->GetName()     << ", " << hist_beta->GetName()
		    << " , "                << hist_res_alpha->GetName() << ", " << hist_res_beta->GetName()
                    << " , "                << hist_dedx_alpha->GetName()
                    << " , "                << hist_dedx_beta->GetName()
                    << std::endl;
#endif	  
	  hist_alpha->Fill( alpha );
	  hist_beta->Fill(  beta  );
	  hist_res_alpha->Fill( delta_x );
	  hist_res_beta->Fill(  delta_y );
          hist_dedx_alpha->Fill( dedx );
          hist_dedx_beta->Fill( dedx );
	}
	//	
      } // loop over RecHits
      //
    } // valid detector
  } // loop over detunits
  
}

void FamosRecHitAnalysis::endJob() {
  //
  theRootFile->cd();
  // before closing file do root macro
  // PXB
  rootMacroPixel( histos_PXB_alpha );
  rootMacroPixel( histos_PXB_beta  );
  // PXF
  rootMacroPixel( histos_PXF_alpha );
  rootMacroPixel( histos_PXF_beta  );
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
  // PXB
  write(histos_PXB_alpha);
  write(histos_PXB_beta);
  write(histos_PXB_dedx_alpha);
  write(histos_PXB_dedx_beta);
  write(histos_PXB_nom_alpha);
  write(histos_PXB_nom_beta);
  write(histos_PXB_res_alpha);
  write(histos_PXB_res_beta);
  write(histos_PXB_nom_res_alpha);
  write(histos_PXB_nom_res_beta);
  // PXF
  write(histos_PXF_alpha);
  write(histos_PXF_beta);
  write(histos_PXF_dedx_alpha);
  write(histos_PXF_dedx_beta);
  write(histos_PXF_nom_alpha);
  write(histos_PXF_nom_beta);
  write(histos_PXF_res_alpha);
  write(histos_PXF_res_beta);
  write(histos_PXF_nom_res_alpha);
  write(histos_PXF_nom_res_beta);
  // TIB
  write(histos_TIB_x);
  write(histos_TIB_y);
  write(histos_TIB_z);
  write(histos_TIB_err_x);
  write(histos_TIB_err_y);
  write(histos_TIB_err_z);
  write(histos_TIB_nom_x);
  write(histos_TIB_dedx);
  // TID
  write(histos_TID_x);
  write(histos_TID_y);
  write(histos_TID_z);
  write(histos_TID_err_x);
  write(histos_TID_err_y);
  write(histos_TID_err_z);
  write(histos_TID_nom_x);
  write(histos_TID_dedx);
  // TOB
  write(histos_TOB_x);
  write(histos_TOB_y);
  write(histos_TOB_z);
  write(histos_TOB_err_x);
  write(histos_TOB_err_y);
  write(histos_TOB_err_z);
  write(histos_TOB_nom_x);
  write(histos_TOB_dedx);
  // TEC
  write(histos_TEC_x);
  write(histos_TEC_y);
  write(histos_TEC_z);
  write(histos_TEC_err_x);
  write(histos_TEC_err_y);
  write(histos_TEC_err_z);
  write(histos_TEC_nom_x);
  write(histos_TEC_dedx);
  //
  rootComparison( histos_PXB_alpha , histos_PXB_nom_alpha , -1 , 0 , 1.05 );
  rootComparison( histos_PXB_beta  , histos_PXB_nom_beta  , -1 , 0 , 1.05 );
  rootComparison( histos_PXF_alpha , histos_PXF_nom_alpha , -1 , 0 , 1.05 );
  rootComparison( histos_PXF_beta  , histos_PXF_nom_beta  , -1 , 0 , 1.05 );
  rootComparison( histos_PXB_res_alpha , histos_PXB_nom_res_alpha , 20 );
  rootComparison( histos_PXB_res_beta  , histos_PXB_nom_res_beta  , 20 );
  rootComparison( histos_PXF_res_alpha , histos_PXF_nom_res_alpha , 20 );
  rootComparison( histos_PXF_res_beta  , histos_PXF_nom_res_beta  , 20 );
  //
  rootComparison( histos_TIB_x , histos_TIB_nom_x , 20 );
  rootComparison( histos_TID_x , histos_TID_nom_x , 20 );
  rootComparison( histos_TOB_x , histos_TOB_nom_x , 40 );
  rootComparison( histos_TEC_x , histos_TEC_nom_x , 40 );
  //
  theRootFile->Close();
  //
}

//
void FamosRecHitAnalysis::chooseHist( unsigned int rawid ,
                                      TH1F*& hist_x , TH1F*& hist_y , TH1F*& hist_z, TH1F*& hist_err_x , TH1F*& hist_err_y , TH1F*& hist_err_z ,
                                      TH1F*& hist_alpha , TH1F*& hist_beta , TH1F*& hist_res_alpha , TH1F*& hist_res_beta , TH1F*& hist_dedx, 
                                      TH1F*& hist_dedx_alpha, TH1F*& hist_dedx_beta,
                                      unsigned int mult_alpha , unsigned int mult_beta ,
                                      double       alpha      , double       beta       ) {
  int subdetid = ((rawid>>25)&0x7);
  
  switch (subdetid) {
    // Pixel Barrel
  case 1:
    // PXB
    {
      PXBDetId module(rawid);
      hist_alpha       = histos_PXB_alpha[mult_alpha-1];
      hist_beta        = histos_PXB_beta[mult_beta-1];
      hist_dedx_alpha  = histos_PXB_dedx_alpha[mult_alpha-1];
      hist_dedx_beta   = histos_PXB_dedx_beta[mult_beta-1];
#ifdef rrDEBUG
      unsigned int theLayer = module.layer();
      std::cout << "\tTracker subdetector " << subdetid << " PXB Layer " << theLayer << std::endl;
#endif
  //
  // resolution
  // search bin (resolution histograms)
      int iAlphaHist = -1;
      if(resAlphaBarrel_binN!=0) {
	for(unsigned int iBin = 1; iBin<=resAlphaBarrel_binN; iBin++) {
	  double binMin = resAlphaBarrel_binMin+(double)(iBin-1)*resAlphaBarrel_binWidth;
	  double binMax = resAlphaBarrel_binMin+(double)(iBin)*resAlphaBarrel_binWidth;
	  if( alpha >= binMin && alpha < binMax ) iAlphaHist = ( (mult_alpha-1)*resAlphaBarrel_binN + iBin ) - 1;
	}
      } else {
	iAlphaHist = mult_alpha - 1;
      }
      if( iAlphaHist==-1 ) {
	double binMin = resAlphaBarrel_binMin;
	double binMax = resAlphaBarrel_binMin+(double)(resAlphaBarrel_binN)*resAlphaBarrel_binWidth;
	if( alpha < binMin)  iAlphaHist = ( (mult_alpha-1)*resAlphaBarrel_binN + 1 ) - 1; // underflow
	if( alpha >= binMax) iAlphaHist = ( (mult_alpha-1)*resAlphaBarrel_binN + resAlphaBarrel_binN ) - 1; // overflow
      }
      //
      int iBetaHist = -1;
      if(resBetaBarrel_binN!=0) {
	for(unsigned int iBin = 1; iBin<=resBetaBarrel_binN; iBin++) {
	  double binMin = resBetaBarrel_binMin+(double)(iBin-1)*resBetaBarrel_binWidth;
	  double binMax = resBetaBarrel_binMin+(double)(iBin)*resBetaBarrel_binWidth;
	  if( beta >= binMin && beta < binMax ) iBetaHist = ( (mult_beta-1)*resBetaBarrel_binN + iBin ) - 1;
	}
      } else {
	iBetaHist = mult_beta - 1;
      }
      if( iBetaHist==-1 ) {
	double binMin = resBetaBarrel_binMin;
	double binMax = resBetaBarrel_binMin+(double)(resBetaBarrel_binN)*resBetaBarrel_binWidth;
	if( beta < binMin)  iBetaHist = ( (mult_beta-1)*resBetaBarrel_binN + 1 ) - 1; // underflow
	if( beta >= binMax) iBetaHist = ( (mult_beta-1)*resBetaBarrel_binN + resBetaBarrel_binN ) - 1; // overflow
      }
      //
#ifdef rrDEBUG
      std::cout << "\tResolution histos chosen alpha " << iAlphaHist << " beta " << iBetaHist << std::endl;
#endif
      hist_res_alpha = histos_PXB_res_alpha[iAlphaHist];
      hist_res_beta  = histos_PXB_res_beta[iBetaHist];
      //
      break;
    } 
    //
  case 2:
    // PXF
    {
      PXFDetId module(rawid);
      hist_alpha       = histos_PXF_alpha[mult_alpha-1];
      hist_beta        = histos_PXF_beta[mult_beta-1];
      hist_dedx_alpha  = histos_PXF_dedx_alpha[mult_alpha-1];
      hist_dedx_beta   = histos_PXF_dedx_beta[mult_beta-1];
#ifdef rrDEBUG
      unsigned int theDisk = module.disk();
      std::cout << "\tTracker subdetector " << subdetid << " PXF Disk " << theDisk << std::endl;
#endif
      //
  // resolution
  // search bin (resolution histograms)
      int iAlphaHist = -1;
      if(resAlphaForward_binN!=0) {
	for(unsigned int iBin = 1; iBin<=resAlphaForward_binN; iBin++) {
	  double binMin = resAlphaForward_binMin+(double)(iBin-1)*resAlphaForward_binWidth;
	  double binMax = resAlphaForward_binMin+(double)(iBin)*resAlphaForward_binWidth;
	  if( alpha >= binMin && alpha < binMax ) iAlphaHist = ( (mult_alpha-1)*resAlphaForward_binN + iBin ) - 1;
	}
      } else {
	iAlphaHist = mult_alpha - 1;
      }
      if( iAlphaHist==-1 ) {
	double binMin = resAlphaForward_binMin;
	double binMax = resAlphaForward_binMin+(double)(resAlphaForward_binN)*resAlphaForward_binWidth;
	if( alpha < binMin)  iAlphaHist = ( (mult_alpha-1)*resAlphaForward_binN + 1 ) - 1; // underflow
	if( alpha >= binMax) iAlphaHist = ( (mult_alpha-1)*resAlphaForward_binN + resAlphaForward_binN ) - 1; // overflow
      }
      //
      int iBetaHist = -1;
      if(resBetaForward_binN!=0) {
	for(unsigned int iBin = 1; iBin<=resBetaForward_binN; iBin++) {
	  double binMin = resBetaForward_binMin+(double)(iBin-1)*resBetaForward_binWidth;
	  double binMax = resBetaForward_binMin+(double)(iBin)*resBetaForward_binWidth;
	  if( beta >= binMin && beta < binMax ) iBetaHist = ( (mult_beta-1)*resBetaForward_binN + iBin ) - 1;
	}
      } else {
	iBetaHist = mult_beta - 1;
      }
      if( iBetaHist==-1 ) {
	double binMin = resBetaForward_binMin;
	double binMax = resBetaForward_binMin+(double)(resBetaForward_binN)*resBetaForward_binWidth;
	if( beta < binMin)  iBetaHist = ( (mult_beta-1)*resBetaForward_binN + 1 ) - 1; // underflow
	if( beta >= binMax) iBetaHist = ( (mult_beta-1)*resBetaForward_binN + resBetaForward_binN ) - 1; // overflow
      }
      //
#ifdef rrDEBUG
      std::cout << "\tResolution histos chosen alpha " << iAlphaHist << " beta " << iBetaHist << std::endl;
#endif
      hist_res_alpha = histos_PXF_res_alpha[iAlphaHist];
      hist_res_beta  = histos_PXF_res_beta[iBetaHist];
      //
      break;
    } 
    //
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
      hist_dedx = histos_TIB_dedx[theLayer-1];
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
      hist_dedx = histos_TID_dedx[theRing-1];
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
      hist_dedx = histos_TOB_dedx[theLayer-1];
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
      hist_dedx = histos_TEC_dedx[theRing-1];
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
    gaussianResolution->FixParameter( 0 , histos_x[iHist]->GetEntries() / ( sqrt(2*PI) * histos_err_x[iHist]->GetMean()) ); // same normalization
    gaussianResolution->FixParameter( 1 , 0.0 ); // mean = 0.
    gaussianResolution->FixParameter( 2 , histos_err_x[iHist]->GetMean() ); // sigma set to RecHit error
    histos_nom_x[iHist]->FillRandom(gaussianResolution->GetName(),(int)histos_x[iHist]->GetEntries());
    histos_nom_x[iHist]->SetEntries(histos_x[iHist]->GetEntries());
    //
  }
}
//

//
void FamosRecHitAnalysis::rootMacroPixel( std::vector<TH1F*>& histos_angle ) {
  // create probabilities
  TH1F* hist_tot = 0;
  for(unsigned int iHist = 0; iHist < histos_angle.size(); iHist++) {
    if(iHist == 0) hist_tot = (TH1F*)histos_angle[0]->Clone("tot");
    for(int iBin = 0; iBin < histos_angle[0]->GetNbinsX(); iBin++) {
      //
      if(iHist!=0) hist_tot->SetBinContent(iBin+1, hist_tot->GetBinContent(iBin+1)+histos_angle[iHist]->GetBinContent(iBin+1) );
      if(iHist!=0) hist_tot->SetBinError(iBin+1, 0.00 );
#ifdef rrDEBUG
      std::cout << "\tTotal Probability after " << histos_angle[iHist]->GetName() << " bin " << iBin << " is " << hist_tot->GetBinContent(iBin+1) << std::endl;
#endif
    }
  }
  
  for(unsigned int iHist = 0; iHist < histos_angle.size(); iHist++) {
    histos_angle[iHist]->Divide(hist_tot);
  }
  
#ifdef rrDEBUG
  for(unsigned int iMult = 0; iMult<histos_angle.size(); iMult++) {
    std::cout << " Multiplicity probability " << histos_angle[iMult]->GetName() << std::endl;
    for(int iBin = 1; iBin<=histos_angle[iMult]->GetNbinsX(); iBin++) {
      std::cout << " Multiplicity " << iMult+1 << " bin " << iBin << " low edge = " << histos_angle[iMult]->GetBinLowEdge(iBin)
		<< " prob = " << (histos_angle[iMult])->GetBinContent(iBin) // remember in ROOT bin starts from 1 (0 underflow, nBin+1 overflow)
		<< std::endl;
    }
  }
#endif
  
}
//

//
void FamosRecHitAnalysis::rootComparison( std::vector<TH1F*> histos_value , std::vector<TH1F*> histos_nominal ,
					  int binFactor , int yLogScale , float yMax) {
  //
  for(unsigned int iHist = 0; iHist < histos_value.size(); iHist++) {
    // canvas
    TCanvas can_comparison("can_comparison","can_comparison",800,800);
    can_comparison.Range(0,0,25,25);
    can_comparison.SetFillColor(kWhite);
    can_comparison.SetGridy(1);
    can_comparison.SetLogy(yLogScale);
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
    if(binFactor!=-1) {
      histos_value[iHist]->Rebin(binFactor);
      histos_nominal[iHist]->Rebin(binFactor);
      histos_value[iHist]->GetXaxis()->SetRangeUser( histos_value[iHist]->GetMean() - 3 * histos_value[iHist]->GetRMS() , 
						     histos_value[iHist]->GetMean() + 3 * histos_value[iHist]->GetRMS() );
      // normalise entries of nominal histo to value histo (useful for pixel)
      if(histos_nominal[iHist]->GetEntries()!=0) histos_nominal[iHist]->Scale(histos_value[iHist]->GetEntries()/histos_nominal[iHist]->GetEntries());
    }
    //
    // Draw
    if(yMax != -1) histos_nominal[iHist]->SetMaximum(yMax);
    histos_nominal[iHist]->Draw("HIST");
    histos_value[iHist]->Draw("HIST P E1 SAME");
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
    //
  }
  //
}

//
//-----------------------------------------------------------------------------
// I COPIED FROM THE PixelCPEBase BECAUSE IT'S BETTER THAN REINVENT IT
// The isFlipped() is a silly way to determine which detectors are inverted.
// In the barrel for every 2nd ladder the E field direction is in the
// global r direction (points outside from the z axis), every other
// ladder has the E field inside. Something similar is in the 
// forward disks (2 sides of the blade). This has to be recognised
// because the charge sharing effect is different.
//
// The isFliped does it by looking and the relation of the local (z always
// in the E direction) to global coordinates. There is probably a much 
// better way.(PJ: And faster!)
//-----------------------------------------------------------------------------
bool FamosRecHitAnalysis::isFlipped(const PixelGeomDetUnit* theDet) const {
  // Check the relative position of the local +/- z in global coordinates.
  float tmp1 = theDet->surface().toGlobal(Local3DPoint(0.,0.,0.)).perp();
  float tmp2 = theDet->surface().toGlobal(Local3DPoint(0.,0.,1.)).perp();
  //  std::cout << " 1: " << tmp1 << " 2: " << tmp2 << std::endl;
  if ( tmp2<tmp1 ) return true;
  else return false;    
}
// 

//
DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(FamosRecHitAnalysis);
