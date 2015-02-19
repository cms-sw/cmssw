#include <memory>

// Framework
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
//

// Geometry
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/RectangularPixelTopology.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
//

// PSimHits
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
//

// RecHits
#include "DataFormats/TrackerRecHit2D/interface/SiTrackerGSRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiTrackerGSRecHit2DCollection.h"
#include "DataFormats/Common/interface/OwnVector.h" 
//

// ROOT
#include <TStyle.h>
#include <TGaxis.h>
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

//#define rrDEBUG

FamosRecHitAnalysis::FamosRecHitAnalysis(edm::ParameterSet const& pset) : 
  _pset(pset),
  theRecHits_( pset.getParameter<edm::InputTag>("RecHits") )
{
#ifdef rrDEBUG
  std::cout << "Start Famos RecHit Analysis" << std::endl;
#endif
  // Switch between old (ORCA) and new (CMSSW) pixel parameterization
  useCMSSWPixelParameterization = pset.getParameter<bool>("UseCMSSWPixelParametrization");
#ifdef FAMOS_DEBUG
  std::cout << (useCMSSWPixelParameterization? "CMSSW" : "ORCA") << " pixel parametrization chosen in config file." << std::endl;
#endif
  //--- PSimHit Containers
  trackerContainers.clear();
  trackerContainers = pset.getParameter<std::vector<edm::InputTag> >("ROUList");
  //
  if(useCMSSWPixelParameterization)
  {
    thePixelMultiplicityFileName = pset.getParameter<std::string>( "PixelMultiplicityFileNew" );
    nAlphaBarrel  = pset.getParameter<int>("AlphaBarrelMultiplicityNew"  );
    nBetaBarrel   = pset.getParameter<int>("BetaBarrelMultiplicityNew"   );
    nAlphaForward = pset.getParameter<int>("AlphaForwardMultiplicityNew" );
    nBetaForward  = pset.getParameter<int>("BetaForwardMultiplicityNew"  );
  }
  else
  {
    thePixelMultiplicityFileName = pset.getParameter<std::string>( "PixelMultiplicityFile" );
    nAlphaBarrel  = pset.getParameter<int>("AlphaBarrelMultiplicity"  );
    nBetaBarrel   = pset.getParameter<int>("BetaBarrelMultiplicity"   );
    nAlphaForward = pset.getParameter<int>("AlphaForwardMultiplicity" );
    nBetaForward  = pset.getParameter<int>("BetaForwardMultiplicity"  );
  }
  // Resolution Barrel    
  if(useCMSSWPixelParameterization)
  {
    thePixelBarrelResolutionFileName = pset.getParameter<std::string>( "PixelBarrelResolutionFileNew" );
    resAlphaBarrel_binMin   = pset.getParameter<double>("AlphaBarrel_BinMinNew"   );
    resAlphaBarrel_binWidth = pset.getParameter<double>("AlphaBarrel_BinWidthNew" );
    resAlphaBarrel_binN     = pset.getParameter<int>(   "AlphaBarrel_BinNNew"     );
    resBetaBarrel_binMin    = pset.getParameter<double>("BetaBarrel_BinMinNew"    );
    resBetaBarrel_binWidth  = pset.getParameter<double>("BetaBarrel_BinWidthNew"  );
    resBetaBarrel_binN      = pset.getParameter<int>(   "BetaBarrel_BinNNew"      );
  }
  else
  {
    thePixelBarrelResolutionFileName = pset.getParameter<std::string>( "PixelBarrelResolutionFile" );
    resAlphaBarrel_binMin   = pset.getParameter<double>("AlphaBarrel_BinMin"   );
    resAlphaBarrel_binWidth = pset.getParameter<double>("AlphaBarrel_BinWidth" );
    resAlphaBarrel_binN     = pset.getParameter<int>(   "AlphaBarrel_BinN"     );
    resBetaBarrel_binMin    = pset.getParameter<double>("BetaBarrel_BinMin"    );
    resBetaBarrel_binWidth  = pset.getParameter<double>("BetaBarrel_BinWidth"  );
    resBetaBarrel_binN      = pset.getParameter<int>(   "BetaBarrel_BinN"      );
  }
  // Resolution Forward
  if(useCMSSWPixelParameterization)
  {
    thePixelForwardResolutionFileName = pset.getParameter<std::string>( "PixelForwardResolutionFileNew" );
    resAlphaForward_binMin   = pset.getParameter<double>("AlphaForward_BinMinNew"   );
    resAlphaForward_binWidth = pset.getParameter<double>("AlphaForward_BinWidthNew" );
    resAlphaForward_binN     = pset.getParameter<int>(   "AlphaBarrel_BinNNew"      );
    resBetaForward_binMin    = pset.getParameter<double>("BetaForward_BinMinNew"    );
    resBetaForward_binWidth  = pset.getParameter<double>("BetaForward_BinWidthNew"  );
    resBetaForward_binN      = pset.getParameter<int>(   "BetaForward_BinNNew"      );
  }
  else
  {
    thePixelForwardResolutionFileName = pset.getParameter<std::string>( "PixelForwardResolutionFile" );
    resAlphaForward_binMin   = pset.getParameter<double>("AlphaForward_BinMin"   );
    resAlphaForward_binWidth = pset.getParameter<double>("AlphaForward_BinWidth" );
    resAlphaForward_binN     = pset.getParameter<int>(   "AlphaBarrel_BinN"      );
    resBetaForward_binMin    = pset.getParameter<double>("BetaForward_BinMin"    );
    resBetaForward_binWidth  = pset.getParameter<double>("BetaForward_BinWidth"  );
    resBetaForward_binN      = pset.getParameter<int>(   "BetaForward_BinN"      );
  }
  // root files
  thePixelMultiplicityFile      = new TFile ( edm::FileInPath( thePixelMultiplicityFileName      ).fullPath().c_str() , "READ" );
  thePixelBarrelResolutionFile  = new TFile ( edm::FileInPath( thePixelBarrelResolutionFileName  ).fullPath().c_str() , "READ" );
  thePixelForwardResolutionFile = new TFile ( edm::FileInPath( thePixelForwardResolutionFileName ).fullPath().c_str() , "READ" );
}

void FamosRecHitAnalysis::beginRun(edm::Run const&, const edm::EventSetup& setup) {
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
  char det[100]; 
  // TIB
  sprintf  (det, "TIB" );
  bookValues( histos_TIB_x , histos_TIB_y , histos_TIB_z , nbin , minmax , det , nHist_TIB );
  bookErrors( histos_TIB_err_x , histos_TIB_err_y , histos_TIB_err_z , 500 , 0.0500 , det , nHist_TIB );
  bookNominals( histos_TIB_nom_x , nbin , minmax , det , nHist_TIB );
  bookEnergyLosses( histos_TIB_dedx, 200, 0.001, det, nHist_TIB );
  // TID
  sprintf  (det, "TID" );
  bookValues( histos_TID_x , histos_TID_y , histos_TID_z , nbin , minmax , det , nHist_TID );
  bookErrors( histos_TID_err_x , histos_TID_err_y , histos_TID_err_z , 500 , 0.0500 , det , nHist_TID );
  bookNominals( histos_TID_nom_x , nbin , minmax , det , nHist_TID );
  bookEnergyLosses( histos_TID_dedx, 200, 0.001, det , nHist_TID );
  // TOB
  sprintf  (det, "TOB" );
  bookValues( histos_TOB_x , histos_TOB_y , histos_TOB_z , nbin , minmax , det , nHist_TOB );
  bookErrors( histos_TOB_err_x , histos_TOB_err_y , histos_TOB_err_z , 500 , 0.0500 , det , nHist_TOB );
  bookNominals( histos_TOB_nom_x , nbin , minmax , det , nHist_TOB );
  bookEnergyLosses( histos_TOB_dedx, 200, 0.002, det, nHist_TOB );
  // TEC
  sprintf  (det, "TEC" );
  bookValues( histos_TEC_x , histos_TEC_y , histos_TEC_z , nbin , minmax , det , nHist_TEC );
  bookErrors( histos_TEC_err_x , histos_TEC_err_y , histos_TEC_err_z , 500 , 0.0500 , det , nHist_TEC );
  bookNominals( histos_TEC_nom_x , nbin , minmax , det , nHist_TEC );
  bookEnergyLosses( histos_TEC_dedx, 200, 0.002, det, nHist_TEC );
  //
  
  // special Analysis of pixels
  loadPixelData(thePixelMultiplicityFile, thePixelBarrelResolutionFile, thePixelForwardResolutionFile);
  //

  sprintf  (det, "PXB" );
  bookPixel( histos_PXB_alpha , histos_PXB_beta , histos_PXB_nom_alpha , histos_PXB_nom_beta ,
             histos_PXB_dedx_alpha, histos_PXB_dedx_beta,
             det );
  bookPixel( histos_PXB_res_alpha , histos_PXB_res_beta , histos_PXB_nom_res_alpha , histos_PXB_nom_res_beta ,
             det ,
             nAlphaBarrel , resAlphaBarrel_binMin , resAlphaBarrel_binWidth , resAlphaBarrel_binN ,
             nBetaBarrel  , resBetaBarrel_binMin  , resBetaBarrel_binWidth  , resBetaBarrel_binN  );

  sprintf  (det, "PXF" );
  bookPixel( histos_PXF_alpha , histos_PXF_beta , histos_PXF_nom_alpha , histos_PXF_nom_beta ,
             histos_PXF_dedx_alpha, histos_PXF_dedx_beta,
             det );
  bookPixel( histos_PXF_res_alpha , histos_PXF_res_beta , histos_PXF_nom_res_alpha , histos_PXF_nom_res_beta ,
             det ,
             nAlphaForward , resAlphaForward_binMin , resAlphaForward_binWidth , resAlphaForward_binN ,
             nBetaForward  , resBetaForward_binMin  , resBetaForward_binWidth  , resBetaForward_binN   );
  //
  
#ifdef rrDEBUG
  std::cout << "Famos histograms " << theRootFile->GetName() << " booked" << std::endl;
  // Print names of histograms inside multiplicity and resolution vectors
  std::cout << "Contents of histos_PXB_alpha vector" << std::endl;
  std::cout << "\tsize is " << histos_PXB_alpha.size() << std::endl;
  for(int i = 0; i < histos_PXB_alpha.size(); i++) {
    std::cout << "\tElement #" << i << " name is " << (histos_PXB_alpha.at(i))->GetName() << std::endl;
  }
  std::cout << "Contents of histos_PXB_beta vector" << std::endl;
  std::cout << "\tsize is " << histos_PXB_beta.size() << std::endl;
  for(int i = 0; i < histos_PXB_beta.size(); i++) {
    std::cout << "\tElement #" << i << " name is " << (histos_PXB_beta.at(i))->GetName() << std::endl;
  }
  std::cout << "Contents of histos_PXF_alpha vector" << std::endl;
  std::cout << "\tsize is " << histos_PXF_alpha.size() << std::endl;
  for(int i = 0; i < histos_PXF_alpha.size(); i++) {
    std::cout << "\tElement #" << i << " name is " << (histos_PXF_alpha.at(i))->GetName() << std::endl;
  }
  std::cout << "Contents of histos_PXF_beta vector" << std::endl;
  std::cout << "\tsize is " << histos_PXF_beta.size() << std::endl;
  for(int i = 0; i < histos_PXF_beta.size(); i++) {
    std::cout << "\tElement #" << i << " name is " << (histos_PXF_beta.at(i))->GetName() << std::endl;
  }
  std::cout << "Contents of histos_PXB_nom_alpha vector" << std::endl;
  std::cout << "\tsize is " << histos_PXB_nom_alpha.size() << std::endl;
  for(int i = 0; i < histos_PXB_nom_alpha.size(); i++) {
    std::cout << "\tElement #" << i << " name is " << (histos_PXB_nom_alpha.at(i))->GetName() << std::endl;
  }
  std::cout << "Contents of histos_PXB_nom_beta vector" << std::endl;
  std::cout << "\tsize is " << histos_PXB_nom_beta.size() << std::endl;
  for(int i = 0; i < histos_PXB_nom_beta.size(); i++) {
    std::cout << "\tElement #" << i << " name is " << (histos_PXB_nom_beta.at(i))->GetName() << std::endl;
  }
  std::cout << "Contents of histos_PXF_nom_alpha vector" << std::endl;
  std::cout << "\tsize is " << histos_PXF_nom_alpha.size() << std::endl;
  for(int i = 0; i < histos_PXF_nom_alpha.size(); i++) {
    std::cout << "\tElement #" << i << " name is " << (histos_PXF_nom_alpha.at(i))->GetName() << std::endl;
  }
  std::cout << "Contents of histos_PXF_nom_beta vector" << std::endl;
  std::cout << "\tsize is " << histos_PXF_nom_beta.size() << std::endl;
  for(int i = 0; i < histos_PXF_nom_beta.size(); i++) {
    std::cout << "\tElement #" << i << " name is " << (histos_PXF_nom_beta.at(i))->GetName() << std::endl;
  }
  std::cout << "Contents of histos_PXB_res_alpha vector" << std::endl;
  std::cout << "\tsize is " << histos_PXB_res_alpha.size() << std::endl;
  for(int i = 0; i < histos_PXB_res_alpha.size(); i++) {
    std::cout << "\tElement #" << i << " name is " << (histos_PXB_res_alpha.at(i))->GetName() << std::endl;
  }
  std::cout << "Contents of histos_PXB_res_beta vector" << std::endl;
  std::cout << "\tsize is " << histos_PXB_res_beta.size() << std::endl;
  for(int i = 0; i < histos_PXB_res_beta.size(); i++) {
    std::cout << "\tElement #" << i << " name is " << (histos_PXB_res_beta.at(i))->GetName() << std::endl;
  }
  std::cout << "Contents of histos_PXF_res_alpha vector" << std::endl;
  std::cout << "\tsize is " << histos_PXF_res_alpha.size() << std::endl;
  for(int i = 0; i < histos_PXF_res_alpha.size(); i++) {
    std::cout << "\tElement #" << i << " name is " << (histos_PXF_res_alpha.at(i))->GetName() << std::endl;
  }
  std::cout << "Contents of histos_PXF_res_beta vector" << std::endl;
  std::cout << "\tsize is " << histos_PXF_res_beta.size() << std::endl;
  for(int i = 0; i < histos_PXF_res_beta.size(); i++) {
    std::cout << "\tElement #" << i << " name is " << (histos_PXF_res_beta.at(i))->GetName() << std::endl;
  }
  std::cout << "Contents of histos_PXB_nom_res_alpha vector" << std::endl;
  std::cout << "\tsize is " << histos_PXB_nom_res_alpha.size() << std::endl;
  for(int i = 0; i < histos_PXB_nom_res_alpha.size(); i++) {
    std::cout << "\tElement #" << i << " name is " << (histos_PXB_nom_res_alpha.at(i))->GetName() << std::endl;
  }
  std::cout << "Contents of histos_PXB_nom_res_beta vector" << std::endl;
  std::cout << "\tsize is " << histos_PXB_nom_res_beta.size() << std::endl;
  for(int i = 0; i < histos_PXB_nom_res_beta.size(); i++) {
    std::cout << "\tElement #" << i << " name is " << (histos_PXB_nom_res_beta.at(i))->GetName() << std::endl;
  }
  std::cout << "Contents of histos_PXF_nom_res_alpha vector" << std::endl;
  std::cout << "\tsize is " << histos_PXF_nom_res_alpha.size() << std::endl;
  for(int i = 0; i < histos_PXF_nom_res_alpha.size(); i++) {
    std::cout << "\tElement #" << i << " name is " << (histos_PXF_nom_res_alpha.at(i))->GetName() << std::endl;
  }
  std::cout << "Contents of histos_PXF_nom_res_beta vector" << std::endl;
  std::cout << "\tsize is " << histos_PXF_nom_res_beta.size() << std::endl;
  for(int i = 0; i < histos_PXF_nom_res_beta.size(); i++) {
    std::cout << "\tElement #" << i << " name is " << (histos_PXF_nom_res_beta.at(i))->GetName() << std::endl;
  }
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
  // Load also big pixel data if CMSSW parametrization is on
  // They are pushed back into the vectors after the normal pixels data:
  // [0, ..., (size/2)-1] -> Normal pixels
  // [size/2, ..., size-1] -> Big pixels
  if(useCMSSWPixelParameterization) {
    // alpha barrel
    loadPixelData( pixelMultiplicityFile, nAlphaBarrel, std::string("hist_alpha_barrel_big"), histos_PXB_nom_alpha, true );
    loadPixelData( pixelBarrelResolutionFile, nAlphaBarrel, resAlphaBarrel_binN, resAlphaBarrel_binWidth, histos_PXB_nom_res_alpha, true, true );
    // 
    // beta barrel
    loadPixelData( pixelMultiplicityFile, nBetaBarrel, std::string("hist_beta_barrel_big"), histos_PXB_nom_beta, true );
    loadPixelData( pixelBarrelResolutionFile, nBetaBarrel, resBetaBarrel_binN, resBetaBarrel_binWidth, histos_PXB_nom_res_beta, false, true );
    // 
    // alpha forward
    loadPixelData( pixelMultiplicityFile, nAlphaForward, std::string("hist_alpha_forward_big"), histos_PXF_nom_alpha, true );
    loadPixelData( pixelForwardResolutionFile, nAlphaForward, resAlphaForward_binN, resAlphaForward_binWidth, histos_PXF_nom_res_alpha, true, true );
    // 
    // beta forward
    loadPixelData( pixelMultiplicityFile, nBetaForward, std::string("hist_beta_forward_big"), histos_PXF_nom_beta, true );
    loadPixelData( pixelForwardResolutionFile, nBetaForward, resBetaForward_binN, resBetaForward_binWidth, histos_PXF_nom_res_beta, false, true );
    //
    //
  }
}


void FamosRecHitAnalysis::loadPixelData( TFile* pixelDataFile, unsigned int nMultiplicity, std::string histName,
					 std::vector<TH1F*>& theMultiplicityProbabilities, bool isBig ) {
  std::string histName_i = histName + "_%u"; // needed to open histograms with a for
  if(!isBig)
    theMultiplicityProbabilities.clear();
  //
  std::vector<double> mult; // vector with fixed multiplicity
  for(unsigned int i = 0; i<nMultiplicity; i++) {
    TH1F addHist = *((TH1F*) pixelDataFile->Get( Form( histName_i.c_str() ,i+1 )));
    theMultiplicityProbabilities.push_back( new TH1F(addHist) );
  }
  
#ifdef rrDEBUG
  const unsigned int maxMult = theMultiplicityProbabilities.size();
  unsigned int iMult, multSize;
  if(useCMSSWPixelParameterization) {
    if(isBig) {     
      iMult = maxMult / 2;
      multSize = maxMult ;
    } else {                
      iMult = 0;
      multSize = maxMult;
    }
  } else {
    iMult = 0;
    multSize = maxMult ;
  }

  std::cout << " Multiplicity probability " << histName << std::endl;
  for(/* void */; iMult<multSize; iMult++) {
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
					 std::vector<TH1F*>& theResolutionHistograms, bool isAlpha, bool isBig) {
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
      std::string histName;
      if(isBig)
        histName = Form( "h%ub" , histN );
      else
        histName = Form( "h%u" , histN );
      TH1F hist = *(TH1F*) pixelDataFile->Get( histName.c_str() );
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
  unsigned int maxHist = 0;
  if(useCMSSWPixelParameterization)
    maxHist = histos_nom_alpha.size() / 2;
  else
    maxHist = histos_nom_alpha.size();
  for(unsigned int iHist = 0; iHist < maxHist; iHist++) {
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
  
  // Book also big pixels if CMSSW parameterization is set
  if(useCMSSWPixelParameterization) {
    for(unsigned int iHist = histos_nom_alpha.size()/2; iHist < histos_nom_alpha.size(); iHist++) {
      unsigned int multiplicity = iHist+1-histos_nom_alpha.size()/2;
      histos_alpha.push_back( new TH1F(Form( "hist_%s_%u_prob_alpha_big" , det , multiplicity ) ,
                                       Form( "Hit Local Position angle #alpha^{Rec} %s (multiplicity %u) probability big pixels;#alpha^{Rec} [rad];Probability" , det , multiplicity ) ,
                                       histos_nom_alpha[iHist]->GetNbinsX() , histos_nom_alpha[iHist]->GetXaxis()->GetXmin() , histos_nom_alpha[iHist]->GetXaxis()->GetXmax() ));
      histos_dedx_alpha.push_back( new TH1F(Form( "hist_%s_%u_dedx_alpha_big" , det , multiplicity ) ,
                                            Form( "Sim Hit energy loss dE/dx %s %u big pixels;dE/dx;Entries/bin" , det , multiplicity ) ,
                                            200, 0, 0.001 ));
      // change name to the nominal one
      histos_nom_alpha[iHist]->SetTitle( Form( "Hit Local Position angle #alpha^{Sim} %s (multiplicity %u) probability big pixels;#alpha^{Sim} [rad];Probability" , det , multiplicity ) );
      histos_nom_alpha[iHist]->SetName(  Form( "hist_%s_%u_nom_prob_alpha_big" , det , multiplicity ) );
#ifdef rrDEBUG
      for(int iBin = 1; iBin<=histos_nom_alpha[iHist]->GetNbinsX(); iBin++ )
        std::cout << "\tNominal Probability " << histos_nom_alpha[iHist]->GetName() << " bin " << iBin << " is " << histos_nom_alpha[iHist]->GetBinContent(iBin) << std::endl;
#endif  
    }
  }
  //
  //
  if(useCMSSWPixelParameterization)
    maxHist = histos_nom_beta.size() / 2;
  else
    maxHist = histos_nom_beta.size();
  for(unsigned int iHist = 0; iHist < maxHist; iHist++) {
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

  // Book also big pixels if CMSSW parameterization is set
  if(useCMSSWPixelParameterization) {
    for(unsigned int iHist = histos_nom_beta.size()/2; iHist < histos_nom_beta.size(); iHist++) {
      unsigned int multiplicity = iHist+1-histos_nom_beta.size()/2;
      histos_beta.push_back( new TH1F(Form( "hist_%s_%u_prob_beta_big" , det , multiplicity ) ,
                                      Form( "Hit Local Position angle #beta^{Rec} %s (multiplicity %u) probability big pixels;#beta^{Rec} [rad];Probability" , det , multiplicity ) ,
                                      histos_nom_beta[iHist]->GetNbinsX() , histos_nom_beta[iHist]->GetXaxis()->GetXmin() , histos_nom_beta[iHist]->GetXaxis()->GetXmax() ));
      histos_dedx_beta.push_back( new TH1F(Form( "hist_%s_%u_dedx_beta_big" , det , multiplicity ) ,
                                           Form( "Sim Hit energy loss dE/dx %s %u big pixels;dE/dx;Entries/bin" , det , multiplicity ) ,
                                           200, 0, 0.001 ));
      // change name to the nominal one
      histos_nom_beta[iHist]->SetTitle( Form( "Hit Local Position angle #beta^{Sim} %s (multiplicity %u) probability big pixels;#beta^{Sim} [rad];Probability" , det , multiplicity ) );
      histos_nom_beta[iHist]->SetName(  Form( "hist_%s_%u_nom_prob_beta_big" , det , multiplicity ) );
#ifdef rrDEBUG
      for(int iBin = 1; iBin<=histos_nom_beta[iHist]->GetNbinsX(); iBin++ )
        std::cout << "\tNominal Probability " << histos_nom_beta[iHist]->GetName() << " bin " << iBin << " is " << histos_nom_beta[iHist]->GetBinContent(iBin) << std::endl;
#endif
    }
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
  
  // Book also big pixels if CMSSW parameterization is set
  if(useCMSSWPixelParameterization) {
    for(unsigned int iMult = 0; iMult<nAlphaMultiplicity; iMult++) {
      for(int iBin = -1; iBin<resAlpha_binN; iBin++) {
        if(iBin<0) iBin++; // to avoid skip loop if nBins==0
        iHist++;
        float binMin = ( resAlpha_binWidth!=0 ? resAlpha_binMin+(float)(iBin)*resAlpha_binWidth   : -PI );
        float binMax = ( resAlpha_binWidth!=0 ? resAlpha_binMin+(float)(iBin+1)*resAlpha_binWidth :  PI );
        histos_alpha.push_back(new TH1F(Form( "hist_%s_%u_%u_res_alpha_big" , det , iMult+1 , iBin+1 ) ,
                                        Form( "Hit Local Position x^{Rec} [%f<#alpha^{Rec}<%f] %s (multiplicity %u) big pixels;x [cm];Events/bin" ,
                                              binMin , binMax ,
                                              det , iMult+1 ) ,
                                        histos_nom_alpha[iHist]->GetNbinsX() , histos_nom_alpha[iHist]->GetXaxis()->GetXmin() , histos_nom_alpha[iHist]->GetXaxis()->GetXmax() ) );
        // change name to the nominal one
        histos_nom_alpha[iHist]->SetName(  Form( "hist_%s_%u_%u_nom_res_alpha_big" , det , iMult+1 , iBin+1 ) );
        histos_nom_alpha[iHist]->SetTitle( Form( "Hit Local Position x^{Sim} [%f<#alpha^{Sim}<%f] %s (multiplicity %u) big pixels;x [cm];Events/bin" ,
					       binMin , binMax ,
					       det , iMult+1 ) );
      }
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

  // Book also big pixels if CMSSW parameterization is set  
  if(useCMSSWPixelParameterization) {
    for(unsigned int iMult = 0; iMult<nBetaMultiplicity; iMult++) {
      for(int iBin = -1; iBin<resBeta_binN; iBin++) {
        if(iBin<0) iBin++; // to avoid skip loop if nBins==0
        iHist++;
        histos_beta.push_back(new TH1F(Form( "hist_%s_%u_%u_res_beta_big" , det , iMult+1 , iBin+1 ) ,
                                       Form( "Hit Local Position y^{Rec} [%f<#beta^{Rec}<%f] %s (multiplicity %u) big pixels;y [cm];Events/bin" ,
                                             resBeta_binMin+(float)(iBin)*resBeta_binWidth , resBeta_binMin+(float)(iBin+1)*resBeta_binWidth ,
                                             det , iMult+1 ) ,
                                       histos_nom_beta[iHist]->GetNbinsX() , histos_nom_beta[iHist]->GetXaxis()->GetXmin() , histos_nom_beta[iHist]->GetXaxis()->GetXmax() ) );
        // change name to the nominal one
        histos_nom_beta[iHist]->SetName(  Form( "hist_%s_%u_%u_nom_res_beta_big" , det , iMult+1 , iBin+1 ) );
        histos_nom_beta[iHist]->SetTitle( Form( "Hit Local Position y^{Sim} [%f<#beta^{Sim}<%f] %s (multiplicity %u) big pixels;y [cm];Events/bin" ,
                                                resBeta_binMin+(float)(iBin)*resBeta_binWidth , resBeta_binMin+(float)(iBin+1)*resBeta_binWidth ,
                                                det , iMult+1 ) );
      }
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

  //Retrieve tracker topology from geometry
  edm::ESHandle<TrackerTopology> tTopoHand;
  setup.get<IdealGeometryRecord>().get(tTopoHand);
  const TrackerTopology *tTopo=tTopoHand.product();


  // Get PSimHit's of the Event
  
  edm::Handle<CrossingFrame<PSimHit> > cf_simhit; 
  std::vector<const CrossingFrame<PSimHit> *> cf_simhitvec;
  for(uint32_t i=0; i<trackerContainers.size(); i++){
    event.getByLabel(trackerContainers[i], cf_simhit);
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
        //
        
        bool hasBigPixelInX = false;
        bool hasBigPixelInY = false;
        
        if( (subdetid == 1 || subdetid==2) && useCMSSWPixelParameterization ) {
          // If the sim track crosses a sensitive region in which there are big pixels,
          // then we set to true the variables above
          
          // Get the topology of the pixel module
          const PixelGeomDetUnit* detUnit = dynamic_cast<const PixelGeomDetUnit*>(geometry->idToDetUnit(DetId(detUnitId)));
	  const RectangularPixelTopology *rectPixelTopology = static_cast<const RectangularPixelTopology*>(&(detUnit->specificType().specificTopology()));
          
          // Get the rows and columns of entry and exit points
          // FIXME - these are not guaranteed to be the same as the cluster limits (as they should be)
          const int firstPixelInX = int(rectPixelTopology->pixel(simHit->entryPoint()).first);
          const int firstPixelInY = int(rectPixelTopology->pixel(simHit->entryPoint()).second);
          const int lastPixelInX = int(rectPixelTopology->pixel(simHit->exitPoint()).first);
          const int lastPixelInY = int(rectPixelTopology->pixel(simHit->exitPoint()).second);
           
          // Check if there is a big pixel inside and set hasBigPixelInX and hasBigPixelInY accordingly
          // This function only works if first <= last
          if(rectPixelTopology->containsBigPixelInX(firstPixelInX < lastPixelInX ? firstPixelInX : lastPixelInX,
                                                   firstPixelInX > lastPixelInX ? firstPixelInX : lastPixelInX))
            hasBigPixelInX = true;
          if(rectPixelTopology->containsBigPixelInY(firstPixelInY < lastPixelInY ? firstPixelInY : lastPixelInY,
                                                   firstPixelInY > lastPixelInY ? firstPixelInY : lastPixelInY))
            hasBigPixelInY = true;
#ifdef rrDEBUG
          std::cout << " Simhit first pixel in X = " << (firstPixelInX < lastPixelInX ? firstPixelInX : lastPixelInX)
                    << " last pixel in X = " << (firstPixelInX > lastPixelInX ? firstPixelInX : lastPixelInX)
                    << " has big pixel in X is " << (hasBigPixelInX ? " true" : " false")
                    << std::endl;
          std::cout << " Simhit first pixel in Y = " << (firstPixelInY < lastPixelInY ? firstPixelInY : lastPixelInY)
                    << " last pixel in Y = " << (firstPixelInY > lastPixelInY ? firstPixelInY : lastPixelInY)
                    << " has big pixel in Y is " << (hasBigPixelInY ? " true" : " false")
                    << std::endl;
#endif 
        }

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
                    alpha      , beta      ,
                    hasBigPixelInX, hasBigPixelInY, tTopo );
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

// Virtual destructor needed.
FamosRecHitAnalysis::~FamosRecHitAnalysis() { 
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
                                      double       alpha      , double       beta,      
                                      const bool hasBigPixelInX, const bool hasBigPixelInY,
				      const TrackerTopology *tTopo) {
  int subdetid = ((rawid>>25)&0x7);

  switch (subdetid) {
    // Pixel Barrel
  case 1:
    // PXB
    {
      unsigned int iAlpha = mult_alpha - 1;
      unsigned int iBeta = mult_beta - 1;
      // Big pixel histograms in second half of vector
      if(hasBigPixelInX)
        iAlpha+=histos_PXB_alpha.size()/2;
      if(hasBigPixelInY)
        iBeta+=histos_PXB_beta.size()/2;
      
      hist_alpha       = histos_PXB_alpha[iAlpha];
      hist_beta        = histos_PXB_beta[iBeta];
      hist_dedx_alpha  = histos_PXB_dedx_alpha[iAlpha];
      hist_dedx_beta   = histos_PXB_dedx_beta[iBeta];
#ifdef rrDEBUG
      unsigned int theLayer = tTopo->pxbLayer(rawid);
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
        if(hasBigPixelInX) // Histogram is in the second half of vector
          iAlphaHist+=nAlphaBarrel*resAlphaBarrel_binN;
      } else {
	iAlphaHist = iAlpha;
      }
      if( iAlphaHist==-1 ) {
	double binMin = resAlphaBarrel_binMin;
	double binMax = resAlphaBarrel_binMin+(double)(resAlphaBarrel_binN)*resAlphaBarrel_binWidth;
	if( alpha < binMin)  iAlphaHist = ( (mult_alpha-1)*resAlphaBarrel_binN + 1 ) - 1; // underflow
	if( alpha >= binMax) iAlphaHist = ( (mult_alpha-1)*resAlphaBarrel_binN + resAlphaBarrel_binN ) - 1; // overflow
        if(hasBigPixelInX) {
          if(resAlphaBarrel_binN != 0)
            iAlphaHist+=nAlphaBarrel*resAlphaBarrel_binN;
          else
            iAlphaHist+=nAlphaBarrel;
        }
      }
      //
      int iBetaHist = -1;
      if(resBetaBarrel_binN!=0) {
	for(unsigned int iBin = 1; iBin<=resBetaBarrel_binN; iBin++) {
	  double binMin = resBetaBarrel_binMin+(double)(iBin-1)*resBetaBarrel_binWidth;
	  double binMax = resBetaBarrel_binMin+(double)(iBin)*resBetaBarrel_binWidth;
	  if( beta >= binMin && beta < binMax ) iBetaHist = ( (mult_beta-1)*resBetaBarrel_binN + iBin ) - 1;
	}
        if(hasBigPixelInY) // Histogram is in the second half of vector
          iBetaHist+=nBetaBarrel*resBetaBarrel_binN;
      } else {
	iBetaHist = iBeta;
      }
      if( iBetaHist==-1 ) {
	double binMin = resBetaBarrel_binMin;
	double binMax = resBetaBarrel_binMin+(double)(resBetaBarrel_binN)*resBetaBarrel_binWidth;
	if( beta < binMin)  iBetaHist = ( (mult_beta-1)*resBetaBarrel_binN + 1 ) - 1; // underflow
	if( beta >= binMax) iBetaHist = ( (mult_beta-1)*resBetaBarrel_binN + resBetaBarrel_binN ) - 1; // overflow
        if(hasBigPixelInY) {
          if(resBetaBarrel_binN != 0)
            iBetaHist+=nBetaBarrel*resBetaBarrel_binN;
          else
            iBetaHist+=nBetaBarrel;
        }
      }
      //
#ifdef rrDEBUG
      std::cout << "\tResolution histos chosen alpha " << iAlphaHist << " beta " << iBetaHist << std::endl;
      std::cout << "\tSize of alpha and beta vectors " << histos_PXB_res_alpha.size() << ", " << histos_PXB_res_beta.size() << std::endl;
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
      unsigned int iAlpha = mult_alpha - 1;
      unsigned int iBeta = mult_beta - 1;
      // Big pixel histograms in second half of vector
      if(hasBigPixelInX)
        iAlpha+=histos_PXF_alpha.size()/2;
      if(hasBigPixelInY)
        iBeta+=histos_PXF_beta.size()/2;
      
      hist_alpha       = histos_PXF_alpha[iAlpha];
      hist_beta        = histos_PXF_beta[iBeta];
      hist_dedx_alpha  = histos_PXF_dedx_alpha[iAlpha];
      hist_dedx_beta   = histos_PXF_dedx_beta[iBeta];
#ifdef rrDEBUG
      unsigned int theDisk = tTopo->pxfDisk(rawid);
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
        if(hasBigPixelInX) // Histogram is in the second half of vector
          iAlphaHist+=nAlphaForward*resAlphaForward_binN; 
      } else {
	iAlphaHist = iAlpha;
      }
      if( iAlphaHist==-1 ) {
	double binMin = resAlphaForward_binMin;
	double binMax = resAlphaForward_binMin+(double)(resAlphaForward_binN)*resAlphaForward_binWidth;
	if( alpha < binMin)  iAlphaHist = ( (mult_alpha-1)*resAlphaForward_binN + 1 ) - 1; // underflow
	if( alpha >= binMax) iAlphaHist = ( (mult_alpha-1)*resAlphaForward_binN + resAlphaForward_binN ) - 1; // overflow
        if(hasBigPixelInX) {
          if(resAlphaForward_binN != 0)
            iAlphaHist+=nAlphaForward*resAlphaForward_binN;
          else
            iAlphaHist+=nAlphaForward;
        }
      }
      //
      int iBetaHist = -1;
      if(resBetaForward_binN!=0) {
	for(unsigned int iBin = 1; iBin<=resBetaForward_binN; iBin++) {
	  double binMin = resBetaForward_binMin+(double)(iBin-1)*resBetaForward_binWidth;
	  double binMax = resBetaForward_binMin+(double)(iBin)*resBetaForward_binWidth;
	  if( beta >= binMin && beta < binMax ) iBetaHist = ( (mult_beta-1)*resBetaForward_binN + iBin ) - 1;
	}
        if(hasBigPixelInY) // Histogram is in the second half of vector
          iBetaHist+=nBetaForward*resBetaForward_binN;
      } else {
	iBetaHist = iBeta;
      }
      if( iBetaHist==-1 ) {
	double binMin = resBetaForward_binMin;
	double binMax = resBetaForward_binMin+(double)(resBetaForward_binN)*resBetaForward_binWidth;
	if( beta < binMin)  iBetaHist = ( (mult_beta-1)*resBetaForward_binN + 1 ) - 1; // underflow
	if( beta >= binMax) iBetaHist = ( (mult_beta-1)*resBetaForward_binN + resBetaForward_binN ) - 1; // overflow
        if(hasBigPixelInY) {
          if(resBetaForward_binN != 0)
            iBetaHist+=nBetaForward*resBetaForward_binN;
          else
            iBetaHist+=nBetaForward;
        }
      }
      //
#ifdef rrDEBUG
      std::cout << "\tResolution histos chosen alpha " << iAlphaHist << " beta " << iBetaHist << std::endl;
      std::cout << "\tSize of alpha and beta vectors " << histos_PXB_res_alpha.size() << ", " << histos_PXB_res_beta.size() << std::endl;      
#endif
      hist_res_alpha = histos_PXF_res_alpha[iAlphaHist];
      hist_res_beta  = histos_PXF_res_beta[iBetaHist];
      //
      break;
    } 
    //
  case 3:
    {
      
      unsigned int theLayer = tTopo->tibLayer(rawid);
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
      
      unsigned int theRing = tTopo->tidRing(rawid);
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
      
      unsigned int theLayer = tTopo->tobLayer(rawid);
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
      
      unsigned int theRing = tTopo->tecRing(rawid);
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
/*
    TF1* gaussianFit_histos_x = new TF1("gaussianFit_histos_x","gaus",
					histos_x[iHist]->GetMean() - 5. * histos_x[iHist]->GetRMS() ,
					histos_x[iHist]->GetMean() + 5. * histos_x[iHist]->GetRMS() );
    histos_x[iHist]->Fit("gaussianFit_histos_x","R");
    //
    TF1* constantFit_histos_y = new TF1("constantFit_histos_y","pol1",
					histos_y[iHist]->GetMean() - 2. * histos_y[iHist]->GetRMS() ,
					histos_y[iHist]->GetMean() + 2. * histos_y[iHist]->GetRMS() );
    histos_y[iHist]->Fit("constantFit_histos_y","R");
*/
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
  unsigned int maxHist = histos_angle.size();
  if(useCMSSWPixelParameterization)
    maxHist/=2;
  for(unsigned int iHist = 0; iHist < maxHist; iHist++) {
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
  for(unsigned int iMult = 0; iMult<maxHist; iMult++) {
    std::cout << " Multiplicity probability " << histos_angle[iMult]->GetName() << std::endl;
    for(int iBin = 1; iBin<=histos_angle[iMult]->GetNbinsX(); iBin++) {
      std::cout << " Multiplicity " << iMult+1 << " bin " << iBin << " low edge = " << histos_angle[iMult]->GetBinLowEdge(iBin)
		<< " prob = " << (histos_angle[iMult])->GetBinContent(iBin) // remember in ROOT bin starts from 1 (0 underflow, nBin+1 overflow)
		<< std::endl;
    }
  }
#endif

  if(useCMSSWPixelParameterization) { // Create probabilities for big pixels too, if CMSSW parameterization is set
    delete hist_tot;
    TH1F* hist_tot = 0;
    unsigned int minHist = histos_angle.size() / 2;
    unsigned int maxHist = histos_angle.size();
    for(unsigned int iHist = minHist; iHist < maxHist; iHist++) {
      if(iHist == minHist) hist_tot = (TH1F*)histos_angle[minHist]->Clone("tot");
      for(int iBin = 0; iBin < histos_angle[minHist]->GetNbinsX(); iBin++) {
        //
        if(iHist!=minHist) hist_tot->SetBinContent(iBin+1, hist_tot->GetBinContent(iBin+1)+histos_angle[iHist]->GetBinContent(iBin+1) );
        if(iHist!=minHist) hist_tot->SetBinError(iBin+1, 0.00 );
#ifdef rrDEBUG
        std::cout << "\tTotal Probability after " << histos_angle[iHist]->GetName() << " bin " << iBin << " is " << hist_tot->GetBinContent(iBin+1) << std::endl;
#endif
      }
    }
    for(unsigned int iHist = minHist; iHist < maxHist; iHist++) {
      histos_angle[iHist]->Divide(hist_tot);
    }
#ifdef rrDEBUG
    for(unsigned int iMult = 0; iMult<histos_angle.size()/2; iMult++) {
      std::cout << " Multiplicity probability " << histos_angle[iMult+minHist]->GetName() << std::endl;
      for(int iBin = 1; iBin<=histos_angle[iMult+minHist]->GetNbinsX(); iBin++) {
        std::cout << " Multiplicity " << iMult+1 << " bin " << iBin << " low edge = " << histos_angle[iMult+minHist]->GetBinLowEdge(iBin)
                  << " prob = " << (histos_angle[iMult+minHist])->GetBinContent(iBin) // remember in ROOT bin starts from 1 (0 underflow, nBin+1 overflow)
                  << std::endl;
      }
    }
#endif
  }
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
    if(histos_value[iHist]->Integral() != 0)
      can_comparison.SetLogy(yLogScale);
    else
      can_comparison.SetLogy(0);
    // settings
    gStyle->SetOptStat(0);
    gStyle->SetOptFit(0);
    //
    histos_value[iHist]->SetMarkerColor(4); // blue
    histos_value[iHist]->SetLineColor(4); // blue
    histos_value[iHist]->SetMarkerStyle(20); // circles
    histos_value[iHist]->SetMarkerSize(1.0); // 
    histos_nominal[iHist]->SetLineColor(2); // red
    histos_nominal[iHist]->SetFillColor(0); // white
    //
    if(binFactor!=-1) {
      histos_value[iHist]->Rebin(binFactor);
      histos_nominal[iHist]->Rebin(binFactor);
//      histos_value[iHist]->GetXaxis()->SetRangeUser( histos_value[iHist]->GetMean() - 3 * histos_value[iHist]->GetRMS() , 
//						     histos_value[iHist]->GetMean() + 3 * histos_value[iHist]->GetRMS() );
      // normalise entries of nominal histo to value histo (useful for pixel)
      if(histos_nominal[iHist]->GetEntries()!=0) histos_nominal[iHist]->Scale(histos_value[iHist]->GetEntries()/histos_nominal[iHist]->GetEntries());
    }
    //
    // Draw
    if(yMax != -1) histos_nominal[iHist]->SetMaximum(yMax);
    histos_nominal[iHist]->Sumw2();
    histos_nominal[iHist]->Draw("HIST");
    histos_value[iHist]->Draw("HIST P E1 SAME");
    //
    // perform chi2 test between obtained and nominal histograms
    double compatibilityFactor = 0;
    if(histos_value[iHist]->Integral() != 0)
      compatibilityFactor = histos_value[iHist]->KolmogorovTest(histos_nominal[iHist],"");
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

DEFINE_FWK_MODULE(FamosRecHitAnalysis);
