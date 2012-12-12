/*
  Script to produce validation plots.
  Run in terminal, type:
  root -l
  .L GemValidation.cc
  main()
  
  You will be prompted to type a production identifier, e.g. "SingleMuPt40_20121128_Pilot".
  The program will look for two files in the *working* directory:
  - SingleMuPt40_20121128_Pilot_SIM.root
  - SingleMuPt40_20121128_Pilot_DIGI.root
  These files are of course the output of the *GEMAnalyzers*
  - GEMSimHitAnalyzer
  - GEMDigiAnalyzer
  In case one is or both are missing, the program will terminate automatically.
  Otherwise an output directory is created in the *working* directory, e.g. GemValidation_SingleMuPt40_20121128_Pilot/.
  Plots (.pdf format) are stored in the output directory. 
  
  FIXME:
  - include check for existing directories
  - include path to ROOT files not in working directory

  Contact sven.dildick@cern.ch for more information
 */

#include "TFile.h"
#include "TTree.h"
#include "TBranch.h"
#include "TLeaf.h"
#include "TDirectory.h"
#include "TROOT.h"
#include "TGraph.h"
#include "TCanvas.h"
#include "TH2D.h"
#include "TH1D.h"
#include "TPad.h"
#include "TStyle.h"
#include "TString.h"

#include <iostream>
#include <cstdlib>
#include <stdio.h>
#include <sys/stat.h>
#include <stdlib.h>
#include <string>
#include <sstream>

// convert value to string
template < typename T > std::string to_string( T const& value )
{
  std::stringstream sstr;
  sstr << value;
  return sstr.str();
}

// create output directory in the working directory
void create_output_directory(const TString dir)
{
  std::cout << ">>> Attempting to create output directory: " << dir << std::endl;
  struct stat sb;
  // FIXME
  // if ( stat(dir.c_str(), &sb) == 0 && S_ISDIR(sb.st_mode) )
  //   std::cout << ">>> Output directory exists. Overwriting results." << std::endl;
  // else
  // const TString command( "mkdir " + dir );
  system( "mkdir " + dir );
}

int main(int argc, char* argv[] )
{
  // To make the plot a bit prettier...
  gStyle->SetPalette(1);
  gStyle->SetCanvasColor(kWhite);
  gStyle->SetPadColor(kWhite);
  gStyle->SetPadTickX(1);
  gStyle->SetPadTickY(1);
  //  gStyle->SetPadTopMargin(.075);
  gStyle->SetPadTopMargin(0.12);
  gStyle->SetPadRightMargin(0.12);
  gStyle->SetPadBottomMargin(.12);
  //  gStyle->SetPadLeftMargin(.1425);
  gStyle->SetPadLeftMargin(.12);
  gStyle->SetTitleSize(.05,"XYZ");
  gStyle->SetTitleFillColor(kWhite);
  gStyle->SetTitleBorderSize(0);
  gStyle->SetStatColor(kWhite);
  gStyle->SetStatBorderSize(1);
  gStyle->SetOptStat( 0 );
  gStyle->SetOptFit( 0 );
  gStyle->SetMarkerStyle(8);
  gStyle->SetFrameBorderMode(0);
  gStyle->SetNumberContours(50);

  // Steering  
  std::cout << ">>> Enter MC production identifier:" << std::endl;
  std::string identifier;
  std::cin >> identifier;

  // create output directory in the working directory
  const TString outputDir( "validationPlots_" + identifier + "/");
  create_output_directory(outputDir);

  // use ROOT files in working directory
  const TString simHitFile( identifier + "_SIM.root" );
  const TString digiFile( identifier + "_DIGI.root" );
  
  std::cout << ">>> Using input files: " << std::endl
	    << "\t" << simHitFile << std::endl
	    << "\t" << digiFile << std::endl;

  // extension for the plots [.pdf/.png]
  TString ext( ".pdf");

  /////////////////////////////
  // SIMHIT VALIDATION PLOTS //
  /////////////////////////////

  TFile *f = new TFile( simHitFile );
  std::cout << "Opening TFile: " << simHitFile << std::endl;  
  if (!f || !f->IsOpen()) {    
    std::cerr << "GemValidationPlots() No such TROOT file. Exiting" << std::endl;
    exit(1);
  } 
  else {
    std::cout << "Opening TDirectory: gemSimHitAnalyzer" << std::endl;  
    TDirectory * dir = (TDirectory*)f->Get("gemSimHitAnalyzer");
    if (!dir){
      std::cerr << "No such TDirectory: gemSimHitAnalyzer. Exiting." << std::endl;
      exit(1);
    }
    else{
      std::cout << "Opening TTree: GEMSimHits" << std::endl;
      TTree* tree = (TTree*) dir->Get("GEMSimHits");
      if (!tree){
	std::cerr << "No such TTree: GEMSimHits. Exiting." << std::endl;
	exit(1);
      }
      else{
	//--------------------//
	// XY occupancy plots //
	//--------------------//

	TCanvas* c = new TCanvas("c","c",600,600);
	tree->Draw("globalY:globalX>>hh(54,-270,270, 54,-270,270)","region==-1&&layer==1");
	TH2D *hh = (TH2D*)gDirectory->Get("hh");
	hh->SetMarkerSize(0.1);    
	hh->SetTitle("SimHit occupancy: region-1, layer1;globalX [cm];globalY [cm]");
	hh->Draw("COLZ");    
	c->SaveAs( outputDir + "globalxy_region-1_layer1_simhit" + ext );

	c->Clear();
	tree->Draw("globalY:globalX>>hh(54,-270,270, 54,-270,270)","region==-1&&layer==2");
	hh = (TH2D*)gDirectory->Get("hh");
	hh->SetMarkerSize(0.1); 
	hh->SetTitle("SimHit occupancy: region-1, layer2;globalX [cm];globalY [cm]");   
	hh->Draw("COLZ");    
	c->SaveAs( outputDir + "globalxy_region-1_layer2_simhit" + ext);

	c->Clear();
	tree->Draw("globalY:globalX>>hh(54,-270,270, 54,-270,270)","region==1&&layer==1");
	hh = (TH2D*)gDirectory->Get("hh");
	hh->SetMarkerSize(0.1);
	hh->SetTitle("SimHit occupancy: region1, layer1;globalX [cm];globalY [cm]");   
	hh->Draw("COLZ");    
	c->SaveAs( outputDir + "globalxy_region1_layer1_simhit" + ext);

	c->Clear();	
	tree->Draw("globalY:globalX>>hh(54,-270,270, 54,-270,270)","region==1&&layer==2");
	hh = (TH2D*)gDirectory->Get("hh");
	hh->SetMarkerSize(0.1);    
	hh->SetTitle("SimHit occupancy: region1, layer2;globalX [cm];globalY [cm]");   
	hh->Draw("COLZ");    
	c->SaveAs(outputDir + "globalxy_region1_layer2_simhit" + ext);

	//--------------------//
	// ZR occupancy plots //
	//--------------------//

	c->Clear();	
	tree->Draw("sqrt(globalX*globalX+globalY*globalY):globalZ>>hh(30,-568,-565, 110,130,240)","region==-1&&layer==1");
	hh = (TH2D*)gDirectory->Get("hh");
	hh->SetMarkerSize(0.1);    
	hh->SetTitle("SimHit occupancy: region-1, layer1;globalZ [cm];globalR [cm]");
	hh->Draw("COLZ");    
	c->SaveAs(outputDir + "globalzr_region-1_layer1_simhit" + ext);

	c->Clear();
	tree->Draw("sqrt(globalX*globalX+globalY*globalY):globalZ>>hh(30,-572.25,-569.25, 110,130,240)","region==-1&&layer==2");
	hh = (TH2D*)gDirectory->Get("hh");
	hh->SetMarkerSize(0.1);    
	hh->SetTitle("SimHit occupancy: region-1, layer2;globalZ [cm];globalR [cm]");
	hh->Draw("COLZ");    
	c->SaveAs(outputDir + "globalzr_region-1_layer2_simhit" + ext);

	c->Clear();
	tree->Draw("sqrt(globalX*globalX+globalY*globalY):globalZ>>hh(30,565,568, 110,130,240)","region==1&&layer==1");
	hh = (TH2D*)gDirectory->Get("hh");
	hh->SetMarkerSize(0.1);    
	hh->SetTitle("SimHit occupancy: region1, layer1;globalZ [cm];globalR [cm]");
	hh->Draw("COLZ");    
	c->SaveAs(outputDir + "globalzr_region1_layer1_simhit" + ext);

	c->Clear();
	tree->Draw("sqrt(globalX*globalX+globalY*globalY):globalZ>>hh(30,569.25,572.25, 110,130,240)","region==1&&layer==2");
	hh = (TH2D*)gDirectory->Get("hh");
	hh->SetMarkerSize(0.1);    
	hh->SetTitle("SimHit occupancy: region1, layer2;globalZ [cm];globalR [cm]");
	hh->Draw("COLZ");    
	c->SaveAs(outputDir + "globalzr_region1_layer2_simhit" + ext);
	
	//--------------------//
	// timeOfFlight plots //
	//--------------------//

	c->Clear();
	tree->Draw("timeOfFlight>>h(300,0,30)","region==-1&&layer==1");
	TH1D* h = (TH1D*)gDirectory->Get("h");
	h->SetMarkerSize(0.1);    
	gPad->SetLogy();
        TString title( "SimHits timeOfFlight: region-1, layer1;Time of flight [ns];entries/" + to_string(h->GetBinWidth(1)) + " ns" );
	h->SetTitle( title );
	h->Draw("");        
	c->SaveAs(outputDir + "timeOfFlight_region-1_layer1_simhit" + ext);

	c->Clear();
	tree->Draw("timeOfFlight>>h(300,0.,30.)","region==-1&&layer==2");
	h = (TH1D*)gDirectory->Get("h");
	h->SetMarkerSize(0.1);    
	gPad->SetLogy();
	title = "SimHits timeOfFlight: region-1, layer2;Time of flight [ns];entries/" + to_string(h->GetBinWidth(1)) + " ns";
	h->SetTitle( title );
	h->Draw("");        
	c->SaveAs(outputDir + "timeOfFlight_region-1_layer2_simhit" + ext);       

	c->Clear();
	tree->Draw("timeOfFlight>>h(300,0.,30.)","region==1&&layer==1");
	h = (TH1D*)gDirectory->Get("h");
	h->SetMarkerSize(0.1);    
	gPad->SetLogy();
	title = "SimHits timeOfFlight: region1, layer1;Time of flight [ns];entries/" + to_string(h->GetBinWidth(1)) + " ns";
	h->SetTitle( title );       
	h->Draw("");        
	c->SaveAs(outputDir + "timeOfFlight_region1_layer1_simhit" + ext);

	c->Clear();
	tree->Draw("timeOfFlight>>h(300,0.,30.)","region==1&&layer==2");
	h = (TH1D*)gDirectory->Get("h");
	h->SetMarkerSize(0.1);    
	gPad->SetLogy();
	title = "SimHits timeOfFlight: region1, layer2;Time of flight [ns];entries/" +  to_string(h->GetBinWidth(1)) + " ns";
	h->SetTitle( title );       
	h->Draw("");        
	c->SaveAs(outputDir + "timeOfFlight_region1_layer2_simhit" + ext);

	delete hh;
	delete h;
	delete c;
      }
      delete tree;
    }
    delete dir;
  }
  delete f;

  ///////////////////////////
  // DIGI VALIDATION PLOTS //
  ///////////////////////////

  TFile* f = new TFile( digiFile );
  std::cout << "Opening TFile: " << digiFile << std::endl;  
  if (!f || !f->IsOpen()) {    
    std::cerr << "GemValidationPlots() No such TROOT file. Exiting." << std::endl;
    exit(1);
  }
  else {
    std::cout << "Opening TDirectory: gemDigiAnalyzer" << std::endl;  
    TDirectory * dir = (TDirectory*)f->Get("gemDigiAnalyzer");
    if (!dir){
      std::cerr << "No such TDirectory: gemDigiAnalyzer. Exiting." << std::endl;
      exit(1);
    }
    else {
      std::cout << "Opening TTree: GemSimDigiTree" << std::endl;
      TTree* tree = (TTree*) dir->Get("GEMDigiTree");
      if (!tree){
	std::cerr << "No such TTree: GemDigiTree. Exiting." << std::endl;
	exit(1);
      } 
      else {
	//--------------------//
	// XY occupancy plots //
	//--------------------//

	TCanvas* c = new TCanvas("c","c",600,600);
	tree->Draw("g_x:g_y>>hh(100,-270,270,100,-270,270)","region==-1&&layer==1");
	TH2D *hh = (TH2D*)gDirectory->Get("hh");  
	hh->SetMarkerSize(0.1);
	hh->SetTitle("Digi occupancy: region-1, layer1; globalX [cm]; globalY [cm]");
	hh->Draw("COLZ");
	c->SaveAs(outputDir + "globalxy_region-1_layer1_digi" + ext);

	c->Clear();
	tree->Draw("g_x:g_y>>hh(100,-270,270,100,-270,270)","region==-1&&layer==2");
	hh = (TH2D*)gDirectory->Get("hh");
	hh->SetMarkerSize(0.1);
	hh->SetTitle("Digi occupancy: region-1, layer2; globalX [cm]; globalY [cm]");
	hh->Draw("COLZ");
	c->SaveAs(outputDir + "globalxy_region-1_layer2_digi" + ext);

	c->Clear();
	tree->Draw("g_x:g_y>>hh(100,-270,270,100,-270,270)","region==1&&layer==1");
	hh = (TH2D*)gDirectory->Get("hh");
	hh->SetMarkerSize(0.1);
	hh->SetTitle("Digi occupancy: region1, layer1; globalX [cm]; globalY [cm]");
	hh->Draw("COLZ");
	c->SaveAs(outputDir + "globalxy_region1_layer1_digi" + ext);

	c->Clear();
	tree->Draw("g_x:g_y>>hh(100,-270,270,100,-270,270)","region==1&&layer==2");
	hh = (TH2D*)gDirectory->Get("hh");
	hh->SetMarkerSize(0.1);
	hh->SetTitle("Digi occupancy: region1, layer1; globalX [cm]; globalY [cm]");	
	hh->Draw("COLZ");
	c->SaveAs(outputDir + "globalxy_region1_layer2_digi" + ext);

	//--------------------//
	// ZR occupancy plots //
	//--------------------//

	c->Clear();	
	tree->Draw("g_r:g_z>>hh(30,-568,-565,22,130,240)","region==-1&&layer==1");
	hh = (TH2D*)gDirectory->Get("hh");
	hh->SetMarkerSize(0.1);
	hh->SetTitle("Digi occupancy: region-1, layer1; globalZ [cm]; globalR [cm]");	
	hh->Draw("COLZ");
	c->SaveAs(outputDir + "globalzr_region-1_layer1_digi" + ext);

	c->Clear();	
	tree->Draw("g_r:g_z>>hh(30,-572.25,-569.25,22,130,240)","region==-1&&layer==2");
	hh = (TH2D*)gDirectory->Get("hh");
	hh->SetMarkerSize(0.1);
	hh->SetTitle("Digi occupancy: region-1, layer2; globalZ [cm]; globalR [cm]");	
	hh->Draw("COLZ");
	c->SaveAs(outputDir + "globalzr_region-1_layer2_digi" + ext);
	  
	c->Clear();	
	tree->Draw("g_r:g_z>>hh(30,565,568,22,130,240)","region==1&&layer==1");
	hh = (TH2D*)gDirectory->Get("hh");
 	hh->SetMarkerSize(0.1);
	hh->SetTitle("Digi occupancy: region1, layer1; globalZ [cm]; globalR [cm]");	
	hh->Draw("COLZ");
	c->SaveAs(outputDir + "globalzr_region1_layer1_digi" + ext);
	
	c->Clear();		
	tree->Draw("g_r:g_z>>hh(30,569.25,572.25,22,130,240)","region==1&&layer==2");
	hh = (TH2D*)gDirectory->Get("hh");
	hh->SetMarkerSize(0.1);
	hh->SetTitle("Digi occupancy: region1, layer2; globalZ [cm]; globalR [cm]");	
	hh->Draw("COLZ");
	c->SaveAs(outputDir + "globalzr_region1_layer2_digi" + ext);

	//--------------------//
	//   PhiStrip plots   //
	//--------------------//

	c->Clear();		
	tree->Draw("strip:g_phi>>hh(140,-3.5,3.5,100,0,400,)","region==-1&&layer==1");
	hh = (TH2D*)gDirectory->Get("hh");
	hh->SetMarkerSize(0.6);
	hh->SetTitle("Digi occupancy: region-1, layer1; phi [rad]; strip");		
	hh->Draw("COLZ");
	c->SaveAs(outputDir + "phiStrip_region-1_layer1_digi" + ext);

	c->Clear();		
	tree->Draw("strip:g_phi>>hh(140,-3.5,3.5,100,0,400)","region==-1&&layer==2");
	hh = (TH2D*)gDirectory->Get("hh");
	hh->SetMarkerSize(0.1);
	hh->SetTitle("Digi occupancy: region-1, layer2; phi [rad]; strip");		
	hh->Draw("COLZ");
	c->SaveAs(outputDir + "phiStrip_region-1_layer2_digi" + ext);

	c->Clear();		
	tree->Draw("strip:g_phi>>hh(140,-3.5,3.5,100,0,400)","region==1&&layer==1");
	hh = (TH2D*)gDirectory->Get("hh");
	hh->SetMarkerSize(0.1);
	hh->SetTitle("Digi occupancy: region1, layer1; phi [rad]; strip");		
	hh->Draw("COLZ");
	c->SaveAs(outputDir + "phiStrip_region1_layer1_digi" + ext);

	c->Clear();		
	tree->Draw("strip:g_phi>>hh(140,-3.5,3.5,100,0,400)","region==1&&layer==2");
	hh = (TH2D*)gDirectory->Get("hh");
	hh->SetMarkerSize(0.1);
	hh->SetTitle("Digi occupancy: region1, layer2; phi [rad]; strip");		
	hh->Draw("COLZ");
	c->SaveAs(outputDir + "phiStrip_region1_layer2_digi" + ext);

	c->Clear();		
	tree->Draw("strip>>h(200,0,400)");
	TH1F* h = (TH1F*)gDirectory->Get("h");
	h->SetMarkerSize(0.2);
	TString title(";strip;entries/" + to_string(h->GetBinWidth(1)) + " strips");
	h->SetTitle(title);		
	h->Draw("");
	c->SaveAs(outputDir + "strip_digi" + ext);

	//-----------------------//
	// Bunch crossing plots  //  
	//-----------------------//

	c->Clear();		
	tree->Draw("bx>>h(20,-2,2)");
	h = (TH1F*)gDirectory->Get("h");
	h->SetMarkerSize(0.2);
	gPad->SetLogy();
	h->SetTitle(";bunch crossing;entries");			
	h->Draw("");
	c->SaveAs(outputDir + "bx_digi" + ext);

	c->Clear();		
	tree->Draw("bx>>h(20,-2,2)","region==-1&&layer==1");
	h = (TH1F*)gDirectory->Get("h");
	h->SetMarkerSize(0.2);
	gPad->SetLogy();
	h->SetTitle("Bunch crossing: region-1,layer1;bunch crossing;entries");			
	h->Draw("");
	c->SaveAs(outputDir + "bx_region-1_layer1_digi" + ext);

	c->Clear();		
	tree->Draw("bx>>h(20,-2,2)","region==-1&&layer==2");
	h = (TH1F*)gDirectory->Get("h");
	h->SetMarkerSize(0.2);
	gPad->SetLogy();
	h->SetTitle("Bunch crossing: region-1,layer2; bunch crossing;entries");			
	h->Draw("");
	c->SaveAs(outputDir + "bx_region-1_layer2_digi" + ext);

	c->Clear();		
	tree->Draw("bx>>h(20,-2,2)","region==1&&layer==1");
	h = (TH1F*)gDirectory->Get("h");
	h->SetMarkerSize(0.2);
	gPad->SetLogy();
	h->SetTitle("Bunch crossing: region1,layer1 ;bunch crossing;entries");			
	h->Draw("");
	c->SaveAs(outputDir + "bx_region1_layer1_digi" + ext);

	c->Clear();		
	tree->Draw("bx>>h(20,-2,2)","region==1&&layer==2");
	h = (TH1F*)gDirectory->Get("h");
	h->SetMarkerSize(0.2);
	gPad->SetLogy();
	h->SetTitle("Bunch crossing: region1,layer2 ;bunch crossing;entries");			
	h->Draw("");
	c->SaveAs(outputDir + "bx_region1_layer2_digi" + ext);

	delete hh;
	delete h;
	delete c;
      }
      delete tree;
    }
    delete dir;
  }
  delete f;

}
