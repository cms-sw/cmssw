#include <TH2.h>
#include <TStyle.h>
#include <TCanvas.h>
#include <TMath.h>
#include <cmath>
#include <TString.h>

void CosmicRateTool_Kinematics(const char* fileName)
{
        TString InputFile= Form("../test/%s",fileName); 
        TFile *file = new TFile(InputFile);
	TTree *tree;
	tree = (TTree*)file->Get("demo/Event");
	
   vector<double>  *pt;
   vector<double>  *charge;
   vector<double>  *chi2;
   vector<double>  *chi2_ndof;
   vector<double>  *eta;
   vector<double>  *theta;
   vector<double>  *phi;
   vector<double>  *p;
   vector<double>  *d0;
   vector<double>  *dz;
   vector<double>  *nvh;
   vector<int>  *v_ntrk;
   
   pt = 0;
   charge = 0;
   chi2 = 0;
   chi2_ndof = 0;
   eta = 0;
   theta = 0;
   phi = 0;
   p = 0;
   d0 = 0;
   dz = 0;
   nvh = 0;

   tree->SetBranchAddress("pt", &pt);
   tree->SetBranchAddress("charge", &charge);
   tree->SetBranchAddress("chi2", &chi2);
   tree->SetBranchAddress("chi2_ndof", &chi2_ndof);
   tree->SetBranchAddress("eta", &eta);
   tree->SetBranchAddress("theta", &theta);
   tree->SetBranchAddress("phi", &phi);
   tree->SetBranchAddress("p", &p);
   tree->SetBranchAddress("d0", &d0);
   tree->SetBranchAddress("dz", &dz);
   tree->SetBranchAddress("nvh", &nvh);
//   tree->SetBranchAddress("v_ntrk", &v_ntrk);

   Long64_t n = tree->GetEntriesFast();

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//					Various Kinematical Histograms Declerations				
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

   TH1D h_pt 		("h_pt","h_pt",200,0,200);
   TH1D h_charge 	("h_charge","h_charge",10,-5,5);
   TH1D h_chi2 		("h_chi2","h_chi2",200,0,100);
   TH1D h_chi2_ndof 	("h_chi2_ndof","h_chi2_ndof",200,0,20);
   TH1D h_eta 		("h_eta","h_eta",500,-3,3);
   TH1D h_theta 	("h_theta","h_theta",500,-3,3);
   TH1D h_phi 		("h_phi","h_phi",400,-3.5,3.5);
   TH1D h_d0 		("h_d0","h_d0",1000,-85,85);
   TH1D h_dz 		("h_dz","h_dz",1500,-350,350);
//   TH1D *h_ntrk		= new TH1D("h_ntrk","h_ntrk",20,0,20);

//----------------------------------------------------------------------------------------------------------------

   
   for (Long64_t jentry=0; jentry<n;jentry++) 
   {
     tree->GetEntry(jentry);

      for (int k = 0; k < pt->size() ; k++)			// Loop to calculate Kinematical distributions
      {
	h_pt.Fill(pt->at(k));
	h_charge.Fill(charge->at(k));
	h_chi2.Fill(chi2->at(k));
	h_chi2_ndof.Fill(chi2_ndof->at(k));
	h_eta.Fill(eta->at(k));
	h_theta.Fill(theta->at(k));
	h_phi.Fill(phi->at(k));
	h_d0.Fill(d0->at(k));
	h_dz.Fill(dz->at(k));


      }								// Loop Closed to calculate Kinematical distributions
     }								// Loop Closed to calculate rates


//++++++++++++++++++++++++++++++++++       Make Directory     ++++++++++++++++++++++++++++++++++++++

	gSystem->Exec("mkdir -p Kinematical_Plots");

//---------------------------------------------------------------------------------------------------
	
	TCanvas c("c","c",800,600);    // Declare canvas

//+++++++++++++++++++++++++++++++       pT Distribution      ++++++++++++++++++++++++++++++++++++++++     
	h_pt.SetLineColor(kBlue);
	h_pt.SetLineWidth(2);
	h_pt.SetTitle("pT distribution");
	h_pt.SetXTitle("pT (in GeV)");
	h_pt.Draw();
	h_pt.SetStats();
	c.SetGrid();
	c.SaveAs("pt.png");
	c.Clear();
	gSystem->Exec("mv pt.png Kinematical_Plots");
//---------------------------------------------------------------------------------------------------


//+++++++++++++++++++++++++++++++       charge Distribution      ++++++++++++++++++++++++++++++++++++++++     

	h_charge.SetLineColor(kBlue);
	h_charge.SetLineWidth(2);
	h_charge.SetTitle("charge");
	h_charge.SetXTitle("");
	h_charge.Draw();
	c.SetGrid();
	c.SaveAs("charge.png");
	c.Clear();
	gSystem->Exec("mv charge.png Kinematical_Plots");
//---------------------------------------------------------------------------------------------------


//+++++++++++++++++++++++++++++++       chi2 Distribution      ++++++++++++++++++++++++++++++++++++++++     

        h_chi2.SetLineColor(kBlue);
        h_chi2.SetLineWidth(2); 
        h_chi2.SetTitle("chi2 distribution");
        h_chi2.SetXTitle("");
        h_chi2.Draw();
        c.SetGrid();                         
        c.SaveAs("chi2.png");                                                                                                                  
        c.Clear();      
	gSystem->Exec("mv chi2.png Kinematical_Plots");
//---------------------------------------------------------------------------------------------------


//+++++++++++++++++++++++++++++++       chi2/ndof Distribution      ++++++++++++++++++++++++++++++++++++++++     

        h_chi2_ndof.SetLineColor(kBlue);
        h_chi2_ndof.SetLineWidth(2); 
        h_chi2_ndof.SetTitle("chi2 per ndof");
        h_chi2_ndof.SetXTitle("");
        h_chi2_ndof.Draw();
        c.SetGrid();    
        c.SaveAs("chi2_ndof.png");                                                                                                                       c.Clear();      
	c.Clear();
	gSystem->Exec("mv chi2_ndof.png Kinematical_Plots");
//---------------------------------------------------------------------------------------------------


//+++++++++++++++++++++++++++++++       eta Distribution      ++++++++++++++++++++++++++++++++++++++++     

        h_eta.SetLineColor(kBlue);
        h_eta.SetLineWidth(2); 
        h_eta.SetTitle("eta Distribution");
        h_eta.SetXTitle("#eta");
        h_eta.Draw();
        c.SetGrid();                            
        c.SaveAs("eta.png");                                                                                                                  
        c.Clear();      
	gSystem->Exec("mv eta.png Kinematical_Plots");
//---------------------------------------------------------------------------------------------------


//+++++++++++++++++++++++++++++++       theta Distribution      ++++++++++++++++++++++++++++++++++++++++     

        h_theta.SetLineColor(kBlue);
        h_theta.SetLineWidth(2); 
        h_theta.SetTitle("theta distribution");
        h_theta.SetXTitle("#theta");
        h_theta.Draw();
        c.SetGrid();    
        c.SaveAs("theta.png");                                                                                                                  
        c.Clear();      
	gSystem->Exec("mv theta.png Kinematical_Plots");
//---------------------------------------------------------------------------------------------------


//+++++++++++++++++++++++++++++++       phi Distribution      ++++++++++++++++++++++++++++++++++++++++     

        h_phi.SetLineColor(kBlue);
        h_phi.SetLineWidth(2); 
        h_phi.SetTitle("phi distribution");
        h_phi.SetXTitle("#phi");
        h_phi.Draw();
        c.SetGrid();    
        c.SaveAs("phi.png");                                                                                                                  
        c.Clear();      
	gSystem->Exec("mv phi.png Kinematical_Plots");
//---------------------------------------------------------------------------------------------------


//+++++++++++++++++++++++++++++++       d0 Distribution      ++++++++++++++++++++++++++++++++++++++++     

        h_d0.SetLineColor(kBlue);
        h_d0.SetLineWidth(2); 
        h_d0.SetTitle("d0 distribution");
        h_d0.SetXTitle("d0");
        h_d0.Draw();
        c.SetGrid();                         
        c.SaveAs("d0.png");                                                                                                                  
        c.Clear();      
	gSystem->Exec("mv d0.png Kinematical_Plots");
//---------------------------------------------------------------------------------------------------


//+++++++++++++++++++++++++++++++       dz Distribution      ++++++++++++++++++++++++++++++++++++++++     

        h_dz.SetLineColor(kBlue);
        h_dz.SetLineWidth(2); 
        h_dz.SetTitle("dz distribution");
        h_dz.SetXTitle("dz");
        h_dz.Draw();
        c.SetGrid();                         
        c.SaveAs("dz.png");                                                                                                                  
        c.Close();      
	gSystem->Exec("mv dz.png Kinematical_Plots");
//---------------------------------------------------------------------------------------------------

}


