#ifndef Histograms_H
#define Histograms_H

/** \class Histograms
 *  Collection of histograms for GLB muon analysis
 *
 *  $Date: 2008/07/03 10:39:21 $
 *  $Revision: 1.1 $
 *  \author S. Bolognesi - INFN Torino / T.Dorigo - INFN Padova
 */

#include <CLHEP/Vector/LorentzVector.h>
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "MuScleFitUtils.h"

#include "TH1D.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TFile.h"
#include "TString.h"
#include "TProfile.h"
#include "TF1.h"
#include "TGraphErrors.h"
#include "TFile.h"
#include "TSystem.h"

#include <vector>
#include <string>
#include <iostream>
#include "TMath.h"

using namespace std;
using std::cout;
using std::endl;

class Histograms {
public:

  // Constructor
  // -----------
  Histograms() : theWeight(1) {};

  // Destructor
  // ----------
  virtual ~Histograms() {};

  // Operations
  // ----------
  virtual void Fill (reco::Particle::LorentzVector p4) {};
  virtual void Fill (HepLorentzVector momentum) {};
  virtual void Fill (reco::Particle::LorentzVector p1, reco::Particle::LorentzVector p2) {};
  virtual void Fill (reco::Particle::LorentzVector p1, reco::Particle::LorentzVector p2, int charge) {};
  virtual void Fill (HepLorentzVector momentum1, HepLorentzVector momentum2) {};
  virtual void Fill (HepLorentzVector momentum1, HepLorentzVector momentum2, int charge) {};
  virtual void Fill (HepLorentzVector p1, reco::Particle::LorentzVector p2) {};
  virtual void Fill (reco::Particle::LorentzVector p4, double likeValue) {};
  virtual void Fill (reco::Particle::LorentzVector p4, double genValue, double recValue) {};
  virtual void Fill (HepLorentzVector p, double likeValue) {};
  virtual void Fill (int number) {};
  
  virtual void Write() = 0;
  virtual void Clear() = 0;

  virtual void SetWeight (double weight) {
    theWeight = weight;
  }

   virtual TString GetName() {
    return name;
  }

protected:
  double theWeight;
  TString name;

private:

};

// -----------------------------------------------------
// A set of histograms of particle kinematical variables
// -----------------------------------------------------
class HParticle : public Histograms {
 public:
  HParticle (std::string name_) {
    TString N = name_.c_str();
    name=N;
    // Kinematical variables
    hPt        = new TH1F (N+"_Pt", "transverse momentum", 40, 0, 100);
    hEta       = new TH1F (N+"_Eta", "pseudorapidity", 30, -6, 6);
    hPhi       = new TH1F (N+"_Phi", "phi angle", 32, -3.2, 3.2);
    hMass      = new TH1F (N+"_Mass", "mass", 40000, 0, 200);
    // hMass_fine = new TH1F (N+"_Mass_fine", "low mass fine binning", 4000, 0., 20. ); //Removed to avoid too many histos (more binning added to hMass)
    hNumber    = new TH1F (N+"_Number", "number", 20, -0.5, 19.5);
   }
  
  HParticle (TString name_, TFile* file) {
    name=name_;
    hPt        = (TH1F *) file->Get(name+"_Pt");
    hEta       = (TH1F *) file->Get(name+"_Eta");
    hPhi       = (TH1F *) file->Get(name+"_Phi");
    hMass      = (TH1F *) file->Get(name+"_Mass");
    //hMass_fine = (TH1F *) file->Get(name+"_Mass_fine");
    hNumber    = (TH1F *) file->Get(name+"_Number");
  }

  ~HParticle() {

    //     delete hDist;
    //     delete hRes;
    //     delete hResVsEta;
    //     delete hResVsPhi;
    //     delete hResVsPos;
    //     delete hPull;
  } 

  virtual void Fill (reco::Particle::LorentzVector p4) {
    Fill (HepLorentzVector(p4.x(),p4.y(),p4.z(),p4.t()));
  }

  virtual void Fill (HepLorentzVector momentum) {
    hPt->Fill (momentum.perp());
    hEta->Fill (momentum.eta());
    hPhi->Fill (momentum.phi());
    hMass->Fill (momentum.m());
    //hMass_fine->Fill (momentum.m());
  }
  
  virtual void Fill (int number) {
    hNumber->Fill (number);
  }

  virtual void Write() {
    hPt->Write();
    hEta->Write();    
    hPhi->Write();
    hMass->Write();
    //hMass_fine->Write();
    hNumber->Write();
  }
  
  virtual void Clear() {
    hPt->Clear();
    hEta->Clear();    
    hPhi->Clear();
    hMass->Clear();
    //hMass_fine->Clear();
    hNumber->Clear();
  }
  
 public:
  TH1F* hPt;
  TH1F* hEta;
  TH1F* hPhi;
  TH1F* hMass;
  //TH1F* hMass_fine;
  TH1F* hNumber;
 
  TString name;

};

// ---------------------------------------------------
// A set of histograms for distances between particles
// ---------------------------------------------------
class HDelta : public Histograms {
 public:
  HDelta (std::string name_) {
    TString N = name_.c_str();
    name = N;

    // Kinematical variables
    // ---------------------
    hEta    = new TH1F (N+"_Eta", "pseudorapidity", 100, -6, 6);
    hPhi    = new TH1F (N+"_Phi", "phi angle", 100,0,3.2);
    hTheta  = new TH1F (N+"_Theta", "theta angle", 100,-3.2,3.2);
    hCotgTheta  = new TH1F (N+"_CotgTheta", "cotangent theta angle", 100,-3.2,3.2);
    hDeltaR = new TH1F (N+"_DeltaR","DeltaR", 400, 0, 10 );
   }
  
  HDelta (TString name_, TFile* file) {
    name = name_;
    hEta       = (TH1F *) file->Get(name+"_Eta");
    hPhi       = (TH1F *) file->Get(name+"_Phi");
    hTheta     = (TH1F *) file->Get(name+"_Theta");
    hCotgTheta = (TH1F *) file->Get(name+"_CotgTheta");
    hDeltaR    = (TH1F *) file->Get(name+"_DeltaR");
   }

  ~HDelta() {
    //     delete hDist;
    //     delete hRes;
    //     delete hResVsEta;
    //     delete hResVsPhi;
    //     delete hResVsPos;
    //     delete hPull;
  }
  
  virtual void Fill (reco::Particle::LorentzVector p1, reco::Particle::LorentzVector p2) {
    Fill (HepLorentzVector(p1.x(),p1.y(),p1.z(),p1.t()), 
	  HepLorentzVector(p2.x(),p2.y(),p2.z(),p2.t()));
  }

  virtual void Fill (HepLorentzVector p1, reco::Particle::LorentzVector p2) {
    Fill (p1,HepLorentzVector(p2.x(),p2.y(),p2.z(),p2.t()));
  }

  virtual void Fill (HepLorentzVector momentum1, HepLorentzVector momentum2) {
    hEta->Fill(momentum1.eta()-momentum2.eta());
    hPhi->Fill(MuScleFitUtils::deltaPhi(momentum1.phi(),momentum2.phi()));
    hTheta->Fill(momentum1.theta()-momentum2.theta());
    hCotgTheta->Fill(1/(TMath::Tan(momentum1.theta()))-1/(TMath::Tan(momentum2.theta())));
    hDeltaR->Fill(sqrt((momentum1.eta()-momentum2.eta())*(momentum1.eta()-momentum2.eta()) +
		       (MuScleFitUtils::deltaPhi(momentum1.phi(),momentum2.phi()))*
		       (MuScleFitUtils::deltaPhi(momentum1.phi(),momentum2.phi()))));  
  }
  
  virtual void Write() {
    hEta->Write();    
    hPhi->Write();
    hTheta->Write();    
    hCotgTheta->Write();    
    hDeltaR->Write();
  }
  
  virtual void Clear() {
    hEta->Clear();    
    hPhi->Clear();
    hTheta->Clear();    
    hDeltaR->Clear();
    hCotgTheta->Clear();    
  }
  
 public:
  TH1F* hEta;
  TH1F* hPhi;
  TH1F* hTheta;
  TH1F* hCotgTheta;
  TH1F* hDeltaR;
  
  TString name;
  
};


// ------------------------------------------------------------
// A set of histograms of particle kinematical variables vs eta
// ------------------------------------------------------------
class HPartVSEta : public Histograms {
 public:
  HPartVSEta(std::string name_) {
    TString N = name_.c_str();
    name = N;
    // Eta bins
    // hForw  = new HParticle (name_+"_Forw");
    // hWm2   = new HParticle (name_+"_Wm2"); 
    // hWm1   = new HParticle (name_+"_Wm1"); 
    // hW0    = new HParticle (name_+"_W0"); 
    // hWp1   = new HParticle (name_+"_Wp1"); 
    // hWp2   = new HParticle (name_+"_Wp2"); 
    // hBackw = new HParticle (name_+"_Backw"); 
    
    hPtVSEta = new TH2F (N+"_PtVSEta", "transverse momentum vs pseudorapidity", 
			 12, -6, 6, 200, 0, 200);
    hMassVSEta = new TH2F (N+"_MassVSEta", "mass vs pseudorapidity", 
			   12, -6, 6, 40, 70, 110);
    // TD profile histograms
    // ---------------------
    hMassVSEta_prof = new TProfile (N+"_MassVSEta_prof", "mass vs pseudorapidity", 
				    12, -3, 3, 86, 116);
    hPtVSEta_prof = new TProfile (N+"_PtVSEta_prof", "mass vs pseudorapidity", 
				  12, -3, 3, 0, 200);
  }
  
  ~HPartVSEta() {
  }

  virtual void Fill (reco::Particle::LorentzVector p4) {
    Fill (HepLorentzVector(p4.x(),p4.y(),p4.z(),p4.t()));
  }

  virtual void Fill (HepLorentzVector momentum) {
    // if (momentum.eta()<-1.2)                              hBackw->Fill (momentum);
    //  else if (momentum.eta()<-0.8 && momentum.eta()>-1.2) hWm2->Fill (momentum);
    //  else if (momentum.eta()<-0.3 && momentum.eta()>-0.8) hWm1->Fill (momentum);
    //  else if ((fabs(momentum.eta())) < 0.3)               hW0 ->Fill (momentum);
    //  else if (momentum.eta()>0.3 && momentum.eta()<0.8)   hWp1->Fill (momentum);
    //  else if (momentum.eta()>0.8 && momentum.eta()<1.2)   hWp2->Fill (momentum);
    //  else if (momentum.eta()>1.2)                         hForw->Fill (momentum);

    hPtVSEta->Fill(momentum.eta(),momentum.perp());
    hPtVSEta_prof->Fill(momentum.eta(),momentum.perp());

    hMassVSEta->Fill(momentum.eta(),momentum.m());
    hMassVSEta_prof->Fill(momentum.eta(),momentum.m());
    
  }
    
  virtual void Write() {
    // hForw->Write();
    // hWm2->Write();
    // hWm1->Write();
    // hW0->Write();
    // hWp1->Write();
    // hWp2->Write();
    // hBackw->Write();

    hPtVSEta->Write();
    hPtVSEta_prof->Write();
    hMassVSEta->Write();
    hMassVSEta_prof->Write();
   
    vector<TGraphErrors*> graphs ((MuScleFitUtils::fitMass(hMassVSEta)));
    for (vector<TGraphErrors*>::const_iterator graph = graphs.begin(); graph != graphs.end(); graph++) {
      (*graph)->Write();
    }
  }
  
  virtual void Clear() {
    // hForw->Clear();
    // hWm2->Clear();
    // hWm1->Clear();
    // hW0->Clear();
    // hWp1->Clear();
    // hWp2->Clear();
    // hBackw->Clear();

    hPtVSEta->Clear();
    hPtVSEta_prof->Clear();
    hMassVSEta->Clear();
    hMassVSEta_prof->Clear();
  }
   
 public:

  /* HParticle *hForw;
  HParticle *hWm2;
  HParticle *hWm1;
  HParticle *hW0;
  HParticle *hWp1;
  HParticle *hWp2;
  HParticle *hBackw;
  */
  TH2F *hPtVSEta;
  TH2F *hMassVSEta; 
  TProfile *hMassVSEta_prof; 
  TProfile *hPtVSEta_prof; 
 
  TString name;

};

//---------------------------------------------------------------------------
// A set of histograms of particle kinematical variables vs phi (in eta bins)
// --------------------------------------------------------------------------
class HPartVSPhi : public Histograms{
 public:
  HPartVSPhi(std::string name_){
    TString N = name_.c_str();
    name=N;
    // Phi bins
    /* hSec1  = new HParticle (name_+"_Sec1");
    hSec2  = new HParticle (name_+"_Sec2");
    hSec3  = new HParticle (name_+"_Sec3");
    hSec4  = new HParticle (name_+"_Sec4");
    */
    hPtVSPhi = new TH2F (N+"_PtVSPhi", "transverse momentum vs phi angle",
			 12, -3.2, 3.2, 200, 0, 200);
    hMassVSPhi = new TH2F (N+"_MassVSPhi", "mass vs phi angle", 
			   7, -3.2, 3.2, 40, 70, 110);
    hMassVSPhiF = new TH2F (N+"_MassVSPhiF", "mass vs phi F", 
			    7, -3.2, 3.2, 40, 70, 110);
    hMassVSPhiWp2 = new TH2F (N+"_MassVSPhiWp2", "mass vs phi Wp2", 
			   7, -3.2, 3.2, 40, 70, 110);
    hMassVSPhiWp1 = new TH2F (N+"_MassVSPhiWp1", "mass vs phi Wp1", 
			      7, -3.2, 3.2, 40, 70, 110);
    hMassVSPhiW0 = new TH2F (N+"_MassVSPhiW0", "mass vs phi W0", 
			     7, -3.2, 3.2, 40, 70, 110);
    hMassVSPhiWm1 = new TH2F (N+"_MassVSPhiWm1", "mass vs phi Wm1", 
			      7, -3.2, 3.2, 40, 70, 110);
    hMassVSPhiWm2 = new TH2F (N+"_MassVSPhiWm2", "mass vs phi Wm2", 
			      7, -3.2, 3.2, 40, 70, 110);
    hMassVSPhiB = new TH2F (N+"_MassVSPhiB", "mass vs phi B", 
			    7, -3.2, 3.2, 40, 70, 110);  

    // TD profile histograms
    hMassVSPhi_prof = new TProfile (N+"_MassVSPhi_prof", "mass vs phi angle", 
				    12, -3.2, 3.2, 70, 110);
    hPtVSPhi_prof = new TProfile (N+"_PtVSPhi_prof", "pt vs phi angle", 
				    12, -3.2, 3.2, 0, 200);

  }

  ~HPartVSPhi(){
  }

  void Fill(reco::Particle::LorentzVector p4) {
    Fill(HepLorentzVector(p4.x(),p4.y(),p4.z(),p4.t()));
  }

  void Fill(HepLorentzVector momentum) {
    //if (momentum.phi()>1.57)                           hSec1->Fill(momentum);
    //else if (momentum.phi()<1.57 && momentum.phi()>0)  hSec2->Fill(momentum);
    //else if (momentum.phi()<0 && momentum.phi()>-1.57) hSec3->Fill(momentum);
    //else if (momentum.phi()<-1.57)                     hSec4->Fill(momentum);
    
    
    hPtVSPhi->Fill(momentum.phi(),momentum.perp());
    hMassVSPhi->Fill(momentum.phi(),momentum.m());
    hMassVSPhi_prof->Fill(momentum.phi(),momentum.m());
    hPtVSPhi_prof->Fill(momentum.phi(),momentum.perp());
 
    if (momentum.eta()<-1.2)                            hMassVSPhiB ->Fill(momentum.phi(),momentum.m());
    else if (momentum.eta()<-0.8 && momentum.eta()>-1.2)hMassVSPhiWm2->Fill(momentum.phi(),momentum.m());
    else if (momentum.eta()<-0.3 && momentum.eta()>-0.8)hMassVSPhiWm1->Fill(momentum.phi(),momentum.m());
    else if ((fabs(momentum.eta())) < 0.3)              hMassVSPhiW0 ->Fill(momentum.phi(),momentum.m());
    else if (momentum.eta()>0.3 && momentum.eta()<0.8)  hMassVSPhiWp1->Fill(momentum.phi(),momentum.m());
    else if (momentum.eta()>0.8 && momentum.eta()<1.2)  hMassVSPhiWp2->Fill(momentum.phi(),momentum.m());
    else if (momentum.eta()>1.2)                        hMassVSPhiF->Fill(momentum.phi(),momentum.m());
  }
  
  virtual void Write() {
    /*    hSec1->Write();
    hSec2->Write();
    hSec3->Write();
    hSec4->Write();
    */
    hPtVSPhi->Write();
    hMassVSPhi->Write();
    hMassVSPhi_prof->Write();
    hPtVSPhi_prof->Write();

    hMassVSPhiB->Write();
    hMassVSPhiWm2->Write(); 
    hMassVSPhiWm1 ->Write();
    hMassVSPhiW0 ->Write();
    hMassVSPhiWp1->Write();
    hMassVSPhiWp2 ->Write();
    hMassVSPhiF ->Write();

    vector<TGraphErrors*> graphs ((MuScleFitUtils::fitMass(hMassVSPhi)));
    for(vector<TGraphErrors*>::const_iterator graph = graphs.begin(); graph != graphs.end(); graph++){
      (*graph)->Write();
    }
    vector<TGraphErrors*> graphsB ((MuScleFitUtils::fitMass(hMassVSPhiB)));
    for(vector<TGraphErrors*>::const_iterator graph = graphsB.begin(); graph != graphsB.end(); graph++){
      (*graph)->Write();
    }
    vector<TGraphErrors*> graphsWm2 ((MuScleFitUtils::fitMass(hMassVSPhiWm2)));
    for(vector<TGraphErrors*>::const_iterator graph = graphsWm2.begin(); graph != graphsWm2.end(); graph++){
      (*graph)->Write();
    }
    vector<TGraphErrors*> graphsWm1 ((MuScleFitUtils::fitMass(hMassVSPhiWm1)));
    for(vector<TGraphErrors*>::const_iterator graph = graphsWm1.begin(); graph != graphsWm1.end(); graph++){
      (*graph)->Write();
    }
    vector<TGraphErrors*> graphsW0 ((MuScleFitUtils::fitMass(hMassVSPhiW0)));
    for(vector<TGraphErrors*>::const_iterator graph = graphsW0.begin(); graph != graphsW0.end(); graph++){
      (*graph)->Write();
    }
    vector<TGraphErrors*> graphsWp1 ((MuScleFitUtils::fitMass(hMassVSPhiWp1)));
    for(vector<TGraphErrors*>::const_iterator graph = graphsWp1.begin(); graph != graphsWp1.end(); graph++){
      (*graph)->Write();
    }
    vector<TGraphErrors*> graphsWp2 ((MuScleFitUtils::fitMass(hMassVSPhiWp2)));
    for(vector<TGraphErrors*>::const_iterator graph = graphsWp2.begin(); graph != graphsWp2.end(); graph++){
      (*graph)->Write();
    }
    vector<TGraphErrors*> graphsF ((MuScleFitUtils::fitMass(hMassVSPhiF)));
    for(vector<TGraphErrors*>::const_iterator graph = graphsF.begin(); graph != graphsF.end(); graph++){
      (*graph)->Write();
    }

  }

  virtual void Clear() {
    /*    hSec1->Clear();
    hSec2->Clear();
    hSec3->Clear();
    hSec4->Clear();
    */
    hPtVSPhi->Clear();
    hMassVSPhi->Clear();
    hPtVSPhi_prof->Clear();
    hMassVSPhi_prof->Clear();

    hMassVSPhiB->Clear();
    hMassVSPhiWm2->Clear(); 
    hMassVSPhiWm1 ->Clear();
    hMassVSPhiW0 ->Clear();
    hMassVSPhiWp1->Clear();
    hMassVSPhiWp2 ->Clear();
    hMassVSPhiF ->Clear();
  }
  
 public:
  /*HParticle *hSec1;
  HParticle *hSec2;
  HParticle *hSec3;
  HParticle *hSec4;
  */
  TH2F *hPtVSPhi;
  TH2F *hMassVSPhi;
  TProfile *hMassVSPhi_prof; 
  TProfile *hPtVSPhi_prof; 
 
  TH2F *hMassVSPhiB;
  TH2F *hMassVSPhiWm2;
  TH2F *hMassVSPhiWm1;
  TH2F *hMassVSPhiW0 ;
  TH2F *hMassVSPhiWp1;
  TH2F *hMassVSPhiWp2 ;
  TH2F *hMassVSPhiF;

  TString name;
 
};
//---------------------------------------------------------------------------------------
// A set of histograms of particle VS pt
class HPartVSPt : public Histograms{
 public:
  HPartVSPt(std::string name_){
    TString N = name_.c_str();
    name=N;
    hMassVSPt = new TH2F (N+"_MassVSPt", "mass vs transverse momentum", 
			   12, -6, 6, 40, 70, 110);
    // TD profile histograms
    hMassVSPt_prof = new TProfile (N+"_MassVSPt_prof", "mass vs transverse momentum", 
			   12, -3, 3, 86, 116);
  }

  ~HPartVSPt(){
  }

  virtual void Fill(reco::Particle::LorentzVector p4) {
    Fill(HepLorentzVector(p4.x(),p4.y(),p4.z(),p4.t()));
  }

  virtual void Fill(HepLorentzVector momentum) {
    hMassVSPt->Fill(momentum.eta(),momentum.m());
    hMassVSPt_prof->Fill(momentum.eta(),momentum.m());    
  }
    
  virtual void Write() {
    hMassVSPt->Write();
    hMassVSPt_prof->Write();
   
    vector<TGraphErrors*> graphs ((MuScleFitUtils::fitMass(hMassVSPt)));
    for(vector<TGraphErrors*>::const_iterator graph = graphs.begin(); graph != graphs.end(); graph++){
    (*graph)->Write();
      }
  }
  
  virtual void Clear() {
    hMassVSPt->Clear();
    hMassVSPt_prof->Clear();
  }
   
 public:
  TH2F *hMassVSPt; 
  TProfile *hMassVSPt_prof; 
 
  TString name;

};

// ---------------------------------------------------
// A set of histograms of Z mass versus muon variables

class HMassVSPart : public Histograms{
 public:
  HMassVSPart(std::string name_){
    TString N = name_.c_str();
    name=N;

    // Kinematical variables
    // ---------------------
    hMassVSPt     = new TH2F (N+"_MassVSPt", "re sonance mass vs muon transverse momentum", 200, 0., 200., 1200, 0, 150.);
    hMassVSEta    = new TH2F (N+"_MassVSEta", "resonance mass vs muon pseudorapidity", 30, -6., 6., 1200, 0, 150.);
    hMassVSPhiPlus    = new TH2F (N+"_MassVSPhiPlus", "resonance mass vs muon+ phi angle", 32, -3.2, 3.2, 1200, 0, 150.);
    hMassVSPhiMinus    = new TH2F (N+"_MassVSPhiMinus", "resonance mass vs muon- phi angle", 32, -3.2, 3.2, 1200, 0, 150.);
    hMassVSPt_prof     = new TProfile (N+"_MassVSPt_prof", "resonance mass vs muon transverse momentum", 100, 0., 200., 0, 150.);
    hMassVSEta_prof    = new TProfile (N+"_MassVSEta_prof", "resonance mass vs muon pseudorapidity", 30, -6., 6., 0, 150.);
    hMassVSPhiPlus_prof    = new TProfile (N+"_MassVSPhiPlus_prof", "resonance mass vs muon+ phi angle", 32, -3.2, 3.2, 0, 150.);
    hMassVSPhiMinus_prof    = new TProfile (N+"_MassVSPhiMinus_prof", "resonance mass vs muon- phi angle", 32, -3.2, 3.2, 0, 150.);
   }
  
  HMassVSPart(TString name_, TFile* file){
    name=name_;
    hMassVSPt     = (TH2F *) file->Get(name+"_MassVSPt");
    hMassVSEta    = (TH2F *) file->Get(name+"_MassVSEta");
    hMassVSPhiPlus    = (TH2F *) file->Get(name+"_MassVSPhiPlus");
    hMassVSPhiMinus    = (TH2F *) file->Get(name+"_MassVSPhiMinus");
    hMassVSPt_prof     = (TProfile *) file->Get(name+"_MassVSPt_prof");
    hMassVSEta_prof    = (TProfile *) file->Get(name+"_MassVSEta_prof");
    hMassVSPhiPlus_prof    = (TProfile *) file->Get(name+"_MassVSPhiPlus_prof");
    hMassVSPhiMinus_prof    = (TProfile *) file->Get(name+"_MassVSPhiMinus_prof");
  }

  ~HMassVSPart(){
    // Do not delete anything...
  } 

  virtual void Fill(reco::Particle::LorentzVector p41, reco::Particle::LorentzVector p42, int charge) {
    Fill(HepLorentzVector(p41.x(),p41.y(),p41.z(),p41.t()),
	 HepLorentzVector(p42.x(),p42.y(),p42.z(),p42.t()), charge);
  }
  
   virtual void Fill(HepLorentzVector momentum1, HepLorentzVector momentum2, int charge) { 
     hMassVSPt->Fill(momentum1.perp(),momentum2.m()); 
     hMassVSPt_prof->Fill(momentum1.perp(),momentum2.m()); 
     hMassVSEta->Fill(momentum1.eta(),momentum2.m()); 
     hMassVSEta_prof->Fill(momentum1.eta(),momentum2.m()); 
     if(charge>0){
       hMassVSPhiPlus->Fill(momentum1.phi(),momentum2.m()); 
       hMassVSPhiPlus_prof->Fill(momentum1.phi(),momentum2.m()); 
     }
     else if(charge<0){
       hMassVSPhiMinus->Fill(momentum1.phi(),momentum2.m()); 
       hMassVSPhiMinus_prof->Fill(momentum1.phi(),momentum2.m()); 
     }
     else 
       abort();
   } 
  
  virtual void Write() {
    hMassVSPt->Write();
    hMassVSEta->Write();    
    hMassVSPhiPlus->Write();
    hMassVSPhiMinus->Write();
    hMassVSPt_prof->Write();
    hMassVSEta_prof->Write();    
    hMassVSPhiPlus_prof->Write();
    hMassVSPhiMinus_prof->Write();

    vector<TGraphErrors*> graphPt ((MuScleFitUtils::fitMass(hMassVSPt)));
    for(vector<TGraphErrors*>::const_iterator graph = graphPt.begin(); graph != graphPt.end(); graph++){
      (*graph)->Write();
    }   
    vector<TGraphErrors*> graphPhiPlus ((MuScleFitUtils::fitMass(hMassVSPhiPlus)));
    for(vector<TGraphErrors*>::const_iterator graph = graphPhiPlus.begin(); graph != graphPhiPlus.end(); graph++){
      (*graph)->Write();
    }   
    vector<TGraphErrors*> graphPhiMinus ((MuScleFitUtils::fitMass(hMassVSPhiMinus)));
    for(vector<TGraphErrors*>::const_iterator graph = graphPhiMinus.begin(); graph != graphPhiMinus.end(); graph++){
      (*graph)->Write();
    }   
    vector<TGraphErrors*> graphEta ((MuScleFitUtils::fitMass(hMassVSEta)));
    for(vector<TGraphErrors*>::const_iterator graph = graphEta.begin(); graph != graphEta.end(); graph++){
      (*graph)->Write();
    }   
  }
  
  virtual void Clear() {
    hMassVSPt->Clear();
    hMassVSEta->Clear();    
    hMassVSPhiPlus->Clear();
    hMassVSPhiMinus->Clear();
    hMassVSPt_prof->Clear();
    hMassVSEta_prof->Clear();    
    hMassVSPhiPlus_prof->Clear();
    hMassVSPhiMinus_prof->Clear();
   }
  
 public:
  TH2F* hMassVSPt;
  TH2F* hMassVSEta;
  TH2F* hMassVSPhiPlus; 
  TH2F* hMassVSPhiMinus; 
  TProfile* hMassVSPt_prof;
  TProfile* hMassVSEta_prof;
  TProfile* hMassVSPhiPlus_prof;
  TProfile* hMassVSPhiMinus_prof;
 
  TString name;

};

//---------------------------------------------------------------------------------------
/// A set of histograms for resolution
class HResolutionVSPart : public Histograms{
 public:
  HResolutionVSPart(std::string name_){
    TString N = name_.c_str();
    name=N;
    // Kinematical variables
    hReso    = new TH1F (N+"_Reso", "resolution", 200, -1, 1);
    hResoVSPt    = new TH2F (N+"_ResoVSPt", "resolution VS pt", 200, 0, 200, 500, -1, 1);
    hResoVSPt_prof = new TProfile (N+"_ResoVSPt_prof", "resolution VS pt", 100, 0, 200, -1, 1);
    hResoVSEta    = new TH2F (N+"_ResoVSEta", "resolution VS eta", 10, -2.5, 2.5, 500, -1, 1);
    hResoVSEta_prof = new TProfile (N+"_ResoVSEta_prof", "resolution VS eta", 10, -2.5, 2.5, -1, 1);
    hResoVSPhi    = new TH2F (N+"_ResoVSPhi", "resolution VS phi", 14, -3.2, 3.2, 500, -1, 1);
    hResoVSPhi_prof = new TProfile (N+"_ResoVSPhi_prof", "resolution VS phi", 14, -3.2, 3.2, -1, 1);
    hAbsReso    = new TH1F (N+"_AbsReso", "resolution", 100, 0, 1);
    hAbsResoVSPt    = new TH2F (N+"_AbsResoVSPt", "Abs resolution VS pt", 200, 0, 500, 100, 0, 1);
    hAbsResoVSEta    = new TH2F (N+"_AbsResoVSEta", "Abs resolution VS eta", 10, -2.5, 2.5, 100, 0, 1);
    hAbsResoVSPhi    = new TH2F (N+"_AbsResoVSPhi", "Abs resolution VS phi", 14, -3.2, 3.2, 100, 0, 1);
  }
  
  HResolutionVSPart(TString name_, TFile* file){
    name=name_;
    hReso      = (TH1F *) file->Get(name+"_Reso");
    hResoVSPt  = (TH2F *) file->Get(name+"_ResoVSPt");
    hResoVSPt_prof  = (TProfile *) file->Get(name+"_ResoVSPt_prof");
    hResoVSEta  = (TH2F *) file->Get(name+"_ResoVSEta");
    hResoVSEta_prof  = (TProfile *) file->Get(name+"_ResoVSEta_prof");
    hResoVSPhi  = (TH2F *) file->Get(name+"_ResoVSPhi");
    hResoVSPhi_prof  = (TProfile *) file->Get(name+"_ResoVSPhi_prof");
    hAbsReso      = (TH1F *) file->Get(name+"_AbsReso");
    hAbsResoVSPt  = (TH2F *) file->Get(name+"_AbsResoVSPt");
    hAbsResoVSEta  = (TH2F *) file->Get(name+"_AbsResoVSEta");
    hAbsResoVSPhi  = (TH2F *) file->Get(name+"_AbsResoVSPhi");
   }

  ~HResolutionVSPart(){
  }

   virtual void Fill(reco::Particle::LorentzVector p4, double resValue) { 
     hReso->Fill(resValue); 
     hResoVSPt->Fill(p4.Pt(),resValue); 
     hResoVSEta->Fill(p4.Eta(),resValue); 
     hResoVSPhi->Fill(p4.Phi(),resValue); 
     hResoVSPt_prof->Fill(p4.Pt(),resValue); 
     hResoVSEta_prof->Fill(p4.Eta(),resValue); 
     hResoVSPhi_prof->Fill(p4.Phi(),resValue); 
     hAbsReso->Fill(fabs(resValue)); 
     hAbsResoVSPt->Fill(p4.Pt(),fabs(resValue)); 
     hAbsResoVSEta->Fill(p4.Eta(),fabs(resValue)); 
     hAbsResoVSPhi->Fill(p4.Phi(),fabs(resValue));     
   } 
  
  virtual void Fill(reco::Particle::LorentzVector p4, double genValue, double recValue) {//1 sim, 2 rec
    Fill(p4, (recValue-genValue)/genValue);
  }

  virtual void Write() {
    hReso->Write();
    hResoVSPt->Write();
    hResoVSPt_prof->Write();
    hResoVSEta->Write();
    hResoVSEta_prof->Write();
    hResoVSPhi->Write();
    hResoVSPhi_prof->Write();
    hAbsReso->Write();
    hAbsResoVSPt->Write();
    hAbsResoVSEta->Write();
    hAbsResoVSPhi->Write();

    vector<TGraphErrors*> graphs ((MuScleFitUtils::fitReso(hResoVSPt)));
    for(vector<TGraphErrors*>::const_iterator graph = graphs.begin(); graph != graphs.end(); graph++){
      (*graph)->Write();
    }

    vector<TGraphErrors*> graphsEta ((MuScleFitUtils::fitReso(hResoVSEta)));
    for(vector<TGraphErrors*>::const_iterator graph = graphsEta.begin(); graph != graphsEta.end(); graph++){
      (*graph)->Write();
    }

    vector<TGraphErrors*> graphsPhi ((MuScleFitUtils::fitReso(hResoVSPhi)));
    for(vector<TGraphErrors*>::const_iterator graph = graphsPhi.begin(); graph != graphsPhi.end(); graph++){
      (*graph)->Write();
    }
 }
  
  virtual void Clear() {
    hReso->Clear();    
    hResoVSPt->Clear();    
    hResoVSPt_prof->Clear();    
    hResoVSEta->Clear();    
    hResoVSEta_prof->Clear();    
    hResoVSPhi->Clear();    
    hResoVSPhi_prof->Clear();    
    hAbsReso->Clear();
    hAbsResoVSPt->Clear();
    hAbsResoVSEta->Clear();
    hAbsResoVSPhi->Clear();

 }
  
 public:
  TH1F* hReso;
  TH2F* hResoVSPt;
  TProfile* hResoVSPt_prof;
  TH2F* hResoVSEta;
  TProfile* hResoVSEta_prof;
  TH2F* hResoVSPhi;
  TProfile* hResoVSPhi_prof;
  TH1F* hAbsReso;
  TH2F* hAbsResoVSPt;
  TH2F* hAbsResoVSEta;
  TH2F* hAbsResoVSPhi;

  TString name;

};
// -------------------------------------------------------------
// A set of histograms of likelihood value versus muon variables
// -------------------------------------------------------------
class HLikelihoodVSPart : public Histograms{
 public:
  HLikelihoodVSPart(std::string name_){
    TString N = name_.c_str();
    name=N;

    // Kinematical variables
    // ---------------------
    hLikeVSPt     = new TH2F (N+"_LikelihoodVSPt", "likelihood vs muon transverse momentum", 100, 0., 100., 100, -100., 100.);
    hLikeVSEta    = new TH2F (N+"_LikelihoodVSEta", "likelihood vs muon pseudorapidity", 100, -4.,4., 100, -100., 100.);
    hLikeVSPhi    = new TH2F (N+"_LikelihoodVSPhi", "likelihood vs muon phi angle", 100, -3.2, 3.2, 100, -100., 100.);
    hLikeVSPt_prof     = new TProfile (N+"_LikelihoodVSPt_prof", "likelihood vs muon transverse momentum", 40, 0., 100., -1000., 1000. );
    hLikeVSEta_prof    = new TProfile (N+"_LikelihoodVSEta_prof", "likelihood vs muon pseudorapidity", 40, -4.,4., -1000., 1000. );
    hLikeVSPhi_prof    = new TProfile (N+"_LikelihoodVSPhi_prof", "likelihood vs muon phi angle", 40, -3.2, 3.2, -1000., 1000.);
   }
  
  HLikelihoodVSPart(TString name_, TFile* file){
    name=name_;
    hLikeVSPt     = (TH2F *) file->Get(name+"_LikelihoodVSPt");
    hLikeVSEta    = (TH2F *) file->Get(name+"_LikelihoodVSEta");
    hLikeVSPhi    = (TH2F *) file->Get(name+"_LikelihoodVSPhi");
    hLikeVSPt_prof     = (TProfile *) file->Get(name+"_LikelihoodVSPt_prof");
    hLikeVSEta_prof    = (TProfile *) file->Get(name+"_LikelihoodVSEta_prof");
    hLikeVSPhi_prof    = (TProfile *) file->Get(name+"_LikelihoodVSPhi_prof");
  }

  ~HLikelihoodVSPart(){
    // Do not delete anything...
  } 

  virtual void Fill(reco::Particle::LorentzVector p4, double likeValue) {
    Fill(HepLorentzVector(p4.x(),p4.y(),p4.z(),p4.t()), likeValue);
  }
  
   virtual void Fill(HepLorentzVector momentum, double likeValue) { 
     hLikeVSPt->Fill(momentum.perp(),likeValue); 
     hLikeVSEta->Fill(momentum.eta(),likeValue); 
     hLikeVSPhi->Fill(momentum.phi(),likeValue); 
     hLikeVSPt_prof->Fill(momentum.perp(),likeValue); 
     hLikeVSEta_prof->Fill(momentum.eta(),likeValue); 
     hLikeVSPhi_prof->Fill(momentum.phi(),likeValue); 
   } 
  
  virtual void Write() {
    hLikeVSPt->Write();
    hLikeVSEta->Write();    
    hLikeVSPhi->Write();
    hLikeVSPt_prof->Write();
    hLikeVSEta_prof->Write();    
    hLikeVSPhi_prof->Write();
  }
  
  virtual void Clear() {
    hLikeVSPt->Reset("ICE");
    hLikeVSEta->Reset("ICE");
    hLikeVSPhi->Reset("ICE");
    hLikeVSPt_prof->Reset("ICE");
    hLikeVSEta_prof->Reset("ICE");    
    hLikeVSPhi_prof->Reset("ICE");
   }
  
 public:
  TH2F* hLikeVSPt;
  TH2F* hLikeVSEta;
  TH2F* hLikeVSPhi; 
  TProfile* hLikeVSPt_prof;
  TProfile* hLikeVSEta_prof;
  TProfile* hLikeVSPhi_prof;
 
  TString name;

};

#endif
