#ifndef Histograms_H
#define Histograms_H

/** \class Histograms
 *  Collection of histograms for GLB muon analysis
 *
 *  $Date: 2008/10/09 15:38:29 $
 *  $Revision: 1.6 $
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
  Histograms() : theWeight_(1), histoDir_(0) {};
  Histograms( const TString & name ) : theWeight_(1), name_(name), histoDir_(0) {};
  Histograms( TFile * outputFile, const TString & name ) :
    theWeight_(1),
    name_(name),
    histoDir_( outputFile->mkdir(name) ) {
    histoDir_->cd();
  };

  // Destructor
  // ----------
  virtual ~Histograms() {};

  // Operations
  // ----------
  virtual void Fill (const reco::Particle::LorentzVector & p4) {};
  virtual void Fill (const HepLorentzVector & momentum) {};
  virtual void Fill (const reco::Particle::LorentzVector & p1, const reco::Particle::LorentzVector & p2) {};
  virtual void Fill (const reco::Particle::LorentzVector & p1, const reco::Particle::LorentzVector & p2, const int charge) {};
  virtual void Fill (const HepLorentzVector & momentum1, const HepLorentzVector & momentum2) {};
  virtual void Fill (const HepLorentzVector & momentum1, const HepLorentzVector & momentum2, const int charge) {};
  virtual void Fill (const HepLorentzVector & p1, const reco::Particle::LorentzVector & p2) {};
  virtual void Fill (const reco::Particle::LorentzVector & p4, const double & likeValue) {};
  virtual void Fill (const reco::Particle::LorentzVector & p4, const double & resValue, const int charge) {};
  virtual void Fill (const reco::Particle::LorentzVector & p4, const double & genValue, const double recValue, const int charge) {};
  virtual void Fill (const HepLorentzVector & p, const double & likeValue) {};
  virtual void Fill (const int & number) {};
  virtual void Fill( const reco::Particle::LorentzVector & recoP1, const int charge1,
                     // const reco::Particle::LorentzVector & genP1,
                     const reco::Particle::LorentzVector & recoP2, const int charge2,
                     // const reco::Particle::LorentzVector & genP2,
                     const double & recoMass, const double & genMass ) {};
  
  virtual void Write() = 0;
  virtual void Clear() = 0;

  virtual void SetWeight (double weight) {
    theWeight_ = weight;
  }

   virtual TString GetName() {
    return name_;
  }

protected:
  double theWeight_;
  TString name_;
  TDirectory * histoDir_;

private:

};

// -----------------------------------------------------
// A set of histograms of particle kinematical variables
// -----------------------------------------------------
class HParticle : public Histograms {
 public:
  HParticle (const TString & name) :
    Histograms(name),
    // Kinematical variables
    hPt(     new TH1F (name+"_Pt", "transverse momentum", 100, 0, 100) ),
    hEta(    new TH1F (name+"_Eta", "pseudorapidity", 60, -6, 6) ),
    hPhi(    new TH1F (name+"_Phi", "phi angle", 64, -3.2, 3.2) ),
    hMass(   new TH1F (name+"_Mass", "mass", 40000, 0, 200) ),
    // hMass_fine = new TH1F (name+"_Mass_fine", "low mass fine binning", 4000, 0., 20. ); //Removed to avoid too many histos (more binning added to hMass)
    hNumber( new TH1F (name+"_Number", "number", 20, -0.5, 19.5) )
  {}

  /// Constructor that puts the histograms inside a TDirectory
  HParticle (TFile* outputFile, const TString & name) :
    Histograms(outputFile, name)
  {
    // Kinematical variables
    hPt =     new TH1F (name+"_Pt", "transverse momentum", 100, 0, 100);
    hEta =    new TH1F (name+"_Eta", "pseudorapidity", 60, -6, 6);
    hPhi =    new TH1F (name+"_Phi", "phi angle", 64, -3.2, 3.2);
    hMass =   new TH1F (name+"_Mass", "mass", 40000, 0, 200);
    // hMass_fine = new TH1F (name+"_Mass_fine", "low mass fine binning", 4000, 0., 20. ); //Removed to avoid too many histos (more binning added to hMass)
    hNumber = new TH1F (name+"_Number", "number", 20, -0.5, 19.5);
  }
  
  HParticle (const TString & name, TFile* file) :
    Histograms(name),
    hPt(     (TH1F *) file->Get(name_+"_Pt") ),
    hEta(    (TH1F *) file->Get(name_+"_Eta") ),
    hPhi(    (TH1F *) file->Get(name_+"_Phi") ),
    hMass(   (TH1F *) file->Get(name_+"_Mass") ),
    //hMass_fine = (TH1F *) file->Get(name_+"_Mass_fine");
    hNumber( (TH1F *) file->Get(name_+"_Number") )
  {}

  ~HParticle() {}

  virtual void Fill (const reco::Particle::LorentzVector & p4) {
    Fill(HepLorentzVector(p4.x(),p4.y(),p4.z(),p4.t()));
  }

  virtual void Fill (HepLorentzVector momentum) {
    hPt->Fill(momentum.perp());
    hEta->Fill(momentum.eta());
    hPhi->Fill(momentum.phi());
    hMass->Fill(momentum.m());
    //hMass_fine->Fill(momentum.m());
  }
  
  virtual void Fill (int number) {
    hNumber->Fill (number);
  }

  virtual void Write() {
    if(histoDir_ != 0) histoDir_->cd();

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
  
 protected:
  TH1F* hPt;
  TH1F* hEta;
  TH1F* hPhi;
  TH1F* hMass;
  //TH1F* hMass_fine;
  TH1F* hNumber;

};

// ---------------------------------------------------
// A set of histograms for distances between particles
// ---------------------------------------------------
class HDelta : public Histograms {
 public:
  HDelta (const TString & name) {
    name_=name;

    // Kinematical variables
    // ---------------------
    hEta    = new TH1F (name+"_Eta", "pseudorapidity", 100, -6, 6);
    hPhi    = new TH1F (name+"_Phi", "phi angle", 100,0,3.2);
    hTheta  = new TH1F (name+"_Theta", "theta angle", 100,-3.2,3.2);
    hCotgTheta  = new TH1F (name+"_CotgTheta", "cotangent theta angle", 100,-3.2,3.2);
    hDeltaR = new TH1F (name+"_DeltaR","DeltaR", 400, 0, 10 );
   }
  
  HDelta (const TString & name, TFile* file) {
    name_ = name;
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
  
  virtual void Fill (const reco::Particle::LorentzVector & p1, const reco::Particle::LorentzVector & p2) {
    Fill (HepLorentzVector(p1.x(),p1.y(),p1.z(),p1.t()), 
	  HepLorentzVector(p2.x(),p2.y(),p2.z(),p2.t()));
  }

  virtual void Fill (const HepLorentzVector & p1, const reco::Particle::LorentzVector & p2) {
    Fill (p1,HepLorentzVector(p2.x(),p2.y(),p2.z(),p2.t()));
  }

  virtual void Fill (const HepLorentzVector & momentum1, const HepLorentzVector & momentum2) {
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
  
};


// ------------------------------------------------------------
// A set of histograms of particle kinematical variables vs eta
// ------------------------------------------------------------
class HPartVSEta : public Histograms {
 public:
  HPartVSEta(const TString & name) {
    name_=name;
    // Eta bins
    // hForw  = new HParticle (name_+"_Forw");
    // hWm2   = new HParticle (name_+"_Wm2"); 
    // hWm1   = new HParticle (name_+"_Wm1"); 
    // hW0    = new HParticle (name_+"_W0"); 
    // hWp1   = new HParticle (name_+"_Wp1"); 
    // hWp2   = new HParticle (name_+"_Wp2"); 
    // hBackw = new HParticle (name_+"_Backw"); 
    
    hPtVSEta = new TH2F (name+"_PtVSEta", "transverse momentum vs pseudorapidity", 
			 12, -6, 6, 200, 0, 200);
    hMassVSEta = new TH2F (name+"_MassVSEta", "mass vs pseudorapidity", 
			   12, -6, 6, 40, 70, 110);
    // TD profile histograms
    // ---------------------
    hMassVSEta_prof = new TProfile (name+"_MassVSEta_prof", "mass vs pseudorapidity", 
				    12, -3, 3, 86, 116);
    hPtVSEta_prof = new TProfile (name+"_PtVSEta_prof", "mass vs pseudorapidity", 
				  12, -3, 3, 0, 200);
  }
  
  ~HPartVSEta() {
  }

  virtual void Fill (const reco::Particle::LorentzVector & p4) {
    Fill (HepLorentzVector(p4.x(),p4.y(),p4.z(),p4.t()));
  }

  virtual void Fill (const HepLorentzVector & momentum) {
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
 
};

//---------------------------------------------------------------------------
// A set of histograms of particle kinematical variables vs phi (in eta bins)
// --------------------------------------------------------------------------
class HPartVSPhi : public Histograms{
 public:
  HPartVSPhi(const TString & name){
    name_ = name;
    // Phi bins
    /* hSec1  = new HParticle (name_+"_Sec1");
    hSec2  = new HParticle (name_+"_Sec2");
    hSec3  = new HParticle (name_+"_Sec3");
    hSec4  = new HParticle (name_+"_Sec4");
    */
    hPtVSPhi = new TH2F (name+"_PtVSPhi", "transverse momentum vs phi angle",
			 12, -3.2, 3.2, 200, 0, 200);
    hMassVSPhi = new TH2F (name+"_MassVSPhi", "mass vs phi angle", 
			   7, -3.2, 3.2, 40, 70, 110);
    hMassVSPhiF = new TH2F (name+"_MassVSPhiF", "mass vs phi F", 
			    7, -3.2, 3.2, 40, 70, 110);
    hMassVSPhiWp2 = new TH2F (name+"_MassVSPhiWp2", "mass vs phi Wp2", 
			   7, -3.2, 3.2, 40, 70, 110);
    hMassVSPhiWp1 = new TH2F (name+"_MassVSPhiWp1", "mass vs phi Wp1", 
			      7, -3.2, 3.2, 40, 70, 110);
    hMassVSPhiW0 = new TH2F (name+"_MassVSPhiW0", "mass vs phi W0", 
			     7, -3.2, 3.2, 40, 70, 110);
    hMassVSPhiWm1 = new TH2F (name+"_MassVSPhiWm1", "mass vs phi Wm1", 
			      7, -3.2, 3.2, 40, 70, 110);
    hMassVSPhiWm2 = new TH2F (name+"_MassVSPhiWm2", "mass vs phi Wm2", 
			      7, -3.2, 3.2, 40, 70, 110);
    hMassVSPhiB = new TH2F (name+"_MassVSPhiB", "mass vs phi B", 
			    7, -3.2, 3.2, 40, 70, 110);  

    // TD profile histograms
    hMassVSPhi_prof = new TProfile (name+"_MassVSPhi_prof", "mass vs phi angle", 
				    12, -3.2, 3.2, 70, 110);
    hPtVSPhi_prof = new TProfile (name+"_PtVSPhi_prof", "pt vs phi angle", 
				    12, -3.2, 3.2, 0, 200);

  }

  ~HPartVSPhi(){
  }

  void Fill(const reco::Particle::LorentzVector & p4) {
    Fill(HepLorentzVector(p4.x(),p4.y(),p4.z(),p4.t()));
  }

  void Fill(const HepLorentzVector & momentum) {
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

};
//---------------------------------------------------------------------------------------
// A set of histograms of particle VS pt
class HPartVSPt : public Histograms{
 public:
  HPartVSPt(const TString & name){
    name_ = name;
    hMassVSPt = new TH2F (name+"_MassVSPt", "mass vs transverse momentum", 
			   12, -6, 6, 40, 70, 110);
    // TD profile histograms
    hMassVSPt_prof = new TProfile (name+"_MassVSPt_prof", "mass vs transverse momentum", 
			   12, -3, 3, 86, 116);
  }

  ~HPartVSPt(){
  }

  virtual void Fill(const reco::Particle::LorentzVector & p4) {
    Fill(HepLorentzVector(p4.x(),p4.y(),p4.z(),p4.t()));
  }

  virtual void Fill(const HepLorentzVector & momentum) {
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
};

// ---------------------------------------------------
// A set of histograms of Z mass versus muon variables

class HMassVSPart : public Histograms{
 public:
  HMassVSPart(const TString & name){
    name_ = name;

    // Kinematical variables
    // ---------------------
    hMassVSPt     = new TH2F (name+"_MassVSPt", "resonance mass vs muon transverse momentum", 200, 0., 200., 6000, 0, 150.);
    hMassVSEta    = new TH2F (name+"_MassVSEta", "resonance mass vs muon pseudorapidity", 60, -6., 6., 6000, 0, 150.);
    hMassVSPhiPlus    = new TH2F (name+"_MassVSPhiPlus", "resonance mass vs muon+ phi angle", 64, -3.2, 3.2, 6000, 0, 150.);
    hMassVSPhiMinus    = new TH2F (name+"_MassVSPhiMinus", "resonance mass vs muon- phi angle", 64, -3.2, 3.2, 6000, 0, 150.);
    //hMassVSPt_prof     = new TProfile (name+"_MassVSPt_prof", "resonance mass vs muon transverse momentum", 100, 0., 200., 0, 150.);
    //hMassVSEta_prof    = new TProfile (name+"_MassVSEta_prof", "resonance mass vs muon pseudorapidity", 30, -6., 6., 0, 150.);
    //hMassVSPhiPlus_prof    = new TProfile (name+"_MassVSPhiPlus_prof", "resonance mass vs muon+ phi angle", 32, -3.2, 3.2, 0, 150.);
    //hMassVSPhiMinus_prof    = new TProfile (name+"_MassVSPhiMinus_prof", "resonance mass vs muon- phi angle", 32, -3.2, 3.2, 0, 150.);
   }
  
  HMassVSPart(const TString & name, TFile* file){
    name_=name;
    hMassVSPt     = (TH2F *) file->Get(name+"_MassVSPt");
    hMassVSEta    = (TH2F *) file->Get(name+"_MassVSEta");
    hMassVSPhiPlus    = (TH2F *) file->Get(name+"_MassVSPhiPlus");
    hMassVSPhiMinus    = (TH2F *) file->Get(name+"_MassVSPhiMinus");
    //hMassVSPt_prof     = (TProfile *) file->Get(name+"_MassVSPt_prof");
    //hMassVSEta_prof    = (TProfile *) file->Get(name+"_MassVSEta_prof");
    //hMassVSPhiPlus_prof    = (TProfile *) file->Get(name+"_MassVSPhiPlus_prof");
    //hMassVSPhiMinus_prof    = (TProfile *) file->Get(name+"_MassVSPhiMinus_prof");
  }

  ~HMassVSPart(){
    // Do not delete anything...
  } 

  virtual void Fill(const reco::Particle::LorentzVector & p41, const reco::Particle::LorentzVector & p42, const int charge) {
    Fill(HepLorentzVector(p41.x(),p41.y(),p41.z(),p41.t()),
	 HepLorentzVector(p42.x(),p42.y(),p42.z(),p42.t()), charge);
  }
  
   virtual void Fill(const HepLorentzVector & momentum1, const HepLorentzVector & momentum2, const int charge) { 
     hMassVSPt->Fill(momentum1.perp(),momentum2.m()); 
     //hMassVSPt_prof->Fill(momentum1.perp(),momentum2.m()); 
     hMassVSEta->Fill(momentum1.eta(),momentum2.m()); 
     //hMassVSEta_prof->Fill(momentum1.eta(),momentum2.m()); 
     if(charge>0){
       hMassVSPhiPlus->Fill(momentum1.phi(),momentum2.m()); 
       //hMassVSPhiPlus_prof->Fill(momentum1.phi(),momentum2.m()); 
     }
     else if(charge<0){
       hMassVSPhiMinus->Fill(momentum1.phi(),momentum2.m()); 
       //hMassVSPhiMinus_prof->Fill(momentum1.phi(),momentum2.m()); 
     }
     else 
       abort();
   } 
  
  virtual void Write() {
    hMassVSPt->Write();
    hMassVSEta->Write();    
    hMassVSPhiPlus->Write();
    hMassVSPhiMinus->Write();
    //hMassVSPt_prof->Write();
    //hMassVSEta_prof->Write();    
    //hMassVSPhiPlus_prof->Write();
    //hMassVSPhiMinus_prof->Write();

    /*vector<TGraphErrors*> graphPt ((MuScleFitUtils::fitMass(hMassVSPt)));
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
      }*/   
  }
  
  virtual void Clear() {
    hMassVSPt->Clear();
    hMassVSEta->Clear();    
    hMassVSPhiPlus->Clear();
    hMassVSPhiMinus->Clear();
    //hMassVSPt_prof->Clear();
    //hMassVSEta_prof->Clear();    
    //hMassVSPhiPlus_prof->Clear();
    //hMassVSPhiMinus_prof->Clear();
   }
  
 public:
  TH2F* hMassVSPt;
  TH2F* hMassVSEta;
  TH2F* hMassVSPhiPlus; 
  TH2F* hMassVSPhiMinus; 
  //TProfile* hMassVSPt_prof;
  //TProfile* hMassVSEta_prof;
  //TProfile* hMassVSPhiPlus_prof;
  //TProfile* hMassVSPhiMinus_prof;
 
};

//---------------------------------------------------------------------------------------
/// A set of histograms for resolution
class HResolutionVSPart : public Histograms{
 public:
  HResolutionVSPart(TFile * outputFile, const TString & name) : Histograms(outputFile, name) {
    // Kinematical variables

    hReso           = new TH1F (name+"_Reso", "resolution", 4000, -1, 1);
    hResoVSPt       = new TH2F (name+"_ResoVSPt", "resolution VS pt", 200, 0, 200, 4000, -1, 1);
    //hResoVSPt_prof  = new TProfile (name+"_ResoVSPt_prof", "resolution VS pt", 100, 0, 200, -1, 1);
    hResoVSEta      = new TH2F (name+"_ResoVSEta", "resolution VS eta", 30, -3, 3, 4000, -1, 1);
    hResoVSTheta    = new TH2F (name+"_ResoVSTheta", "resolution VS theta", 30, 0, TMath::Pi(), 4000, -1, 1);
    //hResoVSEta_prof = new TProfile (name+"_ResoVSEta_prof", "resolution VS eta", 10, -2.5, 2.5, -1, 1);
    hResoVSPhiPlus  = new TH2F (name+"_ResoVSPhiPlus", "resolution VS phi mu+", 14, -3.2, 3.2, 4000, -1, 1);
    hResoVSPhiMinus = new TH2F (name+"_ResoVSPhiMinus", "resolution VS phi mu-", 14, -3.2, 3.2, 4000, -1, 1);
    //hResoVSPhi_prof = new TProfile (name+"_ResoVSPhi_prof", "resolution VS phi", 14, -3.2, 3.2, -1, 1);
    hAbsReso        = new TH1F (name+"_AbsReso", "resolution", 100, 0, 1);
    hAbsResoVSPt    = new TH2F (name+"_AbsResoVSPt", "Abs resolution VS pt", 200, 0, 500, 100, 0, 1);
    hAbsResoVSEta   = new TH2F (name+"_AbsResoVSEta", "Abs resolution VS eta", 30, -3, 3, 100, 0, 1);
    hAbsResoVSPhi   = new TH2F (name+"_AbsResoVSPhi", "Abs resolution VS phi", 14, -3.2, 3.2, 100, 0, 1);
  }
  
  HResolutionVSPart(const TString & name, TFile* file){
    name_=name;
    hReso           = (TH1F *) file->Get(name+"_Reso");
    hResoVSPt       = (TH2F *) file->Get(name+"_ResoVSPt");
    //hResoVSPt_prof  = (TProfile *) file->Get(name+"_ResoVSPt_prof");
    hResoVSEta      = (TH2F *) file->Get(name+"_ResoVSEta");
    hResoVSTheta    = (TH2F *) file->Get(name+"_ResoVSTheta");
    //hResoVSEta_prof = (TProfile *) file->Get(name+"_ResoVSEta_prof");
    hResoVSPhiPlus  = (TH2F *) file->Get(name+"_ResoVSPhiPlus");
    hResoVSPhiMinus = (TH2F *) file->Get(name+"_ResoVSPhiMinus");
    //hResoVSPhi_prof = (TProfile *) file->Get(name+"_ResoVSPhi_prof");
    hAbsReso        = (TH1F *) file->Get(name+"_AbsReso");
    hAbsResoVSPt    = (TH2F *) file->Get(name+"_AbsResoVSPt");
    hAbsResoVSEta   = (TH2F *) file->Get(name+"_AbsResoVSEta");
    hAbsResoVSPhi   = (TH2F *) file->Get(name+"_AbsResoVSPhi");
   }

  ~HResolutionVSPart(){
  }

  virtual void Fill(const reco::Particle::LorentzVector & p4, const double & resValue, const int charge) { 
    hReso->Fill(resValue); 
    hResoVSPt->Fill(p4.Pt(),resValue); 
    hResoVSEta->Fill(p4.Eta(),resValue); 
    hResoVSTheta->Fill(p4.Theta(),resValue); 
    if(charge>0)
      hResoVSPhiPlus->Fill(p4.Phi(),resValue); 
    else if(charge<0)
      hResoVSPhiMinus->Fill(p4.Phi(),resValue); 
    //hResoVSPt_prof->Fill(p4.Pt(),resValue); 
    //hResoVSEta_prof->Fill(p4.Eta(),resValue); 
    //hResoVSPhi_prof->Fill(p4.Phi(),resValue); 
    hAbsReso->Fill(fabs(resValue)); 
    hAbsResoVSPt->Fill(p4.Pt(),fabs(resValue)); 
    hAbsResoVSEta->Fill(p4.Eta(),fabs(resValue)); 
    hAbsResoVSPhi->Fill(p4.Phi(),fabs(resValue));     
  }

  virtual void Write() {
    if(histoDir_ != 0) histoDir_->cd();

    hReso->Write();
    hResoVSPt->Write();
    //hResoVSPt_prof->Write();
    hResoVSEta->Write();
    hResoVSTheta->Write();
    //hResoVSEta_prof->Write();
    hResoVSPhiMinus->Write();
    hResoVSPhiPlus->Write();
    //hResoVSPhi_prof->Write();
    hAbsReso->Write();
    hAbsResoVSPt->Write();
    hAbsResoVSEta->Write();
    hAbsResoVSPhi->Write();
    /*
    vector<TGraphErrors*> graphs ((MuScleFitUtils::fitReso(hResoVSPt)));
    for(vector<TGraphErrors*>::const_iterator graph = graphs.begin(); graph != graphs.end(); graph++){
      (*graph)->Write();
    }

    vector<TGraphErrors*> graphsEta ((MuScleFitUtils::fitReso(hResoVSEta)));
    for(vector<TGraphErrors*>::const_iterator graph = graphsEta.begin(); graph != graphsEta.end(); graph++){
      (*graph)->Write();
    }

    vector<TGraphErrors*> graphsPhiPlus ((MuScleFitUtils::fitReso(hResoVSPhiMinus)));
    for(vector<TGraphErrors*>::const_iterator graph = graphsPhiPlus.begin(); graph != graphsPhiPlus.end(); graph++){
      (*graph)->Write();
    }

    vector<TGraphErrors*> graphsPhiMinus ((MuScleFitUtils::fitReso(hResoVSPhiPlus)));
    for(vector<TGraphErrors*>::const_iterator graph = graphsPhiMinus.begin(); graph != graphsPhiMinus.end(); graph++){
      (*graph)->Write();
      }*/
  }
  
  virtual void Clear() {
    hReso->Clear();    
    hResoVSPt->Clear();    
    //hResoVSPt_prof->Clear();    
    hResoVSEta->Clear();    
    hResoVSTheta->Clear();    
    //hResoVSEta_prof->Clear();    
    hResoVSPhiPlus->Clear();    
    hResoVSPhiMinus->Clear();    
    //hResoVSPhi_prof->Clear();    
    hAbsReso->Clear();
    hAbsResoVSPt->Clear();
    hAbsResoVSEta->Clear();
    hAbsResoVSPhi->Clear();

 }
  
 public:
  TH1F* hReso;
  TH2F* hResoVSPt;
  //TProfile* hResoVSPt_prof;
  TH2F* hResoVSEta;
  TH2F* hResoVSTheta;
  //TProfile* hResoVSEta_prof;
  TH2F* hResoVSPhiMinus;
  TH2F* hResoVSPhiPlus;
  //TProfile* hResoVSPhi_prof;
  TH1F* hAbsReso;
  TH2F* hAbsResoVSPt;
  TH2F* hAbsResoVSEta;
  TH2F* hAbsResoVSPhi;

};
// -------------------------------------------------------------
// A set of histograms of likelihood value versus muon variables
// -------------------------------------------------------------
class HLikelihoodVSPart : public Histograms{
 public:
  HLikelihoodVSPart(const TString & name){
    name_ = name;

    // Kinematical variables
    // ---------------------
    hLikeVSPt       = new TH2F (name+"_LikelihoodVSPt", "likelihood vs muon transverse momentum", 100, 0., 100., 100, -100., 100.);
    hLikeVSEta      = new TH2F (name+"_LikelihoodVSEta", "likelihood vs muon pseudorapidity", 100, -4.,4., 100, -100., 100.);
    hLikeVSPhi      = new TH2F (name+"_LikelihoodVSPhi", "likelihood vs muon phi angle", 100, -3.2, 3.2, 100, -100., 100.);
    hLikeVSPt_prof  = new TProfile (name+"_LikelihoodVSPt_prof", "likelihood vs muon transverse momentum", 40, 0., 100., -1000., 1000. );
    hLikeVSEta_prof = new TProfile (name+"_LikelihoodVSEta_prof", "likelihood vs muon pseudorapidity", 40, -4.,4., -1000., 1000. );
    hLikeVSPhi_prof = new TProfile (name+"_LikelihoodVSPhi_prof", "likelihood vs muon phi angle", 40, -3.2, 3.2, -1000., 1000.);
   }
  
  HLikelihoodVSPart(const TString & name, TFile* file){
    name_ = name;
    hLikeVSPt       = (TH2F *) file->Get(name+"_LikelihoodVSPt");
    hLikeVSEta      = (TH2F *) file->Get(name+"_LikelihoodVSEta");
    hLikeVSPhi      = (TH2F *) file->Get(name+"_LikelihoodVSPhi");
    hLikeVSPt_prof  = (TProfile *) file->Get(name+"_LikelihoodVSPt_prof");
    hLikeVSEta_prof = (TProfile *) file->Get(name+"_LikelihoodVSEta_prof");
    hLikeVSPhi_prof = (TProfile *) file->Get(name+"_LikelihoodVSPhi_prof");
  }

  ~HLikelihoodVSPart(){
    // Do not delete anything...
  } 

  virtual void Fill(const reco::Particle::LorentzVector & p4, const double & likeValue) {
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
 
};

/**
 * A set of histograms for resolution.</br>
 * The fill method requires the two selected muons, their charges and the reconstructed and generated masses.
 * It evaluates the resolution on the mass vs:</br>
 * Pt of the pair</br>
 * DeltaEta of the pair</br>
 * DeltaPhi of the pair</br>
 * pt, eta and phi of the plus and minus muon separately</br>
 */
class HMassResolutionVSPart : public Histograms{
public:
  HMassResolutionVSPart(TFile * outputFile, const TString & name) : Histograms( outputFile, name ) {
    // Kinematical variables
    TString nameSuffix[] = {"Plus", "Minus"};
    TString titleSuffix[] = {" for mu+", " for mu-"};

    // This is done inside Histograms constructor
    // histoDir_->cd();

    hReso                    = new TH1F     (name+"_Reso", "resolution", 4000, -1, 1);
    hResoVSPairPt            = new TH2F     (name+"_ResoVSPairPt", "resolution VS pt of the pair", 200, 0, 200, 4000, -1, 1);
    hResoVSPairPt_prof       = new TProfile (name+"_ResoVSPairPt_prof", "resolution VS pt of the pair", 100, 0, 200, -1, 1);
    hResoVSPairDeltaEta      = new TH2F     (name+"_ResoVSPairDeltaEta", "resolution VS DeltaEta of the pair", 30, -0.1, 6, 4000, -1, 1);
    hResoVSPairDeltaEta_prof = new TProfile (name+"_ResoVSPairDeltaEta_prof", "resolution VS DeltaEta of the pair", 100, -0.1, 6, -1, 1);
    hResoVSPairDeltaPhi      = new TH2F     (name+"_ResoVSPairDeltaPhi", "resolution VS DeltaPhi of the pair", 14, -0.1, 3.2, 4000, -1, 1);
    hResoVSPairDeltaPhi_prof = new TProfile (name+"_ResoVSPairDeltaPhi_prof", "resolution VS DeltaPhi of the pair", 100, -0.1, 3.2, -1, 1);

    for( int i=0; i<2; ++i ) {
      hResoVSPt[i]       = new TH2F     (name+"_ResoVSPt"+nameSuffix[i], "resolution VS pt"+titleSuffix[i], 200, 0, 200, 4000, -1, 1);
      hResoVSPt_prof[i]  = new TProfile (name+"_ResoVSPt"+nameSuffix[i]+"_prof", "resolution VS pt"+titleSuffix[i], 100, 0, 200, -1, 1);
      hResoVSEta[i]      = new TH2F     (name+"_ResoVSEta"+nameSuffix[i], "resolution VS eta"+titleSuffix[i], 30, -3, 3, 4000, -1, 1);
      hResoVSEta_prof[i] = new TProfile (name+"_ResoVSEta"+nameSuffix[i]+"_prof", "resolution VS eta"+titleSuffix[i], 100, -3, 3, -1, 1);
      hResoVSTheta[i]    = new TH2F     (name+"_ResoVSTheta"+nameSuffix[i], "resolution VS theta"+titleSuffix[i], 30, 0, TMath::Pi(), 4000, -1, 1);
      hResoVSPhi[i]      = new TH2F     (name+"_ResoVSPhi"+nameSuffix[i], "resolution VS phi"+titleSuffix[i], 14, -3.2, 3.2, 4000, -1, 1);
      hResoVSPhi_prof[i] = new TProfile (name+"_ResoVSPhi"+nameSuffix[i]+"_prof", "resolution VS phi"+titleSuffix[i], 100, -3.2, 3.2, -1, 1);
      //     hAbsReso[i]        = new TH1F     (name+"_AbsReso", "resolution", 100, 0, 1);
      //     hAbsResoVSPt    = new TH2F (name+"_AbsResoVSPt", "Abs resolution VS pt", 200, 0, 500, 100, 0, 1);
      //     hAbsResoVSEta    = new TH2F (name+"_AbsResoVSEta", "Abs resolution VS eta", 30, -3, 3, 100, 0, 1);
      //     hAbsResoVSPhi    = new TH2F (name+"_AbsResoVSPhi", "Abs resolution VS phi", 14, -3.2, 3.2, 100, 0, 1);
    }
  }

  HMassResolutionVSPart(const TString & name, TFile* file){
    string nameSuffix[] = {"Plus", "Minus"};
    name_ = name;
    hReso                    = (TH1F *)     file->Get(name+"_Reso");
    hResoVSPairPt            = (TH2F *)     file->Get(name+"_ResoVSPairPt");
    hResoVSPairPt_prof       = (TProfile *) file->Get(name+"_ResoVSPairPt_prof");
    hResoVSPairDeltaEta      = (TH2F *)     file->Get(name+"_ResoVSPairDeltaEta");
    hResoVSPairDeltaEta_prof = (TProfile *) file->Get(name+"_ResoVSPairDeltaEta_prof");
    hResoVSPairDeltaPhi      = (TH2F *)     file->Get(name+"_ResoVSPairDeltaPhi");
    hResoVSPairDeltaPhi_prof = (TProfile *) file->Get(name+"_ResoVSPairDeltaPhi_prof");
    for( int i=0; i<2; ++i ) {
      hResoVSPt[i]           = (TH2F *)     file->Get(name+"_ResoVSPt"+nameSuffix[i]);
      hResoVSPt_prof[i]      = (TProfile *) file->Get(name+"_ResoVSPt"+nameSuffix[i]+"_prof");
      hResoVSEta[i]          = (TH2F *)     file->Get(name+"_ResoVSEta"+nameSuffix[i]);
      hResoVSEta_prof[i]     = (TProfile *) file->Get(name+"_ResoVSEta"+nameSuffix[i]+"_prof");
      hResoVSTheta[i]        = (TH2F *)     file->Get(name+"_ResoVSTheta"+nameSuffix[i]);
      hResoVSPhi[i]          = (TH2F *)     file->Get(name+"_ResoVSPhi"+nameSuffix[i]);
      hResoVSPhi_prof[i]     = (TProfile *) file->Get(name+"_ResoVSPhi"+nameSuffix[i]+"_prof");
    //     hAbsReso      = (TH1F *) file->Get(name+"_AbsReso");
    //     hAbsResoVSPt  = (TH2F *) file->Get(name+"_AbsResoVSPt");
    //     hAbsResoVSEta  = (TH2F *) file->Get(name+"_AbsResoVSEta");
    //     hAbsResoVSPhi  = (TH2F *) file->Get(name+"_AbsResoVSPhi");
    }
  }

  ~HMassResolutionVSPart(){
  }

  virtual void Fill( const reco::Particle::LorentzVector & recoP1, const int charge1,
                     // const reco::Particle::LorentzVector & genP1,
                     const reco::Particle::LorentzVector & recoP2, const int charge2,
                     // const reco::Particle::LorentzVector & genP2,
                     const double & recoMass, const double & genMass ) {

    if ( charge1 == charge2 ) cout << "Error: must get two opposite charge particles" << endl;

    double massRes = (recoMass - genMass)/genMass;

    reco::Particle::LorentzVector recoPair( recoP1 + recoP2 );
    double pairPt = recoPair.Pt();

    double recoPt[2]  = {recoP1.Pt(),  recoP2.Pt()};
    double recoEta[2] = {recoP1.Eta(), recoP2.Eta()};
    double recoPhi[2] = {recoP1.Phi(), recoP2.Phi()};

    // cout << "pairPt = " << pairPt << ", massRes = ("<<recoMass<<" - "<<genMass<<")/"<<genMass<<" = " << massRes
    //      << ", recoPt[0] = " << recoPt[0] << ", recoPt[1] = " << recoPt[1] << endl;

    // Index of the histogram. If the muons have charge1 = -1 and charge2 = 1, they already have the
    // correct histogram indeces. Otherwise, swap the indeces.
    // In any case the negative muon has index = 0  and the positive muon has index = 1.
    int id[2] = {0,1};
    if ( charge1 > 0 ) {
      id[0] = 1;
      id[1] = 0;
    }
//     double recoPt2  = recoP2.Pt();
//     double recoEta2 = recoP2.Eta();
//     double recoPhi2 = recoP2.Phi();

    double pairDeltaEta = fabs(recoEta[0] - recoEta[1]);
    double pairDeltaPhi = MuScleFitUtils::deltaPhi(recoPhi[0], recoPhi[1]);

    hReso->Fill(massRes);
    hResoVSPairPt->Fill(pairPt, massRes);
    hResoVSPairPt_prof->Fill(pairPt, massRes);
    hResoVSPairDeltaEta->Fill(pairDeltaEta, massRes);
    hResoVSPairDeltaEta_prof->Fill(pairDeltaEta, massRes);
    hResoVSPairDeltaPhi->Fill(pairDeltaPhi, massRes);
    hResoVSPairDeltaPhi_prof->Fill(pairDeltaPhi, massRes);

    // Resolution vs single muon quantities
    // ------------------------------------
    int index = 0;
    for( int i=0; i<2; ++i ) {
      index = id[i];
      hResoVSPt[index]->Fill(recoPt[i], massRes);
      hResoVSPt_prof[index]->Fill(recoPt[i], massRes);
      hResoVSEta[index]->Fill(recoEta[i], massRes);
      hResoVSEta_prof[index]->Fill(recoEta[i], massRes);
      // hResoVSTheta[index]->Fill(recoTheta[i], massRes);
      hResoVSPhi[index]->Fill(recoPhi[i], massRes);
      hResoVSPhi_prof[index]->Fill(recoPhi[i], massRes);
    }
  } 

  virtual void Write() {
    histoDir_->cd();

    hReso->Write();
    hResoVSPairPt->Write();
    hResoVSPairPt_prof->Write();
    hResoVSPairDeltaEta->Write();
    hResoVSPairDeltaEta_prof->Write();
    hResoVSPairDeltaPhi->Write();
    hResoVSPairDeltaPhi_prof->Write();
    for( int i=0; i<2; ++i ) {
      hResoVSPt[i]->Write();
      hResoVSPt_prof[i]->Write();
      hResoVSEta[i]->Write();
      hResoVSEta_prof[i]->Write();
      hResoVSTheta[i]->Write();
      hResoVSPhi[i]->Write();
      hResoVSPhi_prof[i]->Write();
//     hAbsReso->Write();
//     hAbsResoVSPt->Write();
//     hAbsResoVSEta->Write();
//     hAbsResoVSPhi->Write();
    }
    /*
    vector<TGraphErrors*> graphs ((MuScleFitUtils::fitReso(hResoVSPt)));
    for(vector<TGraphErrors*>::const_iterator graph = graphs.begin(); graph != graphs.end(); graph++){
      (*graph)->Write();
    }

    vector<TGraphErrors*> graphsEta ((MuScleFitUtils::fitReso(hResoVSEta)));
    for(vector<TGraphErrors*>::const_iterator graph = graphsEta.begin(); graph != graphsEta.end(); graph++){
      (*graph)->Write();
    }

    vector<TGraphErrors*> graphsPhiPlus ((MuScleFitUtils::fitReso(hResoVSPhiMinus)));
    for(vector<TGraphErrors*>::const_iterator graph = graphsPhiPlus.begin(); graph != graphsPhiPlus.end(); graph++){
      (*graph)->Write();
    }

    vector<TGraphErrors*> graphsPhiMinus ((MuScleFitUtils::fitReso(hResoVSPhiPlus)));
    for(vector<TGraphErrors*>::const_iterator graph = graphsPhiMinus.begin(); graph != graphsPhiMinus.end(); graph++){
      (*graph)->Write();
      }*/
  }
  
  virtual void Clear() {
    hReso->Clear();
    hResoVSPairPt->Clear();
    hResoVSPairPt_prof->Clear();
    hResoVSPairDeltaEta->Clear();
    hResoVSPairDeltaEta_prof->Clear();
    hResoVSPairDeltaPhi->Clear();
    hResoVSPairDeltaPhi_prof->Clear();    
    for( int i=0; i<2; ++i ) {
      hResoVSPt[i]->Write();
      hResoVSPt_prof[i]->Clear();
      hResoVSEta[i]->Clear();
      hResoVSEta_prof[i]->Clear();
      hResoVSTheta[i]->Clear();
      hResoVSPhi[i]->Clear();
      hResoVSPhi_prof[i]->Clear();
//     hAbsReso->Clear();
//     hAbsResoVSPt->Clear();
//     hAbsResoVSEta->Clear();
//     hAbsResoVSPhi->Clear();
    }
  }
  
 public:
  TH1F* hReso;
  TH2F* hResoVSPairPt;
  TProfile* hResoVSPairPt_prof;
  TH2F* hResoVSPairDeltaEta;
  TProfile* hResoVSPairDeltaEta_prof;
  TH2F* hResoVSPairDeltaPhi;
  TProfile* hResoVSPairDeltaPhi_prof;

  TH2F * hResoVSPt[2];
  TProfile * hResoVSPt_prof[2];
  TH2F * hResoVSEta[2];
  TProfile * hResoVSEta_prof[2];
  TH2F * hResoVSTheta[2];
  TH2F * hResoVSPhi[2];
  TProfile * hResoVSPhi_prof[2];
  // TH1F * hAbsReso[2];
  // TH2F * hAbsResoVSPt[2];
  // TH2F * hAbsResoVSEta[2];
  // TH2F * hAbsResoVSPhi[2];

};

#endif
