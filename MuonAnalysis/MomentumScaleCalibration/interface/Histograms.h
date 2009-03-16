#ifndef Histograms_H
#define Histograms_H

/** \class Histograms
 *  Collection of histograms for GLB muon analysis
 *
 *  $Date: 2009/01/14 15:34:14 $
 *  $Revision: 1.3 $
 *  \author S. Bolognesi - INFN Torino / T.Dorigo - INFN Padova
 */

#include <CLHEP/Vector/LorentzVector.h>
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "MuonAnalysis/MomentumScaleCalibration/interface/MuScleFitUtils.h"

#include "TH1D.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TH3F.h"
#include "TFile.h"
#include "TString.h"
#include "TProfile.h"
#include "TF1.h"
#include "TGraphErrors.h"
#include "TFile.h"
#include "TSystem.h"
#include "TCanvas.h"

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
    outputFile_(outputFile),
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
                     const reco::Particle::LorentzVector & genP1,
                     const reco::Particle::LorentzVector & recoP2, const int charge2,
                     const reco::Particle::LorentzVector & genP2,
                     const double & recoMass, const double & genMass ) {};
  virtual void Fill( const reco::Particle::LorentzVector & recoP1, const int charge1,
                     // const reco::Particle::LorentzVector & genP1,
                     const reco::Particle::LorentzVector & recoP2, const int charge2,
                     // const reco::Particle::LorentzVector & genP2,
                     const double & recoMass, const double & genMass ) {};
  virtual void Fill( const reco::Particle::LorentzVector & recoP1,
                     const reco::Particle::LorentzVector & genP1,
                     const reco::Particle::LorentzVector & recoP2,
                     const reco::Particle::LorentzVector & genP2 ) {};

  virtual double Get( const reco::Particle::LorentzVector & recoP1, const TString & covarianceName ) { return 0.; };

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
  TFile * outputFile_;
  TDirectory * histoDir_;

private:

};

// -----------------------------------------------------
// A set of histograms of particle kinematical variables
// -----------------------------------------------------
class HParticle : public Histograms {
 public:
  HParticle (const TString & name, const double & minMass = 0., const double & maxMass = 200.) :
    Histograms(name),
    // Kinematical variables
    hPt(     new TH1F (name+"_Pt", "transverse momentum", 100, 0, 100) ),
    hEta(    new TH1F (name+"_Eta", "pseudorapidity", 60, -6, 6) ),
    hPhi(    new TH1F (name+"_Phi", "phi angle", 64, -3.2, 3.2) ),
    hMass(   new TH1F (name+"_Mass", "mass", 40000, minMass, maxMass) ),
    // hMass_fine = new TH1F (name+"_Mass_fine", "low mass fine binning", 4000, 0., 20. ); //Removed to avoid too many histos (more binning added to hMass)
    hNumber( new TH1F (name+"_Number", "number", 20, -0.5, 19.5) )
  {}

  /// Constructor that puts the histograms inside a TDirectory
  HParticle (TFile* outputFile, const TString & name, const double & minMass = 0., const double & maxMass = 200.) :
    Histograms(outputFile, name)
  {
    // Kinematical variables
    hPt =     new TH1F (name+"_Pt", "transverse momentum", 100, 0, 100);
    hEta =    new TH1F (name+"_Eta", "pseudorapidity", 60, -6, 6);
    hPhi =    new TH1F (name+"_Phi", "phi angle", 64, -3.2, 3.2);
    hMass =   new TH1F (name+"_Mass", "mass", 40000, minMass, maxMass);
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
  HDelta (const TString & name) :
    Histograms(name),
    // Kinematical variables
    // ---------------------
    hEta( new TH1F (name+"_DeltaEta", "#Delta#eta", 100, 0, 6) ),
    hEtaSign( new TH1F (name+"_DeltaEtaSign", "#Delta#eta with sign", 100, -6, 6) ),
    hPhi( new TH1F (name+"_DeltaPhi", "#Delta#phi", 100,0,3.2) ),
    hTheta( new TH1F (name+"_DeltaTheta", "#Delta#theta", 100,-3.2,3.2) ),
    hCotgTheta( new TH1F (name+"_DeltaCotgTheta", "#Delta Cotg(#theta )", 100,-3.2,3.2) ),
    hDeltaR( new TH1F (name+"_DeltaR","#Delta R", 400, 0, 1 ) )
  {}

  HDelta (TFile* outputFile, const TString & name) :
    Histograms(outputFile, name),
    // Kinematical variables
    // ---------------------
    hEta( new TH1F (name+"_DeltaEta", "#Delta#eta", 100, 0, 6) ),
    hEtaSign( new TH1F (name+"_DeltaEtaSign", "#Delta#eta with sign", 100, -6, 6) ),
    hPhi( new TH1F (name+"_DeltaPhi", "#Delta#phi", 100,0,3.2) ),
    hTheta( new TH1F (name+"_DeltaTheta", "#Delta#theta", 100,-3.2,3.2) ),
    hCotgTheta( new TH1F (name+"_DeltaCotgTheta", "#Delta Cotg(#theta )", 100,-3.2,3.2) ),
    hDeltaR( new TH1F (name+"_DeltaR","#DeltaR", 400, 0, 1 ) )
  {}

  HDelta (const TString & name, TFile* file) {
    name_ = name;
    hEta       = (TH1F *) file->Get(name+"_DeltaEta");
    hEtaSign   = (TH1F *) file->Get(name+"_DeltaEtaSign");
    hPhi       = (TH1F *) file->Get(name+"_DeltaPhi");
    hTheta     = (TH1F *) file->Get(name+"_DeltaTheta");
    hCotgTheta = (TH1F *) file->Get(name+"_DeltaCotgTheta");
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
    hEta->Fill(fabs( momentum1.eta()-momentum2.eta() ));
    hEtaSign->Fill(momentum1.eta()-momentum2.eta());
    hPhi->Fill(MuScleFitUtils::deltaPhi(momentum1.phi(),momentum2.phi()));
    hTheta->Fill(momentum1.theta()-momentum2.theta());
    // hCotgTheta->Fill(1/(TMath::Tan(momentum1.theta()))-1/(TMath::Tan(momentum2.theta())));
    double theta1 = momentum1.theta();
    double theta2 = momentum2.theta();
    hCotgTheta->Fill(TMath::Cos(theta1)/TMath::Sin(theta1) - TMath::Cos(theta2)/TMath::Sin(theta2));
    hDeltaR->Fill(sqrt((momentum1.eta()-momentum2.eta())*(momentum1.eta()-momentum2.eta()) +
		       (MuScleFitUtils::deltaPhi(momentum1.phi(),momentum2.phi()))*
		       (MuScleFitUtils::deltaPhi(momentum1.phi(),momentum2.phi()))));
  }
  
  virtual void Write() {
    if(histoDir_ != 0) histoDir_->cd();

    hEta->Write();
    hEtaSign->Write();
    hPhi->Write();
    hTheta->Write();
    hCotgTheta->Write();
    hDeltaR->Write();
  }
  
  virtual void Clear() {
    hEta->Clear();
    hEtaSign->Clear();
    hPhi->Clear();
    hTheta->Clear();
    hDeltaR->Clear();
    hCotgTheta->Clear();
  }
  
 public:
  TH1F* hEta;
  TH1F* hEtaSign;
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
  HMassVSPart( const TString & name, const double & minMass = 0., const double & maxMass = 150. ) {
    name_ = name;

    // Kinematical variables
    // ---------------------
    hMassVSPt     = new TH2F (name+"_MassVSPt", "resonance mass vs muon transverse momentum", 200, 0., 200., 6000, minMass, maxMass);
    hMassVSEta    = new TH2F (name+"_MassVSEta", "resonance mass vs muon pseudorapidity", 60, -6., 6., 6000, minMass, maxMass);
    hMassVSPhiPlus    = new TH2F (name+"_MassVSPhiPlus", "resonance mass vs muon+ phi angle", 64, -3.2, 3.2, 6000, minMass, maxMass);
    hMassVSPhiMinus    = new TH2F (name+"_MassVSPhiMinus", "resonance mass vs muon- phi angle", 64, -3.2, 3.2, 6000, minMass, maxMass);
    //hMassVSPt_prof     = new TProfile (name+"_MassVSPt_prof", "resonance mass vs muon transverse momentum", 100, 0., 200., minMass, maxMass);
    //hMassVSEta_prof    = new TProfile (name+"_MassVSEta_prof", "resonance mass vs muon pseudorapidity", 30, -6., 6., minMass, maxMass);
    //hMassVSPhiPlus_prof    = new TProfile (name+"_MassVSPhiPlus_prof", "resonance mass vs muon+ phi angle", 32, -3.2, 3.2, minMass, maxMass);
    //hMassVSPhiMinus_prof    = new TProfile (name+"_MassVSPhiMinus_prof", "resonance mass vs muon- phi angle", 32, -3.2, 3.2, minMass, maxMass);
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
  HResolutionVSPart(TFile * outputFile, const TString & name,
                    const double & yMinEta = -1, const double & yMaxEta = 1) : Histograms(outputFile, name) {

    // Kinematical variables
    hReso           = new TH1F (name+"_Reso", "resolution", 4000, -1, 1);
    hResoVSPtEta    = new TH2F (name+"_ResoVSPtEta", "resolution VS pt and #eta", 200, 0, 200, 60, -3, 3);
    hResoVSPt       = new TH2F (name+"_ResoVSPt", "resolution VS pt", 200, 0, 200, 4000, -1, 1);
    //hResoVSPt_prof  = new TProfile (name+"_ResoVSPt_prof", "resolution VS pt", 100, 0, 200, -1, 1);
    hResoVSEta      = new TH2F (name+"_ResoVSEta", "resolution VS eta", 60, -3, 3, 4000, yMinEta, yMaxEta);
    hResoVSTheta    = new TH2F (name+"_ResoVSTheta", "resolution VS theta", 30, 0, TMath::Pi(), 4000, -1, 1);
    //hResoVSEta_prof = new TProfile (name+"_ResoVSEta_prof", "resolution VS eta", 10, -2.5, 2.5, -1, 1);
    hResoVSPhiPlus  = new TH2F (name+"_ResoVSPhiPlus", "resolution VS phi mu+", 14, -3.2, 3.2, 4000, -1, 1);
    hResoVSPhiMinus = new TH2F (name+"_ResoVSPhiMinus", "resolution VS phi mu-", 14, -3.2, 3.2, 4000, -1, 1);
    //hResoVSPhi_prof = new TProfile (name+"_ResoVSPhi_prof", "resolution VS phi", 14, -3.2, 3.2, -1, 1);
    hAbsReso        = new TH1F (name+"_AbsReso", "resolution", 100, 0, 1);
    hAbsResoVSPt    = new TH2F (name+"_AbsResoVSPt", "Abs resolution VS pt", 200, 0, 500, 100, 0, 1);
    hAbsResoVSEta   = new TH2F (name+"_AbsResoVSEta", "Abs resolution VS eta", 60, -3, 3, 100, 0, 1);
    hAbsResoVSPhi   = new TH2F (name+"_AbsResoVSPhi", "Abs resolution VS phi", 14, -3.2, 3.2, 100, 0, 1);
  }
  
  HResolutionVSPart(const TString & name, TFile* file){
    name_=name;
    hReso           = (TH1F *) file->Get(name+"_Reso");
    hResoVSPtEta    = (TH2F *) file->Get(name+"_ResoVSPtEta");
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
    double pt = p4.Pt();
    double eta = p4.Eta();
    hReso->Fill(resValue);
    hResoVSPtEta->Fill(pt, eta,resValue); 
    hResoVSPt->Fill(pt,resValue); 
    hResoVSEta->Fill(eta,resValue);
    hResoVSTheta->Fill(p4.Theta(),resValue); 
    if(charge>0)
      hResoVSPhiPlus->Fill(p4.Phi(),resValue); 
    else if(charge<0)
      hResoVSPhiMinus->Fill(p4.Phi(),resValue); 
    //hResoVSPt_prof->Fill(p4.Pt(),resValue); 
    //hResoVSEta_prof->Fill(p4.Eta(),resValue); 
    //hResoVSPhi_prof->Fill(p4.Phi(),resValue); 
    hAbsReso->Fill(fabs(resValue)); 
    hAbsResoVSPt->Fill(pt,fabs(resValue)); 
    hAbsResoVSEta->Fill(eta,fabs(resValue)); 
    hAbsResoVSPhi->Fill(p4.Phi(),fabs(resValue));     
  }

  virtual void Write() {
    if(histoDir_ != 0) histoDir_->cd();

    hReso->Write();
    hResoVSPtEta->Write();
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
    hResoVSPtEta->Clear();
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
  TH2F* hResoVSPtEta;
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
 * This histogram class fills a TProfile with the resolution evaluated from the resolution
 * functions for single muon quantities. The resolution functions are used by MuScleFit to
 * evaluate the mass resolution, which is the value seen by minuit and through it,
 * corrections are evaluated.
 * In the end we will compare the histograms filled by this class (from the resolution
 * function, reflecting the parameters changes done by minuit) with those filled comparing
 * recoMuons with genMuons (the real resolutions).
 */
class HFunctionResolution : public Histograms {
 public:
  HFunctionResolution(TFile * outputFile, const TString & name, const double ptMax = 200) : Histograms(outputFile, name) {
    name_ = name;
    totBinsX_ = 100;
    totBinsY_ = 60;
    xMin_ = 0.;
    yMin_ = -3.2;
    double xMax = ptMax;
    double yMax = 3.2;
    deltaX_ = xMax - xMin_;
    deltaY_ = yMax - yMin_;
    hReso                = new TH1F( name+"_Reso", "resolution", 1000, 0, 1 );
    hResoVSPtEta         = new TH2F( name+"_ResoVSPtEta", "resolution vs pt and #eta", totBinsX_, xMin_, xMax, totBinsY_, yMin_, yMax );
    // Create and initialize the resolution arrays
    resoVsPtEta_ = new double*[totBinsX_];
    resoCount_ = new int*[totBinsX_];
    for( int i=0; i<totBinsX_; ++i ) {
      resoVsPtEta_[i] = new double[totBinsY_];
      resoCount_[i] = new int[totBinsY_];
      for( int j=0; j<totBinsY_; ++j ) {
        resoVsPtEta_[i][j] = 0;
        resoCount_[i][j] = 0;
      }
    }
    hResoVSPt_prof       = new TProfile( name+"_ResoVSPt_prof", "resolution VS pt", totBinsX_, xMin_, xMax, yMin_, yMax);
    hResoVSEta_prof      = new TProfile( name+"_ResoVSEta_prof", "resolution VS eta", totBinsY_, yMin_, yMax, 0, 1);
    //hResoVSTheta_prof    = new TProfile( name+"_ResoVSTheta_prof", "resolution VS theta", 30, 0, TMath::Pi(), 0, 1);
    hResoVSPhiPlus_prof  = new TProfile( name+"_ResoVSPhiPlus_prof", "resolution VS phi mu+", 14, -3.2, 3.2, 0, 1);
    hResoVSPhiMinus_prof = new TProfile( name+"_ResoVSPhiMinus_prof", "resolution VS phi mu-", 14, -3.2, 3.2, 0, 1);
    hResoVSPhi_prof      = new TProfile( name+"_ResoVSPhi_prof", "resolution VS phi", 14, -3.2, 3.2, -1, 1);
  }
  ~HFunctionResolution() {}
  virtual void Fill(const reco::Particle::LorentzVector & p4, const double & resValue, const int charge) { 
    // cout << "Filling "<<hReso->GetName()<<" with resValue = " << resValue << endl;
    hReso->Fill(resValue);

    // Fill the arrays with the resolution value and count
    int xIndex = getXindex(p4.Pt());
    int yIndex = getYindex(p4.Eta());
    if ( 0 <= xIndex && xIndex < totBinsX_ && 0 <= yIndex && yIndex < totBinsY_ ) {
      resoVsPtEta_[xIndex][yIndex] += resValue;
      // ATTENTION: we count only for positive muons because we are considering di-muon resonances
      // and we use this counter to compute the mean in the end. The resoVsPtEta value is filled with each muon,
      // but they must be considered independently (analogous to a factor 2) so in the end we would have
      // to divide by N/2, that is why we only increase the counter for half the muons.
      // if( charge > 0 )
      // No more. Changing it here influences also other uses of this class. The macro FunctionTerms.cc
      // multiplies the terms by the 2 factor.

      resoCount_[xIndex][yIndex] += 1;

      // hResoVSPtEta->Fill(p4.Pt(), p4.Eta(), resValue);
      hResoVSPt_prof->Fill(p4.Pt(),resValue);
      hResoVSEta_prof->Fill(p4.Eta(),resValue);
      //hResoVSTheta_prof->Fill(p4.Theta(),resValue);
      if(charge>0)
        hResoVSPhiPlus_prof->Fill(p4.Phi(),resValue);
      else if(charge<0)
        hResoVSPhiMinus_prof->Fill(p4.Phi(),resValue);
      hResoVSPhi_prof->Fill(p4.Phi(),resValue);
    }
  }

  virtual void Write() {
    if(histoDir_ != 0) histoDir_->cd();

    hReso->Write();

    for( int i=0; i<totBinsX_; ++i ) {
      for( int j=0; j<totBinsY_; ++j ) {
        int N = resoCount_[i][j];
        // Fill with the mean value
        if( N != 0 ) hResoVSPtEta->SetBinContent( i+1, j+1, resoVsPtEta_[i][j]/N );
        else hResoVSPtEta->SetBinContent( i+1, j+1, 0 );
      }
    }

    hResoVSPtEta->Write();


    hResoVSPt_prof->Write();
    hResoVSEta_prof->Write();
    //hResoVSTheta_prof->Write();
    hResoVSPhiMinus_prof->Write();
    hResoVSPhiPlus_prof->Write();
    hResoVSPhi_prof->Write();

    TCanvas canvas(TString(hResoVSPtEta->GetName())+"_canvas", TString(hResoVSPtEta->GetTitle())+" canvas", 1000, 800);
    canvas.Divide(2);
    canvas.cd(1);
    hResoVSPtEta->Draw("lego");
    canvas.cd(2);
    hResoVSPtEta->Draw("surf5");
    canvas.Write();
    hResoVSPtEta->Write();

    outputFile_->cd();
  }
  
  virtual void Clear() {
    hReso->Clear();
    hResoVSPtEta->Clear();
    hResoVSPt_prof->Clear();
    hResoVSEta_prof->Clear();
    //hResoVSTheta_prof->Clear();
    hResoVSPhiPlus_prof->Clear();
    hResoVSPhiMinus_prof->Clear();
    hResoVSPhi_prof->Clear();
  }

 protected:
  int getXindex(const double & x) const {
    return int((x-xMin_)/deltaX_*totBinsX_);
  }
  int getYindex(const double & y) const {
    return int((y-yMin_)/deltaY_*totBinsY_);
  }
  TH1F* hReso;
  TH2F* hResoVSPtEta;
  double ** resoVsPtEta_;
  int ** resoCount_;
  TProfile* hResoVSPt_prof;
  TProfile* hResoVSEta_prof;
  //TProfile* hResoVSTheta_prof;
  TProfile* hResoVSPhiMinus_prof;
  TProfile* hResoVSPhiPlus_prof;
  TProfile* hResoVSPhi_prof;
  int totBinsX_, totBinsY_;
  double xMin_, yMin_;
  double deltaX_, deltaY_;
};

class HFunctionResolutionVarianceCheck : public HFunctionResolution
{
public:
  HFunctionResolutionVarianceCheck(TFile * outputFile, const TString & name, const double ptMax = 200) :
    HFunctionResolution(outputFile, name, ptMax)
  {
    histoVarianceCheck_ = new TH1D**[totBinsX_];
    for( int i=0; i<totBinsX_; ++i ) {
      histoVarianceCheck_[i] = new TH1D*[totBinsY_];
      for( int j=0; j<totBinsY_; ++j ) {
        stringstream namei;
        stringstream namej;
        namei << i;
        namej << j;
        histoVarianceCheck_[i][j] = new TH1D(name+"_"+namei.str()+"_"+namej.str(), name, 100, 0., 1.);
      }
    }
  }
  ~HFunctionResolutionVarianceCheck() {
  }
  virtual void Fill(const reco::Particle::LorentzVector & p4, const double & resValue, const int charge) { 
    // Need to convert the (x,y) values to the array indeces
    int xIndex = getXindex(p4.Pt());
    int yIndex = getYindex(p4.Eta());
    // Only fill values if they are in the selected range
    if ( 0 <= xIndex && xIndex < totBinsX_ && 0 <= yIndex && yIndex < totBinsY_ ) {
      histoVarianceCheck_[xIndex][yIndex]->Fill(resValue);
    }
    // Call also the fill of the base class
    HFunctionResolution::Fill( p4, resValue, charge );
  }
  void Write() {
    if(histoDir_ != 0) histoDir_->cd();
    for( int xBin=0; xBin<totBinsX_; ++xBin ) {
      for( int yBin=0; yBin<totBinsY_; ++yBin ) {
        histoVarianceCheck_[xBin][yBin]->Write();
      }
    }
    HFunctionResolution::Write();
  }
protected:
  TH1D *** histoVarianceCheck_;
};

/**
 * This histogram class can be used to evaluate the resolution of a variable.</br>
 * It has a TProfile, a TH2F and a TH1F. The TProfile is used to compute the rms of
 * the distribution which is filled in the TH1F (the resolution histogram) in the
 * Write method.</br>
 * If a TDirectory is passed to the constructor, the different histograms are
 * placed in subdirectories.
 */
class HResolution : public TH1F {
public:
  HResolution( const TString & name, const TString & title,
               const int totBins, const double & xMin, const double & xMax,
               const double & yMin, const double & yMax, TDirectory * dir = 0) :
    dir_(dir),
    dir2D_(0),
    diffDir_(0)
  {
    if( dir_ != 0 ) {
      dir2D_ = (TDirectory*) dir_->Get("2D");
      if(dir2D_ == 0) dir2D_ = dir_->mkdir("2D");
      diffDir_ = (TDirectory*) dir_->Get("deltaXoverX");
      if(diffDir_ == 0) diffDir_ = dir->mkdir("deltaXoverX");
    }
    diffHisto_ = new TProfile(name+"_prof", title+" profile", totBins, xMin, xMax, yMin, yMax);
    histo2D_ = new TH2F(name+"2D", title, totBins, xMin, xMax, 4000, yMin, yMax);
    resoHisto_ = new TH1F(name, title, totBins, xMin, xMax);
  }
  virtual Int_t Fill( Double_t x, Double_t y ) {
    diffHisto_->Fill(x, y);
    histo2D_->Fill(x, y);
    return 0;
  }
  virtual Int_t	Write(const char* name = 0, Int_t option = 0, Int_t bufsize = 0) {
    // Loop on all the bins and take the rms.
    // The TProfile bin error is by default the standard error on the mean, that is
    // rms/sqrt(N). If it is created with the "S" option (as we did NOT do), it would
    // already be the rms. Thus we take the error and multiply it by the sqrt of the
    // bin entries to get the rms.
    // bin 0 is the underflow, bin totBins+1 is the overflow.
    unsigned int totBins = diffHisto_->GetNbinsX();
    // cout << "totBins = " << totBins << endl;
    for( unsigned int iBin=1; iBin<=totBins; ++iBin ) {
      // cout << "iBin = " << iBin << ", " << diffHisto_->GetBinError(iBin)*sqrt(diffHisto_->GetBinEntries(iBin)) << endl;
      resoHisto_->SetBinContent( iBin, diffHisto_->GetBinError(iBin)*sqrt(diffHisto_->GetBinEntries(iBin)) );
    }
    if( dir_ != 0 ) dir_->cd();
    resoHisto_->Write();
    if( diffDir_ != 0 ) diffDir_->cd();
    diffHisto_->Write();
    if( dir2D_ != 0 ) dir2D_->cd();
    histo2D_->Write();

    return 0;
  }

protected:
  TDirectory * dir_;
  TDirectory * dir2D_;
  TDirectory * diffDir_;
  TProfile * diffHisto_;
  TH2F * histo2D_;
  TH1F * resoHisto_;
};

/**
 * This class can be used to compute the covariance between two input variables.
 * The Fill method need the two input variables. </br>
 * In the end the covariance method computes the covariance as:
 * cov(x,y) = Sum_i(x_i*y_i)/N - x_mean*y_mean. </br>
 * Of course passing the same variable for x and y gives the variance of that variable.
 */
class Covariance {
public:
  Covariance() :
    productXY_(0),
    sumX_(0),
    sumY_(0),
    N_(0)
  {}
  void fill(const double & x, const double & y) {
    productXY_ += x*y;
    sumX_ += x;
    sumY_ += y;
    ++N_;
  }
  double covariance() {
    if( N_ != 0 ) {
      double meanX = sumX_/N_;
      double meanY = sumY_/N_;
      cout << "meanX*meanY = "<<meanX<<"*"<<meanY<< " = " << meanX*meanY << endl;
      return (productXY_/N_ - meanX*meanY);
    }
    return 0.;
  }
  double getN() {return N_;}
protected:
  double productXY_;
  double sumX_;
  double sumY_;
  int N_;
};

/**
 * This class can be used to compute the covariance of two variables with respect to other two variables
 * (to see e.g. how does the covariance of ptVSphi vary with respect to (pt,eta).
 */
class HCovarianceVSxy {
public:
  HCovarianceVSxy( const TString & name, const TString & title,
                   const int totBinsX, const double & xMin, const double & xMax,
                   const int totBinsY, const double & yMin, const double & yMax,
                   TDirectory * dir = 0, bool varianceCheck = false ) :
    name_(name),
    dir_(dir),
    totBinsX_(totBinsX), totBinsY_(totBinsY),
    xMin_(xMin), deltaX_(xMax-xMin), yMin_(yMin), deltaY_(yMax-yMin),
    readMode_(false),
    varianceCheck_(varianceCheck)
  {
    histoCovariance_ = new TH2D(name+"Covariance", title+" covariance", totBinsX, xMin, xMax, totBinsY, yMin, yMax);

    covariances_ = new Covariance*[totBinsX];
    for( int i=0; i<totBinsX; ++i ) {
      covariances_[i] = new Covariance[totBinsY];
    }
    if( varianceCheck_ ) {
      histoVarianceCheck_ = new TH1D**[totBinsX_];
      for( int i=0; i<totBinsX_; ++i ) {
        histoVarianceCheck_[i] = new TH1D*[totBinsY_];
        for( int j=0; j<totBinsY_; ++j ) {
          stringstream namei;
          stringstream namej;
          namei << i;
          namej << j;
          histoVarianceCheck_[i][j] = new TH1D(name+"_"+namei.str()+"_"+namej.str(), name, 10000, -1, 1);
        }
      }
    }
  }
  /// Contructor to read histograms from file
  HCovarianceVSxy( TFile * inputFile, const TString & name, const TString & dirName ) :
    readMode_(true)
  {
    dir_ = (TDirectory*)(inputFile->Get(dirName.Data()));
    if( dir_ == 0 ) {
      cout << "Error: directory not found" << endl;
      exit(0);
    }
    histoCovariance_ = (TH2D*)(dir_->Get(name));
    totBinsX_ = histoCovariance_->GetNbinsX();
    xMin_ = histoCovariance_->GetXaxis()->GetBinLowEdge(1);
    deltaX_ = histoCovariance_->GetXaxis()->GetBinUpEdge(totBinsX_) - xMin_;
    totBinsY_ = histoCovariance_->GetNbinsY();
    yMin_ = histoCovariance_->GetYaxis()->GetBinLowEdge(1);
    deltaY_ = histoCovariance_->GetYaxis()->GetBinUpEdge(totBinsY_) - yMin_;

//     cout << "For histogram: " << histoCovariance_->GetName() << endl;
//     cout << "totBinsX_ = " << totBinsX_ << endl;
//     cout << "xMin_ = " << xMin_ << endl;
//     cout << "deltaX_ = " << deltaX_ << endl;
//     cout << "totBinsY_ = " << totBinsY_ << endl;
//     cout << "yMin_ = " << yMin_ << endl;
//     cout << "deltaY_ = " << deltaY_ << endl;
  }

  ~HCovarianceVSxy() {
    for(int i=0; i<totBinsX_; ++i) {
      delete[] covariances_[i];
    }
    delete[] covariances_;
  }
  /**
   * x and y should be the variables VS which we are computing the covariance (pt and eta)
   * a and b should be the variables OF which we are computing the covariance </br>
   */
  void Fill( const double & x, const double & y, const double & a, const double & b ) {
    // Need to convert the (x,y) values to the array indeces
    int xIndex = getXindex(x);
    int yIndex = getYindex(y);
    // Only fill values if they are in the selected range
    if ( 0 <= xIndex && xIndex < totBinsX_ && 0 <= yIndex && yIndex < totBinsY_ ) {
      if( TString(histoCovariance_->GetName()).Contains("CovarianceCotgTheta_Covariance") )
//         cout << "(x,y) = (" << xIndex << ", " << yIndex << "), (a,b) = (" << a << ", " << b << ")" << endl;
      covariances_[xIndex][yIndex].fill(a,b);
      // Should be used with the variance, in which case a==b
      if( varianceCheck_ ) histoVarianceCheck_[xIndex][yIndex]->Fill(a);
    }
  }
  double Get( const double & x, const double & y ) const {
    // Need to convert the (x,y) values to the array indeces
    int xIndex = getXindex(x);
    int yIndex = getYindex(y);
    // If the values exceed the limits of the histogram, return the border values
    if ( xIndex < 0 ) xIndex = 0;
    if ( xIndex >= totBinsX_ ) xIndex = totBinsX_-1;
    if ( yIndex < 0 ) yIndex = 0;
    if ( yIndex >= totBinsY_ ) yIndex = totBinsY_-1;
    return histoCovariance_->GetBinContent(xIndex+1, yIndex+1);
  }

  void Write() {
    if( !readMode_ ) {
      cout << "writing: " << histoCovariance_->GetName() << endl;
      for( int xBin=0; xBin<totBinsX_; ++xBin ) {
        for( int yBin=0; yBin<totBinsY_; ++yBin ) {
          double covariance = covariances_[xBin][yBin].covariance();
          // Histogram bins start from 1
          cout << "covariance["<<xBin<<"]["<<yBin<<"] with N = "<<covariances_[xBin][yBin].getN()<<" is: " << covariance << endl;
          histoCovariance_->SetBinContent(xBin+1, yBin+1, covariance);
        }
      }
      if( dir_ != 0 ) dir_->cd();
      TCanvas canvas(TString(histoCovariance_->GetName())+"_canvas", TString(histoCovariance_->GetTitle())+" canvas", 1000, 800);
      canvas.Divide(2);
      canvas.cd(1);
      histoCovariance_->Draw("lego");
      canvas.cd(2);
      histoCovariance_->Draw("surf5");
      canvas.Write();
      histoCovariance_->Write();

      TDirectory * binsDir = 0;
      if( varianceCheck_ ) {
        if ( dir_ != 0 ) {
          dir_->cd();
          if( binsDir == 0 ) binsDir = dir_->mkdir(name_+"Bins");
          binsDir->cd();
        }
        for( int xBin=0; xBin<totBinsX_; ++xBin ) {
          for( int yBin=0; yBin<totBinsY_; ++yBin ) {
            histoVarianceCheck_[xBin][yBin]->Write();
          }
        }
      }
    }
  }
  void Clear() {
    histoCovariance_->Clear();
  }
protected:
  int getXindex(const double & x) const {
    return int((x-xMin_)/deltaX_*totBinsX_);
  }
  int getYindex(const double & y) const {
    return int((y-yMin_)/deltaY_*totBinsY_);
  }
  TString name_;
  TDirectory * dir_;
  TH2D * histoCovariance_;
  Covariance ** covariances_;
  int totBinsX_, totBinsY_, totBinsZ_;
  double xMin_, deltaX_, yMin_, deltaY_;
  bool readMode_;
  bool varianceCheck_;
  TH1D *** histoVarianceCheck_;
};

/**
 * This class uses the HCovariance histograms to compute the covariances between the two input muons kinematic quantities. </br>
 * The covariances are computed against pt and eta.
 */
class HCovarianceVSParts : public Histograms
{
 public:
  HCovarianceVSParts(TFile * outputFile, const TString & name, const double & ptMax ) : Histograms( outputFile, name ) {
    int totBinsX = 40;
    int totBinsY = 40;
    double etaMin = -3.;
    double etaMax = 3.;
    double ptMin = 0.;

    readMode_ = false;

    // Variances
    mapHisto_[name+"Pt"]                    = new HCovarianceVSxy(name+"Pt_", "Pt", totBinsX, ptMin, ptMax, totBinsY, etaMin, etaMax, histoDir_, true);
    mapHisto_[name+"CotgTheta"]             = new HCovarianceVSxy(name+"CotgTheta_", "CotgTheta", totBinsX, ptMin, ptMax, totBinsY, etaMin, etaMax, histoDir_, true);
    mapHisto_[name+"Phi"]                   = new HCovarianceVSxy(name+"Phi_", "Phi", totBinsX, ptMin, ptMax, totBinsY, etaMin, etaMax, histoDir_, true);
    // Covariances
    mapHisto_[name+"Pt-CotgTheta"]          = new HCovarianceVSxy(name+"Pt_CotgTheta_", "Pt-CotgTheta", totBinsX, ptMin, ptMax, totBinsY, etaMin, etaMax, histoDir_);
    mapHisto_[name+"Pt-Phi"]                = new HCovarianceVSxy(name+"Pt_Phi_", "Pt-Phi", totBinsX, ptMin, ptMax, totBinsY, etaMin, etaMax, histoDir_);
    mapHisto_[name+"CotgTheta-Phi"]         = new HCovarianceVSxy(name+"CotgTheta_Phi_", "CotgTheta-Phi", totBinsX, ptMin, ptMax, totBinsY, etaMin, etaMax, histoDir_);
    mapHisto_[name+"Pt1-Pt2"]               = new HCovarianceVSxy(name+"Pt1_Pt2_", "Pt1-Pt2", totBinsX, ptMin, ptMax, totBinsY, etaMin, etaMax, histoDir_);
    mapHisto_[name+"CotgTheta1-CotgTheta2"] = new HCovarianceVSxy(name+"CotgTheta1_CotgTheta2_", "CotgTheta1-CotgTheta2", totBinsX, ptMin, ptMax, totBinsY, etaMin, etaMax, histoDir_);
    mapHisto_[name+"Phi1-Phi2"]             = new HCovarianceVSxy(name+"Phi1_Phi2_", "Phi1-Phi2", totBinsX, ptMin, ptMax, totBinsY, etaMin, etaMax, histoDir_);
    mapHisto_[name+"Pt12-CotgTheta21"]      = new HCovarianceVSxy(name+"Pt12_CotgTheta21_", "Pt12-CotgTheta21", totBinsX, ptMin, ptMax, totBinsY, etaMin, etaMax, histoDir_);
    mapHisto_[name+"Pt12-Phi21"]            = new HCovarianceVSxy(name+"Pt12_Phi21_", "Pt12-Phi21", totBinsX, ptMin, ptMax, totBinsY, etaMin, etaMax, histoDir_);
    mapHisto_[name+"CotgTheta12-Phi21"]     = new HCovarianceVSxy(name+"CotgTheta12_Phi21_", "CotgTheta12-Phi21", totBinsX, ptMin, ptMax, totBinsY, etaMin, etaMax, histoDir_);
  }

  /// Constructor reading the histograms from file
  HCovarianceVSParts( const TString & inputFileName, const TString & name )
  {
    name_ = name;
    TFile * inputFile = new TFile(inputFileName, "READ");
    readMode_ = true;

    // Variances
    mapHisto_[name_+"Pt"]                    = new HCovarianceVSxy(inputFile, name_+"Pt_"+name_, name_);
    mapHisto_[name_+"CotgTheta"]             = new HCovarianceVSxy(inputFile, name_+"CotgTheta_"+name_, name_);
    mapHisto_[name_+"Phi"]                   = new HCovarianceVSxy(inputFile, name_+"Phi_"+name_, name_);
    // Covariances
    mapHisto_[name_+"Pt-CotgTheta"]          = new HCovarianceVSxy(inputFile, name_+"Pt_CotgTheta_"+name_, name_);
    mapHisto_[name_+"Pt-Phi"]                = new HCovarianceVSxy(inputFile, name_+"Pt_Phi_"+name_, name_);
    mapHisto_[name_+"CotgTheta-Phi"]         = new HCovarianceVSxy(inputFile, name_+"CotgTheta_Phi_"+name_, name_);
    mapHisto_[name_+"Pt1-Pt2"]               = new HCovarianceVSxy(inputFile, name_+"Pt1_Pt2_"+name_, name_);
    mapHisto_[name_+"CotgTheta1-CotgTheta2"] = new HCovarianceVSxy(inputFile, name_+"CotgTheta1_CotgTheta2_"+name_, name_);
    mapHisto_[name_+"Phi1-Phi2"]             = new HCovarianceVSxy(inputFile, name_+"Phi1_Phi2_"+name_, name_);
    mapHisto_[name_+"Pt12-CotgTheta21"]      = new HCovarianceVSxy(inputFile, name_+"Pt12_CotgTheta21_"+name_, name_);
    mapHisto_[name_+"Pt12-Phi21"]            = new HCovarianceVSxy(inputFile, name_+"Pt12_Phi21_"+name_, name_);
    mapHisto_[name_+"CotgTheta12-Phi21"]     = new HCovarianceVSxy(inputFile, name_+"CotgTheta12_Phi21_"+name_, name_);
  }

  ~HCovarianceVSParts(){
  }

  virtual double Get( const reco::Particle::LorentzVector & recoP1, const TString & covarianceName ) {
    return mapHisto_[name_+covarianceName]->Get(recoP1.pt(), recoP1.eta());
  }

  virtual void Fill( const reco::Particle::LorentzVector & recoP1,
                     const reco::Particle::LorentzVector & genP1,
                     const reco::Particle::LorentzVector & recoP2,
                     const reco::Particle::LorentzVector & genP2 ) {

    double pt1 = recoP1.pt();
    double eta1 = recoP1.eta();
    double pt2 = recoP2.pt();
    double eta2 = recoP2.eta();

    double diffPt1 = (pt1 - genP1.pt())/genP1.pt();
    double diffPt2 = (pt2 - genP2.pt())/genP2.pt();

    double genTheta1 = genP1.theta();
    double genTheta2 = genP2.theta();
    double recoTheta1 = recoP1.theta();
    double recoTheta2 = recoP2.theta();

    double genCotgTheta1 = TMath::Cos(genTheta1)/(TMath::Sin(genTheta1));
    double genCotgTheta2 = TMath::Cos(genTheta2)/(TMath::Sin(genTheta2));
    double recoCotgTheta1 = TMath::Cos(recoTheta1)/(TMath::Sin(recoTheta1));
    double recoCotgTheta2 = TMath::Cos(recoTheta2)/(TMath::Sin(recoTheta2));

    // double diffCotgTheta1 = (recoCotgTheta1 - genCotgTheta1)/genCotgTheta1;
    // double diffCotgTheta2 = (recoCotgTheta2 - genCotgTheta2)/genCotgTheta2;
    double diffCotgTheta1 = recoCotgTheta1 - genCotgTheta1;
    double diffCotgTheta2 = recoCotgTheta2 - genCotgTheta2;

    // double diffPhi1 = (recoP1.phi() - genP1.phi())/genP1.phi();
    // double diffPhi2 = (recoP2.phi() - genP2.phi())/genP2.phi();
    double diffPhi1 = MuScleFitUtils::deltaPhiNoFabs(recoP1.phi(), genP1.phi());
    double diffPhi2 = MuScleFitUtils::deltaPhiNoFabs(recoP2.phi(), genP2.phi());

//     cout << "pt1 = " << pt1 << endl;
//     cout << "pt2 = " << pt2 << endl;
//     cout << "eta1 = " << eta1 << endl;
//     cout << "eta2 = " << eta2 << endl;
//     cout << "phi1 = " << phi1 << endl;
//     cout << "phi2 = " << phi2 << endl;
//     cout << "theta1 = " << theta1 << endl;
//     cout << "theta2 = " << theta2 << endl;
//     cout << "cotgTheta1 = " << cotgTheta1 << endl;
//     cout << "cothTheta2 = " << cotgTheta2 << endl;

    // Fill the variances
    mapHisto_[name_+"Pt"]->Fill(pt1, eta1, diffPt1, diffPt1);
    mapHisto_[name_+"Pt"]->Fill(pt2, eta2, diffPt2, diffPt2);
    mapHisto_[name_+"CotgTheta"]->Fill(pt1, eta1, diffCotgTheta1, diffCotgTheta1);
    mapHisto_[name_+"CotgTheta"]->Fill(pt2, eta2, diffCotgTheta2, diffCotgTheta2);
    mapHisto_[name_+"Phi"]->Fill(pt1, eta1, diffPhi1, diffPhi1);
    mapHisto_[name_+"Phi"]->Fill(pt2, eta2, diffPhi2, diffPhi2);

    // Fill these histograms with both muons
    mapHisto_[name_+"Pt-CotgTheta"]->Fill(pt1, eta1, diffPt1, diffCotgTheta1 );
    mapHisto_[name_+"Pt-CotgTheta"]->Fill(pt2, eta2, diffPt2, diffCotgTheta2 );
    mapHisto_[name_+"Pt-Phi"]->Fill(pt1, eta1, diffPt1, diffPhi1);
    mapHisto_[name_+"Pt-Phi"]->Fill(pt2, eta2, diffPt2, diffPhi2);
    mapHisto_[name_+"CotgTheta-Phi"]->Fill(pt1, eta1, diffCotgTheta1, diffPhi1);
    mapHisto_[name_+"CotgTheta-Phi"]->Fill(pt2, eta2, diffCotgTheta2, diffPhi2);

    // We fill two (pt, eta) bins for each pair of values. The bin of the
    // first and of the second muon. This should take account for the
    // assumed symmetry between the exchange of the first with the second muon.
    mapHisto_[name_+"Pt1-Pt2"]->Fill(pt1, eta1, diffPt1, diffPt2);
    mapHisto_[name_+"Pt1-Pt2"]->Fill(pt2, eta2, diffPt1, diffPt2);
    mapHisto_[name_+"CotgTheta1-CotgTheta2"]->Fill(pt1, eta1, diffCotgTheta1, diffCotgTheta2);
    mapHisto_[name_+"CotgTheta1-CotgTheta2"]->Fill(pt2, eta2, diffCotgTheta1, diffCotgTheta2);
    mapHisto_[name_+"Phi1-Phi2"]->Fill(pt1, eta1, diffPhi1, diffPhi2);
    mapHisto_[name_+"Phi1-Phi2"]->Fill(pt2, eta2, diffPhi1, diffPhi2);

    // Fill the following histograms again for each muon (pt, eta) bin. Same
    // reason as in the previous case. If the symmetry is true, it does not
    // make any difference the order by which we fill the pt and cotgTheta combinations.
    mapHisto_[name_+"Pt12-CotgTheta21"]->Fill(pt1, eta1, diffPt1, diffCotgTheta2);
    mapHisto_[name_+"Pt12-CotgTheta21"]->Fill(pt2, eta2, diffPt2, diffCotgTheta1);
    mapHisto_[name_+"Pt12-Phi21"]->Fill(pt1, eta1, diffPt1, diffPhi2);
    mapHisto_[name_+"Pt12-Phi21"]->Fill(pt2, eta2, diffPt2, diffPhi1);
    mapHisto_[name_+"CotgTheta12-Phi21"]->Fill(pt1, eta1, diffCotgTheta1, diffPhi2);
    mapHisto_[name_+"CotgTheta12-Phi21"]->Fill(pt2, eta2, diffCotgTheta2, diffPhi1);
  }
  virtual void Write() {
    if( !readMode_ ) {
      histoDir_->cd();

      for (map<TString, HCovarianceVSxy*>::const_iterator histo=mapHisto_.begin(); 
           histo!=mapHisto_.end(); histo++) {
        (*histo).second->Write();
      }
    }
  }
  virtual void Clear() {
    for (map<TString, HCovarianceVSxy*>::const_iterator histo=mapHisto_.begin(); 
         histo!=mapHisto_.end(); histo++) {
      (*histo).second->Clear();
    }
  }
 protected:
  map<TString, HCovarianceVSxy*> mapHisto_;
  bool readMode_;
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
class HMassResolutionVSPart : public Histograms
{
 public:
  HMassResolutionVSPart(TFile * outputFile, const TString & name) : Histograms( outputFile, name ) {
    // Kinematical variables
    nameSuffix_[0] = "Plus";
    nameSuffix_[1] = "Minus";
    TString titleSuffix[] = {" for mu+", " for mu-"};

    mapHisto_[name]                  = new TH1F (name, "#Delta M/M", 4000, -1, 1);
    mapHisto_[name+"VSPairPt"]       = new HResolution (name+"VSPairPt", "resolution VS pt of the pair", 100, 0, 200, -1, 1, histoDir_);
    mapHisto_[name+"VSPairDeltaEta"] = new HResolution (name+"VSPairDeltaEta", "resolution VS #Delta#eta of the pair", 100, -0.1, 6.2, -1, 1, histoDir_);
    mapHisto_[name+"VSPairDeltaPhi"] = new HResolution (name+"VSPairDeltaPhi", "resolution VS #Delta#phi of the pair", 100, -0.1, 3.2, -1, 1, histoDir_);

    for( int i=0; i<2; ++i ) {
      mapHisto_[name+"VSPt"+nameSuffix_[i]]  = new HResolution (name+"VSPt"+nameSuffix_[i], "resolution VS pt"+titleSuffix[i], 100, 0, 200, -1, 1, histoDir_);
      mapHisto_[name+"VSEta"+nameSuffix_[i]] = new HResolution (name+"VSEta"+nameSuffix_[i], "resolution VS #eta"+titleSuffix[i], 100, -3, 3, -1, 1, histoDir_);
      mapHisto_[name+"VSPhi"+nameSuffix_[i]] = new HResolution (name+"VSPhi"+nameSuffix_[i], "resolution VS #phi"+titleSuffix[i], 100, -3.2, 3.2, -1, 1, histoDir_);
    }

    // single particles histograms
    muMinus.reset( new HDelta("muMinus") );
    muPlus.reset( new HDelta("muPlus") );
  }

  ~HMassResolutionVSPart(){
  }

  virtual void Fill( const reco::Particle::LorentzVector & recoP1, const int charge1,
                     const reco::Particle::LorentzVector & genP1,
                     const reco::Particle::LorentzVector & recoP2, const int charge2,
                     const reco::Particle::LorentzVector & genP2,
                     const double & recoMass, const double & genMass ) {
    muMinus->Fill(recoP1, genP1);
    muPlus->Fill(recoP2, genP2);

    Fill( recoP1, charge1, recoP2, charge2, recoMass, genMass );
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

    double pairDeltaEta = fabs(recoEta[0] - recoEta[1]);
    double pairDeltaPhi = MuScleFitUtils::deltaPhi(recoPhi[0], recoPhi[1]);

    mapHisto_[name_]->Fill(massRes);
    mapHisto_[name_+"VSPairPt"]->Fill(pairPt, massRes);
    mapHisto_[name_+"VSPairDeltaEta"]->Fill(pairDeltaEta, massRes);
    mapHisto_[name_+"VSPairDeltaPhi"]->Fill(pairDeltaPhi, massRes);

    // Resolution vs single muon quantities
    // ------------------------------------
    int index = 0;
    for( int i=0; i<2; ++i ) {
      index = id[i];
      mapHisto_[name_+"VSPt"+nameSuffix_[i]]->Fill(recoPt[i], massRes);
      mapHisto_[name_+"VSEta"+nameSuffix_[i]]->Fill(recoEta[i], massRes);
      mapHisto_[name_+"VSPhi"+nameSuffix_[i]]->Fill(recoPhi[i], massRes);
    }
  } 

  virtual void Write() {
    histoDir_->cd();

    for (map<TString, TH1*>::const_iterator histo=mapHisto_.begin(); 
         histo!=mapHisto_.end(); histo++) {
      (*histo).second->Write();
    }

    // Create the new dir and cd into it
    (histoDir_->mkdir("singleMuonsVSgen"))->cd();

    muMinus->Write();
    muPlus->Write();
  }
  
  virtual void Clear() {
    for (map<TString, TH1*>::const_iterator histo=mapHisto_.begin(); 
         histo!=mapHisto_.end(); histo++) {
      (*histo).second->Clear();
    }
    muMinus->Clear();
    muPlus->Clear();
  }

//   HMassResolutionVSPart(const TString & name, TFile* file){
//     string nameSuffix[] = {"Plus", "Minus"};
//     name_ = name;
//     hReso                    = (TH1F *)     file->Get(name+"_Reso");
//     hResoVSPairPt            = (TH2F *)     file->Get(name+"_ResoVSPairPt");
//     hResoVSPairDeltaEta      = (TH2F *)     file->Get(name+"_ResoVSPairDeltaEta");
//     hResoVSPairDeltaPhi      = (TH2F *)     file->Get(name+"_ResoVSPairDeltaPhi");
//     for( int i=0; i<2; ++i ) {
//       hResoVSPt[i]           = (TH2F *)     file->Get(name+"_ResoVSPt"+nameSuffix[i]);
//       hResoVSEta[i]          = (TH2F *)     file->Get(name+"_ResoVSEta"+nameSuffix[i]);
//       hResoVSPhi[i]          = (TH2F *)     file->Get(name+"_ResoVSPhi"+nameSuffix[i]);
//     }
//   }

 protected:
  map<TString, TH1*> mapHisto_;
  TString nameSuffix_[2];
  auto_ptr<HDelta> muMinus;
  auto_ptr<HDelta> muPlus;
};

#endif
