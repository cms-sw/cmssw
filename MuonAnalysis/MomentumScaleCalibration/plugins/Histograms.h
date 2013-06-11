#ifndef Histograms_H
#define Histograms_H

/** \class Histograms
 *  Collection of histograms for GLB muon analysis
 *
 *  $Date: 2012/09/07 07:46:16 $
 *  $Revision: 1.36 $
 *  \author S. Bolognesi - INFN Torino / T.Dorigo - INFN Padova
 */

#include <CLHEP/Vector/LorentzVector.h>
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "MuScleFitUtils.h"

#include "TH1D.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TH3F.h"
#include "TFile.h"
#include "TString.h"
#include "TProfile.h"
#include "TProfile2D.h"
#include "TF1.h"
#include "TGraphErrors.h"
#include "TFile.h"
#include "TSystem.h"
#include "TCanvas.h"

#include "TLorentzVector.h"
#include <vector>
#include <string>
#include <iostream>
#include "TMath.h"

class Histograms {
public:

  // Constructor
  // -----------
  Histograms() : theWeight_(1), histoDir_(0) {};
  Histograms( const TString & name ) : theWeight_(1), name_(name), histoDir_(0) {}
  Histograms( TFile * outputFile, const TString & name ) :
    theWeight_(1),
    name_(name),
    outputFile_(outputFile),
    histoDir_( outputFile->GetDirectory(name) )
  {
    if( histoDir_ == 0 ) {
      histoDir_ = outputFile->mkdir(name);
    }
    histoDir_->cd();
  }

  // Destructor
  // ----------
  virtual ~Histograms() {};

  // Operations
  // ----------
  //   virtual void Fill( const reco::Particle::LorentzVector & p4 ) {};
  //   virtual void Fill( const CLHEP::HepLorentzVector & momentum ) {};
  virtual void Fill( const reco::Particle::LorentzVector & p1, const reco::Particle::LorentzVector & p2 ) {};
  virtual void Fill( const reco::Particle::LorentzVector & p1, const reco::Particle::LorentzVector & p2, const int charge, const double & weight = 1.) {};
  virtual void Fill( const CLHEP::HepLorentzVector & momentum1, const CLHEP::HepLorentzVector & momentum2 ) {};
  virtual void Fill( const CLHEP::HepLorentzVector & momentum1, const CLHEP::HepLorentzVector & momentum2, const int charge, const double & weight = 1.) {};
  virtual void Fill( const CLHEP::HepLorentzVector & p1, const reco::Particle::LorentzVector & p2 ) {};
  virtual void Fill( const reco::Particle::LorentzVector & p4, const double & weight = 1. ) {};

  //virtual void Fill( const CLHEP::HepLorentzVector & momentum, const double & weight ) {};
  //------
  virtual void Fill( const reco::Particle::LorentzVector & p4, const int charge, const double & weight = 1. ) {};

  virtual void Fill( const CLHEP::HepLorentzVector & momentum, const int charge, const double & weight =1.){};
  //------
  // virtual void Fill( const reco::Particle::LorentzVector & p4, const double & likeValue ) {};
  virtual void Fill( const reco::Particle::LorentzVector & p4, const double & resValue, const int charge ) {};
  virtual void Fill( const reco::Particle::LorentzVector & p4, const double & genValue, const double recValue, const int charge ) {};
  virtual void Fill( const CLHEP::HepLorentzVector & p, const double & likeValue ) {};
  virtual void Fill( const unsigned int number ) {};
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
  virtual void Fill( const double & x, const double & y ) {};
  virtual void Fill( const double & x, const double & y, const double & a, const double & b ) {};
  virtual void Fill(const reco::Particle::LorentzVector & p41,
		    const reco::Particle::LorentzVector & p42,
		    const reco::Particle::LorentzVector & p4Res,
		    const double & weight = 1.) {};
  virtual void Fill(const CLHEP::HepLorentzVector & momentum1,
		    const CLHEP::HepLorentzVector & momentum2,
		    const CLHEP::HepLorentzVector & momentumRes,
		    const double & weight = 1.) {};


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

/// A wrapper for the TH2D histogram to allow it to be put inside the same map as all the other classes in this file
class HTH2D : public Histograms
{
public:
  HTH2D( TFile * outputFile, const TString & name, const TString & title, const TString & dirName,
         const int xBins, const double & xMin, const double & xMax,
         const int yBins, const double & yMin, const double & yMax ) : Histograms(outputFile, dirName),
                                                                       tH2d_( new TH2D(name, title, xBins, xMin, xMax, yBins, yMin, yMax) ),
                                                                       tProfile_( new TProfile(name+"Prof", title+" profile", xBins, xMin, xMax, yMin, yMax) ) {}
  ~HTH2D() {
    delete tH2d_;
    delete tProfile_;
  }
  virtual void Fill( const double & x, const double & y ) {
    tH2d_->Fill(x,y);
    tProfile_->Fill(x,y);
  }
  virtual void Write() {
    if(histoDir_ != 0) histoDir_->cd();
    tH2d_->Write();
    tProfile_->Write();
  }
  virtual void Clear() {
    tH2d_->Clear();
    tProfile_->Clear();
  }
  virtual void SetXTitle(const TString & title) {
    tH2d_->GetXaxis()->SetTitle(title);
    tProfile_->GetXaxis()->SetTitle(title);
  }
  virtual void SetYTitle(const TString & title) {
    tH2d_->GetYaxis()->SetTitle(title);
    tProfile_->GetYaxis()->SetTitle(title);
  }
  TH2D * operator->() { return tH2d_; }
  TProfile * getProfile() { return tProfile_; }
protected:
  TH2D * tH2d_;
  TProfile * tProfile_;
};

/// A wrapper for the TH1D histogram to allow it to be put inside the same map as all the other classes in this file
class HTH1D : public Histograms
{
public:
  HTH1D( TFile * outputFile, const TString & name, const TString & title,
         const int xBins, const double & xMin, const double & xMax ) : Histograms(outputFile, name),
                                                                       tH1D_( new TH1D(name, title, xBins, xMin, xMax) ) {}
  ~HTH1D() {
    delete tH1D_;
  }
  virtual void Fill( const double & x, const double & y ) {
    tH1D_->Fill(x, y);
  }
  virtual void Write() {
    if(histoDir_ != 0) histoDir_->cd();
    tH1D_->Write();
  }
  virtual void Clear() {
    tH1D_->Clear();
  }
  virtual void SetXTitle(const TString & title) {
    tH1D_->GetXaxis()->SetTitle(title);
  }
  virtual void SetYTitle(const TString & title) {
    tH1D_->GetYaxis()->SetTitle(title);
  }
  TH1D * operator->() { return tH1D_; }
protected:
  TH1D * tH1D_;
};

/// A wrapper for the TProfile histogram to allow it to be put inside the same map as all the other classes in this file
class HTProfile : public Histograms
{
public:
  HTProfile( TFile * outputFile, const TString & name, const TString & title,
             const int xBins, const double & xMin, const double & xMax,
             const double & yMin, const double & yMax ) : Histograms(outputFile, name),
                                                          tProfile_( new TProfile(name+"Prof", title+" profile", xBins, xMin, xMax, yMin, yMax) ) {}
  ~HTProfile() {
    delete tProfile_;
  }
  virtual void Fill( const double & x, const double & y ) {
    tProfile_->Fill(x,y);
  }
  virtual void Write() {
    if(histoDir_ != 0) histoDir_->cd();
    tProfile_->Write();
  }
  virtual void Clear() {
    tProfile_->Clear();
  }
  virtual void SetXTitle(const TString & title) {
    tProfile_->GetXaxis()->SetTitle(title);
  }
  virtual void SetYTitle(const TString & title) {
    tProfile_->GetYaxis()->SetTitle(title);
  }
  TProfile * operator->() { return tProfile_; }
protected:
  TProfile * tProfile_;
};

// -----------------------------------------------------
// A set of histograms of particle kinematical variables
// -----------------------------------------------------
class HParticle : public Histograms {
 public:
  HParticle( const TString & name, const double & minMass = 0., const double & maxMass = 200., const double & maxPt = 100. ) :
    Histograms(name),
    // Kinematical variables
    hPt_(      new TH1F (name+"_Pt",      "transverse momentum", 100, 0, maxPt) ),
    hPtVsEta_( new TH2F (name+"_PtVsEta", "transverse momentum vs #eta", 100, 0, maxPt, 100, -3.0, 3.0) ),
   
    hCurvVsEtaNeg_( new TProfile(name+"_CurvVsEtaNeg", "q/pT vs #eta neg.", 64, -3.2, 3.2, -1., 0.) ),
    hCurvVsEtaPos_( new TProfile(name+"_CurvVsEtaPos", "q/pT vs #eta pos.", 64, -3.2, 3.2,  0., 1.) ),
    hCurvVsPhiNeg_( new TProfile(name+"_CurvVsPhiNeg", "q/pT vs #phi neg.", 32, -3.2, 3.2, -1., 0.) ),
    hCurvVsPhiPos_( new TProfile(name+"_CurvVsPhiPos", "q/pT vs #phi pos.", 32, -3.2, 3.2,  0., 1.) ),

    hPtVsPhiNeg_( new TProfile(name+"_PtVsPhiNeg", "pT vs #phi neg.", 32, -3.2, 3.2, 0.,100) ),
    hPtVsPhiPos_( new TProfile(name+"_PtVsPhiPos", "pT vs #phi pos.", 32, -3.2, 3.2, 0.,100) ),

    
    hEta_(     new TH1F (name+"_Eta",     "pseudorapidity", 64, -3.2, 3.2) ),
    hPhi_(     new TH1F (name+"_Phi",     "phi angle",      64, -3.2, 3.2) ),
    hMass_(    new TH1F (name+"_Mass",    "mass", 10000, minMass, maxMass) ),
    hNumber_( new TH1F (name+"_Number", "number", 20, -0.5, 19.5) )
  {}

  /// Constructor that puts the histograms inside a TDirectory
  HParticle( TFile* outputFile, const TString & name, const double & minMass = 0., const double & maxMass = 200., const double & maxPt = 100. ) :
    Histograms(outputFile, name)
  {
    // Kinematical variables
    hPt_ =      new TH1F (name+"_Pt", "transverse momentum", 100, 0, maxPt);
    hPtVsEta_ = new TH2F (name+"_PtVsEta", "transverse momentum vs #eta", 100, 0, maxPt, 100, -3.0, 3.0);
    
    hPtVsEta_ = new TH2F (name+"_PtVsEta", "transverse momentum vs #eta", 100, 0, maxPt, 100, -3.0, 3.0);
   
    hCurvVsEtaNeg_ = new TProfile(name+"_CurvVsEtaNeg", "q/pT vs #eta neg.", 100, -3.0, 3.0, -1. ,0.);
    hCurvVsEtaPos_ = new TProfile(name+"_CurvVsEtaPos", "q/pT vs #eta pos.", 100, -3.0, 3.0, 0., 1.);
    hCurvVsPhiNeg_ = new TProfile(name+"_CurvVsPhiNeg", "q/pT vs #phi neg.", 32, -3.2, 3.2, -1. ,0.);
    hCurvVsPhiPos_ = new TProfile(name+"_CurvVsPhiPos", "q/pT vs #phi pos.", 32, -3.2, 3.2, 0., 1.);

    hPtVsPhiNeg_ = new TProfile(name+"_PtVsPhiNeg", "pT vs #phi neg.", 32, -3.2, 3.2, 0.,100);
    hPtVsPhiPos_ = new TProfile(name+"_PtVsPhiPos", "pT vs #phi pos.", 32, -3.2, 3.2, 0.,100);


    //hPtVSPhi_prof_ = new TProfile (name+"_PtVSPhi_prof", "pt vs phi angle",12, -3.2, 3.2, 0, 200);

    hEta_ =     new TH1F (name+"_Eta", "pseudorapidity", 64, -3.2, 3.2);
    hPhi_ =     new TH1F (name+"_Phi", "phi angle",      64, -3.2, 3.2);
    hMass_ =    new TH1F (name+"_Mass", "mass", 40000, minMass, maxMass);
    hNumber_ = new TH1F (name+"_Number", "number", 20, -0.5, 19.5);
  }

  HParticle( const TString & name, TFile* file ) :
    Histograms(name),
    hPt_(      (TH1F *) file->Get(name_+"_Pt") ),
    hPtVsEta_( (TH2F *) file->Get(name_+"_PtVsEta") ),

   
    hCurvVsEtaNeg_( (TProfile *) file->Get(name_+"_CurvVsEtaNeg") ),   
    hCurvVsEtaPos_( (TProfile *) file->Get(name_+"_CurvVsEtaPos") ),
    hCurvVsPhiNeg_( (TProfile *) file->Get(name_+"_CurvVsPhiNeg") ),   
    hCurvVsPhiPos_( (TProfile *) file->Get(name_+"_CurvVsPhiPos") ),

    hPtVsPhiNeg_( (TProfile *) file->Get(name_+"_PtVsPhiNeg") ),   
    hPtVsPhiPos_( (TProfile *) file->Get(name_+"_PtVsPhiPos") ),

    hEta_(     (TH1F *) file->Get(name_+"_Eta") ),
    hPhi_(     (TH1F *) file->Get(name_+"_Phi") ),
    hMass_(    (TH1F *) file->Get(name_+"_Mass") ),
    //hMass_fine_ = (TH1F *) file->Get(name_+"_Mass_fine");
    hNumber_( (TH1F *) file->Get(name_+"_Number") )
  {}

  ~HParticle()
  {
    delete hPt_;
    delete hPtVsEta_;

    delete hCurvVsEtaNeg_; 
    delete hCurvVsEtaPos_;
    delete hCurvVsPhiNeg_; 
    delete hCurvVsPhiPos_;

    delete hPtVsPhiNeg_; 
    delete hPtVsPhiPos_;

    delete hEta_;
    delete hPhi_;
    delete hMass_;
    // delete hMass_fine_;
    delete hNumber_;
  }

  virtual void Fill( const reco::Particle::LorentzVector & p4, const int charge, const double & weight = 1. )
  {
    Fill(CLHEP::HepLorentzVector(p4.x(),p4.y(),p4.z(),p4.t()),charge, weight);
  }

  virtual void Fill( const CLHEP::HepLorentzVector & momentum, const int charge, const double & weight =1.)
  {
    hPt_->Fill(momentum.perp(), weight);
    hPtVsEta_->Fill(momentum.perp(), momentum.eta(), weight);
   
    //   std::cout<< "charge-> " <<charge<<std::endl;
    if(charge<0)hCurvVsEtaNeg_->Fill( momentum.eta(),charge/(momentum.perp()),weight);     
    if(charge>0)hCurvVsEtaPos_->Fill( momentum.eta(),charge/(momentum.perp()),weight);
    if(charge<0)hCurvVsPhiNeg_->Fill( momentum.phi(),charge/(momentum.perp()),weight);     
    if(charge>0)hCurvVsPhiPos_->Fill( momentum.phi(),charge/(momentum.perp()),weight);

    if(charge<0)hPtVsPhiNeg_->Fill( momentum.phi(),momentum.perp(),weight);     
    if(charge>0)hPtVsPhiPos_->Fill( momentum.phi(),momentum.perp(),weight);

    hEta_->Fill(momentum.eta(), weight);
    hPhi_->Fill(momentum.phi(), weight);
    hMass_->Fill(momentum.m(), weight);
    //hMass_fine_->Fill(momentum.m(), weight);
  }


  
  virtual void Fill( unsigned int number )
  {
    hNumber_->Fill(number);
  }

  virtual void Write()
  {
    if(histoDir_ != 0) histoDir_->cd();
    hPt_->Write();
    hPtVsEta_->Write();
   
    hCurvVsEtaNeg_->Write();
    hCurvVsEtaPos_->Write();
    hCurvVsPhiNeg_->Write();
    hCurvVsPhiPos_->Write();

    hPtVsPhiNeg_->Write();
    hPtVsPhiPos_->Write();

    hEta_->Write();    
    hPhi_->Write();
    hMass_->Write();
    //hMass_fine_->Write();
    hNumber_->Write();
  }
  
  virtual void Clear()
  {
    hPt_->Clear();
    hPtVsEta_->Clear();
    
    hCurvVsEtaNeg_->Clear();
    hCurvVsEtaPos_->Clear();
    hCurvVsPhiNeg_->Clear();
    hCurvVsPhiPos_->Clear();
    
    hPtVsPhiNeg_->Clear();
    hPtVsPhiPos_->Clear();

    hEta_->Clear();    
    hPhi_->Clear();
    hMass_->Clear();
    //hMass_fine_->Clear();
    hNumber_->Clear();
  }

 protected:
  TH1F* hPt_;
  TH2F* hPtVsEta_;
 
  TProfile* hCurvVsEtaNeg_; 
  TProfile* hCurvVsEtaPos_;
  TProfile* hCurvVsPhiNeg_; 
  TProfile* hCurvVsPhiPos_;
  
  TProfile* hPtVsPhiNeg_; 
  TProfile* hPtVsPhiPos_;

  TH1F* hEta_;
  TH1F* hPhi_;
  TH1F* hMass_;
  //TH1F* hMass_fine_;
  TH1F* hNumber_;
};

// ---------------------------------------------------
// A set of histograms for distances between particles
// ---------------------------------------------------
class HDelta : public Histograms
{
 public:
  HDelta (const TString & name) :
    Histograms(name),
    // Kinematical variables
    // ---------------------
    hEta_( new TH1F (name+"_DeltaEta", "#Delta#eta", 100, 0, 6) ),
    hEtaSign_( new TH1F (name+"_DeltaEtaSign", "#Delta#eta with sign", 100, -6, 6) ),
    hPhi_( new TH1F (name+"_DeltaPhi", "#Delta#phi", 100,0,3.2) ),
    hTheta_( new TH1F (name+"_DeltaTheta", "#Delta#theta", 100,-3.2,3.2) ),
    hCotgTheta_( new TH1F (name+"_DeltaCotgTheta", "#Delta Cotg(#theta )", 100,-3.2,3.2) ),
    hDeltaR_( new TH1F (name+"_DeltaR","#Delta R", 400, 0, 4 ) )
  {}

  HDelta (TFile* outputFile, const TString & name) :
    Histograms(outputFile, name),
    // Kinematical variables
    // ---------------------
    hEta_( new TH1F (name+"_DeltaEta", "#Delta#eta", 100, 0, 6) ),
    hEtaSign_( new TH1F (name+"_DeltaEtaSign", "#Delta#eta with sign", 100, -6, 6) ),
    hPhi_( new TH1F (name+"_DeltaPhi", "#Delta#phi", 100,0,3.2) ),
    hTheta_( new TH1F (name+"_DeltaTheta", "#Delta#theta", 100,-3.2,3.2) ),
    hCotgTheta_( new TH1F (name+"_DeltaCotgTheta", "#Delta Cotg(#theta )", 100,-3.2,3.2) ),
    hDeltaR_( new TH1F (name+"_DeltaR","#DeltaR", 400, 0, 4 ) )
  {}

  HDelta (const TString & name, TFile* file) {
    name_ = name;
    hEta_       = (TH1F *) file->Get(name+"_DeltaEta");
    hEtaSign_   = (TH1F *) file->Get(name+"_DeltaEtaSign");
    hPhi_       = (TH1F *) file->Get(name+"_DeltaPhi");
    hTheta_     = (TH1F *) file->Get(name+"_DeltaTheta");
    hCotgTheta_ = (TH1F *) file->Get(name+"_DeltaCotgTheta");
    hDeltaR_    = (TH1F *) file->Get(name+"_DeltaR");
   }

  ~HDelta() {
    delete hEta_;
    delete hEtaSign_;
    delete hPhi_;
    delete hTheta_;
    delete hCotgTheta_;
    delete hDeltaR_;
  }
  
  virtual void Fill (const reco::Particle::LorentzVector & p1, const reco::Particle::LorentzVector & p2) {
    Fill (CLHEP::HepLorentzVector(p1.x(),p1.y(),p1.z(),p1.t()), 
	  CLHEP::HepLorentzVector(p2.x(),p2.y(),p2.z(),p2.t()));
  }

  virtual void Fill (const CLHEP::HepLorentzVector & p1, const reco::Particle::LorentzVector & p2) {
    Fill (p1,CLHEP::HepLorentzVector(p2.x(),p2.y(),p2.z(),p2.t()));
  }

  virtual void Fill (const CLHEP::HepLorentzVector & momentum1, const CLHEP::HepLorentzVector & momentum2) {
    hEta_->Fill(fabs( momentum1.eta()-momentum2.eta() ));
    hEtaSign_->Fill(momentum1.eta()-momentum2.eta());
    hPhi_->Fill(MuScleFitUtils::deltaPhi(momentum1.phi(),momentum2.phi()));
    hTheta_->Fill(momentum1.theta()-momentum2.theta());
    // hCotgTheta->Fill(1/(TMath::Tan(momentum1.theta()))-1/(TMath::Tan(momentum2.theta())));
    double theta1 = momentum1.theta();
    double theta2 = momentum2.theta();
    hCotgTheta_->Fill(TMath::Cos(theta1)/TMath::Sin(theta1) - TMath::Cos(theta2)/TMath::Sin(theta2));
    hDeltaR_->Fill(sqrt((momentum1.eta()-momentum2.eta())*(momentum1.eta()-momentum2.eta()) +
                        (MuScleFitUtils::deltaPhi(momentum1.phi(),momentum2.phi()))*
                        (MuScleFitUtils::deltaPhi(momentum1.phi(),momentum2.phi()))));
  }
  
  virtual void Write() {
    if(histoDir_ != 0) histoDir_->cd();

    hEta_->Write();
    hEtaSign_->Write();
    hPhi_->Write();
    hTheta_->Write();
    hCotgTheta_->Write();
    hDeltaR_->Write();
  }
  
  virtual void Clear() {
    hEta_->Clear();
    hEtaSign_->Clear();
    hPhi_->Clear();
    hTheta_->Clear();
    hDeltaR_->Clear();
    hCotgTheta_->Clear();
  }
  
 public:
  TH1F* hEta_;
  TH1F* hEtaSign_;
  TH1F* hPhi_;
  TH1F* hTheta_;
  TH1F* hCotgTheta_;
  TH1F* hDeltaR_;
};

// ------------------------------------------------------------
// A set of histograms of particle kinematical variables vs eta
// ------------------------------------------------------------
class HPartVSEta : public Histograms
{
 public:
  HPartVSEta(const TString & name, const double & minMass = 0., const double & maxMass = 100., const double & maxPt = 100.)
  {
    name_ = name;
    hPtVSEta_ = new TH2F( name+"_PtVSEta", "transverse momentum vs pseudorapidity", 
                          32, -3.2, 3.2, 200, 0, maxPt );
    hMassVSEta_ = new TH2F( name+"_MassVSEta", "mass vs pseudorapidity", 
                            32, -3.2, 3.2, 40, minMass, maxMass );
    // TD profile histograms
    // ---------------------
    hPtVSEta_prof_ = new TProfile( name+"_PtVSEta_prof", "mass vs pseudorapidity", 
                                   32, -3.2, 3.2, 0, maxPt );
    hMassVSEta_prof_ = new TProfile( name+"_MassVSEta_prof", "mass vs pseudorapidity", 
                                     32, -3.2, 3.2, minMass, maxMass );
    hCurvVSEta_prof_ = new TProfile( name+"_CurvVSEta_prof", "curvature vs pseudorapidity", 
                                     32, -3.2, 3.2, 0, 1. );
  }

  ~HPartVSEta() {
    delete hPtVSEta_;
    delete hMassVSEta_;
    delete hPtVSEta_prof_;
    delete hMassVSEta_prof_;
    delete hCurvVSEta_prof_;
  }

  virtual void Fill (const reco::Particle::LorentzVector & p4, const double & weight = 1.) {
    Fill (CLHEP::HepLorentzVector(p4.x(),p4.y(),p4.z(),p4.t()), weight);
  }

  virtual void Fill (const CLHEP::HepLorentzVector & momentum, const double & weight = 1.) {
    hPtVSEta_->Fill(momentum.eta(),momentum.perp(), weight);
    hPtVSEta_prof_->Fill(momentum.eta(),momentum.perp(), weight);

    hMassVSEta_->Fill(momentum.eta(),momentum.m(), weight);
    hMassVSEta_prof_->Fill(momentum.eta(),momentum.m(), weight);
    hCurvVSEta_prof_->Fill(momentum.eta(),1/momentum.perp(), weight);
  }
    
  virtual void Write() {
    hPtVSEta_->Write();
    hPtVSEta_prof_->Write();
    hCurvVSEta_prof_->Write();
    hMassVSEta_->Write();
    hMassVSEta_prof_->Write();

    //     std::vector<TGraphErrors*> graphs( (MuScleFitUtils::fitMass(hMassVSEta_)) );
    //     for (std::vector<TGraphErrors*>::const_iterator graph = graphs.begin(); graph != graphs.end(); graph++) {
    //       (*graph)->Write();
    //     }
  }

  virtual void Clear() {
    hPtVSEta_->Clear();
    hPtVSEta_prof_->Clear();
    hCurvVSEta_prof_->Clear();
    hMassVSEta_->Clear();
    hMassVSEta_prof_->Clear();

  }

 public:

  TH2F *hPtVSEta_;
  TH2F *hMassVSEta_; 
  TProfile *hMassVSEta_prof_; 
  TProfile *hPtVSEta_prof_;  
  TProfile *hCurvVSEta_prof_;  
};

//---------------------------------------------------------------------------
// A set of histograms of particle kinematical variables vs phi (in eta bins)
// --------------------------------------------------------------------------
class HPartVSPhi : public Histograms
{
 public:
  HPartVSPhi(const TString & name){
    name_ = name;
//     hPtVSPhi_ = new TH2F (name+"_PtVSPhi", "transverse momentum vs phi angle",
//                           12, -3.2, 3.2, 200, 0, 200);
//     hMassVSPhi_ = new TH2F (name+"_MassVSPhi", "mass vs phi angle", 
//                             7, -3.2, 3.2, 40, 70, 110);
//     hMassVSPhiF_ = new TH2F (name+"_MassVSPhiF", "mass vs phi F", 
//                              7, -3.2, 3.2, 40, 70, 110);
//     hMassVSPhiWp2_ = new TH2F (name+"_MassVSPhiWp2", "mass vs phi Wp2", 
//                                7, -3.2, 3.2, 40, 70, 110);
//     hMassVSPhiWp1_ = new TH2F (name+"_MassVSPhiWp1", "mass vs phi Wp1", 
//                                7, -3.2, 3.2, 40, 70, 110);
//     hMassVSPhiW0_ = new TH2F (name+"_MassVSPhiW0", "mass vs phi W0", 
//                               7, -3.2, 3.2, 40, 70, 110);
//     hMassVSPhiWm1_ = new TH2F (name+"_MassVSPhiWm1", "mass vs phi Wm1", 
//                                7, -3.2, 3.2, 40, 70, 110);
//     hMassVSPhiWm2_ = new TH2F (name+"_MassVSPhiWm2", "mass vs phi Wm2", 
//                                7, -3.2, 3.2, 40, 70, 110);
//     hMassVSPhiB_ = new TH2F (name+"_MassVSPhiB", "mass vs phi B", 
//                              7, -3.2, 3.2, 40, 70, 110);  

    // TD profile histograms
    hMassVSPhi_prof_ = new TProfile (name+"_MassVSPhi_prof", "mass vs phi angle", 
                                     16, -3.2, 3.2, 70, 110);
    hPtVSPhi_prof_ = new TProfile (name+"_PtVSPhi_prof", "pt vs phi angle", 
                                     16, -3.2, 3.2, 0, 200);
  }

  ~HPartVSPhi() {
    delete hPtVSPhi_;
    delete hMassVSPhi_;
    delete hMassVSPhi_prof_;
    delete hPtVSPhi_prof_;

 //    delete hMassVSPhiB_;
//     delete hMassVSPhiWm2_;
//     delete hMassVSPhiWm1_;
//     delete hMassVSPhiW0_;
//     delete hMassVSPhiWp1_;
//     delete hMassVSPhiWp2_;
//     delete hMassVSPhiF_;
  }

  void Fill(const reco::Particle::LorentzVector & p4, const double & weight =1.) {
    Fill(CLHEP::HepLorentzVector(p4.x(),p4.y(),p4.z(),p4.t()), weight);
  }

  void Fill(const CLHEP::HepLorentzVector & momentum, const double & weight= 1.) {
    hPtVSPhi_->Fill(momentum.phi(),momentum.perp(), weight);
    hMassVSPhi_->Fill(momentum.phi(),momentum.m(), weight);
    hMassVSPhi_prof_->Fill(momentum.phi(),momentum.m(), weight);
    hPtVSPhi_prof_->Fill(momentum.phi(),momentum.perp(), weight);
 
//     if (momentum.eta()<-1.2)                             hMassVSPhiB_->Fill(momentum.phi(),momentum.m(), weight);
//     else if (momentum.eta()<-0.8 && momentum.eta()>-1.2) hMassVSPhiWm2_->Fill(momentum.phi(),momentum.m(), weight);
//     else if (momentum.eta()<-0.3 && momentum.eta()>-0.8) hMassVSPhiWm1_->Fill(momentum.phi(),momentum.m(), weight);
//     else if ((fabs(momentum.eta())) < 0.3)               hMassVSPhiW0_->Fill(momentum.phi(),momentum.m(), weight);
//     else if (momentum.eta()>0.3 && momentum.eta()<0.8)   hMassVSPhiWp1_->Fill(momentum.phi(),momentum.m(), weight);
//     else if (momentum.eta()>0.8 && momentum.eta()<1.2)   hMassVSPhiWp2_->Fill(momentum.phi(),momentum.m(), weight);
//     else if (momentum.eta()>1.2)                         hMassVSPhiF_->Fill(momentum.phi(),momentum.m(), weight);
  }

  virtual void Write() {
    hPtVSPhi_->Write();
    hMassVSPhi_->Write();
    hMassVSPhi_prof_->Write();
    hPtVSPhi_prof_->Write();

 //    hMassVSPhiB_->Write();
//     hMassVSPhiWm2_->Write(); 
//     hMassVSPhiWm1_->Write();
//     hMassVSPhiW0_->Write();
//     hMassVSPhiWp1_->Write();
//     hMassVSPhiWp2_->Write();
//     hMassVSPhiF_->Write();

//     std::vector<TGraphErrors*> graphs ((MuScleFitUtils::fitMass(hMassVSPhi_)));
//     for(std::vector<TGraphErrors*>::const_iterator graph = graphs.begin(); graph != graphs.end(); graph++){
//       (*graph)->Write();
//     }
//     std::vector<TGraphErrors*> graphsB ((MuScleFitUtils::fitMass(hMassVSPhiB_)));
//     for(std::vector<TGraphErrors*>::const_iterator graph = graphsB.begin(); graph != graphsB.end(); graph++){
//       (*graph)->Write();
//     }
//     std::vector<TGraphErrors*> graphsWm2 ((MuScleFitUtils::fitMass(hMassVSPhiWm2_)));
//     for(std::vector<TGraphErrors*>::const_iterator graph = graphsWm2.begin(); graph != graphsWm2.end(); graph++){
//       (*graph)->Write();
//     }
//     std::vector<TGraphErrors*> graphsWm1 ((MuScleFitUtils::fitMass(hMassVSPhiWm1_)));
//     for(std::vector<TGraphErrors*>::const_iterator graph = graphsWm1.begin(); graph != graphsWm1.end(); graph++){
//       (*graph)->Write();
//     }
//     std::vector<TGraphErrors*> graphsW0 ((MuScleFitUtils::fitMass(hMassVSPhiW0_)));
//     for(std::vector<TGraphErrors*>::const_iterator graph = graphsW0.begin(); graph != graphsW0.end(); graph++){
//       (*graph)->Write();
//     }
//     std::vector<TGraphErrors*> graphsWp1 ((MuScleFitUtils::fitMass(hMassVSPhiWp1_)));
//     for(std::vector<TGraphErrors*>::const_iterator graph = graphsWp1.begin(); graph != graphsWp1.end(); graph++){
//       (*graph)->Write();
//     }
//     std::vector<TGraphErrors*> graphsWp2 ((MuScleFitUtils::fitMass(hMassVSPhiWp2_)));
//     for(std::vector<TGraphErrors*>::const_iterator graph = graphsWp2.begin(); graph != graphsWp2.end(); graph++){
//       (*graph)->Write();
//     }
//     std::vector<TGraphErrors*> graphsF ((MuScleFitUtils::fitMass(hMassVSPhiF_)));
//     for(std::vector<TGraphErrors*>::const_iterator graph = graphsF.begin(); graph != graphsF.end(); graph++){
//       (*graph)->Write();
//     }
  }

  virtual void Clear() {
    hPtVSPhi_->Clear();
    hMassVSPhi_->Clear();
    hPtVSPhi_prof_->Clear();
    hMassVSPhi_prof_->Clear();

//     hMassVSPhiB_->Clear();
//     hMassVSPhiWm2_->Clear(); 
//     hMassVSPhiWm1_->Clear();
//     hMassVSPhiW0_->Clear();
//     hMassVSPhiWp1_->Clear();
//     hMassVSPhiWp2_->Clear();
//     hMassVSPhiF_->Clear();
  }
  
 public:
  TH2F *hPtVSPhi_;
  TH2F *hMassVSPhi_;
  TProfile *hMassVSPhi_prof_;
  TProfile *hPtVSPhi_prof_;

//   TH2F *hMassVSPhiB_;
//   TH2F *hMassVSPhiWm2_;
//   TH2F *hMassVSPhiWm1_;
//   TH2F *hMassVSPhiW0_;
//   TH2F *hMassVSPhiWp1_;
//   TH2F *hMassVSPhiWp2_;
//   TH2F *hMassVSPhiF_;
};

//---------------------------------------------------------------------------------------
// A set of histograms of particle VS pt
class HPartVSPt : public Histograms
{
 public:
  HPartVSPt(const TString & name){
    name_ = name;
    hMassVSPt_ = new TH2F( name+"_MassVSPt", "mass vs transverse momentum", 
                           12, -6, 6, 40, 70, 110 );
    // TD profile histograms
    hMassVSPt_prof_ = new TProfile( name+"_MassVSPt_prof", "mass vs transverse momentum", 
                                    12, -3, 3, 86, 116 );
  }

  ~HPartVSPt(){
    delete hMassVSPt_;
    delete hMassVSPt_prof_;
  }

  virtual void Fill(const reco::Particle::LorentzVector & p4, const double & weight = 1.) {
    Fill(CLHEP::HepLorentzVector(p4.x(),p4.y(),p4.z(),p4.t()), weight);
  }

  virtual void Fill(const CLHEP::HepLorentzVector & momentum, const double & weight = 1.) {
    hMassVSPt_->Fill(momentum.eta(),momentum.m(), weight);
    hMassVSPt_prof_->Fill(momentum.eta(),momentum.m(), weight);    
  }
    
  virtual void Write() {
    hMassVSPt_->Write();
    hMassVSPt_prof_->Write();
   
//     std::vector<TGraphErrors*> graphs( (MuScleFitUtils::fitMass(hMassVSPt_)) );
//     for(std::vector<TGraphErrors*>::const_iterator graph = graphs.begin(); graph != graphs.end(); graph++){
//       (*graph)->Write();
//     }
  }
  
  virtual void Clear() {
    hMassVSPt_->Clear();
    hMassVSPt_prof_->Clear();
  }

 public:
  TH2F *hMassVSPt_;
  TProfile *hMassVSPt_prof_;
};

// ---------------------------------------------------
// A set of histograms of resonance mass versus muon variables
class HMassVSPart : public Histograms
{
 public:
  HMassVSPart( const TString & name, const double & minMass = 0., const double & maxMass = 150., const double maxPt = 100. ) {
    name_ = name;

    // Kinematical variables
    // ---------------------
    hMassVSPt_       = new TH2F( name+"_MassVSPt", "resonance mass vs muon transverse momentum", 200, 0., maxPt, 6000, minMass, maxMass );

    hMassVSEta_      = new TH2F( name+"_MassVSEta", "resonance mass vs muon pseudorapidity", 64, -6.4, 6.4, 6000, minMass, maxMass );
    hMassVSEtaPlus_  = new TH2F( name+"_MassVSEtaPlus", "resonance mass vs muon+ pseudorapidity",  64, -6.4, 6.4,  6000, minMass, maxMass );
    hMassVSEtaMinus_ = new TH2F( name+"_MassVSEtaMinus", "resonance mass vs muon- pseudorapidity", 64, -6.4, 6.4,  6000, minMass, maxMass );

    hMassVSPhiPlus_  = new TH2F( name+"_MassVSPhiPlus", "resonance mass vs muon+ phi angle",  64, -3.2, 3.2, 6000, minMass, maxMass );
    hMassVSPhiMinus_ = new TH2F( name+"_MassVSPhiMinus", "resonance mass vs muon- phi angle", 64, -3.2, 3.2, 6000, minMass, maxMass );

    
    // J/Psi mass -----
//     hMassVSEtaPhiPlus_      = new TH3F( name+"_MassVSEtaPhiPlus", "resonance mass vs muon+ phi/pseudorapidity",6, -3.2, 3.2, 20, -2.5, 2.5, 6000, minMass, maxMass );
//     hMassVSEtaPhiMinus_      = new TH3F( name+"_MassVSEtaPhiMinus", "resonance mass vs muon- phi/pseudorapidity", 6, -3.2, 3.2, 20, -2.5, 2.5, 6000, minMass, maxMass );
 
    //Z mass -----------
    hMassVSEtaPhiPlus_      = new TH3F( name+"_MassVSEtaPhiPlus",  "resonance mass vs muon+ phi/pseudorapidity", 16, -3.2, 3.2, 20, -2.4, 2.4, 300, minMass, maxMass );
    hMassVSEtaPhiMinus_     = new TH3F( name+"_MassVSEtaPhiMinus", "resonance mass vs muon- phi/pseudorapidity", 16, -3.2, 3.2, 20, -2.4, 2.4, 300, minMass, maxMass );


    hMassVSCosThetaCS_      = new TH2F( name+"_MassVSCosThetaCS", "resonance mass vs cos(theta) (CS frame)", 40, -1., 1., 6000, minMass, maxMass );
    hMassVSPhiCS_           = new TH2F( name+"_MassVSPhiCS", "resonance mass vs phi (CS frame)", 64, -3.2, 3.2, 6000, minMass, maxMass );


    hMassVSPhiPlusPhiMinus_ = new TH3F( name+"_MassVSPhiPlusPhiMinus", "resonance mass vs muon+ phi/muon- phi",16, -3.2, 3.2,16, -3.2, 3.2, 6000, minMass, maxMass );
    hMassVSEtaPlusEtaMinus_ = new TH3F( name+"_MassVSEtaPlusEtaMinus", "resonance mass vs muon+ eta/muon- eta",16, -3.2, 3.2,16, -3.2, 3.2, 6000, minMass, maxMass );

   
    hMassVSPhiPlusMinusDiff_ = new TH2F( name+"_MassVSPhiPlusMinusDiff", "resonance mass vs delta phi between mu+/mu-", 64, -6.4, 6.4, 6000, minMass, maxMass );
    hMassVSEtaPlusMinusDiff_ = new TH2F( name+"_MassVSEtaPlusMinusDiff", "resonance mass vs delta pseudorapidity between mu+/mu-", 32, -4.4, 4.4, 6000, minMass, maxMass );
    hMassVSCosThetaCS_prof   = new TProfile (name+"_MassVScosTheta_prof", "resonance mass vs cosTheta", 40, -1., 1., 85., 95.);   

    //hMassVSPt_prof       = new TProfile (name+"_MassVSPt_prof", "resonance mass vs muon transverse momentum", 100, 0., 200., minMass, maxMass);
    //hMassVSEta_prof      = new TProfile (name+"_MassVSEta_prof", "resonance mass vs muon pseudorapidity", 30, -6., 6., minMass, maxMass);
    //hMassVSPhiPlus_prof  = new TProfile (name+"_MassVSPhiPlus_prof", "resonance mass vs muon+ phi angle", 32, -3.2, 3.2, minMass, maxMass);
    //hMassVSPhiMinus_prof = new TProfile (name+"_MassVSPhiMinus_prof", "resonance mass vs muon- phi angle", 32, -3.2, 3.2, minMass, maxMass);
   }

  HMassVSPart(const TString & name, TFile* file){
    name_=name;
    hMassVSPt_       = (TH2F *) file->Get(name+"_MassVSPt");
    hMassVSEta_      = (TH2F *) file->Get(name+"_MassVSEta");

    hMassVSEtaPhiPlus_    = (TH3F *) file->Get(name+"_MassVSEtaPlus");
    hMassVSEtaPhiMinus_   = (TH3F *) file->Get(name+"_MassVSEtaMinus");
    hMassVSEtaPlus_      = (TH2F *) file->Get(name+"_MassVSEtaPlus");
    hMassVSEtaMinus_      = (TH2F *) file->Get(name+"_MassVSEtaMinus");

    hMassVSPhiPlusMinusDiff_ = (TH2F *) file->Get(name+"_MassVSPhiPlusMinusDiff");
    hMassVSEtaPlusMinusDiff_ = (TH2F *) file->Get(name+"_MassVSEtaPlusMinusDiff");
    
    hMassVSPhiPlus_  = (TH2F *) file->Get(name+"_MassVSPhiPlus");
    hMassVSPhiMinus_ = (TH2F *) file->Get(name+"_MassVSPhiMinus");
    
    hMassVSCosThetaCS_prof  = (TProfile *) file->Get(name+"_MassVScosTheta_prof");  
    //hMassVSPt_prof       = (TProfile *) file->Get(name+"_MassVSPt_prof");
    //hMassVSEta_prof      = (TProfile *) file->Get(name+"_MassVSEta_prof");
    //hMassVSPhiPlus_prof  = (TProfile *) file->Get(name+"_MassVSPhiPlus_prof");
    //hMassVSPhiMinus_prof = (TProfile *) file->Get(name+"_MassVSPhiMinus_prof");
  }

  ~HMassVSPart(){
    delete hMassVSPt_;
    delete hMassVSEta_;
    delete hMassVSPhiPlus_;
    delete hMassVSPhiMinus_;
    delete hMassVSEtaPhiPlus_;
    delete hMassVSEtaPhiMinus_;
    delete hMassVSEtaPlus_;
    delete hMassVSEtaMinus_;
    delete hMassVSPhiPlusPhiMinus_;
    delete hMassVSEtaPlusEtaMinus_;
    delete hMassVSCosThetaCS_;
    delete hMassVSPhiCS_;
    delete hMassVSPhiPlusMinusDiff_;
    delete hMassVSEtaPlusMinusDiff_;
    delete hMassVSCosThetaCS_prof;  
  }

  virtual void Fill(const reco::Particle::LorentzVector & p41, const reco::Particle::LorentzVector & p42, const int charge, const double & weight = 1.)
  {
    Fill(CLHEP::HepLorentzVector(p41.x(),p41.y(),p41.z(),p41.t()),
	 CLHEP::HepLorentzVector(p42.x(),p42.y(),p42.z(),p42.t()), charge, weight);
  }

  virtual void Fill(const reco::Particle::LorentzVector & p41,
		    const reco::Particle::LorentzVector & p42,
		    const reco::Particle::LorentzVector & p4Res,
		    const double & weight = 1.)
  {
    Fill(CLHEP::HepLorentzVector(p41.x(),p41.y(),p41.z(),p41.t()),
	 CLHEP::HepLorentzVector(p42.x(),p42.y(),p42.z(),p42.t()),
	 CLHEP::HepLorentzVector(p4Res.x(),p4Res.y(),p4Res.z(),p4Res.t()),
	 weight);
  }

  /// Used to fill 2D histograms for comparison of opposite charge muons quantities
  virtual void Fill(const CLHEP::HepLorentzVector & momentum1,
		    const CLHEP::HepLorentzVector & momentum2,
		    const CLHEP::HepLorentzVector & momentumRes,
		    const double & weight = 1.)
  {

    /************************************************************************
     *
     * Observable: cos(theta) = 2 Q^-1 (Q^2+Qt^2)^-(1/2) (mu^+ mubar^- - mu^- mubar^+)
     * (computed in Collins-Soper frame)
     *
     ************************************************************************/
    
    double costhetaCS, phiCS;

    CLHEP::HepLorentzVector mu= momentum1;
    CLHEP::HepLorentzVector mubar= momentum2;    
    CLHEP::HepLorentzVector Q(mu+mubar);
    double muplus  = 1.0/sqrt(2.0) * (mu.e() + mu.z());
    double muminus = 1.0/sqrt(2.0) * (mu.e() - mu.z());
    double mubarplus  = 1.0/sqrt(2.0) * (mubar.e() + mubar.z());
    double mubarminus = 1.0/sqrt(2.0) * (mubar.e() - mubar.z());
    //double costheta = 2.0 / Q.Mag() / sqrt(pow(Q.Mag(), 2) + pow(Q.Pt(), 2)) * (muplus * mubarminus - muminus * mubarplus);
    costhetaCS = 2.0 / Q.mag() / sqrt(pow(Q.mag(), 2) + pow(Q.perp(), 2)) * (muplus * mubarminus - muminus * mubarplus);
    if (momentumRes.rapidity()<0) costhetaCS = -costhetaCS;
    



  /************************************************************************
   *
   * 3) tanphi = (Q^2 + Qt^2)^1/2 / Q (Dt dot R unit) /(Dt dot Qt unit)
   *
   ************************************************************************/

  // unit vector on R direction
    CLHEP::HepLorentzVector Pbeam(0.,0.,3500.,3500.);
    CLHEP::Hep3Vector R = Pbeam.vect().cross(Q.vect());
    CLHEP::Hep3Vector Runit = R.unit();


    // unit vector on Qt
    CLHEP::Hep3Vector Qt = Q.vect(); Qt.setZ(0);
    CLHEP::Hep3Vector Qtunit = Qt.unit();


    CLHEP::HepLorentzVector D(mu-mubar);
    CLHEP::Hep3Vector Dt = D.vect(); Dt.setZ(0);
    double tanphi = sqrt(pow(Q.mag(), 2) + pow(Q.perp(), 2)) / Q.mag() * Dt.dot(Runit) / Dt.dot(Qtunit);
    if (momentumRes.rapidity()<0) tanphi = -tanphi;
    phiCS = atan(tanphi); 

    hMassVSPhiCS_->Fill(phiCS,momentumRes.m(), weight);
    hMassVSCosThetaCS_->Fill(costhetaCS,momentumRes.m(), weight);
    hMassVSCosThetaCS_prof ->Fill(costhetaCS,momentumRes.m());  
    /*************************************************************************
     *************************************************************************/
   
    hMassVSPhiPlusPhiMinus_->Fill(momentum1.phi(), momentum2.phi(), momentumRes.m(), weight);
    hMassVSEtaPlusEtaMinus_->Fill(momentum1.eta(), momentum2.eta(), momentumRes.m(), weight);

    hMassVSPhiPlusMinusDiff_->Fill( (momentum1.phi()-momentum2.phi()), momentumRes.m(), weight);
    hMassVSEtaPlusMinusDiff_->Fill( (momentum1.eta()-momentum2.eta()), momentumRes.m(), weight);
  }
  
  virtual void Fill(const CLHEP::HepLorentzVector & momentum1, const CLHEP::HepLorentzVector & momentum2, const int charge, const double & weight = 1.)
  {
     hMassVSPt_->Fill(momentum1.perp(),momentum2.m(), weight); 
     //hMassVSPt_prof_->Fill(momentum1.perp(),momentum2.m()); 
     

     hMassVSEta_->Fill(momentum1.eta(),momentum2.m(), weight);								 
     //hMassVSEta_prof_->Fill(momentum1.eta(),momentum2.m()); 

     if(charge>0){
       hMassVSPhiPlus_->Fill(momentum1.phi(),momentum2.m(), weight);
       hMassVSEtaPlus_->Fill(momentum1.eta(),momentum2.m(), weight); 
       hMassVSEtaPhiPlus_->Fill(momentum1.phi(),momentum1.eta(),momentum2.m(), weight);
     }
     else if(charge<0){
       hMassVSPhiMinus_->Fill(momentum1.phi(),momentum2.m(), weight);
       hMassVSEtaMinus_->Fill(momentum1.eta(),momentum2.m(), weight); 
       hMassVSEtaPhiMinus_->Fill(momentum1.phi(),momentum1.eta(),momentum2.m(), weight);
     }
     else {
       LogDebug("Histograms") << "HMassVSPart: wrong charge value = " << charge << std::endl;
       abort();
     }
   }

  virtual void Write() {
    hMassVSPt_->Write();
    hMassVSEta_->Write();
    hMassVSPhiPlus_->Write();
    hMassVSPhiMinus_->Write();

    hMassVSEtaPhiPlus_->Write();
    hMassVSEtaPhiMinus_->Write();
    hMassVSEtaPlus_->Write();
    hMassVSEtaMinus_->Write();

    hMassVSPhiPlusPhiMinus_->Write();
    hMassVSEtaPlusEtaMinus_->Write();
    hMassVSCosThetaCS_->Write();
    hMassVSPhiCS_->Write();

    hMassVSPhiPlusMinusDiff_->Write();
    hMassVSEtaPlusMinusDiff_->Write();
    hMassVSCosThetaCS_prof->Write();

    //hMassVSPt_prof_->Write();
    //hMassVSEta_prof_->Write();    
    //hMassVSPhiPlus_prof_->Write();
    //hMassVSPhiMinus_prof_->Write();
  }

  virtual void Clear() {
    hMassVSPt_->Clear();
    hMassVSEta_->Clear();    
    hMassVSPhiPlus_->Clear();
    hMassVSPhiMinus_->Clear();

    hMassVSEtaPhiPlus_->Clear();   
    hMassVSEtaPhiMinus_->Clear();   
    hMassVSEtaPlus_->Clear();   
    hMassVSEtaMinus_->Clear();    

    hMassVSPhiPlusPhiMinus_->Clear();
    hMassVSEtaPlusEtaMinus_->Clear();
    hMassVSCosThetaCS_->Clear();
    hMassVSPhiCS_->Clear();
    hMassVSPhiPlusMinusDiff_->Clear();
    hMassVSEtaPlusMinusDiff_->Clear();
    hMassVSCosThetaCS_prof->Clear();

    //hMassVSPt_prof_->Clear();
    //hMassVSEta_prof_->Clear();    
    //hMassVSPhiPlus_prof_->Clear();
    //hMassVSPhiMinus_prof_->Clear();
  }

 protected:
  TH2F* hMassVSPt_;
  TH2F* hMassVSEta_;
  TH2F* hMassVSPhiPlus_; 
  TH2F* hMassVSPhiMinus_; 
  TH2F* hMassVSCosThetaCS_;
  TH2F* hMassVSPhiCS_;

  TH3F* hMassVSEtaPhiPlus_;
  TH3F* hMassVSEtaPhiMinus_;
  TH2F* hMassVSEtaPlus_;
  TH2F* hMassVSEtaMinus_; 

  TH2F* hMassVSPhiPlusMinusDiff_;
  TH2F* hMassVSEtaPlusMinusDiff_;

  TH3F* hMassVSPhiPlusPhiMinus_;
  TH3F* hMassVSEtaPlusEtaMinus_;

  TProfile*  hMassVSCosThetaCS_prof;

  //TProfile* hMassVSPt_prof_;
  //TProfile* hMassVSEta_prof_;
  //TProfile* hMassVSPhiPlus_prof_;
  //TProfile* hMassVSPhiMinus_prof_;
};


// ---------------------------------------------------
// A set of histograms of Z mass versus muon variables
class HMassVSPartProfile : public Histograms
{
 public:
  HMassVSPartProfile( const TString & name, const double & minMass = 0., const double & maxMass = 150., const double maxPt = 100. ) {
    name_ = name;

    // Kinematical variables
    // ---------------------
    hMassVSPt_       = new TProfile2D( name+"_MassVSPt", "resonance mass vs muon transverse momentum", 200, 0., maxPt, 6000, minMass, maxMass, 0., 100. );
    hMassVSEta_      = new TProfile2D( name+"_MassVSEta", "resonance mass vs muon pseudorapidity", 64, -6.4, 6.4, 6000, minMass, maxMass, 0., 100. );
    hMassVSPhiPlus_  = new TProfile2D( name+"_MassVSPhiPlus", "resonance mass vs muon+ phi angle", 64, -3.2, 3.2, 6000, minMass, maxMass, 0., 100. );
    hMassVSPhiMinus_ = new TProfile2D( name+"_MassVSPhiMinus", "resonance mass vs muon- phi angle", 64, -3.2, 3.2, 6000, minMass, maxMass, 0., 100. );
   }

  HMassVSPartProfile(const TString & name, TFile* file){
    name_=name;
    hMassVSPt_       = (TProfile2D *) file->Get(name+"_MassVSPt");
    hMassVSEta_      = (TProfile2D *) file->Get(name+"_MassVSEta");
    hMassVSPhiPlus_  = (TProfile2D *) file->Get(name+"_MassVSPhiPlus");
    hMassVSPhiMinus_ = (TProfile2D *) file->Get(name+"_MassVSPhiMinus");
  }

  ~HMassVSPartProfile(){
    delete hMassVSPt_;
    delete hMassVSEta_;
    delete hMassVSPhiPlus_;
    delete hMassVSPhiMinus_;
  }

  virtual void Fill(const reco::Particle::LorentzVector & p41, const reco::Particle::LorentzVector & p42, const int charge, const double & weight = 1.) {
    Fill(CLHEP::HepLorentzVector(p41.x(),p41.y(),p41.z(),p41.t()),
	 CLHEP::HepLorentzVector(p42.x(),p42.y(),p42.z(),p42.t()), charge, weight);
  }
  
  virtual void Fill(const CLHEP::HepLorentzVector & momentum1, const CLHEP::HepLorentzVector & momentum2, const int charge, const double & weight = 1.) { 
    hMassVSPt_->Fill(momentum1.perp(),momentum2.m(), weight); 
    hMassVSEta_->Fill(momentum1.eta(),momentum2.m(), weight); 
    if(charge>0){
      hMassVSPhiPlus_->Fill(momentum1.phi(), momentum2.m(), weight);
    }
    else if(charge<0){
      hMassVSPhiMinus_->Fill(momentum1.phi(), momentum2.m(), weight);
    }
    else {
      LogDebug("Histograms") << "HMassVSPartProfile: wrong charge value = " << charge << std::endl;
      abort();
    }
  }

  virtual void Write() {
    hMassVSPt_->Write();
    hMassVSEta_->Write();
    hMassVSPhiPlus_->Write();
    hMassVSPhiMinus_->Write();
  }

  virtual void Clear() {
    hMassVSPt_->Clear();
    hMassVSEta_->Clear();    
    hMassVSPhiPlus_->Clear();
    hMassVSPhiMinus_->Clear();
  }

 protected:
  TProfile2D* hMassVSPt_;
  TProfile2D* hMassVSEta_;
  TProfile2D* hMassVSPhiPlus_; 
  TProfile2D* hMassVSPhiMinus_; 
};

//---------------------------------------------------------------------------------------
/// A set of histograms for resolution
class HResolutionVSPart : public Histograms
{
 public:
  HResolutionVSPart(TFile * outputFile, const TString & name, const double maxPt = 100,
                    const double & yMinEta = 0., const double & yMaxEta = 2.,
                    const double & yMinPt = 0., const double & yMaxPt = 2.,
                    const bool doProfiles = false) : Histograms(outputFile, name), doProfiles_(doProfiles)
  {
    // Kinematical variables

    // hReso           = new TH1F (name+"_Reso", "resolution", 4000, -1, 1);
    // hResoVSPtEta    = new TH2F (name+"_ResoVSPtEta", "resolution VS pt and #eta", 200, 0, 200, 60, -3, 3);
    // hResoVSPt       = new TH2F (name+"_ResoVSPt", "resolution VS pt", 200, 0, 200, 8000, -1, 1);
    // //hResoVSPt_prof  = new TProfile (name+"_ResoVSPt_prof", "resolution VS pt", 100, 0, 200, -1, 1);
    // hResoVSEta      = new TH2F (name+"_ResoVSEta", "resolution VS eta", 60, -3, 3, 8000, yMinEta, yMaxEta);
    // hResoVSTheta    = new TH2F (name+"_ResoVSTheta", "resolution VS theta", 30, 0, TMath::Pi(), 8000, -1, 1);
    // //hResoVSEta_prof = new TProfile (name+"_ResoVSEta_prof", "resolution VS eta", 10, -2.5, 2.5, -1, 1);
    // hResoVSPhiPlus  = new TH2F (name+"_ResoVSPhiPlus", "resolution VS phi mu+", 14, -3.2, 3.2, 8000, -1, 1);
    // hResoVSPhiMinus = new TH2F (name+"_ResoVSPhiMinus", "resolution VS phi mu-", 14, -3.2, 3.2, 8000, -1, 1);
    // //hResoVSPhi_prof = new TProfile (name+"_ResoVSPhi_prof", "resolution VS phi", 14, -3.2, 3.2, -1, 1);
    // hAbsReso        = new TH1F (name+"_AbsReso", "resolution", 100, 0, 1);
    // hAbsResoVSPt    = new TH2F (name+"_AbsResoVSPt", "Abs resolution VS pt", 200, 0, 500, 100, 0, 1);
    // hAbsResoVSEta   = new TH2F (name+"_AbsResoVSEta", "Abs resolution VS eta", 60, -3, 3, 100, 0, 1);
    // hAbsResoVSPhi   = new TH2F (name+"_AbsResoVSPhi", "Abs resolution VS phi", 14, -3.2, 3.2, 100, 0, 1);

    hReso_           = new TH1F( name+"_Reso", "resolution", 200, -1, 1 );
    hResoVSPtEta_    = new TH2F( name+"_ResoVSPtEta", "resolution VS pt and #eta", 200, 0, maxPt, 60, -3, 3 );
    hResoVSPt_       = new TH2F( name+"_ResoVSPt", "resolution VS pt", 200, 0, maxPt, 200, yMinPt, yMaxPt );
    hResoVSPt_Bar_   = new TH2F( name+"_ResoVSPt_Bar", "resolution VS pt Barrel", 200, 0, maxPt, 200, yMinPt, yMaxPt );
    hResoVSPt_Endc_17_  = new TH2F( name+"_ResoVSPt_Endc_1.7", "resolution VS pt Endcap (1.4<eta<1.7)", 200, 0, maxPt, 200, yMinPt, yMaxPt );
    hResoVSPt_Endc_20_  = new TH2F( name+"_ResoVSPt_Endc_2.0", "resolution VS pt Endcap (1.7<eta<2.0)", 200, 0, maxPt, 200, yMinPt, yMaxPt );
    hResoVSPt_Endc_24_  = new TH2F( name+"_ResoVSPt_Endc_2.4", "resolution VS pt Endcap (2.0<eta<2.4)", 200, 0, maxPt, 200, yMinPt, yMaxPt );
    hResoVSPt_Ovlap_ = new TH2F( name+"_ResoVSPt_Ovlap", "resolution VS pt overlap", 200, 0, maxPt, 200, yMinPt, yMaxPt );
    hResoVSEta_      = new TH2F( name+"_ResoVSEta", "resolution VS eta", 200, -3, 3, 200, yMinEta, yMaxEta );
    hResoVSTheta_    = new TH2F( name+"_ResoVSTheta", "resolution VS theta", 30, 0, TMath::Pi(), 200, yMinEta, yMaxEta );
    hResoVSPhiPlus_  = new TH2F( name+"_ResoVSPhiPlus", "resolution VS phi mu+",  16, -3.2, 3.2, 200, -1, 1 );
    hResoVSPhiMinus_ = new TH2F( name+"_ResoVSPhiMinus", "resolution VS phi mu-", 16, -3.2, 3.2, 200, -1, 1 );
    hAbsReso_        = new TH1F( name+"_AbsReso", "resolution", 100, 0, 1 );
    hAbsResoVSPt_    = new TH2F( name+"_AbsResoVSPt", "Abs resolution VS pt", 200, 0, maxPt, 100, 0, 1 );
    hAbsResoVSEta_   = new TH2F( name+"_AbsResoVSEta", "Abs resolution VS eta", 64, -3.2, 3.2, 100, 0, 1 );
    hAbsResoVSPhi_   = new TH2F( name+"_AbsResoVSPhi", "Abs resolution VS phi", 16, -3.2, 3.2, 100, 0, 1 );
    if( doProfiles_ ) {
      hResoVSPt_prof_    = new TProfile(name+"_ResoVSPt_prof", "resolution VS pt", 100, 0, maxPt, yMinPt, yMaxPt );
      hResoVSPt_Bar_prof_    = new TProfile(name+"_ResoVSPt_Bar_prof", "resolution VS pt Barrel", 100, 0, maxPt, yMinPt, yMaxPt );
      hResoVSPt_Endc_17_prof_   = new TProfile(name+"_ResoVSPt_Endc_1.7_prof", "resolution VS pt Endcap (1.4<eta<1.7)", 100, 0, maxPt, yMinPt, yMaxPt );
      hResoVSPt_Endc_20_prof_   = new TProfile(name+"_ResoVSPt_Endc_2.0_prof", "resolution VS pt Endcap (1.7<eta<2.0)", 100, 0, maxPt, yMinPt, yMaxPt );
      hResoVSPt_Endc_24_prof_   = new TProfile(name+"_ResoVSPt_Endc_2.4_prof", "resolution VS pt Endcap (2.0<eta<2.4)", 100, 0, maxPt, yMinPt, yMaxPt );
      hResoVSPt_Ovlap_prof_  = new TProfile(name+"_ResoVSPt_Ovlap_prof", "resolution VS pt Overlap", 100, 0, maxPt, yMinPt, yMaxPt );
      hResoVSEta_prof_       = new TProfile(name+"_ResoVSEta_prof", "resolution VS eta", 200, -3.0, 3.0, yMinEta, yMaxEta );
      hResoVSPhi_prof_       = new TProfile(name+"_ResoVSPhi_prof", "resolution VS phi", 16, -3.2, 3.2, -1, 1 );
    }
  }
  
  HResolutionVSPart(const TString & name, TFile* file) {
    name_=name;
    hReso_           = (TH1F *) file->Get(name+"_Reso");
    hResoVSPtEta_    = (TH2F *) file->Get(name+"_ResoVSPtEta");
    hResoVSPt_       = (TH2F *) file->Get(name+"_ResoVSPt");
    hResoVSPt_Bar_   = (TH2F *) file->Get(name+"_ResoVSPt");
    hResoVSPt_Endc_17_  = (TH2F *) file->Get(name+"_ResoVSPt");
    hResoVSPt_Endc_20_  = (TH2F *) file->Get(name+"_ResoVSPt");
    hResoVSPt_Endc_24_  = (TH2F *) file->Get(name+"_ResoVSPt");
    hResoVSPt_Ovlap_ = (TH2F *) file->Get(name+"_ResoVSPt");
    hResoVSEta_      = (TH2F *) file->Get(name+"_ResoVSEta");
    hResoVSTheta_    = (TH2F *) file->Get(name+"_ResoVSTheta");
    hResoVSPhiPlus_  = (TH2F *) file->Get(name+"_ResoVSPhiPlus");
    hResoVSPhiMinus_ = (TH2F *) file->Get(name+"_ResoVSPhiMinus");
    hAbsReso_        = (TH1F *) file->Get(name+"_AbsReso");
    hAbsResoVSPt_    = (TH2F *) file->Get(name+"_AbsResoVSPt");
    hAbsResoVSEta_   = (TH2F *) file->Get(name+"_AbsResoVSEta");
    hAbsResoVSPhi_   = (TH2F *) file->Get(name+"_AbsResoVSPhi");
    if( doProfiles_ ) {
      hResoVSPt_prof_  = (TProfile *) file->Get(name+"_ResoVSPt_prof");
      hResoVSPt_Bar_prof_ = (TProfile *) file->Get(name+"_ResoVSPt_prof"); 
      hResoVSPt_Endc_17_prof_ = (TProfile *) file->Get(name+"_ResoVSPt_1.7_prof"); 
      hResoVSPt_Endc_20_prof_ = (TProfile *) file->Get(name+"_ResoVSPt_2.0_prof"); 
      hResoVSPt_Endc_24_prof_ = (TProfile *) file->Get(name+"_ResoVSPt_2.4_prof"); 
      hResoVSPt_Ovlap_prof_= (TProfile *) file->Get(name+"_ResoVSPt_prof"); 
      hResoVSEta_prof_ = (TProfile *) file->Get(name+"_ResoVSEta_prof");
      hResoVSPhi_prof_ = (TProfile *) file->Get(name+"_ResoVSPhi_prof");
    }
  }

  ~HResolutionVSPart() {
    delete hReso_;
    delete hResoVSPtEta_;
    delete hResoVSPt_;
    delete hResoVSPt_Bar_;
    delete hResoVSPt_Endc_17_;
    delete hResoVSPt_Endc_20_;
    delete hResoVSPt_Endc_24_;
    delete hResoVSPt_Ovlap_;
    delete hResoVSEta_;
    delete hResoVSTheta_;
    delete hResoVSPhiMinus_;
    delete hResoVSPhiPlus_;
    delete hAbsReso_;
    delete hAbsResoVSPt_;
    delete hAbsResoVSEta_;
    delete hAbsResoVSPhi_;
    if( doProfiles_ ) {
      delete hResoVSPt_prof_;
      delete hResoVSPt_Bar_prof_;  
      delete hResoVSPt_Endc_17_prof_; 
      delete hResoVSPt_Endc_20_prof_; 
      delete hResoVSPt_Endc_24_prof_; 
      delete hResoVSPt_Ovlap_prof_;
      delete hResoVSEta_prof_;
      delete hResoVSPhi_prof_;
    }
  }

  virtual void Fill(const reco::Particle::LorentzVector & p4, const double & resValue, const int charge) { 
    double pt = p4.Pt();
    double eta = p4.Eta();
    hReso_->Fill(resValue);
    hResoVSPtEta_->Fill(pt, eta,resValue); 
    hResoVSPt_->Fill(pt,resValue); 
    if(fabs(eta)<=0.9)
      hResoVSPt_Bar_->Fill(pt,resValue);
    else if(fabs(eta)>0.9 && fabs(eta)<=1.4)
      hResoVSPt_Ovlap_->Fill(pt,resValue);
    else if(fabs(eta)>1.4 && fabs(eta)<=1.7)
      hResoVSPt_Endc_17_->Fill(pt,resValue);
    else if(fabs(eta)>1.7 && fabs(eta)<=2.0)
      hResoVSPt_Endc_20_->Fill(pt,resValue);
    else
      hResoVSPt_Endc_24_->Fill(pt,resValue);
     
    hResoVSEta_->Fill(eta,resValue);
    hResoVSTheta_->Fill(p4.Theta(),resValue);
    if(charge>0)
      hResoVSPhiPlus_->Fill(p4.Phi(),resValue); 
    else if(charge<0)
      hResoVSPhiMinus_->Fill(p4.Phi(),resValue); 
    hAbsReso_->Fill(fabs(resValue)); 
    hAbsResoVSPt_->Fill(pt,fabs(resValue)); 
    hAbsResoVSEta_->Fill(eta,fabs(resValue)); 
    hAbsResoVSPhi_->Fill(p4.Phi(),fabs(resValue));     
    if( doProfiles_ ) {
      hResoVSPt_prof_->Fill(p4.Pt(),resValue); 
      if(fabs(eta)<=0.9)
	hResoVSPt_Bar_prof_->Fill(p4.Pt(),resValue); 
      else if(fabs(eta)>0.9 && fabs(eta)<=1.4)
	hResoVSPt_Ovlap_prof_->Fill(pt,resValue);
      else if(fabs(eta)>1.4 && fabs(eta)<=1.7)
	hResoVSPt_Endc_17_prof_->Fill(pt,resValue);
      else if(fabs(eta)>1.7 && fabs(eta)<=2.0)
	hResoVSPt_Endc_20_prof_->Fill(pt,resValue);
      else
	hResoVSPt_Endc_24_prof_->Fill(pt,resValue);
      
      hResoVSEta_prof_->Fill(p4.Eta(),resValue); 
      hResoVSPhi_prof_->Fill(p4.Phi(),resValue); 
    }
  }

  virtual void Write() {
    if(histoDir_ != 0) histoDir_->cd();

    hReso_->Write();
    hResoVSPtEta_->Write();
    hResoVSPt_->Write();
    hResoVSPt_Bar_->Write();
    hResoVSPt_Endc_17_->Write();
    hResoVSPt_Endc_20_->Write();
    hResoVSPt_Endc_24_->Write();
    hResoVSPt_Ovlap_->Write();
    hResoVSEta_->Write();
    hResoVSTheta_->Write();
    hResoVSPhiMinus_->Write();
    hResoVSPhiPlus_->Write();
    hAbsReso_->Write();
    hAbsResoVSPt_->Write();
    hAbsResoVSEta_->Write();
    hAbsResoVSPhi_->Write();
    if( doProfiles_ ) {
      hResoVSPt_prof_->Write();
      hResoVSPt_Bar_prof_->Write();
      hResoVSPt_Endc_17_prof_->Write();
      hResoVSPt_Endc_20_prof_->Write();
      hResoVSPt_Endc_24_prof_->Write();
      hResoVSPt_Ovlap_prof_->Write();
      hResoVSEta_prof_->Write();
      hResoVSPhi_prof_->Write();
    }
  }
  
  virtual void Clear() {
    hReso_->Clear();
    hResoVSPtEta_->Clear();
    hResoVSPt_->Clear();
    hResoVSPt_Bar_->Clear();
    hResoVSPt_Endc_17_->Clear();
    hResoVSPt_Endc_20_->Clear();
    hResoVSPt_Endc_24_->Clear();
    hResoVSPt_Ovlap_->Clear();
    hResoVSEta_->Clear();
    hResoVSTheta_->Clear();
    hResoVSPhiPlus_->Clear();
    hResoVSPhiMinus_->Clear();
    hAbsReso_->Clear();
    hAbsResoVSPt_->Clear();
    hAbsResoVSEta_->Clear();
    hAbsResoVSPhi_->Clear();
    if( doProfiles_ ) {
      hResoVSPt_prof_->Clear();
      hResoVSPt_Bar_prof_->Clear();
      hResoVSPt_Endc_17_prof_->Clear();
      hResoVSPt_Endc_20_prof_->Clear();
      hResoVSPt_Endc_24_prof_->Clear();
      hResoVSPt_Ovlap_prof_->Clear();
      hResoVSEta_prof_->Clear();
      hResoVSPhi_prof_->Clear();
    }
  }

 public:
  TH1F* hReso_;
  TH2F* hResoVSPtEta_;
  TH2F* hResoVSPt_;
  TH2F* hResoVSPt_Bar_;
  TH2F* hResoVSPt_Endc_17_;
  TH2F* hResoVSPt_Endc_20_;
  TH2F* hResoVSPt_Endc_24_;
  TH2F* hResoVSPt_Ovlap_;
  TProfile* hResoVSPt_prof_;
  TProfile* hResoVSPt_Bar_prof_;
  TProfile* hResoVSPt_Endc_17_prof_;
  TProfile* hResoVSPt_Endc_20_prof_;
  TProfile* hResoVSPt_Endc_24_prof_;
  TProfile* hResoVSPt_Ovlap_prof_;
  TH2F* hResoVSEta_;
  TH2F* hResoVSTheta_;
  TProfile* hResoVSEta_prof_;
  TH2F* hResoVSPhiMinus_;
  TH2F* hResoVSPhiPlus_;
  TProfile* hResoVSPhi_prof_;
  TH1F* hAbsReso_;
  TH2F* hAbsResoVSPt_;
  TH2F* hAbsResoVSEta_;
  TH2F* hAbsResoVSPhi_;
  bool doProfiles_;
};

// -------------------------------------------------------------
// A set of histograms of likelihood value versus muon variables
// -------------------------------------------------------------
class HLikelihoodVSPart : public Histograms
{
 public:
  HLikelihoodVSPart(const TString & name){
    name_ = name;

    // Kinematical variables
    // ---------------------
    hLikeVSPt_       = new TH2F (name+"_LikelihoodVSPt", "likelihood vs muon transverse momentum", 100, 0., 100., 100, -100., 100.);
    hLikeVSEta_      = new TH2F (name+"_LikelihoodVSEta", "likelihood vs muon pseudorapidity", 100, -4.,4., 100, -100., 100.);
    hLikeVSPhi_      = new TH2F (name+"_LikelihoodVSPhi", "likelihood vs muon phi angle", 100, -3.2, 3.2, 100, -100., 100.);
    hLikeVSPt_prof_  = new TProfile (name+"_LikelihoodVSPt_prof", "likelihood vs muon transverse momentum", 40, 0., 100., -1000., 1000. );
    hLikeVSEta_prof_ = new TProfile (name+"_LikelihoodVSEta_prof", "likelihood vs muon pseudorapidity", 40, -4.,4., -1000., 1000. );
    hLikeVSPhi_prof_ = new TProfile (name+"_LikelihoodVSPhi_prof", "likelihood vs muon phi angle", 40, -3.2, 3.2, -1000., 1000.);
   }
  
  HLikelihoodVSPart(const TString & name, TFile* file){
    name_ = name;
    hLikeVSPt_       = (TH2F *) file->Get(name+"_LikelihoodVSPt");
    hLikeVSEta_      = (TH2F *) file->Get(name+"_LikelihoodVSEta");
    hLikeVSPhi_      = (TH2F *) file->Get(name+"_LikelihoodVSPhi");
    hLikeVSPt_prof_  = (TProfile *) file->Get(name+"_LikelihoodVSPt_prof");
    hLikeVSEta_prof_ = (TProfile *) file->Get(name+"_LikelihoodVSEta_prof");
    hLikeVSPhi_prof_ = (TProfile *) file->Get(name+"_LikelihoodVSPhi_prof");
  }

  ~HLikelihoodVSPart(){
    delete hLikeVSPt_;
    delete hLikeVSEta_;
    delete hLikeVSPhi_;
    delete hLikeVSPt_prof_;
    delete hLikeVSEta_prof_;
    delete hLikeVSPhi_prof_;
  } 

  virtual void Fill(const reco::Particle::LorentzVector & p4, const double & likeValue) {
    Fill(CLHEP::HepLorentzVector(p4.x(),p4.y(),p4.z(),p4.t()), likeValue);
  }
  
   virtual void Fill(CLHEP::HepLorentzVector momentum, double likeValue) { 
     hLikeVSPt_->Fill(momentum.perp(),likeValue); 
     hLikeVSEta_->Fill(momentum.eta(),likeValue); 
     hLikeVSPhi_->Fill(momentum.phi(),likeValue); 
     hLikeVSPt_prof_->Fill(momentum.perp(),likeValue); 
     hLikeVSEta_prof_->Fill(momentum.eta(),likeValue); 
     hLikeVSPhi_prof_->Fill(momentum.phi(),likeValue); 
   } 
  
  virtual void Write() {
    hLikeVSPt_->Write();
    hLikeVSEta_->Write();    
    hLikeVSPhi_->Write();
    hLikeVSPt_prof_->Write();
    hLikeVSEta_prof_->Write();    
    hLikeVSPhi_prof_->Write();
  }
  
  virtual void Clear() {
    hLikeVSPt_->Reset("ICE");
    hLikeVSEta_->Reset("ICE");
    hLikeVSPhi_->Reset("ICE");
    hLikeVSPt_prof_->Reset("ICE");
    hLikeVSEta_prof_->Reset("ICE");    
    hLikeVSPhi_prof_->Reset("ICE");
  }
  
 public:
  TH2F* hLikeVSPt_;
  TH2F* hLikeVSEta_;
  TH2F* hLikeVSPhi_;
  TProfile* hLikeVSPt_prof_;
  TProfile* hLikeVSEta_prof_;
  TProfile* hLikeVSPhi_prof_;
};

/**
 * This histogram class fills a TProfile with the resolution evaluated from the resolution
 * functions for single muon quantities. The resolution functions are used by MuScleFit to
 * evaluate the mass resolution, which is the value seen by minuit and through it,
 * corrections are evaluated. <br>
 * In the end we will compare the histograms filled by this class (from the resolution
 * function, reflecting the parameters changes done by minuit) with those filled comparing
 * recoMuons with genMuons (the real resolutions).
 */
class HFunctionResolution : public Histograms
{
 public:
  HFunctionResolution(TFile * outputFile, const TString & name, const double & ptMax = 100, const int totBinsY = 300) : Histograms(outputFile, name) {
    name_ = name;
    totBinsX_ = 300;
    totBinsY_ = totBinsY;
    xMin_ = 0.;
    yMin_ = -3.0;
    double xMax = ptMax;
    double yMax = 3.0;
    deltaX_ = xMax - xMin_;
    deltaY_ = yMax - yMin_;
    hReso_        = new TH1F( name+"_Reso", "resolution", 1000, 0, 1 );
    hResoVSPtEta_ = new TH2F( name+"_ResoVSPtEta", "resolution vs pt and #eta", totBinsX_, xMin_, xMax, totBinsY_, yMin_, yMax );
    // Create and initialize the resolution arrays
    resoVsPtEta_  = new double*[totBinsX_];
    resoCount_    = new int*[totBinsX_];
    for( int i=0; i<totBinsX_; ++i ) {
      resoVsPtEta_[i] = new double[totBinsY_];
      resoCount_[i]   = new int[totBinsY_];
      for( int j=0; j<totBinsY_; ++j ) {
        resoVsPtEta_[i][j] = 0;
        resoCount_[i][j] = 0;
      }
    }
    hResoVSPt_prof_       = new TProfile( name+"_ResoVSPt_prof", "resolution VS pt", totBinsX_, xMin_, xMax, yMin_, yMax);
    hResoVSPt_Bar_prof_   = new TProfile( name+"_ResoVSPt_Bar_prof", "resolution VS pt Barrel", totBinsX_, xMin_, xMax, yMin_, yMax);
    hResoVSPt_Endc_17_prof_  = new TProfile( name+"_ResoVSPt_Endc_1.7_prof", "resolution VS pt Endcap (1.4<eta<1.7)", totBinsX_, xMin_, xMax, yMin_, yMax);
    hResoVSPt_Endc_20_prof_  = new TProfile( name+"_ResoVSPt_Endc_2.0_prof", "resolution VS pt Endcap (1.7<eta<2.0)", totBinsX_, xMin_, xMax, yMin_, yMax);
    hResoVSPt_Endc_24_prof_  = new TProfile( name+"_ResoVSPt_Endc_2.4_prof", "resolution VS pt Endcap (2.0<eta<2.4)", totBinsX_, xMin_, xMax, yMin_, yMax);
    hResoVSPt_Ovlap_prof_ = new TProfile( name+"_ResoVSPt_Ovlap_prof", "resolution VS pt Overlap", totBinsX_, xMin_, xMax, yMin_, yMax);
    hResoVSEta_prof_      = new TProfile( name+"_ResoVSEta_prof", "resolution VS eta", totBinsY_, yMin_, yMax, 0, 1);
    //hResoVSTheta_prof_    = new TProfile( name+"_ResoVSTheta_prof", "resolution VS theta", 30, 0, TMath::Pi(), 0, 1);
    hResoVSPhiPlus_prof_  = new TProfile( name+"_ResoVSPhiPlus_prof", "resolution VS phi mu+", 16, -3.2, 3.2, 0, 1);
    hResoVSPhiMinus_prof_ = new TProfile( name+"_ResoVSPhiMinus_prof", "resolution VS phi mu-", 16, -3.2, 3.2, 0, 1);
    hResoVSPhi_prof_      = new TProfile( name+"_ResoVSPhi_prof", "resolution VS phi", 16, -3.2, 3.2, -1, 1);
  }
  ~HFunctionResolution() {
    delete hReso_;
    delete hResoVSPtEta_;
    // Free the resolution arrays
    for( int i=0; i<totBinsX_; ++i ) {
      delete[] resoVsPtEta_[i];
      delete[] resoCount_[i];
    }
    delete[] resoVsPtEta_;
    delete[] resoCount_;
    // Free the profile histograms
    delete hResoVSPt_prof_;
    delete hResoVSPt_Bar_prof_;
    delete hResoVSPt_Endc_17_prof_;
    delete hResoVSPt_Endc_20_prof_;
    delete hResoVSPt_Endc_24_prof_;
    delete hResoVSPt_Ovlap_prof_;
    delete hResoVSEta_prof_;
    //delete hResoVSTheta_prof_;
    delete hResoVSPhiPlus_prof_;
    delete hResoVSPhiMinus_prof_;
    delete hResoVSPhi_prof_;
  }
  virtual void Fill(const reco::Particle::LorentzVector & p4, const double & resValue, const int charge) {
    if( resValue != resValue ) return;
    hReso_->Fill(resValue);

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
      hResoVSPt_prof_->Fill(p4.Pt(),resValue);
      if(fabs(p4.Eta())<=0.9)
	hResoVSPt_Bar_prof_->Fill(p4.Pt(),resValue);
      else if(fabs(p4.Eta())>0.9 && fabs(p4.Eta())<=1.4 )
	hResoVSPt_Ovlap_prof_->Fill(p4.Pt(),resValue);
      else if(fabs(p4.Eta())>1.4 && fabs(p4.Eta())<=1.7 )
	hResoVSPt_Endc_17_prof_->Fill(p4.Pt(),resValue);
      else if(fabs(p4.Eta())>1.7 && fabs(p4.Eta())<=2.0 )
	hResoVSPt_Endc_20_prof_->Fill(p4.Pt(),resValue);
      else
      	hResoVSPt_Endc_24_prof_->Fill(p4.Pt(),resValue);
      hResoVSEta_prof_->Fill(p4.Eta(),resValue);
      //hResoVSTheta_prof_->Fill(p4.Theta(),resValue);
      if(charge>0)
        hResoVSPhiPlus_prof_->Fill(p4.Phi(),resValue);
      else if(charge<0)
        hResoVSPhiMinus_prof_->Fill(p4.Phi(),resValue);
      hResoVSPhi_prof_->Fill(p4.Phi(),resValue);
    }
  }

  virtual void Write() {
    if(histoDir_ != 0) histoDir_->cd();

    hReso_->Write();

    for( int i=0; i<totBinsX_; ++i ) {
      for( int j=0; j<totBinsY_; ++j ) {
        int N = resoCount_[i][j];
        // Fill with the mean value
        if( N != 0 ) hResoVSPtEta_->SetBinContent( i+1, j+1, resoVsPtEta_[i][j]/N );
        else hResoVSPtEta_->SetBinContent( i+1, j+1, 0 );
      }
    }
    hResoVSPtEta_->Write();

    hResoVSPt_prof_->Write();
    hResoVSPt_Bar_prof_->Write();
    hResoVSPt_Endc_17_prof_->Write();
    hResoVSPt_Endc_20_prof_->Write();
    hResoVSPt_Endc_24_prof_->Write();
    hResoVSPt_Ovlap_prof_->Write();
    hResoVSEta_prof_->Write();
    //hResoVSTheta_prof_->Write();
    hResoVSPhiMinus_prof_->Write();
    hResoVSPhiPlus_prof_->Write();
    hResoVSPhi_prof_->Write();

    TCanvas canvas(TString(hResoVSPtEta_->GetName())+"_canvas", TString(hResoVSPtEta_->GetTitle())+" canvas", 1000, 800);
    canvas.Divide(2);
    canvas.cd(1);
    hResoVSPtEta_->Draw("lego");
    canvas.cd(2);
    hResoVSPtEta_->Draw("surf5");
    canvas.Write();
    hResoVSPtEta_->Write();

    outputFile_->cd();
  }
  
  virtual void Clear() {
    hReso_->Clear();
    hResoVSPtEta_->Clear();
    hResoVSPt_prof_->Clear();
    hResoVSPt_Bar_prof_->Clear();
    hResoVSPt_Endc_17_prof_->Clear();
    hResoVSPt_Endc_20_prof_->Clear();
    hResoVSPt_Endc_24_prof_->Clear();
    hResoVSPt_Ovlap_prof_->Clear();
    hResoVSEta_prof_->Clear();
    //hResoVSTheta_prof_->Clear();
    hResoVSPhiPlus_prof_->Clear();
    hResoVSPhiMinus_prof_->Clear();
    hResoVSPhi_prof_->Clear();
  }

 protected:
  int getXindex(const double & x) const {
    return int((x-xMin_)/deltaX_*totBinsX_);
  }
  int getYindex(const double & y) const {
    return int((y-yMin_)/deltaY_*totBinsY_);
  }
  TH1F* hReso_;
  TH2F* hResoVSPtEta_;
  double ** resoVsPtEta_;
  int ** resoCount_;
  TProfile* hResoVSPt_prof_;
  TProfile* hResoVSPt_Bar_prof_;
  TProfile* hResoVSPt_Endc_17_prof_;
  TProfile* hResoVSPt_Endc_20_prof_;
  TProfile* hResoVSPt_Endc_24_prof_;
  TProfile* hResoVSPt_Ovlap_prof_;
  TProfile* hResoVSEta_prof_;
  //TProfile* hResoVSTheta_prof_;
  TProfile* hResoVSPhiMinus_prof_;
  TProfile* hResoVSPhiPlus_prof_;
  TProfile* hResoVSPhi_prof_;
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
        std::stringstream namei;
        std::stringstream namej;
        namei << i;
        namej << j;
        histoVarianceCheck_[i][j] = new TH1D(name+"_"+namei.str()+"_"+namej.str(), name, 100, 0., 1.);
      }
    }
  }
  ~HFunctionResolutionVarianceCheck() {
    for( int i=0; i<totBinsX_; ++i ) {
      for( int j=0; j<totBinsY_; ++j ) {
        delete histoVarianceCheck_[i][j];
      }
      delete[] histoVarianceCheck_;
    }
    delete[] histoVarianceCheck_;
  }
  virtual void Fill(const reco::Particle::LorentzVector & p4, const double & resValue, const int charge) {
    if( resValue != resValue ) return;
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
  ~HResolution() {
    delete diffHisto_;
    delete histo2D_;
    delete resoHisto_;
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
    // std::cout << "totBins = " << totBins << std::endl;
    for( unsigned int iBin=1; iBin<=totBins; ++iBin ) {
      // std::cout << "iBin = " << iBin << ", " << diffHisto_->GetBinError(iBin)*sqrt(diffHisto_->GetBinEntries(iBin)) << std::endl;
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
 * The Fill method needs the two input variables. </br>
 * In the end the covariance method computes the covariance as:
 * cov(x,y) = Sum_i(x_i*y_i)/N - x_mean*y_mean. </br>
 * Of course passing the same variable for x and y gives the variance of that variable.
 */
class Covariance
{
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
      // std::cout << "meanX*meanY = "<<meanX<<"*"<<meanY<< " = " << meanX*meanY << std::endl;
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
class HCovarianceVSxy : public Histograms
{
 public:
  HCovarianceVSxy( const TString & name, const TString & title,
                   const int totBinsX, const double & xMin, const double & xMax,
                   const int totBinsY, const double & yMin, const double & yMax,
                   TDirectory * dir = 0, bool varianceCheck = false ) :
    totBinsX_(totBinsX), totBinsY_(totBinsY),
    xMin_(xMin), deltaX_(xMax-xMin), yMin_(yMin), deltaY_(yMax-yMin),
    readMode_(false),
    varianceCheck_(varianceCheck)
  {
    name_ = name;
    histoDir_ = dir;
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
          std::stringstream namei;
          std::stringstream namej;
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
    histoDir_ = (TDirectory*)(inputFile->Get(dirName.Data()));
    if( histoDir_ == 0 ) {
      std::cout << "Error: directory not found" << std::endl;
      exit(0);
    }
    histoCovariance_ = (TH2D*)(histoDir_->Get(name));
    totBinsX_ = histoCovariance_->GetNbinsX();
    xMin_ = histoCovariance_->GetXaxis()->GetBinLowEdge(1);
    deltaX_ = histoCovariance_->GetXaxis()->GetBinUpEdge(totBinsX_) - xMin_;
    totBinsY_ = histoCovariance_->GetNbinsY();
    yMin_ = histoCovariance_->GetYaxis()->GetBinLowEdge(1);
    deltaY_ = histoCovariance_->GetYaxis()->GetBinUpEdge(totBinsY_) - yMin_;
  }

  ~HCovarianceVSxy() {
    delete histoCovariance_;
    // Free covariances
    for(int i=0; i<totBinsX_; ++i) {
      delete[] covariances_[i];
    }
    delete[] covariances_;
    // Free variance check histograms
    if( varianceCheck_ ) {
      for( int i=0; i<totBinsX_; ++i ) {
        for( int j=0; j<totBinsY_; ++j ) {
          delete histoVarianceCheck_[i][j];
        }
        delete[] histoVarianceCheck_[i];
      }
      delete[] histoVarianceCheck_;
    }
  }

  /**
   * x and y should be the variables VS which we are computing the covariance (pt and eta)
   * a and b should be the variables OF which we are computing the covariance </br>
   */
  virtual void Fill( const double & x, const double & y, const double & a, const double & b ) {
    // Need to convert the (x,y) values to the array indeces
    int xIndex = getXindex(x);
    int yIndex = getYindex(y);
    // Only fill values if they are in the selected range
    if ( 0 <= xIndex && xIndex < totBinsX_ && 0 <= yIndex && yIndex < totBinsY_ ) {
      // if( TString(histoCovariance_->GetName()).Contains("CovarianceCotgTheta_Covariance") )
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
      std::cout << "writing: " << histoCovariance_->GetName() << std::endl;
      for( int xBin=0; xBin<totBinsX_; ++xBin ) {
        for( int yBin=0; yBin<totBinsY_; ++yBin ) {
          double covariance = covariances_[xBin][yBin].covariance();
          // Histogram bins start from 1
          // std::cout << "covariance["<<xBin<<"]["<<yBin<<"] with N = "<<covariances_[xBin][yBin].getN()<<" is: " << covariance << std::endl;
          histoCovariance_->SetBinContent(xBin+1, yBin+1, covariance);
        }
      }
      if( histoDir_ != 0 ) histoDir_->cd();
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
        if ( histoDir_ != 0 ) {
          histoDir_->cd();
          if( binsDir == 0 ) binsDir = histoDir_->mkdir(name_+"Bins");
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
    if( varianceCheck_ ) {
      for( int i=0; i<totBinsX_; ++i ) {
        for( int j=0; j<totBinsY_; ++j ) {
          histoVarianceCheck_[i][j]->Clear();
        }
      }
    }
  }
 protected:
  int getXindex(const double & x) const {
    return int((x-xMin_)/deltaX_*totBinsX_);
  }
  int getYindex(const double & y) const {
    return int((y-yMin_)/deltaY_*totBinsY_);
  }
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

  ~HCovarianceVSParts() {
    for (std::map<TString, HCovarianceVSxy*>::const_iterator histo=mapHisto_.begin(); 
         histo!=mapHisto_.end(); histo++) {
      delete (*histo).second;
    }
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
      for (std::map<TString, HCovarianceVSxy*>::const_iterator histo=mapHisto_.begin(); 
           histo!=mapHisto_.end(); histo++) {
        (*histo).second->Write();
      }
    }
  }
  virtual void Clear() {
    for (std::map<TString, HCovarianceVSxy*>::const_iterator histo=mapHisto_.begin(); 
         histo!=mapHisto_.end(); histo++) {
      (*histo).second->Clear();
    }
  }
 protected:
  std::map<TString, HCovarianceVSxy*> mapHisto_;
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
    for (std::map<TString, TH1*>::const_iterator histo=mapHisto_.begin(); 
         histo!=mapHisto_.end(); histo++) {
      delete (*histo).second;
    }
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

    if ( charge1 == charge2 ) std::cout << "Error: must get two opposite charge particles" << std::endl;

    double massRes = (recoMass - genMass)/genMass;

    reco::Particle::LorentzVector recoPair( recoP1 + recoP2 );
    double pairPt = recoPair.Pt();

    double recoPt[2]  = {recoP1.Pt(),  recoP2.Pt()};
    double recoEta[2] = {recoP1.Eta(), recoP2.Eta()};
    double recoPhi[2] = {recoP1.Phi(), recoP2.Phi()};

    // std::cout << "pairPt = " << pairPt << ", massRes = ("<<recoMass<<" - "<<genMass<<")/"<<genMass<<" = " << massRes
    //      << ", recoPt[0] = " << recoPt[0] << ", recoPt[1] = " << recoPt[1] << std::endl;

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
      mapHisto_[name_+"VSPt"+nameSuffix_[i]]->Fill(recoPt[index], massRes); // EM [index] or [i]???
      mapHisto_[name_+"VSEta"+nameSuffix_[i]]->Fill(recoEta[index], massRes);
      mapHisto_[name_+"VSPhi"+nameSuffix_[i]]->Fill(recoPhi[index], massRes);
    }
  } 

  virtual void Write() {
    histoDir_->cd();
    for (std::map<TString, TH1*>::const_iterator histo=mapHisto_.begin(); 
         histo!=mapHisto_.end(); histo++) {
      (*histo).second->Write();
    }
    // Create the new dir and cd into it
    (histoDir_->mkdir("singleMuonsVSgen"))->cd();
    muMinus->Write();
    muPlus->Write();
  }
  
  virtual void Clear() {
    for (std::map<TString, TH1*>::const_iterator histo=mapHisto_.begin(); 
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
  std::map<TString, TH1*> mapHisto_;
  TString nameSuffix_[2];
  std::auto_ptr<HDelta> muMinus;
  std::auto_ptr<HDelta> muPlus;
};

#endif
