#include <vector>
#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>
#include "TF1.h"
#include "TH2F.h"
#include "TLegend.h"
#include "TProfile.h"
#include "TGraph.h"
#include "TGraphErrors.h"
#include "Math/SMatrix.h"
#include "Math/SVector.h"
#include "TCanvas.h"
#include <iostream>
#include <math.h>

typedef ROOT::Math::SMatrix<double,1,1,ROOT::Math::MatRepStd<double,1> > StdMatrix11;
typedef ROOT::Math::SMatrix<double,2,2,ROOT::Math::MatRepStd<double,2> > StdMatrix22;
typedef ROOT::Math::SMatrix<double,3,3,ROOT::Math::MatRepStd<double,3> > StdMatrix33;
typedef ROOT::Math::SMatrix<double,4,4,ROOT::Math::MatRepStd<double,4> > StdMatrix44;
typedef ROOT::Math::SMatrix<double,5,5,ROOT::Math::MatRepStd<double,5> > StdMatrix55;
typedef ROOT::Math::SMatrix<double,6,6,ROOT::Math::MatRepStd<double,6> > StdMatrix66;
typedef ROOT::Math::SMatrix<double,7,7,ROOT::Math::MatRepStd<double,7> > StdMatrix77;

TGraphErrors* gra0;
TGraphErrors* gra;
TGraphErrors* grb;
TGraphErrors* grc;
TGraphErrors* graEta0;
TGraphErrors* grbEta0;
TGraphErrors* graX;
TGraphErrors* grbX; 
TGraphErrors* grcX;


unsigned etaEH = 0;
double threshE0 = 3.7; 
//double threshE = 3.5;  // Barrel
//double threshH = 3.0;  // Barrel
double threshE = 3.5;  // End-caps
double threshH = 2.5;  // End-caps
//double threshE = 0.;
//double threshH = 0.;
// Barrel
//double etamin = 0.0;
//double etamax = 1.2;
/*
bool endcap = false;
double etamin = 0.0;
double etamax = 1.6;
double etamin_Fit;
double etamax_Fit;
double etamin_FitE = 0.0;
double etamax_FitE = 1.0;
double etamin_FitEta = 0.0;
double etamax_FitEta = 1.3;
*/
// Endcap
/* */
bool endcap = true;
double etamin = 1.5;
double etamax = 3.0;
double etamin_Fit;
double etamax_Fit;
double etamin_FitE = 1.6;
double etamax_FitE = 2.2;
double etamin_FitEta = 1.6;
double etamax_FitEta = 2.8;
/* */
//double etamin = 1.3;
//double etamax = 3.0;
// Endcap - HF border
//double etamin = 2.6;
//double etamax = 4.9;

using namespace std;

class NTuple {
public :
   TTree          *fChain;   //!pointer to the analyzed TTree or TChain
   Int_t           fCurrent; //!current Tree number in a TChain

   // Declaration of leaf types
   Float_t         true_;
   Float_t         p_;
   Float_t         ecal_;
   Float_t         hcal_;
   Float_t         eta_;
   Float_t         phi_;

   // List of branches
   TBranch        *b_true;   //!
   TBranch        *b_p;   //!
   TBranch        *b_ecal;   //!
   TBranch        *b_hcal;   //!
   TBranch        *b_eta;    //!
   TBranch        *b_phi;    //!

   NTuple(TTree *tree=0);
   virtual ~NTuple();
   virtual Int_t    Cut(Long64_t entry);
   virtual Int_t    GetEntry(Long64_t entry);
   virtual Long64_t LoadTree(Long64_t entry);
   virtual void     Init(TTree *tree);
   // virtual void     Loop();
   virtual Bool_t   Notify();
   virtual void     Show(Long64_t entry = -1);
};

NTuple::NTuple(TTree *tree)
{
// if parameter tree is not specified (or zero), connect the file
// used to generate this class and read the Tree.
   if (tree == 0) {
      TFile *f = (TFile*)gROOT->GetListOfFiles()->FindObject("myTree.root");
      if (!f) {
         f = new TFile("myTree.root");
      }
      tree = (TTree*)gDirectory->Get("ntuple");

   }
   Init(tree);
}

NTuple::~NTuple()
{
   if (!fChain) return;
   delete fChain->GetCurrentFile();
}

Int_t NTuple::GetEntry(Long64_t entry)
{
// Read contents of entry.
   if (!fChain) return 0;
   return fChain->GetEntry(entry);
}
Long64_t NTuple::LoadTree(Long64_t entry)
{
// Set the environment to read one entry
   if (!fChain) return -5;
   Long64_t centry = fChain->LoadTree(entry);
   if (centry < 0) return centry;
   if (!fChain->InheritsFrom(TChain::Class()))  return centry;
   TChain *chain = (TChain*)fChain;
   if (chain->GetTreeNumber() != fCurrent) {
      fCurrent = chain->GetTreeNumber();
      Notify();
   }
   return centry;
}

void NTuple::Init(TTree *tree)
{
   // The Init() function is called when the selector needs to initialize
   // a new tree or chain. Typically here the branch addresses and branch
   // pointers of the tree will be set.
   // It is normally not necessary to make changes to the generated
   // code, but the routine can be extended by the user if needed.
   // Init() will be called many times when running on PROOF
   // (once per file to be processed).

   // Set branch addresses and branch pointers
   if (!tree) return;
   fChain = tree;
   fCurrent = -1;
   fChain->SetMakeClass(1);

   fChain->SetBranchAddress("true", &true_, &b_true);
   fChain->SetBranchAddress("p", &p_, &b_p);
   fChain->SetBranchAddress("ecal", &ecal_, &b_ecal);
   fChain->SetBranchAddress("hcal", &hcal_, &b_hcal);
   fChain->SetBranchAddress("eta", &eta_, &b_eta);
   fChain->SetBranchAddress("phi", &phi_, &b_phi);
   Notify();
}

Bool_t NTuple::Notify()
{
   // The Notify() function is called when a new file is opened. This
   // can be either for a new TTree in a TChain or when when a new TTree
   // is started when using PROOF. It is normally not necessary to make changes
   // to the generated code, but the routine can be extended by the
   // user if needed. The return value is currently not used.

   return kTRUE;
}

void NTuple::Show(Long64_t entry)
{
// Print contents of entry.
// If entry is not specified, print current entry
   if (!fChain) return;
   fChain->Show(entry);
}
Int_t NTuple::Cut(Long64_t entry)
{
// This function may be called from Loop.
// returns  1 if entry is accepted.
// returns -1 otherwise.
   if ( entry >= 0 ) return 1;
   return 0;
}

class Fit {

public:

  Fit(double minE, double maxE) 
    : 
    minE(minE), 
    maxE(maxE),
    Denom(0.),
    NumA(0.),
    NumB(0.),
    ACoeff(0.),
    BCoeff(0.),
    CCoeff(0.),
    DCoeff(0.),
    SigmaA(0.),
    SigmaB(0.),
    SigmaC(0.),
    SigmaD(0.),
    RhoAB(0.)
  { }

  void clear() { 
    Eta.clear();
    S.clear();
    E.clear();
    H.clear();
    T.clear();
    XE.clear();
    XH.clear();
  }

  bool fill(double e, double h, double t, double eta, unsigned input = 0) { 
    bool isFilled = false;
    if ( t > minE && t < maxE )  { 
      double thresh = 0.;
      if ( input == 0 ) thresh = e != 0. ? ( h != 0. ? threshE : threshE0 ) : threshH;
      
      double sigma = endcap ? 
	sqrt(0.04*0.04 + 1.80*1.80*(std::max(e+h,1.))) :
	sqrt(0.08*0.08 + 1.04*1.04*(std::max(e+h,1.)));

      if ( (e+h-t+thresh) < -5.*sigma ) return isFilled;

      if ( eta < etamax_Fit && eta > etamin_Fit ) { 

	if ( endcap ) 
	  S.push_back(sqrt(0.04*0.04 + 1.8*1.8*(std::max(e+h,1.))));  /// End-cap stuff
	else
	  S.push_back(sqrt(0.08*0.08 + 1.04*1.04*(std::max(e+h,1.))));  /// Barrel stuff

	E.push_back(e/S.back());
	H.push_back(h/S.back());
	T.push_back((t-thresh)/S.back());
	Eta.push_back(eta-etamin);
	if ( e != 0. && h != 0. ) { 
	  XE.push_back((e+h)*(e+h)/(e*e));
	  XH.push_back((e+h)*(e+h)/(h*h));
	} else { 
	  XE.push_back(0.);
	  XH.push_back(0.);
	}
      }
      isFilled = true;
    }
    return isFilled;
  }

  bool eMatrices() { 

    bool success = true;

    E2(0,0) = 0.;
    E2(0,1) = 0.;
    E2(1,0) = 0.;
    E2(1,1) = 0.;
    TE2(0)  = 0.;
    TE2(1)  = 0.;
    E1(0,0) = 0.;
    TE1(0)  = 0.;
    H1(0,0) = 0.;
    TH1(0)  = 0.;
    
    for (unsigned i=0; i<T.size(); ++i ) { 

      if ( E[i] != 0. ) { 
	// (a*E +b*sqrt(E)+c*H+d*sqrt(H) fit
	if ( H[i] != 0. ) { 

	  E2(0,0) += 2.*E[i]*E[i];
	  E2(0,1) += 2.*E[i]*H[i];
	  E2(1,0) += 2.*E[i]*H[i];
	  E2(1,1) += 2.*H[i]*H[i];
	  TE2(0) += 2.*T[i]*E[i];
	  TE2(1) += 2.*T[i]*H[i];

	} else {

	  E1(0,0) += 2*E[i]*E[i];
	  TE1(0) += 2*T[i]*E[i];

	}

      } else {  

	H1(0,0) += 2*H[i]*H[i];
	TH1(0) += 2*T[i]*H[i];

      }

    }

    success = E2.Invert();

    if ( H1(0,0) != 0. ) success = success && H1.Invert();
    if ( E1(0,0) != 0. ) success = success && E1.Invert();

    H1Coeffs = H1 * TH1;
    //T1Coeffs = H1 * XH1;
    E2Coeffs = E2 * TE2;
    E1Coeffs = E1 * TE1;
    //T2Coeffs = E2 * XE2;

    return success;

  }

  bool etaMatrices() { 

    bool success = true;
    Eta3(0,0) = 0.;
    Eta3(0,1) = 0.;
    Eta3(1,0) = 0.;
    Eta3(1,1) = 0.;
    TEta3(0) = 0.;
    TEta3(1) = 0.;
    
    Eta2(0,0) = 0.;
    Eta2(0,1) = 0.;
    Eta2(1,0) = 0.;
    Eta2(1,1) = 0.;
    TEta2(0) = 0.;
    TEta2(1) = 0.;

    bool debug = T.size() == 5323;
    for (unsigned i=0; i<T.size(); ++i ) {
      if ( H[i] == 0. ) continue;

      /*
      // (1+a1*eta+a2*eta**2) E + (1+b1*eta+b2*eta**2) H fit
      Eta6(0,0) += 2.*E[i]*E[i];
      Eta6(0,1) += 2.*Eta[i]*E[i]*E[i];
      Eta6(0,2) += 2.*Eta[i]*Eta[i]*E[i]*E[i];
      Eta6(0,3) += 2.*E[i]*H[i];
      Eta6(0,4) += 2.*Eta[i]*E[i]*H[i];
      Eta6(0,5) += 2.*Eta[i]*Eta[i]*E[i]*H[i];
      Eta6(1,0) += 2.*Eta[i]*E[i]*E[i];
      Eta6(1,1) += 2.*Eta[i]*Eta[i]*E[i]*E[i];
      Eta6(1,2) += 2.*Eta[i]*Eta[i]*Eta[i]*E[i]*E[i];
      Eta6(1,3) += 2.*Eta[i]*E[i]*H[i];
      Eta6(1,4) += 2.*Eta[i]*Eta[i]*E[i]*H[i];
      Eta6(1,5) += 2.*Eta[i]*Eta[i]*Eta[i]*E[i]*H[i];
      Eta6(2,0) += 2.*Eta[i]*Eta[i]*E[i]*E[i];
      Eta6(2,1) += 2.*Eta[i]*Eta[i]*Eta[i]*E[i]*E[i];
      Eta6(2,2) += 2.*Eta[i]*Eta[i]*Eta[i]*Eta[i]*E[i]*E[i];
      Eta6(2,3) += 2.*Eta[i]*Eta[i]*E[i]*H[i];
      Eta6(2,4) += 2.*Eta[i]*Eta[i]*Eta[i]*E[i]*H[i];
      Eta6(2,5) += 2.*Eta[i]*Eta[i]*Eta[i]*Eta[i]*E[i]*H[i];
      Eta6(3,0) += 2.*E[i]*H[i];
      Eta6(3,1) += 2.*Eta[i]*E[i]*H[i];
      Eta6(3,2) += 2.*Eta[i]*Eta[i]*E[i]*H[i];
      Eta6(3,3) += 2.*H[i]*H[i];
      Eta6(3,4) += 2.*Eta[i]*H[i]*H[i];
      Eta6(3,5) += 2.*Eta[i]*Eta[i]*H[i]*H[i];
      Eta6(4,0) += 2.*Eta[i]*E[i]*H[i];
      Eta6(4,1) += 2.*Eta[i]*Eta[i]*E[i]*H[i];
      Eta6(4,2) += 2.*Eta[i]*Eta[i]*Eta[i]*E[i]*H[i];
      Eta6(4,3) += 2.*Eta[i]*H[i]*H[i];
      Eta6(4,4) += 2.*Eta[i]*Eta[i]*H[i]*H[i];
      Eta6(4,5) += 2.*Eta[i]*Eta[i]*Eta[i]*H[i]*H[i];
      Eta6(5,0) += 2.*Eta[i]*Eta[i]*E[i]*H[i];
      Eta6(5,1) += 2.*Eta[i]*Eta[i]*Eta[i]*E[i]*H[i];
      Eta6(5,2) += 2.*Eta[i]*Eta[i]*Eta[i]*Eta[i]*E[i]*H[i];
      Eta6(5,3) += 2.*Eta[i]*Eta[i]*H[i]*H[i];
      Eta6(5,4) += 2.*Eta[i]*Eta[i]*Eta[i]*H[i]*H[i];
      Eta6(5,5) += 2.*Eta[i]*Eta[i]*Eta[i]*Eta[i]*H[i]*H[i];
      TEta6(0) += 2.*(T[i]-E[i]-H[i])*E[i];
      TEta6(1) += 2.*Eta[i]*(T[i]-E[i]-H[i])*E[i];
      TEta6(2) += 2.*Eta[i]*Eta[i]*(T[i]-E[i]-H[i])*E[i];
      TEta6(3) += 2.*(T[i]-E[i]-H[i])*H[i];
      TEta6(4) += 2.*Eta[i]*(T[i]-E[i]-H[i])*H[i];
      TEta6(5) += 2.*Eta[i]*Eta[i]*(T[i]-E[i]-H[i])*H[i];

      //(a0+a2*eta**2) E + (b0+b2*eta**2) H fit
      Eta4(0,0) += 2.*E[i]*E[i];
      Eta4(0,1) += 2.*Eta[i]*Eta[i]*E[i]*E[i];
      Eta4(0,2) += 2.*E[i]*H[i];
      Eta4(0,3) += 2.*Eta[i]*Eta[i]*E[i]*H[i];
      Eta4(1,0) += 2.*Eta[i]*Eta[i]*E[i]*E[i];
      Eta4(1,1) += 2.*Eta[i]*Eta[i]*Eta[i]*Eta[i]*E[i]*E[i];
      Eta4(1,2) += 2.*Eta[i]*Eta[i]*E[i]*H[i];
      Eta4(1,3) += 2.*Eta[i]*Eta[i]*Eta[i]*Eta[i]*E[i]*H[i];
      Eta4(2,0) += 2.*E[i]*H[i];
      Eta4(2,1) += 2.*Eta[i]*Eta[i]*E[i]*H[i];
      Eta4(2,2) += 2.*H[i]*H[i];
      Eta4(2,3) += 2.*Eta[i]*Eta[i]*H[i]*H[i];
      Eta4(3,0) += 2.*Eta[i]*Eta[i]*E[i]*H[i];
      Eta4(3,1) += 2.*Eta[i]*Eta[i]*Eta[i]*Eta[i]*E[i]*H[i];
      Eta4(3,2) += 2.*Eta[i]*Eta[i]*H[i]*H[i];
      Eta4(3,3) += 2.*Eta[i]*Eta[i]*Eta[i]*Eta[i]*H[i]*H[i];
      TEta4(0) += 2.*(T[i]-E[i]-H[i])*E[i];
      TEta4(1) += 2.*Eta[i]*Eta[i]*(T[i]-E[i]-H[i])*E[i];
      TEta4(2) += 2.*(T[i]-E[i]-H[i])*H[i];
      TEta4(3) += 2.*Eta[i]*Eta[i]*(T[i]-E[i]-H[i])*H[i];
      */
      // if ( E[i] == 0. ) continue;
      if ( endcap ) { 

	// (a0+a2*eta**2) (E+H) fit
	/* */
	double etaPow = Eta[i]*Eta[i]*Eta[i]*Eta[i];
	double factor = 0.5;
	/* */
	Eta3(0,0) += 2.*(factor*E[i]+H[i])*(factor*E[i]+H[i]);
	Eta3(0,1) += 2.*etaPow*(factor*E[i]+H[i])*(factor*E[i]+H[i]);
	Eta3(1,0) += 2.*etaPow*(factor*E[i]+H[i])*(factor*E[i]+H[i]);
	Eta3(1,1) += 2.*etaPow*etaPow*(factor*E[i]+H[i])*(factor*E[i]+H[i]);
	TEta3(0) += 2.*(T[i]-E[i]-H[i])*(factor*E[i]+H[i]);
	TEta3(1) += 2.*etaPow*(T[i]-E[i]-H[i])*(factor*E[i]+H[i]);
	/* */
	/* 
	// E+(a0+a2*eta**2)H) fit
	Eta3(0,0) += 2.*H[i]*H[i];
	Eta3(0,1) += 2.*etaPow*H[i]*H[i];
	Eta3(1,0) += 2.*etaPow*H[i]*H[i];
	Eta3(1,1) += 2.*etaPow*etaPow*H[i]*H[i];
	TEta3(0) += 2.*(T[i]-E[i]-H[i])*H[i];
	TEta3(1) += 2.*etaPow*(T[i]-E[i]-H[i])*H[i];
	*/
      } else { 
	double etaPow = Eta[i]*Eta[i];
	// (a0+a1*eta)E + H) fit
	Eta3(0,0) += 2.*E[i]*E[i];
	Eta3(0,1) += 2.*etaPow*E[i]*E[i];
	Eta3(1,0) += 2.*etaPow*E[i]*E[i];
	Eta3(1,1) += 2.*etaPow*etaPow*E[i]*E[i];
	TEta3(0) += 2.*(T[i]-E[i]-H[i])*E[i];
	TEta3(1) += 2.*etaPow*(T[i]-E[i]-H[i])*E[i];


      }

    }
    
    success = 
      //Eta2.Invert() && 
      Eta3.Invert(); // && 
      //Eta4.Invert() && 
      //Eta6.Invert();

    //Eta2Coeffs = Eta2 * TEta2;
    Eta3Coeffs = Eta3 * TEta3;
    //Eta4Coeffs = Eta4 * TEta4;
    //Eta6Coeffs = Eta6 * TEta6;

    return success;

  }

  bool xMatrices() { 

    bool success = true;

    for (unsigned i=0; i<T.size(); ++i ) { 
      if ( E[i] == 0. || H[i] == 0. ) continue;

      // (a+b*XE+c*xH) (E+H) fit
      X3(0,0) += 2.*(E[i]+H[i])*(E[i]+H[i]);
      X3(0,1) += 2.*XE[i]*(E[i]+H[i])*(E[i]+H[i]);
      X3(0,2) += 2.*XH[i]*(E[i]+H[i])*(E[i]+H[i]);
      X3(1,0) += 2.*XE[i]*(E[i]+H[i])*(E[i]+H[i]);
      X3(1,1) += 2.*XE[i]*XE[i]*(E[i]+H[i])*(E[i]+H[i]);
      X3(1,2) += 2.*XE[i]*XH[i]*(E[i]+H[i])*(E[i]+H[i]);
      X3(2,0) += 2.*XH[i]*(E[i]+H[i])*(E[i]+H[i]);
      X3(2,1) += 2.*XH[i]*XE[i]*(E[i]+H[i])*(E[i]+H[i]);
      X3(2,2) += 2.*XH[i]*XH[i]*(E[i]+H[i])*(E[i]+H[i]);
      TX3(0) += 2.*(T[i]-E[i]-H[i])*(E[i]+H[i]);
      TX3(1) += 2.*XE[i]*(T[i]-E[i]-H[i])*(E[i]+H[i]);
      TX3(2) += 2.*XH[i]*(T[i]-E[i]-H[i])*(E[i]+H[i]);

    }

    success = X3.Invert();

    X3Coeffs = X3 * TX3;

    return success;

  }

  double denom() {
    if ( Denom != 0. ) return Denom;
    for (unsigned i=0; i<T.size(); ++i ) { 
      if ( E[i] == 0. ) continue;
      for (unsigned j=0; j<T.size(); ++j ) {
	if ( E[j] == 0. ) continue;
	Denom += E[i]*H[j]*(E[i]*H[j]-E[j]*H[i]);
      }
    }
    return Denom;
  }

  double numA() {
    if ( NumA != 0. ) return NumA;
    for (unsigned i=0; i<T.size(); ++i ) { 
      if ( E[i] == 0. ) continue;
      for (unsigned j=0; j<T.size(); ++j ) {
	if ( E[j] == 0. ) continue;
	NumA += T[i]*H[j]*(E[i]*H[j]-E[j]*H[i]);
      }
    }
    return NumA;
  }

  double numB() {
    if ( NumB != 0. ) return NumB;
    for (unsigned i=0; i<T.size(); ++i ) { 
      if ( E[i] == 0. ) continue;
      for (unsigned j=0; j<T.size(); ++j ) {
	if ( E[j] == 0. ) continue;
	NumB -= T[i]*E[j]*(E[i]*H[j]-E[j]*H[i]);
      }
    }
    return NumB;
  }

  double aCoeff() {
    if ( ACoeff != 0. ) return ACoeff;
    ACoeff = denom()!= 0. ? numA()/denom() : 0.;
    return ACoeff;
  }

  double bCoeff() {
    if ( BCoeff != 0. ) return BCoeff;
    BCoeff = denom()!= 0 ? numB()/denom() : 0.;
    return BCoeff;
  }

  double sigmaA() {
    if ( SigmaA != 0. ) return SigmaA;
    double num = 0.;
    for (unsigned i=0; i<T.size(); ++i ) {
      if ( E[i] == 0. ) continue;
      num += H[i]*H[i];
    }
    
    SigmaA = denom()!= 0. ? sqrt(num/denom()) : 0.;
    return SigmaA;
  }

  double sigmaB() {
    if ( SigmaB != 0. ) return SigmaB;
    double num = 0.;
    for (unsigned i=0; i<T.size(); ++i ) {
      if ( E[i] == 0. ) continue;
      num += E[i]*E[i];
    }
    
    SigmaB = denom()!= 0 ? sqrt(num/denom()) : 0.;
    return SigmaB;
  }

  double rhoAB() {
    if ( RhoAB != 0. ) return RhoAB;
    if ( sigmaA()*sigmaB() == 0. ) return 0.;
    double num = 0.;
    for (unsigned i=0; i<T.size(); ++i ) {
      if ( E[i] == 0. ) continue;
      num += -E[i]*H[i];
    }
    
    RhoAB = denom()!= 0 ? num/denom() : 0.;
    RhoAB /= sigmaA()*sigmaB();
    return RhoAB;
  }

  double cCoeff() {
    if ( CCoeff != 0. ) return CCoeff;
    double den = 0.;
    double num = 0.;
    for (unsigned i=0; i<T.size(); ++i ) {
      if ( E[i] != 0. ) continue;
      den += H[i]*H[i];
      num += T[i]*H[i];
    }
    CCoeff = den != 0 ? num/den : 0.;
    SigmaC = den != 0 ? 1./sqrt(den) : 0.;
    return CCoeff;
  }

  double sigmaC() {
    if ( SigmaC != 0. ) return SigmaC;
    cCoeff();
    return SigmaC;
  }

  double dCoeff() {
    if ( DCoeff != 0. ) return DCoeff;
    double den = 0.;
    double num = 0.;
    for (unsigned i=0; i<T.size(); ++i ) {
      if ( E[i] == 0. ) continue;
      den += (E[i]+H[i])*(E[i]+H[i]);
      num += T[i]*(E[i]+H[i]);
    }
    DCoeff = den != 0 ? num/den : 0.;
    SigmaD = den != 0 ? 1./sqrt(den) : 0.;
    return DCoeff;
  }

  double sigmaD() {
    if ( SigmaD != 0. ) return SigmaD;
    dCoeff();
    return SigmaD;
  }

  double average(unsigned flag, unsigned input = 0) { 
    double ave = 0.;
    unsigned int n = 0;
    for (unsigned i=0; i<T.size(); ++i ) { 
      if ( flag == 0 && E[i]*H[i] == 0. ) continue;
      if ( flag == 1 && E[i] != 0.  ) continue;
      // if ( flag == 2 && H[i] > 0.  ) continue;
      double thresh = E[i] != 0. ? threshE : threshH;
      if ( input == 0 ) 
	ave += T[i]*S[i] + thresh;
      else if ( input == 1 ) 
	ave += E[i]*S[i];	
      ++n;
    }
    ave /= n;
    return ave;    
  }

  double rms(double ave,unsigned flag, unsigned input = 0) { 
    double rms = 0.;
    unsigned int n = 0;
    for (unsigned i=0; i<T.size(); ++i ) {
      if ( flag == 0 && E[i] == 0. ) continue;
      if ( flag == 1 && E[i] != 0. && H[i] != 0. )  continue;
      double thresh = E[i] != 0. ? threshE : threshH;
      if ( input == 0 ) 
	rms += (T[i]*S[i]+thresh)*(T[i]*S[i]+thresh);
      else if ( input == 1 ) 
	rms += E[i]*E[i]*S[i]*S[i];
      ++n;
    }
    rms = sqrt((rms/n -ave*ave)/n);
    return rms;    
  }

  unsigned int size() { 
    return T.size();
  }
  
  
  double theta() {
    if ( sigmaA()*sigmaB() == 0. ) return 0.;
    double s2 = -2.*rhoAB() * sigmaA() * sigmaB();
    double c2 = -(sigmaA()*sigmaA() - sigmaB()*sigmaB());
    double one = sqrt(s2*s2 + c2*c2);
    s2 /= one;
    return 0.5*asin(s2);
    /*
    double t2 
      = 2.*rhoAB() * sigmaA() * sigmaB()
      / (sigmaA()*sigmaA() - sigmaB()*sigmaB());
    // double at2 = atan(t2)/2;
    //if ( at2 < 0. ) at2 += 2.*3.1415926535;
    double t = fabs(0.5 * atan(t2));
    // if ( t > 3.1415926535/2. ) t = 3.1415926535/2.-t;
    return t;
    */
  }

  double sigmaU() {
    if ( sigmaA()*sigmaB() == 0. ) return 0.;
    double sA2 = sigmaA()*sigmaA();
    double sB2 = sigmaB()*sigmaB();
    double rh2 = rhoAB()*rhoAB();
    double s = sA2 + sB2 + sqrt((sA2-sB2)*(sA2-sB2) + 4.*rh2*sA2*sB2);
    return sqrt(0.5*s);
  }
    
  double sigmaV() {
    if ( sigmaA()*sigmaB() == 0. ) return 0.;
    double sA2 = sigmaA()*sigmaA();
    double sB2 = sigmaB()*sigmaB();
    double rh2 = rhoAB()*rhoAB();
    double s = sA2 + sB2 - sqrt((sA2-sB2)*(sA2-sB2) + 4.*rh2*sA2*sB2);
    return sqrt(0.5*s);
  }

  double costheta() { 
    return cos(theta());
  }

  double sintheta() {
    return sin(theta());
  }

  double uCoeff() {
    return costheta()*aCoeff() - sintheta()*bCoeff();
  }
    
  double vCoeff() {
    return sintheta()*aCoeff() + costheta()*bCoeff();
  }
    
  // Members (all public)
  double minE;
  double maxE;

  double Denom;
  double NumA;
  double NumB;

  double ACoeff;
  double BCoeff;
  double CCoeff;
  double DCoeff;

  double SigmaA;
  double SigmaB;
  double SigmaC;
  double SigmaD;

  double RhoAB;

  vector<double> Eta;
  vector<double> E;
  vector<double> H;
  vector<double> T;
  vector<double> S;
  vector<double> XE;
  vector<double> XH;

  StdMatrix22 Eta2;
  StdMatrix22 Eta3;
  StdMatrix44 Eta4;
  StdMatrix66 Eta6;
  
  ROOT::Math::SVector<double,2> TEta2, Eta2Coeffs;
  ROOT::Math::SVector<double,2> TEta3, Eta3Coeffs;
  ROOT::Math::SVector<double,4> TEta4, Eta4Coeffs;
  ROOT::Math::SVector<double,6> TEta6,Eta6Coeffs;

  StdMatrix33 X3;
  ROOT::Math::SVector<double,3> TX3, X3Coeffs;

  StdMatrix11 H1;
  StdMatrix22 E2;
  StdMatrix11 E1;

  ROOT::Math::SVector<double,1> TH1, H1Coeffs;
  ROOT::Math::SVector<double,1> TE1, E1Coeffs;
  ROOT::Math::SVector<double,2> TE2, E2Coeffs;

};


TGraphErrors* Gra() { return gra; }
TGraphErrors* Grb() { return grb; }
TGraphErrors* Grc() { return grc; }
TGraphErrors* GraEta0() { return graEta0; }
TGraphErrors* GrbEta0() { return grbEta0; }
TGraphErrors* GraX() { return graX; }
TGraphErrors* GrbX() { return grbX; }
TGraphErrors* GrcX() { return grcX; }

TGraph* 
FitReso(TH2F* h, string hname, double xmin = 7., double xmax = 1000., unsigned rebin = 1) { 

  vector<TH1F*> histos;
  vector<double> energies, sigmas, means, rms, aver;
  rebin = 1;
  for ( unsigned bin=2; bin<1000; ) { 

    string shname = hname;
    char type[3];
    sprintf(type,"%i_%i",bin,bin+4*rebin);
    shname += type;

    histos.push_back((TH1F*)h->ProjectionY(shname.c_str(),bin,bin+4*rebin));
    // histos.back()->Fit("gaus",(Option_t*)("L"),"",-0.7,0.7);
    histos.back()->Fit("gaus","","",-1.,1.);
    TF1* gaus = histos.back()->GetFunction( "gaus" );
    histos.back()->Fit("gaus","","",gaus->GetParameter(1)-2.*gaus->GetParameter(2),1.);
    gaus = histos.back()->GetFunction( "gaus" );
    histos.back()->Fit("gaus","","",gaus->GetParameter(1)-2.*gaus->GetParameter(2),1.);
    gaus = histos.back()->GetFunction( "gaus" );

    energies.push_back(bin+2.*rebin);
    sigmas.push_back(gaus->GetParameter(2)/(1.+min(0.,gaus->GetParameter(1))));
    means.push_back(gaus->GetParameter(1));
    rms.push_back(histos.back()->GetRMS());
    aver.push_back(histos.back()->GetMean());

    std::cout << "Bin : " << bin << " - ";
    bin += 4*rebin;
    std::cout << bin << " GeV" << std::endl;
    if ( bin > 10 ) rebin = 2.;
    if ( bin > 100 ) rebin = 20.;
    if ( bin > 1000 ) rebin = 50.;

  }

  TGraph* reso = new TGraph(energies.size(),&energies[0],&sigmas[0]);
  TGraph* resp = new TGraph(energies.size(),&energies[0],&means[0]);

  TH2F* hres = new TH2F("hres","",100,0,1000,100,0.,0.5);
  TH2F* hrep = new TH2F("hrep","",100,0,1000,100,-1.,1.);
  TH2F* hrep2 = new TH2F("hrep2","",100,0,100,100,-0.5,0.5);

  TCanvas *C = new TCanvas("C","",1000, 600);
  C->Divide(2,1);
  C->cd(1);
  hres->SetStats(0);
  hres->Draw();
  gPad->SetGridx();
  gPad->SetGridy();
  reso->SetMarkerStyle(22);						
  reso->SetMarkerSize(0.8);						
  reso->SetMarkerColor(4);						
  reso->SetLineColor(4);						  
  reso->SetLineWidth(2);						  
  TF1* fres = new TF1("fres","sqrt([0]*[0]+[1]*[1]/x+[2]*[2]/(x*x))",1,1000);
  fres->SetParameters(0.06,1.20,0.);
  if ( etamin < 0.1 ) fres->FixParameter(2,0.);
  reso->Fit("fres","","",xmin,xmax);
  reso->Draw("P");

  string legend;
  int fres0 = (int)(fres->GetParameter(0)*100.);
  int fres1 = (int)(10.*(fres->GetParameter(0)*100.-fres0));
  int fres2 = (int)(fres->GetParameter(1)*100.);
  char text[100];
  sprintf(text,"#sigma/E = %i%/#sqrt{E} + %i.%i%",fres2,fres0,fres1);
  legend += text;
  TLegend *leg=new TLegend(0.30,0.75,0.85,0.85);
  leg->AddEntry(reso,legend.c_str(),"lp");
  leg->SetTextSize(0.04);
  leg->Draw();

  C->cd(2);
  hrep->SetStats(0);
  hrep->Draw();
  gPad->SetGridx();
  gPad->SetGridy();
  //gPad->SetLogx();
  resp->SetMarkerStyle(22);						
  resp->SetMarkerSize(0.8);						
  resp->SetMarkerColor(2);						
  resp->SetLineColor(2);						  
  resp->SetLineWidth(2);						  
  resp->Draw("P");

  string filename = hname + ".gif";
  C->Print(filename.c_str());

  TCanvas *C1 = new TCanvas("C1","",600, 600);
  C1->Divide(1,1);
  C1->cd(9);

  hrep2->SetStats(0);
  hrep2->Draw();
  gPad->SetGridx();
  gPad->SetGridy();
  //gPad->SetLogx();
  resp->SetMarkerStyle(22);						
  resp->SetMarkerSize(0.8);						
  resp->SetMarkerColor(2);						
  resp->SetLineColor(2);						  
  resp->SetLineWidth(2);						  
  resp->Draw("P");
  string filename2 = hname + "_insert.gif";
  C1->Print(filename2.c_str());
  
  return resp;

}

double findABC(std::vector<Fit*>& fits, double tE0, double tE, double tH) { 
  
  TFile* theFile = TFile::Open("pfcalib_all.root");
  //TFile* theFile = TFile::Open("pfcalib_piall.root");
  //TFile* theFile = TFile::Open("pfcalib_pi.root");
  //TFile* theFile = TFile::Open("pfcalib_k0L.root");
  TTree* TT = (TTree*)theFile->Get("s");
  NTuple* ntuple = new NTuple(TT);
  
  threshE0 = tE0;
  threshE = tE;
  threshH = tH;

  etamin_Fit = etamin_FitE;
  etamax_Fit = etamax_FitE;
  
  for ( unsigned ifit=0; ifit<fits.size(); ++ifit ) fits[ifit]->clear();
  
  unsigned nEntries = TT->GetEntriesFast();
  for ( unsigned entry=0; entry<nEntries; ++entry ) {
    // if ( entry/10000*10000 == entry ) cout << "Process entry " << entry << endl;
    ntuple->LoadTree(entry);
    TT->GetEntry(entry);
    double e = ntuple->ecal_;
    double h = ntuple->hcal_;
    double t = ntuple->true_;
    double eta = fabs(ntuple->eta_);

    if ( eta < etamin || eta > etamax ) continue;
    // This cut will fit a, b and c in a limited eta range.
    // The eta correction can then be applied afterwards
    // Proven not to work better than without this cut
    if ((e+h) < 0.5 ) continue;
    if ( h == 0. ) continue;
    if (t < 1. ) continue;
    for ( unsigned ifit=0; ifit<fits.size(); ++ifit ) {
      if ( h == 0. ) break;
      if ( fits[ifit]->fill(e,h,t,eta) ) break;
    }
  }
  
  double ave0 = 0.;
  double rms0 = 0.;
  double aveA = 0.;
  double rmsA = 0.;
  double aveB = 0.;
  double rmsB = 0.;
  double aveC = 0.;
  double rmsC = 0.;
  unsigned nfit = 0;
  for ( unsigned ifit=2; ifit<fits.size()-1; ++ifit ) {
    if ( !fits[ifit]->size() ) continue;
    fits[ifit]->eMatrices();
    if (ifit<10) continue;
    ave0 += fits[ifit]->E1Coeffs(0);
    aveA += fits[ifit]->E2Coeffs(0);
    aveB += fits[ifit]->E2Coeffs(1);
    aveC += fits[ifit]->H1Coeffs(0);
    rms0 += fits[ifit]->E1Coeffs(0) * fits[ifit]->E1Coeffs(0);
    rmsA += fits[ifit]->E2Coeffs(0) * fits[ifit]->E2Coeffs(0);
    rmsB += fits[ifit]->E2Coeffs(1) * fits[ifit]->E2Coeffs(1);
    rmsC += fits[ifit]->H1Coeffs(0) * fits[ifit]->H1Coeffs(0);
    nfit += 1;
  }
  ave0 /= (float)nfit;
  aveA /= (float)nfit;
  aveB /= (float)nfit;
  aveC /= (float)nfit;
  rms0 = sqrt(rms0/nfit - ave0*ave0);
  rmsA = sqrt(rmsA/nfit - aveA*aveA);
  rmsB = sqrt(rmsB/nfit - aveB*aveB);
  rmsC = sqrt(rmsC/nfit - aveC*aveC);
  cout << tE << " " << tH << " RMS 0/A+B/C = " 
       << rms0 << ", " << sqrt(rmsA*rmsA+rmsB*rmsB) << ", " << rmsC
       << endl;
  
  delete ntuple;
  return sqrt(rms0*rms0+rmsA*rmsA+rmsB*rmsB+rmsC+rmsC);
}

TGraphErrors* 
computeBarrelCoefficients(const char* calibFile) {


  gROOT->Reset();
  /*
  TTree* T = new TTree("ntuple","");
  T->ReadFile(calibFile, "eta:phi:E:Ecor:Hcor:Ecal:Hcal:DR");
  */

  std::cout << "Open the file" << std::endl;
  TFile* theFile = TFile::Open("pfcalib_all.root");
  //TFile* theFile = TFile::Open("pfcalib_piall.root");
  //TFile* theFile = TFile::Open("pfcalib_pi.root");
  //TFile* theFile = TFile::Open("pfcalib_k0L.root");
  std::cout << "Get the tree" << std::endl;
  TTree* TT = (TTree*)theFile->Get("s");
  std::cout << "Create the ntuple" << std::endl;
  NTuple* ntuple = new NTuple(TT);
  std::cout << "Get the number of entries " << std::endl;
  unsigned nEntries = TT->GetEntriesFast();
  std::cout << "Number of entries = " << nEntries << std::endl;

  TH2F* rawEvsPBarrel = new TH2F("rawEvsPBarrel","ECAL+HCAL raw Energy vs P (barrel)",
			   1000,0.,1000.,150,-1.5,1.5);
  TH2F* rawEvsPEndcap = new TH2F("rawEvsPEndcap","ECAL+HCAL raw Energy vs P (end-cap)",
			   1000,0.,1000.,150,-1.5,1.5);

  TT->Draw("(ecal+hcal-true)/true:true>>rawEvsPBarrel","ecal+hcal>1&&true>1&&abs(eta)<1.4");
  TT->Draw("(ecal+hcal-true)/true:true>>rawEvsPEndcap","ecal+hcal>1&&true>1&&abs(eta)>1.4&&abs(eta)<2.6");

  TCanvas *c1 = new TCanvas();
  c1->cd();
  rawEvsPBarrel->ProfileX()->Draw();

  TCanvas *c2 = new TCanvas();
  c2->cd();
  rawEvsPEndcap->ProfileX()->Draw();

  //TCanvas *c4 = new TCanvas();
  //TCanvas *c5 = new TCanvas();
  TCanvas *c6 = new TCanvas();
  TCanvas *c7 = new TCanvas();

  vector<Fit*> fits;
  
  for ( double bin=0.; bin<10.; bin=bin+1.0) { 
    fits.push_back(new Fit(bin,bin+1.0));    
  }

  for ( double bin=10.; bin<100.; bin=bin+2.) { 
    fits.push_back(new Fit(bin,bin+2.));    
  }

  for ( double bin=100.; bin<1000.; bin=bin+10.) { 
    fits.push_back(new Fit(bin,bin+10.));    
  }

  //std::cout << "Delete the ntuple" << std::endl;
  //delete ntuple;
  //  std::cout << "Close the file" << std::endl;
  //theFile->Close();  
  //std::cout << "File closed" << std::endl;

  /* 
  double rmsMaxE0 = 999.;
  for ( double te0=0.; te0<10; te0=te0+0.1) {
    double rmsCoeff = findABC(fits, te0, threshE, threshH);
    if ( rmsCoeff < rmsMaxE0 ) { 
      rmsMaxE0 = rmsCoeff;
    } else { 
      threshE0 = te0-0.1;
      break;
    }
  }
  */

  /*
  std::cout << "Call findABC (1) " << std::endl;
  double rmsMaxE = 999.;
  for ( double te=1.; te<10; te=te+0.1) {
    double rmsCoeff = findABC(fits, threshE0, te, threshH);
    if ( rmsCoeff < rmsMaxE ) { 
      rmsMaxE = rmsCoeff;
    } else { 
      threshE = te-0.1;
      break;
    }
  }

  std::cout << "Call findABC (2) " << std::endl;
  double rmsMaxH = 999.;
  for ( double th=1.; th<10.; th=th+0.1) {
    double rmsCoeff = findABC(fits, threshE0, threshE, th);
    if ( rmsCoeff < rmsMaxH ) { 
      rmsMaxH = rmsCoeff;
    } else { 
      threshH = th-0.1;
      break;
    }
  }
  */

  cout << "the thresholds are " << threshE0 << ", " << threshE << " and " << threshH << endl;
  findABC(fits, threshE0, threshE, threshH);

  /* */
  
  vector<double> xa0, xab, xc, a, a0, b, c, sxa0, sxab, sxc, sa, sa0, sb, sc;
  vector<double> at, bt, ct, dt, aht, bht, sat, sbt, sct, sdt, saht, sbht;
  for ( unsigned ifit=1; ifit<fits.size()-1; ++ifit ) {
    if ( !fits[ifit]->size() ) continue;
    //fits[ifit]->eMatrices();
    cout << "Bin " << ifit 
	 << "; a, b, c, a0  = " << fits[ifit]->E2Coeffs(0)
	 << " +/- " << sqrt(fits[ifit]->E2(0,0))
	 << ", "  << fits[ifit]->E2Coeffs(1)
	 << " +/- " << sqrt(fits[ifit]->E2(1,1))
	 << ", "  << fits[ifit]->H1Coeffs(0)
	 << " +/- " << sqrt(fits[ifit]->H1(0,0))
	 << ", "  << fits[ifit]->E1Coeffs(0)
	 << " +/- " << sqrt(fits[ifit]->E1(0,0))
	 << endl;

    // if ( fits[ifit].average(2) == 0. ) continue 
    a.push_back(fits[ifit]->E2Coeffs(0));
    b.push_back(fits[ifit]->E2Coeffs(1));
    c.push_back(fits[ifit]->H1Coeffs(0));
    // if ( fits[ifit]->E1Coeffs(0) != 0. ) 
    //  a0.push_back(fits[ifit]->E1Coeffs(0));

    sa.push_back(sqrt(fits[ifit]->E2(0,0)));
    sb.push_back(sqrt(fits[ifit]->E2(1,1)));
    sc.push_back(sqrt(fits[ifit]->H1(0,0)));
    // if ( fits[ifit]->E1Coeffs(0) != 0. ) 
    //  sa0.push_back(sqrt(fits[ifit]->E1(0,0)));

    /*
    if ( fits[ifit]->E1Coeffs(0) != 0. ) { 
      xa0.push_back(fits[ifit]->average(2));
      sxa0.push_back(fits[ifit]->rms(xa0.back(),2));
    }
    */

    xab.push_back(fits[ifit]->average(0));
    sxab.push_back(fits[ifit]->rms(xab.back(),0));

    xc.push_back(fits[ifit]->average(1));
    sxc.push_back(fits[ifit]->rms(xc.back(),1));

  }
  
  // std::cout << "a0 coeffs = " << xa0.size()  << " " << a0.size() << " " << sa0.size() << std::endl;

  // gra0 = new TGraphErrors ( xa0.size(), &xa0[0], &a0[0], &sxa0[0], &sa0[0]);
  gra = new TGraphErrors ( xab.size(), &xab[0], &a[0], &sxab[0], &sa[0]);
  grb = new TGraphErrors ( xab.size(), &xab[0], &b[0], &sxab[0], &sb[0]);
  grc = new TGraphErrors ( xc.size(), &xc[0], &c[0], &sxc[0], &sc[0]);

  TH2F *h = new TH2F("a & b","", 100, 0., 1000., 10, -0.5, 2.5 );

  c1->cd();
  h->SetStats(0);
  h->Draw();

  /*
  TF1* fa0 = new TF1("fa0","[0]+([1]+[2]/sqrt(x))*exp(-x/[3])-[4]*exp(-x*x/[5])",0,1000);
  //fb->SetParameters(1.2,0.3,2,50,1.5,10); // 0.0
  //fb->SetParameters(1.2,0.5,-1,50,1.2,30); // 2.5
  fa0->SetParameters(1.2,0.5,-1.5,40,1.2,30); // 3.0
  gra0->Fit("fa0","","",3,50);  
  gra0->Fit("fa0","W","",3,50);
  */

  TF1* fa = new TF1("fa","[0]+([1]+[2]/sqrt(x))*exp(-x/[3])-[4]*exp(-x*x/[5])",0,1000);
  if ( endcap ) {
    fa->SetParameters(1.10,0.12,-0.2,119,1.3,80); // 3.0
    fa->FixParameter(4,2.0);
  } else {
    fa->SetParameters(1.10,0.12,-0.2,119,1.3,40); // 3.0
  }
  gra->Fit("fa","","",1.5,1000);  
  gra->Fit("fa","","",1.5,1000);
  gra->Fit("fa","","",1.5,1000);

  TF1* fb = new TF1("fb","[0]+([1]+[2]/sqrt(x))*exp(-x/[3])-[4]*exp(-x*x/[5])",0,1000);
  if ( endcap ) {
    fb->SetParameters(1.04,0.27,-0.6,70,1.05,50); // 3.0
    fb->FixParameter(4,1.2);
  } else {
    fb->SetParameters(1.04,0.27,-0.6,70,1.05,24); // 3.0
    fb->FixParameter(4,1.0);
  }
  grb->Fit("fb","","",1.5,1000);  
  grb->Fit("fb","","",1.5,1000);
  grb->Fit("fb","","",1.5,1000);

  TF1* fc = new TF1("fc","[0]+([1]+[2]/sqrt(x))*exp(-x/[3])-[4]*exp(-x*x/[5])",0,1000);

  fc->SetParameters(0.95,0.04,0.07,76,1.23,25); // 2.9
  grc->Fit("fc","","",1.5,1000);  
  grc->Fit("fc","","",1.5,1000);
  grc->Fit("fc","","",1.5,1000);

  gra->SetMarkerStyle(25);						
  gra->SetMarkerSize(0.1);						
  gra->SetMarkerColor(2);						
  gra->SetLineColor(2);						  
  gra->SetLineWidth(2);						  
  gra->Draw("P");

  grb->SetMarkerStyle(22);						
  grb->SetMarkerSize(0.1);						
  grb->SetMarkerColor(4);						
  grb->SetLineColor(4);						  
  grb->SetLineWidth(2);						  
  grb->Draw("P");

  c2->cd();
  h->SetStats(0);
  h->Draw();

  grc->SetMarkerStyle(25);						
  grc->SetMarkerSize(0.1);						
  grc->SetMarkerColor(4);						
  grc->SetLineColor(4);						  
  grc->SetLineWidth(2);						  
  grc->Draw("P");

  /*
  gra0->SetMarkerStyle(25);						
  gra0->SetMarkerSize(0.1);						
  gra0->SetMarkerColor(2);						
  gra0->SetLineColor(2);						  
  gra0->SetLineWidth(2);						  
  gra0->Draw("P");
  fa0->Draw("same");
  */

  TH2F* result = new TH2F("result","Resultat",1000,0,1000.,150,-1.5,1.5);
  TH2F* resultCor = new TH2F("resultCor","Resultat",1000,0,1000.,150,-1.5,1.5);
  TH2F* resultRaw = new TH2F("resultRaw","Resultat",1000,0,1000.,150,-1.5,1.5);

  TH2F* resultE = new TH2F("resultE","Resultat",1000,0,1000.,150,-1.5,1.5);
  TH2F* resultCorE = new TH2F("resultCorE","Resultat",1000,0,1000.,150,-1.5,1.5);
  TH2F* resultRawE = new TH2F("resultRawE","Resultat",1000,0,1000.,150,-1.5,1.5);

  TH2F* resultE0 = new TH2F("resultE0","Resultat",1000,0,1000.,150,-1.5,1.5);
  TH2F* resultCorE0 = new TH2F("resultCorE0","Resultat",1000,0,1000.,150,-1.5,1.5);
  TH2F* resultRawE0 = new TH2F("resultRawE0","Resultat",1000,0,1000.,150,-1.5,1.5);

  TH2F* resultH = new TH2F("resultH","Resultat",1000,0,1000.,150,-1.5,1.5);
  TH2F* resultCorH = new TH2F("resultCorH","Resultat",1000,0,1000.,150,-1.5,1.5);
  TH2F* resultRawH = new TH2F("resultRawH","Resultat",1000,0,1000.,150,-1.5,1.5);

  TH2F* etadep = new TH2F("etadep","Eta Dependence",150,etamin,etamax,150,-1.5,1.5);
  TH2F* etadepE0 = new TH2F("etadepE0","Eta Dependence",150,etamin,etamax,150,-1.5,1.5);
  TH2F* etadepE = new TH2F("etadepE","Eta Dependence",150,etamin,etamax,150,-1.5,1.5);
  TH2F* etadepH = new TH2F("etadepH","Eta Dependence",150,etamin,etamax,150,-1.5,1.5);
  TH2F* etadep_1_10 = new TH2F("etadep_1_10","Eta Dependence 1 a 10",150,etamin,etamax,150,-1.5,1.5);
  TH2F* etadep_10_100 = new TH2F("etadep_10_100","Eta Dependence 10 a 100",150,etamin,etamax,150,-1.5,1.5);
  TH2F* etadep_100_1000 = new TH2F("etadep_100_1000","Eta Dependence 100 a 1000",150,etamin,etamax,150,-1.5,1.5);
  TH2F* etadepH_1_10 = new TH2F("etadepH_1_10","Eta Dependence 1 a 10",150,etamin,etamax,150,-1.5,1.5);
  TH2F* etadepH_10_100 = new TH2F("etadepH_10_100","Eta Dependence 10 a 100",150,etamin,etamax,150,-1.5,1.5);
  TH2F* etadepH_100_1000 = new TH2F("etadepH_100_1000","Eta Dependence 100 a 1000",150,etamin,etamax,150,-1.5,1.5);

  // Clear fit inputs
  for ( unsigned ifit=0; ifit<fits.size(); ++ifit ) fits[ifit]->clear();
  fits.clear();

  for ( double bin=0.; bin<10.; bin=bin+2.) { 
    fits.push_back(new Fit(bin,bin+2.));    
  }

  for ( double bin=10.; bin<100.; bin=bin+10.) { 
    fits.push_back(new Fit(bin,bin+10.));    
  }

  for ( double bin=100.; bin<1000.; bin=bin+50.) { 
    fits.push_back(new Fit(bin,bin+50.));    
  }



  //unsigned nEntries = TT->GetEntriesFast();
  for ( unsigned entry=0; entry<nEntries; ++entry ) {
    if ( entry/10000*10000 == entry ) cout << "Process entry " << entry << endl;
    ntuple->LoadTree(entry);
    TT->GetEntry(entry);
    double e = ntuple->ecal_;
    double h = ntuple->hcal_;
    double t = ntuple->true_;
    double eta = fabs(ntuple->eta_);

    if ( eta < etamin || eta > etamax ) continue;
    if (e+h < 0.5 ) continue;
    if (t < 1. ) continue;
    // if (e!=0.) continue;
    if (h==0.) continue;

    double a = h>0. ? fa->Eval(t) : fa->Eval(t);
    double b = e>0. ? fb->Eval(t) : fc->Eval(t);
    double thresh = e > 0. ? ( h>0? threshE : threshE0) : threshH;
    double eCorr = thresh + a*e + b*h;

    etamin_Fit = etamin_FitEta;
    etamax_Fit = etamax_FitEta;

    unsigned myifit = 0;
    for ( unsigned ifit=1; ifit<fits.size(); ++ifit ) {
      if ( h == 0. ) break;
      // Offset independent of eta
      myifit = ifit;
      if ( fits[ifit]->fill(a*e,b*h,t,eta) ) break;
      // Offset dependent on eta
      //if ( fits[ifit]->fill(a*e,thresh+b*h,t,eta,1) ) break;
    }
    // if ( myifit == 1 ) 
    // std::cout << e << " " << a*e << " " << h << " " << b*h << " " << t << std::endl;
    
    if ( eta < etamax_FitE && eta > etamin_FitE ) { 
      result->Fill( t, (eCorr-t)/t );
      if ( e>0. )
	if ( h>0. ) 
	  resultE->Fill( t, (eCorr-t)/t ); 
	else
	  resultE0->Fill( t, (eCorr-t)/t ); 
      else
	resultH->Fill( t, (eCorr-t)/t );
    }
    
    etadep->Fill( fabs(ntuple->eta_),  (eCorr-t)/t );
    if ( e>0. ) 
      if ( h>0. )
	etadepE->Fill( fabs(ntuple->eta_),  (eCorr-t)/t );
      else
	etadepE0->Fill( fabs(ntuple->eta_),  (eCorr-t)/t );
    else
      etadepH->Fill( fabs(ntuple->eta_),  (eCorr-t)/t );

    if ( t < 10 ) {
      etadep_1_10->Fill( fabs(ntuple->eta_),  (eCorr-t)/t );
      if ( e == 0. && h > 0. ) 
	etadepH_1_10->Fill( fabs(ntuple->eta_),  (eCorr-t)/t );
    } else if ( t < 100 ) { 
      etadep_10_100->Fill( fabs(ntuple->eta_),  (eCorr-t)/t);
      if ( e == 0. && h > 0. ) 
	etadepH_10_100->Fill( fabs(ntuple->eta_),  (eCorr-t)/t );
    } else if ( t < 1000 ) {
      etadep_100_1000->Fill( fabs(ntuple->eta_),  (eCorr-t)/t);
      if ( e == 0. && h > 0. ) 
	etadepH_100_1000->Fill( fabs(ntuple->eta_),  (eCorr-t)/t );
    }

    eCorr = ntuple->ecal_+ntuple->hcal_;
    if ( eta < etamax_FitE && eta > etamin_FitE ) { 
      resultRaw->Fill( t, (eCorr-t)/t );
      if ( e>0. ) 
	if ( h>0. )
	  resultRawE->Fill( t, (eCorr-t)/t ); 
	else
	  resultRawE0->Fill( t, (eCorr-t)/t ); 
      else
	resultRawH->Fill( t, (eCorr-t)/t );
    }

  }

  vector<double> aEta0, bEta0, cEta0, aEta1, bEta1, cEta1, aEta2, bEta2, cEta2;
  vector<double> saEta0, sbEta0, scEta0, saEta1, sbEta1, scEta1, saEta2, sbEta2, scEta2;
  vector<double> xEta, sxEta;
  for ( unsigned ifit=1; ifit<fits.size()-1; ++ifit ) {
    if ( !fits[ifit]->size() ) continue;
    fits[ifit]->etaMatrices();
    cout << "Bin " << ifit 
	 << "; a0, b0 = " 
	 << "   " << fits[ifit]->Eta3Coeffs(0) 
	 << " +/- " << sqrt(fits[ifit]->Eta3(0,0))
	 << ", "  << fits[ifit]->Eta3Coeffs(1)
	 << " +/- " << sqrt(fits[ifit]->Eta3(1,1))
	 << endl;
    aEta0.push_back(fits[ifit]->Eta3Coeffs(0));
    bEta0.push_back(fits[ifit]->Eta3Coeffs(1));
    saEta0.push_back(sqrt(fits[ifit]->Eta3(0,0)));
    sbEta0.push_back(sqrt(fits[ifit]->Eta3(1,1)));

    xEta.push_back(fits[ifit]->average(2));
    sxEta.push_back(fits[ifit]->rms(xEta.back(),2));



  }

  graEta0 = new TGraphErrors ( xEta.size(), &xEta[0], &aEta0[0], &sxEta[0], &saEta0[0]);
  grbEta0 = new TGraphErrors ( xEta.size(), &xEta[0], &bEta0[0], &sxEta[0], &sbEta0[0]);

  TH2F *heta = new TH2F("eta","", 100, 0., 1000., 1000, -1.0, 1.0 );

  c6->cd();
  heta->SetStats(0);
  heta->Draw();

  //TF1* faEta = new TF1("faEta","[0]+[1]*exp(-x/[2])+[3]*[3]*exp(-x*x/([4]*[4]))",0,1000);
  TF1* faEta = new TF1("faEta","[0]+[1]*exp(-x/[2])",0,1000);
  faEta->SetParameters(0.02,-0.1,200);
  if (etamin<0.1) { 
    // faEta->FixParameter(0,0.02);
  } else { 
  }

  graEta0->Fit("faEta","","",2.0,1000);  
  graEta0->Fit("faEta","","",2.0,1000);

  TF1* fbEta = new TF1("fbEta","[0]+[1]*exp(-x/[2])+[3]*[3]*exp(-x*x/([4]*[4]))",0,1000);
  //TF1* fbEta = new TF1("fbEta","[0]+[1]*exp(-x/[2])",0,1000);
  if ( etamin < 0.1 ) {
    //fbEta->SetParameters(-0.02,0.4,200,-0.1,5.);
    fbEta->SetParameters(-0.02,0.4,200,0.,0.);
    fbEta->FixParameter(3,0.);
    fbEta->FixParameter(4,0.);
    //fbEta->FixParameter(0,-0.02);
  } else {
    fbEta->SetParameters(0.07,-2.5,6.0,0.3,175.);
    //fbEta->SetParameters(0.07,-2.5,6.0);
  }

  grbEta0->Fit("fbEta","","",2.0,1000);  
  grbEta0->Fit("fbEta","","",2.0,1000);

  graEta0->SetMarkerStyle(25);						
  graEta0->SetMarkerSize(0.1);						
  graEta0->SetMarkerColor(2);						
  graEta0->SetLineColor(2);						  
  graEta0->SetLineWidth(2);						  
  graEta0->Draw("P");

  grbEta0->SetMarkerStyle(22);						
  grbEta0->SetMarkerSize(0.1);						
  grbEta0->SetMarkerColor(4);						
  grbEta0->SetLineColor(4);						  
  grbEta0->SetLineWidth(2);						  
  grbEta0->Draw("P");

  c7->cd();
  heta->SetStats(0);
  heta->Draw();

  TH2F* resultEta = new TH2F("resultEta","Resultat eta",1000,0,1000.,150,-1.5,1.5);
  TH2F* resultEtaE0 = new TH2F("resultEtaE0","Resultat eta",1000,0,1000.,150,-1.5,1.5);
  TH2F* resultEtaE = new TH2F("resultEtaE","Resultat eta",1000,0,1000.,150,-1.5,1.5);
  TH2F* resultEtaH = new TH2F("resultEtaH","Resultat eta",1000,0,1000.,150,-1.5,1.5);
  TH2F* etadep_Cor = new TH2F("etadep_Cor","Eta Dependence",150,etamin,etamax,150,-1.5,1.5);
  TH2F* etadepE0_Cor = new TH2F("etadepE0_Cor","Eta Dependence",150,etamin,etamax,150,-1.5,1.5);
  TH2F* etadepE_Cor = new TH2F("etadepE_Cor","Eta Dependence",150,etamin,etamax,150,-1.5,1.5);
  TH2F* etadepH_Cor = new TH2F("etadepH_Cor","Eta Dependence",150,etamin,etamax,150,-1.5,1.5);
  TH2F* etadep_1_10_Cor = new TH2F("etadep_1_10_Cor","Eta Dependence 1 a 10",150,etamin,etamax,150,-1.5,1.5);
  TH2F* etadep_10_100_Cor = new TH2F("etadep_10_100_Cor","Eta Dependence 10 a 100",150,etamin,etamax,150,-1.5,1.5);
  TH2F* etadep_100_1000_Cor = new TH2F("etadep_100_1000_Cor","Eta Dependence 100 a 1000",150,etamin,etamax,150,-1.5,1.5);
  TH2F* etadepH_1_10_Cor = new TH2F("etadepH_1_10_Cor","Eta Dependence 1 a 10",150,etamin,etamax,150,-1.5,1.5);
  TH2F* etadepH_10_100_Cor = new TH2F("etadepH_10_100_Cor","Eta Dependence 10 a 100",150,etamin,etamax,150,-1.5,1.5);
  TH2F* etadepH_100_1000_Cor = new TH2F("etadepH_100_1000_Cor","Eta Dependence 100 a 1000",150,etamin,etamax,150,-1.5,1.5);

  // Clear fit inputs
  for ( unsigned ifit=0; ifit<fits.size(); ++ifit ) fits[ifit]->clear();
  fits.clear();

  for ( double bin=0.; bin<10.; bin=bin+2.) { 
    fits.push_back(new Fit(bin,bin+2.));    
  }

  for ( double bin=10.; bin<100.; bin=bin+20.) { 
    fits.push_back(new Fit(bin,bin+20.));    
  }

  for ( double bin=100.; bin<1000.; bin=bin+200.) { 
    fits.push_back(new Fit(bin,bin+200.));    
  }


  for ( unsigned entry=0; entry<nEntries; ++entry ) {
    if ( entry/10000*10000 == entry ) cout << "Process entry " << entry << endl;
    ntuple->LoadTree(entry);
    TT->GetEntry(entry);
    double e = ntuple->ecal_;
    double h = ntuple->hcal_;
    double t = ntuple->true_;
    double eta = fabs(ntuple->eta_);

    if ( eta < etamin || eta > etamax ) continue;
    if (e+h < 0.5 ) continue;
    if (t < 1. ) continue;
    // if (e!=0.) continue;
    if (h==0.) continue;

    double a = h>0. ? fa->Eval(t) : fa->Eval(t);
    double b = e>0. ? fb->Eval(t) : fc->Eval(t);
    double thresh = e > 0. ? ( h>0? threshE : threshE0) : threshH;

    double factor = 0.5;
    double etaCorrE = endcap ?
      1. + faEta->Eval(t) + factor*fbEta->Eval(t)*(eta-etamin)*(eta-etamin)*(eta-etamin)*(eta-etamin) : 
      1. + faEta->Eval(t) + fbEta->Eval(t)*(eta-etamin)*(eta-etamin);
    double etaCorrH = endcap ?
      1. + faEta->Eval(t) + fbEta->Eval(t)*(eta-etamin)*(eta-etamin)*(eta-etamin)*(eta-etamin) : 
      1.;

    double eCorr = thresh + etaCorrE * a * e + etaCorrH * b * h;
    // double eCorr = etaCorrE * a * e + etaCorrH * (thresh + b * h);
    /*
    for ( unsigned ifit=0; ifit<fits.size(); ++ifit ) {
      if ( fits[ifit]->fill(etaCorrE*a*e,etaCorrH*b*h,t,eta) ) break;
    }
    */
    if ( eta < etamax_FitEta && eta > etamin_FitEta ) { 
      resultEta->Fill( t, (eCorr-t)/t );
      if ( e>0. ) 
	if ( h>0. ) 
	  resultEtaE->Fill( t, (eCorr-t)/t );
	else
	  resultEtaE0->Fill( t, (eCorr-t)/t );
      else
	resultEtaH->Fill( t, (eCorr-t)/t );
    }

    etadep_Cor->Fill( fabs(ntuple->eta_),  (eCorr-t)/t );
    if ( e>0. ) 
      if ( h>0. )
	etadepE_Cor->Fill( fabs(ntuple->eta_),  (eCorr-t)/t );
      else
	etadepE0_Cor->Fill( fabs(ntuple->eta_),  (eCorr-t)/t );
    else
      etadepH_Cor->Fill( fabs(ntuple->eta_),  (eCorr-t)/t );
    if ( t < 10 ) {
      etadep_1_10_Cor->Fill( fabs(ntuple->eta_),  (eCorr-t)/t );
      if ( e == 0. && h > 0. ) 
	etadepH_1_10_Cor->Fill( fabs(ntuple->eta_),  (eCorr-t)/t );
    } else if ( t < 100 ) { 
      etadep_10_100_Cor->Fill( fabs(ntuple->eta_),  (eCorr-t)/t);
      if ( e == 0. && h > 0. ) 
	etadepH_10_100_Cor->Fill( fabs(ntuple->eta_),  (eCorr-t)/t );
    } else if ( t < 1000 ) { 
      etadep_100_1000_Cor->Fill( fabs(ntuple->eta_),  (eCorr-t)/t);
      if ( e == 0. && h > 0. ) 
	etadepH_100_1000_Cor->Fill( fabs(ntuple->eta_),  (eCorr-t)/t );
    }

  }


  // output for further coding 
  const char* fa_expression = fa->GetTitle();
  const char* fb_expression = fb->GetTitle();
  const char* fc_expression = fc->GetTitle();
  const char* faEta_expression = faEta->GetTitle();
  const char* fbEta_expression = fbEta->GetTitle();

  cout << "  threshE = " << threshE << ";" << endl; 
  cout << "  threshH = " << threshH << ";" << endl; 
  if ( endcap ) { 
    //cout << "  fa0Endcap = new TF1(\"faEndcap\",\"" << fa_expression << "\",1.,1000.);" << endl;
    cout << "  faEndcap = new TF1(\"faEndcap\",\"" << fa_expression << "\",1.,1000.);" << endl;
    cout << "  fbEndcap = new TF1(\"fbEndcap\",\"" << fb_expression << "\",1.,1000.);" << endl;
    cout << "  fcEndcap = new TF1(\"fcEndcap\",\"" << fc_expression << "\",1.,1000.);" << endl;
    cout << "  faEtaEndcap = new TF1(\"faEtaEndcap\",\"" << faEta_expression << "\",1.,1000.);" << endl;
    cout << "  fbEtaEndcap = new TF1(\"fbEtaEndcap\",\"" << fbEta_expression << "\",1.,1000.);" << endl;
  } else { 
    //cout << "  fa0Barrel = new TF1(\"fa0Barrel\",\"" << fa_expression << "\",1.,1000.);" << endl;
    cout << "  faBarrel = new TF1(\"faBarrel\",\"" << fa_expression << "\",1.,1000.);" << endl;
    cout << "  fbBarrel = new TF1(\"fbBarrel\",\"" << fb_expression << "\",1.,1000.);" << endl;
    cout << "  fcBarrel = new TF1(\"fcBarrel\",\"" << fc_expression << "\",1.,1000.);" << endl;
    cout << "  faEtaBarrel = new TF1(\"faEtaBarrel\",\"" << faEta_expression << "\",1.,1000.);" << endl;
    cout << "  fbEtaBarrel = new TF1(\"fbEtaBarrel\",\"" << fbEta_expression << "\",1.,1000.);" << endl;
  }

  for ( unsigned ip=0; ip < 10 ; ++ip ) { 

    //double param_fa0 = fa0->GetParameter(ip);
    double param_fa = fa->GetParameter(ip);
    double param_fb = fb->GetParameter(ip);
    double param_fc = fc->GetParameter(ip);
    double param_faEta = faEta->GetParameter(ip);
    double param_fbEta = fbEta->GetParameter(ip);
    if ( endcap ) { 
      //if ( param_fa0 != 0. ) cout << "  fa0Endcap->SetParameter(" << ip << "," << param_fa0 << ");" << endl;
      if ( param_fa != 0. ) cout << "  faEndcap->SetParameter(" << ip << "," << param_fa << ");" << endl;
      if ( param_fb != 0. ) cout << "  fbEndcap->SetParameter(" << ip << "," << param_fb << ");" << endl;
      if ( param_fc != 0. ) cout << "  fcEndcap->SetParameter(" << ip << "," << param_fc << ");" << endl;
      if ( param_faEta != 0. ) cout << "  faEtaEndcap->SetParameter(" << ip << "," << param_faEta << ");" << endl;
      if ( param_fbEta != 0. ) cout << "  fbEtaEndcap->SetParameter(" << ip << "," << param_fbEta << ");" << endl;
    } else if ( etamin < 1.7 ) { 
      //if ( param_fa0 != 0. ) cout << "  fa0Barrel->SetParameter(" << ip << "," << param_fa0 << ");" << endl;
      if ( param_fa != 0. ) cout << "  faBarrel->SetParameter(" << ip << "," << param_fa << ");" << endl;
      if ( param_fb != 0. ) cout << "  fbBarrel->SetParameter(" << ip << "," << param_fb << ");" << endl;
      if ( param_fc != 0. ) cout << "  fcBarrel->SetParameter(" << ip << "," << param_fc << ");" << endl;
      if ( param_faEta != 0. ) cout << "  faEtaBarrel->SetParameter(" << ip << "," << param_faEta << ");" << endl;
      if ( param_fbEta != 0. ) cout << "  fbEtaBarrel->SetParameter(" << ip << "," << param_fbEta << ");" << endl;
    }    

  }

  if ( endcap ) { 
    cout << "  double a = h>0. ? faEndcap->Eval(t) : faEndcap->Eval(t);" << endl
	 << "  double b = e>0. ? fbEndcap->Eval(t) : fcEndcap->Eval(t);" << endl
	 << "  double etaCorr = 1. + faEtaEndcap->Eval(t) + fbEtaEndcap->Eval(t)*(fabs(eta)-" << etamin 
	 << ")*(fabs(eta)-" << etamin << ");" << endl
	 << "  double thresh = e > 0. ? threshE : threshH;" << endl
	 << "  double eCorr = thresh + etaCorr * ( a*e + b*h );" << endl;
  } else { 
    cout << "  double a = h>0. ? faBarrel->Eval(t) : faBarrel->Eval(t);" << endl
	 << "  double b = e>0. ? fbBarrel->Eval(t) : fcBarrel->Eval(t);" << endl
	 << "  double etaCorr = 1. + faEtaBarrel->Eval(t) + fbEtaBarrel->Eval(t)*eta*eta;" << endl
	 << "  double thresh = e > 0. ? threshE : threshH;" << endl
	 << "  double eCorr = thresh + etaCorr * ( a*e + b*h );" << endl;
  }  
  
  return grb;

}
 
void calib() {
  computeBarrelCoefficients("calib_Endcap_130.txt");
  //computeBarrelCoefficients("calib_Barrel_130.txt");
  //computeBarrelCoefficients("calib_130.txt");
  //computeBarrelCoefficients("calib_130_Fast.txt");
}
