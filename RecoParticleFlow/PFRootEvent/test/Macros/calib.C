#include <vector>
#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>
#include "TF1.h"
#include "TH2F.h"
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

TGraphErrors* gra;
TGraphErrors* grb;
TGraphErrors* grc;
TGraphErrors* graEta0;
TGraphErrors* grbEta0;
TGraphErrors* graX;
TGraphErrors* grbX; 
TGraphErrors* grcX;

double threshE = 3.7;
double threshH = 2.9;

using namespace std;

class NTuple {
public :
   TTree          *fChain;   //!pointer to the analyzed TTree or TChain
   Int_t           fCurrent; //!current Tree number in a TChain

   // Declaration of leaf types
   Float_t         eta;
   Float_t         phi;
   Float_t         E;
   Float_t         Ecal;
   Float_t         Hcal;
   Float_t         Ecol;
   Float_t         Hcol;
   Float_t         Ejam;
   Float_t         Hjam;

   // List of branches
   TBranch        *b_eta;   //!
   TBranch        *b_phi;   //!
   TBranch        *b_E;   //!
   TBranch        *b_Ecal;   //!
   TBranch        *b_Hcal;   //!
   TBranch        *b_Ecol;   //!
   TBranch        *b_Hcol;   //!
   TBranch        *b_Ejam;   //!
   TBranch        *b_Hjam;   //!

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

   fChain->SetBranchAddress("eta", &eta, &b_eta);
   fChain->SetBranchAddress("phi", &phi, &b_phi);
   fChain->SetBranchAddress("E", &E, &b_E);
   fChain->SetBranchAddress("Ecal", &Ecal, &b_Ecal);
   fChain->SetBranchAddress("Hcal", &Hcal, &b_Hcal);
   fChain->SetBranchAddress("Ecol", &Ecol, &b_Ecol);
   fChain->SetBranchAddress("Hcol", &Hcol, &b_Hcol);
   fChain->SetBranchAddress("Ejam", &Ejam, &b_Ejam);
   fChain->SetBranchAddress("Hjam", &Hjam, &b_Hjam);
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
    if ( ( input == 0 && t > minE && t < maxE ) ||
	 ( input == 1 && e > minE && e < maxE ) ) { 
      S.push_back(sqrt(0.06*0.06 + 1.20*(e+h)));
      //S.push_back(1.);
      double thresh = e > 0 ? threshE : threshH;
      E.push_back(e/S.back());
      H.push_back(h/S.back());
      T.push_back((t-thresh)/S.back());
      Eta.push_back(eta);
      if ( e > 0. && h > 0. ) { 
	XE.push_back((e+h)*(e+h)/(e*e));
	XH.push_back((e+h)*(e+h)/(h*h));
      } else { 
	XE.push_back(0.);
	XH.push_back(0.);
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
    H1(0,0) = 0.;
    TH1(0)  = 0.;

    for (unsigned i=0; i<T.size(); ++i ) { 

      if ( E[i] != 0. ) { 
	// (a*E +b*sqrt(E)+c*H+d*sqrt(H) fit
	E2(0,0) += 2.*E[i]*E[i];
	E2(0,1) += 2.*E[i]*H[i];
	E2(1,0) += 2.*E[i]*H[i];
	E2(1,1) += 2.*H[i]*H[i];
	TE2(0) += 2.*T[i]*E[i];
	TE2(1) += 2.*T[i]*H[i];

      } else {  

	H1(0,0) += 2*H[i]*H[i];
	TH1(0) += 2*T[i]*H[i];

      }

    }

    success = 
      H1.Invert() &&
      E2.Invert();

    H1Coeffs = H1 * TH1;
    T1Coeffs = H1 * XH1;
    E2Coeffs = E2 * TE2;
    T2Coeffs = E2 * XE2;

    return success;

  }

  bool etaMatrices() { 

    bool success = true;

    for (unsigned i=0; i<T.size(); ++i ) { 
      // if ( E[i] == 0. ) continue;
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

      // (1+a*eta+a*eta**2) (E+H) fit
      Eta3(0,0) += 2.*(E[i]+H[i])*(E[i]+H[i]);
      Eta3(0,1) += 2.*Eta[i]*(E[i]+H[i])*(E[i]+H[i]);
      Eta3(0,2) += 2.*Eta[i]*Eta[i]*(E[i]+H[i])*(E[i]+H[i]);
      Eta3(1,0) += 2.*Eta[i]*(E[i]+H[i])*(E[i]+H[i]);
      Eta3(1,1) += 2.*Eta[i]*Eta[i]*(E[i]+H[i])*(E[i]+H[i]);
      Eta3(1,2) += 2.*Eta[i]*Eta[i]*Eta[i]*(E[i]+H[i])*(E[i]+H[i]);
      Eta3(2,0) += 2.*Eta[i]*Eta[i]*(E[i]+H[i])*(E[i]+H[i]);
      Eta3(2,1) += 2.*Eta[i]*Eta[i]*Eta[i]*(E[i]+H[i])*(E[i]+H[i]);
      Eta3(2,2) += 2.*Eta[i]*Eta[i]*Eta[i]*Eta[i]*(E[i]+H[i])*(E[i]+H[i]);
      TEta3(0) += 2.*(T[i]-E[i]-H[i])*(E[i]+H[i]);
      TEta3(1) += 2.*Eta[i]*(T[i]-E[i]-H[i])*(E[i]+H[i]);
      TEta3(2) += 2.*Eta[i]*Eta[i]*(T[i]-E[i]-H[i])*(E[i]+H[i]);

      // (a0+a2*eta**2) (E+H) fit
      Eta2(0,0) += 2.*(E[i]+H[i])*(E[i]+H[i]);
      Eta2(0,1) += 2.*Eta[i]*Eta[i]*(E[i]+H[i])*(E[i]+H[i]);
      Eta2(1,0) += 2.*Eta[i]*Eta[i]*(E[i]+H[i])*(E[i]+H[i]);
      Eta2(1,1) += 2.*Eta[i]*Eta[i]*Eta[i]*Eta[i]*(E[i]+H[i])*(E[i]+H[i]);
      TEta2(0) += 2.*(T[i]-E[i]-H[i])*(E[i]+H[i]);
      TEta2(1) += 2.*Eta[i]*Eta[i]*(T[i]-E[i]-H[i])*(E[i]+H[i]);

    }

    success = 
      Eta2.Invert() && 
      Eta3.Invert() && 
      Eta4.Invert() && 
      Eta6.Invert();

    Eta2Coeffs = Eta2 * TEta2;
    Eta3Coeffs = Eta3 * TEta3;
    Eta4Coeffs = Eta4 * TEta4;
    Eta6Coeffs = Eta6 * TEta6;

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
      if ( E[i] > 0. ) continue;
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
      if ( flag == 0 && E[i] == 0. ) continue;
      if ( flag == 1 && E[i] != 0. ) continue;
      double thresh = E[i] > 0 ? threshE : threshH;
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
      if ( flag == 1 && E[i] != 0. ) continue;
      double thresh = E[i] > 0 ? threshE : threshH;
      if ( input == 0 ) 
	rms += (T[i]*S[i]+thresh)*(T[i]*S[i]+thresh);
      else if ( input == 1 ) 
	rms += E[i]*E[i]*S[i]*S[i];
      ++n;
    }
    rms = sqrt((rms/n -ave*ave)/n);
    return rms;    
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
  StdMatrix33 Eta3;
  StdMatrix44 Eta4;
  StdMatrix66 Eta6;
  
  ROOT::Math::SVector<double,2> TEta2, Eta2Coeffs;
  ROOT::Math::SVector<double,3> TEta3, Eta3Coeffs;
  ROOT::Math::SVector<double,4> TEta4, Eta4Coeffs;
  ROOT::Math::SVector<double,6> TEta6,Eta6Coeffs;

  StdMatrix33 X3;
  ROOT::Math::SVector<double,3> TX3, X3Coeffs;

  StdMatrix11 H1;
  StdMatrix22 E2;

  ROOT::Math::SVector<double,1> XH1, TH1, T1Coeffs, H1Coeffs;
  ROOT::Math::SVector<double,2> XE2, TE2, T2Coeffs, E2Coeffs;

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
FitReso(TH2F* h, string hname, double xmin = 7.) { 

  vector<TH1F*> histos;
  vector<double> energies, sigmas, means, rms, aver;
  for ( unsigned bin=2; bin<1000; bin=bin+4 ) { 

    string shname = hname;
    char type[3];
    sprintf(type,"%i_%i",bin,bin+4);
    shname += type;

    histos.push_back((TH1F*)h->ProjectionY(shname.c_str(),bin,bin+4));
    // histos.back()->Fit("gaus",(Option_t*)("L"),"",-0.7,0.7);
    histos.back()->Fit("gaus","","",-1.,1.);
    TF1* gaus = histos.back()->GetFunction( "gaus" );

    energies.push_back(bin+2.);
    sigmas.push_back(gaus->GetParameter(2)/(1.+min(0.,gaus->GetParameter(1))));
    means.push_back(gaus->GetParameter(1));
    rms.push_back(histos.back()->GetRMS());
    aver.push_back(histos.back()->GetMean());

  }
  
  TGraph* reso = new TGraph(energies.size(),&energies[0],&sigmas[0]);
  TGraph* resp = new TGraph(energies.size(),&energies[0],&means[0]);

  TH2F* hres = new TH2F("hres","",100,0,1000,100,0.,0.5);
  TH2F* hrep = new TH2F("hrep","",100,0,1000,100,-1.,1.);

  TCanvas *C = new TCanvas("C","",1000, 600);
  C->Divide(2,1);
  C->cd(1);
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
  reso->Fit("fres","","",xmin,1000);
  reso->Draw("P");
  C->cd(2);
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

  return reso;

}

double findABC(std::vector<Fit*>& fits, double tE, double tH) { 
  
  TFile* theFile = TFile::Open("myTree.root");
  TTree* TT = (TTree*)theFile->Get("ntuple");
  NTuple* ntuple = new NTuple(TT);
  
  threshE = tE;
  threshH = tH;
  
  for ( unsigned ifit=0; ifit<fits.size(); ++ifit ) fits[ifit]->clear();
  
  unsigned nEntries = TT->GetEntriesFast();
  for ( unsigned entry=0; entry<nEntries; ++entry ) {
    // if ( entry/10000*10000 == entry ) cout << "Process entry " << entry << endl;
    ntuple->LoadTree(entry);
    TT->GetEntry(entry);
    double e = ntuple->Ecal;
    double h = ntuple->Hcal;
    double t = ntuple->E;
    double eta = fabs(ntuple->eta);
    if ( eta > 1.4 ) continue;
    if ((e+h) < 0.5 ) continue;
    if ( h == 0. ) continue;
    if (t < 1. ) continue;
    for ( unsigned ifit=0; ifit<fits.size(); ++ifit ) {
      if ( fits[ifit]->fill(e,h,t,eta) ) break;
    }
  }
  
  double aveA = 0.;
  double rmsA = 0.;
  double aveB = 0.;
  double rmsB = 0.;
  double aveC = 0.;
  double rmsC = 0.;
  unsigned nfit = 0;
  for ( unsigned ifit=5; ifit<fits.size()-1; ++ifit ) {
    fits[ifit]->eMatrices();
    if (ifit<20) continue;
    aveA += fits[ifit]->E2Coeffs(0);
    aveB += fits[ifit]->E2Coeffs(1);
    aveC += fits[ifit]->H1Coeffs(0);
    rmsA += fits[ifit]->E2Coeffs(0) * fits[ifit]->E2Coeffs(0);
    rmsB += fits[ifit]->E2Coeffs(1) * fits[ifit]->E2Coeffs(1);
    rmsC += fits[ifit]->H1Coeffs(0) * fits[ifit]->H1Coeffs(0);
    nfit += 1;
  }
  aveA /= (float)nfit;
  aveB /= (float)nfit;
  aveC /= (float)nfit;
  rmsA = sqrt(rmsA/nfit - aveA*aveA);
  rmsB = sqrt(rmsB/nfit - aveB*aveB);
  rmsC = sqrt(rmsC/nfit - aveC*aveC);
  cout << tE << " " << tH << " RMS A+B/C = " 
       << sqrt(rmsA*rmsA+rmsB*rmsB) << ", " << rmsC
       << endl;
  
  delete ntuple;
  return sqrt(rmsA*rmsA+rmsB*rmsB+rmsC+rmsC);
}

void
computeBarrelCoefficients(const char* calibFile) {


  gROOT->Reset();
  TTree* T = new TTree("ntuple","");
  T->ReadFile(calibFile, "eta:phi:E:Ecal:Hcal:Ecol:Hcol:Ejam:Hjam");

  TH2F* rawEvsP = new TH2F("rawEvsP","ECAL+HCAL raw Energy vs P",
			   1000,0.,1000.,150,-1.5,1.5);
  TH2F* colEvsP = new TH2F("colEvsP","ECAL+HCAL col Energy vs P",
			   1000,0.,1000.,150,-1.5,1.5);
  TH2F* jamEvsP = new TH2F("jamEvsP","ECAL+HCAL jam Energy vs P",
			   1000,0.,1000.,150,-1.5,1.5);

  T->Draw("(Ecal+Hcal-E)/E:E>>rawEvsP","Ecal+Hcal>1&&E>1&&abs(eta)<1.4");
  T->Draw("(Ecol+Hcol-E)/E:E>>colEvsP","Ecal+Hcal>1&&E>1&&abs(eta)<1.4");
  T->Draw("(Ejam+Hjam-E)/E:E>>jamEvsP","Ecal+Hcal>1&&E>1&&abs(eta)<1.4");

  TCanvas *c1 = new TCanvas();
  c1->cd();
  rawEvsP->ProfileX()->Draw();

  TCanvas *c2 = new TCanvas();
  c2->cd();
  colEvsP->ProfileX()->Draw();

  TCanvas *c3 = new TCanvas();
  c3->cd();
  jamEvsP->ProfileX()->Draw();

  //TCanvas *c4 = new TCanvas();
  //TCanvas *c5 = new TCanvas();
  TCanvas *c6 = new TCanvas();
  TCanvas *c7 = new TCanvas();
  TCanvas *c8 = new TCanvas();

  vector<Fit*> fits;
  
  for ( double bin=0.; bin<10.; bin=bin+0.5) { 
    fits.push_back(new Fit(bin,bin+1.));    
  }

  for ( double bin=10.; bin<100.; bin=bin+1.) { 
    fits.push_back(new Fit(bin,bin+1.));    
  }

  for ( double bin=100.; bin<1000.; bin=bin+5.) { 
    fits.push_back(new Fit(bin,bin+10.));    
  }

  TFile* myFile = new TFile("myTree.root","recreate");
  myFile->cd();
  T->Write();
  T->Print();
  delete myFile;

  TFile* theFile = TFile::Open("myTree.root");
  TTree* TT = (TTree*)theFile->Get("ntuple");
  NTuple* ntuple = new NTuple(TT);
  unsigned nEntries = TT->GetEntriesFast();

  /*
  double rmsMaxE = 999.;
  for ( double te=0.; te<5; te=te+0.1) {
    double rmsCoeff = findABC(fits, te, 0.);
    if ( rmsCoeff < rmsMaxE ) { 
      rmsMaxE = rmsCoeff;
    } else { 
      threshE = te-0.1;
      break;
    }
  }

  double rmsMaxH = 999.;
  for ( double th=0.; th<5.; th=th+0.1) {
    double rmsCoeff = findABC(fits, threshE, th);
    if ( rmsCoeff < rmsMaxH ) { 
      rmsMaxH = rmsCoeff;
    } else { 
      threshH = th-0.1;
      break;
    }
  }
  */

  cout << "the thresholds are " << threshE << " and " << threshH << endl;
  findABC(fits, threshE, threshH);

  /* */
  
  vector<double> xab, xc, a, b, c, sxab, sxc, sa, sb, sc;
  vector<double> at, bt, ct, dt, aht, bht, sat, sbt, sct, sdt, saht, sbht;
  for ( unsigned ifit=5; ifit<fits.size()-1; ++ifit ) {
    //fits[ifit]->eMatrices();
    cout << "Bin " << ifit 
	 << "; a, b, c  = " << fits[ifit]->E2Coeffs(0)
	 << " +/- " << sqrt(fits[ifit]->E2(0,0))
	 << ", "  << fits[ifit]->E2Coeffs(1)
	 << " +/- " << sqrt(fits[ifit]->E2(1,1))
	 << ", "  << fits[ifit]->H1Coeffs(0)
	 << " +/- " << sqrt(fits[ifit]->H1(0,0))
	 << endl;

    // if ( fits[ifit].average(2) == 0. ) continue 
    a.push_back(fits[ifit]->E2Coeffs(0));
    b.push_back(fits[ifit]->E2Coeffs(1));
    c.push_back(fits[ifit]->H1Coeffs(0));

    xab.push_back(fits[ifit]->average(0));
    sxab.push_back(fits[ifit]->rms(xab.back(),0));

    xc.push_back(fits[ifit]->average(1));
    sxc.push_back(fits[ifit]->rms(xc.back(),1));

    sa.push_back(sqrt(fits[ifit]->E2(0,0)));
    sb.push_back(sqrt(fits[ifit]->E2(1,1)));
    sc.push_back(sqrt(fits[ifit]->H1(0,0)));
  }
  
  gra = new TGraphErrors ( xab.size(), &xab[0], &a[0], &sxab[0], &sa[0]);
  grb = new TGraphErrors ( xab.size(), &xab[0], &b[0], &sxab[0], &sb[0]);
  grc = new TGraphErrors ( xc.size(), &xc[0], &c[0], &sxc[0], &sc[0]);
  /*
  TGraphErrors* grd
    = new TGraphErrors ( xd.size(), &xd[0], &d[0], &sxd[0], &sd[0]);  
  TGraphErrors* gru
    = new TGraphErrors ( xab.size(), &xab[0], &u[0], &sxab[0], &su[0]);
  TGraphErrors* grv
    = new TGraphErrors ( xab.size(), &xab[0], &v[0], &sxab[0], &sv[0]);

  TGraphErrors* grs
    = new TGraphErrors ( xab.size(), &xab[0], &st[0], &sxab[0], 0);
  */

  TH2F *h = new TH2F("a & b","", 100, 0., 1000., 10, -0.5, 2.5 );

  /*
  TH2F *hu = new TH2F("u","", 100, 0., 1000., 10, -0.5, 2.0 );
  TH2F *hv = new TH2F("v","", 100, 0., 1000., 10, 0.0, 2.5 );
  TH2F *hs = new TH2F("s","", 100, 0., 1000., 10, -1.0, 1.0 );
  */

  c1->cd();
  h->SetStats(0);
  h->Draw();

  TF1* fa = new TF1("fa","[0]+([1]+[2]/sqrt(x))*exp(-x/[3])-[4]*exp(-x*x/[5])",0,1000);
  fa->SetParameters(1.15,0.2,0.,50,0.5,100); // 0.0
  fa->SetParameters(1.15,0.2,-0.5,100,1.0,70); // 2.5
  fa->SetParameters(1.15,0.2,-0.8,100,1.0,70); // 3.0
  gra->Fit("fa");  
  gra->Fit("fa","W");

  TF1* fb = new TF1("fb","[0]+([1]+[2]/sqrt(x))*exp(-x/[3])-[4]*exp(-x*x/[5])",0,1000);
  fb->SetParameters(1.2,0.3,2,50,1.5,10); // 0.0
  fb->SetParameters(1.2,0.5,-1,50,1.2,30); // 2.5
  fb->SetParameters(1.2,0.5,-1.5,40,1.2,30); // 3.0
  grb->Fit("fb");  
  grb->Fit("fb","W");

  TF1* fc = new TF1("fc","[0]+([1]+[2]/sqrt(x))*exp(-x/[3])-[4]*exp(-x*x/[5])",0,1000);
  fc->SetParameters(1.15,0.2,2,50,0.5,100); // 0
  fc->SetParameters(1.1,0.01,0.6,120,1.0,20); // 1
  fc->SetParameters(1.1,0.005,0.3,90,1.0,22); // 1.5
  fc->SetParameters(1.1,0.005,0.0,100,1.0,22); // 1.8
  fc->SetParameters(1.1,0.01,-0.1,75,1.0,22); // 2.0
  grc->Fit("fc");  
  grc->Fit("fc","W");

  /*
  TF1* fd = new TF1("fd","[0]+([1]+[2]/sqrt(x))*exp(-x/[3])-[4]*exp(-x*x/[5])",0,1000);
  fd->SetParameters(1.2,0.3,2,50,1.5,10);
  grd->Fit("fd");  
  grd->Fit("fd","W");
  */

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
  grd->SetMarkerStyle(25);						
  grd->SetMarkerSize(0.1);						
  grd->SetMarkerColor(2);						
  grd->SetLineColor(2);						  
  grd->SetLineWidth(2);						  
  grd->Draw("P");
  */

  /*
  c3->cd();
  hu->SetStats(0);
  hu->Draw();

  // TF1* fu = new TF1("fu","max(0,[0]*abs(x-[1])-[2]*x*sqrt(x)+[3]*x*x*x*x)",0.001,1);
  TF1* fu = new TF1("fu","[0]+([1]+[2]/sqrt(x))*exp(-x/[3])-[4]*exp(-x*x/[5])",0,1000);
  fu->SetParameters(0.6,-1.,0.,150,-1,50);
  gru->Fit("fu");  
  //gru->Fit("fu","W");

  TF1* fv = new TF1("fv","[0]+([1]+[2]/sqrt(x))*exp(-x/[3])-[4]*exp(-x*x/[5])",0,1000);
  fv->SetParameters(1.6,0.5,2,150,1,30);
  grv->Fit("fv");

  gru->SetMarkerStyle(25);						
  gru->SetMarkerSize(0.2);						
  gru->SetMarkerColor(2);						
  gru->SetLineColor(2);						  
  gru->SetLineWidth(2);						  
  gru->Draw("P");

  c4->cd();
  hv->SetStats(0);
  hv->Draw();

  //grv->Fit("fv","W");

  grv->SetMarkerStyle(22);						
  grv->SetMarkerSize(0.2);						
  grv->SetMarkerColor(4);						
  grv->SetLineColor(4);						  
  grv->SetLineWidth(2);						  
  grv->Draw("P");

  c5->cd();
  hs->SetStats(0);
  hs->Draw();

  TF1* fs = new TF1("fs","[0]+([1]+[2]*x+[3]*x*x+[4]*x*x*x)*exp(-x/[5])",0,1000);
  fs->SetParameters(0.3,-0.5,0.03,-3E-4,1E-6,60);
  grs->Fit("fs","W");

  grs->SetMarkerStyle(23);						
  grs->SetMarkerSize(0.2);						
  grs->SetMarkerColor(3);						
  grs->SetLineColor(3);						  
  grs->SetLineWidth(2);						  
  grs->Draw("P");

  */

  TH2F* result = new TH2F("result","Resultat",1000,0,1000.,150,-1.5,1.5);
  TH2F* resultCol = new TH2F("resultCol","Resultat",1000,0,1000.,150,-1.5,1.5);
  TH2F* resultJam = new TH2F("resultJam","Resultat",1000,0,1000.,150,-1.5,1.5);
  TH2F* resultRaw = new TH2F("resultRaw","Resultat",1000,0,1000.,150,-1.5,1.5);

  TH2F* resultE = new TH2F("resultE","Resultat",1000,0,1000.,150,-1.5,1.5);
  TH2F* resultColE = new TH2F("resultColE","Resultat",1000,0,1000.,150,-1.5,1.5);
  TH2F* resultJamE = new TH2F("resultJamE","Resultat",1000,0,1000.,150,-1.5,1.5);
  TH2F* resultRawE = new TH2F("resultRawE","Resultat",1000,0,1000.,150,-1.5,1.5);

  TH2F* resultH = new TH2F("resultH","Resultat",1000,0,1000.,150,-1.5,1.5);
  TH2F* resultColH = new TH2F("resultColH","Resultat",1000,0,1000.,150,-1.5,1.5);
  TH2F* resultJamH = new TH2F("resultJamH","Resultat",1000,0,1000.,150,-1.5,1.5);
  TH2F* resultRawH = new TH2F("resultRawH","Resultat",1000,0,1000.,150,-1.5,1.5);

  TH2F* etadep = new TH2F("etadep","Eta Dependence",150,0,1.5,150,-1.5,1.5);
  TH2F* etadep_1_10 = new TH2F("etadep_1_10","Eta Dependence 1 a 10",150,0,1.5,150,-1.5,1.5);
  TH2F* etadep_10_100 = new TH2F("etadep_10_100","Eta Dependence 1 a 10",150,0,1.5,150,-1.5,1.5);
  TH2F* etadep_100_1000 = new TH2F("etadep_100_1000","Eta Dependence 1 a 10",150,0,1.5,150,-1.5,1.5);


  // Clear fit inputs
  for ( unsigned ifit=0; ifit<fits.size(); ++ifit ) fits[ifit]->clear();
  fits.clear();

  for ( double bin=0.; bin<10.; bin=bin+2.) { 
    fits.push_back(new Fit(bin,bin+2.));    
  }

  for ( double bin=10.; bin<100.; bin=bin+5.) { 
    fits.push_back(new Fit(bin,bin+5.));    
  }

  for ( double bin=100.; bin<1000.; bin=bin+20.) { 
    fits.push_back(new Fit(bin,bin+20.));    
  }



  //unsigned nEntries = TT->GetEntriesFast();
  for ( unsigned entry=0; entry<nEntries; ++entry ) {
    if ( entry/10000*10000 == entry ) cout << "Process entry " << entry << endl;
    ntuple->LoadTree(entry);
    TT->GetEntry(entry);
    double e = ntuple->Ecal;
    double h = ntuple->Hcal;
    double t = ntuple->E;
    double eta = fabs(ntuple->eta);
    if ( eta > 1.4 ) continue;
    if (e+h < 0.5 ) continue;
    if (t < 1. ) continue;
    // if (e!=0.) continue;
    if (h==0.) continue;

    double a = fa->Eval(t);
    double b = e>0. ? fb->Eval(t) : fc->Eval(t);
    double thresh = e > 0. ? threshE : threshH;
    double eCorr = thresh + a*e + b*h;

    for ( unsigned ifit=0; ifit<fits.size(); ++ifit ) {
      if ( fits[ifit]->fill(a*e,b*h,t,eta) ) break;
    }
    
    result->Fill( t, (eCorr-t)/t );
    if ( e>0. ) 
      resultE->Fill( t, (eCorr-t)/t ); 
    else
      resultH->Fill( t, (eCorr-t)/t );
    
    etadep->Fill( fabs(ntuple->eta),  (eCorr-t)/t );
    if ( t < 10 ) 
      etadep_1_10->Fill( fabs(ntuple->eta),  (eCorr-t)/t );
    else if ( t < 100 ) 
      etadep_10_100->Fill( fabs(ntuple->eta),  (eCorr-t)/t);
    else if ( t < 1000 ) 
      etadep_100_1000->Fill( fabs(ntuple->eta),  (eCorr-t)/t);

    eCorr = ntuple->Ecol+ntuple->Hcol;
    resultCol->Fill( t, (eCorr-t)/t );
    if ( e>0. ) 
      resultColE->Fill( t, (eCorr-t)/t ); 
    else
      resultColH->Fill( t, (eCorr-t)/t );
    
    eCorr = ntuple->Ejam+ntuple->Hjam;
    resultJam->Fill( t, (eCorr-t)/t );
    if ( e>0. ) 
      resultJamE->Fill( t, (eCorr-t)/t ); 
    else
      resultJamH->Fill( t, (eCorr-t)/t );
    
    eCorr = ntuple->Ecal+ntuple->Hcal;
    resultRaw->Fill( t, (eCorr-t)/t );
    if ( e>0. ) 
      resultRawE->Fill( t, (eCorr-t)/t ); 
    else
      resultRawH->Fill( t, (eCorr-t)/t );

  }

  vector<double> aEta0, bEta0, cEta0, aEta1, bEta1, cEta1, aEta2, bEta2, cEta2;
  vector<double> saEta0, sbEta0, scEta0, saEta1, sbEta1, scEta1, saEta2, sbEta2, scEta2;
  vector<double> xEta, sxEta;
  for ( unsigned ifit=2; ifit<fits.size()-1; ++ifit ) {
    bool invert = fits[ifit]->etaMatrices();
    cout << "Bin " << ifit 
	 << "; a0, b0 = " 
	 << "   " << fits[ifit]->Eta2Coeffs(0) 
	 << " +/- " << sqrt(fits[ifit]->Eta2(0,0))
	 << ", "  << fits[ifit]->Eta2Coeffs(1)
	 << " +/- " << sqrt(fits[ifit]->Eta2(1,1))
	 << endl
	 << "a1, b1, a2, b2 = " 
	 << "   " << fits[ifit]->Eta4Coeffs(0)
	 << " +/- " << sqrt(fits[ifit]->Eta4(0,0))
	 << ", "  << fits[ifit]->Eta4Coeffs(1)
	 << " +/- " << sqrt(fits[ifit]->Eta4(1,1))
	 << ", "  << fits[ifit]->Eta4Coeffs(2)
	 << " +/- " << sqrt(fits[ifit]->Eta4(2,2))
	 << ", "  << fits[ifit]->Eta4Coeffs(3)
	 << " +/- " << sqrt(fits[ifit]->Eta4(3,3))
	 << endl;
    aEta0.push_back(fits[ifit]->Eta2Coeffs(0));
    bEta0.push_back(fits[ifit]->Eta2Coeffs(1));
    aEta1.push_back(fits[ifit]->Eta4Coeffs(0));
    bEta1.push_back(fits[ifit]->Eta4Coeffs(1));
    aEta2.push_back(fits[ifit]->Eta4Coeffs(2));
    bEta2.push_back(fits[ifit]->Eta4Coeffs(3));
    saEta0.push_back(sqrt(fits[ifit]->Eta2(0,0)));
    sbEta0.push_back(sqrt(fits[ifit]->Eta2(1,1)));
    saEta1.push_back(sqrt(fits[ifit]->Eta4(0,0)));
    sbEta1.push_back(sqrt(fits[ifit]->Eta4(1,1)));
    saEta2.push_back(sqrt(fits[ifit]->Eta4(2,2)));
    sbEta2.push_back(sqrt(fits[ifit]->Eta4(3,3)));

    xEta.push_back(fits[ifit]->average(2));
    sxEta.push_back(fits[ifit]->rms(xEta.back(),2));



  }

  graEta0 = new TGraphErrors ( xEta.size(), &xEta[0], &aEta0[0], &sxEta[0], &saEta0[0]);
  grbEta0 = new TGraphErrors ( xEta.size(), &xEta[0], &bEta0[0], &sxEta[0], &sbEta0[0]);
  // TGraphErrors* grcEta0
  //  = new TGraphErrors ( xEta.size(), &xEta[0], &cEta0[0], &sxEta[0], &scEta0[0]);
  TGraphErrors* graEta1  
    = new TGraphErrors ( xEta.size(), &xEta[0], &aEta1[0], &sxEta[0], &saEta1[0]);
  TGraphErrors* grbEta1
    = new TGraphErrors ( xEta.size(), &xEta[0], &bEta1[0], &sxEta[0], &sbEta1[0]);
  //TGraphErrors* grcEta1
  //  = new TGraphErrors ( xEta.size(), &xEta[0], &cEta1[0], &sxEta[0], &scEta1[0]);
  TGraphErrors* graEta2  
    = new TGraphErrors ( xEta.size(), &xEta[0], &aEta2[0], &sxEta[0], &saEta2[0]);
  TGraphErrors* grbEta2
    = new TGraphErrors ( xEta.size(), &xEta[0], &bEta2[0], &sxEta[0], &sbEta2[0]);
  //TGraphErrors* grcEta2
  //  = new TGraphErrors ( xEta.size(), &xEta[0], &cEta2[0], &sxEta[0], &scEta2[0]);

  TH2F *heta = new TH2F("eta","", 100, 0., 1000., 1000, -1.0, 1.0 );

  c6->cd();
  heta->SetStats(0);
  heta->Draw();

  //TF1* fs = new TF1("fs","[0]+([1]+[2]*x+[3]*x*x+[4]*x*x*x)*exp(-x/[5])",0,1000);
  //fs->SetParameters(0.3,-0.5,0.03,-3E-4,1E-6,60);
  //grs->Fit("fs","W");

  TF1* faEta = new TF1("faEta","[0]+[1]*x+[2]*exp(-x/[3])",0,1000);
  faEta->SetParameters(-0.05,-2E-5,-0.05,150);
  graEta0->Fit("faEta");  
  //graEta0->Fit("faEta","W");

  TF1* fbEta = new TF1("fbEta","[0]+[1]*x+[2]*exp(-x/[3])",0,1000);
  fbEta->SetParameters(0.02,4E-5,0.08,150);
  grbEta0->Fit("fbEta");  
  //grbEta0->Fit("fbEta","W");

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

  /*
  grcEta0->SetMarkerStyle(22);						
  grcEta0->SetMarkerSize(0.1);						
  grcEta0->SetMarkerColor(3);						
  grcEta0->SetLineColor(3);						  
  grcEta0->SetLineWidth(2);						  
  grcEta0->Draw("P");
  */

  c7->cd();
  heta->SetStats(0);
  heta->Draw();

  //TF1* fs = new TF1("fs","[0]+([1]+[2]*x+[3]*x*x+[4]*x*x*x)*exp(-x/[5])",0,1000);
  //fs->SetParameters(0.3,-0.5,0.03,-3E-4,1E-6,60);
  //grs->Fit("fs","W");

  graEta1->SetMarkerStyle(25);						
  graEta1->SetMarkerSize(0.1);						
  graEta1->SetMarkerColor(2);						
  graEta1->SetLineColor(2);						  
  graEta1->SetLineWidth(2);						  
  graEta1->Draw("P");

  grbEta1->SetMarkerStyle(22);						
  grbEta1->SetMarkerSize(0.1);						
  grbEta1->SetMarkerColor(4);						
  grbEta1->SetLineColor(4);						  
  grbEta1->SetLineWidth(2);						  
  grbEta1->Draw("P");

  /*
  grcEta1->SetMarkerStyle(22);						
  grcEta1->SetMarkerSize(0.1);						
  grcEta1->SetMarkerColor(1);						
  grcEta1->SetLineColor(1);						  
  grcEta1->SetLineWidth(2);						  
  grcEta1->Draw("P");
  */

  graEta2->SetMarkerStyle(25);						
  graEta2->SetMarkerSize(0.1);						
  graEta2->SetMarkerColor(3);						
  graEta2->SetLineColor(3);						  
  graEta2->SetLineWidth(2);						  
  graEta2->Draw("P");

  grbEta2->SetMarkerStyle(22);						
  grbEta2->SetMarkerSize(0.1);						
  grbEta2->SetMarkerColor(6);						
  grbEta2->SetLineColor(6);						  
  grbEta2->SetLineWidth(2);						  
  grbEta2->Draw("P");

  /*
  grcEta2->SetMarkerStyle(22);						
  grcEta2->SetMarkerSize(0.1);						
  grcEta2->SetMarkerColor(7);						
  grcEta2->SetLineColor(7);						  
  grcEta2->SetLineWidth(2);						  
  grcEta2->Draw("P");
  */

  TH2F* resultEta = new TH2F("resultEta","Resultat eta",1000,0,1000.,150,-1.5,1.5);
  TH2F* resultEtaE = new TH2F("resultEtaE","Resultat eta",1000,0,1000.,150,-1.5,1.5);
  TH2F* resultEtaH = new TH2F("resultEtaH","Resultat eta",1000,0,1000.,150,-1.5,1.5);
  TH2F* etadep_Cor = new TH2F("etadep_Cor","Eta Dependence",150,0,1.5,150,-1.5,1.5);
  TH2F* etadep_1_10_Cor = new TH2F("etadep_1_10_Cor","Eta Dependence 1 a 10",150,0,1.5,150,-1.5,1.5);
  TH2F* etadep_10_100_Cor = new TH2F("etadep_10_100_Cor","Eta Dependence 1 a 10",150,0,1.5,150,-1.5,1.5);
  TH2F* etadep_100_1000_Cor = new TH2F("etadep_100_1000_Cor","Eta Dependence 1 a 10",150,0,1.5,150,-1.5,1.5);
  TH2F* edep = new TH2F("Edep","E Dependence",100,0,100.,150,-30.,30.);
  TH2F* edep_1_10 = new TH2F("Edep_1_10","E Dependence 1 a 10",100,0,100.,150,-30.,30.);
  TH2F* edep_10_100 = new TH2F("Edep_10_100","E Dependence 10 a 100",100,0,100.,150,-30.,30.);
  TH2F* edep_100_1000 = new TH2F("Edep_100_1000","E Dependence 100 a 1000",100,0,100.,150,-30.,30.);

  TH2F* hdep = new TH2F("Hdep","E Dependence",100,0,100.,150,-30.,30.);
  TH2F* hdep_1_10 = new TH2F("Hdep_1_10","H Dependence 1 a 10",100,0,100.,150,-30,30);
  TH2F* hdep_10_100 = new TH2F("Hdep_10_100","H Dependence 10 a 100",100,0,100.,150,-30,30);
  TH2F* hdep_100_1000 = new TH2F("Hdep_100_1000","H Dependence 100 a 1000",100,0,100.,150,-30,30);

  // Clear fit inputs
  for ( unsigned ifit=0; ifit<fits.size(); ++ifit ) fits[ifit]->clear();
  fits.clear();

  for ( double bin=0.; bin<1.; bin=bin+0.2) { 
    fits.push_back(new Fit(bin,bin+0.2));    
  }

  for ( double bin=1.; bin<10.; bin=bin+1.) { 
    fits.push_back(new Fit(bin,bin+1.));    
  }

  for ( double bin=10.; bin<100.; bin=bin+5.) { 
    fits.push_back(new Fit(bin,bin+5.));    
  }

  for ( double bin=100.; bin<1000.; bin=bin+20.) { 
    fits.push_back(new Fit(bin,bin+20.));    
  }

  for ( unsigned entry=0; entry<nEntries; ++entry ) {
    if ( entry/10000*10000 == entry ) cout << "Process entry " << entry << endl;
    ntuple->LoadTree(entry);
    TT->GetEntry(entry);
    double e = ntuple->Ecal;
    double h = ntuple->Hcal;
    double t = ntuple->E;
    double eta = fabs(ntuple->eta);
    if ( eta > 1.4 ) continue;
    if (e+h < 0.5 ) continue;
    if (t < 1. ) continue;
    // if (e!=0.) continue;
    if (h==0.) continue;

    double a = fa->Eval(t);
    double b = e>0. ? fb->Eval(t) : fc->Eval(t);
    double etaCorr = 1. + faEta->Eval(t) + fbEta->Eval(t)*eta*eta;
    double thresh = e > 0. ? threshE : threshH;
    double eCorr = etaCorr * ( thresh + a*e + b*h );
    // double xEcal = etaCorr * a * e / eCorr; 

    for ( unsigned ifit=0; ifit<fits.size(); ++ifit ) {
      if ( fits[ifit]->fill(etaCorr*a*e,etaCorr*b*h,t,eta,1) ) break;
    }
    
    resultEta->Fill( t, (eCorr-t)/t );
    if ( e>0. ) 
      resultEtaE->Fill( t, (eCorr-t)/t );
    else
      resultEtaH->Fill( t, (eCorr-t)/t );

    etadep_Cor->Fill( fabs(ntuple->eta),  (eCorr-t)/t );
    edep->Fill( etaCorr*a*e, (eCorr-t) );
    hdep->Fill( etaCorr*b*h, (eCorr-t) );
    if ( t < 10 ) {
      etadep_1_10_Cor->Fill( fabs(ntuple->eta),  (eCorr-t)/t );
      edep_1_10->Fill( etaCorr*a*e, (eCorr-t) );
      hdep_1_10->Fill( etaCorr*b*h, (eCorr-t) );
    } else if ( t < 100 ) { 
      etadep_10_100_Cor->Fill( fabs(ntuple->eta),  (eCorr-t)/t);
      edep_10_100->Fill( etaCorr*a*e, (eCorr-t) );
      hdep_10_100->Fill( etaCorr*b*h, (eCorr-t) );
    } else if ( t < 1000 ) { 
      etadep_100_1000_Cor->Fill( fabs(ntuple->eta),  (eCorr-t)/t);
      edep_100_1000->Fill( etaCorr*a*e, (eCorr-t) );
      hdep_100_1000->Fill( etaCorr*b*h, (eCorr-t) );
    }

  }

  /*
  TH2F* resultEta = new TH2F("resultEta","Resultat eta",1000,0,1000.,150,-1.5,1.5);
  TH2F* resultEtaE = new TH2F("resultEtaE","Resultat eta",1000,0,1000.,150,-1.5,1.5);
  TH2F* resultEtaH = new TH2F("resultEtaH","Resultat eta",1000,0,1000.,150,-1.5,1.5);
  TH2F* etadep_Cor = new TH2F("etadep_Cor","Eta Dependence",150,0,1.5,150,-1.5,1.5);
  TH2F* etadep_1_10_Cor = new TH2F("etadep_1_10_Cor","Eta Dependence 1 a 10",150,0,1.5,150,-1.5,1.5);
  TH2F* etadep_10_100_Cor = new TH2F("etadep_10_100_Cor","Eta Dependence 1 a 10",150,0,1.5,150,-1.5,1.5);
  TH2F* etadep_100_1000_Cor = new TH2F("etadep_100_1000_Cor","Eta Dependence 1 a 10",150,0,1.5,150,-1.5,1.5);
  */

  TH2F* xdep = new TH2F("xdep","X Dependence",100,0,1.,150,-1.5,1.5);
  TH2F* xdep_1_10 = new TH2F("xdep_1_10","X Dependence 1 a 10",100,0,1.,150,-1.5,1.5);
  TH2F* xdep_10_100 = new TH2F("xdep_10_100","X Dependence 1 a 10",100,0,1.,150,-1.5,1.5);
  TH2F* xdep_100_1000 = new TH2F("xdep_100_1000","X Dependence 1 a 10",100,0,1.,150,-1.5,1.5);


  // Clear fit inputs
  for ( unsigned ifit=0; ifit<fits.size(); ++ifit ) fits[ifit]->clear();
  fits.clear();

  for ( double bin=0.; bin<10.; bin=bin+2.) { 
    fits.push_back(new Fit(bin,bin+2.));    
  }

  for ( double bin=10.; bin<100.; bin=bin+5.) { 
    fits.push_back(new Fit(bin,bin+5.));    
  }

  for ( double bin=100.; bin<1000.; bin=bin+20.) { 
    fits.push_back(new Fit(bin,bin+20.));    
  }


  for ( unsigned entry=0; entry<nEntries; ++entry ) {
    if ( entry/10000*10000 == entry ) cout << "Process entry " << entry << endl;
    ntuple->LoadTree(entry);
    TT->GetEntry(entry);
    double e = ntuple->Ecal;
    double h = ntuple->Hcal;
    double t = ntuple->E;
    double eta = fabs(ntuple->eta);
    if ( eta > 1.4 ) continue;
    if (e+h < 0.5 ) continue;
    if (t < 1. ) continue;
    // if (e!=0.) continue;
    if (h==0.) continue;

    double a = fa->Eval(t);
    double b = e>0. ? fb->Eval(t) : fc->Eval(t);
    double etaCorr = 1. + faEta->Eval(t) + fbEta->Eval(t)*eta*eta;
    double eCorr = etaCorr * ( a*e + b*h );
    double xEcal = etaCorr * a * e / eCorr; 

    for ( unsigned ifit=0; ifit<fits.size(); ++ifit ) {
      if ( fits[ifit]->fill(etaCorr*a*e,etaCorr*b*h,t,eta) ) break;
    }
    
    /*
    resultEta->Fill( t, (eCorr-t)/t );
    if ( e>0. ) 
      resultEtaE->Fill( t, (eCorr-t)/t );
    else
      resultEtaH->Fill( t, (eCorr-t)/t );

    etadep_Cor->Fill( fabs(ntuple->eta),  (eCorr-t)/t );
    */
    xdep->Fill( xEcal,  (eCorr-t)/t );
    if ( t < 10 ) {
      // etadep_1_10_Cor->Fill( fabs(ntuple->eta),  (eCorr-t)/t );
      xdep_1_10->Fill( xEcal,  (eCorr-t)/t );
    } else if ( t < 100 ) { 
      // etadep_10_100_Cor->Fill( fabs(ntuple->eta),  (eCorr-t)/t);
      xdep_10_100->Fill( xEcal,  (eCorr-t)/t );
    } else if ( t < 1000 ) { 
      // etadep_100_1000_Cor->Fill( fabs(ntuple->eta),  (eCorr-t)/t);
      xdep_100_1000->Fill( xEcal,  (eCorr-t)/t );
    }

  }

  vector<double> aX, bX, cX, saX, sbX, scX;
  vector<double> xX, sxX;
  for ( unsigned ifit=2; ifit<fits.size()-1; ++ifit ) {
    bool invert = fits[ifit]->xMatrices();
    cout << "Bin " << ifit 
	 << "; a, b, c = " 
	 << "   " << fits[ifit]->X3Coeffs(0) 
	 << " +/- " << sqrt(fits[ifit]->X3(0,0))
	 << ", "  << fits[ifit]->X3Coeffs(1)
	 << " +/- " << sqrt(fits[ifit]->X3(1,1))
	 << ", "  << fits[ifit]->X3Coeffs(2)
	 << " +/- " << sqrt(fits[ifit]->X3(2,2))
	 << endl;

    aX.push_back(1.*fits[ifit]->X3Coeffs(0));
    bX.push_back(1.*fits[ifit]->X3Coeffs(1));
    cX.push_back(1.*fits[ifit]->X3Coeffs(2));
    saX.push_back(1.*sqrt(fits[ifit]->X3(0,0)));
    sbX.push_back(1.*sqrt(fits[ifit]->X3(1,1)));
    scX.push_back(1.*sqrt(fits[ifit]->X3(2,2)));

    xX.push_back(fits[ifit]->average(0));
    sxX.push_back(fits[ifit]->rms(xEta.back(),0));

  }

  graX = new TGraphErrors ( xX.size(), &xX[0], &aX[0], &sxX[0], &saX[0]);
  grbX = new TGraphErrors ( xX.size(), &xX[0], &bX[0], &sxX[0], &sbX[0]);
  grcX = new TGraphErrors ( xX.size(), &xX[0], &cX[0], &sxX[0], &scX[0]);

  TH2F *hx = new TH2F("X","", 100, 0., 1000., 1000, -1.0, 1.0 );

  c8->cd();
  hx->SetStats(0);
  hx->Draw();

  TF1* faX = new TF1("faX","[0]*exp(-x/[1])+[2]",2,1000);
  faX->SetParameters(0.02,10);
  graX->Fit("faX","W","",3.,1000.);  
  //graX->Fit("faX","W","",3.,1000.);  
  TF1* fbX = new TF1("fbX","[0]*exp(-x/[1])+[2]*exp(-x/[3])",2,1000);
  fbX->SetParameters(-0.0006,10,-0.02,1);
  grbX->Fit("fbX","W","",3.,1000.);  
  TF1* fcX = new TF1("fcX","[0]/(x*x)+[1]*exp(-x/[2])",2,1000);
  fcX->SetParameters(-0.5,-0.05,5);
  grcX->Fit("fcX","","",3.,1000.);  

  graX->SetMarkerStyle(25);						
  graX->SetMarkerSize(0.1);						
  graX->SetMarkerColor(2);						
  graX->SetLineColor(2);						  
  graX->SetLineWidth(2);						  
  graX->Draw("P");

  grbX->SetMarkerStyle(22);						
  grbX->SetMarkerSize(0.1);						
  grbX->SetMarkerColor(4);						
  grbX->SetLineColor(4);						  
  grbX->SetLineWidth(2);						  
  grbX->Draw("P");

  grcX->SetMarkerStyle(22);						
  grcX->SetMarkerSize(0.1);						
  grcX->SetMarkerColor(3);						
  grcX->SetLineColor(3);						  
  grcX->SetLineWidth(2);						  
  grcX->Draw("P");


  TH2F* resultX = new TH2F("resultX","Resultat X",1000,0,1000.,150,-1.5,1.5);
  TH2F* resultXE = new TH2F("resultXE","Resultat X",1000,0,1000.,150,-1.5,1.5);
  TH2F* resultXH = new TH2F("resultXH","Resultat X",1000,0,1000.,150,-1.5,1.5);
  TH2F* etadep_XCor = new TH2F("etadep_XCor","X Dependence",150,0,1.5,150,-1.5,1.5);
  TH2F* etadep_1_10_XCor = new TH2F("etadep_1_10_XCor","X Dependence 1 a 10",150,0,1.5,150,-1.5,1.5);
  TH2F* etadep_10_100_XCor = new TH2F("etadep_10_100_XCor","X Dependence 1 a 10",150,0,1.5,150,-1.5,1.5);
  TH2F* etadep_100_1000_XCor = new TH2F("etadep_100_1000_XCor","X Dependence 1 a 10",150,0,1.5,150,-1.5,1.5);
  TH2F* xdep_XCor = new TH2F("xdep_XCor","X Dependence",100,0,1.,150,-1.5,1.5);
  TH2F* xdep_1_10_XCor = new TH2F("xdep_1_10_XCor","X Dependence 1 a 10",100,0,1.,150,-1.5,1.5);
  TH2F* xdep_10_100_XCor = new TH2F("xdep_10_100_XCor","X Dependence 1 a 10",100,0,1.,150,-1.5,1.5);
  TH2F* xdep_100_1000_XCor = new TH2F("xdep_100_1000_XCor","X Dependence 1 a 10",100,0,1.,150,-1.5,1.5);

  for ( unsigned entry=0; entry<nEntries; ++entry ) {
    if ( entry/10000*10000 == entry ) cout << "Process entry " << entry << endl;
    ntuple->LoadTree(entry);
    TT->GetEntry(entry);
    double e = ntuple->Ecal;
    double h = ntuple->Hcal;
    double t = ntuple->E;
    double eta = fabs(ntuple->eta);
    if ( eta > 1.4 ) continue;
    if (e+h< 0.5 ) continue;
    if (t < 1. ) continue;
    // if (e!=0.) continue;
    if (h==0.) continue;

    double a = fa->Eval(t);
    double b = e>0. ? fb->Eval(t) : fc->Eval(t);
    double etaCorr = 1. + faEta->Eval(t) + fbEta->Eval(t)*eta*eta;
    double xEcal = a * e / (a * e + b * h) ; 
    double xCorr = xEcal != 0. && xEcal != 1. ? 
      1. + faX->Eval(t) + fbX->Eval(t)/(xEcal*xEcal) + fcX->Eval(t)/((1-xEcal)*(1.-xEcal)) : 1.; 
    double thresh = e > 0. ? threshE : threshH;
    double eCorr = xCorr * etaCorr * ( thresh + a*e + b*h );

    /*
    for ( unsigned ifit=0; ifit<fits.size(); ++ifit ) {
      if ( fits[ifit]->fill(a*e,b*h,t,eta) ) break;
    }
    */

    resultX->Fill( t, (eCorr-t)/t );
    if ( e>0. ) 
      resultXE->Fill( t, (eCorr-t)/t );
    else
      resultXH->Fill( t, (eCorr-t)/t );

    etadep_XCor->Fill( fabs(ntuple->eta),  (eCorr-t)/t );
    xdep_XCor->Fill( xEcal,  (eCorr-t)/t );
    if ( t < 10 ) {
      etadep_1_10_XCor->Fill( fabs(ntuple->eta),  (eCorr-t)/t );
      xdep_1_10_XCor->Fill( xEcal,  (eCorr-t)/t );
    } else if ( t < 100 ) { 
      etadep_10_100_XCor->Fill( fabs(ntuple->eta),  (eCorr-t)/t);
      xdep_10_100_XCor->Fill( xEcal,  (eCorr-t)/t );
    } else if ( t < 1000 ) { 
      etadep_100_1000_XCor->Fill( fabs(ntuple->eta),  (eCorr-t)/t);
      xdep_100_1000_XCor->Fill( xEcal,  (eCorr-t)/t );
    }

  }

}
 
void calib() {
  computeBarrelCoefficients("calib_130.txt");
}
