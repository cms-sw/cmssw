//////////////////////////////////////////////////////////
// This class has been automatically generated on
// Fri Apr 27 12:47:47 2007 by ROOT version 5.12/00
// from TTree sim/simulated showers
// found on file: protons_150gev.root
//////////////////////////////////////////////////////////

#ifndef sim_h
#define sim_h

#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>
const Int_t kMaxshower = 1;
const Int_t kMaxparticle_ = 100000;
const Int_t kMaxlong = 10000;
const Int_t kMaxcerenkov = 1;

class sim {
public:
  TTree *fChain;   //!pointer to the analyzed TTree or TChain
  Int_t fCurrent;  //!current Tree number in a TChain

  // Declaration of leave types
  //crsIO::TShower  *shower.;
  UInt_t shower_TObject_fUniqueID;
  UInt_t shower_TObject_fBits;
  Int_t shower_EventID;
  Float_t shower_Energy;
  Float_t shower_StartingAltitude;
  Int_t shower_FirstTarget;
  Float_t shower_FirstHeight;
  Float_t shower_Theta;
  Float_t shower_Phi;
  Int_t shower_RandomSeed[10];
  Int_t shower_RandomOffset[10];
  Float_t shower_nPhotons;
  Float_t shower_nElectrons;
  Float_t shower_nHadrons;
  Float_t shower_nMuons;
  Int_t shower_nParticlesWritten;
  Int_t shower_nPhotonsWritten;
  Int_t shower_nElectronsWritten;
  Int_t shower_nHadronsWritten;
  Int_t shower_nMuonsWritten;
  Float_t shower_GH_Nmax;
  Float_t shower_GH_t0;
  Float_t shower_GH_tmax;
  Float_t shower_GH_a;
  Float_t shower_GH_b;
  Float_t shower_GH_c;
  Float_t shower_GH_Chi2;
  Int_t shower_nPreshower;
  Int_t shower_CPUtime;
  Int_t particle__;
  UInt_t particle__fUniqueID[kMaxparticle_];          //[particle._]
  UInt_t particle__fBits[kMaxparticle_];              //[particle._]
  Int_t particle__ParticleID[kMaxparticle_];          //[particle._]
  Int_t particle__ObservationLevel[kMaxparticle_];    //[particle._]
  Int_t particle__HadronicGeneration[kMaxparticle_];  //[particle._]
  Double_t particle__Px[kMaxparticle_];               //[particle._]
  Double_t particle__Py[kMaxparticle_];               //[particle._]
  Double_t particle__Pz[kMaxparticle_];               //[particle._]
  Double_t particle__x[kMaxparticle_];                //[particle._]
  Double_t particle__y[kMaxparticle_];                //[particle._]
  Double_t particle__Time[kMaxparticle_];             //[particle._]
  Double_t particle__Weight[kMaxparticle_];           //[particle._]
  Int_t long_;
  UInt_t long_fUniqueID[kMaxlong];      //[long_]
  UInt_t long_fBits[kMaxlong];          //[long_]
  Float_t long_Depth[kMaxlong];         //[long_]
  ULong64_t long_nGammas[kMaxlong];     //[long_]
  ULong64_t long_nElectrons[kMaxlong];  //[long_]
  ULong64_t long_nPositrons[kMaxlong];  //[long_]
  ULong64_t long_nMuons[kMaxlong];      //[long_]
  ULong64_t long_nAntiMuons[kMaxlong];  //[long_]
  ULong64_t long_nHadrons[kMaxlong];    //[long_]
  ULong64_t long_nCharged[kMaxlong];    //[long_]
  ULong64_t long_nNuclei[kMaxlong];     //[long_]
  ULong64_t long_nCerenkov[kMaxlong];   //[long_]
  Int_t cerenkov_;
  UInt_t cerenkov_fUniqueID[kMaxcerenkov];          //[cerenkov_]
  UInt_t cerenkov_fBits[kMaxcerenkov];              //[cerenkov_]
  Float_t cerenkov_nPhotons[kMaxcerenkov];          //[cerenkov_]
  Float_t cerenkov_x[kMaxcerenkov];                 //[cerenkov_]
  Float_t cerenkov_y[kMaxcerenkov];                 //[cerenkov_]
  Float_t cerenkov_u[kMaxcerenkov];                 //[cerenkov_]
  Float_t cerenkov_v[kMaxcerenkov];                 //[cerenkov_]
  Float_t cerenkov_Time[kMaxcerenkov];              //[cerenkov_]
  Float_t cerenkov_ProductionHeight[kMaxcerenkov];  //[cerenkov_]
  Float_t cerenkov_Weight[kMaxcerenkov];            //[cerenkov_]

  // List of branches
  TBranch *b_shower_TObject_fUniqueID;      //!
  TBranch *b_shower_TObject_fBits;          //!
  TBranch *b_shower_EventID;                //!
  TBranch *b_shower_Energy;                 //!
  TBranch *b_shower_StartingAltitude;       //!
  TBranch *b_shower_FirstTarget;            //!
  TBranch *b_shower_FirstHeight;            //!
  TBranch *b_shower_Theta;                  //!
  TBranch *b_shower_Phi;                    //!
  TBranch *b_shower_RandomSeed;             //!
  TBranch *b_shower_RandomOffset;           //!
  TBranch *b_shower_nPhotons;               //!
  TBranch *b_shower_nElectrons;             //!
  TBranch *b_shower_nHadrons;               //!
  TBranch *b_shower_nMuons;                 //!
  TBranch *b_shower_nParticlesWritten;      //!
  TBranch *b_shower_nPhotonsWritten;        //!
  TBranch *b_shower_nElectronsWritten;      //!
  TBranch *b_shower_nHadronsWritten;        //!
  TBranch *b_shower_nMuonsWritten;          //!
  TBranch *b_shower_GH_Nmax;                //!
  TBranch *b_shower_GH_t0;                  //!
  TBranch *b_shower_GH_tmax;                //!
  TBranch *b_shower_GH_a;                   //!
  TBranch *b_shower_GH_b;                   //!
  TBranch *b_shower_GH_c;                   //!
  TBranch *b_shower_GH_Chi2;                //!
  TBranch *b_shower_nPreshower;             //!
  TBranch *b_shower_CPUtime;                //!
  TBranch *b_particle__;                    //!
  TBranch *b_particle__fUniqueID;           //!
  TBranch *b_particle__fBits;               //!
  TBranch *b_particle__ParticleID;          //!
  TBranch *b_particle__ObservationLevel;    //!
  TBranch *b_particle__HadronicGeneration;  //!
  TBranch *b_particle__Px;                  //!
  TBranch *b_particle__Py;                  //!
  TBranch *b_particle__Pz;                  //!
  TBranch *b_particle__x;                   //!
  TBranch *b_particle__y;                   //!
  TBranch *b_particle__Time;                //!
  TBranch *b_particle__Weight;              //!
  TBranch *b_long_;                         //!
  TBranch *b_long_fUniqueID;                //!
  TBranch *b_long_fBits;                    //!
  TBranch *b_long_Depth;                    //!
  TBranch *b_long_nGammas;                  //!
  TBranch *b_long_nElectrons;               //!
  TBranch *b_long_nPositrons;               //!
  TBranch *b_long_nMuons;                   //!
  TBranch *b_long_nAntiMuons;               //!
  TBranch *b_long_nHadrons;                 //!
  TBranch *b_long_nCharged;                 //!
  TBranch *b_long_nNuclei;                  //!
  TBranch *b_long_nCerenkov;                //!
  TBranch *b_cerenkov_;                     //!
  TBranch *b_cerenkov_fUniqueID;            //!
  TBranch *b_cerenkov_fBits;                //!
  TBranch *b_cerenkov_nPhotons;             //!
  TBranch *b_cerenkov_x;                    //!
  TBranch *b_cerenkov_y;                    //!
  TBranch *b_cerenkov_u;                    //!
  TBranch *b_cerenkov_v;                    //!
  TBranch *b_cerenkov_Time;                 //!
  TBranch *b_cerenkov_ProductionHeight;     //!
  TBranch *b_cerenkov_Weight;               //!

  sim(TTree *tree = nullptr);
  virtual ~sim();
  virtual Int_t Cut(Long64_t entry);
  virtual Int_t GetEntry(Long64_t entry);
  virtual Long64_t LoadTree(Long64_t entry);
  virtual void Init(TTree *tree);
  //virtual void     Loop();
  virtual Bool_t Notify();
  virtual void Show(Long64_t entry = -1);
};

#endif

#ifdef sim_cxx
inline sim::sim(TTree *tree) {
  // if parameter tree is not specified (or zero), connect the file
  // used to generate this class and read the Tree.
  if (tree == nullptr) {
    std::cout << "sim::sim: tree = 0" << std::endl;
    TFile *f = (TFile *)gROOT->GetListOfFiles()->FindObject("protons_150gev.root");
    if (!f) {
      f = new TFile("protons_150gev.root");
    }
    tree = (TTree *)gDirectory->Get("sim");
  }
  //else std::cout << "sim::sim: tree != 0 => Alright!" << std::endl;
  Init(tree);
}

inline sim::~sim() {
  if (!fChain)
    return;
  delete fChain->GetCurrentFile();
}

inline Int_t sim::GetEntry(Long64_t entry) {
  // Read contents of entry.
  if (!fChain)
    return 0;
  return fChain->GetEntry(entry);
}
inline Long64_t sim::LoadTree(Long64_t entry) {
  // Set the environment to read one entry
  std::cout << "sim::LoadTree: " << std::endl;
  if (fChain)
    std::cout << " fChain<>0" << std::endl;
  else
    std::cout << " fChain=0" << std::endl;
  if (!fChain)
    return -5;
  Long64_t centry = fChain->LoadTree(entry);
  if (centry < 0)
    return centry;
  if (fChain->IsA() != TChain::Class())
    return centry;
  TChain *chain = (TChain *)fChain;
  if (chain->GetTreeNumber() != fCurrent) {
    fCurrent = chain->GetTreeNumber();
    Notify();
  }
  return centry;
}

inline void sim::Init(TTree *tree) {
  // The Init() function is called when the selector needs to initialize
  // a new tree or chain. Typically here the branch addresses and branch
  // pointers of the tree will be set.
  // It is normaly not necessary to make changes to the generated
  // code, but the routine can be extended by the user if needed.
  // Init() will be called many times when running on PROOF
  // (once per file to be processed).

  // Set branch addresses and branch pointers
  if (!tree)
    return;
  fChain = tree;
  fCurrent = -1;
  fChain->SetMakeClass(1);

  fChain->SetBranchAddress("shower.TObject.fUniqueID", &shower_TObject_fUniqueID, &b_shower_TObject_fUniqueID);
  fChain->SetBranchAddress("shower.TObject.fBits", &shower_TObject_fBits, &b_shower_TObject_fBits);
  fChain->SetBranchAddress("shower.EventID", &shower_EventID, &b_shower_EventID);
  fChain->SetBranchAddress("shower.Energy", &shower_Energy, &b_shower_Energy);
  fChain->SetBranchAddress("shower.StartingAltitude", &shower_StartingAltitude, &b_shower_StartingAltitude);
  fChain->SetBranchAddress("shower.FirstTarget", &shower_FirstTarget, &b_shower_FirstTarget);
  fChain->SetBranchAddress("shower.FirstHeight", &shower_FirstHeight, &b_shower_FirstHeight);
  fChain->SetBranchAddress("shower.Theta", &shower_Theta, &b_shower_Theta);
  fChain->SetBranchAddress("shower.Phi", &shower_Phi, &b_shower_Phi);
  fChain->SetBranchAddress("shower.RandomSeed[10]", shower_RandomSeed, &b_shower_RandomSeed);
  fChain->SetBranchAddress("shower.RandomOffset[10]", shower_RandomOffset, &b_shower_RandomOffset);
  fChain->SetBranchAddress("shower.nPhotons", &shower_nPhotons, &b_shower_nPhotons);
  fChain->SetBranchAddress("shower.nElectrons", &shower_nElectrons, &b_shower_nElectrons);
  fChain->SetBranchAddress("shower.nHadrons", &shower_nHadrons, &b_shower_nHadrons);
  fChain->SetBranchAddress("shower.nMuons", &shower_nMuons, &b_shower_nMuons);
  fChain->SetBranchAddress("shower.nParticlesWritten", &shower_nParticlesWritten, &b_shower_nParticlesWritten);
  fChain->SetBranchAddress("shower.nPhotonsWritten", &shower_nPhotonsWritten, &b_shower_nPhotonsWritten);
  fChain->SetBranchAddress("shower.nElectronsWritten", &shower_nElectronsWritten, &b_shower_nElectronsWritten);
  fChain->SetBranchAddress("shower.nHadronsWritten", &shower_nHadronsWritten, &b_shower_nHadronsWritten);
  fChain->SetBranchAddress("shower.nMuonsWritten", &shower_nMuonsWritten, &b_shower_nMuonsWritten);
  fChain->SetBranchAddress("shower.GH_Nmax", &shower_GH_Nmax, &b_shower_GH_Nmax);
  fChain->SetBranchAddress("shower.GH_t0", &shower_GH_t0, &b_shower_GH_t0);
  fChain->SetBranchAddress("shower.GH_tmax", &shower_GH_tmax, &b_shower_GH_tmax);
  fChain->SetBranchAddress("shower.GH_a", &shower_GH_a, &b_shower_GH_a);
  fChain->SetBranchAddress("shower.GH_b", &shower_GH_b, &b_shower_GH_b);
  fChain->SetBranchAddress("shower.GH_c", &shower_GH_c, &b_shower_GH_c);
  fChain->SetBranchAddress("shower.GH_Chi2", &shower_GH_Chi2, &b_shower_GH_Chi2);
  fChain->SetBranchAddress("shower.nPreshower", &shower_nPreshower, &b_shower_nPreshower);
  fChain->SetBranchAddress("shower.CPUtime", &shower_CPUtime, &b_shower_CPUtime);
  fChain->SetBranchAddress("particle.", &particle__, &b_particle__);
  fChain->SetBranchAddress("particle..fUniqueID", particle__fUniqueID, &b_particle__fUniqueID);
  fChain->SetBranchAddress("particle..fBits", particle__fBits, &b_particle__fBits);
  fChain->SetBranchAddress("particle..ParticleID", particle__ParticleID, &b_particle__ParticleID);
  fChain->SetBranchAddress("particle..ObservationLevel", particle__ObservationLevel, &b_particle__ObservationLevel);
  fChain->SetBranchAddress(
      "particle..HadronicGeneration", particle__HadronicGeneration, &b_particle__HadronicGeneration);
  fChain->SetBranchAddress("particle..Px", particle__Px, &b_particle__Px);
  fChain->SetBranchAddress("particle..Py", particle__Py, &b_particle__Py);
  fChain->SetBranchAddress("particle..Pz", particle__Pz, &b_particle__Pz);
  fChain->SetBranchAddress("particle..x", particle__x, &b_particle__x);
  fChain->SetBranchAddress("particle..y", particle__y, &b_particle__y);
  fChain->SetBranchAddress("particle..Time", particle__Time, &b_particle__Time);
  fChain->SetBranchAddress("particle..Weight", particle__Weight, &b_particle__Weight);
  fChain->SetBranchAddress("long", &long_, &b_long_);
  fChain->SetBranchAddress("long.fUniqueID", long_fUniqueID, &b_long_fUniqueID);
  fChain->SetBranchAddress("long.fBits", long_fBits, &b_long_fBits);
  fChain->SetBranchAddress("long.Depth", long_Depth, &b_long_Depth);
  fChain->SetBranchAddress("long.nGammas", long_nGammas, &b_long_nGammas);
  fChain->SetBranchAddress("long.nElectrons", long_nElectrons, &b_long_nElectrons);
  fChain->SetBranchAddress("long.nPositrons", long_nPositrons, &b_long_nPositrons);
  fChain->SetBranchAddress("long.nMuons", long_nMuons, &b_long_nMuons);
  fChain->SetBranchAddress("long.nAntiMuons", long_nAntiMuons, &b_long_nAntiMuons);
  fChain->SetBranchAddress("long.nHadrons", long_nHadrons, &b_long_nHadrons);
  fChain->SetBranchAddress("long.nCharged", long_nCharged, &b_long_nCharged);
  fChain->SetBranchAddress("long.nNuclei", long_nNuclei, &b_long_nNuclei);
  fChain->SetBranchAddress("long.nCerenkov", long_nCerenkov, &b_long_nCerenkov);
  fChain->SetBranchAddress("cerenkov", &cerenkov_, &b_cerenkov_);
  fChain->SetBranchAddress("cerenkov.fUniqueID", &cerenkov_fUniqueID, &b_cerenkov_fUniqueID);
  fChain->SetBranchAddress("cerenkov.fBits", &cerenkov_fBits, &b_cerenkov_fBits);
  fChain->SetBranchAddress("cerenkov.nPhotons", &cerenkov_nPhotons, &b_cerenkov_nPhotons);
  fChain->SetBranchAddress("cerenkov.x", &cerenkov_x, &b_cerenkov_x);
  fChain->SetBranchAddress("cerenkov.y", &cerenkov_y, &b_cerenkov_y);
  fChain->SetBranchAddress("cerenkov.u", &cerenkov_u, &b_cerenkov_u);
  fChain->SetBranchAddress("cerenkov.v", &cerenkov_v, &b_cerenkov_v);
  fChain->SetBranchAddress("cerenkov.Time", &cerenkov_Time, &b_cerenkov_Time);
  fChain->SetBranchAddress("cerenkov.ProductionHeight", &cerenkov_ProductionHeight, &b_cerenkov_ProductionHeight);
  fChain->SetBranchAddress("cerenkov.Weight", &cerenkov_Weight, &b_cerenkov_Weight);
  Notify();
}

inline Bool_t sim::Notify() {
  // The Notify() function is called when a new file is opened. This
  // can be either for a new TTree in a TChain or when when a new TTree
  // is started when using PROOF. It is normaly not necessary to make changes
  // to the generated code, but the routine can be extended by the
  // user if needed. The return value is currently not used.

  return kTRUE;
}

inline void sim::Show(Long64_t entry) {
  // Print contents of entry.
  // If entry is not specified, print current entry
  if (!fChain)
    return;
  fChain->Show(entry);
}
inline Int_t sim::Cut(Long64_t entry) {
  // This function may be called from Loop.
  // returns  1 if entry is accepted.
  // returns -1 otherwise.
  return 1;
}
#endif  // #ifdef sim_cxx
