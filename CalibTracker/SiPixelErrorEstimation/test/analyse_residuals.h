//////////////////////////////////////////////////////////
// This class has been automatically generated on
// Fri Jul  6 20:43:01 2007 by ROOT version 5.14/00e
// from TChain TrackHitNtuple/
//////////////////////////////////////////////////////////

#ifndef analyse_residuals_h
#define analyse_residuals_h

#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>

class analyse_residuals {
public :
   TTree          *fChain;   //!pointer to the analyzed TTree or TChain
   Int_t           fCurrent; //!current Tree number in a TChain

   // Declaration of leave types
   Int_t           evt;
   Int_t           run;
   Int_t           subdetId;
   Int_t           layer;
   Int_t           ladder;
   Int_t           mod;
   Int_t           side;
   Int_t           disk;
   Int_t           blade;
   Int_t           panel;
   Int_t           plaq;
   Int_t           half;
   Int_t           flipped;
   Float_t         rechitx;
   Float_t         rechity;
   Float_t         rechitz;
   Float_t         rechiterrx;
   Float_t         rechiterry;
   Float_t         rechitresx;
   Float_t         rechitresy;
   Float_t         rechitpullx;
   Float_t         rechitpully;
   Int_t           npix;
   Int_t           nxpix;
   Int_t           nypix;
   Float_t         charge;
   Int_t           edgex;
   Int_t           edgey;
   Int_t           bigx;
   Int_t           bigy;
   Float_t         alpha;
   Float_t         beta;
   Float_t         trk_alpha;
   Float_t         trk_beta;
   Float_t         phi;
   Float_t         eta;
   Float_t         simhitx;
   Float_t         simhity;
   Int_t           nsimhit;
   Int_t           pidhit;
   Int_t           simproc;

   // List of branches
   TBranch        *b_evt;   //!
   TBranch        *b_run;   //!
   TBranch        *b_subdetId;   //!
   TBranch        *b_layer;   //!
   TBranch        *b_ladder;   //!
   TBranch        *b_mod;   //!
   TBranch        *b_side;   //!
   TBranch        *b_disk;   //!
   TBranch        *b_blade;   //!
   TBranch        *b_panel;   //!
   TBranch        *b_plaq;   //!
   TBranch        *b_half;   //!
   TBranch        *b_flipped;   //!
   TBranch        *b_rechitx;   //!
   TBranch        *b_rechity;   //!
   TBranch        *b_rechitz;   //!
   TBranch        *b_rechiterrx;   //!
   TBranch        *b_rechiterry;   //!
   TBranch        *b_rechitresx;   //!
   TBranch        *b_rechitresy;   //!
   TBranch        *b_rechitpullx;   //!
   TBranch        *b_rechitpully;   //!
   TBranch        *b_npix;   //!
   TBranch        *b_nxpix;   //!
   TBranch        *b_nypix;   //!
   TBranch        *b_charge;   //!
   TBranch        *b_edgex;   //!
   TBranch        *b_edgey;   //!
   TBranch        *b_bigx;   //!
   TBranch        *b_bigy;   //!
   TBranch        *b_alpha;   //!
   TBranch        *b_beta;   //!
   TBranch        *b_trk_alpha;   //!
   TBranch        *b_trk_beta;   //!
   TBranch        *b_phi;   //!
   TBranch        *b_eta;   //!
   TBranch        *b_simhitx;   //!
   TBranch        *b_simhity;   //!
   TBranch        *b_nsimhit;   //!
   TBranch        *b_pidhit;   //!
   TBranch        *b_simproc;   //!

   analyse_residuals(TTree *tree=0);
   virtual ~analyse_residuals();
   virtual Int_t    Cut(Long64_t entry);
   virtual Int_t    GetEntry(Long64_t entry);
   virtual Long64_t LoadTree(Long64_t entry);
   virtual void     Init(TTree *tree);
   virtual void     Loop();
   virtual Bool_t   Notify();
   virtual void     Show(Long64_t entry = -1);
};

#endif

#ifdef analyse_residuals_cxx
analyse_residuals::analyse_residuals(TTree *tree)
{
// if parameter tree is not specified (or zero), connect the file
// used to generate this class and read the Tree.
   if (tree == 0) {

#ifdef SINGLE_TREE
      // The following code should be used if you want this class to access
      // a single tree instead of a chain
      TFile *f = (TFile*)gROOT->GetListOfFiles()->FindObject("Memory Directory");
      if (!f) {
         f = new TFile("Memory Directory");
         f->cd("Rint:/");
      }
      tree = (TTree*)gDirectory->Get("TrackHitNtuple");

#else // SINGLE_TREE

      // The following code should be used if you want this class to access a chain
      // of trees.
      TChain * chain = new TChain("TrackHitNtuple","");
   
      chain->Add("/uscms_data/d1/ggiurgiu/SiPixelErrorEstimation_Ntuple_after4.root/TrackHitNtuple");

      tree = chain;
#endif // SINGLE_TREE

   }
   Init(tree);
}

analyse_residuals::~analyse_residuals()
{
   if (!fChain) return;
   delete fChain->GetCurrentFile();
}

Int_t analyse_residuals::GetEntry(Long64_t entry)
{
// Read contents of entry.
   if (!fChain) return 0;
   return fChain->GetEntry(entry);
}
Long64_t analyse_residuals::LoadTree(Long64_t entry)
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

void analyse_residuals::Init(TTree *tree)
{
   // The Init() function is called when the selector needs to initialize
   // a new tree or chain. Typically here the branch addresses and branch
   // pointers of the tree will be set.
   // It is normaly not necessary to make changes to the generated
   // code, but the routine can be extended by the user if needed.
   // Init() will be called many times when running on PROOF
   // (once per file to be processed).

   // Set branch addresses and branch pointers
   if (!tree) return;
   fChain = tree;
   fCurrent = -1;
   fChain->SetMakeClass(1);

   fChain->SetBranchAddress("evt", &evt, &b_evt);
   fChain->SetBranchAddress("run", &run, &b_run);
   fChain->SetBranchAddress("subdetId", &subdetId, &b_subdetId);
   fChain->SetBranchAddress("layer", &layer, &b_layer);
   fChain->SetBranchAddress("ladder", &ladder, &b_ladder);
   fChain->SetBranchAddress("mod", &mod, &b_mod);
   fChain->SetBranchAddress("side", &side, &b_side);
   fChain->SetBranchAddress("disk", &disk, &b_disk);
   fChain->SetBranchAddress("blade", &blade, &b_blade);
   fChain->SetBranchAddress("panel", &panel, &b_panel);
   fChain->SetBranchAddress("plaq", &plaq, &b_plaq);
   fChain->SetBranchAddress("half", &half, &b_half);
   fChain->SetBranchAddress("flipped", &flipped, &b_flipped);
   fChain->SetBranchAddress("rechitx", &rechitx, &b_rechitx);
   fChain->SetBranchAddress("rechity", &rechity, &b_rechity);
   fChain->SetBranchAddress("rechitz", &rechitz, &b_rechitz);
   fChain->SetBranchAddress("rechiterrx", &rechiterrx, &b_rechiterrx);
   fChain->SetBranchAddress("rechiterry", &rechiterry, &b_rechiterry);
   fChain->SetBranchAddress("rechitresx", &rechitresx, &b_rechitresx);
   fChain->SetBranchAddress("rechitresy", &rechitresy, &b_rechitresy);
   fChain->SetBranchAddress("rechitpullx", &rechitpullx, &b_rechitpullx);
   fChain->SetBranchAddress("rechitpully", &rechitpully, &b_rechitpully);
   fChain->SetBranchAddress("npix", &npix, &b_npix);
   fChain->SetBranchAddress("nxpix", &nxpix, &b_nxpix);
   fChain->SetBranchAddress("nypix", &nypix, &b_nypix);
   fChain->SetBranchAddress("charge", &charge, &b_charge);
   fChain->SetBranchAddress("edgex", &edgex, &b_edgex);
   fChain->SetBranchAddress("edgey", &edgey, &b_edgey);
   fChain->SetBranchAddress("bigx", &bigx, &b_bigx);
   fChain->SetBranchAddress("bigy", &bigy, &b_bigy);
   fChain->SetBranchAddress("alpha", &alpha, &b_alpha);
   fChain->SetBranchAddress("beta", &beta, &b_beta);
   fChain->SetBranchAddress("trk_alpha", &trk_alpha, &b_trk_alpha);
   fChain->SetBranchAddress("trk_beta", &trk_beta, &b_trk_beta);
   fChain->SetBranchAddress("phi", &phi, &b_phi);
   fChain->SetBranchAddress("eta", &eta, &b_eta);
   fChain->SetBranchAddress("simhitx", &simhitx, &b_simhitx);
   fChain->SetBranchAddress("simhity", &simhity, &b_simhity);
   fChain->SetBranchAddress("nsimhit", &nsimhit, &b_nsimhit);
   fChain->SetBranchAddress("pidhit", &pidhit, &b_pidhit);
   fChain->SetBranchAddress("simproc", &simproc, &b_simproc);
   Notify();
}

Bool_t analyse_residuals::Notify()
{
   // The Notify() function is called when a new file is opened. This
   // can be either for a new TTree in a TChain or when when a new TTree
   // is started when using PROOF. It is normaly not necessary to make changes
   // to the generated code, but the routine can be extended by the
   // user if needed. The return value is currently not used.

   return kTRUE;
}

void analyse_residuals::Show(Long64_t entry)
{
// Print contents of entry.
// If entry is not specified, print current entry
   if (!fChain) return;
   fChain->Show(entry);
}
Int_t analyse_residuals::Cut(Long64_t entry)
{
// This function may be called from Loop.
// returns  1 if entry is accepted.
// returns -1 otherwise.
   return 1;
}
#endif // #ifdef analyse_residuals_cxx
