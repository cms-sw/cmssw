#include "TreeAnalysisReadGen.h"
#include <TH2.h>
#include <TStyle.h>
#include <TCanvas.h>
#include <vector>

Bool_t FillChain(TChain *chain, const TString &inputFileList);

int main(Int_t argc, Char_t *argv[]) {
  if( argc<6 ){
    std::cerr << "Please give 7 arguments "
              << "runList Seed" << "\n" 
	      << "outputFileName" << "\n"
	      << "L1 Trigger Name" << "\n" 
	      << "dRCut for L1" << "\n" 
	      << "maximum sample size" << "\n" 
	      << "iRange" << "\n" 
	      << "fRange" << "" 
	      << std::endl;
    return -1;
  }
  
  const char *inputFileList = argv[1];
  const char *outFileName   = argv[2];
  const char *name          = argv[3];
  const char *DRCut         = argv[4];
  double totalTracks        = atof(argv[5]);
  const int iRange          = atoi(argv[6]);
  const int fRange          = atoi(argv[7]);
  
  // Reading Tree                                                                                   
  std::cout << "---------------------" << std::endl;
  std::cout << "Reading List of input trees from " << inputFileList << std::endl;

  //  bool debug=true;
  bool debug=false;
  const char *ranges[15] = {"0to5",     "5to15",    "15to30",   "30to50",
			    "50to80",   "80to120",  "120to170", "170to300", 
			    "300to470", "470to600", "600to800", "800to1000", 
			    "1000to1400", "1400to1800", "1800"};
  
  double evFrac[15]      = {4.844e10, 3.675e10, 8.159e08, 5.312e07, 
			    6.359e06, 7.843e05, 1.151e05, 2.426e04, 
			    1.168e03, 7.022e01, 1.555e01, 1.844, 
			    3.321e-01, 1.087e-02, 3.575e-04};
  double nEvents[15]     = {1082851, 1649302, 6582850, 10999212,  
			    6599873, 6589860, 6127443, 5593629,
			    6255698, 3890287, 3379490, 3585297,
			    2051327, 2196167, 293135 };

  std::vector<std::string> Ranges, rangesV;
  std::vector<double>      fraction, events;
  for(int i=0; i<15; i++) rangesV.push_back(ranges[i]);
  if (iRange != fRange) {
    for (unsigned int i=iRange; i<=fRange; i++) {
      Ranges.push_back(ranges[i]);
      fraction.push_back(evFrac[i]);
      events.push_back(nEvents[i]);
      if(debug) std::cout<< "range " << ranges[i] <<" fraction " <<  evFrac[i] << " nevents " << nEvents[i] << std::endl;
    }
  } else {
    Ranges.push_back(ranges[iRange]);
    fraction.push_back(1.0);
    events.push_back(totalTracks);
    std::cout << "range " << iRange << std::endl;
  }
  
  TreeAnalysisReadGen tree(outFileName, rangesV);
  tree.debug  = debug;
  tree.l1Name = name;
  tree.dRCut  = atof(DRCut);
  tree.iRange = iRange;
  tree.fRange = fRange;
  for (unsigned int i=0; i<Ranges.size(); i++) {
    char fileList[200], treeName[200];
    sprintf (fileList, "%s_%s.txt", inputFileList, Ranges[i].c_str());
    std::cout << "try to create a chain for " << fileList << std::endl;
    TChain *chain = new TChain("/isolatedGenParticles/tree");
    if( ! FillChain(chain, fileList) ) {
      std::cerr << "Cannot get the tree " << std::endl;
      return(0);
    } else {
      tree.Init(chain);
      tree.setRange(i+iRange);
      tree.Loop();
      tree.weights[i]= (fraction[i]*totalTracks)/events[i];
      std::cout << "range " << Ranges[i].c_str() << " cross-section " << fraction[i] << " nevents  " << events[i] << " weight " << tree.weights[i] << std::endl;
      tree.clear();
      std::cout << iRange << " tree cleared" << std::endl;
    }
  }
  std::cout << "Here I am " << iRange << ":" << fRange << std::endl;
  if (iRange != fRange) tree.AddWeight();
  return 0;
}


Bool_t FillChain(TChain *chain, const TString &inputFileList) {

  ifstream infile(inputFileList);
  std::string buffer;

  if(!infile.is_open()) {
    std::cerr << "** ERROR: Can't open '" << inputFileList << "' for input" << std::endl;
    return kFALSE;
  }
  
  std::cout << "TreeUtilities : FillChain " << std::endl;
  while(1) {
    infile >> buffer;
    if(!infile.good()) break;
    //std::cout << "Adding tree from " << buffer.c_str() << std::endl;
    chain->Add(buffer.c_str());
    //std::cout << "No. of Entries in this tree : " << chain->GetEntries() << std::endl;
  }
  return kTRUE;
}

TreeAnalysisReadGen::TreeAnalysisReadGen(const char *outFileName, std::vector<std::string>& ranges) {

  double tempgen_TH[NPBins+1] = { 0.0,  1.0,  2.0,  3.0,  4.0,  
				  5.0,  6.0,  7.0,  8.0,  9.0, 
				  10.0, 12.0, 15.0, 20.0, 25.0, 
				  30.0, 40.0, 60.0, 70.0, 80.0, 100., 200.};

  for(int i=0; i<NPBins+1; i++)  genPartPBins[i]  = tempgen_TH[i];
  
  double tempgen_Eta[NEtaBins+1] = {0.0, 0.5, 1.1, 1.7, 2.3};

  for(int i=0; i<NEtaBins+1; i++) genPartEtaBins[i] = tempgen_Eta[i];

  // if parameter tree is not specified (or zero), connect the file
  // used to generate this class and read the Tree.
  //  Init(tree);  
  BookHistograms(outFileName, ranges);
}

TreeAnalysisReadGen::~TreeAnalysisReadGen() {
  if (!fChain) return;

  fout->cd();
  fout->Write();
  fout->Close();
  std::cout << "after Close\n";
  //  delete fChain->GetCurrentFile();
}

Int_t TreeAnalysisReadGen::Cut(Long64_t entry) {
  // This function may be called from Loop.
  // returns  1 if entry is accepted.
  // returns -1 otherwise.
  return 1;
}

Int_t TreeAnalysisReadGen::GetEntry(Long64_t entry) {

  // Read contents of entry.
  if (!fChain) return 0;
  return fChain->GetEntry(entry);
}

Long64_t TreeAnalysisReadGen::LoadTree(Long64_t entry) {
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

void TreeAnalysisReadGen::Init(TChain *tree) {
  // The Init() function is called when the selector needs to initialize
  // a new tree or chain. Typically here the branch addresses and branch
  // pointers of the tree will be set.
  // It is normally not necessary to make changes to the generated
  // code, but the routine can be extended by the user if needed.
  // Init() will be called many times when running on PROOF
  // (once per file to be processed).

  // Set object pointer
  t_isoTrkPAll = 0;
  t_isoTrkPtAll = 0;
  t_isoTrkPhiAll = 0;
  t_isoTrkEtaAll = 0;
  t_isoTrkDPhiAll = 0;
  t_isoTrkDEtaAll = 0;
  t_isoTrkPdgIdAll = 0;
  t_isoTrkP = 0;
  t_isoTrkPt = 0;
  t_isoTrkEne = 0;
  t_isoTrkEta = 0;
  t_isoTrkPhi = 0;
  t_isoTrkPdgId = 0;
  t_maxNearP31x31 = 0;
  t_cHadronEne31x31 = 0;
  t_cHadronEne31x31_1 = 0;
  t_cHadronEne31x31_2 = 0;
  t_cHadronEne31x31_3 = 0;
  t_nHadronEne31x31 = 0;
  t_photonEne31x31 = 0;
  t_eleEne31x31 = 0;
  t_muEne31x31 = 0;
  t_maxNearP25x25 = 0;
  t_cHadronEne25x25 = 0;
  t_cHadronEne25x25_1 = 0;
  t_cHadronEne25x25_2 = 0;
  t_cHadronEne25x25_3 = 0;
  t_nHadronEne25x25 = 0;
  t_photonEne25x25 = 0;
  t_eleEne25x25 = 0;
  t_muEne25x25 = 0;
  t_maxNearP21x21 = 0;
  t_cHadronEne21x21 = 0;
  t_cHadronEne21x21_1 = 0;
  t_cHadronEne21x21_2 = 0;
  t_cHadronEne21x21_3 = 0;
  t_nHadronEne21x21 = 0;
  t_photonEne21x21 = 0;
  t_eleEne21x21 = 0;
  t_muEne21x21 = 0;
  t_maxNearP15x15 = 0;
  t_cHadronEne15x15 = 0;
  t_cHadronEne15x15_1 = 0;
  t_cHadronEne15x15_2 = 0;
  t_cHadronEne15x15_3 = 0;
  t_nHadronEne15x15 = 0;
  t_photonEne15x15 = 0;
  t_eleEne15x15 = 0;
  t_muEne15x15 = 0;
  t_maxNearP11x11 = 0;
  t_cHadronEne11x11 = 0;
  t_cHadronEne11x11_1 = 0;
  t_cHadronEne11x11_2 = 0;
  t_cHadronEne11x11_3 = 0;
  t_nHadronEne11x11 = 0;
  t_photonEne11x11 = 0;
  t_eleEne11x11 = 0;
  t_muEne11x11 = 0;
  t_maxNearP9x9 = 0;
  t_cHadronEne9x9 = 0;
  t_cHadronEne9x9_1 = 0;
  t_cHadronEne9x9_2 = 0;
  t_cHadronEne9x9_3 = 0;
  t_nHadronEne9x9 = 0;
  t_photonEne9x9 = 0;
  t_eleEne9x9 = 0;
  t_muEne9x9 = 0;
  t_maxNearP7x7 = 0;
  t_cHadronEne7x7 = 0;
  t_cHadronEne7x7_1 = 0;
  t_cHadronEne7x7_2 = 0;
  t_cHadronEne7x7_3 = 0;
  t_nHadronEne7x7 = 0;
  t_photonEne7x7 = 0;
  t_eleEne7x7 = 0;
  t_muEne7x7 = 0;
  t_maxNearPHC3x3 = 0;
  t_cHadronEneHC3x3 = 0;
  t_cHadronEneHC3x3_1 = 0;
  t_cHadronEneHC3x3_2 = 0;
  t_cHadronEneHC3x3_3 = 0;
  t_nHadronEneHC3x3 = 0;
  t_photonEneHC3x3 = 0;
  t_eleEneHC3x3 = 0;
  t_muEneHC3x3 = 0;
  t_maxNearPHC5x5 = 0;
  t_cHadronEneHC5x5 = 0;
  t_cHadronEneHC5x5_1 = 0;
  t_cHadronEneHC5x5_2 = 0;
  t_cHadronEneHC5x5_3 = 0;
  t_nHadronEneHC5x5 = 0;
  t_photonEneHC5x5 = 0;
  t_eleEneHC5x5 = 0;
  t_muEneHC5x5 = 0;

  t_maxNearPHC7x7 = 0;
  t_cHadronEneHC7x7 = 0;
  t_cHadronEneHC7x7_1 = 0;
  t_cHadronEneHC7x7_2 = 0;
  t_cHadronEneHC7x7_3 = 0;
  t_nHadronEneHC7x7 = 0;
  t_photonEneHC7x7 = 0;
  t_eleEneHC7x7 = 0;
  t_muEneHC7x7 = 0;

  t_maxNearPR     = 0;
  t_cHadronEneR   = 0;
  t_cHadronEneR_1 = 0;
  t_cHadronEneR_2 = 0;
  t_cHadronEneR_3 = 0;
  t_nHadronEneR   = 0;
  t_photonEneR    = 0;
  t_eleEneR       = 0;
  t_muEneR        = 0;

  t_maxNearPIsoR     = 0;
  t_cHadronEneIsoR   = 0;
  t_cHadronEneIsoR_1 = 0;
  t_cHadronEneIsoR_2 = 0;
  t_cHadronEneIsoR_3 = 0;
  t_nHadronEneIsoR   = 0;
  t_photonEneIsoR    = 0;
  t_eleEneIsoR       = 0;
  t_muEneIsoR        = 0;
  
   t_maxNearPHCR     = 0;
  t_cHadronEneHCR   = 0;
  t_cHadronEneHCR_1 = 0;
  t_cHadronEneHCR_2 = 0;
  t_cHadronEneHCR_3 = 0;
  t_nHadronEneHCR   = 0;
  t_photonEneHCR    = 0;
  t_eleEneHCR       = 0;
  t_muEneHCR        = 0;

  t_maxNearPIsoHCR     = 0;
  t_cHadronEneIsoHCR   = 0;
  t_cHadronEneIsoHCR_1 = 0;
  t_cHadronEneIsoHCR_2 = 0;
  t_cHadronEneIsoHCR_3 = 0;
  t_nHadronEneIsoHCR   = 0;
  t_photonEneIsoHCR    = 0;
  t_eleEneIsoHCR       = 0;
  t_muEneIsoHCR        = 0;

  t_L1Decision = 0;
  t_L1CenJetPt = 0;
  t_L1CenJetEta = 0;
  t_L1CenJetPhi = 0;
  t_L1FwdJetPt = 0;
  t_L1FwdJetEta = 0;
  t_L1FwdJetPhi = 0;
  t_L1TauJetPt = 0;
  t_L1TauJetEta = 0;
  t_L1TauJetPhi = 0;
  t_L1MuonPt = 0;
  t_L1MuonEta = 0;
  t_L1MuonPhi = 0;
  t_L1IsoEMPt = 0;
  t_L1IsoEMEta = 0;
  t_L1IsoEMPhi = 0;
  t_L1NonIsoEMPt = 0;
  t_L1NonIsoEMEta = 0;
  t_L1NonIsoEMPhi = 0;
  t_L1METPt = 0;
  t_L1METEta = 0;
  t_L1METPhi = 0;
  // Set branch addresses and branch pointers
  if (!tree) return;
  fChain = tree;
  fCurrent = -1;
  fChain->SetMakeClass(1);
  
  fChain->SetBranchAddress("t_isoTrkPAll", &t_isoTrkPAll, &b_t_isoTrkPAll);
  fChain->SetBranchAddress("t_isoTrkPtAll", &t_isoTrkPtAll, &b_t_isoTrkPtAll);
  fChain->SetBranchAddress("t_isoTrkPhiAll", &t_isoTrkPhiAll, &b_t_isoTrkPhiAll);
  fChain->SetBranchAddress("t_isoTrkEtaAll", &t_isoTrkEtaAll, &b_t_isoTrkEtaAll);
  fChain->SetBranchAddress("t_isoTrkDPhiAll", &t_isoTrkDPhiAll, &b_t_isoTrkDPhiAll);
  fChain->SetBranchAddress("t_isoTrkDEtaAll", &t_isoTrkDEtaAll, &b_t_isoTrkDEtaAll);
  fChain->SetBranchAddress("t_isoTrkPdgIdAll", &t_isoTrkPdgIdAll, &b_t_isoTrkPdgIdAll);
  fChain->SetBranchAddress("t_isoTrkP", &t_isoTrkP, &b_t_isoTrkP);
  fChain->SetBranchAddress("t_isoTrkPt", &t_isoTrkPt, &b_t_isoTrkPt);
  fChain->SetBranchAddress("t_isoTrkEne", &t_isoTrkEne, &b_t_isoTrkEne);
  fChain->SetBranchAddress("t_isoTrkEta", &t_isoTrkEta, &b_t_isoTrkEta);
  fChain->SetBranchAddress("t_isoTrkPhi", &t_isoTrkPhi, &b_t_isoTrkPhi);
  fChain->SetBranchAddress("t_isoTrkPdgId", &t_isoTrkPdgId, &b_t_isoTrkPdgId);
  fChain->SetBranchAddress("t_maxNearP31x31", &t_maxNearP31x31, &b_t_maxNearP31x31);
  fChain->SetBranchAddress("t_cHadronEne31x31", &t_cHadronEne31x31, &b_t_cHadronEne31x31);
  fChain->SetBranchAddress("t_cHadronEne31x31_1", &t_cHadronEne31x31_1, &b_t_cHadronEne31x31_1);
  fChain->SetBranchAddress("t_cHadronEne31x31_2", &t_cHadronEne31x31_2, &b_t_cHadronEne31x31_2);
  fChain->SetBranchAddress("t_cHadronEne31x31_3", &t_cHadronEne31x31_3, &b_t_cHadronEne31x31_3);
  fChain->SetBranchAddress("t_nHadronEne31x31", &t_nHadronEne31x31, &b_t_nHadronEne31x31);
  fChain->SetBranchAddress("t_photonEne31x31", &t_photonEne31x31, &b_t_photonEne31x31);
  fChain->SetBranchAddress("t_eleEne31x31", &t_eleEne31x31, &b_t_eleEne31x31);
  fChain->SetBranchAddress("t_muEne31x31", &t_muEne31x31, &b_t_muEne31x31);
  fChain->SetBranchAddress("t_maxNearP25x25", &t_maxNearP25x25, &b_t_maxNearP25x25);
  fChain->SetBranchAddress("t_cHadronEne25x25", &t_cHadronEne25x25, &b_t_cHadronEne25x25);
  fChain->SetBranchAddress("t_cHadronEne25x25_1", &t_cHadronEne25x25_1, &b_t_cHadronEne25x25_1);
  fChain->SetBranchAddress("t_cHadronEne25x25_2", &t_cHadronEne25x25_2, &b_t_cHadronEne25x25_2);
  fChain->SetBranchAddress("t_cHadronEne25x25_3", &t_cHadronEne25x25_3, &b_t_cHadronEne25x25_3);
  fChain->SetBranchAddress("t_nHadronEne25x25", &t_nHadronEne25x25, &b_t_nHadronEne25x25);
  fChain->SetBranchAddress("t_photonEne25x25", &t_photonEne25x25, &b_t_photonEne25x25);
  fChain->SetBranchAddress("t_eleEne25x25", &t_eleEne25x25, &b_t_eleEne25x25);
  fChain->SetBranchAddress("t_muEne25x25", &t_muEne25x25, &b_t_muEne25x25);
  fChain->SetBranchAddress("t_maxNearP21x21", &t_maxNearP21x21, &b_t_maxNearP21x21);
  fChain->SetBranchAddress("t_cHadronEne21x21", &t_cHadronEne21x21, &b_t_cHadronEne21x21);
  fChain->SetBranchAddress("t_cHadronEne21x21_1", &t_cHadronEne21x21_1, &b_t_cHadronEne21x21_1);
  fChain->SetBranchAddress("t_cHadronEne21x21_2", &t_cHadronEne21x21_2, &b_t_cHadronEne21x21_2);
  fChain->SetBranchAddress("t_cHadronEne21x21_3", &t_cHadronEne21x21_3, &b_t_cHadronEne21x21_3);
  fChain->SetBranchAddress("t_nHadronEne21x21", &t_nHadronEne21x21, &b_t_nHadronEne21x21);
  fChain->SetBranchAddress("t_photonEne21x21", &t_photonEne21x21, &b_t_photonEne21x21);
  fChain->SetBranchAddress("t_eleEne21x21", &t_eleEne21x21, &b_t_eleEne21x21);
  fChain->SetBranchAddress("t_muEne21x21", &t_muEne21x21, &b_t_muEne21x21);
  fChain->SetBranchAddress("t_maxNearP15x15", &t_maxNearP15x15, &b_t_maxNearP15x15);
  fChain->SetBranchAddress("t_cHadronEne15x15", &t_cHadronEne15x15, &b_t_cHadronEne15x15);
  fChain->SetBranchAddress("t_cHadronEne15x15_1", &t_cHadronEne15x15_1, &b_t_cHadronEne15x15_1);
  fChain->SetBranchAddress("t_cHadronEne15x15_2", &t_cHadronEne15x15_2, &b_t_cHadronEne15x15_2);
  fChain->SetBranchAddress("t_cHadronEne15x15_3", &t_cHadronEne15x15_3, &b_t_cHadronEne15x15_3);
  fChain->SetBranchAddress("t_nHadronEne15x15", &t_nHadronEne15x15, &b_t_nHadronEne15x15);
  fChain->SetBranchAddress("t_photonEne15x15", &t_photonEne15x15, &b_t_photonEne15x15);
  fChain->SetBranchAddress("t_eleEne15x15", &t_eleEne15x15, &b_t_eleEne15x15);
  fChain->SetBranchAddress("t_muEne15x15", &t_muEne15x15, &b_t_muEne15x15);
  fChain->SetBranchAddress("t_maxNearP11x11", &t_maxNearP11x11, &b_t_maxNearP11x11);
  fChain->SetBranchAddress("t_cHadronEne11x11", &t_cHadronEne11x11, &b_t_cHadronEne11x11);
  fChain->SetBranchAddress("t_cHadronEne11x11_1", &t_cHadronEne11x11_1, &b_t_cHadronEne11x11_1);
  fChain->SetBranchAddress("t_cHadronEne11x11_2", &t_cHadronEne11x11_2, &b_t_cHadronEne11x11_2);
  fChain->SetBranchAddress("t_cHadronEne11x11_3", &t_cHadronEne11x11_3, &b_t_cHadronEne11x11_3);
  fChain->SetBranchAddress("t_nHadronEne11x11", &t_nHadronEne11x11, &b_t_nHadronEne11x11);
  fChain->SetBranchAddress("t_photonEne11x11", &t_photonEne11x11, &b_t_photonEne11x11);
  fChain->SetBranchAddress("t_eleEne11x11", &t_eleEne11x11, &b_t_eleEne11x11);
  fChain->SetBranchAddress("t_muEne11x11", &t_muEne11x11, &b_t_muEne11x11);
  fChain->SetBranchAddress("t_maxNearP9x9", &t_maxNearP9x9, &b_t_maxNearP9x9);
  fChain->SetBranchAddress("t_cHadronEne9x9", &t_cHadronEne9x9, &b_t_cHadronEne9x9);
  fChain->SetBranchAddress("t_cHadronEne9x9_1", &t_cHadronEne9x9_1, &b_t_cHadronEne9x9_1);
  fChain->SetBranchAddress("t_cHadronEne9x9_2", &t_cHadronEne9x9_2, &b_t_cHadronEne9x9_2);
  fChain->SetBranchAddress("t_cHadronEne9x9_3", &t_cHadronEne9x9_3, &b_t_cHadronEne9x9_3);
  fChain->SetBranchAddress("t_nHadronEne9x9", &t_nHadronEne9x9, &b_t_nHadronEne9x9);
  fChain->SetBranchAddress("t_photonEne9x9", &t_photonEne9x9, &b_t_photonEne9x9);
  fChain->SetBranchAddress("t_eleEne9x9", &t_eleEne9x9, &b_t_eleEne9x9);
  fChain->SetBranchAddress("t_muEne9x9", &t_muEne9x9, &b_t_muEne9x9);
  fChain->SetBranchAddress("t_maxNearP7x7", &t_maxNearP7x7, &b_t_maxNearP7x7);
  fChain->SetBranchAddress("t_cHadronEne7x7", &t_cHadronEne7x7, &b_t_cHadronEne7x7);
  fChain->SetBranchAddress("t_cHadronEne7x7_1", &t_cHadronEne7x7_1, &b_t_cHadronEne7x7_1);
  fChain->SetBranchAddress("t_cHadronEne7x7_2", &t_cHadronEne7x7_2, &b_t_cHadronEne7x7_2);
  fChain->SetBranchAddress("t_cHadronEne7x7_3", &t_cHadronEne7x7_3, &b_t_cHadronEne7x7_3);
  fChain->SetBranchAddress("t_nHadronEne7x7", &t_nHadronEne7x7, &b_t_nHadronEne7x7);
  fChain->SetBranchAddress("t_photonEne7x7", &t_photonEne7x7, &b_t_photonEne7x7);
  fChain->SetBranchAddress("t_eleEne7x7", &t_eleEne7x7, &b_t_eleEne7x7);
  fChain->SetBranchAddress("t_muEne7x7", &t_muEne7x7, &b_t_muEne7x7);
  fChain->SetBranchAddress("t_maxNearPHC3x3", &t_maxNearPHC3x3, &b_t_maxNearPHC3x3);
  fChain->SetBranchAddress("t_cHadronEneHC3x3", &t_cHadronEneHC3x3, &b_t_cHadronEneHC3x3);
  fChain->SetBranchAddress("t_cHadronEneHC3x3_1", &t_cHadronEneHC3x3_1, &b_t_cHadronEneHC3x3_1);
  fChain->SetBranchAddress("t_cHadronEneHC3x3_2", &t_cHadronEneHC3x3_2, &b_t_cHadronEneHC3x3_2);
  fChain->SetBranchAddress("t_cHadronEneHC3x3_3", &t_cHadronEneHC3x3_3, &b_t_cHadronEneHC3x3_3);
  fChain->SetBranchAddress("t_nHadronEneHC3x3", &t_nHadronEneHC3x3, &b_t_nHadronEneHC3x3);
  fChain->SetBranchAddress("t_photonEneHC3x3", &t_photonEneHC3x3, &b_t_photonEneHC3x3);
  fChain->SetBranchAddress("t_eleEneHC3x3", &t_eleEneHC3x3, &b_t_eleEneHC3x3);
  fChain->SetBranchAddress("t_muEneHC3x3", &t_muEneHC3x3, &b_t_muEneHC3x3);
  fChain->SetBranchAddress("t_maxNearPHC5x5", &t_maxNearPHC5x5, &b_t_maxNearPHC5x5);
  fChain->SetBranchAddress("t_cHadronEneHC5x5", &t_cHadronEneHC5x5, &b_t_cHadronEneHC5x5);
  fChain->SetBranchAddress("t_cHadronEneHC5x5_1", &t_cHadronEneHC5x5_1, &b_t_cHadronEneHC5x5_1);
  fChain->SetBranchAddress("t_cHadronEneHC5x5_2", &t_cHadronEneHC5x5_2, &b_t_cHadronEneHC5x5_2);
  fChain->SetBranchAddress("t_cHadronEneHC5x5_3", &t_cHadronEneHC5x5_3, &b_t_cHadronEneHC5x5_3);
  fChain->SetBranchAddress("t_nHadronEneHC5x5", &t_nHadronEneHC5x5, &b_t_nHadronEneHC5x5);
  fChain->SetBranchAddress("t_photonEneHC5x5", &t_photonEneHC5x5, &b_t_photonEneHC5x5);
  fChain->SetBranchAddress("t_eleEneHC5x5", &t_eleEneHC5x5, &b_t_eleEneHC5x5);
  fChain->SetBranchAddress("t_muEneHC5x5", &t_muEneHC5x5, &b_t_muEneHC5x5);
  fChain->SetBranchAddress("t_maxNearPHC7x7", &t_maxNearPHC7x7, &b_t_maxNearPHC7x7);
  fChain->SetBranchAddress("t_cHadronEneHC7x7", &t_cHadronEneHC7x7, &b_t_cHadronEneHC7x7);
  fChain->SetBranchAddress("t_cHadronEneHC7x7_1", &t_cHadronEneHC7x7_1, &b_t_cHadronEneHC7x7_1);
  fChain->SetBranchAddress("t_cHadronEneHC7x7_2", &t_cHadronEneHC7x7_2, &b_t_cHadronEneHC7x7_2);
  fChain->SetBranchAddress("t_cHadronEneHC7x7_3", &t_cHadronEneHC7x7_3, &b_t_cHadronEneHC7x7_3);
  fChain->SetBranchAddress("t_nHadronEneHC7x7", &t_nHadronEneHC7x7, &b_t_nHadronEneHC7x7);
  fChain->SetBranchAddress("t_photonEneHC7x7", &t_photonEneHC7x7, &b_t_photonEneHC7x7);
  fChain->SetBranchAddress("t_eleEneHC7x7", &t_eleEneHC7x7, &b_t_eleEneHC7x7);
  fChain->SetBranchAddress("t_muEneHC7x7", &t_muEneHC7x7, &b_t_muEneHC7x7);
  fChain->SetBranchAddress("t_maxNearPR", &t_maxNearPR, &b_t_maxNearPR);
  fChain->SetBranchAddress("t_cHadronEneR", &t_cHadronEneR, &b_t_cHadronEneR);
  fChain->SetBranchAddress("t_cHadronEneR_1", &t_cHadronEneR_1, &b_t_cHadronEneR_1);
  fChain->SetBranchAddress("t_cHadronEneR_2", &t_cHadronEneR_2, &b_t_cHadronEneR_2);
  fChain->SetBranchAddress("t_cHadronEneR_3", &t_cHadronEneR_3, &b_t_cHadronEneR_3);
  fChain->SetBranchAddress("t_nHadronEneR", &t_nHadronEneR, &b_t_nHadronEneR);
  fChain->SetBranchAddress("t_photonEneR", &t_photonEneR, &b_t_photonEneR);
  fChain->SetBranchAddress("t_eleEneR", &t_eleEneR, &b_t_eleEneR);
  fChain->SetBranchAddress("t_muEneR", &t_muEneR, &b_t_muEneR);
  fChain->SetBranchAddress("t_maxNearPIsoR", &t_maxNearPIsoR, &b_t_maxNearPIsoR);
  fChain->SetBranchAddress("t_cHadronEneIsoR", &t_cHadronEneIsoR, &b_t_cHadronEneIsoR);
  fChain->SetBranchAddress("t_cHadronEneIsoR_1", &t_cHadronEneIsoR_1, &b_t_cHadronEneIsoR_1);
  fChain->SetBranchAddress("t_cHadronEneIsoR_2", &t_cHadronEneIsoR_2, &b_t_cHadronEneIsoR_2);
  fChain->SetBranchAddress("t_cHadronEneIsoR_3", &t_cHadronEneIsoR_3, &b_t_cHadronEneIsoR_3);
  fChain->SetBranchAddress("t_nHadronEneIsoR", &t_nHadronEneIsoR, &b_t_nHadronEneIsoR);
  fChain->SetBranchAddress("t_photonEneIsoR", &t_photonEneIsoR, &b_t_photonEneIsoR);
  fChain->SetBranchAddress("t_eleEneIsoR", &t_eleEneIsoR, &b_t_eleEneIsoR);
  fChain->SetBranchAddress("t_muEneIsoR", &t_muEneIsoR, &b_t_muEneIsoR);
  fChain->SetBranchAddress("t_maxNearPHCR", &t_maxNearPHCR, &b_t_maxNearPHCR);
  fChain->SetBranchAddress("t_cHadronEneHCR", &t_cHadronEneHCR, &b_t_cHadronEneHCR);
  fChain->SetBranchAddress("t_cHadronEneHCR_1", &t_cHadronEneHCR_1, &b_t_cHadronEneHCR_1);
  fChain->SetBranchAddress("t_cHadronEneHCR_2", &t_cHadronEneHCR_2, &b_t_cHadronEneHCR_2);
  fChain->SetBranchAddress("t_cHadronEneHCR_3", &t_cHadronEneHCR_3, &b_t_cHadronEneHCR_3);
  fChain->SetBranchAddress("t_nHadronEneHCR", &t_nHadronEneHCR, &b_t_nHadronEneHCR);
  fChain->SetBranchAddress("t_photonEneHCR", &t_photonEneHCR, &b_t_photonEneHCR);
  fChain->SetBranchAddress("t_eleEneHCR", &t_eleEneHCR, &b_t_eleEneHCR);
  fChain->SetBranchAddress("t_muEneHCR", &t_muEneHCR, &b_t_muEneHCR);
  fChain->SetBranchAddress("t_maxNearPIsoHCR", &t_maxNearPIsoHCR, &b_t_maxNearPIsoHCR);
  fChain->SetBranchAddress("t_cHadronEneIsoHCR", &t_cHadronEneIsoHCR, &b_t_cHadronEneIsoHCR);
  fChain->SetBranchAddress("t_cHadronEneIsoHCR_1", &t_cHadronEneIsoHCR_1, &b_t_cHadronEneIsoHCR_1);
  fChain->SetBranchAddress("t_cHadronEneIsoHCR_2", &t_cHadronEneIsoHCR_2, &b_t_cHadronEneIsoHCR_2);
  fChain->SetBranchAddress("t_cHadronEneIsoHCR_3", &t_cHadronEneIsoHCR_3, &b_t_cHadronEneIsoHCR_3);
  fChain->SetBranchAddress("t_nHadronEneIsoHCR", &t_nHadronEneIsoHCR, &b_t_nHadronEneIsoHCR);
  fChain->SetBranchAddress("t_photonEneIsoHCR", &t_photonEneIsoHCR, &b_t_photonEneIsoHCR);
  fChain->SetBranchAddress("t_eleEneIsoHCR", &t_eleEneIsoHCR, &b_t_eleEneIsoHCR);
  fChain->SetBranchAddress("t_muEneIsoHCR", &t_muEneIsoHCR, &b_t_muEneIsoHCR);
  fChain->SetBranchAddress("t_L1Decision", &t_L1Decision, &b_t_L1Decision);
  fChain->SetBranchAddress("t_L1CenJetPt", &t_L1CenJetPt, &b_t_L1CenJetPt);
  fChain->SetBranchAddress("t_L1CenJetEta", &t_L1CenJetEta, &b_t_L1CenJetEta);
  fChain->SetBranchAddress("t_L1CenJetPhi", &t_L1CenJetPhi, &b_t_L1CenJetPhi);
  fChain->SetBranchAddress("t_L1FwdJetPt", &t_L1FwdJetPt, &b_t_L1FwdJetPt);
  fChain->SetBranchAddress("t_L1FwdJetEta", &t_L1FwdJetEta, &b_t_L1FwdJetEta);
  fChain->SetBranchAddress("t_L1FwdJetPhi", &t_L1FwdJetPhi, &b_t_L1FwdJetPhi);
  fChain->SetBranchAddress("t_L1TauJetPt", &t_L1TauJetPt, &b_t_L1TauJetPt);
  fChain->SetBranchAddress("t_L1TauJetEta", &t_L1TauJetEta, &b_t_L1TauJetEta);
  fChain->SetBranchAddress("t_L1TauJetPhi", &t_L1TauJetPhi, &b_t_L1TauJetPhi);
  fChain->SetBranchAddress("t_L1MuonPt", &t_L1MuonPt, &b_t_L1MuonPt);
  fChain->SetBranchAddress("t_L1MuonEta", &t_L1MuonEta, &b_t_L1MuonEta);
  fChain->SetBranchAddress("t_L1MuonPhi", &t_L1MuonPhi, &b_t_L1MuonPhi);
  fChain->SetBranchAddress("t_L1IsoEMPt", &t_L1IsoEMPt, &b_t_L1IsoEMPt);
  fChain->SetBranchAddress("t_L1IsoEMEta", &t_L1IsoEMEta, &b_t_L1IsoEMEta);
  fChain->SetBranchAddress("t_L1IsoEMPhi", &t_L1IsoEMPhi, &b_t_L1IsoEMPhi);
  fChain->SetBranchAddress("t_L1NonIsoEMPt", &t_L1NonIsoEMPt, &b_t_L1NonIsoEMPt);
  fChain->SetBranchAddress("t_L1NonIsoEMEta", &t_L1NonIsoEMEta, &b_t_L1NonIsoEMEta);
  fChain->SetBranchAddress("t_L1NonIsoEMPhi", &t_L1NonIsoEMPhi, &b_t_L1NonIsoEMPhi);
  fChain->SetBranchAddress("t_L1METPt", &t_L1METPt, &b_t_L1METPt);
  fChain->SetBranchAddress("t_L1METEta", &t_L1METEta, &b_t_L1METEta);
  fChain->SetBranchAddress("t_L1METPhi", &t_L1METPhi, &b_t_L1METPhi);
  Notify();
}

void TreeAnalysisReadGen::Loop() {
  
  getL1Names();
  int ibit = -1;
  for (std::map<std::string,int>::iterator it=l1Names.begin(); it != l1Names.end(); ++it) {
    if (!strcmp(l1Name.c_str(),(it->first).c_str())) {
      ibit =(it->second);
      break;
    }
  }
  if (debug) std::cout << "liName " << l1Name.c_str() << " " << ibit << std::endl;  

  if (fChain == 0) return;  
  Long64_t nentries = fChain->GetEntries();
  std::cout << "No. of Entries in tree " << nentries << std::endl;
  
  Long64_t nbytes = 0, nb = 0;
  unsigned int nTrk=0;    
  unsigned int nTrk_Bins=0;    
  unsigned int nTrk_maxNearP=0;
  unsigned int nIsoTrk=0;
  for (Long64_t jentry=0; jentry<nentries;jentry++) {
    
    Long64_t ientry = LoadTree(jentry);
    if (ientry < 0) break;
    nb = fChain->GetEntry(jentry);   nbytes += nb;
    
    if( !(jentry%100000) )
      std::cout << "processing event " << jentry+1 << std::endl;
    
    // get the inclusive distributions here
    for(int itrk=0; itrk<t_isoTrkPAll->size(); itrk++ ){
      double p     = (*t_isoTrkPAll)    [itrk];
      double pt    = (*t_isoTrkPtAll)   [itrk];
      double eta   = (*t_isoTrkEtaAll)  [itrk];
      double phi   = (*t_isoTrkPhiAll)  [itrk];
      double pdgid = (*t_isoTrkPdgIdAll)[itrk];
      double deta  = (*t_isoTrkDEtaAll) [itrk];
      double dphi  = (*t_isoTrkDPhiAll) [itrk];
      
      if (debug) std::cout << "p " << p << " pt " << pt << " eta " << eta << " phi " << phi << " pdgid " << pdgid << " deta " << deta << " dphi " << dphi << std::endl;
      int iTrkEtaBin=-1, iTrkMomBin=-1;
      for(int ieta=0; ieta<NEtaBins; ieta++)   {
	if (std::abs(eta)>genPartEtaBins[ieta] && std::abs(eta)<genPartEtaBins[ieta+1] ) iTrkEtaBin = ieta;
      }
      for(int ipt=0;  ipt<NPBins;   ipt++)  {
	if (p>genPartPBins[ipt] &&  p<genPartPBins[ipt+1] )  iTrkMomBin = ipt;
      }
      if (debug) std::cout << " etabin " << iTrkEtaBin << " mombin " << iTrkMomBin <<std::endl;
      if (std::abs(pdgid) == 211 ) {
	h_trkPAll  [0][iRangeBin] ->Fill(p);
	h_trkPtAll [0][iRangeBin] ->Fill(pt);
	h_trkEtaAll[0][iRangeBin] ->Fill(eta);
	h_trkPhiAll[0][iRangeBin] ->Fill(phi);
	if( iTrkMomBin>=0 && iTrkEtaBin>=0 ) {
	  h_trkDEta[iTrkMomBin][iTrkEtaBin][iRangeBin]->Fill(deta);
	  h_trkDPhi[iTrkMomBin][iTrkEtaBin][iRangeBin]->Fill(dphi);
	}
      }
      else if( std::abs(pdgid)==321 ) {
	h_trkPAll  [1][iRangeBin] ->Fill(p);
	h_trkPtAll [1][iRangeBin] ->Fill(pt);
	h_trkEtaAll[1][iRangeBin] ->Fill(eta);
	h_trkPhiAll[1][iRangeBin] ->Fill(phi);
	if( iTrkMomBin>=0 && iTrkEtaBin>=0 ) {
	  h_trkDEta[iTrkMomBin][iTrkEtaBin][iRangeBin]->Fill(deta);
	  h_trkDPhi[iTrkMomBin][iTrkEtaBin][iRangeBin]->Fill(dphi);
	}
      }
      else if( pdgid==2212 ) {
	h_trkPAll  [2][iRangeBin] ->Fill(p);
	h_trkPtAll [2][iRangeBin] ->Fill(pt);
	h_trkEtaAll[2][iRangeBin] ->Fill(eta);
	h_trkPhiAll[2][iRangeBin] ->Fill(phi);
	if( iTrkMomBin>=0 && iTrkEtaBin>=0 ) {
	  h_trkDEta[iTrkMomBin][iTrkEtaBin][iRangeBin]->Fill(deta);
	  h_trkDPhi[iTrkMomBin][iTrkEtaBin][iRangeBin]->Fill(dphi);
	}
      }
      else if( pdgid==-2212 ) {
	h_trkPAll  [3][iRangeBin] ->Fill(p);
	h_trkPtAll [3][iRangeBin] ->Fill(pt);
	h_trkEtaAll[3][iRangeBin] ->Fill(eta);
	h_trkPhiAll[3][iRangeBin] ->Fill(phi);
	if( iTrkMomBin>=0 && iTrkEtaBin>=0 ) {
	  h_trkDEta[iTrkMomBin][iTrkEtaBin][iRangeBin]->Fill(deta);
	  h_trkDPhi[iTrkMomBin][iTrkEtaBin][iRangeBin]->Fill(dphi);
	}
      }
    } /// Loop over IsoTrkpAll
    
    // L1 Trigger Information       
    double leadL1JetPt=0.0, leadL1JetEta=-0.999, leadL1JetPhi = -0.999;
    std::vector<int> myDec;
    bool l1SingleJet=false;
    if( (*t_L1Decision)[l1Names["L1_SingleJet6"]]  || (*t_L1Decision)[l1Names["L1_SingleJet20"]] || 
	(*t_L1Decision)[l1Names["L1_SingleJet30"]] || (*t_L1Decision)[l1Names["L1_SingleJet40"]] ||
	(*t_L1Decision)[l1Names["L1_SingleJet50"]] || (*t_L1Decision)[l1Names["L1_SingleJet60"]]) {
      l1SingleJet=true;  myDec.push_back(L1SingleJet);
      //std::cout<<"Single Jet "<<(*t_L1Decision)[l1Names["L1_SingleJet6"]] <<std::endl;
      if(t_L1CenJetPt->size()>0)  h_L1CenJetPt[iRangeBin]->Fill((*t_L1CenJetPt)[0]);
      if(t_L1FwdJetPt->size()>0)  h_L1FwdJetPt[iRangeBin]->Fill((*t_L1FwdJetPt)[0]);
      for(int i=0; i<t_L1CenJetPt->size(); i++) {
	if( (*t_L1CenJetPt)[i] > leadL1JetPt ) {
	  leadL1JetPt  = (*t_L1CenJetPt)[i];
	  leadL1JetEta = (*t_L1CenJetEta)[i];
	  leadL1JetPhi = (*t_L1CenJetPhi)[i];
	}
      }
      for(int i=0; i<t_L1FwdJetPt->size(); i++) {
	if( (*t_L1FwdJetPt)[i] > leadL1JetPt ) {
	  leadL1JetPt  = (*t_L1FwdJetPt)[i];
	  leadL1JetEta = (*t_L1FwdJetEta)[i];
	  leadL1JetPhi = (*t_L1FwdJetPhi)[i];
	}
      }
      //      if (debug) std::cout << " L1_SingleJet: LeadJet : pT " << leadL1JetPt << " eta " << leadL1JetEta << " phi " << leadL1JetPhi << std::endl;
    }

    bool l1SingleTauJet=false;
    if( (*t_L1Decision)[l1Names["L1_SingleTauJet10"]] || (*t_L1Decision)[l1Names["L1_SingleTauJet20"]] || 
	(*t_L1Decision)[l1Names["L1_SingleTauJet30"]] || (*t_L1Decision)[l1Names["L1_SingleTauJet50"]]) {
      l1SingleTauJet=true; myDec.push_back(L1SingleTauJet);
      if(t_L1TauJetPt->size()>0) h_L1TauJetPt[iRangeBin]->Fill((*t_L1TauJetPt)[0]);
      for(int i=0; i<t_L1TauJetPt->size(); i++) {
	if( (*t_L1TauJetPt)[i] > leadL1JetPt ) {
	  leadL1JetPt  = (*t_L1TauJetPt)[i];
	  leadL1JetEta = (*t_L1TauJetEta)[i];
	  leadL1JetPhi = (*t_L1TauJetPhi)[i];
	}
      }
      //      if (debug) std::cout << " L1TauJet: LeadTauJet: pT " << leadL1JetPt << " eta " << leadL1JetEta << " phi " << leadL1JetPhi << std::endl; 
    }
    if( l1SingleTauJet || l1SingleJet ) h_L1LeadJetPt[iRangeBin]->Fill(leadL1JetPt);
    bool l1SingleIsoEG=false;
    if( (*t_L1Decision)[l1Names["L1_SingleIsoEG5"]]  || (*t_L1Decision)[l1Names["L1_SingleIsoEG8"]]  || 
	(*t_L1Decision)[l1Names["L1_SingleIsoEG10"]] || (*t_L1Decision)[l1Names["L1_SingleIsoEG12"]] ||
	(*t_L1Decision)[l1Names["L1_SingleIsoEG15"]] ) {
      l1SingleIsoEG=true; myDec.push_back(L1SingleIsoEG);
    }
    bool l1SingleEG=false;
    if( (*t_L1Decision)[l1Names["L1_SingleEG1"]]  || (*t_L1Decision)[l1Names["L1_SingleEG2"]]  || 
	(*t_L1Decision)[l1Names["L1_SingleEG5"]]  || (*t_L1Decision)[l1Names["L1_SingleEG8"]]  ||
	(*t_L1Decision)[l1Names["L1_SingleEG10"]] || (*t_L1Decision)[l1Names["L1_SingleEG12"]] ||
	(*t_L1Decision)[l1Names["L1_SingleEG15"]] || (*t_L1Decision)[l1Names["L1_SingleEG20"]] ) {
      l1SingleEG=true;  myDec.push_back(L1SingleEG);
    }
    bool l1L1_SingleMu=false;
    if( (*t_L1Decision)[l1Names["L1_SingleMu0"]]  || (*t_L1Decision)[l1Names["L1_SingleMu3"]]  || 
	(*t_L1Decision)[l1Names["L1_SingleMu5"]]  || (*t_L1Decision)[l1Names["L1_SingleMu7"]]  ||
	(*t_L1Decision)[l1Names["L1_SingleMu10"]] || (*t_L1Decision)[l1Names["L1_SingleMu14"]] ||
	(*t_L1Decision)[l1Names["L1_SingleMu20"]] ) {
      l1L1_SingleMu=true;  myDec.push_back(L1SingleMu);
    }
    //    if (debug) std::cout << " L1Decision (L1SingleJet/L1SingleTauJet/L1SingleIsoEG/L1SingleEG/L1SingleMu) " << l1SingleJet << "/" << l1SingleTauJet << "/" << l1SingleIsoEG << "/" << l1SingleEG << "/" << l1L1_SingleMu << std::endl;

    bool checkL1=false, checkTest=false;
    if (ibit >= 0) {
      checkTest = true;
      if( (*t_L1Decision)[ibit] ) checkL1 = true;
    } else {
      if (l1Name=="L1Jet" || l1Name=="L1JetL1Tau" || 
	  l1Name=="L1JetL1TauL1EM" || l1Name=="L1JetL1EM") {
	checkTest = true;
	if (l1SingleJet) checkL1 = true;
      }
      if (l1Name=="L1Tau" || l1Name=="L1JetL1Tau" || l1Name=="L1JetL1TauL1EM") {
	checkTest = true;
	if (l1SingleTauJet) checkL1 = true;
      }
      if (l1Name=="L1EM" || l1Name=="L1JETL1EM" || l1Name=="L1JetL1TauL1EM") {
	checkTest = true;
	if (l1SingleEG) checkL1 = true;
      }
    }
    //    if (debug) std::cout << " ibit " << ibit << std::endl;
    if (debug) std::cout << "isotrkP size " << t_isoTrkP->size() << std::endl;
    for(int itrk=0; itrk<t_isoTrkP->size(); itrk++ ){      
      nTrk++;
      double p1            = (*t_isoTrkP)[itrk];
      double pt1           = (*t_isoTrkPt)[itrk];
      double eta1          = (*t_isoTrkEta)[itrk];
      double phi1          = (*t_isoTrkPhi)[itrk];
      double maxNearP31x31 = (*t_maxNearP31x31)[itrk];
      double maxNearP25x25 = (*t_maxNearP25x25)[itrk];
      double maxNearP21x21 = (*t_maxNearP21x21)[itrk];
      double maxNearP15x15 = (*t_maxNearP15x15)[itrk];
      double maxNearP11x11 = (*t_maxNearP11x11)[itrk];
      double maxNearPHC3x3 = (*t_maxNearPHC3x3)[itrk];
      double maxNearPHC5x5 = (*t_maxNearPHC5x5)[itrk];
      double maxNearPHC7x7 = (*t_maxNearPHC7x7)[itrk];
      double maxNearPIsoR   = (*t_maxNearPIsoR)[itrk];
      double maxNearPIsoHCR = (*t_maxNearPIsoHCR)[itrk];
      double pdgid1        = (*t_isoTrkPdgId)[itrk];

      if (debug)  std:: cout << "tracks: p " << p1 << " pt1 " << pt1 << " eta1 " << eta1 << " phi1 " << phi1 << " maxNearP(31x31/25x25/21x21/15x15/11x11//H3x3/H5x5/H7x7//IsoR//IsoHR " <<  maxNearP31x31 << "/" << maxNearP25x25 << "/" << maxNearP21x21 << "/" << maxNearP15x15 << "/" << maxNearP11x11 << "//" << maxNearPHC3x3 << "/" << maxNearPHC5x5 << "/" << maxNearPHC7x7 << "//" << maxNearPIsoR << "//" << maxNearPIsoHCR << std::endl; 
       if( pt1<1.0) continue;       
       for(int i=0; i<myDec.size(); i++) h_L1Decision[iRangeBin]->Fill(myDec[i]);
       int iTrkEtaBin=-1, iTrkMomBin=-1;
       for(int ieta=0; ieta<NEtaBins; ieta++)   {
	 if(std::abs(eta1)>genPartEtaBins[ieta] && std::abs(eta1)<genPartEtaBins[ieta+1] ) iTrkEtaBin = ieta;
       }
       for(int ipt=0;  ipt<NPBins;   ipt++)  {
	 if( p1>genPartPBins[ipt] &&  p1<genPartPBins[ipt+1] )  iTrkMomBin = ipt;
       }
       h_maxNearPIsoHCR_allbins[iRangeBin]->Fill( maxNearPIsoHCR );
       if( iTrkMomBin>=0) h_maxNearPIsoHCR_pbins[iTrkMomBin][iRangeBin]->Fill( maxNearPIsoHCR );

       if( maxNearP31x31<0 )  h_trkP_iso31x31[iRangeBin]->Fill(p1);
       if( maxNearP25x25<0 )  h_trkP_iso25x25[iRangeBin]->Fill(p1);
       if( maxNearP21x21<0 )  h_trkP_iso21x21[iRangeBin]->Fill(p1);
       if( maxNearP15x15<0 )  h_trkP_iso15x15[iRangeBin]->Fill(p1);
       if( maxNearP11x11<0 )  h_trkP_iso11x11[iRangeBin]->Fill(p1);
       // Optimize Charge Isolation
       if( iTrkMomBin>=0 && iTrkEtaBin>=0 ) {
	 nTrk_Bins++;
	 h_maxNearP31x31[iTrkMomBin][iTrkEtaBin][iRangeBin]->Fill( maxNearP31x31 );
	 h_maxNearP25x25[iTrkMomBin][iTrkEtaBin][iRangeBin]->Fill( maxNearP25x25 );
	 h_maxNearP21x21[iTrkMomBin][iTrkEtaBin][iRangeBin]->Fill( maxNearP21x21 );
	 h_maxNearP15x15[iTrkMomBin][iTrkEtaBin][iRangeBin]->Fill( maxNearP15x15 );
	 h_maxNearP11x11[iTrkMomBin][iTrkEtaBin][iRangeBin]->Fill( maxNearP11x11 );

	 h_maxNearPIsoR[iTrkMomBin][iTrkEtaBin][iRangeBin]->Fill( maxNearPIsoR );
	 h_maxNearPIsoHCR[iTrkMomBin][iTrkEtaBin][iRangeBin]->Fill( maxNearPIsoHCR );

	 // dR cut from trigger object
	 double dR=-999.0;
	 if (checkL1) {
	   dR = DeltaR( (*t_isoTrkEta)[itrk],(*t_isoTrkPhi)[itrk], leadL1JetEta, leadL1JetPhi);
	 } else if (!checkTest) {
	   dR = 999.0;
	 }

	 //===================================================================================================
	 if (debug) std::cout << " maxNearP31x31 " << maxNearP31x31;
	 if (maxNearP31x31<0) {
	   nTrk_maxNearP++;
	   double etotal1 = (*t_photonEne31x31)[itrk]+(*t_cHadronEne31x31_1)[itrk]+(*t_nHadronEne31x31)[itrk];
	   h_photon_iso31x31[iTrkMomBin][iTrkEtaBin][iRangeBin]        ->Fill( (*t_photonEne31x31)[itrk] );
	   h_charged_iso31x31[iTrkMomBin][iTrkEtaBin][iRangeBin]       ->Fill( (*t_cHadronEne31x31_1)[itrk] );
	   h_neutral_iso31x31[iTrkMomBin][iTrkEtaBin][iRangeBin]       ->Fill( (*t_nHadronEne31x31)[itrk] );
	   h_contamination_iso31x31[iTrkMomBin][iTrkEtaBin][iRangeBin] ->Fill( etotal1 );
	   
	   double etotal2 = (*t_photonEne11x11)[itrk]+(*t_cHadronEne11x11_1)[itrk]+(*t_nHadronEne11x11)[itrk];
	   h_photon11x11_iso31x31[iTrkMomBin][iTrkEtaBin][iRangeBin]       ->Fill( (*t_photonEne11x11)[itrk] );
	   h_charged11x11_iso31x31[iTrkMomBin][iTrkEtaBin][iRangeBin]      ->Fill( (*t_cHadronEne11x11_1)[itrk] );
	   h_neutral11x11_iso31x31[iTrkMomBin][iTrkEtaBin][iRangeBin]      ->Fill( (*t_nHadronEne11x11)[itrk] );
	   h_contamination11x11_iso31x31[iTrkMomBin][iTrkEtaBin][iRangeBin]->Fill( etotal2 );

	   bool eNeutIso = (etotal1-etotal2 < 0.1);
	   if (debug) std::cout << " etotal1 " << etotal1  << " etotal1 " << etotal1 << " eNeutIso " << eNeutIso;
	   if (eNeutIso) {
	     nIsoTrk++;
	     h_photon11x11_isoEcal_NxN[iTrkMomBin][iTrkEtaBin][iRangeBin]       ->Fill( (*t_photonEne11x11)[itrk] );
	     h_charged11x11_isoEcal_NxN[iTrkMomBin][iTrkEtaBin][iRangeBin]      ->Fill( (*t_cHadronEne11x11_1)[itrk] );
	     h_neutral11x11_isoEcal_NxN[iTrkMomBin][iTrkEtaBin][iRangeBin]      ->Fill( (*t_nHadronEne11x11)[itrk] );
	     h_contamination11x11_isoEcal_NxN[iTrkMomBin][iTrkEtaBin][iRangeBin]->Fill( etotal2 );
	   } 
	   for(int i=0; i<myDec.size(); i++) {
	     h_L1_iso31x31[iTrkMomBin][iTrkEtaBin][iRangeBin]->Fill(myDec[i]);
	     if( (*t_photonEne11x11)[itrk] > 0.1)  h_L1_iso31x31_isoPhoton_11x11_1[iTrkMomBin][iTrkEtaBin][iRangeBin]->Fill(myDec[i]);
	     else                                  h_L1_iso31x31_isoPhoton_11x11_2[iTrkMomBin][iTrkEtaBin][iRangeBin]->Fill(myDec[i]);
	     if( (*t_nHadronEne11x11)[itrk] > 0.1) h_L1_iso31x31_isoNeutral_11x11_1[iTrkMomBin][iTrkEtaBin][iRangeBin]->Fill(myDec[i]);
	     else                                  h_L1_iso31x31_isoNeutral_11x11_2[iTrkMomBin][iTrkEtaBin][iRangeBin]->Fill(myDec[i]);
	   }
	   if (debug) std::cout << " maxNearPHC7x7 " << maxNearPHC7x7 <<" ";;
	   if (maxNearPHC7x7<0) {
	     double htotal1 = (*t_photonEneHC7x7)[itrk]+(*t_cHadronEneHC7x7_1)[itrk]+(*t_nHadronEneHC7x7)[itrk];
	     double htotal2 = (*t_photonEneHC3x3)[itrk]+(*t_cHadronEneHC3x3_1)[itrk]+(*t_nHadronEneHC3x3)[itrk];
	     bool hNeutIso = (htotal1-htotal2 < 0.1);
	     if (eNeutIso && hNeutIso) {
	       h_photonHC5x5_IsoNxN[iTrkMomBin][iTrkEtaBin][iRangeBin]       ->Fill( (*t_photonEneHC5x5)[itrk] );
	       h_chargedHC5x5_IsoNxN[iTrkMomBin][iTrkEtaBin][iRangeBin]      ->Fill( (*t_cHadronEneHC5x5_1)[itrk] );
	       h_neutralHC5x5_IsoNxN[iTrkMomBin][iTrkEtaBin][iRangeBin]      ->Fill( (*t_nHadronEneHC5x5)[itrk] );
	       h_contaminationHC5x5_IsoNxN[iTrkMomBin][iTrkEtaBin][iRangeBin]->Fill( (*t_photonEneHC5x5)[itrk]+(*t_cHadronEneHC5x5_1)[itrk]+(*t_nHadronEneHC5x5)[itrk] );
	     }
	     if (debug) std::cout << " dR(dRcut) " << dR << "(" << dRCut << ")" << std::endl;
	     if ((dR > dRCut) && eNeutIso && hNeutIso) {
	       if( std::abs(pdgid1) == 211 ) {
		 h_trkPIsoNxN  [0][iRangeBin] ->Fill(p1);
		 h_trkPtIsoNxN [0][iRangeBin] ->Fill(pt1);
		 h_trkEtaIsoNxN[0][iRangeBin] ->Fill(eta1);
		 h_trkPhiIsoNxN[0][iRangeBin] ->Fill(phi1);
	       } else if (std::abs(pdgid1)==321 ) {
		 h_trkPIsoNxN  [1][iRangeBin] ->Fill(p1);
		 h_trkPtIsoNxN [1][iRangeBin] ->Fill(pt1);
		 h_trkEtaIsoNxN[1][iRangeBin] ->Fill(eta1);
		 h_trkPhiIsoNxN[1][iRangeBin] ->Fill(phi1);
	       } else if (pdgid1==2212 ) {
		 h_trkPIsoNxN  [2][iRangeBin] ->Fill(p1);
		 h_trkPtIsoNxN [2][iRangeBin] ->Fill(pt1);
		 h_trkEtaIsoNxN[2][iRangeBin] ->Fill(eta1);
		 h_trkPhiIsoNxN[2][iRangeBin] ->Fill(phi1);
	       } else if (pdgid1==-2212 ) {
		 h_trkPIsoNxN  [3][iRangeBin] ->Fill(p1);
		 h_trkPtIsoNxN [3][iRangeBin] ->Fill(pt1);
		 h_trkEtaIsoNxN[3][iRangeBin] ->Fill(eta1);
		 h_trkPhiIsoNxN[3][iRangeBin] ->Fill(phi1);
	       }
	     }
	   }

	 } // if isolated in 31x31

	 //===================================================================================================
	 //////// CHARGE ISOLATION CHANGED TO 2GEV
	 if (maxNearPIsoR<0) {
	   double etotal1_R = (*t_photonEneIsoR)[itrk]+(*t_cHadronEneIsoR_1)[itrk]+(*t_nHadronEneIsoR)[itrk];
	   double etotal2_R = (*t_photonEneR)[itrk]+(*t_cHadronEneR_1)[itrk]+(*t_nHadronEneR)[itrk];
	   bool eNeutIsoR = (etotal1_R-etotal2_R < 0.1);
	   if (eNeutIsoR) {
	     h_photonR_isoEcal_R[iTrkMomBin][iTrkEtaBin][iRangeBin]       ->Fill( (*t_photonEneR)[itrk] );
	     h_chargedR_isoEcal_R[iTrkMomBin][iTrkEtaBin][iRangeBin]      ->Fill( (*t_cHadronEneR_1)[itrk] );
	     h_neutralR_isoEcal_R[iTrkMomBin][iTrkEtaBin][iRangeBin]      ->Fill( (*t_nHadronEneR)[itrk] );
	     h_contaminationR_isoEcal_R[iTrkMomBin][iTrkEtaBin][iRangeBin]->Fill( etotal2_R );
	   }
	   if (maxNearPIsoHCR<0) {
	     double htotal1_R = (*t_photonEneIsoHCR)[itrk]+(*t_cHadronEneIsoHCR_1)[itrk]+(*t_nHadronEneIsoHCR)[itrk];
	     double htotal2_R = (*t_photonEneHCR)[itrk]+(*t_cHadronEneHCR_1)[itrk]+(*t_nHadronEneHCR)[itrk];
	     bool hNeutIsoR = (htotal1_R-htotal2_R < 0.1);
	     if (eNeutIsoR && hNeutIsoR) {
	       h_photonHCR_IsoR[iTrkMomBin][iTrkEtaBin][iRangeBin]       ->Fill( (*t_photonEneHCR)[itrk] );
	       h_chargedHCR_IsoR[iTrkMomBin][iTrkEtaBin][iRangeBin]      ->Fill( (*t_cHadronEneHCR_1)[itrk] );
	       h_neutralHCR_IsoR[iTrkMomBin][iTrkEtaBin][iRangeBin]      ->Fill( (*t_nHadronEneHCR)[itrk] );
	       h_contaminationHCR_IsoR[iTrkMomBin][iTrkEtaBin][iRangeBin]->Fill( (*t_photonEneHCR)[itrk]+(*t_cHadronEneHCR_1)[itrk]+(*t_nHadronEneHCR)[itrk] );
	     }
	     if ((dR > dRCut) && eNeutIsoR && hNeutIsoR) {
	       if( std::abs(pdgid1) == 211 ) {
		 h_trkPIsoR  [0][iRangeBin] ->Fill(p1);
		 h_trkPtIsoR [0][iRangeBin] ->Fill(pt1);
		 h_trkEtaIsoR[0][iRangeBin] ->Fill(eta1);
		 h_trkPhiIsoR[0][iRangeBin] ->Fill(phi1);
	       } else if (std::abs(pdgid1)==321 ) {
		 h_trkPIsoR  [1][iRangeBin] ->Fill(p1);
		 h_trkPtIsoR [1][iRangeBin] ->Fill(pt1);
		 h_trkEtaIsoR[1][iRangeBin] ->Fill(eta1);
		 h_trkPhiIsoR[1][iRangeBin] ->Fill(phi1);
	       } else if (pdgid1==2212 ) {
		 h_trkPIsoR  [2][iRangeBin] ->Fill(p1);
		 h_trkPtIsoR [2][iRangeBin] ->Fill(pt1);
		 h_trkEtaIsoR[2][iRangeBin] ->Fill(eta1);
		 h_trkPhiIsoR[2][iRangeBin] ->Fill(phi1);
	       } else if (pdgid1==-2212 ) {
		 h_trkPIsoR  [3][iRangeBin] ->Fill(p1);
		 h_trkPtIsoR [3][iRangeBin] ->Fill(pt1);
		 h_trkEtaIsoR[3][iRangeBin] ->Fill(eta1);
		 h_trkPhiIsoR[3][iRangeBin] ->Fill(phi1);
	       }
	     }
	   }

	 } // if isolated in IsoR (ECAL)
	 //===================================================================================================
	 if( maxNearP25x25<0 ) {
	   double total = (*t_photonEne25x25)[itrk]+(*t_cHadronEne25x25_1)[itrk]+(*t_nHadronEne25x25)[itrk];
	   h_photon_iso25x25[iTrkMomBin][iTrkEtaBin][iRangeBin]       ->Fill( (*t_photonEne25x25)[itrk] );
	   h_charged_iso25x25[iTrkMomBin][iTrkEtaBin][iRangeBin]      ->Fill( (*t_cHadronEne25x25_1)[itrk] );
	   h_neutral_iso25x25[iTrkMomBin][iTrkEtaBin][iRangeBin]      ->Fill( (*t_nHadronEne25x25)[itrk] );
	   h_contamination_iso25x25[iTrkMomBin][iTrkEtaBin][iRangeBin]->Fill( total );
	 }
	 if( maxNearP21x21<0 ) {
	   double total = (*t_photonEne21x21)[itrk]+(*t_cHadronEne21x21_1)[itrk]+(*t_nHadronEne21x21)[itrk];
	   h_photon_iso21x21[iTrkMomBin][iTrkEtaBin][iRangeBin]       ->Fill( (*t_photonEne21x21)[itrk] );
	   h_charged_iso21x21[iTrkMomBin][iTrkEtaBin][iRangeBin]      ->Fill( (*t_cHadronEne21x21_1)[itrk] );
	   h_neutral_iso21x21[iTrkMomBin][iTrkEtaBin][iRangeBin]      ->Fill( (*t_nHadronEne21x21)[itrk] );
	   h_contamination_iso21x21[iTrkMomBin][iTrkEtaBin][iRangeBin]->Fill( total );
	 }
	 if( maxNearP15x15<0 ) {
	   double total = (*t_photonEne15x15)[itrk]+(*t_cHadronEne15x15_1)[itrk]+(*t_nHadronEne15x15)[itrk];
	   h_photon_iso15x15[iTrkMomBin][iTrkEtaBin][iRangeBin]       ->Fill( (*t_photonEne15x15)[itrk] );
	   h_charged_iso15x15[iTrkMomBin][iTrkEtaBin][iRangeBin]      ->Fill( (*t_cHadronEne15x15_1)[itrk] );
	   h_neutral_iso15x15[iTrkMomBin][iTrkEtaBin][iRangeBin]      ->Fill( (*t_nHadronEne15x15)[itrk] );
	   h_contamination_iso15x15[iTrkMomBin][iTrkEtaBin][iRangeBin]->Fill( total );
	 }
	 if( maxNearP11x11<0 ) {
	   double total = (*t_photonEne11x11)[itrk]+(*t_cHadronEne11x11_1)[itrk]+(*t_nHadronEne11x11)[itrk];
	   h_photon_iso11x11[iTrkMomBin][iTrkEtaBin][iRangeBin]       ->Fill( (*t_photonEne11x11)[itrk] );
	   h_charged_iso11x11[iTrkMomBin][iTrkEtaBin][iRangeBin]      ->Fill( (*t_cHadronEne11x11_1)[itrk] );
	   h_neutral_iso11x11[iTrkMomBin][iTrkEtaBin][iRangeBin]      ->Fill( (*t_nHadronEne11x11)[itrk] );
	   h_contamination_iso11x11[iTrkMomBin][iTrkEtaBin][iRangeBin]->Fill( total );
	 }
	 //===================================================================================================
       }
    }    
  } // loop over entries
  std::cout << "number of tracks " << nTrk << std::endl
	    << "number of tracks selected  in bins " << nTrk_Bins << std::endl
	    << "number of tracks selected in maxnearP " << nTrk_maxNearP << std::endl
	    << "number of isolated tracks " << nIsoTrk << std::endl;
}

Bool_t TreeAnalysisReadGen::Notify() {

  // The Notify() function is called when a new file is opened. This
  // can be either for a new TTree in a TChain or when when a new TTree
  // is started when using PROOF. It is normally not necessary to make changes
  // to the generated code, but the routine can be extended by the
  // user if needed. The return value is currently not used.
  
  return kTRUE;
}

void TreeAnalysisReadGen::Show(Long64_t entry) {

  // Print contents of entry.
  // If entry is not specified, print current entry
  if (!fChain) return;
  fChain->Show(entry);
}

void TreeAnalysisReadGen::getL1Names(){

  l1Names.insert( std::pair<std::string,int>("L1_SingleJet6"    ,15) );
  l1Names.insert( std::pair<std::string,int>("L1_SingleJet20"   ,17) );
  l1Names.insert( std::pair<std::string,int>("L1_SingleJet30"   ,18) );
  l1Names.insert( std::pair<std::string,int>("L1_SingleJet40"   ,19) );
  l1Names.insert( std::pair<std::string,int>("L1_SingleJet50"   ,20) );
  l1Names.insert( std::pair<std::string,int>("L1_SingleJet60"   ,21) );
  l1Names.insert( std::pair<std::string,int>("L1_SingleTauJet10",30) );
  l1Names.insert( std::pair<std::string,int>("L1_SingleTauJet20",31) );
  l1Names.insert( std::pair<std::string,int>("L1_SingleTauJet30",32) );
  l1Names.insert( std::pair<std::string,int>("L1_SingleTauJet50",33) );
  l1Names.insert( std::pair<std::string,int>("L1_SingleIsoEG5"  ,40) );
  l1Names.insert( std::pair<std::string,int>("L1_SingleIsoEG8"  ,41) );
  l1Names.insert( std::pair<std::string,int>("L1_SingleIsoEG10" ,42) );
  l1Names.insert( std::pair<std::string,int>("L1_SingleIsoEG12" ,43) );
  l1Names.insert( std::pair<std::string,int>("L1_SingleIsoEG15" ,44) );
  l1Names.insert( std::pair<std::string,int>("L1_SingleEG1"     ,45) );
  l1Names.insert( std::pair<std::string,int>("L1_SingleEG2"     ,46) );
  l1Names.insert( std::pair<std::string,int>("L1_SingleEG5"     ,47) );
  l1Names.insert( std::pair<std::string,int>("L1_SingleEG8"     ,48) );
  l1Names.insert( std::pair<std::string,int>("L1_SingleEG10"    ,49) );
  l1Names.insert( std::pair<std::string,int>("L1_SingleEG12"    ,50) );
  l1Names.insert( std::pair<std::string,int>("L1_SingleEG15"    ,51) );
  l1Names.insert( std::pair<std::string,int>("L1_SingleEG20"    ,52) );
  l1Names.insert( std::pair<std::string,int>("L1_SingleMu0"     ,56) );
  l1Names.insert( std::pair<std::string,int>("L1_SingleMu3"     ,57) );
  l1Names.insert( std::pair<std::string,int>("L1_SingleMu5"     ,58) );
  l1Names.insert( std::pair<std::string,int>("L1_SingleMu7"     ,59) );
  l1Names.insert( std::pair<std::string,int>("L1_SingleMu10"    ,60) );
  l1Names.insert( std::pair<std::string,int>("L1_SingleMu14"    ,61) );
  l1Names.insert( std::pair<std::string,int>("L1_SingleMu20"    ,62) );  
}

void TreeAnalysisReadGen::BookHistograms(const char *outFileName, std::vector<std::string>& ranges) {

  fout = new TFile(outFileName, "RECREATE");
  fout->cd();
  char name[100], hname[100], htit[100];
  // inclusive distributions
  fout->cd();
  TDirectory *d_maxNearP              = fout->mkdir( "MaxNearP" );
  TDirectory *d_chargeIso31x31        = fout->mkdir( "chargeIso31x31" );
  TDirectory *d_chargeIso25x25        = fout->mkdir( "chargeIso25x25" );
  TDirectory *d_chargeIso21x21        = fout->mkdir( "chargeIso21x21" );
  TDirectory *d_chargeIso15x15        = fout->mkdir( "chargeIso15x15" );
  TDirectory *d_chargeIso11x11        = fout->mkdir( "chargeIso11x11" );
  TDirectory *d_E11x11_chargeIso31x31 = fout->mkdir( "signalE11x11_chargeIso31x31" );
  TDirectory *d_R_chargeIsoIsoR       = fout->mkdir( "signalR_chargeIsoIsoR" );
  TDirectory *d_H5x5_IsoNxN           = fout->mkdir( "d_H5x5_IsoNxN" );
  TDirectory *d_HCR_IsoR              = fout->mkdir( "d_HCR_IsoR" );
  TDirectory *d_trigger = fout->mkdir("trigger");
  TDirectory *d_inclusive           = fout->mkdir( "InclusiveTracks" ); 
  std::string PNames[PTypes] = {"Pions", "Kaons", "Protons", "AntiProtons"};
  for (unsigned int j=0; j<ranges.size()+1; j++) {
    if(j==ranges.size()) sprintf(name, "all");
    else sprintf(name, "%s", ranges[j].c_str());
    d_inclusive ->cd();
    for(int itype=0; itype<PTypes; itype++){
      sprintf(hname, "h_trkPAll_%i_%s",itype, name);
      sprintf(htit,  "tracks : P(%s)_%s", PNames[itype].c_str(), name);
      h_trkPAll[itype][j] = new TH1F(hname, htit, NPBins, genPartPBins);
      h_trkPAll[itype][j]->Sumw2();
      
      sprintf(hname, "h_trkPtAll_%i_%s",itype, name);
      sprintf(htit,  "tracks : Pt(%s)_%s", PNames[itype].c_str(), name);
      h_trkPtAll[itype][j] = new TH1F(hname, htit, NPBins, genPartPBins);
      h_trkPtAll[itype][j]->Sumw2();

      sprintf(hname, "h_trkEtaAll_%i_%s",itype, name);
      sprintf(htit,  "tracks : Eta(%s)_%s", PNames[itype].c_str(), name);
      h_trkEtaAll[itype][j] = new TH1F(hname, htit, 200, -10.0, 10.0);
      h_trkEtaAll[itype][j]->Sumw2();

      sprintf(hname, "h_trkPhiAll_%i_%s",itype, name);
      sprintf(htit,  "tracks : Phi(%s)_%s", PNames[itype].c_str(), name);
      h_trkPhiAll[itype][j] = new TH1F(hname, htit, 100, -5.0, 5.0);    
      h_trkPhiAll[itype][j]->Sumw2();
      
      sprintf(hname, "h_trkPIsoNxN_%i_%s",itype, name);
      sprintf(htit,  "tracks : P(%s)_%s", PNames[itype].c_str(), name);
      h_trkPIsoNxN[itype][j] = new TH1F(hname, htit, NPBins, genPartPBins);
      h_trkPIsoNxN[itype][j]->Sumw2();

      sprintf(hname, "h_trkPtIsoNxN_%i_%s",itype, name);
      sprintf(htit,  "tracks : Pt(%s)_%s", PNames[itype].c_str(), name);
      h_trkPtIsoNxN[itype][j] = new TH1F(hname, htit, NPBins, genPartPBins);
      h_trkPtIsoNxN[itype][j]->Sumw2();
      
      sprintf(hname, "h_trkEtaIsoNxN_%i_%s",itype, name);
      sprintf(htit,  "tracks : Eta(%s)_%s", PNames[itype].c_str(), name);
      h_trkEtaIsoNxN[itype][j] = new TH1F(hname, htit, 100, -5.0, 5.0);
      h_trkEtaIsoNxN[itype][j]->Sumw2();
      
      sprintf(hname, "h_trkPhiIsoNxN_%i_%s",itype, name);
      sprintf(htit,  "tracks : Phi(%s)_%s", PNames[itype].c_str(), name);
      h_trkPhiIsoNxN[itype][j] = new TH1F(hname, htit, 100, -5.0, 5.0);    
      h_trkPhiIsoNxN[itype][j]->Sumw2();
      
      sprintf(hname, "h_trkPIsoR_%i_%s",itype, name);
      sprintf(htit,  "tracks : P(%s)_%s", PNames[itype].c_str(), name);
      h_trkPIsoR[itype][j] = new TH1F(hname, htit, NPBins, genPartPBins);
      h_trkPIsoR[itype][j]->Sumw2();
      
      sprintf(hname, "h_trkPtIsoR_%i_%s",itype, name);
      sprintf(htit,  "tracks : Pt(%s)_%s", PNames[itype].c_str(), name);
      h_trkPtIsoR[itype][j] = new TH1F(hname, htit, NPBins, genPartPBins);
      h_trkPtIsoR[itype][j]->Sumw2();
      
      sprintf(hname, "h_trkEtaIsoR_%i_%s",itype, name);
      sprintf(htit,  "tracks : Eta(%s)_%s", PNames[itype].c_str(), name);
      h_trkEtaIsoR[itype][j] = new TH1F(hname, htit, 100, -5.0, 5.0);
      h_trkEtaIsoR[itype][j]->Sumw2();
      
      sprintf(hname, "h_trkPhiIsoR_%i_%s",itype, name);
      sprintf(htit,  "tracks : Phi(%s)_%s", PNames[itype].c_str(), name);
      h_trkPhiIsoR[itype][j] = new TH1F(hname, htit, 100, -5.0, 5.0);    
      h_trkPhiIsoR[itype][j]->Sumw2();
    }
    for(int ieta=0; ieta<NEtaBins; ieta++) {
      double lowEta=-5.0, highEta= 5.0;
      lowEta  = genPartEtaBins[ieta];
      highEta = genPartEtaBins[ieta+1];
      
      for(int ipt=0; ipt<NPBins; ipt++) {
	double lowP=0.0, highP=300.0;
	lowP    = genPartPBins[ipt];
	highP   = genPartPBins[ipt+1];
	
	sprintf(hname, "h_trkDEta_ptBin%i_etaBin%i_%s",ipt, ieta, name);
	sprintf(htit,  "#Delta(#eta(track),#eta(EcalImpactPoint)) (%3.2f<|#eta|<%3.2f), (%2.0f<trkP<%3.0f) for %s", lowEta, highEta, lowP, highP , name);
	h_trkDEta[ipt][ieta][j] = new TH1F(hname, htit, 250, -0.5, 0.5);
	h_trkDEta[ipt][ieta][j]->Sumw2();
	
	sprintf(hname, "h_trkDPhi_ptBin%i_etaBin%i_%s",ipt, ieta, name);
	sprintf(htit,  "#Delta(#phi(track),#phi(EcalImpactPoint)) (%3.2f<|#eta|<%3.2f), (%2.0f<trkP<%3.0f) for %s", lowEta, highEta, lowP, highP , name);
	h_trkDPhi[ipt][ieta][j] = new TH1F(hname, htit, 350, -0.2, 1.5);
	h_trkDPhi[ipt][ieta][j]->Sumw2();
      }
    }
    
    sprintf(hname, "h_trkP_iso31x31_%s", name);
    h_trkP_iso31x31[j] = new TH1F(hname, hname, NPBins, genPartPBins);
    h_trkP_iso31x31[j]->Sumw2();
    sprintf(hname, "h_trkP_iso25x25_%s", name);
    h_trkP_iso25x25[j] = new TH1F(hname, hname, NPBins, genPartPBins);
    h_trkP_iso25x25[j]->Sumw2();
    sprintf(hname, "h_trkP_iso21x21_%s", name);
    h_trkP_iso21x21[j] = new TH1F(hname, hname, NPBins, genPartPBins);
    h_trkP_iso21x21[j]->Sumw2();
    sprintf(hname, "h_trkP_iso15x15_%s", name);
    h_trkP_iso15x15[j] = new TH1F(hname, hname, NPBins, genPartPBins);
    h_trkP_iso15x15[j]->Sumw2();
    sprintf(hname, "h_trkP_iso11x11_%s", name);
    h_trkP_iso11x11[j] = new TH1F(hname, hname, NPBins, genPartPBins);
    h_trkP_iso11x11[j]->Sumw2();
    
    sprintf(hname, "h_L1Decision_%s", name);
    h_L1Decision[j]    = new TH1F(hname,    hname,    10, -0.5, 9.5);
    h_L1Decision[j]->Sumw2();
    h_L1Decision[j]->GetXaxis()->SetBinLabel(1,"L1SingleJet");
    h_L1Decision[j]->GetXaxis()->SetBinLabel(2,"L1SingleTauJet");
    h_L1Decision[j]->GetXaxis()->SetBinLabel(3,"L1SingleEG");
    h_L1Decision[j]->GetXaxis()->SetBinLabel(4,"L1SingleIsoEG");
    h_L1Decision[j]->GetXaxis()->SetBinLabel(5,"L1SingleMu");
    
    d_maxNearP->cd();
    sprintf(hname, "h_maxNearPIsoHCR_allbins_%s", name);
    h_maxNearPIsoHCR_allbins[j] = new TH1F(hname, hname, 220, -2.0, 100.0);
    h_maxNearPIsoHCR_allbins[j]->Sumw2();

    for(int ipt=0; ipt<NPBins; ipt++) {
      double lowP=0.0, highP=300.0;
      lowP    = genPartPBins[ipt];
      highP   = genPartPBins[ipt+1];
      
      sprintf(hname, "h_maxNearPIsoHCR_ptBin%i_%s",ipt, name);
      sprintf(htit,  "maxNearP in IsoHCR (%2.0f<trkP<%3.0f) for %s", lowP, highP, name);
      h_maxNearPIsoHCR_pbins[ipt][j] = new TH1F(hname, htit, 220, -2.0, 100.0);
      h_maxNearPIsoHCR_pbins[ipt][j]->Sumw2();
    }
    for(int ieta=0; ieta<NEtaBins; ieta++) {
      double lowEta=-5.0, highEta= 5.0;
      lowEta  = genPartEtaBins[ieta];
      highEta = genPartEtaBins[ieta+1];
      
      for(int ipt=0; ipt<NPBins; ipt++) {
	double lowP=0.0, highP=300.0;
	lowP    = genPartPBins[ipt];
	highP   = genPartPBins[ipt+1];
	
	d_maxNearP->cd();
	sprintf(hname, "h_maxNearPIsoR_ptBin%i_etaBin%i_%s",ipt, ieta, name);
	sprintf(htit,  "maxNearP in IsoR (%3.2f<|#eta|<%3.2f), (%2.0f<trkP<%3.0f) for %s", lowEta, highEta, lowP, highP, name);
	h_maxNearPIsoR[ipt][ieta][j] = new TH1F(hname, htit, 220, -2.0, 100.0);
	h_maxNearPIsoR[ipt][ieta][j]->Sumw2();
	sprintf(hname, "h_maxNearPIsoHCR_ptBin%i_etaBin%i_%s",ipt, ieta, name);
	sprintf(htit,  "maxNearP in IsoHCR (%3.2f<|#eta|<%3.2f), (%2.0f<trkP<%3.0f) for %s", lowEta, highEta, lowP, highP, name);
	h_maxNearPIsoHCR[ipt][ieta][j] = new TH1F(hname, htit, 220, -2.0, 100.0);
	h_maxNearPIsoHCR[ipt][ieta][j]->Sumw2();

	sprintf(hname, "h_maxNearP31x31_ptBin%i_etaBin%i_%s",ipt, ieta, name);
	sprintf(htit,  "maxNearP in 31x31 (%3.2f<|#eta|<%3.2f), (%2.0f<trkP<%3.0f) for %s", lowEta, highEta, lowP, highP, name);
	h_maxNearP31x31[ipt][ieta][j] = new TH1F(hname, htit, 220, -2.0, 100.0);
	h_maxNearP31x31[ipt][ieta][j]->Sumw2();
	sprintf(hname, "h_maxNearP25x25_ptBin%i_etaBin%i_%s",ipt, ieta, name);
	sprintf(htit,  "maxNearP in 25x25 (%3.2f<|#eta|<%3.2f), (%2.0f<trkP<%3.0f) for %s", lowEta, highEta, lowP, highP, name);
	h_maxNearP25x25[ipt][ieta][j] = new TH1F(hname, htit, 220, -2.0, 100.0);
	h_maxNearP25x25[ipt][ieta][j]->Sumw2();
	sprintf(hname, "h_maxNearP21x21_ptBin%i_etaBin%i_%s",ipt, ieta, name);
	sprintf(htit,  "maxNearP in 21x21 (%3.2f<|#eta|<%3.2f), (%2.0f<trkP<%3.0f) for %s", lowEta, highEta, lowP, highP, name);
	h_maxNearP21x21[ipt][ieta][j] = new TH1F(hname, htit, 220, -2.0, 100.0);
	h_maxNearP21x21[ipt][ieta][j]->Sumw2();
	sprintf(hname, "h_maxNearP15x15_ptBin%i_etaBin%i_%s",ipt, ieta, name);
	sprintf(htit,  "maxNearP in 15x15 (%3.2f<|#eta|<%3.2f), (%2.0f<trkP<%3.0f) for %s", lowEta, highEta, lowP, highP, name);
	h_maxNearP15x15[ipt][ieta][j] = new TH1F(hname, htit, 220, -2.0, 100.0);
	h_maxNearP15x15[ipt][ieta][j]->Sumw2();
	sprintf(hname, "h_maxNearP11x11_ptBin%i_etaBin%i_%s",ipt, ieta, name);
	sprintf(htit,  "maxNearP in 11x11 (%3.2f<|#eta|<%3.2f), (%2.0f<trkP<%3.0f) for %s", lowEta, highEta, lowP, highP, name);
	h_maxNearP11x11[ipt][ieta][j] = new TH1F(hname, htit, 220, -2.0, 100.0);
	h_maxNearP11x11[ipt][ieta][j]->Sumw2();


	d_chargeIso31x31->cd();
	sprintf(hname, "h_photon_iso31x31_ptBin%i_etaBin%i_%s",ipt, ieta, name);
	sprintf(htit,  "photon in 31x31 (%3.2f<|#eta|<%3.2f), (%2.0f<trkP<%3.0f) for %s", lowEta, highEta, lowP, highP, name);
	h_photon_iso31x31[ipt][ieta][j] = new TH1F(hname, htit, 220, -2.0, 20.0);
	h_photon_iso31x31[ipt][ieta][j]->Sumw2();
	sprintf(hname, "h_charged_iso31x31_ptBin%i_etaBin%i_%s",ipt, ieta, name);
	sprintf(htit,  "charged in 31x31 (%3.2f<|#eta|<%3.2f), (%2.0f<trkP<%3.0f) for %s", lowEta, highEta, lowP, highP, name);
	h_charged_iso31x31[ipt][ieta][j] = new TH1F(hname, htit, 220, -2.0, 20.0);
	h_charged_iso31x31[ipt][ieta][j]->Sumw2();
	sprintf(hname, "h_neutral_iso31x31_ptBin%i_etaBin%i_%s",ipt, ieta, name);
	sprintf(htit,  "neutral in 31x31 (%3.2f<|#eta|<%3.2f), (%2.0f<trkP<%3.0f) for %s", lowEta, highEta, lowP, highP, name);
	h_neutral_iso31x31[ipt][ieta][j] = new TH1F(hname, htit, 220, -2.0, 20.0);
	h_neutral_iso31x31[ipt][ieta][j]->Sumw2();
	sprintf(hname, "h_contamination_iso31x31_ptBin%i_etaBin%i_%s",ipt, ieta, name);
	sprintf(htit,  "contamination in 31x31 (%3.2f<|#eta|<%3.2f), (%2.0f<trkP<%3.0f) for %s", lowEta, highEta, lowP, highP, name);
	h_contamination_iso31x31[ipt][ieta][j] = new TH1F(hname, htit, 220, -2.0, 20.0);
	h_contamination_iso31x31[ipt][ieta][j]->Sumw2();
	sprintf(hname, "h_L1_iso31x31_ptBin%i_etaBin%i_%s",ipt, ieta, name);
	sprintf(htit,  "L1 (%3.2f<|#eta|<%3.2f), (%2.0f<trkP<%3.0f) for %s", lowEta, highEta, lowP, highP, name);
	h_L1_iso31x31[ipt][ieta][j] = new TH1F(hname, htit, 10, -0.5, 9.5);
	h_L1_iso31x31[ipt][ieta][j]->Sumw2();
	h_L1_iso31x31[ipt][ieta][j]->GetXaxis()->SetBinLabel(1,"L1SingleJet");
	h_L1_iso31x31[ipt][ieta][j]->GetXaxis()->SetBinLabel(2,"L1SingleTauJet");
	h_L1_iso31x31[ipt][ieta][j]->GetXaxis()->SetBinLabel(3,"L1SingleEG");
	h_L1_iso31x31[ipt][ieta][j]->GetXaxis()->SetBinLabel(4,"L1SingleIsoEG");
	h_L1_iso31x31[ipt][ieta][j]->GetXaxis()->SetBinLabel(5,"L1SingleMu");
		
	d_chargeIso25x25->cd();
	sprintf(hname, "h_photon_iso25x25_ptBin%i_etaBin%i_%s",ipt, ieta, name);
	sprintf(htit,  "photon in 25x25 (%3.2f<|#eta|<%3.2f), (%2.0f<trkP<%3.0f) for %s", lowEta, highEta, lowP, highP , name);
	h_photon_iso25x25[ipt][ieta][j] = new TH1F(hname, htit, 220, -2.0, 20.0);
	h_photon_iso25x25[ipt][ieta][j]->Sumw2();
	sprintf(hname, "h_charged_iso25x25_ptBin%i_etaBin%i_%s",ipt, ieta, name);
	sprintf(htit,  "charged in 25x25 (%3.2f<|#eta|<%3.2f), (%2.0f<trkP<%3.0f) for %s", lowEta, highEta, lowP, highP , name);
	h_charged_iso25x25[ipt][ieta][j] = new TH1F(hname, htit, 220, -2.0, 20.0);
	h_charged_iso25x25[ipt][ieta][j]->Sumw2();
	sprintf(hname, "h_neutral_iso25x25_ptBin%i_etaBin%i_%s",ipt, ieta, name);
	sprintf(htit,  "neutral in 25x25 (%3.2f<|#eta|<%3.2f), (%2.0f<trkP<%3.0f) for %s", lowEta, highEta, lowP, highP , name);
	h_neutral_iso25x25[ipt][ieta][j] = new TH1F(hname, htit, 220, -2.0, 20.0);
	h_neutral_iso25x25[ipt][ieta][j]->Sumw2();
	sprintf(hname, "h_contamination_iso25x25_ptBin%i_etaBin%i_%s",ipt, ieta, name);
	sprintf(htit,  "contamination in 25x25 (%3.2f<|#eta|<%3.2f), (%2.0f<trkP<%3.0f) for %s", lowEta, highEta, lowP, highP , name);
	h_contamination_iso25x25[ipt][ieta][j] = new TH1F(hname, htit, 220, -2.0, 20.0);
	h_contamination_iso25x25[ipt][ieta][j]->Sumw2();
	
	d_chargeIso21x21->cd();
	sprintf(hname, "h_photon_iso21x21_ptBin%i_etaBin%i_%s",ipt, ieta, name);
	sprintf(htit,  "photon in 21x21 (%3.2f<|#eta|<%3.2f), (%2.0f<trkP<%3.0f) for %s", lowEta, highEta, lowP, highP , name);
	h_photon_iso21x21[ipt][ieta][j] = new TH1F(hname, htit, 220, -2.0, 20.0);
	h_photon_iso21x21[ipt][ieta][j]->Sumw2();
	sprintf(hname, "h_charged_iso21x21_ptBin%i_etaBin%i_%s",ipt, ieta, name);
	sprintf(htit,  "charged in 21x21 (%3.2f<|#eta|<%3.2f), (%2.0f<trkP<%3.0f) for %s", lowEta, highEta, lowP, highP , name);
	h_charged_iso21x21[ipt][ieta][j] = new TH1F(hname, htit, 220, -2.0, 20.0);
	h_charged_iso21x21[ipt][ieta][j]->Sumw2();
	sprintf(hname, "h_neutral_iso21x21_ptBin%i_etaBin%i_%s",ipt, ieta, name);
	sprintf(htit,  "neutral in 21x21 (%3.2f<|#eta|<%3.2f), (%2.0f<trkP<%3.0f) for %s", lowEta, highEta, lowP, highP , name);
	h_neutral_iso21x21[ipt][ieta][j] = new TH1F(hname, htit, 220, -2.0, 20.0);
	h_neutral_iso21x21[ipt][ieta][j]->Sumw2();
	sprintf(hname, "h_contamination_iso21x21_ptBin%i_etaBin%i_%s",ipt, ieta, name);
	sprintf(htit,  "contamination in 21x21 (%3.2f<|#eta|<%3.2f), (%2.0f<trkP<%3.0f) for %s", lowEta, highEta, lowP, highP , name);
	h_contamination_iso21x21[ipt][ieta][j] = new TH1F(hname, htit, 220, -2.0, 20.0);
	h_contamination_iso21x21[ipt][ieta][j]->Sumw2();
	
	d_chargeIso15x15->cd();
	sprintf(hname, "h_photon_iso15x15_ptBin%i_etaBin%i_%s",ipt, ieta, name);
	sprintf(htit,  "photon in 15x15 (%3.2f<|#eta|<%3.2f), (%2.0f<trkP<%3.0f) for %s", lowEta, highEta, lowP, highP , name);
	h_photon_iso15x15[ipt][ieta][j] = new TH1F(hname, htit, 220, -2.0, 20.0);
	h_photon_iso15x15[ipt][ieta][j]->Sumw2();
	sprintf(hname, "h_charged_iso15x15_ptBin%i_etaBin%i_%s",ipt, ieta, name);
	sprintf(htit,  "charged in 15x15 (%3.2f<|#eta|<%3.2f), (%2.0f<trkP<%3.0f) for %s", lowEta, highEta, lowP, highP , name);
	h_charged_iso15x15[ipt][ieta][j] = new TH1F(hname, htit, 220, -2.0, 20.0);
	h_charged_iso15x15[ipt][ieta][j]->Sumw2();

	sprintf(hname, "h_neutral_iso15x15_ptBin%i_etaBin%i_%s",ipt, ieta, name);
	sprintf(htit,  "neutral in 15x15 (%3.2f<|#eta|<%3.2f), (%2.0f<trkP<%3.0f) for %s", lowEta, highEta, lowP, highP , name);
	h_neutral_iso15x15[ipt][ieta][j] = new TH1F(hname, htit, 220, -2.0, 20.0);
	h_neutral_iso15x15[ipt][ieta][j]->Sumw2();
	sprintf(hname, "h_contamination_iso15x15_ptBin%i_etaBin%i_%s",ipt, ieta, name);
	sprintf(htit,  "contamination in 15x15 (%3.2f<|#eta|<%3.2f), (%2.0f<trkP<%3.0f) for %s", lowEta, highEta, lowP, highP , name);
	h_contamination_iso15x15[ipt][ieta][j] = new TH1F(hname, htit, 220, -2.0, 20.0);
	h_contamination_iso15x15[ipt][ieta][j]->Sumw2();
	
	d_chargeIso11x11->cd();
	sprintf(hname, "h_photon_iso11x11_ptBin%i_etaBin%i_%s",ipt, ieta, name);
	sprintf(htit,  "photon in 11x11 (%3.2f<|#eta|<%3.2f), (%2.0f<trkP<%3.0f) for %s", lowEta, highEta, lowP, highP , name);
	h_photon_iso11x11[ipt][ieta][j] = new TH1F(hname, htit, 220, -2.0, 20.0);
	h_photon_iso11x11[ipt][ieta][j]->Sumw2();
	sprintf(hname, "h_charged_iso11x11_ptBin%i_etaBin%i_%s",ipt, ieta, name);
	sprintf(htit,  "charged in 11x11 (%3.2f<|#eta|<%3.2f), (%2.0f<trkP<%3.0f) for %s", lowEta, highEta, lowP, highP , name);
	h_charged_iso11x11[ipt][ieta][j] = new TH1F(hname, htit, 220, -2.0, 20.0);
	h_charged_iso11x11[ipt][ieta][j]->Sumw2();
	sprintf(hname, "h_neutral_iso11x11_ptBin%i_etaBin%i_%s",ipt, ieta, name);
	sprintf(htit,  "neutral in 11x11 (%3.2f<|#eta|<%3.2f), (%2.0f<trkP<%3.0f) for %s", lowEta, highEta, lowP, highP , name);
	h_neutral_iso11x11[ipt][ieta][j] = new TH1F(hname, htit, 220, -2.0, 20.0);
	h_neutral_iso11x11[ipt][ieta][j]->Sumw2();
	sprintf(hname, "h_contamination_iso11x11_ptBin%i_etaBin%i_%s",ipt, ieta, name);
	sprintf(htit,  "contamination in 11x11 (%3.2f<|#eta|<%3.2f), (%2.0f<trkP<%3.0f) for %s", lowEta, highEta, lowP, highP , name);
	h_contamination_iso11x11[ipt][ieta][j] = new TH1F(hname, htit, 220, -2.0, 20.0);
	h_contamination_iso11x11[ipt][ieta][j]->Sumw2();
	
	d_E11x11_chargeIso31x31->cd();
	sprintf(hname, "h_photon11x11_iso31x31_ptBin%i_etaBin%i_%s",ipt, ieta, name);
	sprintf(htit,  "photon in 11x11 (iso31x31) (%3.2f<|#eta|<%3.2f), (%2.0f<trkP<%3.0f) for %s", lowEta, highEta, lowP, highP , name);
	h_photon11x11_iso31x31[ipt][ieta][j] = new TH1F(hname, htit, 220, -2.0, 20.0);
	h_photon11x11_iso31x31[ipt][ieta][j]->Sumw2();
	sprintf(hname, "h_charged11x11_iso31x31_ptBin%i_etaBin%i_%s",ipt, ieta, name);
	sprintf(htit,  "charged in 11x11 (iso31x31) (%3.2f<|#eta|<%3.2f), (%2.0f<trkP<%3.0f) for %s", lowEta, highEta, lowP, highP , name);
	h_charged11x11_iso31x31[ipt][ieta][j] = new TH1F(hname, htit, 220, -2.0, 20.0);
	h_charged11x11_iso31x31[ipt][ieta][j]->Sumw2();
	sprintf(hname, "h_neutral11x11_iso31x31_ptBin%i_etaBin%i_%s",ipt, ieta, name);
	sprintf(htit,  "neutral in 11x11 (iso31x31) (%3.2f<|#eta|<%3.2f), (%2.0f<trkP<%3.0f) for %s", lowEta, highEta, lowP, highP , name);
	h_neutral11x11_iso31x31[ipt][ieta][j] = new TH1F(hname, htit, 220, -2.0, 20.0);
	h_neutral11x11_iso31x31[ipt][ieta][j]->Sumw2();
	sprintf(hname, "h_contamination11x11_iso31x31_ptBin%i_etaBin%i_%s",ipt, ieta, name);
	sprintf(htit,  "contamination in 11x11 (iso31x31) (%3.2f<|#eta|<%3.2f), (%2.0f<trkP<%3.0f) for %s", lowEta, highEta, lowP, highP , name);
	h_contamination11x11_iso31x31[ipt][ieta][j] = new TH1F(hname, htit, 220, -2.0, 20.0);
	h_contamination11x11_iso31x31[ipt][ieta][j]->Sumw2();
	sprintf(hname, "h_L1_iso31x31_isoPhoton_11x11_1_ptBin%i_etaBin%i_%s",ipt, ieta, name);
	sprintf(htit,  "L1(iso31x31, photonEne>0)) (%3.2f<|#eta|<%3.2f), (%2.0f<trkP<%3.0f) for %s", lowEta, highEta, lowP, highP , name);
	h_L1_iso31x31_isoPhoton_11x11_1[ipt][ieta][j] = new TH1F(hname, htit, 10, -0.5, 9.5);
	h_L1_iso31x31_isoPhoton_11x11_1[ipt][ieta][j]->Sumw2();
	h_L1_iso31x31_isoPhoton_11x11_1[ipt][ieta][j]->GetXaxis()->SetBinLabel(1,"L1SingleJet");
	h_L1_iso31x31_isoPhoton_11x11_1[ipt][ieta][j]->GetXaxis()->SetBinLabel(2,"L1SingleTauJet");
	h_L1_iso31x31_isoPhoton_11x11_1[ipt][ieta][j]->GetXaxis()->SetBinLabel(3,"L1SingleEG");
	h_L1_iso31x31_isoPhoton_11x11_1[ipt][ieta][j]->GetXaxis()->SetBinLabel(4,"L1SingleIsoEG");
	h_L1_iso31x31_isoPhoton_11x11_1[ipt][ieta][j]->GetXaxis()->SetBinLabel(5,"L1SingleMu");
	sprintf(hname, "h_L1_iso31x31_isoPhoton_11x11_2_ptBin%i_etaBin%i_%s",ipt, ieta, name);
	sprintf(htit,  "L1(iso31x31, photonEne==0)) (%3.2f<|#eta|<%3.2f), (%2.0f<trkP<%3.0f) for %s", lowEta, highEta, lowP, highP , name);
	h_L1_iso31x31_isoPhoton_11x11_2[ipt][ieta][j] = new TH1F(hname, htit, 10, -0.5, 9.5);
	h_L1_iso31x31_isoPhoton_11x11_2[ipt][ieta][j]->Sumw2();
	h_L1_iso31x31_isoPhoton_11x11_2[ipt][ieta][j]->GetXaxis()->SetBinLabel(1,"L1SingleJet");
	h_L1_iso31x31_isoPhoton_11x11_2[ipt][ieta][j]->GetXaxis()->SetBinLabel(2,"L1SingleTauJet");
	h_L1_iso31x31_isoPhoton_11x11_2[ipt][ieta][j]->GetXaxis()->SetBinLabel(3,"L1SingleEG");
	h_L1_iso31x31_isoPhoton_11x11_2[ipt][ieta][j]->GetXaxis()->SetBinLabel(4,"L1SingleIsoEG");
	h_L1_iso31x31_isoPhoton_11x11_2[ipt][ieta][j]->GetXaxis()->SetBinLabel(5,"L1SingleMu");
	sprintf(hname, "h_L1_iso31x31_isoNeutral_11x11_1_ptBin%i_etaBin%i_%s",ipt, ieta, name);
	sprintf(htit,  "L1(iso31x31, photonEne>0)) (%3.2f<|#eta|<%3.2f), (%2.0f<trkP<%3.0f) for %s", lowEta, highEta, lowP, highP , name);
	h_L1_iso31x31_isoNeutral_11x11_1[ipt][ieta][j] = new TH1F(hname, htit, 10, -0.5, 9.5);
	h_L1_iso31x31_isoNeutral_11x11_1[ipt][ieta][j]->Sumw2();
	h_L1_iso31x31_isoNeutral_11x11_1[ipt][ieta][j]->GetXaxis()->SetBinLabel(1,"L1SingleJet");
	h_L1_iso31x31_isoNeutral_11x11_1[ipt][ieta][j]->GetXaxis()->SetBinLabel(2,"L1SingleTauJet");
	h_L1_iso31x31_isoNeutral_11x11_1[ipt][ieta][j]->GetXaxis()->SetBinLabel(3,"L1SingleEG");
	h_L1_iso31x31_isoNeutral_11x11_1[ipt][ieta][j]->GetXaxis()->SetBinLabel(4,"L1SingleIsoEG");
	h_L1_iso31x31_isoNeutral_11x11_1[ipt][ieta][j]->GetXaxis()->SetBinLabel(5,"L1SingleMu");
	sprintf(hname, "h_L1_iso31x31_isoNeutral_11x11_2_ptBin%i_etaBin%i_%s",ipt, ieta, name);
	sprintf(htit,  "L1(iso31x31, photonEne==0)) (%3.2f<|#eta|<%3.2f), (%2.0f<trkP<%3.0f) for %s", lowEta, highEta, lowP, highP , name);
	h_L1_iso31x31_isoNeutral_11x11_2[ipt][ieta][j] = new TH1F(hname, htit, 10, -0.5, 9.5);
	h_L1_iso31x31_isoNeutral_11x11_2[ipt][ieta][j]->Sumw2();
	h_L1_iso31x31_isoNeutral_11x11_2[ipt][ieta][j]->GetXaxis()->SetBinLabel(1,"L1SingleJet");
	h_L1_iso31x31_isoNeutral_11x11_2[ipt][ieta][j]->GetXaxis()->SetBinLabel(2,"L1SingleTauJet");
	h_L1_iso31x31_isoNeutral_11x11_2[ipt][ieta][j]->GetXaxis()->SetBinLabel(3,"L1SingleEG");
	h_L1_iso31x31_isoNeutral_11x11_2[ipt][ieta][j]->GetXaxis()->SetBinLabel(4,"L1SingleIsoEG");
	h_L1_iso31x31_isoNeutral_11x11_2[ipt][ieta][j]->GetXaxis()->SetBinLabel(5,"L1SingleMu");
	
	sprintf(hname, "h_photon11x11_isoEcal_NxN_ptBin%i_etaBin%i_%s",ipt, ieta, name);
	sprintf(htit,  "photon in 11x11 (iso31x31-11x11) (%3.2f<|#eta|<%3.2f), (%2.0f<trkP<%3.0f) for %s", lowEta, highEta, lowP, highP , name);
	h_photon11x11_isoEcal_NxN[ipt][ieta][j] = new TH1F(hname, htit, 220, -2.0, 20.0);
	h_photon11x11_isoEcal_NxN[ipt][ieta][j]->Sumw2();
	sprintf(hname, "h_charged11x11_isoEcal_NxN_ptBin%i_etaBin%i_%s",ipt, ieta, name);
	sprintf(htit,  "charged in 11x11 (iso31x31-11x11) (%3.2f<|#eta|<%3.2f), (%2.0f<trkP<%3.0f) for %s", lowEta, highEta, lowP, highP , name);
	h_charged11x11_isoEcal_NxN[ipt][ieta][j] = new TH1F(hname, htit, 220, -2.0, 20.0);
	h_charged11x11_isoEcal_NxN[ipt][ieta][j]->Sumw2();
	sprintf(hname, "h_neutral11x11_isoEcal_NxN_ptBin%i_etaBin%i_%s",ipt, ieta, name);
	sprintf(htit,  "neutral in 11x11 (iso31x31-11x11) (%3.2f<|#eta|<%3.2f), (%2.0f<trkP<%3.0f) for %s", lowEta, highEta, lowP, highP , name);
	h_neutral11x11_isoEcal_NxN[ipt][ieta][j] = new TH1F(hname, htit, 220, -2.0, 20.0);
	h_neutral11x11_isoEcal_NxN[ipt][ieta][j]->Sumw2();
	sprintf(hname, "h_contamination11x11_isoEcal_NxN_ptBin%i_etaBin%i_%s",ipt, ieta, name);
	sprintf(htit,  "contamination in 11x11 (iso31x31-11x11) (%3.2f<|#eta|<%3.2f), (%2.0f<trkP<%3.0f) for %s", lowEta, highEta, lowP, highP , name);
	h_contamination11x11_isoEcal_NxN[ipt][ieta][j] = new TH1F(hname, htit, 220, -2.0, 20.0);
	h_contamination11x11_isoEcal_NxN[ipt][ieta][j]->Sumw2();
	
	d_R_chargeIsoIsoR->cd();
	sprintf(hname, "h_photonR_isoEcal_R_ptBin%i_etaBin%i_%s",ipt, ieta, name);
	sprintf(htit,  "photon in R iso(IsoR-R) (%3.2f<|#eta|<%3.2f), (%2.0f<trkP<%3.0f) for %s", lowEta, highEta, lowP, highP , name);
	h_photonR_isoEcal_R[ipt][ieta][j] = new TH1F(hname, htit, 220, -2.0, 20.0);
	h_photonR_isoEcal_R[ipt][ieta][j]->Sumw2();
	sprintf(hname, "h_chargedR_isoEcal_R_ptBin%i_etaBin%i_%s",ipt, ieta, name);
	sprintf(htit,  "charged in R iso(IsoR-R) (%3.2f<|#eta|<%3.2f), (%2.0f<trkP<%3.0f) for %s", lowEta, highEta, lowP, highP , name);
	h_chargedR_isoEcal_R[ipt][ieta][j] = new TH1F(hname, htit, 220, -2.0, 20.0);
	h_chargedR_isoEcal_R[ipt][ieta][j]->Sumw2();
	sprintf(hname, "h_neutralR_isoEcal_R_ptBin%i_etaBin%i_%s",ipt, ieta, name);
	sprintf(htit,  "neutral in R iso(IsoR-R) (%3.2f<|#eta|<%3.2f), (%2.0f<trkP<%3.0f) for %s", lowEta, highEta, lowP, highP , name);
	h_neutralR_isoEcal_R[ipt][ieta][j] = new TH1F(hname, htit, 220, -2.0, 20.0);
	h_neutralR_isoEcal_R[ipt][ieta][j]->Sumw2();
	sprintf(hname, "h_contaminationR_isoEcal_R_ptBin%i_etaBin%i_%s",ipt, ieta, name);
	sprintf(htit,  "contamination in R iso(IsoR-R) (%3.2f<|#eta|<%3.2f), (%2.0f<trkP<%3.0f) for %s", lowEta, highEta, lowP, highP , name);
	h_contaminationR_isoEcal_R[ipt][ieta][j] = new TH1F(hname, htit, 220, -2.0, 20.0);
	h_contaminationR_isoEcal_R[ipt][ieta][j]->Sumw2();
	
	d_H5x5_IsoNxN->cd();
	sprintf(hname, "h_photonHC5x5_IsoNxN_ptBin%i_etaBin%i_%s",ipt, ieta, name);
	sprintf(htit,  "photon in HC5x5 (IsoNxN) (%3.2f<|#eta|<%3.2f), (%2.0f<trkP<%3.0f) for %s", lowEta, highEta, lowP, highP , name);
	h_photonHC5x5_IsoNxN[ipt][ieta][j] = new TH1F(hname, htit, 220, -2.0, 20.0);
	h_photonHC5x5_IsoNxN[ipt][ieta][j]->Sumw2();
	sprintf(hname, "h_chargedHC5x5_IsoNxN_ptBin%i_etaBin%i_%s",ipt, ieta, name);
	sprintf(htit,  "charged in HC5x5 (IsoNxN) (%3.2f<|#eta|<%3.2f), (%2.0f<trkP<%3.0f) for %s", lowEta, highEta, lowP, highP , name);
	h_chargedHC5x5_IsoNxN[ipt][ieta][j] = new TH1F(hname, htit, 220, -2.0, 20.0);
	h_chargedHC5x5_IsoNxN[ipt][ieta][j]->Sumw2();
	sprintf(hname, "h_neutralHC5x5_IsoNxN_ptBin%i_etaBin%i_%s",ipt, ieta, name);
	sprintf(htit,  "neutral in HC5x5 (IsoNxN) (%3.2f<|#eta|<%3.2f), (%2.0f<trkP<%3.0f) for %s", lowEta, highEta, lowP, highP , name);
	h_neutralHC5x5_IsoNxN[ipt][ieta][j] = new TH1F(hname, htit, 220, -2.0, 20.0);
	h_neutralHC5x5_IsoNxN[ipt][ieta][j]->Sumw2();
	sprintf(hname, "h_contaminationHC5x5_IsoNxN_ptBin%i_etaBin%i_%s",ipt, ieta, name);
	sprintf(htit,  "contamination in HC5x5 (IsoNxN) (%3.2f<|#eta|<%3.2f), (%2.0f<trkP<%3.0f) for %s", lowEta, highEta, lowP, highP , name);
	h_contaminationHC5x5_IsoNxN[ipt][ieta][j] = new TH1F(hname, htit, 220, -2.0, 20.0);
	h_contaminationHC5x5_IsoNxN[ipt][ieta][j]->Sumw2();
	
	d_HCR_IsoR->cd();
	sprintf(hname, "h_photonHCR_IsoR_ptBin%i_etaBin%i_%s",ipt, ieta, name);
	sprintf(htit,  "photon in HCR (IsoR) (%3.2f<|#eta|<%3.2f), (%2.0f<trkP<%3.0f) for %s", lowEta, highEta, lowP, highP , name);
	h_photonHCR_IsoR[ipt][ieta][j] = new TH1F(hname, htit, 220, -2.0, 20.0);
	h_photonHCR_IsoR[ipt][ieta][j]->Sumw2();
	sprintf(hname, "h_chargedHCR_IsoR_ptBin%i_etaBin%i_%s",ipt, ieta, name);
	sprintf(htit,  "charged in HCR (IsoR) (%3.2f<|#eta|<%3.2f), (%2.0f<trkP<%3.0f) for %s", lowEta, highEta, lowP, highP , name);
	h_chargedHCR_IsoR[ipt][ieta][j] = new TH1F(hname, htit, 220, -2.0, 20.0);
	h_chargedHCR_IsoR[ipt][ieta][j]->Sumw2();
	sprintf(hname, "h_neutralHCR_IsoR_ptBin%i_etaBin%i_%s",ipt, ieta, name);
	sprintf(htit,  "neutral in HCR (IsoR) (%3.2f<|#eta|<%3.2f), (%2.0f<trkP<%3.0f) for %s", lowEta, highEta, lowP, highP , name);
	h_neutralHCR_IsoR[ipt][ieta][j] = new TH1F(hname, htit, 220, -2.0, 20.0);
	h_neutralHCR_IsoR[ipt][ieta][j]->Sumw2();
	sprintf(hname, "h_contaminationHCR_IsoR_ptBin%i_etaBin%i_%s",ipt, ieta, name);
	sprintf(htit,  "contamination in HCR (IsoR) (%3.2f<|#eta|<%3.2f), (%2.0f<trkP<%3.0f) for %s", lowEta, highEta, lowP, highP , name);
	h_contaminationHCR_IsoR[ipt][ieta][j] = new TH1F(hname, htit, 220, -2.0, 20.0);
	h_contaminationHCR_IsoR[ipt][ieta][j]->Sumw2();
      }
    }
    
    fout->cd();
    d_trigger->cd();
    sprintf(hname, "h_L1CenJetPt_%s", name);
    h_L1CenJetPt[j]  = new TH1F(hname, hname, 500, 0.0, 500);
    h_L1CenJetPt[j]->Sumw2();
    sprintf(hname, "h_L1FwdJetPt_%s", name);
    h_L1FwdJetPt[j]  = new TH1F(hname, hname, 500, 0.0, 500);
    h_L1FwdJetPt[j]->Sumw2();
    sprintf(hname, "h_L1TauJetPt_%s", name);
    h_L1TauJetPt[j]  = new TH1F(hname, hname, 500, 0.0, 500);
    h_L1TauJetPt[j]->Sumw2();
    sprintf(hname, "h_L1LeadJetPt_%s", name);
    h_L1LeadJetPt[j] = new TH1F(hname, hname, 500, 0.0, 500);
    h_L1LeadJetPt[j]->Sumw2();    
  }
}

double TreeAnalysisReadGen::DeltaPhi(double v1, double v2) {

  // Computes the correctly normalized phi difference
  // v1, v2 = phi of object 1 and 2

  double pi    = 3.141592654;
  double twopi = 6.283185307;
  
  double diff = std::abs(v2 - v1);
  double corr = twopi - diff;
  if (diff < pi){ return diff;} else { return corr;}
}

double TreeAnalysisReadGen::DeltaR(double eta1, double phi1, double eta2, double phi2) {
  
  double deta = eta1 - eta2;
  double dphi = DeltaPhi(phi1, phi2);
  return std::sqrt(deta*deta + dphi*dphi);
}

void TreeAnalysisReadGen::AddWeight(){

  for (unsigned int i=0; i<fRange-iRange+1; i++) {
    for (int itype=0; itype<PTypes; itype++) {
      h_trkPAll[itype][NRanges]          ->Add(h_trkPAll[itype][i+iRange]          , weights[i]);
      h_trkPtAll[itype][NRanges]         ->Add(h_trkPtAll[itype][i+iRange]         , weights[i]);
      h_trkEtaAll[itype][NRanges]        ->Add(h_trkEtaAll[itype][i+iRange]        , weights[i]);
      h_trkPhiAll[itype][NRanges]        ->Add(h_trkPhiAll[itype][i+iRange]        , weights[i]);
      h_trkPIsoNxN[itype][NRanges]       ->Add(h_trkPIsoNxN[itype][i+iRange]       , weights[i]);
      h_trkPtIsoNxN[itype][NRanges]      ->Add(h_trkPtIsoNxN[itype][i+iRange]      , weights[i]);
      h_trkEtaIsoNxN[itype][NRanges]     ->Add(h_trkEtaIsoNxN[itype][i+iRange]     , weights[i]);
      h_trkPhiIsoNxN[itype][NRanges]     ->Add(h_trkPhiIsoNxN[itype][i+iRange]     , weights[i]);
      h_trkPIsoR[itype][NRanges]         ->Add(h_trkPIsoR[itype][i+iRange]         , weights[i]);
      h_trkPtIsoR[itype][NRanges]        ->Add(h_trkPtIsoR[itype][i+iRange]        , weights[i]);
      h_trkEtaIsoR[itype][NRanges]       ->Add(h_trkEtaIsoR[itype][i+iRange]       , weights[i]);
      h_trkPhiIsoR[itype][NRanges]       ->Add(h_trkPhiIsoR[itype][i+iRange]       , weights[i]);
    }
    h_trkP_iso31x31[NRanges] ->Add(h_trkP_iso31x31[i+iRange] , weights[i]);
    h_trkP_iso25x25[NRanges] ->Add(h_trkP_iso25x25[i+iRange] , weights[i]);
    h_trkP_iso21x21[NRanges] ->Add(h_trkP_iso21x21[i+iRange] , weights[i]);
    h_trkP_iso15x15[NRanges] ->Add(h_trkP_iso15x15[i+iRange] , weights[i]);
    h_trkP_iso11x11[NRanges] ->Add(h_trkP_iso11x11[i+iRange] , weights[i]);
    h_L1Decision[NRanges]    ->Add(h_L1Decision[i+iRange]    , weights[i]);

    for (int ieta=0; ieta<NEtaBins; ieta++) {
      for (int ipt=0; ipt<NPBins; ipt++) {
	h_trkDEta[ipt][ieta][NRanges] ->Add(h_trkDEta[ipt][ieta][i+iRange] , weights[i]);
	h_trkDPhi[ipt][ieta][NRanges] ->Add(h_trkDPhi[ipt][ieta][i+iRange] , weights[i]);
	h_maxNearP31x31[ipt][ieta][NRanges]                  ->Add(h_maxNearP31x31[ipt][ieta][i+iRange]                  , weights[i]);
	h_maxNearP25x25[ipt][ieta][NRanges]                  ->Add(h_maxNearP25x25[ipt][ieta][i+iRange]                  , weights[i]);
	h_maxNearP21x21[ipt][ieta][NRanges]                  ->Add(h_maxNearP21x21[ipt][ieta][i+iRange]                  , weights[i]);
	h_maxNearP15x15[ipt][ieta][NRanges]                  ->Add(h_maxNearP15x15[ipt][ieta][i+iRange]                  , weights[i]);
	h_maxNearP11x11[ipt][ieta][NRanges]                  ->Add(h_maxNearP11x11[ipt][ieta][i+iRange]                  , weights[i]);
	h_photon_iso31x31[ipt][ieta][NRanges]                ->Add(h_photon_iso31x31[ipt][ieta][i+iRange]                , weights[i]);
	h_charged_iso31x31[ipt][ieta][NRanges]               ->Add(h_charged_iso31x31[ipt][ieta][i+iRange]               , weights[i]);
	h_neutral_iso31x31[ipt][ieta][NRanges]               ->Add(h_neutral_iso31x31[ipt][ieta][i+iRange]               , weights[i]);
	h_contamination_iso31x31[ipt][ieta][NRanges]         ->Add(h_contamination_iso31x31[ipt][ieta][i+iRange]         , weights[i]);
	h_L1_iso31x31[ipt][ieta][NRanges]                    ->Add(h_L1_iso31x31[ipt][ieta][i+iRange]                    , weights[i]);
	h_photon_iso25x25[ipt][ieta][NRanges]                ->Add(h_photon_iso25x25[ipt][ieta][i+iRange]                , weights[i]);
	h_charged_iso25x25[ipt][ieta][NRanges]               ->Add(h_charged_iso25x25[ipt][ieta][i+iRange]               , weights[i]);
	h_neutral_iso25x25[ipt][ieta][NRanges]               ->Add(h_neutral_iso25x25[ipt][ieta][i+iRange]               , weights[i]);
	h_contamination_iso25x25[ipt][ieta][NRanges]         ->Add(h_contamination_iso25x25[ipt][ieta][i+iRange]         , weights[i]);
	h_photon_iso21x21[ipt][ieta][NRanges]                ->Add(h_photon_iso21x21[ipt][ieta][i+iRange]                , weights[i]);
	h_charged_iso21x21[ipt][ieta][NRanges]               ->Add(h_charged_iso21x21[ipt][ieta][i+iRange]               , weights[i]);
	h_neutral_iso21x21[ipt][ieta][NRanges]               ->Add(h_neutral_iso21x21[ipt][ieta][i+iRange]               , weights[i]);
	h_contamination_iso21x21[ipt][ieta][NRanges]         ->Add(h_contamination_iso21x21[ipt][ieta][i+iRange]         , weights[i]);
	h_photon_iso15x15[ipt][ieta][NRanges]                ->Add(h_photon_iso15x15[ipt][ieta][i+iRange]                , weights[i]);
	h_charged_iso15x15[ipt][ieta][NRanges]               ->Add(h_charged_iso15x15[ipt][ieta][i+iRange]               , weights[i]);
	h_neutral_iso15x15[ipt][ieta][NRanges]               ->Add(h_neutral_iso15x15[ipt][ieta][i+iRange]               , weights[i]);
	h_contamination_iso15x15[ipt][ieta][NRanges]         ->Add(h_contamination_iso15x15[ipt][ieta][i+iRange]         , weights[i]);
	h_photon_iso11x11[ipt][ieta][NRanges]                ->Add(h_photon_iso11x11[ipt][ieta][i+iRange]                , weights[i]);
	h_charged_iso11x11[ipt][ieta][NRanges]               ->Add(h_charged_iso11x11[ipt][ieta][i+iRange]               , weights[i]);
	h_neutral_iso11x11[ipt][ieta][NRanges]               ->Add(h_neutral_iso11x11[ipt][ieta][i+iRange]               , weights[i]);
	h_contamination_iso11x11[ipt][ieta][NRanges]         ->Add(h_contamination_iso11x11[ipt][ieta][i+iRange]         , weights[i]);
	h_photon11x11_iso31x31[ipt][ieta][NRanges]           ->Add(h_photon11x11_iso31x31[ipt][ieta][i+iRange]           , weights[i]);
	h_charged11x11_iso31x31[ipt][ieta][NRanges]          ->Add(h_charged11x11_iso31x31[ipt][ieta][i+iRange]          , weights[i]);
	h_neutral11x11_iso31x31[ipt][ieta][NRanges]          ->Add(h_neutral11x11_iso31x31[ipt][ieta][i+iRange]          , weights[i]);
	h_contamination11x11_iso31x31[ipt][ieta][NRanges]    ->Add(h_contamination11x11_iso31x31[ipt][ieta][i+iRange]    , weights[i]);
	h_L1_iso31x31_isoPhoton_11x11_1[ipt][ieta][NRanges]  ->Add(h_L1_iso31x31_isoPhoton_11x11_1[ipt][ieta][i+iRange]  , weights[i]);
	h_L1_iso31x31_isoPhoton_11x11_2[ipt][ieta][NRanges]  ->Add(h_L1_iso31x31_isoPhoton_11x11_2[ipt][ieta][i+iRange]  , weights[i]);
	h_L1_iso31x31_isoNeutral_11x11_1[ipt][ieta][NRanges] ->Add(h_L1_iso31x31_isoNeutral_11x11_1[ipt][ieta][i+iRange] , weights[i]);
	h_L1_iso31x31_isoNeutral_11x11_2[ipt][ieta][NRanges] ->Add(h_L1_iso31x31_isoNeutral_11x11_2[ipt][ieta][i+iRange] , weights[i]);
	h_photon11x11_isoEcal_NxN[ipt][ieta][NRanges]        ->Add(h_photon11x11_isoEcal_NxN[ipt][ieta][i+iRange]        , weights[i]);
	h_charged11x11_isoEcal_NxN[ipt][ieta][NRanges]       ->Add(h_charged11x11_isoEcal_NxN[ipt][ieta][i+iRange]       , weights[i]);
	h_neutral11x11_isoEcal_NxN[ipt][ieta][NRanges]       ->Add(h_neutral11x11_isoEcal_NxN[ipt][ieta][i+iRange]       , weights[i]);
	h_contamination11x11_isoEcal_NxN[ipt][ieta][NRanges] ->Add(h_contamination11x11_isoEcal_NxN[ipt][ieta][i+iRange] , weights[i]);
	h_photonR_isoEcal_R[ipt][ieta][NRanges]              ->Add(h_photonR_isoEcal_R[ipt][ieta][i+iRange]              , weights[i]);
	h_chargedR_isoEcal_R[ipt][ieta][NRanges]             ->Add(h_chargedR_isoEcal_R[ipt][ieta][i+iRange]             , weights[i]);
	h_neutralR_isoEcal_R[ipt][ieta][NRanges]             ->Add(h_neutralR_isoEcal_R[ipt][ieta][i+iRange]             , weights[i]);
	h_contaminationR_isoEcal_R[ipt][ieta][NRanges]       ->Add(h_contaminationR_isoEcal_R[ipt][ieta][i+iRange]       , weights[i]);
	h_photonHC5x5_IsoNxN[ipt][ieta][NRanges]             ->Add(h_photonHC5x5_IsoNxN[ipt][ieta][i+iRange]             , weights[i]);
	h_chargedHC5x5_IsoNxN[ipt][ieta][NRanges]            ->Add(h_chargedHC5x5_IsoNxN[ipt][ieta][i+iRange]            , weights[i]);
	h_neutralHC5x5_IsoNxN[ipt][ieta][NRanges]            ->Add(h_neutralHC5x5_IsoNxN[ipt][ieta][i+iRange]            , weights[i]);
	h_contaminationHC5x5_IsoNxN[ipt][ieta][NRanges]      ->Add(h_contaminationHC5x5_IsoNxN[ipt][ieta][i+iRange]      , weights[i]);
	h_photonHCR_IsoR[ipt][ieta][NRanges]                 ->Add(h_photonHCR_IsoR[ipt][ieta][i+iRange]                 , weights[i]);
	h_chargedHCR_IsoR[ipt][ieta][NRanges]                ->Add(h_chargedHCR_IsoR[ipt][ieta][i+iRange]                , weights[i]);
	h_neutralHCR_IsoR[ipt][ieta][NRanges]                ->Add(h_neutralHCR_IsoR[ipt][ieta][i+iRange]                , weights[i]);
	h_contaminationHCR_IsoR[ipt][ieta][NRanges]          ->Add(h_contaminationHCR_IsoR[ipt][ieta][i+iRange]          , weights[i]);
      }
    }
    h_L1CenJetPt[NRanges]  ->Add(h_L1CenJetPt[i+iRange]  , weights[i]);
    h_L1FwdJetPt[NRanges]  ->Add(h_L1FwdJetPt[i+iRange]  , weights[i]);
    h_L1TauJetPt[NRanges]  ->Add(h_L1TauJetPt[i+iRange]  , weights[i]);
    h_L1LeadJetPt[NRanges] ->Add(h_L1LeadJetPt[i+iRange] , weights[i]);
  }
}

void TreeAnalysisReadGen::setRange(unsigned int ir) {
  iRangeBin = ir;
}

void TreeAnalysisReadGen::clear() {
  std::cout << fChain << std::endl;
  if (!fChain) return;
  delete fChain->GetCurrentFile();
}
