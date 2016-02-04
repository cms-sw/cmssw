#include "TreeAnalysisHcalNZS.h"
#include <TStyle.h>
#include <TCanvas.h>

Bool_t FillChain(TChain *chain, const TString &inputFileList);

int main(Int_t argc, Char_t *argv[]) {

  if ( argc<12 ) {
    std::cerr << "Please give 10 arguments "
              << "runList \n"
              << "outputFileName \n"
              << "dataType \n"
              << "ecalCharIso \n"
              << "hcalCharIso \n"
              << "ebNeutIso \n"
              << "eeNeutIso \n"
              << "hhNeutIso \n"
	      << "L1Seed \n"
              << "dRL1Jet \n"
	      << "ZS/NZS Flag \n"
	      << "nGoodPV \n"
	      << std::endl;
    return -1;
  }

  const char *inputFileList = argv[1];
  const char *outFileName   = argv[2];
  const char *dataType      = argv[3];
  const char *ecalCharIso   = argv[4];
  const char *hcalCharIso   = argv[5];
  const char *ebNeutIso     = argv[6];
  const char *eeNeutIso     = argv[7];
  const char *hhNeutIso     = argv[8];
  const char *L1Seed        = argv[9];
  const char *dRL1Jet       = argv[10];
  const char *ZSNZSFlag     = argv[11];
  const char *nGoodPV       = argv[12];

  int cut = 0;
  int flag = atoi(ZSNZSFlag);

  std::cout << "Command line arguments"          << std::endl;
  std::cout << "inputFileList " << inputFileList << std::endl;
  std::cout << "outFileName   " << outFileName   << std::endl;
  std::cout << "dataType      " << dataType      << std::endl;
  std::cout << "ecalCharIso   " << ecalCharIso   << std::endl;
  std::cout << "hcalCharIso   " << hcalCharIso   << std::endl;
  std::cout << "ebNeutIso     " << ebNeutIso     << std::endl;
  std::cout << "eeNeutIso     " << eeNeutIso     << std::endl;
  std::cout << "hhNeutIso     " << hhNeutIso     << std::endl;
  std::cout << "dRL1Jet       " << dRL1Jet       << std::endl;
  std::cout << "L1Seed        " << L1Seed        << std::endl;
  std::cout << "ZS/NZS Flag   " << flag          << "(1 for ZS, 0 for NZS)" << std::endl;
  std::cout << "nGoodPV       " << nGoodPV       << "(0: nGoodPV=1, -1: firstGoodPV, -2: anyGoodPV)" << std::endl;

  // Reading Tree 
  std::cout << "---------------------" << std::endl;
  std::cout << "Reading List of input trees from " << inputFileList << std::endl;
  
  TChain *chain;
  if (flag) chain = new TChain("/isolatedTracksNxN/tree");
  else  chain = new TChain("/isolatedTracksNxN_NZS/tree");
  if( ! FillChain(chain, inputFileList) ) {
    std::cerr << "Cannot get the tree " << std::endl;
    return(0);
  }

  TreeAnalysisHcalNZS tree(chain, outFileName);
  
  tree.ecalCharIso = ecalCharIso;
  tree.hcalCharIso = hcalCharIso;
  tree.dataType    = dataType;
  tree.ebNeutIso   = atof(ebNeutIso);
  tree.eeNeutIso   = atof(eeNeutIso);
  tree.hhNeutIso   = atof(hhNeutIso);
  tree.dRL1Jet     = atof(dRL1Jet);
  tree.L1Seed      = L1Seed;
  tree.GoodPVCut   = atoi(nGoodPV);

  tree.Loop(cut);

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
  }
  std::cout << "No. of Entries in this tree : " << chain->GetEntries() << std::endl;
  return kTRUE;
}

TreeAnalysisHcalNZS::TreeAnalysisHcalNZS(TChain *tree, const char *outFileName) {

  double tempgen_TH[NPBins+1] = { 0.0,  1.0,  2.0,  3.0,  4.0,  
				  5.0,  6.0,  7.0,  9.0,  11.0, 
				  15.0, 20.0, 30.0, 50.0, 75.0, 100.0};

  for(int i=0; i<NPBins+1; i++)  genPartPBins[i]  = tempgen_TH[i];

  double tempgen_Eta[NEtaBins+1] = {0.0, 0.2, 0.4, 0.6, 0.8, 
                                    1.0, 1.2, 1.4, 1.6, 1.8, 
                                    2.0, 2.2, 2.4};
  
  for(int i=0; i<NEtaBins+1; i++) genPartEtaBins[i] = tempgen_Eta[i];

  Init(tree);
   
  BookHistograms(outFileName);
}

TreeAnalysisHcalNZS::~TreeAnalysisHcalNZS(){
   if (!fChain) return;
   delete fChain->GetCurrentFile();
   fout->cd();
   fout->Write();
   fout->Close();   
}

Int_t TreeAnalysisHcalNZS::Cut(Long64_t entry) {
  // This function may be called from Loop.
  // returns  1 if entry is accepted.
  // returns -1 otherwise.
  return 1;
}

Int_t TreeAnalysisHcalNZS::GetEntry(Long64_t entry) {
  // Read contents of entry.
  if (!fChain) return 0;
  return fChain->GetEntry(entry);
}

Long64_t TreeAnalysisHcalNZS::LoadTree(Long64_t entry)
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

void TreeAnalysisHcalNZS::Init(TChain *tree)
{
   // The Init() function is called when the selector needs to initialize
   // a new tree or chain. Typically here the branch addresses and branch
   // pointers of the tree will be set.
   // It is normally not necessary to make changes to the generated
   // code, but the routine can be extended by the user if needed.
   // Init() will be called many times when running on PROOF
   // (once per file to be processed).

   // Set object pointer
   PVx = 0;
   PVy = 0;
   PVz = 0;
   PVisValid = 0;
   PVndof = 0;
   PVNTracks = 0;
   PVNTracksWt = 0;
   t_PVTracksSumPt = 0;
   t_PVTracksSumPtWt = 0;
   PVNTracksHP = 0;
   PVNTracksHPWt = 0;
   t_PVTracksSumPtHP = 0;
   t_PVTracksSumPtHPWt = 0;

   t_L1AlgoNames = 0;
   t_L1PreScale = 0;
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

   t_jetPt = 0;
   t_jetEta = 0;
   t_jetPhi = 0;
   t_nTrksJetCalo = 0;
   t_nTrksJetVtx = 0;

   t_trackPAll = 0;
   t_trackEtaAll = 0;
   t_trackPhiAll = 0;
   t_trackPtAll = 0;
   t_trackDxyAll = 0;
   t_trackDzAll = 0;
   t_trackDxyPVAll = 0;
   t_trackDzPVAll = 0;
   t_trackChiSqAll = 0;

   t_trackP = 0;
   t_trackPt = 0;
   t_trackEta = 0;
   t_trackPhi = 0;
   t_trackEcalEta = 0;
   t_trackEcalPhi = 0;
   t_trackHcalEta = 0;
   t_trackHcalPhi = 0;
   t_trackNOuterHits = 0;
   t_NLayersCrossed = 0;
   t_trackHitsTOB = 0;
   t_trackHitsTEC = 0;
   t_trackHitInMissTOB = 0;
   t_trackHitInMissTEC = 0;
   t_trackHitInMissTIB = 0;
   t_trackHitInMissTID = 0;
   t_trackHitInMissTIBTID = 0;
   t_trackHitOutMissTOB = 0;
   t_trackHitOutMissTEC = 0;
   t_trackHitOutMissTIB = 0;
   t_trackHitOutMissTID = 0;
   t_trackHitOutMissTOBTEC = 0;
   t_trackHitInMeasTOB = 0;
   t_trackHitInMeasTEC = 0;
   t_trackHitInMeasTIB = 0;
   t_trackHitInMeasTID = 0;
   t_trackHitOutMeasTOB = 0;
   t_trackHitOutMeasTEC = 0;
   t_trackHitOutMeasTIB = 0;
   t_trackHitOutMeasTID = 0;
   t_trackOutPosOutHitDr = 0;
   t_trackL = 0;
   t_trackDxy = 0;
   t_trackDxyBS = 0;
   t_trackDz = 0;
   t_trackDzBS = 0;
   t_trackDxyPV = 0;
   t_trackDzPV = 0;
   t_trackChiSq = 0;
   t_trackPVIdx = 0;

   t_maxNearP31x31 = 0;
   t_maxNearP21x21 = 0;
   t_ecalSpike11x11 = 0;
   t_e7x7 = 0;
   t_e9x9 = 0;
   t_e11x11 = 0;
   t_e15x15 = 0;
   t_e7x7_10Sig = 0;
   t_e9x9_10Sig = 0;
   t_e11x11_10Sig = 0;
   t_e15x15_10Sig = 0;
   t_e7x7_15Sig = 0;
   t_e9x9_15Sig = 0;
   t_e11x11_15Sig = 0;
   t_e15x15_15Sig = 0;
   t_e7x7_20Sig = 0;
   t_e9x9_20Sig = 0;
   t_e11x11_20Sig = 0;
   t_e15x15_20Sig = 0;
   t_e7x7_25Sig = 0;
   t_e9x9_25Sig = 0;
   t_e11x11_25Sig = 0;
   t_e15x15_25Sig = 0;
   t_e7x7_30Sig = 0;
   t_e9x9_30Sig = 0;
   t_e11x11_30Sig = 0;
   t_e15x15_30Sig = 0;

   t_maxNearHcalP3x3 = 0;
   t_maxNearHcalP5x5 = 0;
   t_maxNearHcalP7x7 = 0;
   t_h3x3 = 0;
   t_h5x5 = 0;
   t_h7x7 = 0;
   t_h3x3Sig = 0;
   t_h5x5Sig = 0;
   t_h7x7Sig = 0;
   t_infoHcal = 0;

   // Set branch addresses and branch pointers
   if (!tree) return;
   fChain = tree;
   fCurrent = -1;
   fChain->SetMakeClass(1);

   fChain->SetBranchAddress("t_EvtNo", &t_EvtNo, &b_t_EvtNo);
   fChain->SetBranchAddress("t_RunNo", &t_RunNo, &b_t_RunNo);
   fChain->SetBranchAddress("t_Lumi", &t_Lumi, &b_t_Lumi);
   fChain->SetBranchAddress("t_Bunch", &t_Bunch, &b_t_Bunch);
   fChain->SetBranchAddress("PVx", &PVx, &b_PVx);
   fChain->SetBranchAddress("PVy", &PVy, &b_PVy);
   fChain->SetBranchAddress("PVz", &PVz, &b_PVz);
   fChain->SetBranchAddress("PVisValid", &PVisValid, &b_PVisValid);
   fChain->SetBranchAddress("PVndof", &PVndof, &b_PVndof);
   fChain->SetBranchAddress("PVNTracks", &PVNTracks, &b_PVNTracks);
   fChain->SetBranchAddress("PVNTracksWt", &PVNTracksWt, &b_PVNTracksWt);
   fChain->SetBranchAddress("t_PVTracksSumPt", &t_PVTracksSumPt, &b_t_PVTracksSumPt);
   fChain->SetBranchAddress("t_PVTracksSumPtWt", &t_PVTracksSumPtWt, &b_t_PVTracksSumPtWt);
   fChain->SetBranchAddress("PVNTracksHP", &PVNTracksHP, &b_PVNTracksHP);
   fChain->SetBranchAddress("PVNTracksHPWt", &PVNTracksHPWt, &b_PVNTracksHPWt);
   fChain->SetBranchAddress("t_PVTracksSumPtHP", &t_PVTracksSumPtHP, &b_t_PVTracksSumPtHP);
   fChain->SetBranchAddress("t_PVTracksSumPtHPWt", &t_PVTracksSumPtHPWt, &b_t_PVTracksSumPtHPWt);

   fChain->SetBranchAddress("t_L1Decision",  t_L1Decision,   &b_t_L1Decision);
   fChain->SetBranchAddress("t_L1AlgoNames", &t_L1AlgoNames, &b_t_L1AlgoNames);
   fChain->SetBranchAddress("t_L1PreScale",  &t_L1PreScale,  &b_t_L1PreScale);
   fChain->SetBranchAddress("t_L1CenJetPt",  &t_L1CenJetPt,  &b_t_L1CenJetPt);
   fChain->SetBranchAddress("t_L1CenJetEta", &t_L1CenJetEta, &b_t_L1CenJetEta);
   fChain->SetBranchAddress("t_L1CenJetPhi", &t_L1CenJetPhi, &b_t_L1CenJetPhi);
   fChain->SetBranchAddress("t_L1FwdJetPt",  &t_L1FwdJetPt,  &b_t_L1FwdJetPt);
   fChain->SetBranchAddress("t_L1FwdJetEta", &t_L1FwdJetEta, &b_t_L1FwdJetEta);
   fChain->SetBranchAddress("t_L1FwdJetPhi", &t_L1FwdJetPhi, &b_t_L1FwdJetPhi);
   fChain->SetBranchAddress("t_L1TauJetPt",  &t_L1TauJetPt,  &b_t_L1TauJetPt);
   fChain->SetBranchAddress("t_L1TauJetEta", &t_L1TauJetEta, &b_t_L1TauJetEta);
   fChain->SetBranchAddress("t_L1TauJetPhi", &t_L1TauJetPhi, &b_t_L1TauJetPhi);
   fChain->SetBranchAddress("t_L1MuonPt",    &t_L1MuonPt,    &b_t_L1MuonPt);
   fChain->SetBranchAddress("t_L1MuonEta",   &t_L1MuonEta,   &b_t_L1MuonEta);
   fChain->SetBranchAddress("t_L1MuonPhi",   &t_L1MuonPhi,   &b_t_L1MuonPhi);
   fChain->SetBranchAddress("t_L1IsoEMPt",   &t_L1IsoEMPt,   &b_t_L1IsoEMPt);
   fChain->SetBranchAddress("t_L1IsoEMEta",  &t_L1IsoEMEta,  &b_t_L1IsoEMEta);
   fChain->SetBranchAddress("t_L1IsoEMPhi",  &t_L1IsoEMPhi,  &b_t_L1IsoEMPhi);
   fChain->SetBranchAddress("t_L1NonIsoEMPt",&t_L1NonIsoEMPt,&b_t_L1NonIsoEMPt);
   fChain->SetBranchAddress("t_L1NonIsoEMEta", &t_L1NonIsoEMEta, &b_t_L1NonIsoEMEta);
   fChain->SetBranchAddress("t_L1NonIsoEMPhi", &t_L1NonIsoEMPhi, &b_t_L1NonIsoEMPhi);
   fChain->SetBranchAddress("t_L1METPt",     &t_L1METPt,     &b_t_L1METPt);
   fChain->SetBranchAddress("t_L1METEta",    &t_L1METEta,    &b_t_L1METEta);
   fChain->SetBranchAddress("t_L1METPhi",    &t_L1METPhi,    &b_t_L1METPhi);

   fChain->SetBranchAddress("t_jetPt", &t_jetPt, &b_t_jetPt);
   fChain->SetBranchAddress("t_jetEta", &t_jetEta, &b_t_jetEta);
   fChain->SetBranchAddress("t_jetPhi", &t_jetPhi, &b_t_jetPhi);
   fChain->SetBranchAddress("t_nTrksJetCalo", &t_nTrksJetCalo, &b_t_nTrksJetCalo);
   fChain->SetBranchAddress("t_nTrksJetVtx", &t_nTrksJetVtx, &b_t_nTrksJetVtx);

   fChain->SetBranchAddress("t_trackPAll",   &t_trackPAll,   &b_t_trackPAll);
   fChain->SetBranchAddress("t_trackEtaAll", &t_trackEtaAll, &b_t_trackEtaAll);
   fChain->SetBranchAddress("t_trackPhiAll", &t_trackPhiAll, &b_t_trackPhiAll);
   fChain->SetBranchAddress("t_trackPtAll",  &t_trackPtAll,  &b_t_trackPtAll);
   fChain->SetBranchAddress("t_trackDxyAll", &t_trackDxyAll, &b_t_trackDxyAll);
   fChain->SetBranchAddress("t_trackDzAll",  &t_trackDzAll,  &b_t_trackDzAll);
   fChain->SetBranchAddress("t_trackDxyPVAll", &t_trackDxyPVAll, &b_t_trackDxyPVAll);
   fChain->SetBranchAddress("t_trackDzPVAll", &t_trackDzPVAll, &b_t_trackDzPVAll);
   fChain->SetBranchAddress("t_trackChiSqAll", &t_trackChiSqAll, &b_t_trackChiSqAll);
   fChain->SetBranchAddress("t_trackP", &t_trackP, &b_t_trackP);
   fChain->SetBranchAddress("t_trackPt", &t_trackPt, &b_t_trackPt);
   fChain->SetBranchAddress("t_trackEta", &t_trackEta, &b_t_trackEta);
   fChain->SetBranchAddress("t_trackPhi", &t_trackPhi, &b_t_trackPhi);
   fChain->SetBranchAddress("t_trackEcalEta", &t_trackEcalEta, &b_t_trackEcalEta);
   fChain->SetBranchAddress("t_trackEcalPhi", &t_trackEcalPhi, &b_t_trackEcalPhi);
   fChain->SetBranchAddress("t_trackHcalEta", &t_trackHcalEta, &b_t_trackHcalEta);
   fChain->SetBranchAddress("t_trackHcalPhi", &t_trackHcalPhi, &b_t_trackHcalPhi);
   fChain->SetBranchAddress("t_trackNOuterHits", &t_trackNOuterHits, &b_t_trackNOuterHits);
   fChain->SetBranchAddress("t_NLayersCrossed", &t_NLayersCrossed, &b_t_NLayersCrossed);
   fChain->SetBranchAddress("t_trackHitsTOB", &t_trackHitsTOB, &b_t_trackHitsTOB);
   fChain->SetBranchAddress("t_trackHitsTEC", &t_trackHitsTEC, &b_t_trackHitsTEC);
   fChain->SetBranchAddress("t_trackHitInMissTOB", &t_trackHitInMissTOB, &b_t_trackHitInMissTOB);
   fChain->SetBranchAddress("t_trackHitInMissTEC", &t_trackHitInMissTEC, &b_t_trackHitInMissTEC);
   fChain->SetBranchAddress("t_trackHitInMissTIB", &t_trackHitInMissTIB, &b_t_trackHitInMissTIB);
   fChain->SetBranchAddress("t_trackHitInMissTID", &t_trackHitInMissTID, &b_t_trackHitInMissTID);
   fChain->SetBranchAddress("t_trackHitInMissTIBTID", &t_trackHitInMissTIBTID, &b_t_trackHitInMissTIBTID);
   fChain->SetBranchAddress("t_trackHitOutMissTOB", &t_trackHitOutMissTOB, &b_t_trackHitOutMissTOB);
   fChain->SetBranchAddress("t_trackHitOutMissTEC", &t_trackHitOutMissTEC, &b_t_trackHitOutMissTEC);
   fChain->SetBranchAddress("t_trackHitOutMissTIB", &t_trackHitOutMissTIB, &b_t_trackHitOutMissTIB);
   fChain->SetBranchAddress("t_trackHitOutMissTID", &t_trackHitOutMissTID, &b_t_trackHitOutMissTID);
   fChain->SetBranchAddress("t_trackHitOutMissTOBTEC", &t_trackHitOutMissTOBTEC, &b_t_trackHitOutMissTOBTEC);
   fChain->SetBranchAddress("t_trackHitInMeasTOB", &t_trackHitInMeasTOB, &b_t_trackHitInMeasTOB);
   fChain->SetBranchAddress("t_trackHitInMeasTEC", &t_trackHitInMeasTEC, &b_t_trackHitInMeasTEC);
   fChain->SetBranchAddress("t_trackHitInMeasTIB", &t_trackHitInMeasTIB, &b_t_trackHitInMeasTIB);
   fChain->SetBranchAddress("t_trackHitInMeasTID", &t_trackHitInMeasTID, &b_t_trackHitInMeasTID);
   fChain->SetBranchAddress("t_trackHitOutMeasTOB", &t_trackHitOutMeasTOB, &b_t_trackHitOutMeasTOB);
   fChain->SetBranchAddress("t_trackHitOutMeasTEC", &t_trackHitOutMeasTEC, &b_t_trackHitOutMeasTEC);
   fChain->SetBranchAddress("t_trackHitOutMeasTIB", &t_trackHitOutMeasTIB, &b_t_trackHitOutMeasTIB);
   fChain->SetBranchAddress("t_trackHitOutMeasTID", &t_trackHitOutMeasTID, &b_t_trackHitOutMeasTID);
   fChain->SetBranchAddress("t_trackDxy", &t_trackDxy, &b_t_trackDxy);
   fChain->SetBranchAddress("t_trackDxyBS", &t_trackDxyBS, &b_t_trackDxyBS);
   fChain->SetBranchAddress("t_trackDz", &t_trackDz, &b_t_trackDz);
   fChain->SetBranchAddress("t_trackDzBS", &t_trackDzBS, &b_t_trackDzBS);
   fChain->SetBranchAddress("t_trackDxyPV", &t_trackDxyPV, &b_t_trackDxyPV);
   fChain->SetBranchAddress("t_trackDzPV", &t_trackDzPV, &b_t_trackDzPV);
   fChain->SetBranchAddress("t_trackChiSq", &t_trackChiSq, &b_t_trackChiSq);
   fChain->SetBranchAddress("t_trackPVIdx", &t_trackPVIdx, &b_t_trackPVIdx);

   fChain->SetBranchAddress("t_maxNearP31x31", &t_maxNearP31x31, &b_t_maxNearP31x31);
   fChain->SetBranchAddress("t_maxNearP21x21", &t_maxNearP21x21, &b_t_maxNearP21x21);
   fChain->SetBranchAddress("t_ecalSpike11x11", &t_ecalSpike11x11, &b_t_ecalSpike11x11);
   fChain->SetBranchAddress("t_e7x7", &t_e7x7, &b_t_e7x7);
   fChain->SetBranchAddress("t_e9x9", &t_e9x9, &b_t_e9x9);
   fChain->SetBranchAddress("t_e11x11", &t_e11x11, &b_t_e11x11);
   fChain->SetBranchAddress("t_e15x15", &t_e15x15, &b_t_e15x15);
   fChain->SetBranchAddress("t_e7x7_10Sig", &t_e7x7_10Sig, &b_t_e7x7_10Sig);
   fChain->SetBranchAddress("t_e9x9_10Sig", &t_e9x9_10Sig, &b_t_e9x9_10Sig);
   fChain->SetBranchAddress("t_e11x11_10Sig", &t_e11x11_10Sig, &b_t_e11x11_10Sig);
   fChain->SetBranchAddress("t_e15x15_10Sig", &t_e15x15_10Sig, &b_t_e15x15_10Sig);
   fChain->SetBranchAddress("t_e7x7_15Sig", &t_e7x7_15Sig, &b_t_e7x7_15Sig);
   fChain->SetBranchAddress("t_e9x9_15Sig", &t_e9x9_15Sig, &b_t_e9x9_15Sig);
   fChain->SetBranchAddress("t_e11x11_15Sig", &t_e11x11_15Sig, &b_t_e11x11_15Sig);
   fChain->SetBranchAddress("t_e15x15_15Sig", &t_e15x15_15Sig, &b_t_e15x15_15Sig);
   fChain->SetBranchAddress("t_e7x7_20Sig", &t_e7x7_20Sig, &b_t_e7x7_20Sig);
   fChain->SetBranchAddress("t_e9x9_20Sig", &t_e9x9_20Sig, &b_t_e9x9_20Sig);
   fChain->SetBranchAddress("t_e11x11_20Sig", &t_e11x11_20Sig, &b_t_e11x11_20Sig);
   fChain->SetBranchAddress("t_e15x15_20Sig", &t_e15x15_20Sig, &b_t_e15x15_20Sig);
   fChain->SetBranchAddress("t_e7x7_25Sig", &t_e7x7_25Sig, &b_t_e7x7_25Sig);
   fChain->SetBranchAddress("t_e9x9_25Sig", &t_e9x9_25Sig, &b_t_e9x9_25Sig);
   fChain->SetBranchAddress("t_e11x11_25Sig", &t_e11x11_25Sig, &b_t_e11x11_25Sig);
   fChain->SetBranchAddress("t_e15x15_25Sig", &t_e15x15_25Sig, &b_t_e15x15_25Sig);
   fChain->SetBranchAddress("t_e7x7_30Sig", &t_e7x7_30Sig, &b_t_e7x7_30Sig);
   fChain->SetBranchAddress("t_e9x9_30Sig", &t_e9x9_30Sig, &b_t_e9x9_30Sig);
   fChain->SetBranchAddress("t_e11x11_30Sig", &t_e11x11_30Sig, &b_t_e11x11_30Sig);
   fChain->SetBranchAddress("t_e15x15_30Sig", &t_e15x15_30Sig, &b_t_e15x15_30Sig);

   fChain->SetBranchAddress("t_maxNearHcalP3x3", &t_maxNearHcalP3x3, &b_t_maxNearHcalP3x3);
   fChain->SetBranchAddress("t_maxNearHcalP5x5", &t_maxNearHcalP5x5, &b_t_maxNearHcalP5x5);
   fChain->SetBranchAddress("t_maxNearHcalP7x7", &t_maxNearHcalP7x7, &b_t_maxNearHcalP7x7);
   fChain->SetBranchAddress("t_h3x3", &t_h3x3, &b_t_h3x3);
   fChain->SetBranchAddress("t_h5x5", &t_h5x5, &b_t_h5x5);
   fChain->SetBranchAddress("t_h7x7", &t_h7x7, &b_t_h7x7);
   fChain->SetBranchAddress("t_h3x3Sig", &t_h3x3Sig, &b_t_h3x3Sig);
   fChain->SetBranchAddress("t_h5x5Sig", &t_h5x5Sig, &b_t_h5x5Sig);
   fChain->SetBranchAddress("t_h7x7Sig", &t_h7x7Sig, &b_t_h7x7Sig);
   fChain->SetBranchAddress("t_infoHcal", &t_infoHcal, &b_t_infoHcal);
   fChain->SetBranchAddress("t_nTracks", &t_nTracks, &b_t_nTracks);
   Notify();
}

void TreeAnalysisHcalNZS::Loop(int cut) {

  std::cout << "Track isolation thresholds "          <<std::endl;
  std::cout << "dataType      " << dataType      << std::endl;
  std::cout << "ecalCharIso   " << ecalCharIso   << std::endl;
  std::cout <<" hcalCharIso   " << hcalCharIso   << std::endl;
  std::cout << "ebNeutIso     " << ebNeutIso     << std::endl;
  std::cout << "eeNeutIso     " << eeNeutIso     << std::endl;
  std::cout << "hhNeutIso     " << hhNeutIso     << std::endl;
  std::cout << "L1Seed        " << L1Seed        << std::endl;
  if (fChain == 0) {
    std::cout << "\n Tree does not exist "<<std::endl;
    return;
  }

  bool debug = false;
  Long64_t nentries = fChain->GetEntriesFast();
  std::cout << "No. of Entries in tree " << nentries << std::endl;

  Long64_t nEventsGoodRuns=0, nEventsValidPV=0, nEventsPVTracks=0;

  Long64_t nbytes = 0, nb = 0;
  std::map<unsigned int, unsigned int> runEvtList;
  std::map<unsigned int, unsigned int> runNTrkList;
  std::map<unsigned int, unsigned int> runNIsoTrkList;

  for (Long64_t jentry=0; jentry<nentries;jentry++) {

    // load tree and get current entry
    Long64_t ientry = LoadTree(jentry);
    if (ientry < 0) break;
    nb = fChain->GetEntry(jentry);   nbytes += nb;

    if( !(jentry%100000) ) {
      std::cout << "processing event " << jentry+1 << std::endl;
    }

    bool goodRun= false;
    goodRun     = true;
    
    bool evtSel = true;
    if(dataType=="Data" && !goodRun ) {
      evtSel = false;
    }
    if( !evtSel ) continue;  //<=================    

    if( runEvtList.find(t_RunNo) != runEvtList.end() ) {
      runEvtList[t_RunNo] += 1; 
   } else {
      runEvtList.insert( std::pair<unsigned int, unsigned int>(t_RunNo,1) );
      if (debug) std::cout << "runNo " << t_RunNo <<" "<<runEvtList[t_RunNo]<<std::endl;
    }
    
    nEventsGoodRuns++;
  
    // I guess the vertex filter works on any PV of the collection
    // All PVs in the event

    std::vector<int> isGoodPV(PVz->size(), -1);
    int  nGoodPV=0;
    bool anyGoodPV  =false;
    bool firstPVGood=false;

    for(int ipv=0; ipv<PVz->size(); ipv++) {
      double rho = sqrt((*PVx)[ipv]*(*PVx)[ipv] + (*PVy)[ipv]*(*PVy)[ipv]);      
      h_PVx[0]            ->Fill( (*PVx)[ipv]             );
      h_PVy[0]            ->Fill( (*PVy)[ipv]             );
      h_PVr[0]            ->Fill( rho                   );
      h_PVz[0]            ->Fill( (*PVz)[ipv]             );
      h_PVNDOF[0]         ->Fill( (*PVndof)[ipv]          );
      h_PVNTracks[0]      ->Fill( (*PVNTracks)[ipv]       );
      h_PVTracksSumPt[0]  ->Fill( (*t_PVTracksSumPt)[ipv] ); 
      h_PVNTracksWt[0]    ->Fill( (*PVNTracksWt)[ipv]       );
      h_PVTracksSumPtWt[0]->Fill( (*t_PVTracksSumPtWt)[ipv] ); 
      h_PVNTracksSumPt[0] ->Fill( (*PVNTracks)[ipv], (*t_PVTracksSumPt)[ipv]);
      
      if( std::abs((*PVz)[ipv])<=15.0 && rho<=2.0 && (*PVndof)[ipv]>4 ) 
	isGoodPV[ipv] = 1;
      else 
	isGoodPV[ipv] = 0;
      
      if (debug) std::cout << "PV : " << ipv << " rho " << rho << " z " << (*PVz)[ipv] << " ndf " << (*PVndof)[ipv] << " Good " << isGoodPV[ipv] << std::endl;
      if( isGoodPV[ipv]>0 ) {
	nGoodPV++;
	anyGoodPV = true;
	if(ipv==0)              firstPVGood = true;
	h_PVx[1]            ->Fill( (*PVx)[ipv]             );
	h_PVy[1]            ->Fill( (*PVy)[ipv]             );
	h_PVr[1]            ->Fill( rho                   );
	h_PVz[1]            ->Fill( (*PVz)[ipv]             );
	h_PVNDOF[1]         ->Fill( (*PVndof)[ipv]          );
	h_PVNTracks[1]      ->Fill( (*PVNTracks)[ipv]       );
	h_PVTracksSumPt[1]  ->Fill( (*t_PVTracksSumPt)[ipv] ); 
	h_PVNTracksWt[1]    ->Fill( (*PVNTracksWt)[ipv]       );
	h_PVTracksSumPtWt[1]->Fill( (*t_PVTracksSumPtWt)[ipv] ); 
	h_PVNTracksSumPt[1] ->Fill( (*PVNTracks)[ipv], (*t_PVTracksSumPt)[ipv]);
      }     
    }
    h_NPV[0] ->Fill(PVz->size());
    h_NPV[1] ->Fill(nGoodPV);
    
    if( firstPVGood ) { 
      nEventsValidPV++;
      nEventsPVTracks++;
    }

    bool pvSel = true;
    //if( !firstPVGood )     pvSel = false;
    if (debug) std::cout << "GoodPVCut " << GoodPVCut << " anyGoodPV " << anyGoodPV << " firstPVGood " << firstPVGood << " nGoodPV " << nGoodPV << std::endl;
    if (GoodPVCut == -2 && !anyGoodPV           ) pvSel = false;
    else if (GoodPVCut == -1 && !firstPVGood    ) pvSel = false;
    else if (GoodPVCut==0 && nGoodPV != 1        ) pvSel = false; 
    else if (GoodPVCut>0 && (nGoodPV > GoodPVCut || nGoodPV < 1)) pvSel = false;

    if(dataType=="DiPion") pvSel = true;  // no PV selection for DiPion sample
    if (debug) std::cout << "NGoodPV " << nGoodPV << " First " << firstPVGood << " pvSel " << pvSel << std::endl;
    if( !pvSel ) continue;

    //================= avoid trigger bias ===================
    double maxL1Pt=-1.0, maxL1Eta=999.0, maxL1Phi=999.0;
    bool checkL1=false;

    if( L1Seed=="L1Jet" || L1Seed=="L1JetL1Tau" || L1Seed=="L1JetL1TauL1EM" || L1Seed=="L1JetL1EM") {
      if( t_L1Decision[15]>0 || t_L1Decision[16]>0 ||  t_L1Decision[17]>0	||
	  t_L1Decision[18]>0 || t_L1Decision[19]>0 ||  t_L1Decision[20]>0	||
	  t_L1Decision[21]>0 ){ 
	if( t_L1CenJetPt->size()>0 && (*t_L1CenJetPt)[0]>maxL1Pt )
	  {maxL1Pt=(*t_L1CenJetPt)[0]; maxL1Eta=(*t_L1CenJetEta)[0]; maxL1Phi=(*t_L1CenJetPhi)[0];}
	if( t_L1FwdJetPt->size()>0 && (*t_L1FwdJetPt)[0]>maxL1Pt )
	  {maxL1Pt=(*t_L1FwdJetPt)[0]; maxL1Eta=(*t_L1FwdJetEta)[0]; maxL1Phi=(*t_L1FwdJetPhi)[0];}
	checkL1=true;
      } 
    }
    if( L1Seed=="L1Tau" || L1Seed=="L1JetL1Tau" || L1Seed=="L1JetL1TauL1EM") {
      if( t_L1Decision[30]>0 ||  t_L1Decision[31]>0	|| t_L1Decision[32]>0 ||
	  t_L1Decision[33]>0 ) {
	if( t_L1TauJetPt->size()>0 && (*t_L1TauJetPt)[0]>maxL1Pt )
	  {maxL1Pt=(*t_L1TauJetPt)[0]; maxL1Eta=(*t_L1TauJetEta)[0]; maxL1Phi=(*t_L1TauJetPhi)[0];}
	checkL1=true;
      } 
    }
    if( L1Seed=="L1EM" || L1Seed=="L1JetL1TauL1EM" || L1Seed=="L1JetL1EM") {      
      if( t_L1Decision[46]>0 || t_L1Decision[47]>0 || t_L1Decision[48]>0 ||  
	  t_L1Decision[49]>0 || t_L1Decision[50]>0 || t_L1Decision[51]>0 ||  
	  t_L1Decision[52]>0	) {
	if( t_L1IsoEMPt->size()>0 && (*t_L1IsoEMPt)[0]>maxL1Pt )
	  {maxL1Pt=(*t_L1IsoEMPt)[0]; maxL1Eta=(*t_L1IsoEMEta)[0]; maxL1Phi=(*t_L1IsoEMPhi)[0];}
	checkL1=true;
      } 
    }
    if( maxL1Pt<0 ) checkL1=false;
    if (debug) {
      std::cout << "L1Seed " << L1Seed << " pt/eta/phi " << maxL1Pt << "|" << maxL1Eta << "|" << maxL1Phi << " checkL1 " << checkL1 << " L1Decision " << std::endl;
      for (int i=-14; i<53; ++i) std::cout << " i " << i << ":" << t_L1Decision[i];
      std::cout << std::endl;
    }

    // good events satisfying good first PV selection
    for(int itrk=0; itrk<t_trackPAll->size(); itrk++ ){
      h_trackP[0]    ->Fill((*t_trackPAll)[itrk]    );  
      h_trackPt[0]   ->Fill((*t_trackPtAll)[itrk]    );  
      h_trackEta[0]  ->Fill((*t_trackEtaAll)[itrk]  );  
      h_trackPhi[0]  ->Fill((*t_trackPhiAll)[itrk]  );  
      h_trackDxy[0]  ->Fill((*t_trackDxyPVAll)[itrk]  );  
      h_trackDz[0]   ->Fill((*t_trackDzPVAll)[itrk]  );  
      h_trackChiSq[0]->Fill((*t_trackChiSqAll)[itrk]  );  
    }
    
    // if charged isolated in ecal 31x31
    for(int itrk=0; itrk<t_trackP->size(); itrk++ ){
      h_trackP[1]    ->Fill((*t_trackP)[itrk]    );  
      h_trackPt[1]   ->Fill((*t_trackPt)[itrk]   );  
      h_trackEta[1]  ->Fill((*t_trackEta)[itrk]  );  
      h_trackPhi[1]  ->Fill((*t_trackPhi)[itrk]  );  
      h_trackDxy[1]  ->Fill((*t_trackDxyPV)[itrk]);  
      h_trackDz[1]   ->Fill((*t_trackDzPV )[itrk]);  
      h_trackChiSq[1]->Fill((*t_trackChiSq)[itrk]  );  
    }
    
    if( runNTrkList.find(t_RunNo) != runNTrkList.end() ) {
      runNTrkList[t_RunNo] += t_trackP->size(); 
    } else {
      runNTrkList.insert( std::pair<unsigned int, unsigned int>(t_RunNo,t_trackP->size()) );
      if (debug) std::cout << "runNo " << t_RunNo <<" "<<runNTrkList[t_RunNo]<<std::endl;
    }
    
    unsigned int NIsoTrk=0;
    for(int itrk=0; itrk<t_trackP->size(); itrk++ ){
      
      // reject soft tracks
      if (debug) std::cout << "Track " << itrk << " pt " << (*t_trackPt)[itrk] << std::endl;
      if( (*t_trackPt)[itrk] < 1.0 ) continue;

      double p1              = (*t_trackP)[itrk];
      double pt1             = (*t_trackPt)[itrk];
      double eta1            = (*t_trackEta)[itrk];
      double phi1            = (*t_trackPhi)[itrk];
      double etaEcal1        = (*t_trackEcalEta)[itrk];
      double phiEcal1        = (*t_trackEcalPhi)[itrk];
      double etaHcal1        = (*t_trackHcalEta)[itrk];
      double phiHcal1        = (*t_trackHcalPhi)[itrk];
      int    trackNOuterHits = (*t_trackNOuterHits)[itrk];
      int    NLayersCrossed  = (*t_NLayersCrossed)[itrk];
      int    ecalSpike11x11  = (*t_ecalSpike11x11)[itrk];

      double maxNearP31x31   = (*t_maxNearP31x31)[itrk];
      
      double e7x7            = (*t_e7x7)[itrk]; 
      double e9x9            = (*t_e9x9)[itrk]; 
      double e11x11          = (*t_e11x11)[itrk]; 
      double e15x15          = (*t_e15x15)[itrk];
      
      double maxNearHcalP3x3 = (*t_maxNearHcalP3x3)[itrk];
      double maxNearHcalP5x5 = (*t_maxNearHcalP5x5)[itrk];
      double maxNearHcalP7x7 = (*t_maxNearHcalP7x7)[itrk];
      
      double h3x3            = (*t_h3x3)[itrk];
      double h5x5            = (*t_h5x5)[itrk];
      double h7x7            = (*t_h7x7)[itrk];
      
      if( (*t_infoHcal)[itrk] < 1 ) {
	h3x3            = 0.0;
	h5x5            = 0.0;
	h7x7            = 0.0;
      }

      h_trackP[2]    ->Fill((*t_trackP)[itrk]    );
      h_trackPt[2]   ->Fill((*t_trackPt)[itrk]   );  
      h_trackEta[2]  ->Fill((*t_trackEta)[itrk]  );  
      h_trackPhi[2]  ->Fill((*t_trackPhi)[itrk]  );  
      h_trackDxy[2]  ->Fill((*t_trackDxyPV)[itrk]);  
      h_trackDz[2]   ->Fill((*t_trackDzPV )[itrk]);  
      h_trackChiSq[2]->Fill((*t_trackChiSq)[itrk]  );  
      
      int iTrkEtaBin=-1, iTrkMomBin=-1;
      for(int ieta=0; ieta<NEtaBins; ieta++)   {
	if(std::abs(etaEcal1)>genPartEtaBins[ieta] && std::abs(etaEcal1)<genPartEtaBins[ieta+1] ) iTrkEtaBin = ieta;
      }
      for(int ipt=0;  ipt<NPBins;   ipt++)  {
	if( p1>genPartPBins[ipt] &&  p1<genPartPBins[ipt+1] )  iTrkMomBin = ipt;
      }
      if (debug) std::cout << "p " << p1 << " " << iTrkMomBin << " eta " << eta1 << " " << iTrkEtaBin << std::endl;       
      
      bool trackChargeIso = true;
      if(ecalCharIso=="MaxNearP31x31"     && maxNearP31x31>0)     trackChargeIso=false;
      
      if(hcalCharIso=="MaxNearHcalP7x7"   && maxNearHcalP7x7>0)   trackChargeIso=false;
      if(hcalCharIso=="MaxNearHcalP5x5"   && maxNearHcalP5x5>0)   trackChargeIso=false;
      if(hcalCharIso=="MaxNearHcalP3x3"   && maxNearHcalP3x3>0)   trackChargeIso=false;
      if (debug) std::cout << "Isolation " << ecalCharIso << ":" << hcalCharIso << " ==> " << trackChargeIso << std::endl;
      
      bool hcalNeutIso = true;
      if( h7x7-h5x5   > hhNeutIso )                             hcalNeutIso=false;
      
      if( iTrkMomBin>=0 && iTrkEtaBin>=0) {	 
	h_trackPhi_2_2[iTrkEtaBin] ->Fill((*t_trackPhi)[itrk]);
      }

      bool trackSel = true;
      // reject interactions in tracker
      if( (*t_trackHitInMissTOB )[itrk]>0 || (*t_trackHitInMissTEC )[itrk]>0 || 
	  (*t_trackHitInMissTIB )[itrk]>0 || (*t_trackHitInMissTID )[itrk]>0 || 
	  (*t_trackHitOutMissTOB)[itrk]>0 || (*t_trackHitOutMissTEC)[itrk]>0 ) 
	trackSel = false;
      if( ecalSpike11x11==0 ) trackSel=false ;
      if( trackSel ) {
	h_trackP[3]    ->Fill((*t_trackP)[itrk]    );  
	h_trackPt[3]   ->Fill((*t_trackPt)[itrk]   );  
	h_trackEta[3]  ->Fill((*t_trackEta)[itrk]  );  
	h_trackPhi[3]  ->Fill((*t_trackPhi)[itrk]  );  
	h_trackDxy[3]  ->Fill((*t_trackDxyPV)[itrk]);  
	h_trackDz[3]   ->Fill((*t_trackDzPV )[itrk]);  
	h_trackChiSq[3]->Fill((*t_trackChiSq)[itrk]  );  
	if( iTrkMomBin>=0 && iTrkEtaBin>=0) {	 
	  h_trackPhi_3_3[iTrkEtaBin] ->Fill((*t_trackPhi)[itrk]);
	}
      }

      if( iTrkMomBin>=0 && iTrkEtaBin>=0) {	 
	if((*t_trackHitInMissTIB )[itrk]<1 && (*t_trackHitInMissTID )[itrk]<1 )
	  h_trackPhi_3_Inner[iTrkEtaBin] ->Fill((*t_trackPhi)[itrk]);
	if((*t_trackHitOutMissTOB)[itrk]<1 && (*t_trackHitOutMissTEC)[itrk]<1 )
	  h_trackPhi_3_Outer[iTrkEtaBin] ->Fill((*t_trackPhi)[itrk]);
      }
      if( NLayersCrossed<7 ) trackSel=false ;

      if(dataType=="Data" && (std::abs((*t_trackDxyPV)[itrk])>0.2 || std::abs((*t_trackDzPV)[itrk])>0.2 || (*t_trackChiSq)[itrk]>5.0 || (*t_trackPVIdx)[itrk]>0 ) ) 
	trackSel = false;
      if(dataType=="MC"   && (std::abs((*t_trackDxyPV)[itrk])>0.2 || std::abs((*t_trackDzPV)[itrk])>0.2 || (*t_trackChiSq)[itrk]>5.0 ) ) 
	trackSel = false;
      if(dataType=="DiPion" && (*t_trackChiSq)[itrk]>5.0 ) trackSel = false;

      double drTrackL1=0.0;
      if( (dataType=="Data" || dataType=="MC") ) {
	if( checkL1 ){
	  drTrackL1 = DeltaR(eta1, phi1, maxL1Eta, maxL1Phi);	  

	  bool tempIso=true;
	  if( std::abs(eta1)<1.47 ) if( e15x15-e11x11 > ebNeutIso ) tempIso=false;
	  if( std::abs(eta1)>1.47 ) if( e15x15-e11x11 > eeNeutIso ) tempIso=false;
	  if (debug) std::cout << "Track Selection " << trackSel << " Isolation " << tempIso << ":" << hcalNeutIso << " dr " << drTrackL1 << ":" << dRL1Jet << std::endl;
	  if( drTrackL1<dRL1Jet) trackSel=false;
	  if( trackChargeIso &&  hcalNeutIso && tempIso) {
	    if( drTrackL1>100.0) {
	      std::cout<<"trackP,eta,phi "<<std::setw(10)<<(*t_trackP)[itrk]<<std::setw(10)<<(*t_trackEta)[itrk]
		       <<std::setw(10)<<(*t_trackPhi)[itrk]
		       <<" L1eta,phi "<<std::setw(10)<<maxL1Eta<<std::setw(10)<<maxL1Phi
		       <<" dEta,dPhi "<<std::setw(10)<<eta1-maxL1Eta<<std::setw(10)<<DeltaPhi(phi1,maxL1Phi)
		       <<" dR "<<std::setw(10)<<drTrackL1<<" maxL1Pt "<<std::setw(5)<<maxL1Pt<<" "<<checkL1
		       <<std::endl;
	    }
	    h_drTrackL1[iTrkMomBin][iTrkEtaBin]->Fill(drTrackL1);
	  }
	} else {
	  trackSel = false; // use only the events triigered by some L1
	}
      }

      if( trackSel ) {
	h_trackP[4]    ->Fill((*t_trackP)[itrk]    );  
	h_trackPt[4]   ->Fill((*t_trackPt)[itrk]   );  
	h_trackEta[4]  ->Fill((*t_trackEta)[itrk]  );  
	h_trackPhi[4]  ->Fill((*t_trackPhi)[itrk]  );  
	h_trackDxy[4]  ->Fill((*t_trackDxyPV)[itrk]);  
	h_trackDz[4]   ->Fill((*t_trackDzPV )[itrk]);  
	h_trackChiSq[4]->Fill((*t_trackChiSq)[itrk]  );  
      }
      
      //if( iTrkMomBin>=0 && iTrkEtaBin>=0 && trackNOuterHits>=4 && NLayersCrossed>=8 && trackSel) {	 
      bool ecalNeutIso = true;
      if( std::abs(eta1)<1.47 ) if( e15x15-e11x11 > ebNeutIso ) ecalNeutIso=false;
      if( std::abs(eta1)>1.47 ) if( e15x15-e11x11 > eeNeutIso ) ecalNeutIso=false;
      if (debug) std::cout << "All selection " << iTrkMomBin << ":" << iTrkEtaBin << ":" << trackSel << ":" << trackChargeIso << ":" << hcalNeutIso << ":" << ecalNeutIso << std::endl;

      if( iTrkMomBin>=0 && iTrkEtaBin>=0 && trackSel) {	 
	 
	h_maxNearP31x31[iTrkMomBin][iTrkEtaBin]->Fill( maxNearP31x31 );

	if( trackChargeIso ) {
	  h_diff_e15x15e11x11      [iTrkMomBin][iTrkEtaBin]->Fill(e15x15-e11x11);
	  h_diff_e15x15e11x11_20Sig[iTrkMomBin][iTrkEtaBin]->Fill((*t_e15x15_20Sig)[itrk]-(*t_e11x11_20Sig)[itrk]);
	  h_diff_h7x7h5x5          [iTrkMomBin][iTrkEtaBin]->Fill( h7x7-h5x5 ) ;
	}

	if( trackChargeIso &&  hcalNeutIso) { 

	  if(ecalNeutIso ) {

	    h_trackP[5]    ->Fill((*t_trackP)[itrk]    );  
	    h_trackPt[5]   ->Fill((*t_trackPt)[itrk]   );  
	    h_trackEta[5]  ->Fill((*t_trackEta)[itrk]  );  
	    h_trackPhi[5]  ->Fill((*t_trackPhi)[itrk]  );  
	    h_trackDxy[5]  ->Fill((*t_trackDxyPV)[itrk]);  
	    h_trackDz[5]   ->Fill((*t_trackDzPV )[itrk]);  
	    h_trackChiSq[5]->Fill((*t_trackChiSq)[itrk]  );  
	    
	    NIsoTrk++;

	    if( pt1>1.0 && std::abs(eta1)<2.3 ) {
	      double etot11x11=0, etot9x9=0, etot7x7=0;
	      if( std::abs(eta1)<1.7 ){
		etot11x11=e11x11+h3x3;
		etot9x9  =e9x9+h3x3;
		etot7x7  =e7x7+h3x3;
	      } else {
		etot11x11=(*t_e11x11_20Sig)[itrk]+h3x3;
		etot9x9  =(*t_e9x9_20Sig)[itrk]+h3x3;
		etot7x7  =(*t_e7x7_20Sig)[itrk]+h3x3;
	      }	      
	      h_trackPCaloE11x11H3x3[NEtaBins]->Fill( p1, etot11x11);
	      h_trackPCaloE9x9H3x3[NEtaBins]  ->Fill( p1, etot9x9);
	      h_trackPCaloE7x7H3x3[NEtaBins]  ->Fill( p1, etot7x7);
	      
	      h_trackPCaloE11x11H3x3[iTrkEtaBin]->Fill( p1, etot11x11);
	      h_trackPCaloE9x9H3x3[iTrkEtaBin]  ->Fill( p1, etot9x9);
	      h_trackPCaloE7x7H3x3[iTrkEtaBin]  ->Fill( p1, etot7x7);	       
	    }

	    if( p1>3.0 &&  p1<4.0)  h_eECAL11x11VsHCAL3x3[iTrkEtaBin]->Fill(e11x11, h3x3);
	    
	    // Ecal tranverse profile
	    h_eECAL7x7_Frac  [iTrkMomBin][iTrkEtaBin]->Fill(e7x7/p1);
	    h_eECAL9x9_Frac  [iTrkMomBin][iTrkEtaBin]->Fill(e9x9/p1);
	    h_eECAL11x11_Frac[iTrkMomBin][iTrkEtaBin]->Fill(e11x11/p1);
	    h_eECAL15x15_Frac[iTrkMomBin][iTrkEtaBin]->Fill(e15x15/p1);
	    
	    hh_eECAL7x7_Frac  [iTrkEtaBin]->Fill(e7x7/p1);
	    hh_eECAL9x9_Frac  [iTrkEtaBin]->Fill(e9x9/p1);
	    hh_eECAL11x11_Frac[iTrkEtaBin]->Fill(e11x11/p1);
	    hh_eECAL15x15_Frac[iTrkEtaBin]->Fill(e15x15/p1);
	    
	    // Hcal transverse profile
	    h_eHCAL3x3_Frac[iTrkMomBin][iTrkEtaBin]->Fill(h3x3/p1);
	    h_eHCAL5x5_Frac[iTrkMomBin][iTrkEtaBin]->Fill(h5x5/p1);
	    h_eHCAL7x7_Frac[iTrkMomBin][iTrkEtaBin]->Fill(h7x7/p1);
	    if( e7x7<0.7) {
	      h_eHCAL3x3MIP_Frac[iTrkMomBin][iTrkEtaBin]->Fill(h3x3/p1);
	      h_eHCAL5x5MIP_Frac[iTrkMomBin][iTrkEtaBin]->Fill(h5x5/p1);
	      h_eHCAL7x7MIP_Frac[iTrkMomBin][iTrkEtaBin]->Fill(h7x7/p1);
	    }
	    hh_eHCAL3x3_Frac[iTrkEtaBin]->Fill(h3x3/p1);
	    hh_eHCAL5x5_Frac[iTrkEtaBin]->Fill(h5x5/p1);
	    hh_eHCAL7x7_Frac[iTrkEtaBin]->Fill(h7x7/p1);
	    
	    // Response : Ecal+Hcal
	    h_eHCAL3x3_eECAL11x11_response[iTrkMomBin][iTrkEtaBin]->Fill((h3x3+e11x11)/p1);
	    h_eHCAL5x5_eECAL11x11_response[iTrkMomBin][iTrkEtaBin]->Fill((h5x5+e11x11)/p1);
	    h_eHCAL7x7_eECAL11x11_response[iTrkMomBin][iTrkEtaBin]->Fill((h7x7+e11x11)/p1);
	    if( e7x7<0.7) {
	      h_eHCAL3x3_eECAL11x11_responseMIP[iTrkMomBin][iTrkEtaBin]->Fill((h3x3+e11x11)/p1);
	      h_eHCAL5x5_eECAL11x11_responseMIP[iTrkMomBin][iTrkEtaBin]->Fill((h5x5+e11x11)/p1);
	      h_eHCAL7x7_eECAL11x11_responseMIP[iTrkMomBin][iTrkEtaBin]->Fill((h7x7+e11x11)/p1);
	    } else {
	      h_eHCAL3x3_eECAL11x11_responseInteract[iTrkMomBin][iTrkEtaBin]->Fill((h3x3+e11x11)/p1);
	      h_eHCAL5x5_eECAL11x11_responseInteract[iTrkMomBin][iTrkEtaBin]->Fill((h5x5+e11x11)/p1);
	      h_eHCAL7x7_eECAL11x11_responseInteract[iTrkMomBin][iTrkEtaBin]->Fill((h7x7+e11x11)/p1);
	    }
	    
	    h_eHCAL3x3_eECAL9x9_response[iTrkMomBin][iTrkEtaBin]->Fill((h3x3+e9x9)/p1);
	    h_eHCAL5x5_eECAL9x9_response[iTrkMomBin][iTrkEtaBin]->Fill((h5x5+e9x9)/p1);
	    h_eHCAL7x7_eECAL9x9_response[iTrkMomBin][iTrkEtaBin]->Fill((h7x7+e9x9)/p1);
	    if( e7x7<0.700) {
	      h_eHCAL3x3_eECAL9x9_responseMIP[iTrkMomBin][iTrkEtaBin]->Fill((h3x3+e9x9)/p1);
	      h_eHCAL5x5_eECAL9x9_responseMIP[iTrkMomBin][iTrkEtaBin]->Fill((h5x5+e9x9)/p1);
	      h_eHCAL7x7_eECAL9x9_responseMIP[iTrkMomBin][iTrkEtaBin]->Fill((h7x7+e9x9)/p1);
	    } else {
	      h_eHCAL3x3_eECAL9x9_responseInteract[iTrkMomBin][iTrkEtaBin]->Fill((h3x3+e9x9)/p1);
	      h_eHCAL5x5_eECAL9x9_responseInteract[iTrkMomBin][iTrkEtaBin]->Fill((h5x5+e9x9)/p1);
	      h_eHCAL7x7_eECAL9x9_responseInteract[iTrkMomBin][iTrkEtaBin]->Fill((h7x7+e9x9)/p1);
	    }
	    
	    h_eHCAL3x3_eECAL7x7_response[iTrkMomBin][iTrkEtaBin]->Fill((h3x3+e7x7)/p1);
	    h_eHCAL5x5_eECAL7x7_response[iTrkMomBin][iTrkEtaBin]->Fill((h5x5+e7x7)/p1);
	    h_eHCAL7x7_eECAL7x7_response[iTrkMomBin][iTrkEtaBin]->Fill((h7x7+e7x7)/p1);
	    if( e7x7<0.700) {
	      h_eHCAL3x3_eECAL7x7_responseMIP[iTrkMomBin][iTrkEtaBin]->Fill((h3x3+e7x7)/p1);
	      h_eHCAL5x5_eECAL7x7_responseMIP[iTrkMomBin][iTrkEtaBin]->Fill((h5x5+e7x7)/p1);
	      h_eHCAL7x7_eECAL7x7_responseMIP[iTrkMomBin][iTrkEtaBin]->Fill((h7x7+e7x7)/p1);
	    } else {
	      h_eHCAL3x3_eECAL7x7_responseInteract[iTrkMomBin][iTrkEtaBin]->Fill((h3x3+e7x7)/p1);
	      h_eHCAL5x5_eECAL7x7_responseInteract[iTrkMomBin][iTrkEtaBin]->Fill((h5x5+e7x7)/p1);
	      h_eHCAL7x7_eECAL7x7_responseInteract[iTrkMomBin][iTrkEtaBin]->Fill((h7x7+e7x7)/p1);
	    }	    
	  } // ecal neutral isolation


	  // cut on xtal noise


	  bool ecalNeutIso20Sig = true;
	  if( std::abs(eta1)<1.47 ) if( (*t_e15x15_20Sig)[itrk]-(*t_e11x11_20Sig)[itrk] > ebNeutIso ) ecalNeutIso20Sig=false;
	  if( std::abs(eta1)>1.47 ) if( (*t_e15x15_20Sig)[itrk]-(*t_e11x11_20Sig)[itrk] > eeNeutIso ) ecalNeutIso20Sig=false;	  
	  if(ecalNeutIso20Sig) {

	    h_meanTrackP[iTrkMomBin][iTrkEtaBin]->Fill(p1);
	    h_eECAL11x11_Frac_20Sig[iTrkMomBin][iTrkEtaBin]->Fill((*t_e11x11_20Sig)[itrk]/p1);
	    h_eECAL9x9_Frac_20Sig  [iTrkMomBin][iTrkEtaBin]->Fill((*t_e9x9_20Sig)[itrk]/p1);
	    h_eECAL7x7_Frac_20Sig  [iTrkMomBin][iTrkEtaBin]->Fill((*t_e7x7_20Sig)[itrk]/p1);

	    hh_eECAL11x11_Frac_20Sig[iTrkEtaBin]->Fill((*t_e11x11_20Sig)[itrk]/p1);
	    hh_eECAL9x9_Frac_20Sig  [iTrkEtaBin]->Fill((*t_e9x9_20Sig)[itrk]/p1);
	    hh_eECAL7x7_Frac_20Sig  [iTrkEtaBin]->Fill((*t_e7x7_20Sig)[itrk]/p1);

	    h_eHCAL3x3_Frac_20Sig[iTrkMomBin][iTrkEtaBin]->Fill(h3x3/p1);
	    h_eHCAL5x5_Frac_20Sig[iTrkMomBin][iTrkEtaBin]->Fill(h5x5/p1);
	    h_eHCAL7x7_Frac_20Sig[iTrkMomBin][iTrkEtaBin]->Fill(h7x7/p1);
	    if((*t_e7x7_20Sig)[itrk]<0.7 ) {
	      h_eHCAL3x3MIP_Frac_20Sig[iTrkMomBin][iTrkEtaBin]->Fill(h3x3/p1);
	      h_eHCAL5x5MIP_Frac_20Sig[iTrkMomBin][iTrkEtaBin]->Fill(h5x5/p1);
	      h_eHCAL7x7MIP_Frac_20Sig[iTrkMomBin][iTrkEtaBin]->Fill(h7x7/p1);
	    }
	    hh_eHCAL3x3_Frac_20Sig[iTrkEtaBin]->Fill(h3x3/p1);
	    hh_eHCAL5x5_Frac_20Sig[iTrkEtaBin]->Fill(h5x5/p1);
	    hh_eHCAL7x7_Frac_20Sig[iTrkEtaBin]->Fill(h7x7/p1);

	    h_eHCAL3x3_eECAL11x11_response_20Sig[iTrkMomBin][iTrkEtaBin]->Fill((h3x3+(*t_e11x11_20Sig)[itrk])/p1);
	    if((*t_e7x7_20Sig)[itrk]<0.7 ) {
	      h_eHCAL3x3_eECAL11x11_responseMIP_20Sig[iTrkMomBin][iTrkEtaBin]->Fill((h3x3+(*t_e11x11_20Sig)[itrk])/p1);
	    } else {
	      h_eHCAL3x3_eECAL11x11_responseInteract_20Sig[iTrkMomBin][iTrkEtaBin]->Fill((h3x3+(*t_e11x11_20Sig)[itrk])/p1);
	    }
	    h_eHCAL3x3_eECAL9x9_response_20Sig  [iTrkMomBin][iTrkEtaBin]->Fill((h3x3+(*t_e9x9_20Sig)[itrk])/p1);
	    if((*t_e7x7_20Sig)[itrk]<0.7 ) {
	      h_eHCAL3x3_eECAL9x9_responseMIP_20Sig  [iTrkMomBin][iTrkEtaBin]->Fill((h3x3+(*t_e9x9_20Sig)[itrk])/p1);
	    } else {
	      h_eHCAL3x3_eECAL9x9_responseInteract_20Sig  [iTrkMomBin][iTrkEtaBin]->Fill((h3x3+(*t_e9x9_20Sig)[itrk])/p1);
	    }
	    h_eHCAL3x3_eECAL7x7_response_20Sig  [iTrkMomBin][iTrkEtaBin]->Fill((h3x3+(*t_e7x7_20Sig)[itrk])/p1);
	    if((*t_e7x7_20Sig)[itrk]<0.7 ) {
	      h_eHCAL3x3_eECAL7x7_responseMIP_20Sig  [iTrkMomBin][iTrkEtaBin]->Fill((h3x3+(*t_e7x7_20Sig)[itrk])/p1);
	    } else {
	      h_eHCAL3x3_eECAL7x7_responseInteract_20Sig  [iTrkMomBin][iTrkEtaBin]->Fill((h3x3+(*t_e7x7_20Sig)[itrk])/p1);
	    }
	    h_eHCAL5x5_eECAL11x11_response_20Sig[iTrkMomBin][iTrkEtaBin]->Fill((h5x5+(*t_e11x11_20Sig)[itrk])/p1);
	    if((*t_e7x7_20Sig)[itrk]<0.7 ) {
	      h_eHCAL5x5_eECAL11x11_responseMIP_20Sig[iTrkMomBin][iTrkEtaBin]->Fill((h5x5+(*t_e11x11_20Sig)[itrk])/p1);
	    } else {
	      h_eHCAL5x5_eECAL11x11_responseInteract_20Sig[iTrkMomBin][iTrkEtaBin]->Fill((h5x5+(*t_e11x11_20Sig)[itrk])/p1);
	    }

	  }

	}	 // if charged & hcal iso
	
      } // momentum and eta bins 
      
    } // loop over tracks in the event 

    if( runNIsoTrkList.find(t_RunNo) != runNIsoTrkList.end() ) {
      runNIsoTrkList[t_RunNo] += NIsoTrk;
    } else {
      runNIsoTrkList.insert( std::pair<unsigned int, unsigned int>(t_RunNo,NIsoTrk) );
      if (debug) std::cout << "runNo " << t_RunNo <<" "<<runNIsoTrkList[t_RunNo]<<std::endl;
    }
  
  } //loop over tree entries

  std::cout<<"Number of entries in tree "<<nentries                                                         << "\n"
	   <<"nEventsGoodRuns           "<<nEventsGoodRuns                                                  << "\n"
	   <<"nEventsValidPV            "<<nEventsValidPV  << " " <<(double)nEventsValidPV/nEventsGoodRuns  << "\n"
	   <<"nEventsPVTracks           "<<nEventsPVTracks << " " <<(double)nEventsPVTracks/nEventsGoodRuns << "\n"
	   << std::endl;

  std::cout << "saved runEvtList " << std::endl;
  std::map<unsigned int, unsigned int>::iterator runEvtListItr = runEvtList.begin();
  long total=0;
  for(runEvtListItr=runEvtList.begin(); runEvtListItr != runEvtList.end(); runEvtListItr++) {
    std::cout<<runEvtListItr->first << " "<< runEvtListItr->second << std::endl;
    total += runEvtListItr->second;
  }
  std::cout<<"Total   " << total << std::endl << std::endl;

  total = 0;
  std::cout << "Number of tracks in runs " << std::endl;
  std::map<unsigned int, unsigned int>::iterator runNTrkListItr = runNTrkList.begin();
  for(runNTrkListItr=runNTrkList.begin(); runNTrkListItr != runNTrkList.end(); runNTrkListItr++) {
    std::cout<<runNTrkListItr->first << " "<< runNTrkListItr->second << std::endl;
    total += runNTrkListItr->second;
  }
  std::cout<<"Total   " << total << std::endl << std::endl;

  total = 0;
  std::cout << "Number of isolated tracks in runs " << std::endl;
  std::map<unsigned int, unsigned int>::iterator runNIsoTrkListItr = runNIsoTrkList.begin();
  for(runNIsoTrkListItr=runNIsoTrkList.begin(); runNIsoTrkListItr != runNIsoTrkList.end(); runNIsoTrkListItr++) {
    std::cout<<runNIsoTrkListItr->first << " "<< runNIsoTrkListItr->second << std::endl;
    total += runNIsoTrkListItr->second;
  }
  std::cout<<"Total   " << total << std::endl << std::endl;  
}

Bool_t TreeAnalysisHcalNZS::Notify() {
  // The Notify() function is called when a new file is opened. This
  // can be either for a new TTree in a TChain or when when a new TTree
  // is started when using PROOF. It is normally not necessary to make changes
  // to the generated code, but the routine can be extended by the
  // user if needed. The return value is currently not used.

  return kTRUE;
}

void TreeAnalysisHcalNZS::Show(Long64_t entry) {
  // Print contents of entry.
  // If entry is not specified, print current entry
  if (!fChain) return;
  fChain->Show(entry);
}

double TreeAnalysisHcalNZS::DeltaPhi(double v1, double v2) {
  // Computes the correctly normalized phi difference 
  // v1, v2 = phi of object 1 and 2 

  double pi    = 3.141592654;
  double twopi = 6.283185307;

  double diff = std::abs(v2 - v1);
  double corr = twopi - diff;
  if (diff < pi){ return diff;} else { return corr;}
}

double TreeAnalysisHcalNZS::DeltaR(double eta1, double phi1, double eta2, double phi2) {
  double deta = eta1 - eta2;
  double dphi = DeltaPhi(phi1, phi2);
  return std::sqrt(deta*deta + dphi*dphi);
}

void TreeAnalysisHcalNZS::BookHistograms(const char *outFileName) {

  fout = new TFile(outFileName, "RECREATE");

  fout->cd();

  char hname[1000], htit[1000];

  TDirectory *d_caloTrkPProfile = fout->mkdir( "ProfileTrackPcaloEne" );
  d_caloTrkPProfile->cd();
  for (int ibin=0; ibin<=NEtaBins; ibin++) {
    if (ibin == NEtaBins) sprintf (htit, "CaloEne(E11x11+H3x3) as trackP:|#eta|<%f", genPartEtaBins[NEtaBins]);
    else sprintf (htit, "CaloEne(E11x11+H3x3) as trackP:%f<|#eta|<%f", genPartEtaBins[ibin],genPartEtaBins[ibin+1]);
    sprintf (hname, "h_trackPCaloE11x11H3x3_%d", ibin);
    h_trackPCaloE11x11H3x3[ibin] = new TProfile(hname, htit,   15, genPartPBins);
    if (ibin == NEtaBins) sprintf (htit, "CaloEne(E9x9+H3x3) as trackP:|#eta|<%f", genPartEtaBins[NEtaBins]);
    else sprintf (htit, "CaloEne(E9x9+H3x3) as trackP:%f<|#eta|<%f", genPartEtaBins[ibin],genPartEtaBins[ibin+1]);
    sprintf (hname, "h_trackPCaloE9x9H3x3_%d", ibin);
    h_trackPCaloE9x9H3x3[ibin] = new TProfile(hname, htit,   15, genPartPBins);
    if (ibin == NEtaBins) sprintf (htit, "CaloEne(E7x7+H3x3) as trackP:|#eta|<%f", genPartEtaBins[NEtaBins]);
    else sprintf (htit, "CaloEne(E7x7+H3x3) as trackP:%f<|#eta|<%f", genPartEtaBins[ibin],genPartEtaBins[ibin+1]);
    sprintf (hname, "h_trackPCaloE7x7H3x3_%d", ibin);
    h_trackPCaloE7x7H3x3[ibin] = new TProfile(hname, htit,   15, genPartPBins);
  }

  fout->cd();
  for (int i=0; i<2; ++i) {
    sprintf (hname, "h_NPV_%d", i);
    h_NPV[i]              = new TH1F(hname, hname,    10, -0.5,   9.5); 
    h_NPV[i]              ->Sumw2(); 
    sprintf (hname, "h_PVx_%d", i);
    h_PVx[i]              = new TH1F(hname, hname,    40, -2.0,   2.0); 
    h_PVx[i]              ->Sumw2(); 
    sprintf (hname, "h_PVy_%d", i);
    h_PVy[i]              = new TH1F(hname, hname,    40, -2.0,   2.0); 
    h_PVy[i]              ->Sumw2(); 
    sprintf (hname, "h_PVr_%d", i);
    h_PVr[i]              = new TH1F(hname, hname,   100,  0.0,   3.0); 
    h_PVr[i]              ->Sumw2(); 
    sprintf (hname, "h_PVz_%d", i);
    h_PVz[i]              = new TH1F(hname, hname,   100,-20.0,  20.0); 
    h_PVz[i]              ->Sumw2(); 
    sprintf (hname, "h_PVNDOF_%d", i);
    h_PVNDOF[i]           = new TH1F(hname, hname,   300, -0.5, 299.5); 
    h_PVNDOF[i]           ->Sumw2();
    sprintf (hname, "h_PVNTracks_%d", i);
    h_PVNTracks[i]        = new TH1F(hname, hname,   300,  0.0, 300.0); 
    h_PVNTracks[i]        ->Sumw2(); 
    sprintf (hname, "h_PVTracksSumPt_%d", i);
    h_PVTracksSumPt[i]    = new TH1F(hname, hname,   300,  0.0, 300.0);
    h_PVTracksSumPt[i]    ->Sumw2();
    sprintf (hname, "h_PVNTracksWt_%d", i);
    h_PVNTracksWt[i]      = new TH1F(hname, hname,   300,  0.0, 300.0); 
    h_PVNTracksWt[i]      ->Sumw2(); 
    sprintf (hname, "h_PVTracksSumPtWt_%d", i);
    h_PVTracksSumPtWt[i]  = new TH1F(hname, hname,   300,  0.0, 300.0);
    h_PVTracksSumPtWt[i]  ->Sumw2();
    sprintf (hname, "h_PVNTracksSumPt_%d", i);
    h_PVNTracksSumPt[i] = new TH2F(hname, hname, 100,0.0,100.0, 100,0.0,100.0);
  }

  std::string subtitle[6] = {"All HighPirity Tracks",
			     "good first PV & Iso in Ecal31x31",
			     "good first PV & Iso in Ecal31x31 & Pt>1.0",
			     "good first PV & Iso in Ecal31x31 & Pt>1.0 & NoIntInTracker",
			     "good first PV & Iso in Ecal31x31 & Pt>1.0 & NoIntInTracker & |dxy|<0.2,|dz|<0.2,chisq<5.0",
			     "Fully Isolated"};
  for (int i=0; i<6; ++i) {
    sprintf (hname, "h_trackP_%d", i);
    sprintf (htit,  "P:    %s", subtitle[i].c_str());
    h_trackP[i]     = new TH1F(hname, htit,   15, genPartPBins);
    h_trackP[i]     ->Sumw2();
    sprintf (hname, "h_trackPt_%d", i);
    sprintf (htit,  "Pt:   %s", subtitle[i].c_str());
    h_trackPt[i]    = new TH1F(hname, htit,   15, genPartPBins);
    h_trackPt[i]    ->Sumw2();
    sprintf (hname, "h_trackEta_%d", i);
    sprintf (htit,  "Eta:  %s", subtitle[i].c_str());
    h_trackEta[i]  = new TH1F(hname, htit,   100, -3.0, 3.0);
    h_trackEta[i]  ->Sumw2();
    sprintf (hname, "h_trackPhi_%d", i);
    sprintf (htit,  "Phi:  %s", subtitle[i].c_str());
    h_trackPhi[i]   = new TH1F(hname, htit,  100, -3.14159, 3.14159);
    h_trackPhi[i]  ->Sumw2();
    sprintf (hname, "h_trackChiSq_%d", i);
    sprintf (htit,  "ChiSq:%s", subtitle[i].c_str());
    h_trackChiSq[i] = new TH1F(hname, htit,  200,  0.0,20.0);
    h_trackChiSq[i]->Sumw2();  
    sprintf (hname, "h_trackDxyPV_%d", i);
    sprintf (htit,  "DxyPV:%s", subtitle[i].c_str());
    h_trackDxy[i]   = new TH1F(hname, htit,  200, -1.0, 1.0);
    h_trackDxy[i]  ->Sumw2();  
    sprintf (hname, "h_trackDzPV_%d", i);
    sprintf (htit,  "DzPV:%s", subtitle[i].c_str());
    h_trackDz[i]    = new TH1F(hname, htit,  200, -1.0, 1.0);
    h_trackDz[i]   ->Sumw2();  
  }

  for(int ieta=0; ieta<NEtaBins; ieta++) {
    double lowEta=-5.0, highEta= 5.0;
    lowEta  = genPartEtaBins[ieta];
    highEta = genPartEtaBins[ieta+1];
    
     sprintf(hname, "h_trackPhi_2_2_etaBin%i",ieta);
     sprintf(htit,  "h_trackPhi(pt>1.0, iso in 31x31) : %3.2f<|#eta|<%3.2f)", lowEta, highEta);
    h_trackPhi_2_2[ieta]= new TH1F(hname, htit,  100, -3.14159, 3.14159);
     sprintf(hname, "h_trackPhi_3_3_etaBin%i",ieta);
     sprintf(htit,  "h_trackPhi(pt>1.0, iso in 31x31,no missing hit) : %3.2f<|#eta|<%3.2f)", lowEta, highEta);
    h_trackPhi_3_3[ieta]= new TH1F(hname, htit,  100, -3.14159, 3.14159);

     sprintf(hname, "h_trackPhi_3_Inner_etaBin%i",ieta);
     sprintf(htit,  "h_trackPhi(pt>1.0, iso in 31x31,no missing Inner hit) : %3.2f<|#eta|<%3.2f)", lowEta, highEta);
    h_trackPhi_3_Inner[ieta]= new TH1F(hname, htit,  100, -3.14159, 3.14159);

     sprintf(hname, "h_trackPhi_3_Outer_etaBin%i",ieta);
     sprintf(htit,  "h_trackPhi(pt>1.0, iso in 31x31,no missing Outer hit) : %3.2f<|#eta|<%3.2f)", lowEta, highEta);
    h_trackPhi_3_Outer[ieta]= new TH1F(hname, htit,  100, -3.14159, 3.14159);

  }

  TDirectory *d_meanTrackP   = fout->mkdir( "MeanTrackP" );
  TDirectory *d_maxNearP     = fout->mkdir( "MaxNearP" );
  TDirectory *d_neutralIso   = fout->mkdir( "NeutralIsolation" );
  TDirectory* d_trProf1      = fout->mkdir( "EcalTranverseProfile" );
  TDirectory* d_trProf2      = fout->mkdir( "HcalTranverseProfile" );
  TDirectory* d_response     = fout->mkdir( "Response" );
  TDirectory* d_response10Sig= fout->mkdir( "Response10Sig" );
  TDirectory* d_response15Sig= fout->mkdir( "Response15Sig" ); 
  TDirectory* d_response20Sig= fout->mkdir( "Response20Sig" );
  TDirectory* d_response25Sig= fout->mkdir( "Response25Sig" );
  TDirectory* d_response30Sig= fout->mkdir( "Response30Sig" );
  TDirectory *d_drTrackL1    = fout->mkdir( "DrTrackL1" );

  for (int ieta=0; ieta<NEtaBins; ieta++) {
    double lowEta=-5.0, highEta= 5.0;
    lowEta  = genPartEtaBins[ieta];
    highEta = genPartEtaBins[ieta+1];

    sprintf(hname, "h_eECAL11x11VsHCAL3x3_etaBin%i",ieta);
    sprintf(htit,  "eEvseH : %3.2f<|#eta|<%3.2f)", lowEta, highEta);
    h_eECAL11x11VsHCAL3x3[ieta] = new TH2F(hname, hname, 220, -1.0, 10.0, 220, -1.0, 10.0);

    d_trProf1->cd();
    sprintf(hname, "hh_eECAL7x7Frac_etaBin%i", ieta);
    sprintf(htit,  "eECAL7x7/trkP (%3.2f<|#eta|<%3.2f)", lowEta, highEta );
    hh_eECAL7x7_Frac[ieta] =  new TH1F(hname, htit, 500, -1.0, 4.0);
    hh_eECAL7x7_Frac[ieta] ->Sumw2();
    sprintf(hname, "hh_eECAL9x9Frac_etaBin%i", ieta);
    sprintf(htit,  "eECAL9x9/trkP (%3.2f<|#eta|<%3.2f)", lowEta, highEta );
    hh_eECAL9x9_Frac[ieta] =  new TH1F(hname, htit, 500, -1.0, 4.0);
    hh_eECAL9x9_Frac[ieta] ->Sumw2();
    sprintf(hname, "hh_eECAL11x11Frac_etaBin%i", ieta);
    sprintf(htit,  "eECAL11x11/trkP (%3.2f<|#eta|<%3.2f)", lowEta, highEta );
    hh_eECAL11x11_Frac[ieta] =  new TH1F(hname, htit, 500, -1.0, 4.0);
    hh_eECAL11x11_Frac[ieta] ->Sumw2();
    sprintf(hname, "hh_eECAL15x15Frac_etaBin%i", ieta);
    sprintf(htit,  "eECAL15x15/trkP (%3.2f<|#eta|<%3.2f)", lowEta, highEta );
    hh_eECAL15x15_Frac[ieta] =  new TH1F(hname, htit, 500, -1.0, 4.0);
    hh_eECAL15x15_Frac[ieta] ->Sumw2();

    sprintf(hname, "hh_eECAL7x7Frac20Sig_etaBin%i", ieta);
    sprintf(htit,  "eECAL7x7/trkP(Xtal>2.0#sigma) (%3.2f<|#eta|<%3.2f)", lowEta, highEta );
    hh_eECAL7x7_Frac_20Sig[ieta] =  new TH1F(hname, htit, 500, -1.0, 4.0);
    hh_eECAL7x7_Frac_20Sig[ieta] ->Sumw2();
    sprintf(hname, "hh_eECAL9x9Frac20Sig_etaBin%i", ieta);
    sprintf(htit,  "eECAL9x9/trkP(Xtal>2.0#sigma) (%3.2f<|#eta|<%3.2f)", lowEta, highEta );
    hh_eECAL9x9_Frac_20Sig[ieta] =  new TH1F(hname, htit, 500, -1.0, 4.0);
    hh_eECAL9x9_Frac_20Sig[ieta] ->Sumw2();
    sprintf(hname, "hh_eECAL11x11Frac20Sig_etaBin%i", ieta);
    sprintf(htit,  "eECAL11x11/trkP(Xtal>2.0#sigma) (%3.2f<|#eta|<%3.2f)", lowEta, highEta );
    hh_eECAL11x11_Frac_20Sig[ieta] =  new TH1F(hname, htit, 500, -1.0, 4.0);
    hh_eECAL11x11_Frac_20Sig[ieta] ->Sumw2();
    
    d_trProf2->cd();
    sprintf(hname, "hh_eHCAL3x3Frac_etaBin%i", ieta);
    sprintf(htit,  "eHCAL3x3/trkP (%3.2f<|#eta|<%3.2f)", lowEta, highEta );
    hh_eHCAL3x3_Frac[ieta] = new TH1F(hname, htit, 500, -2.0, 4.0);
    hh_eHCAL3x3_Frac[ieta] ->Sumw2();
    sprintf(hname, "hh_eHCAL5x5Frac_etaBin%i", ieta);
    sprintf(htit,  "eHCAL5x5/trkP (%3.2f<|#eta|<%3.2f)", lowEta, highEta );
    hh_eHCAL5x5_Frac[ieta] = new TH1F(hname, htit, 500, -2.0, 4.0);
    hh_eHCAL5x5_Frac[ieta] ->Sumw2();
    sprintf(hname, "hh_eHCAL7x7Frac_etaBin%i", ieta);
    sprintf(htit,  "eHCAL7x7/trkP (%3.2f<|#eta|<%3.2f)", lowEta, highEta );
    hh_eHCAL7x7_Frac[ieta] = new TH1F(hname, htit, 500, -2.0, 4.0);
    hh_eHCAL7x7_Frac[ieta] ->Sumw2();

    sprintf(hname, "hh_eHCAL3x3Frac20Sig_etaBin%i", ieta);
    sprintf(htit,  "eHCAL3x3/trkP(Xtal>2.0#sigma) (%3.2f<|#eta|<%3.2f)", lowEta, highEta );
    hh_eHCAL3x3_Frac_20Sig[ieta] = new TH1F(hname, htit, 500, -2.0, 4.0);
    hh_eHCAL3x3_Frac_20Sig[ieta] ->Sumw2();
    sprintf(hname, "hh_eHCAL5x5Frac20Sig_etaBin%i", ieta);
    sprintf(htit,  "eHCAL5x5/trkP(Xtal>2.0#sigma) (%3.2f<|#eta|<%3.2f)", lowEta, highEta );
    hh_eHCAL5x5_Frac_20Sig[ieta] = new TH1F(hname, htit, 500, -2.0, 4.0);
    hh_eHCAL5x5_Frac_20Sig[ieta] ->Sumw2();
    sprintf(hname, "hh_eHCAL7x7Frac20Sig_etaBin%i", ieta);
    sprintf(htit,  "eHCAL7x7/trkP(Xtal>2.0#sigma) (%3.2f<|#eta|<%3.2f)", lowEta, highEta );
    hh_eHCAL7x7_Frac_20Sig[ieta] = new TH1F(hname, htit, 500, -2.0, 4.0);
    hh_eHCAL7x7_Frac_20Sig[ieta] ->Sumw2();

    for(int ipt=0; ipt<NPBins; ipt++) {
      double lowP=0.0, highP=300.0;
      lowP    = genPartPBins[ipt];
      highP   = genPartPBins[ipt+1];

      d_neutralIso->cd();
      sprintf(hname, "h_diff_e15x15e11x11_ptBin%i_etaBin%i", ipt, ieta);
      sprintf(htit,  "h_diff_e15x15e11x11: (%3.2f<|#eta|<%3.2f), (%2.0f<trkP<%3.0f)", lowEta, highEta, lowP, highP );
      h_diff_e15x15e11x11      [ipt][ieta] = new TH1F(hname, htit, 600, -10.0, 50.0);
      sprintf(hname, "h_diff20Sig_e15x15e11x11_ptBin%i_etaBin%i", ipt, ieta);
      sprintf(htit,  "h_diff20Sig_e15x15e11x11: (%3.2f<|#eta|<%3.2f), (%2.0f<trkP<%3.0f)", lowEta, highEta, lowP, highP );
      h_diff_e15x15e11x11_20Sig[ipt][ieta] = new TH1F(hname, htit, 600, -10.0, 50.0);
      sprintf(hname, "h_diff_h7x7h5x5_ptBin%i_etaBin%i", ipt, ieta);
      sprintf(htit,  "h_diff_h7x7h5x5: (%3.2f<|#eta|<%3.2f), (%2.0f<trkP<%3.0f)", lowEta, highEta, lowP, highP );
      h_diff_h7x7h5x5[ipt][ieta] = new TH1F(hname, htit, 600, -10.0, 50.0);

      d_maxNearP->cd();
      sprintf(hname, "h_maxNearP31x31_ptBin%i_etaBin%i",ipt, ieta);
      sprintf(htit,  "maxNearP in 31x31 (%3.2f<|#eta|<%3.2f), (%2.0f<trkP<%3.0f)", lowEta, highEta, lowP, highP );
      h_maxNearP31x31[ipt][ieta] = new TH1F(hname, htit, 220, -2.0, 20.0);
      h_maxNearP31x31[ipt][ieta] ->Sumw2();
	
      //==============================
      // Ecal plots
      //==============================
      d_trProf1->cd();
      sprintf(hname, "h_eECAL7x7Frac_ptBin%i_etaBin%i", ipt, ieta);
      sprintf(htit,  "eECAL7x7/trkP (%3.2f<|#eta|<%3.2f), (%2.0f<trkP<%3.0f)", lowEta, highEta, lowP, highP );
      h_eECAL7x7_Frac[ipt][ieta] =  new TH1F(hname, htit, 1500, -1.0, 4.0);
      h_eECAL7x7_Frac[ipt][ieta] ->Sumw2();
      sprintf(hname, "h_eECAL9x9Frac_ptBin%i_etaBin%i", ipt, ieta);
      sprintf(htit,  "eECAL9x9/trkP (%3.2f<|#eta|<%3.2f), (%2.0f<trkP<%3.0f)", lowEta, highEta, lowP, highP );
      h_eECAL9x9_Frac[ipt][ieta] =  new TH1F(hname, htit, 1500, -1.0, 4.0);
      h_eECAL9x9_Frac[ipt][ieta] ->Sumw2();
      sprintf(hname, "h_eECAL11x11Frac_ptBin%i_etaBin%i", ipt, ieta);
      sprintf(htit,  "eECAL11x11/trkP (%3.2f<|#eta|<%3.2f), (%2.0f<trkP<%3.0f)", lowEta, highEta, lowP, highP );
      h_eECAL11x11_Frac[ipt][ieta] =  new TH1F(hname, htit, 1500, -1.0, 4.0);
      h_eECAL11x11_Frac[ipt][ieta] ->Sumw2();
      sprintf(hname, "h_eECAL15x15Frac_ptBin%i_etaBin%i", ipt, ieta);
      sprintf(htit,  "eECAL15x15/trkP (%3.2f<|#eta|<%3.2f), (%2.0f<trkP<%3.0f)", lowEta, highEta, lowP, highP );
      h_eECAL15x15_Frac[ipt][ieta] =  new TH1F(hname, htit, 1500, -1.0, 4.0);
      h_eECAL15x15_Frac[ipt][ieta] ->Sumw2();
	
      //==== additional plots with Sigma cut
      sprintf(hname, "h_eECAL7x7Frac20Sig_ptBin%i_etaBin%i", ipt, ieta);
      sprintf(htit,  "eECAL7x7/trkP(Xtal>2.0#sigma) (%3.2f<|#eta|<%3.2f), (%2.0f<trkP<%3.0f)", lowEta, highEta, lowP, highP );
      h_eECAL7x7_Frac_20Sig[ipt][ieta] =  new TH1F(hname, htit, 1500, -1.0, 4.0);
      h_eECAL7x7_Frac_20Sig[ipt][ieta] ->Sumw2();
      sprintf(hname, "h_eECAL9x9Frac20Sig_ptBin%i_etaBin%i", ipt, ieta);
      sprintf(htit,  "eECAL9x9/trkP(Xtal>2.0#sigma) (%3.2f<|#eta|<%3.2f), (%2.0f<trkP<%3.0f)", lowEta, highEta, lowP, highP );
      h_eECAL9x9_Frac_20Sig[ipt][ieta] =  new TH1F(hname, htit, 1500, -1.0, 4.0);
      h_eECAL9x9_Frac_20Sig[ipt][ieta] ->Sumw2();
      sprintf(hname, "h_eECAL11x11Frac20Sig_ptBin%i_etaBin%i", ipt, ieta);
      sprintf(htit,  "eECAL11x11/trkP(Xtal>2.0#sigma) (%3.2f<|#eta|<%3.2f), (%2.0f<trkP<%3.0f)", lowEta, highEta, lowP, highP );
      h_eECAL11x11_Frac_20Sig[ipt][ieta] =  new TH1F(hname, htit, 1500, -1.0, 4.0);
      h_eECAL11x11_Frac_20Sig[ipt][ieta] ->Sumw2();
      //==============================
      // Hcal plots
      //==============================
      d_trProf2->cd();
      sprintf(hname, "h_eHCAL3x3Frac_ptBin%i_etaBin%i", ipt, ieta);
      sprintf(htit,  "eHCAL3x3/trkP (%3.2f<|#eta|<%3.2f), (%2.0f<trkP<%3.0f)", lowEta, highEta, lowP, highP );
      h_eHCAL3x3_Frac[ipt][ieta] = new TH1F(hname, htit, 1500, -1.0, 4.0);
      h_eHCAL3x3_Frac[ipt][ieta] ->Sumw2();
      sprintf(hname, "h_eHCAL5x5Frac_ptBin%i_etaBin%i", ipt, ieta);
      sprintf(htit,  "eHCAL5x5/trkP (%3.2f<|#eta|<%3.2f), (%2.0f<trkP<%3.0f)", lowEta, highEta, lowP, highP );
      h_eHCAL5x5_Frac[ipt][ieta] = new TH1F(hname, htit, 1500, -1.0, 4.0);
      h_eHCAL5x5_Frac[ipt][ieta] ->Sumw2();
      sprintf(hname, "h_eHCAL7x7Frac_ptBin%i_etaBin%i", ipt, ieta);
      sprintf(htit,  "eHCAL7x7/trkP (%3.2f<|#eta|<%3.2f), (%2.0f<trkP<%3.0f)", lowEta, highEta, lowP, highP );
      h_eHCAL7x7_Frac[ipt][ieta] = new TH1F(hname, htit, 1500, -1.0, 4.0);
      h_eHCAL7x7_Frac[ipt][ieta] ->Sumw2();
      
      sprintf(hname, "h_eHCAL3x3Frac20Sig_ptBin%i_etaBin%i", ipt, ieta);
      sprintf(htit,  "eHCAL3x3/trkP(Xtal>2.0#sigma) (%3.2f<|#eta|<%3.2f), (%2.0f<trkP<%3.0f)", lowEta, highEta, lowP, highP );
      h_eHCAL3x3_Frac_20Sig[ipt][ieta] = new TH1F(hname, htit, 1500, -1.0, 4.0);
      h_eHCAL3x3_Frac_20Sig[ipt][ieta] ->Sumw2();
      sprintf(hname, "h_eHCAL5x5Frac20Sig_ptBin%i_etaBin%i", ipt, ieta);
      sprintf(htit,  "eHCAL5x5/trkP(Xtal>2.0#sigma) (%3.2f<|#eta|<%3.2f), (%2.0f<trkP<%3.0f)", lowEta, highEta, lowP, highP );
      h_eHCAL5x5_Frac_20Sig[ipt][ieta] = new TH1F(hname, htit, 1500, -1.0, 4.0);
      h_eHCAL5x5_Frac_20Sig[ipt][ieta] ->Sumw2();
      sprintf(hname, "h_eHCAL7x7Frac20Sig_ptBin%i_etaBin%i", ipt, ieta);
      sprintf(htit,  "eHCAL7x7/trkP(Xtal>2.0#sigma) (%3.2f<|#eta|<%3.2f), (%2.0f<trkP<%3.0f)", lowEta, highEta, lowP, highP );
      h_eHCAL7x7_Frac_20Sig[ipt][ieta] = new TH1F(hname, htit, 1500, -1.0, 4.0);
      h_eHCAL7x7_Frac_20Sig[ipt][ieta] ->Sumw2();
      
      sprintf(hname, "h_eHCAL3x3FracMIP_ptBin%i_etaBin%i", ipt, ieta);
      sprintf(htit,  "eHCAL3x3MIP/trkP (%3.2f<|#eta|<%3.2f), (%2.0f<trkP<%3.0f)", lowEta, highEta, lowP, highP );
      h_eHCAL3x3MIP_Frac[ipt][ieta] = new TH1F(hname, htit, 1500, -1.0, 4.0);
      h_eHCAL3x3MIP_Frac[ipt][ieta] ->Sumw2();
      sprintf(hname, "h_eHCAL5x5FracMIP_ptBin%i_etaBin%i", ipt, ieta);
      sprintf(htit,  "eHCAL5x5MIP/trkP (%3.2f<|#eta|<%3.2f), (%2.0f<trkP<%3.0f)", lowEta, highEta, lowP, highP );
      h_eHCAL5x5MIP_Frac[ipt][ieta] = new TH1F(hname, htit, 1500, -1.0, 4.0);
      h_eHCAL5x5MIP_Frac[ipt][ieta] ->Sumw2();
      sprintf(hname, "h_eHCAL7x7FracMIP_ptBin%i_etaBin%i", ipt, ieta);
      sprintf(htit,  "eHCAL7x7MIP/trkP (%3.2f<|#eta|<%3.2f), (%2.0f<trkP<%3.0f)", lowEta, highEta, lowP, highP );
      h_eHCAL7x7MIP_Frac[ipt][ieta] = new TH1F(hname, htit, 1500, -1.0, 4.0);
      h_eHCAL7x7MIP_Frac[ipt][ieta] ->Sumw2();
      
      sprintf(hname, "h_eHCAL3x3FracMIP20Sig_ptBin%i_etaBin%i", ipt, ieta);
      sprintf(htit,  "eHCAL3x3MIP/trkP(Xtal>2.0#sigma) (%3.2f<|#eta|<%3.2f), (%2.0f<trkP<%3.0f)", lowEta, highEta, lowP, highP );
      h_eHCAL3x3MIP_Frac_20Sig[ipt][ieta] = new TH1F(hname, htit, 1500, -1.0, 4.0);
      h_eHCAL3x3MIP_Frac_20Sig[ipt][ieta] ->Sumw2();
      sprintf(hname, "h_eHCAL5x5FracMIP20Sig_ptBin%i_etaBin%i", ipt, ieta);
      sprintf(htit,  "eHCAL5x5MIP/trkP(Xtal>2.0#sigma) (%3.2f<|#eta|<%3.2f), (%2.0f<trkP<%3.0f)", lowEta, highEta, lowP, highP );
      h_eHCAL5x5MIP_Frac_20Sig[ipt][ieta] = new TH1F(hname, htit, 1500, -1.0, 4.0);
      h_eHCAL5x5MIP_Frac_20Sig[ipt][ieta] ->Sumw2();
      sprintf(hname, "h_eHCAL7x7FracMIP20Sig_ptBin%i_etaBin%i", ipt, ieta);
      sprintf(htit,  "eHCAL7x7MIP/trkP(Xtal>2.0#sigma) (%3.2f<|#eta|<%3.2f), (%2.0f<trkP<%3.0f)", lowEta, highEta, lowP, highP );
      h_eHCAL7x7MIP_Frac_20Sig[ipt][ieta] = new TH1F(hname, htit, 1500, -1.0, 4.0);
      h_eHCAL7x7MIP_Frac_20Sig[ipt][ieta] ->Sumw2();

      //=================== Ecal+Hcal Response ====================
      d_response->cd();
      sprintf(hname, "h_Response_eHCAL3x3_eECAL11x11_ptBin%i_etaBin%i", ipt, ieta);
      sprintf(htit,  "(eHCAL3x3+eECAL11x11)/trkP (%3.2f<|#eta|<%3.2f), (%2.0f<trkP<%3.0f)", lowEta, highEta, lowP, highP );
      h_eHCAL3x3_eECAL11x11_response[ipt][ieta] = new TH1F(hname, htit, 1500, -1.0, 4.0);
      h_eHCAL3x3_eECAL11x11_response[ipt][ieta] ->Sumw2();
      sprintf(hname, "h_Response_eHCAL5x5_eECAL11x11_ptBin%i_etaBin%i", ipt, ieta);
      sprintf(htit,  "(eHCAL5x5+eECAL11x11)/trkP (%3.2f<|#eta|<%3.2f), (%2.0f<trkP<%3.0f)", lowEta, highEta, lowP, highP );
      h_eHCAL5x5_eECAL11x11_response[ipt][ieta] = new TH1F(hname, htit, 1500, -1.0, 4.0);
      h_eHCAL5x5_eECAL11x11_response[ipt][ieta] ->Sumw2();
      sprintf(hname, "h_Response_eHCAL7x7_eECAL11x11_ptBin%i_etaBin%i", ipt, ieta);
      sprintf(htit,  "(eHCAL7x7+eECAL11x11)/trkP (%3.2f<|#eta|<%3.2f), (%2.0f<trkP<%3.0f)", lowEta, highEta, lowP, highP );
      h_eHCAL7x7_eECAL11x11_response[ipt][ieta] = new TH1F(hname, htit, 1500, -1.0, 4.0);
      h_eHCAL7x7_eECAL11x11_response[ipt][ieta] ->Sumw2();
      
      sprintf(hname, "h_ResponseMIP_eHCAL3x3_eECAL11x11_ptBin%i_etaBin%i", ipt, ieta);
      sprintf(htit,  "(eHCAL3x3MIP+eECAL11x11)/trkP (%3.2f<|#eta|<%3.2f), (%2.0f<trkP<%3.0f)", lowEta, highEta, lowP, highP );
      h_eHCAL3x3_eECAL11x11_responseMIP[ipt][ieta] = new TH1F(hname, htit, 1500, -1.0, 4.0);
      h_eHCAL3x3_eECAL11x11_responseMIP[ipt][ieta] ->Sumw2();
      sprintf(hname, "h_ResponseMIP_eHCAL5x5_eECAL11x11_ptBin%i_etaBin%i", ipt, ieta);
      sprintf(htit,  "(eHCAL5x5MIP+eECAL11x11)/trkP (%3.2f<|#eta|<%3.2f), (%2.0f<trkP<%3.0f)", lowEta, highEta, lowP, highP );
      h_eHCAL5x5_eECAL11x11_responseMIP[ipt][ieta] = new TH1F(hname, htit, 1500, -1.0, 4.0);
      h_eHCAL5x5_eECAL11x11_responseMIP[ipt][ieta] ->Sumw2();
      sprintf(hname, "h_ResponseMIP_eHCAL7x7_eECAL11x11_ptBin%i_etaBin%i", ipt, ieta);
      sprintf(htit,  "(eHCAL7x7MIP+eECAL11x11)/trkP (%3.2f<|#eta|<%3.2f), (%2.0f<trkP<%3.0f)", lowEta, highEta, lowP, highP );
      h_eHCAL7x7_eECAL11x11_responseMIP[ipt][ieta] = new TH1F(hname, htit, 1500, -1.0, 4.0);
      h_eHCAL7x7_eECAL11x11_responseMIP[ipt][ieta] ->Sumw2();
      
      sprintf(hname, "h_ResponseInteract_eHCAL3x3_eECAL11x11_ptBin%i_etaBin%i", ipt, ieta);
      sprintf(htit,  "(eHCAL3x3Interact+eECAL11x11)/trkP (%3.2f<|#eta|<%3.2f), (%2.0f<trkP<%3.0f)", lowEta, highEta, lowP, highP );
      h_eHCAL3x3_eECAL11x11_responseInteract[ipt][ieta] = new TH1F(hname, htit, 1500, -1.0, 4.0);
      h_eHCAL3x3_eECAL11x11_responseInteract[ipt][ieta] ->Sumw2();
      sprintf(hname, "h_ResponseInteract_eHCAL5x5_eECAL11x11_ptBin%i_etaBin%i", ipt, ieta);
      sprintf(htit,  "(eHCAL5x5Interact+eECAL11x11)/trkP (%3.2f<|#eta|<%3.2f), (%2.0f<trkP<%3.0f)", lowEta, highEta, lowP, highP );
      h_eHCAL5x5_eECAL11x11_responseInteract[ipt][ieta] = new TH1F(hname, htit, 1500, -1.0, 4.0);
      h_eHCAL5x5_eECAL11x11_responseInteract[ipt][ieta] ->Sumw2();
      sprintf(hname, "h_ResponseInteract_eHCAL7x7_eECAL11x11_ptBin%i_etaBin%i", ipt, ieta);
      sprintf(htit,  "(eHCAL7x7Interact+eECAL11x11)/trkP (%3.2f<|#eta|<%3.2f), (%2.0f<trkP<%3.0f)", lowEta, highEta, lowP, highP );
      h_eHCAL7x7_eECAL11x11_responseInteract[ipt][ieta] = new TH1F(hname, htit, 1500, -1.0, 4.0);
      h_eHCAL7x7_eECAL11x11_responseInteract[ipt][ieta] ->Sumw2();
      
      sprintf(hname, "h_Response_eHCAL3x3_eECAL9x9_ptBin%i_etaBin%i", ipt, ieta);
      sprintf(htit,  "(eHCAL3x3+eECAL9x9)/trkP (%3.2f<|#eta|<%3.2f), (%2.0f<trkP<%3.0f)", lowEta, highEta, lowP, highP );
      h_eHCAL3x3_eECAL9x9_response[ipt][ieta] = new TH1F(hname, htit, 1500, -1.0, 4.0);
      h_eHCAL3x3_eECAL9x9_response[ipt][ieta] ->Sumw2();
      sprintf(hname, "h_Response_eHCAL5x5_eECAL9x9_ptBin%i_etaBin%i", ipt, ieta);
      sprintf(htit,  "(eHCAL5x5+eECAL9x9)/trkP (%3.2f<|#eta|<%3.2f), (%2.0f<trkP<%3.0f)", lowEta, highEta, lowP, highP );
      h_eHCAL5x5_eECAL9x9_response[ipt][ieta] = new TH1F(hname, htit, 1500, -1.0, 4.0);
      h_eHCAL5x5_eECAL9x9_response[ipt][ieta] ->Sumw2();
      sprintf(hname, "h_Response_eHCAL7x7_eECAL9x9_ptBin%i_etaBin%i", ipt, ieta);
      sprintf(htit,  "(eHCAL7x7+eECAL9x9)/trkP (%3.2f<|#eta|<%3.2f), (%2.0f<trkP<%3.0f)", lowEta, highEta, lowP, highP );
      h_eHCAL7x7_eECAL9x9_response[ipt][ieta] = new TH1F(hname, htit, 1500, -1.0, 4.0);
      h_eHCAL7x7_eECAL9x9_response[ipt][ieta] ->Sumw2();
      
      sprintf(hname, "h_ResponseMIP_eHCAL3x3_eECAL9x9_ptBin%i_etaBin%i", ipt, ieta);
      sprintf(htit,  "(eHCAL3x3MIP+eECAL9x9)/trkP (%3.2f<|#eta|<%3.2f), (%2.0f<trkP<%3.0f)", lowEta, highEta, lowP, highP );
      h_eHCAL3x3_eECAL9x9_responseMIP[ipt][ieta] = new TH1F(hname, htit, 1500, -1.0, 4.0);
      h_eHCAL3x3_eECAL9x9_responseMIP[ipt][ieta] ->Sumw2();
      sprintf(hname, "h_ResponseMIP_eHCAL5x5_eECAL9x9_ptBin%i_etaBin%i", ipt, ieta);
      sprintf(htit,  "(eHCAL5x5MIP+eECAL9x9)/trkP (%3.2f<|#eta|<%3.2f), (%2.0f<trkP<%3.0f)", lowEta, highEta, lowP, highP );
      h_eHCAL5x5_eECAL9x9_responseMIP[ipt][ieta] = new TH1F(hname, htit, 1500, -1.0, 4.0);
      h_eHCAL5x5_eECAL9x9_responseMIP[ipt][ieta] ->Sumw2();
      sprintf(hname, "h_ResponseMIP_eHCAL7x7_eECAL9x9_ptBin%i_etaBin%i", ipt, ieta);
      sprintf(htit,  "(eHCAL7x7MIP+eECAL9x9)/trkP (%3.2f<|#eta|<%3.2f), (%2.0f<trkP<%3.0f)", lowEta, highEta, lowP, highP );
      h_eHCAL7x7_eECAL9x9_responseMIP[ipt][ieta] = new TH1F(hname, htit, 1500, -1.0, 4.0);
      h_eHCAL7x7_eECAL9x9_responseMIP[ipt][ieta] ->Sumw2();
      
      sprintf(hname, "h_ResponseInteract_eHCAL3x3_eECAL9x9_ptBin%i_etaBin%i", ipt, ieta);
      sprintf(htit,  "(eHCAL3x3Interact+eECAL9x9)/trkP (%3.2f<|#eta|<%3.2f), (%2.0f<trkP<%3.0f)", lowEta, highEta, lowP, highP );
      h_eHCAL3x3_eECAL9x9_responseInteract[ipt][ieta] = new TH1F(hname, htit, 1500, -1.0, 4.0);
      h_eHCAL3x3_eECAL9x9_responseInteract[ipt][ieta] ->Sumw2();
      sprintf(hname, "h_ResponseInteract_eHCAL5x5_eECAL9x9_ptBin%i_etaBin%i", ipt, ieta);
      sprintf(htit,  "(eHCAL5x5Interact+eECAL9x9)/trkP (%3.2f<|#eta|<%3.2f), (%2.0f<trkP<%3.0f)", lowEta, highEta, lowP, highP );
      h_eHCAL5x5_eECAL9x9_responseInteract[ipt][ieta] = new TH1F(hname, htit, 1500, -1.0, 4.0);
      h_eHCAL5x5_eECAL9x9_responseInteract[ipt][ieta] ->Sumw2();
      sprintf(hname, "h_ResponseInteract_eHCAL7x7_eECAL9x9_ptBin%i_etaBin%i", ipt, ieta);
      sprintf(htit,  "(eHCAL7x7Interact+eECAL9x9)/trkP (%3.2f<|#eta|<%3.2f), (%2.0f<trkP<%3.0f)", lowEta, highEta, lowP, highP );
      h_eHCAL7x7_eECAL9x9_responseInteract[ipt][ieta] = new TH1F(hname, htit, 1500, -1.0, 4.0);
      h_eHCAL7x7_eECAL9x9_responseInteract[ipt][ieta] ->Sumw2();


      sprintf(hname, "h_Response_eHCAL3x3_eECAL7x7_ptBin%i_etaBin%i", ipt, ieta);
      sprintf(htit,  "(eHCAL3x3+eECAL7x7)/trkP (%3.2f<|#eta|<%3.2f), (%2.0f<trkP<%3.0f)", lowEta, highEta, lowP, highP );
      h_eHCAL3x3_eECAL7x7_response[ipt][ieta] = new TH1F(hname, htit, 1500, -1.0, 4.0);
      h_eHCAL3x3_eECAL7x7_response[ipt][ieta] ->Sumw2();
      sprintf(hname, "h_Response_eHCAL5x5_eECAL7x7_ptBin%i_etaBin%i", ipt, ieta);
      sprintf(htit,  "(eHCAL5x5+eECAL7x7)/trkP (%3.2f<|#eta|<%3.2f), (%2.0f<trkP<%3.0f)", lowEta, highEta, lowP, highP );
      h_eHCAL5x5_eECAL7x7_response[ipt][ieta] = new TH1F(hname, htit, 1500, -1.0, 4.0);
      h_eHCAL5x5_eECAL7x7_response[ipt][ieta] ->Sumw2();
      sprintf(hname, "h_Response_eHCAL7x7_eECAL7x7_ptBin%i_etaBin%i", ipt, ieta);
      sprintf(htit,  "(eHCAL7x7+eECAL7x7)/trkP (%3.2f<|#eta|<%3.2f), (%2.0f<trkP<%3.0f)", lowEta, highEta, lowP, highP );
      h_eHCAL7x7_eECAL7x7_response[ipt][ieta] = new TH1F(hname, htit, 1500, -1.0, 4.0);
      h_eHCAL7x7_eECAL7x7_response[ipt][ieta] ->Sumw2();

      sprintf(hname, "h_ResponseMIP_eHCAL3x3_eECAL7x7_ptBin%i_etaBin%i", ipt, ieta);
      sprintf(htit,  "(eHCAL3x3MIP+eECAL7x7)/trkP (%3.2f<|#eta|<%3.2f), (%2.0f<trkP<%3.0f)", lowEta, highEta, lowP, highP );
      h_eHCAL3x3_eECAL7x7_responseMIP[ipt][ieta] = new TH1F(hname, htit, 1500, -1.0, 4.0);
      h_eHCAL3x3_eECAL7x7_responseMIP[ipt][ieta] ->Sumw2();
      sprintf(hname, "h_ResponseMIP_eHCAL5x5_eECAL7x7_ptBin%i_etaBin%i", ipt, ieta);
      sprintf(htit,  "(eHCAL5x5MIP+eECAL7x7)/trkP (%3.2f<|#eta|<%3.2f), (%2.0f<trkP<%3.0f)", lowEta, highEta, lowP, highP );
      h_eHCAL5x5_eECAL7x7_responseMIP[ipt][ieta] = new TH1F(hname, htit, 1500, -1.0, 4.0);
      h_eHCAL5x5_eECAL7x7_responseMIP[ipt][ieta] ->Sumw2();
      sprintf(hname, "h_ResponseMIP_eHCAL7x7_eECAL7x7_ptBin%i_etaBin%i", ipt, ieta);
      sprintf(htit,  "(eHCAL7x7MIP+eECAL7x7)/trkP (%3.2f<|#eta|<%3.2f), (%2.0f<trkP<%3.0f)", lowEta, highEta, lowP, highP );
      h_eHCAL7x7_eECAL7x7_responseMIP[ipt][ieta] = new TH1F(hname, htit, 1500, -1.0, 4.0);
      h_eHCAL7x7_eECAL7x7_responseMIP[ipt][ieta] ->Sumw2();
      
      sprintf(hname, "h_ResponseInteract_eHCAL3x3_eECAL7x7_ptBin%i_etaBin%i", ipt, ieta);
      sprintf(htit,  "(eHCAL3x3Interact+eECAL7x7)/trkP (%3.2f<|#eta|<%3.2f), (%2.0f<trkP<%3.0f)", lowEta, highEta, lowP, highP );
      h_eHCAL3x3_eECAL7x7_responseInteract[ipt][ieta] = new TH1F(hname, htit, 1500, -1.0, 4.0);
      h_eHCAL3x3_eECAL7x7_responseInteract[ipt][ieta] ->Sumw2();
      sprintf(hname, "h_ResponseInteract_eHCAL5x5_eECAL7x7_ptBin%i_etaBin%i", ipt, ieta);
      sprintf(htit,  "(eHCAL5x5Interact+eECAL7x7)/trkP (%3.2f<|#eta|<%3.2f), (%2.0f<trkP<%3.0f)", lowEta, highEta, lowP, highP );
      h_eHCAL5x5_eECAL7x7_responseInteract[ipt][ieta] = new TH1F(hname, htit, 1500, -1.0, 4.0);
      h_eHCAL5x5_eECAL7x7_responseInteract[ipt][ieta] ->Sumw2();
      sprintf(hname, "h_ResponseInteract_eHCAL7x7_eECAL7x7_ptBin%i_etaBin%i", ipt, ieta);
      sprintf(htit,  "(eHCAL7x7Interact+eECAL7x7)/trkP (%3.2f<|#eta|<%3.2f), (%2.0f<trkP<%3.0f)", lowEta, highEta, lowP, highP );
      h_eHCAL7x7_eECAL7x7_responseInteract[ipt][ieta] = new TH1F(hname, htit, 1500, -1.0, 4.0);
      h_eHCAL7x7_eECAL7x7_responseInteract[ipt][ieta] ->Sumw2();
      
      d_response20Sig->cd();
      sprintf(hname, "h_Response20Sig_eHCAL3x3_eECAL11x11_ptBin%i_etaBin%i", ipt, ieta);
      sprintf(htit,  "(eHCAL3x3+eECAL11x11)(Xtal>2.0#sigma)/trkP (%3.2f<|#eta|<%3.2f), (%2.0f<trkP<%3.0f)", lowEta, highEta, lowP, highP );
      h_eHCAL3x3_eECAL11x11_response_20Sig[ipt][ieta] = new TH1F(hname, htit, 1500, -1.0, 4.0);
      h_eHCAL3x3_eECAL11x11_response_20Sig[ipt][ieta] ->Sumw2();
      sprintf(hname, "h_ResponseMIP20Sig_eHCAL3x3_eECAL11x11_ptBin%i_etaBin%i", ipt, ieta);
      sprintf(htit,  "(eHCAL3x3MIP+eECAL11x11)/trkP(Xtal>2.0#sigma) (%3.2f<|#eta|<%3.2f), (%2.0f<trkP<%3.0f)", lowEta, highEta, lowP, highP );
      h_eHCAL3x3_eECAL11x11_responseMIP_20Sig[ipt][ieta] = new TH1F(hname, htit, 1500, -1.0, 4.0);
      h_eHCAL3x3_eECAL11x11_responseMIP_20Sig[ipt][ieta] ->Sumw2();
      sprintf(hname, "h_ResponseInteract20Sig_eHCAL3x3_eECAL11x11_ptBin%i_etaBin%i", ipt, ieta);
      sprintf(htit,  "(eHCAL3x3Interact+eECAL11x11)/trkP(Xtal>2.0#sigma) (%3.2f<|#eta|<%3.2f), (%2.0f<trkP<%3.0f)", lowEta, highEta, lowP, highP );
      h_eHCAL3x3_eECAL11x11_responseInteract_20Sig[ipt][ieta] = new TH1F(hname, htit, 1500, -1.0, 4.0);
      h_eHCAL3x3_eECAL11x11_responseInteract_20Sig[ipt][ieta] ->Sumw2();
      sprintf(hname, "h_Response20Sig_eHCAL5x5_eECAL11x11_ptBin%i_etaBin%i", ipt, ieta);
      sprintf(htit,  "(eHCAL5x5+eECAL11x11)(Xtal>2.0#sigma)/trkP (%3.2f<|#eta|<%3.2f), (%2.0f<trkP<%3.0f)", lowEta, highEta, lowP, highP );
      h_eHCAL5x5_eECAL11x11_response_20Sig[ipt][ieta] = new TH1F(hname, htit, 1500, -1.0, 4.0);
      h_eHCAL5x5_eECAL11x11_response_20Sig[ipt][ieta] ->Sumw2();
      sprintf(hname, "h_ResponseMIP20Sig_eHCAL5x5_eECAL11x11_ptBin%i_etaBin%i", ipt, ieta);
      sprintf(htit,  "(eHCAL5x5MIP+eECAL11x11)/trkP(Xtal>2.0#sigma) (%3.2f<|#eta|<%3.2f), (%2.0f<trkP<%3.0f)", lowEta, highEta, lowP, highP );
      h_eHCAL5x5_eECAL11x11_responseMIP_20Sig[ipt][ieta] = new TH1F(hname, htit, 1500, -1.0, 4.0);
      h_eHCAL5x5_eECAL11x11_responseMIP_20Sig[ipt][ieta] ->Sumw2();
      sprintf(hname, "h_ResponseInteract20Sig_eHCAL5x5_eECAL11x11_ptBin%i_etaBin%i", ipt, ieta);
      sprintf(htit,  "(eHCAL5x5Interact+eECAL11x11)/trkP(Xtal>2.0#sigma) (%3.2f<|#eta|<%3.2f), (%2.0f<trkP<%3.0f)", lowEta, highEta, lowP, highP );
      h_eHCAL5x5_eECAL11x11_responseInteract_20Sig[ipt][ieta] = new TH1F(hname, htit, 1500, -1.0, 4.0);
      h_eHCAL5x5_eECAL11x11_responseInteract_20Sig[ipt][ieta] ->Sumw2();
      
      sprintf(hname, "h_Response20Sig_eHCAL3x3_eECAL9x9_ptBin%i_etaBin%i", ipt, ieta);
      sprintf(htit,  "(eHCAL3x3+eECAL9x9)(Xtal>2.0#sigma)/trkP (%3.2f<|#eta|<%3.2f), (%2.0f<trkP<%3.0f)", lowEta, highEta, lowP, highP );
      h_eHCAL3x3_eECAL9x9_response_20Sig[ipt][ieta] = new TH1F(hname, htit, 1500, -1.0, 4.0);
      h_eHCAL3x3_eECAL9x9_response_20Sig[ipt][ieta] ->Sumw2();
      sprintf(hname, "h_ResponseMIP20Sig_eHCAL3x3_eECAL9x9_ptBin%i_etaBin%i", ipt, ieta);
      sprintf(htit,  "(eHCAL3x3MIP+eECAL9x9)/trkP(Xtal>2.0#sigma) (%3.2f<|#eta|<%3.2f), (%2.0f<trkP<%3.0f)", lowEta, highEta, lowP, highP );
      h_eHCAL3x3_eECAL9x9_responseMIP_20Sig[ipt][ieta] = new TH1F(hname, htit, 1500, -1.0, 4.0);
      h_eHCAL3x3_eECAL9x9_responseMIP_20Sig[ipt][ieta] ->Sumw2();
      sprintf(hname, "h_ResponseInteract20Sig_eHCAL3x3_eECAL9x9_ptBin%i_etaBin%i", ipt, ieta);
      sprintf(htit,  "(eHCAL3x3Interact+eECAL9x9)/trkP(Xtal>2.0#sigma) (%3.2f<|#eta|<%3.2f), (%2.0f<trkP<%3.0f)", lowEta, highEta, lowP, highP );
      h_eHCAL3x3_eECAL9x9_responseInteract_20Sig[ipt][ieta] = new TH1F(hname, htit, 1500, -1.0, 4.0);
      h_eHCAL3x3_eECAL9x9_responseInteract_20Sig[ipt][ieta] ->Sumw2();

      sprintf(hname, "h_Response20Sig_eHCAL3x3_eECAL7x7_ptBin%i_etaBin%i", ipt, ieta);
      sprintf(htit,  "(eHCAL3x3+eECAL7x7)(Xtal>2.0#sigma)/trkP (%3.2f<|#eta|<%3.2f), (%2.0f<trkP<%3.0f)", lowEta, highEta, lowP, highP );
      h_eHCAL3x3_eECAL7x7_response_20Sig[ipt][ieta] = new TH1F(hname, htit, 1500, -1.0, 4.0);
      h_eHCAL3x3_eECAL7x7_response_20Sig[ipt][ieta] ->Sumw2();
      sprintf(hname, "h_ResponseMIP20Sig_eHCAL3x3_eECAL7x7_ptBin%i_etaBin%i", ipt, ieta);
      sprintf(htit,  "(eHCAL3x3MIP+eECAL7x7)/trkP(Xtal>2.0#sigma) (%3.2f<|#eta|<%3.2f), (%2.0f<trkP<%3.0f)", lowEta, highEta, lowP, highP );
      h_eHCAL3x3_eECAL7x7_responseMIP_20Sig[ipt][ieta] = new TH1F(hname, htit, 1500, -1.0, 4.0);
      h_eHCAL3x3_eECAL7x7_responseMIP_20Sig[ipt][ieta] ->Sumw2();
      sprintf(hname, "h_ResponseInteract20Sig_eHCAL3x3_eECAL7x7_ptBin%i_etaBin%i", ipt, ieta);
      sprintf(htit,  "(eHCAL3x3Interact+eECAL7x7)/trkP(Xtal>2.0#sigma) (%3.2f<|#eta|<%3.2f), (%2.0f<trkP<%3.0f)", lowEta, highEta, lowP, highP );
      h_eHCAL3x3_eECAL7x7_responseInteract_20Sig[ipt][ieta] = new TH1F(hname, htit, 1500, -1.0, 4.0);
      h_eHCAL3x3_eECAL7x7_responseInteract_20Sig[ipt][ieta] ->Sumw2();
      
      d_maxNearP->cd();
      sprintf(hname, "h_meanTrackP_ptBin%i_etaBin%i", ipt, ieta);
      sprintf(htit,  "Track Momentum (%3.2f<|#eta|<%3.2f), (%2.0f<trkP<%3.0f)", lowEta, highEta, lowP, highP );             
      h_meanTrackP[ipt][ieta] = new TH1F(hname, htit, 100, lowP*0.9, highP*1.1);
      h_meanTrackP[ipt][ieta] ->Sumw2();
      
      d_drTrackL1->cd();
      sprintf(hname, "h_drTrackL1_ptBin%i_etaBin%i", ipt, ieta);
      sprintf(htit,  "dr(Track,L1) (%3.2f<|#eta|<%3.2f), (%2.0f<trkP<%3.0f)", lowEta, highEta, lowP, highP );             
      h_drTrackL1[ipt][ieta] = new TH1F(hname, htit, 100, -0.01, 10.0);
      h_drTrackL1[ipt][ieta] ->Sumw2();

    }
  }

  
  fout->cd();
  double hcalEtaBins[55] = {-2.650,-2.500,-2.322,-2.172,-2.043,-1.930,
			    -1.830,-1.740,-1.653,-1.566,-1.479,-1.392,-1.305,
			    -1.218,-1.131,-1.044,-0.957,-0.879,-0.783,-0.696,
			    -0.609,-0.522,-0.435,-0.348,-0.261,-0.174,-0.087,
			    0.000,
			    0.087, 0.174, 0.261, 0.348, 0.435, 0.522, 0.609,
			    0.696, 0.783, 0.879, 0.957, 1.044, 1.131, 1.218,
			    1.305, 1.392, 1.479, 1.566, 1.653, 1.740, 1.830,
			    1.930, 2.043, 2.172, 2.322, 2.500, 2.650};
  h_HcalMeanEneVsEta = new TProfile("h_HcalMeanEneVsEta", "h_HcalMeanEneVsEta", 54, hcalEtaBins);
  h_HcalMeanEneVsEta->Sumw2();

}
