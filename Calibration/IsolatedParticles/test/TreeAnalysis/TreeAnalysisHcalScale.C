#define TreeAnalysisHcalScale_cxx
#include "TreeAnalysisHcalScale.h"
#include <TStyle.h>
#include <TCanvas.h>

Bool_t FillChain(TChain *chain, const char *inputFileList);

int main(Int_t argc, Char_t *argv[]) {
  if( argc<4 ){
    std::cerr << "Please give 4 arguments \n"
              << "runList SeedName" << "\n" 
	      << "outputFileName" << "\n" 
	      << "maximum sample size" << "\n"
	      << "Which sample to do (-1 means all)" << "\n"
              << std::endl;
    return -1;
  }
  
  const char *inputFileList = argv[1];
  const char *outFileName   = argv[2];
  double totalTracks        = atof(argv[3]);
  int    sample             = atoi(argv[4]);
  if (sample < 0 || sample > 5) sample = -1;

  int    cut = 0;

  std::cout << "runList SeeName        " << inputFileList << "\n"
	    << "outputFileName         " << outFileName << "\n"
	    << "total number of tracks " << totalTracks 
	    << " Sample "                << sample << std::endl;

  // Reading Tree                                                        
  std::cout << "---------------------" << std::endl;
  std::cout << "Reading List of input trees from " << inputFileList << std::endl;
  std::string particles[6] = {"pi+","pi-","K+", "K-", "p+", "p-"};
  double evFrac[6]         = {0.695, 0.695, 0.163, 0.163, 0.142, 0.142};
  std::vector<std::string> particleNames;
  std::vector<double>      fraction;
  if (sample < 0) {
    for (unsigned int i=0; i<6; i++) {
      particleNames.push_back(particles[i]);
      fraction.push_back(evFrac[i]);
    }
  } else {
    particleNames.push_back(particles[sample]);
    fraction.push_back(1.0);
  }
  
  TreeAnalysisHcalScale tree(outFileName, particleNames);

  for (unsigned int i=0; i<particleNames.size(); i++) {
    char fileList[200], treeName[200];
    sprintf (fileList, "%s%s.txt", inputFileList, particles[i].c_str());
    //    sprintf (treeName, "/IsolatedTracksHcalScale/tree");
    sprintf (treeName, "/isolatedTracksHcal/tree");
    TChain *chain = new TChain(treeName);
    std::cout << "try to create a chain for " << fileList << std::endl;
    if  (! FillChain(chain, fileList) ) {
      std::cerr << "Cannot get the tree " << std::endl;
    } else {
      unsigned int nmax = (unsigned int)(fraction[i]*totalTracks);
      tree.Init(chain);
      tree.setParticle(i, nmax);
      tree.Loop(cut);
      tree.weights[i]= (double)nmax/(double)tree.nIsoTrkTotal;
      std::cout << "particle " << i << " nmax " << nmax << " nisotrktotal  " << tree.nIsoTrkTotal << " weight " << tree.weights[i] << std::endl;
      tree.clear();
      
    }
  }
  tree.AddWeight(particleNames);
  
  return 0;
}

Bool_t FillChain(TChain *chain, const char *inputFileList) {

  ifstream infile(inputFileList);
  std::string buffer;

  if(!infile.is_open()) {
    std::cerr << "** ERROR: Can't open '" << inputFileList << "' for input" << std::endl;
    return kFALSE;
  }
  
  //  std::cout << "TreeUtilities : FillChain " << std::endl;
  while(1) {
    infile >> buffer;
    if(!infile.good()) break;

    //    std::cout << "Adding tree from " << buffer.c_str() << std::endl;
    chain->Add(buffer.c_str());
  }
  std::cout << "No. of Entries in this tree : " << chain->GetEntries() << std::endl;
  return kTRUE;
}

TreeAnalysisHcalScale::TreeAnalysisHcalScale(const char *outFileName, std::vector<std::string>& particles) {
  
  ipBin = nmaxBin = 0;
  double tempgen_Eta[NPBins+1] = {20.0, 30.0, 40.0, 60.0, 100.0, 1000.0};
  
  for(int i=0; i<NPBins+1; i++)  genPartPBins[i]  = tempgen_Eta[i];
  BookHistograms(outFileName, particles);
}

TreeAnalysisHcalScale::~TreeAnalysisHcalScale() {
  fout->cd();
  fout->Write();
  fout->Close();   
}

Int_t TreeAnalysisHcalScale::Cut(Long64_t entry) {
  // This function may be called from Loop.
  // returns  1 if entry is accepted.
  // returns -1 otherwise.
  return 1;
}

Int_t TreeAnalysisHcalScale::GetEntry(Long64_t entry) {

  // Read contents of entry.
  if (!fChain) return 0;
  return fChain->GetEntry(entry);
}

Long64_t TreeAnalysisHcalScale::LoadTree(Long64_t entry) {

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

void TreeAnalysisHcalScale::Init(TChain *tree) {

  // The Init() function is called when the selector needs to initialize
  // a new tree or chain. Typically here the branch addresses and branch
  // pointers of the tree will be set.
  // It is normally not necessary to make changes to the generated
  // code, but the routine can be extended by the user if needed.
  // Init() will be called many times when running on PROOF
  // (once per file to be processed).
  // Set object pointer
  t_trackP        = 0;
  t_trackPt       = 0;
  t_trackEta      = 0;
  t_trackPhi      = 0;
  t_trackHcalEta  = 0;
  t_trackHcalPhi  = 0;
  t_hCone         = 0;
  t_conehmaxNearP = 0;
  t_eMipDR        = 0;
  t_eMipDR_2        = 0;
  t_eECALDR       = 0;
  t_eECALDR_2       = 0;
  t_eHCALDR       = 0;
  t_e11x11_20Sig  = 0;
  t_e15x15_20Sig  = 0;

  // Set branch addresses and branch pointers
  if (!tree) return;
  fChain = tree;
  fCurrent = -1;
  fChain->SetMakeClass(1);

  //   fChain->SetBranchAddress("t_EvtNo", &t_EvtNo, &b_t_EvtNo);
  fChain->SetBranchAddress("t_RunNo",    &t_RunNo,    &b_t_RunNo);
  fChain->SetBranchAddress("t_Lumi",     &t_Lumi,     &b_t_Lumi);
  fChain->SetBranchAddress("t_Bunch",    &t_Bunch,    &b_t_Bunch);
  fChain->SetBranchAddress("t_trackP",   &t_trackP,   &b_t_trackP);
  fChain->SetBranchAddress("t_trackPt",  &t_trackPt,  &b_t_trackPt);
  fChain->SetBranchAddress("t_trackEta", &t_trackEta, &b_t_trackEta);
  fChain->SetBranchAddress("t_trackPhi", &t_trackPhi, &b_t_trackPhi);
  fChain->SetBranchAddress("t_trackHcalEta", &t_trackHcalEta, &b_t_trackHcalEta);
  fChain->SetBranchAddress("t_trackHcalPhi", &t_trackHcalPhi, &b_t_trackHcalPhi);
  fChain->SetBranchAddress("t_hCone",    &t_hCone,     &b_t_hCone);
  fChain->SetBranchAddress("t_conehmaxNearP", &t_conehmaxNearP, &b_t_conehmaxNearP);
  fChain->SetBranchAddress("t_eMipDR",   &t_eMipDR,    &b_t_eMipDR);
  fChain->SetBranchAddress("t_eMipDR_2",   &t_eMipDR_2,    &b_t_eMipDR_2);
  fChain->SetBranchAddress("t_eECALDR",  &t_eECALDR,   &b_t_eECALDR);
  fChain->SetBranchAddress("t_eECALDR_2",  &t_eECALDR_2,   &b_t_eECALDR_2);
  fChain->SetBranchAddress("t_eHCALDR",  &t_eHCALDR,   &b_t_eHCALDR);
  fChain->SetBranchAddress("t_e11x11_20Sig",  &t_e11x11_20Sig,   &b_t_e11x11_20Sig);
  fChain->SetBranchAddress("t_e15x15_20Sig",  &t_e15x15_20Sig,   &b_t_e15x15_20Sig);
  Notify();
}

void TreeAnalysisHcalScale::Loop(int cut) {

  if (fChain == 0) return;

  Long64_t nentries = fChain->GetEntriesFast();
  std::cout << "No. of Entries in tree " << nentries << std::endl;

  Long64_t nEventsGoodRuns=0, nEventsValidPV=0, nEventsPVTracks=0;

  Long64_t nbytes = 0, nb = 0;
  std::map<unsigned int, unsigned int> runEvtList, runNTrkList, runNTrkEtaPList,  runNTrkMipList,  runNTrkMipCharIsoList,  runNIsoTrkList;
 nIsoTrkTotal=0;

 int nIsoTrkEtaBin[NEtaBins], nGoodTrkEtaBin[NEtaBins];
 for(int i=0; i<NEtaBins; i++){
   nIsoTrkEtaBin[i]=0;
   nGoodTrkEtaBin[i]=0;
 }
 for (Long64_t jentry=0; jentry<nentries;jentry++) {
   
   // load tree and get current entry
   Long64_t ientry = LoadTree(jentry);
   if (ientry < 0) break;
   nb = fChain->GetEntry(jentry);   nbytes += nb;
   
   if( !(jentry%100000) ) {
     std::cout << "procesing event " << jentry+1 << std::endl;
   }
   
   bool goodRun=false;
   goodRun = true;
   bool evtSel = true;
   if( runEvtList.find(t_RunNo) != runEvtList.end() ) {
     runEvtList[t_RunNo] += 1; 
   } else {
     runEvtList.insert( std::pair<unsigned int, unsigned int>(t_RunNo,1) );
     std::cout << "runNo " << t_RunNo <<" "<<runEvtList[t_RunNo]<<std::endl;
   }
   
   nEventsGoodRuns++;
   
   if( runNTrkList.find(t_RunNo) != runNTrkList.end() ) {
     runNTrkList[t_RunNo] += t_trackP->size(); 
   } else {
     runNTrkList.insert( std::pair<unsigned int, unsigned int>(t_RunNo,t_trackP->size()) );
     std::cout << "runNo " << t_RunNo <<" "<<runNTrkList[t_RunNo]<<std::endl;
   }
   
   unsigned int NIsoTrk=0, NEtaPTrk=0, NMipTrk=0, NMipCharIsoTrk=0;
   for(int itrk=0; itrk<t_trackP->size(); itrk++ ){      
     int iTrkEtaBin=-1, iTrkMomBin=-1;
     if(std::abs((*t_trackHcalEta)[itrk]) < 26)
       iTrkEtaBin = ((*t_trackHcalEta)[itrk] + 25);
     else continue;
     
     for(int ip=0;  ip<NPBins;   ip++)  {
       if( (*t_trackP)[itrk]>genPartPBins[ip] &&  (*t_trackP)[itrk]<genPartPBins[ip+1] )  iTrkMomBin = ip;
     }
     if(iTrkMomBin==2) nGoodTrkEtaBin[iTrkEtaBin]++;
     
     if(iTrkEtaBin>=0  && iTrkMomBin>=0) {	
       NEtaPTrk++;
       h_trackP[ipBin]          ->Fill((*t_trackP)[itrk]        );  
       h_trackPt[ipBin]         ->Fill((*t_trackPt)[itrk]       );  
       h_trackEta[ipBin]        ->Fill((*t_trackEta)[itrk]      );  
       h_trackPhi[ipBin]        ->Fill((*t_trackPhi)[itrk]      );  
       h_trackHcalEta[ipBin]    ->Fill((*t_trackHcalEta)[itrk]  );  
       h_trackHcalPhi[ipBin]    ->Fill((*t_trackHcalPhi)[itrk]  );  
       
       h_hCone[ipBin]           ->Fill( (*t_hCone)[itrk]         );
       h_conehmaxNearP[ipBin]   ->Fill( (*t_conehmaxNearP)[itrk] );
       h_eMipDR[ipBin]          ->Fill( (*t_eMipDR)[itrk]        );
       h_eECALDR[ipBin]         ->Fill( (*t_eECALDR)[itrk]       );
       h_eHCALDR[ipBin]         ->Fill( (*t_eHCALDR)[itrk]       );
       h_e11x11_20Sig[ipBin]    ->Fill( (*t_e11x11_20Sig)[itrk]  );
       h_e15x15_20Sig[ipBin]    ->Fill( (*t_e15x15_20Sig)[itrk]  ); 
       
       bool ChargedIso = true, IfMip = true;
       bool eNeutIso = true, hNeutIso = true;
       
       if ((*t_conehmaxNearP)[itrk] >= 0.0) ChargedIso = false;
       if ((*t_eMipDR)[itrk] >= 1.0)        IfMip      = false;
       
       if (std::abs((*t_trackEta)[itrk])<1.47 && ((*t_e15x15_20Sig)[itrk]-(*t_e11x11_20Sig)[itrk]) > 0.5) eNeutIso = false;
       else if(((*t_e15x15_20Sig)[itrk]-(*t_e11x11_20Sig)[itrk]) > 2.0) eNeutIso = false;
       
       if (((*t_eHCALDR)[itrk]-(*t_hCone)[itrk]) > 3) hNeutIso = false;
       
       if(IfMip){
	 NMipTrk++;
	 if(ChargedIso) NMipCharIsoTrk++;
       }
       
       if(eNeutIso && hNeutIso && ChargedIso && IfMip){
	 h_IsotrackPhi[ipBin]        ->Fill((*t_trackPhi)[itrk]);
	 if(iTrkMomBin==2) nIsoTrkEtaBin[iTrkEtaBin]++;
	 NIsoTrk++;
	 h_IsotrackHcalIEta[ipBin]->Fill((*t_trackHcalEta)[itrk]);
	 
	 h_Response[ipBin+1][iTrkMomBin][iTrkEtaBin]        ->Fill(((*t_hCone)[itrk]+(*t_eMipDR_2)[itrk])/(*t_trackP)[itrk]);
	 h_Response_trunc[ipBin+1][iTrkMomBin][iTrkEtaBin]  ->Fill(((*t_hCone)[itrk]+(*t_eMipDR_2)[itrk])/(*t_trackP)[itrk]);
	 h_Response_E11x11[ipBin+1][iTrkMomBin][iTrkEtaBin]        ->Fill(((*t_hCone)[itrk]+(*t_e11x11_20Sig)[itrk])/(*t_trackP)[itrk]);
	 h_Response_E11x11_trunc[ipBin+1][iTrkMomBin][iTrkEtaBin]  ->Fill(((*t_hCone)[itrk]+(*t_e11x11_20Sig)[itrk])/(*t_trackP)[itrk]);
	 
	 h_eHcalFrac[ipBin+1][iTrkMomBin][iTrkEtaBin]       ->Fill((*t_hCone)[itrk]/(*t_trackP)[itrk]);
	 h_eHcalFrac[ipBin+1][iTrkMomBin][iTrkEtaBin]       ->Fill((*t_hCone)[itrk]/(*t_trackP)[itrk]);
	 h_eHcalFrac_trunc[ipBin+1][iTrkMomBin][iTrkEtaBin] ->Fill((*t_hCone)[itrk]/(*t_trackP)[itrk]);
	 h_hneutIso[ipBin+1][iTrkMomBin][iTrkEtaBin]        ->Fill((*t_eHCALDR)[itrk] - (*t_hCone)[itrk]);
	 h_eneutIso[ipBin+1][iTrkMomBin][iTrkEtaBin]        ->Fill((*t_eECALDR)[itrk] - (*t_eMipDR)[itrk]);
	 h_eneutIsoNxN[ipBin+1][iTrkMomBin][iTrkEtaBin]     ->Fill((*t_e15x15_20Sig)[itrk]-(*t_e11x11_20Sig)[itrk]);
	 nIsoTrkTotal++;
	 if (nIsoTrkTotal <= nmaxBin) {
	   h_Response[0][iTrkMomBin][iTrkEtaBin]            ->Fill(((*t_hCone)[itrk]+(*t_eMipDR_2)[itrk])/(*t_trackP)[itrk]);
	   h_Response_trunc[0][iTrkMomBin][iTrkEtaBin]      ->Fill(((*t_hCone)[itrk]+(*t_eMipDR_2)[itrk])/(*t_trackP)[itrk]);
	   h_Response_E11x11[0][iTrkMomBin][iTrkEtaBin] ->Fill(((*t_hCone)[itrk]+(*t_e11x11_20Sig)[itrk])/(*t_trackP)[itrk]);
	   h_Response_E11x11_trunc[0][iTrkMomBin][iTrkEtaBin]->Fill(((*t_hCone)[itrk]+(*t_e11x11_20Sig)[itrk])/(*t_trackP)[itrk]);
	   
	   h_eHcalFrac[0][iTrkMomBin][iTrkEtaBin]           ->Fill((*t_hCone)[itrk]/(*t_trackP)[itrk]);
	   h_eHcalFrac_trunc[0][iTrkMomBin][iTrkEtaBin]     ->Fill((*t_hCone)[itrk]/(*t_trackP)[itrk]);
	   h_hneutIso[0][iTrkMomBin][iTrkEtaBin]            ->Fill((*t_eHCALDR)[itrk] - (*t_hCone)[itrk]);
	   h_eneutIso[0][iTrkMomBin][iTrkEtaBin]            ->Fill((*t_eECALDR)[itrk] - (*t_eMipDR)[itrk]);
	   h_eneutIsoNxN[0][iTrkMomBin][iTrkEtaBin]         ->Fill((*t_e15x15_20Sig)[itrk]-(*t_e11x11_20Sig)[itrk]);
	 }
       }
     }
   } // loop over tracks in the event 
   if( runNIsoTrkList.find(t_RunNo) != runNIsoTrkList.end() ) {
     runNIsoTrkList[t_RunNo] += NIsoTrk;
   } else {
     runNIsoTrkList.insert( std::pair<unsigned int, unsigned int>(t_RunNo,NIsoTrk) );
     std::cout << "runNo " << t_RunNo <<" "<<runNIsoTrkList[t_RunNo]<<std::endl;
   }
   if( runNTrkEtaPList.find(t_RunNo) != runNTrkEtaPList.end() ) {
     runNTrkEtaPList[t_RunNo] += NEtaPTrk;
   } else {
     runNTrkEtaPList.insert( std::pair<unsigned int, unsigned int>(t_RunNo,NEtaPTrk) );
     std::cout << "runNo " << t_RunNo <<" "<<runNTrkEtaPList[t_RunNo]<<std::endl;
   }
   
    if( runNTrkMipList.find(t_RunNo) != runNTrkMipList.end() ) {
      runNTrkMipList[t_RunNo] += NMipTrk;
    } else {
      runNTrkMipList.insert( std::pair<unsigned int, unsigned int>(t_RunNo,NMipTrk) );
      std::cout << "runNo " << t_RunNo <<" "<<runNTrkMipList[t_RunNo]<<std::endl;
    }
    
    if( runNTrkMipCharIsoList.find(t_RunNo) != runNTrkMipCharIsoList.end() ) {
      runNTrkMipCharIsoList[t_RunNo] += NMipCharIsoTrk;
    } else {
      runNTrkMipCharIsoList.insert( std::pair<unsigned int, unsigned int>(t_RunNo,NMipCharIsoTrk) );
      std::cout << "runNo " << t_RunNo <<" "<<runNTrkMipCharIsoList[t_RunNo]<<std::endl;
    }
          
  
  } //loop over tree entries

  std::cout <<"Number of entries in tree "<< nentries <<" # of isolated tracks "<< nIsoTrkTotal 
	    <<" should be at least " << nmaxBin << std::endl;
  
 
  std::cout << "saved runEvtList " << std::endl;
  std::map<unsigned int, unsigned int>::iterator runEvtListItr = runEvtList.begin();
  for(runEvtListItr=runEvtList.begin(); runEvtListItr != runEvtList.end(); runEvtListItr++) {
    std::cout<<runEvtListItr->first << " "<< runEvtListItr->second << std::endl;
  }

  std::cout << "Number of tracks in runs " << std::endl;
  std::map<unsigned int, unsigned int>::iterator runNTrkListItr = runNTrkList.begin();
  for(runNTrkListItr=runNTrkList.begin(); runNTrkListItr != runNTrkList.end(); runNTrkListItr++) {
    std::cout<<runNTrkListItr->first << " "<< runNTrkListItr->second << std::endl;
  }
  
  std::cout << "Number of tracks in eta range in runs " << std::endl;
  std::map<unsigned int, unsigned int>::iterator runNTrkEtaPListItr = runNTrkEtaPList.begin();
  for(runNTrkEtaPListItr=runNTrkEtaPList.begin(); runNTrkEtaPListItr != runNTrkEtaPList.end(); runNTrkEtaPListItr++) {
    std::cout<<runNTrkEtaPListItr->first << " "<< runNTrkEtaPListItr->second << std::endl;
  }
  
  std::cout << "Number of Mip tracks in runs " << std::endl;
  std::map<unsigned int, unsigned int>::iterator runNTrkMipListItr = runNTrkMipList.begin();
  for(runNTrkMipListItr=runNTrkMipList.begin(); runNTrkMipListItr != runNTrkMipList.end(); runNTrkMipListItr++) {
    std::cout<<runNTrkMipListItr->first << " "<< runNTrkMipListItr->second << std::endl;
   }


  std::cout << "Number of Charged isolated Mip tracks in runs " << std::endl;
  std::map<unsigned int, unsigned int>::iterator runNTrkMipCharIsoListItr = runNTrkMipCharIsoList.begin();
  for(runNTrkMipCharIsoListItr=runNTrkMipCharIsoList.begin(); runNTrkMipCharIsoListItr != runNTrkMipCharIsoList.end(); runNTrkMipCharIsoListItr++) {
    std::cout<<runNTrkMipCharIsoListItr->first << " "<< runNTrkMipCharIsoListItr->second << std::endl;
   }

  std::cout << "Number of isolated tracks in runs " << std::endl;
  std::map<unsigned int, unsigned int>::iterator runNIsoTrkListItr = runNIsoTrkList.begin();
  for(runNIsoTrkListItr=runNIsoTrkList.begin(); runNIsoTrkListItr != runNIsoTrkList.end(); runNIsoTrkListItr++) {
    std::cout<<runNIsoTrkListItr->first << " "<< runNIsoTrkListItr->second << std::endl;
  } 
  for(int i=0; i<NEtaBins;i++){
    std::cout<< "Number of tracks in ieta " << i-25 << ": " <<  nIsoTrkEtaBin[i] << std::endl;
    if(nGoodTrkEtaBin[i]!=0){
      double frac = (double)nIsoTrkEtaBin[i]/(double)nGoodTrkEtaBin[i];
      h_FracIsotrackHcalIEta[ipBin]->Fill(i-25, frac);
    }
  }
}

Bool_t TreeAnalysisHcalScale::Notify() {
  // The Notify() function is called when a new file is opened. This
  // can be either for a new TTree in a TChain or when when a new TTree
  // is started when using PROOF. It is normally not necessary to make changes
  // to the generated code, but the routine can be extended by the
  // user if needed. The return value is currently not used.

  return kTRUE;
}

void TreeAnalysisHcalScale::Show(Long64_t entry) {
  // Print contents of entry.
  // If entry is not specified, print current entry
  if (!fChain) return;
  fChain->Show(entry);
}

void TreeAnalysisHcalScale::BookHistograms(const char *outFileName, std::vector<std::string>& particles) {

  fout = new TFile(outFileName, "RECREATE");

  fout->cd();

  char hname[1000], htit[1000];

  for (unsigned int i=0; i<particles.size(); i++) {
    sprintf (hname, "trackEtaAll%s", particles[i].c_str());
    sprintf (htit,  "Eta:    All Tracks for %s", particles[i].c_str());
    h_trackEtaAll[i]     = new TH1F(hname, htit,    100,  -3.0,   3.0    );
    h_trackEtaAll[i]     ->Sumw2();
    sprintf (hname, "trackP%s", particles[i].c_str());
    sprintf (htit,  "P:    Tracks in eta region for %s", particles[i].c_str());
    h_trackP[i]          = new TH1F(hname, htit,   5000,  0.0,   1000   );
    h_trackP[i]          ->Sumw2();
    sprintf (hname, "trackPt%s", particles[i].c_str());
    sprintf (htit,  "Pt:   Tracks in eta region for %s", particles[i].c_str());
    h_trackPt[i]         = new TH1F(hname, htit,   5000,  0.0,   1000   );
    h_trackPt[i]         ->Sumw2();
    sprintf (hname, "trackEta%s", particles[i].c_str());
    sprintf (htit,  "Eta:  Tracks in eta region for %s", particles[i].c_str());
    h_trackEta[i]        = new TH1F(hname, htit,    100,  -3.0,   3.0    );
    h_trackEta[i]        ->Sumw2();
    sprintf (hname, "trackPhi%s", particles[i].c_str());
    sprintf (htit,  "Phi:  Tracks in eta region for %s", particles[i].c_str());
    h_trackPhi[i]        = new TH1F(hname, htit,    100,  -4.0,   4.0    );
    h_trackPhi[i]       ->Sumw2();
    sprintf (hname, "IsotrackPhi%s", particles[i].c_str());
    sprintf (htit,  "Phi:  Isolated tracks for %s", particles[i].c_str());
    h_IsotrackPhi[i]        = new TH1F(hname, htit,    100,  -4.0,   4.0    );
    h_IsotrackPhi[i]       ->Sumw2();
    sprintf (hname, "trackHcalEta%s", particles[i].c_str());
    sprintf (htit,  "iEta:  Track Hit at Hcal for %s", particles[i].c_str());
    h_trackHcalEta[i]    = new TH1F(hname, htit,    100,  -50.0,  50.0   );
    h_trackHcalEta[i]    ->Sumw2();
    sprintf (hname, "IsotrackHcalIEta%s", particles[i].c_str());
    sprintf (htit,  "iEta:  Isolated Track Hit at Hcal for %s", particles[i].c_str());
    h_IsotrackHcalIEta[i]    = new TH1F(hname, htit,    60,  -30.0,  30.0   );
    h_IsotrackHcalIEta[i]    ->Sumw2();
    sprintf (hname, "FracIsotrackHcalIEta%s", particles[i].c_str());
    sprintf (htit,  "iEta:  Fraction of Isolated Track Hit at Hcal for %s", particles[i].c_str());
    h_FracIsotrackHcalIEta[i]    = new TH1F(hname, htit,    60,  -30.0,  30.0   );
    h_FracIsotrackHcalIEta[i]    ->Sumw2();

    sprintf (hname, "trackHcalPhi%s", particles[i].c_str());
    sprintf (htit,  "iPhi:  Track hit at Hcal for %s", particles[i].c_str());
    h_trackHcalPhi[i]    = new TH1F(hname, htit,     100,   0.0,   100.   );
    h_trackHcalPhi[i]    ->Sumw2();
    sprintf (hname, "h_hcone%s", particles[i].c_str());
    sprintf (htit,  "Energy in Hcal in the cone for %s", particles[i].c_str());
    h_hCone[i]           = new TH1F(hname, htit,    5000, -2.0,   1000.0 );
    h_hCone[i]          ->Sumw2();
    sprintf (hname, "h_conehmaxNearP%s", particles[i].c_str());
    sprintf (htit,  "Max energy in charge isolation for %s", particles[i].c_str());
    h_conehmaxNearP[i]   = new TH1F(hname, htit,    5000, -2.0,   1000.0 );
    h_conehmaxNearP[i]   ->Sumw2();
    sprintf (hname, "h_eMipDR%s", particles[i].c_str());
    sprintf (htit,  "Energy in the MIP region for %s", particles[i].c_str());
    h_eMipDR[i]          = new TH1F(hname, htit,    5000, -2.0,   1000.0 );
    h_eMipDR[i]         ->Sumw2();
    sprintf (hname, "h_eECALDR%s", particles[i].c_str());
    sprintf (htit,  "Energy in ECAL iso region for %s", particles[i].c_str());
    h_eECALDR[i]         = new TH1F(hname, htit,    5000, -2.0,   1000.0 );
    h_eECALDR[i]         ->Sumw2();
    sprintf (hname, "h_eHCALDR%s", particles[i].c_str());
    sprintf (htit,  "Energy in HCAL iso region for %s", particles[i].c_str());
    h_eHCALDR[i]         = new TH1F(hname, htit,    5000, -2.0,   1000.0 );
    h_eHCALDR[i]         ->Sumw2();
    sprintf (hname, "h_e11x11_20Sig%s", particles[i].c_str());
    sprintf (htit,  "Energy in 11x11 for %s", particles[i].c_str());
    h_e11x11_20Sig[i]    = new TH1F(hname ,htit,    5000, -2.0,   1000.0 );
    h_e11x11_20Sig[i]    ->Sumw2();
    sprintf (hname, "h_e15x15_20Sig%s", particles[i].c_str());
    sprintf (htit,  "Energy in 15x15 for %s", particles[i].c_str());
    h_e15x15_20Sig[i]    = new TH1F(hname, htit,    5000, -2.0,   1000.0 );
    h_e15x15_20Sig[i]    ->Sumw2();
  }

  TDirectory *d_HcalFrac     = fout->mkdir( "HcalFrac"    );
  TDirectory *d_eNeutIso     = fout->mkdir( "eNeutIso"    );
  TDirectory *d_hNeutIso     = fout->mkdir( "hNeutIso"    );
  TDirectory *d_eNeutIsoNxN  = fout->mkdir( "eNeutIsoNxN" );
  TDirectory *d_Response     = fout->mkdir( "Response" );

  for (unsigned int i=0; i<particles.size()+1; i++) {
    char part[10];
    if (i == 0) sprintf (part, "all");
    else        sprintf (part, "%s", particles[i-1].c_str());
    for (int ieta=0; ieta<NEtaBins; ieta++) {
      int iEta = ieta;
      for (int ip=0; ip<NPBins; ip++) {
	double lowMom=-5.0, highMom= 5.0;
	lowMom  = genPartPBins[ip];
	highMom = genPartPBins[ip+1];
          
	d_Response->cd();
	sprintf(hname, "h_Response_etaBin%i_pBin%i_particle%s", ieta,ip,part);
	sprintf(htit,  "Response (|i#eta|=%i),(%3.2f<|#p|<%3.2f) for %s", iEta, lowMom, highMom, part);
	h_Response[i][ip][ieta] = new TH1F(hname, htit, 500, -1.0, 4.0);
	h_Response[i][ip][ieta] ->Sumw2();

	sprintf(hname, "h_Response_trunc_etaBin%i_pBin%i_particle%s", ieta,ip,part);
	sprintf(htit,  "Response_trunc (|i#eta|=%i),(%3.2f<|#p|<%3.2f) for %s", iEta, lowMom, highMom, part);
	h_Response_trunc[i][ip][ieta] = new TH1F(hname, htit, 500, 0.2, 4.0);
	h_Response_trunc[i][ip][ieta] ->Sumw2();

	sprintf(hname, "h_Response_E11x11_etaBin%i_pBin%i_particle%s", ieta,ip,part);
	sprintf(htit,  "Response_E11x11 (|i#eta|=%i),(%3.2f<|#p|<%3.2f) for %s", iEta, lowMom, highMom, part);
	h_Response_E11x11[i][ip][ieta] = new TH1F(hname, htit, 500, -1.0, 4.0);
	h_Response_E11x11[i][ip][ieta] ->Sumw2();

	sprintf(hname, "h_Response_E11x11_trunc_etaBin%i_pBin%i_particle%s", ieta,ip,part);
	sprintf(htit,  "Response_E11x11_trunc (|i#eta|=%i),(%3.2f<|#p|<%3.2f) for %s", iEta, lowMom, highMom, part);
	h_Response_E11x11_trunc[i][ip][ieta] = new TH1F(hname, htit, 500, 0.2, 4.0);
	h_Response_E11x11_trunc[i][ip][ieta] ->Sumw2();

	d_HcalFrac->cd();
	sprintf(hname, "h_eHcalFrac_etaBin%i_pBin%i_particle%s", ieta,ip,part);
	sprintf(htit,  "eHcalFrac (|i#eta|=%i),(%3.2f<|#p|<%3.2f) for %s", iEta, lowMom, highMom, part);
	h_eHcalFrac[i][ip][ieta] = new TH1F(hname, htit, 500, -1.0, 4.0);
	h_eHcalFrac[i][ip][ieta] ->Sumw2();

	sprintf(hname, "h_eHcalFrac_trunc_etaBin%i_pBin%i_particle%s", ieta,ip,part);
	sprintf(htit,  "eHcalFrac_trunc (|i#eta|=%i),(%3.2f<|#p|<%3.2f) for %s", iEta, lowMom, highMom, part);
	h_eHcalFrac_trunc[i][ip][ieta] = new TH1F(hname, htit, 500, 0.2, 4.0);
	h_eHcalFrac_trunc[i][ip][ieta] ->Sumw2();
      
	d_hNeutIso->cd();
	sprintf(hname, "h_hneutIso_etaBin%i_pBin%i_particle%s", ieta,ip,part);
	sprintf(htit,  "hneutIso (|i#eta|=%i),(%3.2f<|#p|<%3.2f for %s)", iEta, lowMom, highMom, part);
	h_hneutIso[i][ip][ieta] = new TH1F(hname, htit, 1000, -5.0, 20.0);
	h_hneutIso[i][ip][ieta] ->Sumw2();
      
	d_eNeutIso->cd();
	sprintf(hname, "h_eneutIso_etaBin%i_pBin%i_particle%s", ieta,ip,part);
	sprintf(htit,  "eneutIso (|i#eta|=%i),(%3.2f<|#p|<%3.2f for %s)", iEta, lowMom, highMom, part);
	h_eneutIso[i][ip][ieta] = new TH1F(hname, htit, 1000, -1.0, 25.0);
	h_eneutIso[i][ip][ieta] ->Sumw2();
      
	d_eNeutIsoNxN->cd();
	sprintf(hname, "h_eneutIsoNxN_etaBin%i_pBin%i_particle%s",ieta,ip,part);
	sprintf(htit,  "eneutIsoNxN (|i#eta|=%i),(%3.2f<|#p|<%3.2f for %s)", iEta, lowMom, highMom, part);
	h_eneutIsoNxN[i][ip][ieta] = new TH1F(hname, htit, 1000, -1.0, 10.0);
	h_eneutIsoNxN[i][ip][ieta] ->Sumw2();
      }
    }  
  }
  for (int ieta=0; ieta<NEtaBins; ieta++) {
    int iEta = ieta;
    for (int ip=0; ip<NPBins; ip++) {
      double lowMom=-5.0, highMom= 5.0;
      lowMom  = genPartPBins[ip];
      highMom = genPartPBins[ip+1];
      
      d_HcalFrac->cd();
      sprintf(hname, "h_eHcalFrac_all_etaBin%i_pBin%i", ieta,ip);
      sprintf(htit,  "eHcalFrac allWeighted (|i#eta|=%i),(%3.2f<|#p|<%3.2f)", iEta, lowMom, highMom);
      h_eHcalFrac_all[ip][ieta] = new TH1F(hname, htit, 500, -1.0, 4.0);
      h_eHcalFrac_all[ip][ieta] ->Sumw2();
      
      sprintf(hname, "h_eHcalFrac_trunc_all_etaBin%i_pBin%i", ieta,ip);
      sprintf(htit,  "eHcalFrac_trunc allWeigthed (|i#eta|=%i),(%3.2f<|#p|<%3.2f)", iEta, lowMom, highMom);
      h_eHcalFrac_trunc_all[ip][ieta] = new TH1F(hname, htit, 500, 0.2, 4.0);
      h_eHcalFrac_trunc_all[ip][ieta] ->Sumw2();

      d_Response->cd();
      sprintf(hname, "h_Response_all_etaBin%i_pBin%i", ieta,ip);
      sprintf(htit,  "Response allWeighted (|i#eta|=%i),(%3.2f<|#p|<%3.2f)", iEta, lowMom, highMom);
      h_Response_all[ip][ieta] = new TH1F(hname, htit, 500, -1.0, 4.0);
      h_Response_all[ip][ieta] ->Sumw2();
      
      sprintf(hname, "h_Response_trunc_all_etaBin%i_pBin%i", ieta,ip);
      sprintf(htit,  "Response_trunc allWeigthed (|i#eta|=%i),(%3.2f<|#p|<%3.2f)", iEta, lowMom, highMom);
      h_Response_trunc_all[ip][ieta] = new TH1F(hname, htit, 500, 0.2, 4.0);
      h_Response_trunc_all[ip][ieta] ->Sumw2();

      sprintf(hname, "h_Response_E11x11_all_etaBin%i_pBin%i", ieta,ip);
      sprintf(htit,  "Response_E11x11 allWeighted (|i#eta|=%i),(%3.2f<|#p|<%3.2f)", iEta, lowMom, highMom);
      h_Response_E11x11_all[ip][ieta] = new TH1F(hname, htit, 500, -1.0, 4.0);
      h_Response_E11x11_all[ip][ieta] ->Sumw2();
      
      sprintf(hname, "h_Response_E11x11_trunc_all_etaBin%i_pBin%i", ieta,ip);
      sprintf(htit,  "Response_E11x11_trunc allWeigthed (|i#eta|=%i),(%3.2f<|#p|<%3.2f)", iEta, lowMom, highMom);
      h_Response_E11x11_trunc_all[ip][ieta] = new TH1F(hname, htit, 500, 0.2, 4.0);
      h_Response_E11x11_trunc_all[ip][ieta] ->Sumw2();

    }
  }
  fout->cd();
  
}
			  
void TreeAnalysisHcalScale::clear() {
   if (!fChain) return;
   delete fChain->GetCurrentFile();
}
			  
void TreeAnalysisHcalScale::setParticle(unsigned int ip, unsigned int nmax) {
  ipBin = ip;
  nmaxBin = nmax;
}

void TreeAnalysisHcalScale::AddWeight( std::vector<std::string> particleNames){

 for (unsigned int i=0; i<particleNames.size(); i++) {
   char fileList[200], treeName[200];
   std::cout << "particle " << i << " weight " << weights[i] << std::endl; 
   for (int ieta=0; ieta<NEtaBins; ieta++) {
     for (int ip=0; ip<NPBins; ip++) {
       h_eHcalFrac_all[ip][ieta]->Add(h_eHcalFrac[i+1][ip][ieta], weights[i]);
       h_eHcalFrac_trunc_all[ip][ieta]->Add(h_eHcalFrac_trunc[i+1][ip][ieta], weights[i]);

       h_Response_all[ip][ieta]->Add(h_Response[i+1][ip][ieta], weights[i]);
       h_Response_trunc_all[ip][ieta]->Add(h_Response_trunc[i+1][ip][ieta], weights[i]);

       h_Response_E11x11_all[ip][ieta]->Add(h_Response_E11x11[i+1][ip][ieta], weights[i]);
       h_Response_E11x11_trunc_all[ip][ieta]->Add(h_Response_E11x11_trunc[i+1][ip][ieta], weights[i]);

     }
   }
 }
}

