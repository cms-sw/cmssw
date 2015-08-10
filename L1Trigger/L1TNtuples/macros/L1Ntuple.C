#define l1ntuple_cxx
#include "L1Ntuple.h"


void L1Ntuple::Test()
{ 

  if (fChain == 0)  return;
 
  Long64_t nentries = fChain->GetEntriesFast();
  Long64_t nbytes = 0, nb = 0;
  unsigned int nevents =0;

  std::cout << nentries << " events to process"<<std::endl;
  for (Long64_t jentry=0; jentry<nentries;jentry++)
  {
    Long64_t ientry = LoadTree(jentry);
    if (ientry < 0) break;
    
    nb = fChain->GetEntry(jentry);   nbytes += nb;
    
    //fChain->GetEvent(jentry);
  
    nevents++;
    if (nevents<9)  //eight first events
      { 
    std::cout << "--------------------- Event "<<jentry<<" ---------------------"<<std::endl;

    //event_
    std::cout << "L1Tree         : event_->run = "<<event_->run<<std::endl;

    //gct_
    std::cout << "L1Tree         : gct_->IsoEmEta.Size() = "<<gct_->IsoEmEta.size()<<std::endl;
    if (gct_->IsoEmEta.size()!=0)
      std::cout << "L1Tree         : gct->IsoEmEta[0] = "<<gct_->IsoEmEta[0]<<std::endl; 

    //gmt_
    std::cout << "L1Tree         : gmt_->Ndt        = "<<gmt_->Ndt<<std::endl;
    if (gmt_->Ndt!=0)
      std::cout << "L1Tree         : gmt->Bxdt[0] = "<<gmt_->Bxdt[0]<<std::endl;

    //gt_
    std::cout << "L1Tree         : gt_->tw1.size = "<<gt_->tw1.size()<<std::endl;
    if (gt_->tw1.size()!=0)
      std::cout << "L1Tree         : gt->tw1[0] = "<<gt_->tw1[0]<<std::endl;

    //rct_
    std::cout << "L1Tree         : rct->RegSize    = "<<rct_->RegSize<<std::endl;
    if (rct_->RegSize!=0)
      std::cout << "L1Tree         : rct->RegEta[0] = "<<rct_->RegEta[0]<<std::endl; 

    //dttf_
    std::cout << "L1Tree         : dttf->trSize     = "<<dttf_->trSize<<std::endl;
    if (dttf_->trSize!=0)
      std::cout << "L1Tree         : dttf->trBx[0] = "<<dttf_->trBx[0]<<std::endl;

    //csctf_
    std::cout << "L1Tree         : csctf_->lctSize    = "<<csctf_->lctSize<<std::endl;
    if (csctf_->lctBx.size()!=0)
      std::cout << "L1Tree         : csctf->lctBx[0] = "<<csctf_->lctBx[0]<<std::endl;

    //recoMuon
    if (domuonreco) std::cout << "L1MuonRecoTree : nb muons   = " << recoMuon_->nMuons << std::endl;

    //recoMet
    if (doreco)     std::cout << "L1RecoTree     : met        = " << recoMet_->met     << std::endl;
  
    //recoJet_
    if (doreco)     std::cout << "L1RecoTree     : nb jets    = " << recoJet_->nJets   << std::endl;

    //recoBasicCluster_
    if (doreco)     std::cout << "L1RecoTree     : nb BasicCluster   = " << recoBasicCluster_->nClusters   << std::endl;

    //recoSuperCluster_
    if (doreco)     std::cout << "L1RecoTree     : nb SuperCluster   = " << recoSuperCluster_->nClusters   << std::endl;

    //recoTrack_
    if (doreco)     std::cout << "L1RecoTree     : nTrk       = " << recoTrack_->nTrk     << std::endl;

    //recoVertex
    if (doreco)     std::cout << "L1RecoTree     : nVtx       = " << recoVertex_->nVtx     << std::endl;    

    //l1extra_
    if (dol1extra)  std::cout << "L1ExtraTree    : et         = " << l1extra_->et[0]      << std::endl;

    // l1menu
    if (dol1menu)   std::cout << "L1MenuTree     : Is PrescaleIndex for AlgoTrigg valid ? " << l1menu_->AlgoTrig_PrescaleFactorIndexValid << std::endl;

    if (dol1menu)   std::cout << "L1MenuTree     : PrescaleIndex for AlgoTrigg  " << l1menu_->AlgoTrig_PrescaleFactorIndex << std::endl;

    if (dol1menu)   std::cout << "L1MenuTree     : Is PrescaleIndex for TechTrigg valid ? " << l1menu_->TechTrig_PrescaleFactorIndexValid << std::endl;

    if (dol1menu)   std::cout << "L1MenuTree     : PrescaleIndex for TechTrigg  " << l1menu_->TechTrig_PrescaleFactorIndex << std::endl;


    }
  }
   
}




void L1Ntuple::Test2()
{ 

  if (fChain == 0)  return;
 
  Long64_t nentries = fChain->GetEntriesFast();
  Long64_t nbytes = 0, nb = 0;
  //unsigned int nevents =0;

  std::cout << nentries << " events to process"<<std::endl;

  TCanvas * can = new TCanvas;
  can->Divide(2,2);

  TH1D * dttf_histo = new TH1D("dttf","dttf",20,0,20);
  TH1D * csctf_histo = new TH1D("csctf","csctf",20,0,20);

  for (Long64_t jentry=0; jentry<nentries;jentry++)
  {
    Long64_t ientry = LoadTree(jentry);
    if (ientry < 0) break;
    
    nb = fChain->GetEntry(jentry);   nbytes += nb;

    dttf_histo->Fill(dttf_->trBx.size());
    csctf_histo->Fill(csctf_->lctBx.size());
  }

  can->cd(1); dttf_histo->Draw();
  can->cd(2); csctf_histo->Draw();
}
