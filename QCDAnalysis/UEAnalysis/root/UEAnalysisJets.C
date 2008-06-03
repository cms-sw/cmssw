#include "UEAnalysisJets.h"
#include <vector>
#include <math.h>

UEAnalysisJets::UEAnalysisJets()
{
  cout << "UEAnalysisJets constructor " <<endl;
  piG = acos(-1.);

}

void UEAnalysisJets::Begin(TFile * f, string hltBit){

  f->cd( hltBit.c_str() );

  dr_chgcalo = new TH1F("dr_chgcalo","#Delta R Charged RECO vs Calorimeter",100,0.,10.);
  dr_chginc = new TH1F("dr_chginc","#Delta R Charged RECO vs Inclusive",100,0.,10.);
  dr_chgmcreco = new TH1F("dr_chgmcreco","#Delta R Charged RECO vs Charged MC",100,0.,10.);
  dr_caloinc = new TH1F("dr_caloinc","#Delta R Calorimeter vs Inclusive",100,0.,10.);
  numb_cal = new TH1F("numb_cal","Number calo Jet",30,0.,30.);
  pT_cal = new TH1F("pT_cal","P_{T} calo",100,0.,200.);
  eta_cal = new TH1F("eta_cal","#eta Calo",100,-3.,3.);
  eta_cal_res = new TH1F("eta_cal_res","#eta_{calo} - #eta_{inc}",100,-3.,3.);
  phi_cal = new TH1F("phi_cal","#phi Calo",50,-3.14,3.14);
  phi_cal_res = new TH1F("phi_cal_res","#phi_{calo} - #phi_{inc}",100,-3.,3.);
  numb_chgmc = new TH1F("numb_chgmc","Number Charged MC Jet",30,0.,30.);
  pT_chgmc = new TH1F("pT_chgmc","P_{T} Charged MC",100,0.,200.);
  eta_chgmc = new TH1F("eta_chgmc","#eta Charged MC",100,-3.,3.);
  eta_chgmc_res = new TH1F("eta_chgmc_res","#eta_{chg MC} - #eta_{inc}",100,-3.,3.);
  phi_chgmc = new TH1F("phi_chgmc","#phi Charged MC",50,-3.14,3.14);
  phi_chgmc_res = new TH1F("phi_chgmc_res","#phi_{chg MC} - #phi_{inc}",100,-3.,3.);
  numb_chgreco = new TH1F("numb_chgreco","Number Charged RECO Jet",30,0.,30.);
  pT_chgreco = new TH1F("pT_chgreco","P_{T} Charged RECO",100,0.,200.);
  eta_chgreco = new TH1F("eta_chgreco","#eta Charged RECO",100,-3.,3.);
  eta_chgreco_res = new TH1F("eta_chgreco_res","#eta_{chg RECO} - #eta_{inc}",100,-3.,3.);
  phi_chgreco = new TH1F("phi_chgreco","#phi Charged RECO",50,-3.14,3.14);
  phi_chgreco_res = new TH1F("phi_chgreco_res","#phi_{chg RECO} - #phi_{inc}",100,-3.,3.);
  numb_inc = new TH1F("numb_inc","Number Inclusive Jet",30,0.,30.);
  pT_inc = new TH1F("pT_inc","P_{T} Inclusive",100,0.,200.);
  eta_inc = new TH1F("eta_inc","#eta Inclusive",100,-3.,3.);
  phi_inc = new TH1F("phi_inc","#phi Inclusive",50,-3.14,3.14);
  calib_chgcalo  = new TProfile("calib_chgcalo","#frac{P_{T}^{Chg RECO}}{P_{T}^{Calo}} vs P_{T}^{Calo}",100,0,200,-4,4);
  calib_chginc  = new TProfile("calib_chginc","#frac{P_{T}^{Chg RECO}}{P_{T}^{Inc}} vs P_{T}^{Inc}",100,0,200,-4,4);
  calib_chgmcreco  = new TProfile("calib_chgmcreco","#frac{P_{T}^{Chg MC}}{P_{T}^{Chg RECO}} vs P_{T}^{Chg RECO}",100,0,200,-4,4);
  calib_caloinc  = new TProfile("calib_caloinc","#frac{P_{T}^{Calo}}{P_{T}^{Inc}} vs P_{T}^{Inc}",100,0,200,-4,4);
  calib_chgcalo_eta  = new TProfile("calib_chgcalo_eta","#frac{P_{T}^{Chg RECO}}{P_{T}^{Calo}} vs #eta^{Calo}",100,-3,3,-4,4);
  calib_chginc_eta  = new TProfile("calib_chginc_eta","#frac{P_{T}^{Chg RECO}}{P_{T}^{Inc}} vs #eta^{Inc}",100,-3,3,-4,4);
  calib_chgmcreco_eta  = new TProfile("calib_chgmcreco_eta","#frac{P_{T}^{Chg MC}}{P_{T}^{Chg RECO}} vs #eta^{Chg RECO}",100,-3,3,-4,4);
  calib_caloinc_eta  = new TProfile("calib_caloinc_eta","#frac{P_{T}^{Calo}}{P_{T}^{Inc}} vs #eta^{Inc}",100,-3,3,-4,4);
  calib_chgcalo_phi  = new TProfile("calib_chgcalo_phi","#frac{P_{T}^{Chg RECO}}{P_{T}^{Calo}} vs #phi^{Calo}",100,-3,3,-4,4);
  calib_chginc_phi  = new TProfile("calib_chginc_phi","#frac{P_{T}^{Chg RECO}}{P_{T}^{Inc}} vs #phi^{Inc}",100,-3,3,-4,4);
  calib_chgmcreco_phi  = new TProfile("calib_chgmcreco_phi","#frac{P_{T}^{Chg MC}}{P_{T}^{Chg RECO}} vs #phi^{Chg RECO}",100,-3,3,-4,4);
  calib_caloinc_phi  = new TProfile("calib_caloinc_phi","#frac{P_{T}^{Calo}}{P_{T}^{Inc}} vs #phi^{Inc}",100,-3,3,-4,4);
  
  h2d_weightVSratioPtTracksJetPtCaloJet 
    = new TH2D("h2d_weightVSratioPtTracksJetPtCaloJet",
	       "h2d_weightVSratioPtTracksJetPtCaloJet;p_{T}(jet from tracks)/p_{T}(calo jet);event weight",
	       100, 0., 4., 100, 0., 10.);
  
  h2d_calib_chgcalo
    = new TH2D("h2d_calib_chgcalo",
	       "h2d_calib_chgcalo;p_{T}(calo jet) (GeV/c);p_{T}(jet from tracks)/p_{T}(calo jet)",
	       100, 0., 200., 100, 0., 4.);
  //
  // hlt1jet30
  // hlt1jet50
  // hlt1jet80
  // hlt1jet110
  // hlt1jet180
  // hlt1jet250
  
  pT_cal_hlt1jet30  = new TH1F("pT_cal_hlt1jet30","P_{T} calo (hlt1jet30)",100,0.,200.);
  pT_cal_hlt1jet50  = new TH1F("pT_cal_hlt1jet50","P_{T} calo (hlt1jet50)",100,0.,200.);
  pT_cal_hlt1jet80  = new TH1F("pT_cal_hlt1jet80","P_{T} calo (hlt1jet80)",100,0.,200.);
  pT_cal_hlt1jet110 = new TH1F("pT_cal_hlt1jet110","P_{T} calo (hlt1jet110)",100,0.,200.);
  pT_cal_hlt1jet180 = new TH1F("pT_cal_hlt1jet180","P_{T} calo (hlt1jet180)",100,0.,200.);
  pT_cal_hlt1jet250 = new TH1F("pT_cal_hlt1jet250","P_{T} calo (hlt1jet250)",100,0.,200.);
  
  pT_chgreco_hlt1jet30  = new TH1F("pT_chgreco_hlt1jet30","P_{T} Charged RECO (hlt1jet30)",100,0.,200.);
  pT_chgreco_hlt1jet50  = new TH1F("pT_chgreco_hlt1jet50","P_{T} Charged RECO (hlt1jet50)",100,0.,200.);
  pT_chgreco_hlt1jet80  = new TH1F("pT_chgreco_hlt1jet80","P_{T} Charged RECO (hlt1jet80)",100,0.,200.);
  pT_chgreco_hlt1jet110 = new TH1F("pT_chgreco_hlt1jet110","P_{T} Charged RECO (hlt1jet110)",100,0.,200.);
  pT_chgreco_hlt1jet180 = new TH1F("pT_chgreco_hlt1jet180","P_{T} Charged RECO (hlt1jet180)",100,0.,200.);
  pT_chgreco_hlt1jet250 = new TH1F("pT_chgreco_hlt1jet250","P_{T} Charged RECO (hlt1jet250)",100,0.,200.);
  
  pT_chgmc_hlt1jet30  = new TH1F("pT_chgmc_hlt1jet30","P_{T} Charged MC (hlt1jet30)",100,0.,200.);
  pT_chgmc_hlt1jet50  = new TH1F("pT_chgmc_hlt1jet50","P_{T} Charged MC (hlt1jet50)",100,0.,200.);
  pT_chgmc_hlt1jet80  = new TH1F("pT_chgmc_hlt1jet80","P_{T} Charged MC (hlt1jet80)",100,0.,200.);
  pT_chgmc_hlt1jet110 = new TH1F("pT_chgmc_hlt1jet110","P_{T} Charged MC (hlt1jet110)",100,0.,200.);
  pT_chgmc_hlt1jet180 = new TH1F("pT_chgmc_hlt1jet180","P_{T} Charged MC (hlt1jet180)",100,0.,200.);
  pT_chgmc_hlt1jet250 = new TH1F("pT_chgmc_hlt1jet250","P_{T} Charged MC (hlt1jet250)",100,0.,200.);
  
  pT_inc_hlt1jet30  = new TH1F("pT_inc_hlt1jet30","P_{T} Inclusive (hlt1jet30)",100,0.,200.);
  pT_inc_hlt1jet50  = new TH1F("pT_inc_hlt1jet50","P_{T} Inclusive (hlt1jet50)",100,0.,200.);
  pT_inc_hlt1jet80  = new TH1F("pT_inc_hlt1jet80","P_{T} Inclusive (hlt1jet80)",100,0.,200.);
  pT_inc_hlt1jet110 = new TH1F("pT_inc_hlt1jet110","P_{T} Inclusive (hlt1jet110)",100,0.,200.);
  pT_inc_hlt1jet180 = new TH1F("pT_inc_hlt1jet180","P_{T} Inclusive (hlt1jet180)",100,0.,200.);
  pT_inc_hlt1jet250 = new TH1F("pT_inc_hlt1jet250","P_{T} Inclusive (hlt1jet250)",100,0.,200.);
  //
  
}

void UEAnalysisJets::jetCalibAnalysis(float weight,float etaRegion,TClonesArray * InclusiveJet,TClonesArray * ChargedJet,TClonesArray * TracksJet,TClonesArray * CalorimeterJet, TClonesArray *acceptedTriggers, TFile* f, string hltBit)
{

  f->cd( hltBit.c_str() );
  
  if(InclusiveJet->GetEntries()!=0 && ChargedJet->GetEntries()!=0 && TracksJet->GetEntries()!=0 && CalorimeterJet->GetEntries()!=0){
    
    float phiEHJ = -666;
    float phiTJ  = -666;
    float phiIJ  = -666;
    float phiCJ  = -666;
    
    float ptEHJ = -666;
    float ptTJ  = -666;
    float ptIJ  = -666;
    float ptCJ  = -666;
    
    float etaEHJ = -666;
    float etaTJ  = -666;
    float etaIJ  = -666;
    float etaCJ  = -666;
	
    TLorentzVector *m=0;
    
    int nIncJet=0;
    
    for(int i=0;i<InclusiveJet->GetSize();++i) {
      TLorentzVector *v = (TLorentzVector*)InclusiveJet->At(i);
      if(fabs(v->Eta())<etaRegion)
	nIncJet++;
    } 
    
    int nChgRECOJet=0;
    for(int i=0;i<TracksJet->GetSize();++i)
      {
	TLorentzVector *v = (TLorentzVector*)TracksJet->At(i);
	if(fabs(v->Eta())<etaRegion)
	  nChgRECOJet++;
      }
    
    int nChgMCJet=0;
    for(int i=0;i<ChargedJet->GetSize();++i)
      {
	TLorentzVector *v = (TLorentzVector*)ChargedJet->At(i);
	if(fabs(v->Eta())<etaRegion)
	  nChgMCJet++;
      }
    
    int nCaloJet=0;
    for(int i=0;i<CalorimeterJet->GetSize();++i)
      {
	TLorentzVector *v = (TLorentzVector*)CalorimeterJet->At(i);
	if(fabs(v->Eta())<etaRegion)
	  nCaloJet++;
      }
    
    numb_cal->Fill(nCaloJet);
    numb_chgmc->Fill(nChgMCJet);
    numb_chgreco->Fill(nChgRECOJet);
    numb_inc->Fill(nIncJet);
    
    TLorentzVector *IJ0 = (TLorentzVector*)InclusiveJet->At(0);
    TLorentzVector *TJ0 = (TLorentzVector*)TracksJet->At(0);
    TLorentzVector *CJ0 = (TLorentzVector*)CalorimeterJet->At(0);
    
    if(fabs(IJ0->Eta())<etaRegion && fabs(TJ0->Eta())<etaRegion){
      eta_chgreco_res->Fill(IJ0->Eta()-TJ0->Eta());
      phi_chgreco_res->Fill(IJ0->Phi()-TJ0->Phi());
    }
    
    if(fabs(IJ0->Eta())<etaRegion && fabs(CJ0->Eta())<etaRegion){
      eta_chgreco_res->Fill(IJ0->Eta()-CJ0->Eta());
      phi_chgreco_res->Fill(IJ0->Phi()-CJ0->Phi());
    }
    
    for(int i=0;i<InclusiveJet->GetSize();++i)
      {
	TLorentzVector *v = (TLorentzVector*)InclusiveJet->At(i);
	if(fabs(v->Eta())<etaRegion)
	  {
	    etaIJ = v->Eta();
	    ptIJ  = v->Pt();
	    phiIJ = v->Phi();
	    break;
	  }
      }
    
    //TLorentzVector* JetFromTracks;
    for(int i=0;i<TracksJet->GetSize();++i)
      {
	TLorentzVector *v = (TLorentzVector*)TracksJet->At(i);
	if(fabs(v->Eta())<etaRegion)
	  {
	    etaTJ = v->Eta();
	    ptTJ  = v->Pt();
	    phiTJ = v->Phi();
	    
	    //JetFromTracks = new TLorentzVector(v->Px(), v->Py(), v->Pz(), v->E() );
	    
	    break;
	  }
      }
    
    for(int i=0;i<ChargedJet->GetSize();++i)
      {
	TLorentzVector *v = (TLorentzVector*)ChargedJet->At(i);
	if(fabs(v->Eta())<etaRegion)
	  {
	    etaCJ = v->Eta();
	    ptCJ  = v->Pt();
	    phiCJ = v->Phi();
	    break;
	  }
      }
    
    //TLorentzVector* CaloJet;
    for(int i=0;i<CalorimeterJet->GetSize();i++)
      {
	TLorentzVector *v = (TLorentzVector*)CalorimeterJet->At(i);
	if(fabs(v->Eta())<etaRegion)
	  {
	    etaEHJ = v->Eta();
	    ptEHJ  = v->Pt();
	    phiEHJ = v->Phi();
	    
	    //CaloJet = new TLorentzVector(v->Px(), v->Py(), v->Pz(), v->E() );
	    
	    break;
	  }
      }
    
    
    if(etaEHJ!=-666&&etaTJ!=-666){
      float dPhiEHJTJ = fabs(phiEHJ-phiTJ);
      if(dPhiEHJTJ>piG)
	dPhiEHJTJ=2*piG-dPhiEHJTJ;
      
      
      float delR_chgcalo=sqrt((etaEHJ-etaTJ)*(etaEHJ-etaTJ)+dPhiEHJTJ*dPhiEHJTJ);
      dr_chgcalo->Fill(delR_chgcalo,weight);
      if(ptEHJ>0)
	{
	  h2d_weightVSratioPtTracksJetPtCaloJet->Fill( ptTJ/ptEHJ, weight );
	  h2d_calib_chgcalo->Fill( ptEHJ, ptTJ/ptEHJ, weight);
	  
	  calib_chgcalo->Fill(ptEHJ,ptTJ/ptEHJ,weight);
	  calib_chgcalo_eta->Fill(etaEHJ,ptTJ/ptEHJ,weight);
	  calib_chgcalo_phi->Fill(phiEHJ,ptTJ/ptEHJ,weight);
	}
    }
    
    if(etaIJ!=-666&&etaTJ!=-666){
      float dPhiIJTJ = fabs(phiIJ-phiTJ);
      if(dPhiIJTJ>piG)
	dPhiIJTJ=2*piG-dPhiIJTJ;
      float delR_chginc=sqrt((etaIJ-etaTJ)*(etaIJ-etaTJ)+dPhiIJTJ*dPhiIJTJ);
      dr_chginc->Fill(delR_chginc,weight);
      if(ptIJ>0)
	{
	  calib_chginc->Fill(ptIJ,ptTJ/ptIJ,weight);
	  calib_chginc_eta->Fill(etaIJ,ptTJ/ptIJ,weight);
	  calib_chginc_phi->Fill(phiIJ,ptTJ/ptIJ,weight);
	}	  
    }
    
    if(etaIJ!=-666&&etaCJ!=-666){
      eta_chgmc_res->Fill((etaIJ-etaCJ));
      phi_chgmc_res->Fill((phiIJ-phiCJ));
    }    
    
    if(etaCJ!=-666&&etaTJ!=-666){
      float dPhiCJTJ = fabs(phiCJ-phiTJ);
      if(dPhiCJTJ>piG)
	dPhiCJTJ=2*piG-dPhiCJTJ;
      float delR_chgmcreco=sqrt((etaCJ-etaTJ)*(etaCJ-etaTJ)+dPhiCJTJ*dPhiCJTJ);
      dr_chgmcreco->Fill(delR_chgmcreco,weight);
      if(ptTJ>0)
	{
	  calib_chgmcreco->Fill(ptTJ,ptCJ/ptTJ,weight);
	  calib_chgmcreco_eta->Fill(etaTJ,ptCJ/ptTJ,weight);
	  calib_chgmcreco_phi->Fill(phiTJ,ptCJ/ptTJ,weight);
	}	  
    }
    
    if(etaEHJ!=-666&&etaIJ!=-666){
      float dPhiIJEHJ = fabs(phiEHJ-phiIJ);
      if(dPhiIJEHJ>piG)
	dPhiIJEHJ=2*piG-dPhiIJEHJ;
      float delR_caloinc=sqrt((etaIJ-etaEHJ)*(etaIJ-etaEHJ)+dPhiIJEHJ*dPhiIJEHJ);
      dr_caloinc->Fill(delR_caloinc,weight);
      if(ptIJ>0)
	{
	  calib_caloinc->Fill(ptIJ,ptEHJ/ptIJ,weight);
	  calib_caloinc_eta->Fill(etaIJ,ptEHJ/ptIJ,weight);
	  calib_caloinc_phi->Fill(phiIJ,ptEHJ/ptIJ,weight);
	}
    }
    
    if(etaEHJ!=-666){
      pT_cal->Fill(ptEHJ,weight);
      eta_cal->Fill(etaEHJ,weight);
      phi_cal->Fill(phiEHJ,weight);
      
      //
      int nAcceptedTriggers( 0 );
      nAcceptedTriggers = acceptedTriggers->GetSize();
      for ( int iAcceptedTrigger(0); iAcceptedTrigger<nAcceptedTriggers; ++iAcceptedTrigger )
	{
	  // HLT1jet30
	  // HLT1jet50
	  // HLT1jet80
	  // HLT1jet110
	  // HLT1jet180
	  // HLT1jet250
	  
	  std::string filterName( acceptedTriggers->At(iAcceptedTrigger)->GetName() );
	  if      ( filterName=="HLT1jet30"  ) pT_cal_hlt1jet30->Fill(ptEHJ,weight);
	  else if ( filterName=="HLT1jet50"  ) pT_cal_hlt1jet50->Fill(ptEHJ,weight);
	  else if ( filterName=="HLT1jet80"  ) pT_cal_hlt1jet80->Fill(ptEHJ,weight);
	  else if ( filterName=="HLT1jet110" ) pT_cal_hlt1jet110->Fill(ptEHJ,weight);
	  else if ( filterName=="HLT1jet180" ) pT_cal_hlt1jet180->Fill(ptEHJ,weight);
	  else if ( filterName=="HLT1jet250" ) pT_cal_hlt1jet250->Fill(ptEHJ,weight);
	    }
      //
    }
    
    if(etaTJ!=-666){
      pT_chgreco->Fill(ptTJ,weight);
      eta_chgreco->Fill(etaTJ,weight);
      phi_chgreco->Fill(phiTJ,weight);
      
      //
      int nAcceptedTriggers( 0 );
      nAcceptedTriggers = acceptedTriggers->GetSize();
	  for ( int iAcceptedTrigger(0); iAcceptedTrigger<nAcceptedTriggers; ++iAcceptedTrigger )
	    {
	      // HLT1jet30
	      // HLT1jet50
	      // HLT1jet80
	      // HLT1jet110
	      // HLT1jet180
	      // HLT1jet250
	      
	      std::string filterName( acceptedTriggers->At(iAcceptedTrigger)->GetName() );
	      if      ( filterName=="HLT1jet30"  ) pT_chgreco_hlt1jet30->Fill(ptTJ,weight);
	      else if ( filterName=="HLT1jet50"  ) pT_chgreco_hlt1jet50->Fill(ptTJ,weight);
	      else if ( filterName=="HLT1jet80"  ) pT_chgreco_hlt1jet80->Fill(ptTJ,weight);
	      else if ( filterName=="HLT1jet110" ) pT_chgreco_hlt1jet110->Fill(ptTJ,weight);
	      else if ( filterName=="HLT1jet180" ) pT_chgreco_hlt1jet180->Fill(ptTJ,weight);
	      else if ( filterName=="HLT1jet250" ) pT_chgreco_hlt1jet250->Fill(ptTJ,weight);
	    }
	  //
	}
	
	if(etaCJ!=-666){
	  pT_chgmc->Fill(ptCJ,weight);
	  eta_chgmc->Fill(etaCJ,weight);
	  phi_chgmc->Fill(phiCJ,weight);
	  
	  //
	  int nAcceptedTriggers( 0 );
	  nAcceptedTriggers = acceptedTriggers->GetSize();
	  for ( int iAcceptedTrigger(0); iAcceptedTrigger<nAcceptedTriggers; ++iAcceptedTrigger )
	    {
	      // HLT1jet30
	      // HLT1jet50
	      // HLT1jet80
	      // HLT1jet110
	      // HLT1jet180
	      // HLT1jet250
	      
	      std::string filterName( acceptedTriggers->At(iAcceptedTrigger)->GetName() );
	      if      ( filterName=="HLT1jet30"  ) pT_chgmc_hlt1jet30->Fill(ptCJ,weight);
	      else if ( filterName=="HLT1jet50"  ) pT_chgmc_hlt1jet50->Fill(ptCJ,weight);
	      else if ( filterName=="HLT1jet80"  ) pT_chgmc_hlt1jet80->Fill(ptCJ,weight);
	      else if ( filterName=="HLT1jet110" ) pT_chgmc_hlt1jet110->Fill(ptCJ,weight);
	      else if ( filterName=="HLT1jet180" ) pT_chgmc_hlt1jet180->Fill(ptCJ,weight);
	      else if ( filterName=="HLT1jet250" ) pT_chgmc_hlt1jet250->Fill(ptCJ,weight);
	    }
	  //
	}
	
	if(etaIJ!=-666){
	  pT_inc->Fill(ptIJ,weight);
	  eta_inc->Fill(etaIJ,weight);
	  phi_inc->Fill(phiIJ,weight);
	  
	  //
	  int nAcceptedTriggers( 0 );
	  nAcceptedTriggers = acceptedTriggers->GetSize();
	  for ( int iAcceptedTrigger(0); iAcceptedTrigger<nAcceptedTriggers; ++iAcceptedTrigger )
	    {
	      // HLT1jet30
	      // HLT1jet50
	      // HLT1jet80
	      // HLT1jet110
	      // HLT1jet180
	      // HLT1jet250

	      std::string filterName( acceptedTriggers->At(iAcceptedTrigger)->GetName() );
	      if      ( filterName=="HLT1jet30"  ) pT_inc_hlt1jet30->Fill(ptIJ,weight);
	      else if ( filterName=="HLT1jet50"  ) pT_inc_hlt1jet50->Fill(ptIJ,weight);
	      else if ( filterName=="HLT1jet80"  ) pT_inc_hlt1jet80->Fill(ptIJ,weight);
	      else if ( filterName=="HLT1jet110" ) pT_inc_hlt1jet110->Fill(ptIJ,weight);
	      else if ( filterName=="HLT1jet180" ) pT_inc_hlt1jet180->Fill(ptIJ,weight);
	      else if ( filterName=="HLT1jet250" ) pT_inc_hlt1jet250->Fill(ptIJ,weight);
	    }
	  //
	}
	
  }
      
      //    }
}


