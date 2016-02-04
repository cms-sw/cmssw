#include "UEAnalysisUE.h"
#include <vector>
#include <math.h>

UEAnalysisUE::UEAnalysisUE()
{
  std::cout << "UEAnalysisUE constructor " <<std::endl;
  piG = acos(-1.);
  cc = new UEAnalysisCorrCali();
}

void UEAnalysisUE::Begin(TFile * f)
{

  f->cd();

  //Underlying Event analysis
  fHistPtDistMC   = new TH1F(  "HistPtDistMC"  , "Pt Spectra", 100,  0., 4. ) ;
  fHistEtaDistMC  = new TH1F(  "HistEtaDistMC" , "#eta Spectra", 100, -5., 5. ) ;
  fHistPhiDistMC  = new TH1F(  "HistPhiDistMC" , "#phi Spectra", 100, -4., 4. ) ;

  pdN_vs_etaMC              = new TProfile("dN_vs_etaMC","#delta N vs #eta",100,0.,5.);
  pdN_vs_ptMC               = new TProfile("dN_vs_ptMC","#delta N vs P_{T}",1000,0.,100.);
  pdN_vs_dphiMC             = new TProfile("dN_vs_dphiMC","#frac{dN}{d#phid#eta} vs #delta #phi",100,-180.,180.,0,100);
  pdPt_vs_dphiMC            = new TProfile("dPt_vs_dphiMC","#frac{dP_{T}^{sum}}{d#phid#eta} vs #delta #phi",100,-180.,180.,0,100);

  // add histo for ue fluctuation
  h2d_dN_vs_ptJTransMC = new TH2D("h2d_dN_vs_ptJTransMC","#frac{dN}{d#phid#eta} vs P_{T}^{Chg Jet} ''Trans''",100,0.,200,100,0.,20.);

  pdN_vs_ptJTransMC         = new TProfile("dN_vs_ptJTransMC","#frac{dN}{d#phid#eta} vs P_{T}^{Chg Jet} ''Trans''",100,0.,200);
  pdN_vs_ptJTransMaxMC      = new TProfile("dN_vs_ptJTransMaxMC","#frac{dN}{d#phid#eta} vs P_{T}^{Chg Jet} ''Trans Max''",100,0.,200);
  pdN_vs_ptJTransMinMC      = new TProfile("dN_vs_ptJTransMinMC","#frac{dN}{d#phid#eta} vs P_{T}^{Chg Jet} ''Trans Min''",100,0.,200);

  pdPt_vs_ptJTransMC        = new TProfile("dPt_vs_ptJTransMC","#frac{dP_{T}^{sum}}{d#phid#eta} vs P_{T}^{Chg Jet} ''Trans''",100,0.,200);
  pdPt_vs_ptJTransMaxMC     = new TProfile("dPt_vs_ptJTransMaxMC","#frac{dP_{T}^{sum}}{d#phid#eta} vs P_{T}^{Chg Jet} ''Trans Max''",100,0.,200);
  pdPt_vs_ptJTransMinMC     = new TProfile("dPt_vs_ptJTransMinMC","#frac{dP_{T}^{sum}}{d#phid#eta} vs P_{T}^{Chg Jet} ''Trans Min''",100,0.,200);

  pdN_vs_ptJTowardMC       = new TProfile("dN_vs_ptJTowardMC","#frac{dN}{d#phid#eta} vs P_{T}^{Chg Jet} ''Toward''",100,0.,200);
  pdN_vs_ptJAwayMC          = new TProfile("dN_vs_ptJAwayMC","#frac{dN}{d#phid#eta} vs P_{T}^{Chg Jet} ''Away''",100,0.,200);

  pdPt_vs_ptJTowardMC      = new TProfile("dPt_vs_ptJTowardMC","#frac{dP_{T}^{sum}}{d#phid#eta} vs P_{T}^{Chg Jet} ''Toward''",100,0.,200);
  pdPt_vs_ptJAwayMC         = new TProfile("dPt_vs_ptJAwayMC","#frac{dP_{T}^{sum}}{d#phid#eta} vs P_{T}^{Chg Jet} ''Away''",100,0.,200);

  temp1MC = new TH1F("temp1MC","temp",100,-180.,180.);
  temp2MC = new TH1F("temp2MC","temp",100,-180.,180.);
  temp3MC = new TH1F("temp3MC","temp",100,0.,5.);
  temp4MC = new TH1F("temp4MC","temp",1000,0.,100.);

  fHistPtDistRECO   = new TH1F(  "HistPtDistRECO"  , "Pt Spectra", 100,  0., 4. ) ;
  fHistEtaDistRECO  = new TH1F(  "HistEtaDistRECO" , "#eta Spectra", 100, -5., 5. ) ;
  fHistPhiDistRECO  = new TH1F(  "HistPhiDistRECO" , "#phi Spectra", 100, -4., 4. ) ;

  pdN_vs_etaRECO              = new TProfile("dN_vs_etaRECO","#delta N vs #eta",100,0.,5.);
  pdN_vs_ptRECO               = new TProfile("dN_vs_ptRECO","#delta N vs P_{T}",1000,0.,100.);

  pdN_vs_dphiRECO             = new TProfile("dN_vs_dphiRECO","#frac{dN}{d#phid#eta} vs #delta #phi",100,-180.,180.,0,100);
  pdPt_vs_dphiRECO            = new TProfile("dPt_vs_dphiRECO","#frac{dP_{T}^{sum}}{d#phid#eta} vs #delta #phi",100,-180.,180.,0,100);

  pdN_vs_ptJTransRECO         = new TProfile("dN_vs_ptJTransRECO","#frac{dN}{d#phid#eta} vs P_{T}^{Chg Jet} ''Trans''",100,0.,200);
  pdN_vs_ptJTransMaxRECO      = new TProfile("dN_vs_ptJTransMaxRECO","#frac{dN}{d#phid#eta} vs P_{T}^{Chg Jet} ''Trans Max''",100,0.,200);
  pdN_vs_ptJTransMinRECO      = new TProfile("dN_vs_ptJTransMinRECO","#frac{dN}{d#phid#eta} vs P_{T}^{Chg Jet} ''Trans Min''",100,0.,200);

  pdPt_vs_ptJTransRECO        = new TProfile("dPt_vs_ptJTransRECO","#frac{dP_{T}^{sum}}{d#phid#eta} vs P_{T}^{Chg Jet} ''Trans''",100,0.,200);
  pdPt_vs_ptJTransMaxRECO     = new TProfile("dPt_vs_ptJTransMaxRECO","#frac{dP_{T}^{sum}}{d#phid#eta} vs P_{T}^{Chg Jet} ''Trans Max''",100,0.,200);
  pdPt_vs_ptJTransMinRECO     = new TProfile("dPt_vs_ptJTransMinRECO","#frac{dP_{T}^{sum}}{d#phid#eta} vs P_{T}^{Chg Jet} ''Trans Min''",100,0.,200);

  pdN_vs_ptJTowardRECO       = new TProfile("dN_vs_ptJTowardRECO","#frac{dN}{d#phid#eta} vs P_{T}^{Chg Jet} ''Toward''",100,0.,200);
  pdN_vs_ptJAwayRECO          = new TProfile("dN_vs_ptJAwayRECO","#frac{dN}{d#phid#eta} vs P_{T}^{Chg Jet} ''Away''",100,0.,200);

  pdPt_vs_ptJTowardRECO      = new TProfile("dPt_vs_ptJTowardRECO","#frac{dP_{T}^{sum}}{d#phid#eta} vs P_{T}^{Chg Jet} ''Toward''",100,0.,200);
  pdPt_vs_ptJAwayRECO         = new TProfile("dPt_vs_ptJAwayRECO","#frac{dP_{T}^{sum}}{d#phid#eta} vs P_{T}^{Chg Jet} ''Away''",100,0.,200);

  pdN_vs_ptCJTransRECO         = new TProfile("dN_vs_ptCJTransRECO","#frac{dN}{d#phid#eta} vs P_{T}^{Chg Jet MC} ''Trans''",100,0.,200);
  pdN_vs_ptCJTransMaxRECO      = new TProfile("dN_vs_ptCJTransMaxRECO","#frac{dN}{d#phid#eta} vs P_{T}^{Chg Jet MC} ''Trans Max''",100,0.,200);
  pdN_vs_ptCJTransMinRECO      = new TProfile("dN_vs_ptCJTransMinRECO","#frac{dN}{d#phid#eta} vs P_{T}^{Chg Jet MC} ''Trans Min''",100,0.,200);

  pdPt_vs_ptCJTransRECO        = new TProfile("dPt_vs_ptCJTransRECO","#frac{dP_{T}^{sum}}{d#phid#eta} vs P_{T}^{Chg Jet MC} ''Trans''",100,0.,200);
  pdPt_vs_ptCJTransMaxRECO     = new TProfile("dPt_vs_ptCJTransMaxRECO","#frac{dP_{T}^{sum}}{d#phid#eta} vs P_{T}^{Chg Jet MC} ''Trans Max''",100,0.,200);
  pdPt_vs_ptCJTransMinRECO     = new TProfile("dPt_vs_ptCJTransMinRECO","#frac{dP_{T}^{sum}}{d#phid#eta} vs P_{T}^{Chg Jet MC} ''Trans Min''",100,0.,200);

  pdN_vs_ptCJTowardRECO       = new TProfile("dN_vs_ptCJTowardRECO","#frac{dN}{d#phid#eta} vs P_{T}^{Chg Jet MC} ''Toward''",100,0.,200);
  pdN_vs_ptCJAwayRECO          = new TProfile("dN_vs_ptCJAwayRECO","#frac{dN}{d#phid#eta} vs P_{T}^{Chg Jet MC} ''Away''",100,0.,200);

  pdPt_vs_ptCJTowardRECO      = new TProfile("dPt_vs_ptCJTowardRECO","#frac{dP_{T}^{sum}}{d#phid#eta} vs P_{T}^{Chg Jet MC} ''Toward''",100,0.,200);
  pdPt_vs_ptCJAwayRECO         = new TProfile("dPt_vs_ptCJAwayRECO","#frac{dP_{T}^{sum}}{d#phid#eta} vs P_{T}^{Chg Jet MC} ''Away''",100,0.,200);

  temp1RECO = new TH1F("temp1RECO","temp",100,-180.,180.);
  temp2RECO = new TH1F("temp2RECO","temp",100,-180.,180.);
  temp3RECO = new TH1F("temp3RECO","temp",100,0.,5.);
  temp4RECO = new TH1F("temp4RECO","temp",1000,0.,100.);

}

void UEAnalysisUE::ueAnalysisMC(float weight,std::string tkpt,float etaRegion, float ptThreshold, TClonesArray* MonteCarlo, TClonesArray* ChargedJet)
{

  //std::cout << "UEAnalysisUE::ueAnalysisMC(...)" << std::endl;

  TLorentzVector* leadingJet;
  Float_t PTLeadingCJ = -10;
  for(int j=0;j<ChargedJet->GetSize();++j){
    TLorentzVector *v = (TLorentzVector*)ChargedJet->At(j);
    if(fabs(v->Eta())<etaRegion){
      leadingJet = v;
      PTLeadingCJ= v->Pt();
      break;
    }
  }


  //std::cout << "PTLeadingCJ " << PTLeadingCJ << std::endl;

  if ( PTLeadingCJ == -10. )
    {
      //std::cout << "return" << std::endl;
      return;
    }

  //std::cout << "for(int i=0;i<MonteCarlo->GetSize();i++){" << std::endl;

  for(int i=0;i<MonteCarlo->GetSize();i++){
    TLorentzVector *v = (TLorentzVector*)MonteCarlo->At(i);    

    if(v->Pt()>=ptThreshold){
      fHistPtDistMC->Fill(v->Pt(),weight);
      fHistEtaDistMC->Fill(v->Eta(),weight);
      fHistPhiDistMC->Fill(v->Phi(),weight);
      temp3MC->Fill(fabs(v->Eta()));
      temp4MC->Fill(fabs(v->Pt()));
    }

    if(fabs(v->Eta())<etaRegion && v->Pt()>=ptThreshold){
      Float_t conv = 180/piG;
      Float_t Dphi_mc = conv * leadingJet->DeltaPhi(*v);
      temp1MC->Fill(Dphi_mc);
      temp2MC->Fill(Dphi_mc,v->Pt());
    }
  }
  
  //std::cout << "for(int i=0;i<100;i++){" << std::endl;

  for(int i=0;i<100;i++){
    pdN_vs_etaMC->Fill((i*0.05)+0.025,temp3MC->GetBinContent(i+1)/0.1,weight);
  }
  for(int i=0;i<1000;i++){
    pdN_vs_ptMC->Fill((i*0.1)+0.05,temp4MC->GetBinContent(i+1)/0.1,weight);
  }

  temp3MC->Reset();
  temp4MC->Reset();
    
  Float_t transN1=0;
  Float_t transN2=0;
  Float_t transP1=0;
  Float_t transP2=0;
  Float_t towardN=0;
  Float_t towardP=0;
  Float_t awayN=0;
  Float_t awayP=0;
  
  for(int i=0;i<100;i++){
    if(i<=14){
      awayN += temp1MC->GetBinContent(i+1);
      awayP += temp2MC->GetBinContent(i+1);
    }
    if(i>14 && i<33 ){
      transN1 += temp1MC->GetBinContent(i+1);
      transP1 += temp2MC->GetBinContent(i+1);
    }
    if(i>=33 && i<=64 ){
      towardN += temp1MC->GetBinContent(i+1);
      towardP += temp2MC->GetBinContent(i+1);
    }
    if(i>64 && i<83 ){
      transN2 += temp1MC->GetBinContent(i+1);
      transP2 += temp2MC->GetBinContent(i+1);
    }
    if(i>=83){
      awayN += temp1MC->GetBinContent(i+1);
      awayP += temp2MC->GetBinContent(i+1);
    }

    Float_t bincont1_mc=temp1MC->GetBinContent(i+1);
    pdN_vs_dphiMC->Fill(-180.+i*3.6+1.8,bincont1_mc/(3.6*2*etaRegion*(piG/180.)),weight);
    
    Float_t bincont2_mc=temp2MC->GetBinContent(i+1);
    pdPt_vs_dphiMC->Fill(-180.+i*3.6+1.8,bincont2_mc/(3.6*2*etaRegion*(piG/180.)),weight);
    
  }
  
  //std::cout << "bool orderedN = false;" << std::endl;

  bool orderedN = false;
  //  bool orderedP = false;
  
  pdN_vs_ptJTowardMC->Fill(PTLeadingCJ,(towardN)/(120.*(2*etaRegion)*(piG/180.)),weight);
  
  pdPt_vs_ptJTowardMC->Fill(PTLeadingCJ,(towardP)/(120.*(2*etaRegion)*(piG/180.)),weight);
  
  pdN_vs_ptJAwayMC->Fill(PTLeadingCJ,(awayN)/(120.*(2*etaRegion)*(piG/180.)),weight);
  
  pdPt_vs_ptJAwayMC->Fill(PTLeadingCJ,(awayP)/(120.*(2*etaRegion)*(piG/180.)),weight);
  
  if( transN1>=transN2 ) orderedN = true;
  //  if( transP1>=transP2 ) orderedP = true;

  // add histo for ue fluctuation
  h2d_dN_vs_ptJTransMC->Fill(PTLeadingCJ,(transN1+transN2)/(120.*(2*etaRegion)*(piG/180.)),weight);

  pdN_vs_ptJTransMC->Fill(PTLeadingCJ,(transN1+transN2)/(120.*(2*etaRegion)*(piG/180.)),weight);
  
  pdPt_vs_ptJTransMC->Fill(PTLeadingCJ,(transP1+transP2)/(120.*(2*etaRegion)*(piG/180.)),weight);
  
  if(orderedN){
    //dN
    pdN_vs_ptJTransMinMC->Fill(PTLeadingCJ,transN2/(60.*(2*etaRegion)*(piG/180.)),weight);
    
    pdN_vs_ptJTransMaxMC->Fill(PTLeadingCJ,transN1/(60.*(2*etaRegion)*(piG/180.)),weight);
    //dP
    pdPt_vs_ptJTransMinMC->Fill(PTLeadingCJ,transP2/(60.*(2.*etaRegion)*(piG/180.)),weight);
    
    pdPt_vs_ptJTransMaxMC->Fill(PTLeadingCJ,transP1/(60.*(2.*etaRegion)*(piG/180.)),weight);
  }else{
    //dN
    pdN_vs_ptJTransMinMC->Fill(PTLeadingCJ,transN1/(60.*(2.*etaRegion)*(piG/180.)),weight);

    pdN_vs_ptJTransMaxMC->Fill(PTLeadingCJ,transN2/(60.*(2.*etaRegion)*(piG/180.)),weight);
    //dP
    pdPt_vs_ptJTransMinMC->Fill(PTLeadingCJ,transP1/(60.*(2.*etaRegion)*(piG/180.)),weight);
    
    pdPt_vs_ptJTransMaxMC->Fill(PTLeadingCJ,transP2/(60.*(2.*etaRegion)*(piG/180.)),weight);
  }
  /*
  if(orderedP){
    //dN
    pdN_vs_ptJTransMinMC->Fill(PTLeadingCJ,transN2/(60.*(2*etaRegion)*(piG/180.)),weight);
    
    pdN_vs_ptJTransMaxMC->Fill(PTLeadingCJ,transN1/(60.*(2*etaRegion)*(piG/180.)),weight);
    //dP
    pdPt_vs_ptJTransMinMC->Fill(PTLeadingCJ,transP2/(60.*(2.*etaRegion)*(piG/180.)),weight);
    
    pdPt_vs_ptJTransMaxMC->Fill(PTLeadingCJ,transP1/(60.*(2.*etaRegion)*(piG/180.)),weight);
  }else{
    //dN
    pdN_vs_ptJTransMinMC->Fill(PTLeadingCJ,transN1/(60.*(2.*etaRegion)*(piG/180.)),weight);

    pdN_vs_ptJTransMaxMC->Fill(PTLeadingCJ,transN2/(60.*(2.*etaRegion)*(piG/180.)),weight);
    //dP
    pdPt_vs_ptJTransMinMC->Fill(PTLeadingCJ,transP1/(60.*(2.*etaRegion)*(piG/180.)),weight);
    
    pdPt_vs_ptJTransMaxMC->Fill(PTLeadingCJ,transP2/(60.*(2.*etaRegion)*(piG/180.)),weight);
  }
  */
  temp1MC->Reset();
  temp2MC->Reset();

  //std::cout << "done" << std::endl;

}

void UEAnalysisUE::ueAnalysisRECO(float weight,std::string tkpt,float etaRegion,float ptThreshold, TClonesArray* Track, TClonesArray* TracksJet)
{
  
  TLorentzVector* leadingJet;
  Float_t PTLeadingTJ = -10;
  for(int j=0;j<TracksJet->GetSize();++j){
    TLorentzVector *v = (TLorentzVector*)TracksJet->At(j);
    if(fabs(v->Eta())<etaRegion){
      leadingJet = v;
      PTLeadingTJ= v->Pt();
      break;
    }
  }

  Float_t  PTLeadingCJ = cc->calibrationPt(PTLeadingTJ,tkpt)*PTLeadingTJ;

  for(int i=0;i<Track->GetSize();++i){
    TLorentzVector *v = (TLorentzVector*)Track->At(i);

    if(v->Pt()>ptThreshold){
      fHistPtDistRECO->Fill(v->Pt(),weight);
      fHistEtaDistRECO->Fill(v->Eta(),weight);
      fHistPhiDistRECO->Fill(v->Phi(),weight);
      temp3RECO->Fill(fabs(v->Eta()));
      temp4RECO->Fill(fabs(v->Pt()));
    }

    if(fabs(v->Eta())<etaRegion&&v->Pt()>=ptThreshold){
      
      // use ROOT method to calculate dphi                                                                                    
      // convert dphi from radiants to degrees                                                                                
      Float_t conv = 180/piG;
      Float_t Dphi_reco = conv * leadingJet->DeltaPhi(*v);
      
      temp1RECO->Fill(Dphi_reco);
      temp2RECO->Fill(Dphi_reco,v->Pt());
    }
  }

  
  for(int i=0;i<100;i++){
    pdN_vs_etaRECO->Fill((i*0.05)+0.025,temp3RECO->GetBinContent(i+1)/0.1,weight);
  }
  for(int i=0;i<1000;i++){
    pdN_vs_ptRECO->Fill((i*0.1)+0.05,temp4RECO->GetBinContent(i+1)/0.1,weight);
  }
  
  temp3RECO->Reset();
  temp4RECO->Reset();
  
  Float_t transN1=0;
  Float_t transN2=0;
  Float_t transP1=0;
  Float_t transP2=0;
  Float_t towardN=0;
  Float_t towardP=0;
  Float_t awayN=0;
  Float_t awayP=0;
  
  for(int i=0;i<100;i++){
    if(i<=14){
      awayN += temp1RECO->GetBinContent(i+1);
      awayP += temp2RECO->GetBinContent(i+1);
    }
    if(i>14 && i<33 ){
      transN1 += temp1RECO->GetBinContent(i+1);
      transP1 += temp2RECO->GetBinContent(i+1);
    }
    if(i>=33 && i<=64 ){
      towardN += temp1RECO->GetBinContent(i+1);
      towardP += temp2RECO->GetBinContent(i+1);
    }
    if(i>64 && i<83 ){
      transN2 += temp1RECO->GetBinContent(i+1);
      transP2 += temp2RECO->GetBinContent(i+1);
    }
    if(i>=83){
      awayN += temp1RECO->GetBinContent(i+1);
      awayP += temp2RECO->GetBinContent(i+1);
    }

    Float_t bincont1_reco=temp1RECO->GetBinContent(i+1);
    pdN_vs_dphiRECO->Fill(-180.+i*3.6+1.8,bincont1_reco/(3.6*2*etaRegion*(piG/180.)),weight);
    
    Float_t bincont2_reco=temp2RECO->GetBinContent(i+1);
    pdPt_vs_dphiRECO->Fill(-180.+i*3.6+1.8,bincont2_reco/(3.6*2*etaRegion*(piG/180.)),weight);
    
  }
  
  bool orderedN = false;
  //  bool orderedP = false;
  
  pdN_vs_ptJTowardRECO->Fill(PTLeadingTJ,(towardN)/(120.*(2*etaRegion)*(piG/180.)),weight);
  
  pdPt_vs_ptJTowardRECO->Fill(PTLeadingTJ,(towardP)/(120.*(2*etaRegion)*(piG/180.)),weight);
  
  pdN_vs_ptJAwayRECO->Fill(PTLeadingTJ,(awayN)/(120.*(2*etaRegion)*(piG/180.)),weight);
  
  pdPt_vs_ptJAwayRECO->Fill(PTLeadingTJ,(awayP)/(120.*(2*etaRegion)*(piG/180.)),weight);
  
  pdN_vs_ptCJTowardRECO->Fill(PTLeadingCJ,(towardN*cc->correctionNToward(PTLeadingTJ,tkpt))/(120.*(2*etaRegion)*(piG/180.)),weight);
  
  pdPt_vs_ptCJTowardRECO->Fill(PTLeadingCJ,(towardP*cc->correctionPtToward(PTLeadingTJ,tkpt))/(120.*(2*etaRegion)*(piG/180.)),weight);
  
  pdN_vs_ptCJAwayRECO->Fill(PTLeadingCJ,(awayN*cc->correctionNAway(PTLeadingTJ,tkpt))/(120.*(2*etaRegion)*(piG/180.)),weight);
  
  pdPt_vs_ptCJAwayRECO->Fill(PTLeadingCJ,(awayP*cc->correctionPtAway(PTLeadingTJ,tkpt))/(120.*(2*etaRegion)*(piG/180.)),weight);

  /*
  pdN_vs_ptCJTowardRECO->Fill(PTLeadingCJ,(towardN)/(120.*(2*etaRegion)*(piG/180.)),weight);
  
  pdPt_vs_ptCJTowardRECO->Fill(PTLeadingCJ,(towardP)/(120.*(2*etaRegion)*(piG/180.)),weight);
  
  pdN_vs_ptCJAwayRECO->Fill(PTLeadingCJ,(awayN)/(120.*(2*etaRegion)*(piG/180.)),weight);
  
  pdPt_vs_ptCJAwayRECO->Fill(PTLeadingCJ,(awayP)/(120.*(2*etaRegion)*(piG/180.)),weight);
  */
  
  if( transN1>=transN2 ) orderedN = true;
  //  if( transP1>=transP2 ) orderedP = true;
  
  pdN_vs_ptJTransRECO->Fill(PTLeadingTJ,(transN1+transN2)/(120.*(2*etaRegion)*(piG/180.)),weight);
  
  pdPt_vs_ptJTransRECO->Fill(PTLeadingTJ,(transP1+transP2)/(120.*(2*etaRegion)*(piG/180.)),weight);

  pdN_vs_ptCJTransRECO->Fill(PTLeadingCJ,((transN1+transN2)*cc->correctionNTrans(PTLeadingTJ,tkpt))/(120.*(2*etaRegion)*(piG/180.)),weight);
  
  pdPt_vs_ptCJTransRECO->Fill(PTLeadingCJ,((transP1+transP2)*cc->correctionPtTrans(PTLeadingTJ,tkpt))/(120.*(2*etaRegion)*(piG/180.)),weight);

  /*  
  pdN_vs_ptCJTransRECO->Fill(PTLeadingCJ,((transN1+transN2))/(120.*(2*etaRegion)*(piG/180.)),weight);
  
  pdPt_vs_ptCJTransRECO->Fill(PTLeadingCJ,((transP1+transP2))/(120.*(2*etaRegion)*(piG/180.)),weight);
  */

  if(orderedN){

    //dN

    pdN_vs_ptJTransMinRECO->Fill(PTLeadingTJ,transN2/(60.*(2*etaRegion)*(piG/180.)),weight);
    
    pdN_vs_ptJTransMaxRECO->Fill(PTLeadingTJ,transN1/(60.*(2*etaRegion)*(piG/180.)),weight);

    pdN_vs_ptCJTransMinRECO->Fill(PTLeadingCJ,(transN2*cc->correctionNTrans(PTLeadingTJ,tkpt))/(60.*(2*etaRegion)*(piG/180.)),weight);
    
    pdN_vs_ptCJTransMaxRECO->Fill(PTLeadingCJ,(transN1*cc->correctionNTrans(PTLeadingTJ,tkpt))/(60.*(2*etaRegion)*(piG/180.)),weight);

    /*
    pdN_vs_ptCJTransMinRECO->Fill(PTLeadingCJ,(transN2)/(60.*(2*etaRegion)*(piG/180.)),weight);
    
    pdN_vs_ptCJTransMaxRECO->Fill(PTLeadingCJ,(transN1)/(60.*(2*etaRegion)*(piG/180.)),weight);
    */

    //dP

    pdPt_vs_ptJTransMinRECO->Fill(PTLeadingTJ,transP2/(60.*(2.*etaRegion)*(piG/180.)),weight);
    
    pdPt_vs_ptJTransMaxRECO->Fill(PTLeadingTJ,transP1/(60.*(2.*etaRegion)*(piG/180.)),weight);

    pdPt_vs_ptCJTransMinRECO->Fill(PTLeadingCJ,(transP2*cc->correctionPtTrans(PTLeadingTJ,tkpt))/(60.*(2.*etaRegion)*(piG/180.)),weight);
    
    pdPt_vs_ptCJTransMaxRECO->Fill(PTLeadingCJ,(transP1*cc->correctionPtTrans(PTLeadingTJ,tkpt))/(60.*(2.*etaRegion)*(piG/180.)),weight);

    /*
    pdPt_vs_ptCJTransMinRECO->Fill(PTLeadingCJ,(transP2)/(60.*(2.*etaRegion)*(piG/180.)),weight);
    
    pdPt_vs_ptCJTransMaxRECO->Fill(PTLeadingCJ,(transP1)/(60.*(2.*etaRegion)*(piG/180.)),weight);
    */

  }else{

    //dN

    pdN_vs_ptJTransMinRECO->Fill(PTLeadingTJ,transN1/(60.*(2.*etaRegion)*(piG/180.)),weight);

    pdN_vs_ptJTransMaxRECO->Fill(PTLeadingTJ,transN2/(60.*(2.*etaRegion)*(piG/180.)),weight);

    pdN_vs_ptCJTransMinRECO->Fill(PTLeadingCJ,(transN1*cc->correctionNTrans(PTLeadingTJ,tkpt))/(60.*(2.*etaRegion)*(piG/180.)),weight);

    pdN_vs_ptCJTransMaxRECO->Fill(PTLeadingCJ,(transN2*cc->correctionNTrans(PTLeadingTJ,tkpt))/(60.*(2.*etaRegion)*(piG/180.)),weight);

    /*
    pdN_vs_ptCJTransMinRECO->Fill(PTLeadingCJ,(transN1)/(60.*(2*etaRegion)*(piG/180.)),weight);
    
    pdN_vs_ptCJTransMaxRECO->Fill(PTLeadingCJ,(transN2)/(60.*(2*etaRegion)*(piG/180.)),weight);
    */

    //dP

    pdPt_vs_ptJTransMinRECO->Fill(PTLeadingTJ,transP1/(60.*(2.*etaRegion)*(piG/180.)),weight);
    
    pdPt_vs_ptJTransMaxRECO->Fill(PTLeadingTJ,transP2/(60.*(2.*etaRegion)*(piG/180.)),weight);

    pdPt_vs_ptCJTransMinRECO->Fill(PTLeadingCJ,(transP1*cc->correctionPtTrans(PTLeadingTJ,tkpt))/(60.*(2.*etaRegion)*(piG/180.)),weight);
    
    pdPt_vs_ptCJTransMaxRECO->Fill(PTLeadingCJ,(transP2*cc->correctionPtTrans(PTLeadingTJ,tkpt))/(60.*(2.*etaRegion)*(piG/180.)),weight);

    /*
    pdPt_vs_ptCJTransMinRECO->Fill(PTLeadingCJ,(transP1)/(60.*(2.*etaRegion)*(piG/180.)),weight);
    
    pdPt_vs_ptCJTransMaxRECO->Fill(PTLeadingCJ,(transP2)/(60.*(2.*etaRegion)*(piG/180.)),weight);
    */
  }
  
  /*
  if(orderedP){

    //dN

    pdN_vs_ptJTransMinRECO->Fill(PTLeadingTJ,transN2/(60.*(2*etaRegion)*(piG/180.)),weight);
    
    pdN_vs_ptJTransMaxRECO->Fill(PTLeadingTJ,transN1/(60.*(2*etaRegion)*(piG/180.)),weight);

    pdN_vs_ptCJTransMinRECO->Fill(PTLeadingCJ,(transN2*cc->correctionNTrans(PTLeadingTJ,tkpt))/(60.*(2*etaRegion)*(piG/180.)),weight);
    
    pdN_vs_ptCJTransMaxRECO->Fill(PTLeadingCJ,(transN1*cc->correctionNTrans(PTLeadingTJ,tkpt))/(60.*(2*etaRegion)*(piG/180.)),weight);

    
    //pdN_vs_ptCJTransMinRECO->Fill(PTLeadingCJ,(transN2)/(60.*(2*etaRegion)*(piG/180.)),weight);
    
    //pdN_vs_ptCJTransMaxRECO->Fill(PTLeadingCJ,(transN1)/(60.*(2*etaRegion)*(piG/180.)),weight);
    

    //dP

    pdPt_vs_ptJTransMinRECO->Fill(PTLeadingTJ,transP2/(60.*(2.*etaRegion)*(piG/180.)),weight);
    
    pdPt_vs_ptJTransMaxRECO->Fill(PTLeadingTJ,transP1/(60.*(2.*etaRegion)*(piG/180.)),weight);

    pdPt_vs_ptCJTransMinRECO->Fill(PTLeadingCJ,(transP2*cc->correctionPtTrans(PTLeadingTJ,tkpt))/(60.*(2.*etaRegion)*(piG/180.)),weight);
    
    pdPt_vs_ptCJTransMaxRECO->Fill(PTLeadingCJ,(transP1*cc->correctionPtTrans(PTLeadingTJ,tkpt))/(60.*(2.*etaRegion)*(piG/180.)),weight);

    
    //pdPt_vs_ptCJTransMinRECO->Fill(PTLeadingCJ,(transP2)/(60.*(2.*etaRegion)*(piG/180.)),weight);
    
    //pdPt_vs_ptCJTransMaxRECO->Fill(PTLeadingCJ,(transP1)/(60.*(2.*etaRegion)*(piG/180.)),weight);
    

  }else{

    //dN

    pdN_vs_ptJTransMinRECO->Fill(PTLeadingTJ,transN1/(60.*(2.*etaRegion)*(piG/180.)),weight);

    pdN_vs_ptJTransMaxRECO->Fill(PTLeadingTJ,transN2/(60.*(2.*etaRegion)*(piG/180.)),weight);

    pdN_vs_ptCJTransMinRECO->Fill(PTLeadingCJ,(transN1*cc->correctionNTrans(PTLeadingTJ,tkpt))/(60.*(2.*etaRegion)*(piG/180.)),weight);

    pdN_vs_ptCJTransMaxRECO->Fill(PTLeadingCJ,(transN2*cc->correctionNTrans(PTLeadingTJ,tkpt))/(60.*(2.*etaRegion)*(piG/180.)),weight);

    
    //pdN_vs_ptCJTransMinRECO->Fill(PTLeadingCJ,(transN1)/(60.*(2*etaRegion)*(piG/180.)),weight);
    
    //pdN_vs_ptCJTransMaxRECO->Fill(PTLeadingCJ,(transN2)/(60.*(2*etaRegion)*(piG/180.)),weight);
    

    //dP

    pdPt_vs_ptJTransMinRECO->Fill(PTLeadingTJ,transP1/(60.*(2.*etaRegion)*(piG/180.)),weight);
    
    pdPt_vs_ptJTransMaxRECO->Fill(PTLeadingTJ,transP2/(60.*(2.*etaRegion)*(piG/180.)),weight);

    pdPt_vs_ptCJTransMinRECO->Fill(PTLeadingCJ,(transP1*cc->correctionPtTrans(PTLeadingTJ,tkpt))/(60.*(2.*etaRegion)*(piG/180.)),weight);
    
    pdPt_vs_ptCJTransMaxRECO->Fill(PTLeadingCJ,(transP2*cc->correctionPtTrans(PTLeadingTJ,tkpt))/(60.*(2.*etaRegion)*(piG/180.)),weight);

    
    //pdPt_vs_ptCJTransMinRECO->Fill(PTLeadingCJ,(transP1)/(60.*(2.*etaRegion)*(piG/180.)),weight);
    
    //pdPt_vs_ptCJTransMaxRECO->Fill(PTLeadingCJ,(transP2)/(60.*(2.*etaRegion)*(piG/180.)),weight);
    
  }
  */
  temp1RECO->Reset();
  temp2RECO->Reset();

}

void UEAnalysisUE::writeToFile(TFile * file){
  file->Write();
}
