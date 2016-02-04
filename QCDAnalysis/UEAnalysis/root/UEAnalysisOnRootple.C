#define UEAnalysisOnRootple_cxx
#include "UEAnalysisOnRootple.h"
#include <TH2.h>
#include <TStyle.h>
#include <TCanvas.h>
#include <TVector3.h>

#include <vector>
#include <math.h>

//
#include <TClonesArray.h>
#include <TObjString.h>
//

typedef std::pair<TVector3*,TVector3*> AssociatedObject;

void UEAnalysisOnRootple::MultiAnalysis(char* filelist,char* outname,Float_t weight[7],Float_t eta,
					  Float_t triggerPt,std::string type,std::string trigger,std::string tkpt,Float_t ptCut)
{
  BeginJob(outname);
  etaRegion = eta;
  ptThreshold = ptCut/1000.;
  char RootTupleName[255];
  char RootListFileName[255];
  strcpy(RootListFileName,filelist);
  ifstream inFile(RootListFileName);
  int filenumber = 0;
  while(inFile.getline(RootTupleName,255)) {
    if (RootTupleName[0] != '#') {
      std::cout<<"I'm analyzing file "<<RootTupleName<<std::endl;

      //TFile *f =  new TFile(RootTupleName);
      f = TFile::Open(RootTupleName);

      // TFileService puts UEAnalysisTree in a directory named after the module
      // which called the EDAnalyzer
      f->cd("ueAnalysisRootple");

      TTree * tree = (TTree*)gDirectory->Get("AnalysisTree");
      Init(tree);

      Loop(weight[filenumber],triggerPt,type,trigger,tkpt);
    
      f->Close();
    
    } else {
      if (RootTupleName[1] == '#') break;     
    }
    filenumber++;
  }

  EndJob();

}


void UEAnalysisOnRootple::Loop(Float_t we,Float_t triggerPt,std::string type,std::string trigger,std::string tkpt)
{
  if (fChain == 0) 
    {
      std::cout << "fChain == 0 return." << std::endl;
      return;
    }

  Long64_t nentries = fChain->GetEntriesFast();

  std::cout << "number of entries: " << nentries << std::endl;

  
  Long64_t nbytes = 0, nb = 0;
  for (Long64_t jentry=0; jentry<nentries;jentry++) {
    Long64_t ientry = LoadTree(jentry);
    if (ientry < 0) break;
    nb = fChain->GetEntry(jentry);   nbytes += nb;


//     int nAcceptedTriggers( acceptedTriggers->GetSize() );
//     if (nAcceptedTriggers) std::cout << std::endl << "Event has been accepted by " << acceptedTriggers->GetSize() << std::endl;
//     for ( int iAcceptedTrigger(0); iAcceptedTrigger<nAcceptedTriggers; ++iAcceptedTrigger )
//       {
// 	std::cout << "\t(" << iAcceptedTrigger << ") trigger path ";
// 	std::cout << (acceptedTriggers->At(iAcceptedTrigger))->GetName() << std::endl;
//       } 

    if(type=="Jet"){
      if(trigger=="MB"){
	if( EventKind != 92 && EventKind != 93 && EventKind != 94 ){
	  JetCalibAnalysis(we,tkpt);
	}
      }else{
	if(TrasverseMomentumEHJ[0]>=triggerPt)
	  JetCalibAnalysis(we,tkpt);
      }
    }
    if(type=="MPI"){
      if(trigger=="MB"){
	if( EventKind != 92 && EventKind != 93 && EventKind != 94 ){
	  MPIAnalysisMC(we,tkpt);
	  MPIAnalysisRECO(we,tkpt);
	}
      }else{
	if(TrasverseMomentumEHJ[0]>=triggerPt){
	  MPIAnalysisMC(we,tkpt);
	  MPIAnalysisRECO(we,tkpt);
	}
      }
     }
    if(type=="UE"){
      if(trigger=="MB"){
	if( EventKind != 92 && EventKind != 93 && EventKind != 94 ){
	  UEAnalysisMC(we,tkpt);
	  UEAnalysisRECO(we,tkpt);
	}
      }else{
	if(TrasverseMomentumEHJ[0]>=triggerPt){
	  UEAnalysisMC(we,tkpt);
	  UEAnalysisRECO(we,tkpt);
	}
      }
    }
  }
}

void UEAnalysisOnRootple::UEAnalysisMC(Float_t weight,std::string tkpt)
{

  for(int i=0;i<NumberMCParticles;i++){
    if(TransverseMomentumMC[i]>=ptThreshold){
      fHistPtDistMC->Fill(TransverseMomentumMC[i],weight);
      fHistEtaDistMC->Fill(EtaMC[i],weight);
      fHistPhiDistMC->Fill(PhiMC[i],weight);
      temp3MC->Fill(fabs(EtaMC[i]));
      temp4MC->Fill(fabs(TransverseMomentumMC[i]));
    }
  }
  
  
  for(int i=0;i<100;i++){
    pdN_vs_etaMC->Fill((i*0.05)+0.025,temp3MC->GetBinContent(i+1)/0.1,weight);
  }
  for(int i=0;i<1000;i++){
    pdN_vs_ptMC->Fill((i*0.1)+0.05,temp4MC->GetBinContent(i+1)/0.1,weight);
  }
  
  temp3MC->Reset();
  temp4MC->Reset();
  
  // get 3-vector of jet 
  TVector3 * jetvector = new TVector3;
  Float_t PTLeadingCJ = -10;
  for(int j=0;j<NumberChargedJet;j++){
    if(fabs(EtaCJ[j])<etaRegion){
      jetvector->SetPtEtaPhi(TrasverseMomentumCJ[j], EtaCJ[j], PhiCJ[j]);
      PTLeadingCJ= TrasverseMomentumCJ[j];
      break;
    }
  }
  
  for(int i=0;i<NumberMCParticles;i++){
    if(fabs(EtaMC[i])<etaRegion && TransverseMomentumMC[i]>=ptThreshold){
      // get 3-vector of particle                                                                                             
      TVector3 * particlevector = new TVector3;
      particlevector->SetPtEtaPhi(TransverseMomentumMC[i], EtaMC[i], PhiMC[i]);
      
      // use ROOT method to calculate dphi                                                                                    
      // convert dphi from radiants to degrees                                                                                
      Float_t conv = 180/piG;
      Float_t Dphi_mc = conv * jetvector->DeltaPhi(*particlevector);
      
      temp1MC->Fill(Dphi_mc);
      temp2MC->Fill(Dphi_mc,TransverseMomentumMC[i]);
    }
  }
  
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
  
  bool orderedN = false;
  bool orderedP = false;
  
  pdN_vs_ptJTowardMC->Fill(PTLeadingCJ,(towardN)/(120.*(2*etaRegion)*(piG/180.)),weight);
  
  pdPt_vs_ptJTowardMC->Fill(PTLeadingCJ,(towardP)/(120.*(2*etaRegion)*(piG/180.)),weight);
  
  pdN_vs_ptJAwayMC->Fill(PTLeadingCJ,(awayN)/(120.*(2*etaRegion)*(piG/180.)),weight);
  
  pdPt_vs_ptJAwayMC->Fill(PTLeadingCJ,(awayP)/(120.*(2*etaRegion)*(piG/180.)),weight);
  
  if( transN1>=transN2 ) orderedN = true;
  if( transP1>=transP2 ) orderedP = true;

  // add histo for ue fluctuation
  h2d_dN_vs_ptJTransMC->Fill(PTLeadingCJ,(transN1+transN2)/(120.*(2*etaRegion)*(piG/180.)),weight);

  pdN_vs_ptJTransMC->Fill(PTLeadingCJ,(transN1+transN2)/(120.*(2*etaRegion)*(piG/180.)),weight);
  
  pdPt_vs_ptJTransMC->Fill(PTLeadingCJ,(transP1+transP2)/(120.*(2*etaRegion)*(piG/180.)),weight);
  
  if(orderedN){
    pdN_vs_ptJTransMinMC->Fill(PTLeadingCJ,transN2/(60.*(2*etaRegion)*(piG/180.)),weight);
    
    pdN_vs_ptJTransMaxMC->Fill(PTLeadingCJ,transN1/(60.*(2*etaRegion)*(piG/180.)),weight);
  }else{
    pdN_vs_ptJTransMinMC->Fill(PTLeadingCJ,transN1/(60.*(2.*etaRegion)*(piG/180.)),weight);

    pdN_vs_ptJTransMaxMC->Fill(PTLeadingCJ,transN2/(60.*(2.*etaRegion)*(piG/180.)),weight);
  }
  
  if(orderedP){
    pdPt_vs_ptJTransMinMC->Fill(PTLeadingCJ,transP2/(60.*(2.*etaRegion)*(piG/180.)),weight);
    
    pdPt_vs_ptJTransMaxMC->Fill(PTLeadingCJ,transP1/(60.*(2.*etaRegion)*(piG/180.)),weight);
  }else{
    pdPt_vs_ptJTransMinMC->Fill(PTLeadingCJ,transP1/(60.*(2.*etaRegion)*(piG/180.)),weight);
    
    pdPt_vs_ptJTransMaxMC->Fill(PTLeadingCJ,transP2/(60.*(2.*etaRegion)*(piG/180.)),weight);
  }
  temp1MC->Reset();
  temp2MC->Reset();

}

void UEAnalysisOnRootple::UEAnalysisRECO(Float_t weight,std::string tkpt)
{
  
  for(int i=0;i<NumberTracks;i++){
    if(TrasverseMomentumTK[i]>ptThreshold){
      fHistPtDistRECO->Fill(TrasverseMomentumTK[i],weight);
      fHistEtaDistRECO->Fill(EtaTK[i],weight);
      fHistPhiDistRECO->Fill(PhiTK[i],weight);
      temp3RECO->Fill(fabs(EtaTK[i]));
      temp4RECO->Fill(fabs(TrasverseMomentumTK[i]));
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
  
  // get 3-vector of jet 
  TVector3 * jetvector = new TVector3;
  Float_t PTLeadingTJ = -10;
  for(int j=0;j<NumberTracksJet;j++){
    if(fabs(EtaTJ[j])<etaRegion){
      jetvector->SetPtEtaPhi(TrasverseMomentumTJ[j], EtaTJ[j], PhiTJ[j]);
      PTLeadingTJ= TrasverseMomentumTJ[j];
      break;
    }
  }

  Float_t PTLeadingCJ = CalibrationPt(PTLeadingTJ,tkpt)*PTLeadingTJ;
  
  /*
  Float_t PTLeadingCJ = -10;
  if(NumberChargedJet>0)
    PTLeadingCJ=TrasverseMomentumCJ[0];
  */

  for(int i=0;i<NumberTracks;i++){
    // get 3-vector of particle                                                                                             
    if(fabs(EtaTK[i])<etaRegion&&TrasverseMomentumTK[i]>=ptThreshold){
      TVector3 * particlevector = new TVector3;
      particlevector->SetPtEtaPhi(TrasverseMomentumTK[i], EtaTK[i], PhiTK[i]);
      
      // use ROOT method to calculate dphi                                                                                    
      // convert dphi from radiants to degrees                                                                                
      Float_t conv = 180/piG;
      Float_t Dphi_reco = conv * jetvector->DeltaPhi(*particlevector);
      
      temp1RECO->Fill(Dphi_reco);
      temp2RECO->Fill(Dphi_reco,TrasverseMomentumTK[i]);
    }
  }
  
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
  bool orderedP = false;
  
  pdN_vs_ptJTowardRECO->Fill(PTLeadingTJ,(towardN)/(120.*(2*etaRegion)*(piG/180.)),weight);
  
  pdPt_vs_ptJTowardRECO->Fill(PTLeadingTJ,(towardP)/(120.*(2*etaRegion)*(piG/180.)),weight);
  
  pdN_vs_ptJAwayRECO->Fill(PTLeadingTJ,(awayN)/(120.*(2*etaRegion)*(piG/180.)),weight);
  
  pdPt_vs_ptJAwayRECO->Fill(PTLeadingTJ,(awayP)/(120.*(2*etaRegion)*(piG/180.)),weight);
  
  pdN_vs_ptCJTowardRECO->Fill(PTLeadingCJ,(towardN*CorrectionNToward(PTLeadingTJ,tkpt))/(120.*(2*etaRegion)*(piG/180.)),weight);
  
  pdPt_vs_ptCJTowardRECO->Fill(PTLeadingCJ,(towardP*CorrectionPtToward(PTLeadingTJ,tkpt))/(120.*(2*etaRegion)*(piG/180.)),weight);
  
  pdN_vs_ptCJAwayRECO->Fill(PTLeadingCJ,(awayN*CorrectionNAway(PTLeadingTJ,tkpt))/(120.*(2*etaRegion)*(piG/180.)),weight);
  
  pdPt_vs_ptCJAwayRECO->Fill(PTLeadingCJ,(awayP*CorrectionPtAway(PTLeadingTJ,tkpt))/(120.*(2*etaRegion)*(piG/180.)),weight);

  /*
  pdN_vs_ptCJTowardRECO->Fill(PTLeadingCJ,(towardN)/(120.*(2*etaRegion)*(piG/180.)),weight);
  
  pdPt_vs_ptCJTowardRECO->Fill(PTLeadingCJ,(towardP)/(120.*(2*etaRegion)*(piG/180.)),weight);
  
  pdN_vs_ptCJAwayRECO->Fill(PTLeadingCJ,(awayN)/(120.*(2*etaRegion)*(piG/180.)),weight);
  
  pdPt_vs_ptCJAwayRECO->Fill(PTLeadingCJ,(awayP)/(120.*(2*etaRegion)*(piG/180.)),weight);
  */
  
  if( transN1>=transN2 ) orderedN = true;
  if( transP1>=transP2 ) orderedP = true;
  
  pdN_vs_ptJTransRECO->Fill(PTLeadingTJ,(transN1+transN2)/(120.*(2*etaRegion)*(piG/180.)),weight);
  
  pdPt_vs_ptJTransRECO->Fill(PTLeadingTJ,(transP1+transP2)/(120.*(2*etaRegion)*(piG/180.)),weight);

  pdN_vs_ptCJTransRECO->Fill(PTLeadingCJ,((transN1+transN2)*CorrectionNTrans(PTLeadingTJ,tkpt))/(120.*(2*etaRegion)*(piG/180.)),weight);
  
  pdPt_vs_ptCJTransRECO->Fill(PTLeadingCJ,((transP1+transP2)*CorrectionPtTrans(PTLeadingTJ,tkpt))/(120.*(2*etaRegion)*(piG/180.)),weight);

  /*  
  pdN_vs_ptCJTransRECO->Fill(PTLeadingCJ,((transN1+transN2))/(120.*(2*etaRegion)*(piG/180.)),weight);
  
  pdPt_vs_ptCJTransRECO->Fill(PTLeadingCJ,((transP1+transP2))/(120.*(2*etaRegion)*(piG/180.)),weight);
  */

  if(orderedN){

    pdN_vs_ptJTransMinRECO->Fill(PTLeadingTJ,transN2/(60.*(2*etaRegion)*(piG/180.)),weight);
    
    pdN_vs_ptJTransMaxRECO->Fill(PTLeadingTJ,transN1/(60.*(2*etaRegion)*(piG/180.)),weight);

    pdN_vs_ptCJTransMinRECO->Fill(PTLeadingCJ,(transN2*CorrectionNTrans(PTLeadingTJ,tkpt))/(60.*(2*etaRegion)*(piG/180.)),weight);
    
    pdN_vs_ptCJTransMaxRECO->Fill(PTLeadingCJ,(transN1*CorrectionNTrans(PTLeadingTJ,tkpt))/(60.*(2*etaRegion)*(piG/180.)),weight);

    /*
    pdN_vs_ptCJTransMinRECO->Fill(PTLeadingCJ,(transN2)/(60.*(2*etaRegion)*(piG/180.)),weight);
    
    pdN_vs_ptCJTransMaxRECO->Fill(PTLeadingCJ,(transN1)/(60.*(2*etaRegion)*(piG/180.)),weight);
    */

  }else{

    pdN_vs_ptJTransMinRECO->Fill(PTLeadingTJ,transN1/(60.*(2.*etaRegion)*(piG/180.)),weight);

    pdN_vs_ptJTransMaxRECO->Fill(PTLeadingTJ,transN2/(60.*(2.*etaRegion)*(piG/180.)),weight);

    pdN_vs_ptCJTransMinRECO->Fill(PTLeadingCJ,(transN1*CorrectionNTrans(PTLeadingTJ,tkpt))/(60.*(2.*etaRegion)*(piG/180.)),weight);

    pdN_vs_ptCJTransMaxRECO->Fill(PTLeadingCJ,(transN2*CorrectionNTrans(PTLeadingTJ,tkpt))/(60.*(2.*etaRegion)*(piG/180.)),weight);

    /*
    pdN_vs_ptCJTransMinRECO->Fill(PTLeadingCJ,(transN1)/(60.*(2*etaRegion)*(piG/180.)),weight);
    
    pdN_vs_ptCJTransMaxRECO->Fill(PTLeadingCJ,(transN2)/(60.*(2*etaRegion)*(piG/180.)),weight);
    */

  }
  
  if(orderedP){

    pdPt_vs_ptJTransMinRECO->Fill(PTLeadingTJ,transP2/(60.*(2.*etaRegion)*(piG/180.)),weight);
    
    pdPt_vs_ptJTransMaxRECO->Fill(PTLeadingTJ,transP1/(60.*(2.*etaRegion)*(piG/180.)),weight);

    pdPt_vs_ptCJTransMinRECO->Fill(PTLeadingCJ,(transP2*CorrectionPtTrans(PTLeadingTJ,tkpt))/(60.*(2.*etaRegion)*(piG/180.)),weight);
    
    pdPt_vs_ptCJTransMaxRECO->Fill(PTLeadingCJ,(transP1*CorrectionPtTrans(PTLeadingTJ,tkpt))/(60.*(2.*etaRegion)*(piG/180.)),weight);

    /*
    pdPt_vs_ptCJTransMinRECO->Fill(PTLeadingCJ,(transP2)/(60.*(2.*etaRegion)*(piG/180.)),weight);
    
    pdPt_vs_ptCJTransMaxRECO->Fill(PTLeadingCJ,(transP1)/(60.*(2.*etaRegion)*(piG/180.)),weight);
    */

  }else{

    pdPt_vs_ptJTransMinRECO->Fill(PTLeadingTJ,transP1/(60.*(2.*etaRegion)*(piG/180.)),weight);
    
    pdPt_vs_ptJTransMaxRECO->Fill(PTLeadingTJ,transP2/(60.*(2.*etaRegion)*(piG/180.)),weight);

    pdPt_vs_ptCJTransMinRECO->Fill(PTLeadingCJ,(transP1*CorrectionPtTrans(PTLeadingTJ,tkpt))/(60.*(2.*etaRegion)*(piG/180.)),weight);
    
    pdPt_vs_ptCJTransMaxRECO->Fill(PTLeadingCJ,(transP2*CorrectionPtTrans(PTLeadingTJ,tkpt))/(60.*(2.*etaRegion)*(piG/180.)),weight);

    /*
    pdPt_vs_ptCJTransMinRECO->Fill(PTLeadingCJ,(transP1)/(60.*(2.*etaRegion)*(piG/180.)),weight);
    
    pdPt_vs_ptCJTransMaxRECO->Fill(PTLeadingCJ,(transP2)/(60.*(2.*etaRegion)*(piG/180.)),weight);
    */
  }
  temp1RECO->Reset();
  temp2RECO->Reset();

}

void UEAnalysisOnRootple::MPIAnalysisMC(Float_t weight,std::string tkpt)
{
  std::vector<TVector3*> JetMC;
  JetMC.clear();
  
  for(int j=0;j<NumberChargedJet;j++){
    if(fabs(EtaCJ[j])<etaRegion){
      TVector3* jetvector = new TVector3;
      jetvector->SetPtEtaPhi(TrasverseMomentumCJ[j], EtaCJ[j], PhiCJ[j]);
      JetMC.push_back(jetvector);
    }
  }
  
  std::vector<AssociatedObject> assoJetMC;
  assoJetMC.clear();

  while(JetMC.size()>1){
    int oldSize = JetMC.size();
    std::vector<TVector3*>::iterator itH = JetMC.begin();
    if((*itH)->Pt()>=ptThreshold){
      for(std::vector<TVector3*>::iterator it=JetMC.begin();it!=JetMC.end();it++){
	float azimuthDistanceJet = fabs( (*itH)->Phi() - (*it)->Phi() );
	if((*it)->Pt()/(*itH)->Pt()>=0.3){
	  if( (piG - rangePhi) <  azimuthDistanceJet && azimuthDistanceJet < (piG + rangePhi)) {
	    AssociatedObject tmpPair((*itH),(*it));
	    assoJetMC.push_back(tmpPair);
	    JetMC.erase(it);
	    int newSize = oldSize -1;
	    oldSize = newSize;
	    JetMC.resize(newSize);
	    break;
	  }
	}
      }
    }
    JetMC.erase(itH);
    int newSize = oldSize -1;
    JetMC.resize(newSize);
  }
  
  if(assoJetMC.size()){
    fNumbMPIMC->Fill(assoJetMC.size());
    std::vector<AssociatedObject>::iterator at= assoJetMC.begin();
    
    const TVector3* leadingJet((*at).first);
    const TVector3* secondJet((*at).second);

    pPtRatio_vs_PtJleadMC->Fill(leadingJet->Pt(),(secondJet->Pt()/leadingJet->Pt()));
    pPtRatio_vs_EtaJleadMC->Fill(fabs(leadingJet->Eta()),(secondJet->Pt()/leadingJet->Pt()));
    pPtRatio_vs_PhiJleadMC->Fill(leadingJet->Phi(),(secondJet->Pt()/leadingJet->Pt()));
    
    fdEtaLeadingPairMC->Fill(leadingJet->Eta()-secondJet->Eta());
    float dPhiJet = fabs(leadingJet->Phi()-secondJet->Phi());
    if(dPhiJet> piG) dPhiJet = 2*piG -dPhiJet;
    dPhiJet = (180*dPhiJet)/piG;
    fdPhiLeadingPairMC->Fill(dPhiJet);
    fptRatioLeadingPairMC->Fill(secondJet->Pt()/leadingJet->Pt());
  }
}

void UEAnalysisOnRootple::MPIAnalysisRECO(Float_t weight,std::string tkpt)
{
  std::vector<TVector3*> JetRECO;
  JetRECO.clear();
  
  for(int j=0;j<NumberTracksJet;j++){
    if(fabs(EtaCJ[j])<etaRegion){
      TVector3* jetvector = new TVector3;
      //jetvector->SetPtEtaPhi(CalibrationPt(TrasverseMomentumTJ[j],tkpt)*TrasverseMomentumTJ[j], EtaTJ[j], PhiTJ[j]);
      jetvector->SetPtEtaPhi(TrasverseMomentumTJ[j], EtaTJ[j], PhiTJ[j]);
      JetRECO.push_back(jetvector);
    }
  }
  
  std::vector<AssociatedObject> assoJetRECO;
  assoJetRECO.clear();

  while(JetRECO.size()>1){
    int oldSize = JetRECO.size();
    std::vector<TVector3*>::iterator itH = JetRECO.begin();
    if((*itH)->Pt()>=ptThreshold){
      for(std::vector<TVector3*>::iterator it=JetRECO.begin();it!=JetRECO.end();it++){
	float azimuthDistanceJet = fabs( (*itH)->Phi() - (*it)->Phi() );
	if((*it)->Pt()/(*itH)->Pt()>=0.3){
	  if( (piG - rangePhi) <  azimuthDistanceJet && azimuthDistanceJet < (piG + rangePhi)) {
	    AssociatedObject tmpPair((*itH),(*it));
	    assoJetRECO.push_back(tmpPair);
	    JetRECO.erase(it);
	    int newSize = oldSize -1;
	    oldSize = newSize;
	    JetRECO.resize(newSize);
	    break;
	  }
	}
      }
    }
    JetRECO.erase(itH);
    int newSize = oldSize -1;
    JetRECO.resize(newSize);
  }
  
  if(assoJetRECO.size()){
    fNumbMPIRECO->Fill(assoJetRECO.size());
    std::vector<AssociatedObject>::iterator at= assoJetRECO.begin();
    
    const TVector3* leadingJet((*at).first);
    const TVector3* secondJet((*at).second);

    pPtRatio_vs_PtJleadRECO->Fill(leadingJet->Pt(),(secondJet->Pt()/leadingJet->Pt()));
    pPtRatio_vs_EtaJleadRECO->Fill(fabs(leadingJet->Eta()),(secondJet->Pt()/leadingJet->Pt()));
    pPtRatio_vs_PhiJleadRECO->Fill(leadingJet->Phi(),(secondJet->Pt()/leadingJet->Pt()));
    
    fdEtaLeadingPairRECO->Fill(leadingJet->Eta()-secondJet->Eta());
    float dPhiJet = fabs(leadingJet->Phi()-secondJet->Phi());
    if(dPhiJet> piG) dPhiJet = 2*piG -dPhiJet;
    dPhiJet = (180*dPhiJet)/piG;
    fdPhiLeadingPairRECO->Fill(dPhiJet);
    fptRatioLeadingPairRECO->Fill(secondJet->Pt()/leadingJet->Pt());
  }
}

void UEAnalysisOnRootple::JetCalibAnalysis(Float_t weight,std::string tkpt)
{

  if(NumberCaloJet!=0&&NumberChargedJet!=0&&NumberTracksJet!=0&&NumberInclusiveJet!=0){
    
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

    int nIncJet=0;
    for(int i=0;i<NumberInclusiveJet;i++)
      {
	if(fabs(EtaIJ[i])<etaRegion)
	  nIncJet++;
      }
    
    int nChgRECOJet=0;
    for(int i=0;i<NumberTracksJet;i++)
      {
	if(fabs(EtaTJ[i])<etaRegion)
	  nChgRECOJet++;
      }
    
    int nChgMCJet=0;
    for(int i=0;i<NumberChargedJet;i++)
      {
	if(fabs(EtaCJ[i])<etaRegion)
	  nChgMCJet++;
      }
    
    int nCaloJet=0;
    for(int i=0;i<NumberCaloJet;i++)
      {
	if(fabs(EtaEHJ[i])<etaRegion)
	  nCaloJet++;
      }
    
    numb_cal->Fill(nCaloJet);
    numb_chgmc->Fill(nChgMCJet);
    numb_chgreco->Fill(nChgRECOJet);
    numb_inc->Fill(nIncJet);

    if(fabs(EtaIJ[0])<etaRegion && fabs(EtaTJ[0])<etaRegion){
      eta_chgreco_res->Fill((EtaIJ[0]-EtaTJ[0]));
      phi_chgreco_res->Fill((PhiIJ[0]-PhiTJ[0]));
    }

    if(fabs(EtaIJ[0])<etaRegion && fabs(EtaEHJ[0])<etaRegion){
      eta_cal_res->Fill((EtaIJ[0]-EtaEHJ[0]));
      phi_cal_res->Fill((PhiIJ[0]-PhiEHJ[0]));
    }

    for(int i=0;i<NumberInclusiveJet;i++)
      {
	if(fabs(EtaIJ[i])<etaRegion)
	  {
	    etaIJ = EtaIJ[i];
	    ptIJ  = TrasverseMomentumIJ[i];
	    phiIJ = PhiIJ[i];
	    break;
	  }
      }
    
    for(int i=0;i<NumberTracksJet;i++)
      {
	if(fabs(EtaTJ[i])<etaRegion)
	  {
	    etaTJ = EtaTJ[i];
	    ptTJ  = TrasverseMomentumTJ[i];
	    phiTJ = PhiTJ[i];
	    break;
	  }
    }
    
    for(int i=0;i<NumberChargedJet;i++)
      {
	if(fabs(EtaCJ[i])<etaRegion)
	  {
	    etaCJ = EtaCJ[i];
	    ptCJ  = TrasverseMomentumCJ[i];
	    phiCJ = PhiCJ[i];
	    break;
	  }
    }
    
    for(int i=0;i<NumberCaloJet;i++)
      {
	if(fabs(EtaEHJ[i])<etaRegion)
	  {
	    etaEHJ = EtaEHJ[i];
	    ptEHJ  = TrasverseMomentumEHJ[i];
	    phiEHJ = PhiEHJ[i];
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
    }
    
    if(etaTJ!=-666){
      pT_chgreco->Fill(ptTJ,weight);
      eta_chgreco->Fill(etaTJ,weight);
      phi_chgreco->Fill(phiTJ,weight);
    }
    
    if(etaCJ!=-666){
      pT_chgmc->Fill(ptCJ,weight);
      eta_chgmc->Fill(etaCJ,weight);
      phi_chgmc->Fill(phiCJ,weight);
    }
    
    if(etaIJ!=-666){
      pT_inc->Fill(ptIJ,weight);
      eta_inc->Fill(etaIJ,weight);
      phi_inc->Fill(phiIJ,weight);
    }
    
  }

}



void UEAnalysisOnRootple::BeginJob(char* outname)
{
    
  hFile = new TFile(outname, "RECREATE" );
  //Charged Jet caharacterization
  dr_chgcalo = new TH1F("dr_chgcalo","#Delta R Charged RECO vs Calorimeter",100,0.,10.);
  dr_chginc = new TH1F("dr_chginc","#Delta R Charged RECO vs Inclusive",100,0.,10.);
  dr_chgmcreco = new TH1F("dr_chgmcreco","#Delta R Charged RECO vs Charged MC",100,0.,10.);
  dr_caloinc = new TH1F("dr_caloinc","#Delta R Calorimeter vs Inclusive",100,0.,10.);
  numb_cal = new TH1F("numb_cal","Number calo Jet",30,0.,30.);
  pT_cal = new TH1F("pT_cal","P_{T} calo",50,0.,50.);
  eta_cal = new TH1F("eta_cal","#eta Calo",100,-3.,3.);
  eta_cal_res = new TH1F("eta_cal_res","#eta_{calo} - #eta_{inc}",100,-3.,3.);
  phi_cal = new TH1F("phi_cal","#phi Calo",50,-3.14,3.14);
  phi_cal_res = new TH1F("phi_cal_res","#phi_{calo} - #phi_{inc}",100,-3.,3.);
  numb_chgmc = new TH1F("numb_chgmc","Number Charged MC Jet",30,0.,30.);
  pT_chgmc = new TH1F("pT_chgmc","P_{T} Charged MC",50,0.,50.);
  eta_chgmc = new TH1F("eta_chgmc","#eta Charged MC",100,-3.,3.);
  eta_chgmc_res = new TH1F("eta_chgmc_res","#eta_{chg MC} - #eta_{inc}",100,-3.,3.);
  phi_chgmc = new TH1F("phi_chgmc","#phi Charged MC",50,-3.14,3.14);
  phi_chgmc_res = new TH1F("phi_chgmc_res","#phi_{chg MC} - #phi_{inc}",100,-3.,3.);
  numb_chgreco = new TH1F("numb_chgreco","Number Charged RECO Jet",30,0.,30.);
  pT_chgreco = new TH1F("pT_chgreco","P_{T} Charged RECO",50,0.,50.);
  eta_chgreco = new TH1F("eta_chgreco","#eta Charged RECO",100,-3.,3.);
  eta_chgreco_res = new TH1F("eta_chgreco_res","#eta_{chg RECO} - #eta_{inc}",100,-3.,3.);
  phi_chgreco = new TH1F("phi_chgreco","#phi Charged RECO",50,-3.14,3.14);
  phi_chgreco_res = new TH1F("phi_chgreco_res","#phi_{chg RECO} - #phi_{inc}",100,-3.,3.);
  numb_inc = new TH1F("numb_inc","Number Inclusive Jet",30,0.,30.);
  pT_inc = new TH1F("pT_inc","P_{T} Inclusive",50,0.,50.);
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

  fNumbMPIMC             = new TH1D("NumbMPIMC","Number of MPI",20,0.5,20.5);
  fdEtaLeadingPairMC     = new TH1D("dEtaLeadingPairMC","#Delta #eta Jet in the pair",100,-5,5);
  fdPhiLeadingPairMC     = new TH1D("dPhiLeadingPairMC","#Delta #phi Jet in the pair",40,120,200);
  fptRatioLeadingPairMC  = new TH1D("ptRatioLeadingPairMC","P_{T}^{2^{o} Jet}/P_{T}^{1^{o} Jet}",100,0,1.2);
  pPtRatio_vs_PtJleadMC  = new TProfile("PtRation_vs_PtJleadMC","P_{T}^{2^{o} Jet}/P_{T}^{1^{o} Jet} vs P_{T}^{1^{o} Jet}", 100,0.,50.);
  pPtRatio_vs_EtaJleadMC = new TProfile("PtRation_vs_EtaJleadMC","P_{T}^{2^{o} Jet}/P_{T}^{1^{o} Jet} vs #eta^{1^{o} Jet}", 100,0.,5.);
  pPtRatio_vs_PhiJleadMC = new TProfile("PtRation_vs_PhiJleadMC","P_{T}^{2^{o} Jet}/P_{T}^{1^{o} Jet} vs #phi^{1^{o} Jet}", 101,-4.,4.);

  fNumbMPIRECO             = new TH1D("NumbMPIRECO","Number of MPI",20,0.5,20.5);
  fdEtaLeadingPairRECO     = new TH1D("dEtaLeadingPairRECO","#Delta #eta Jet in the pair",100,-5,5);
  fdPhiLeadingPairRECO     = new TH1D("dPhiLeadingPairRECO","#Delta #phi Jet in the pair",40,120,200);
  fptRatioLeadingPairRECO  = new TH1D("ptRatioLeadingPairRECO","P_{T}^{2^{o} Jet}/P_{T}^{1^{o} Jet}",100,0,1.2);
  pPtRatio_vs_PtJleadRECO  = new TProfile("PtRation_vs_PtJleadRECO","P_{T}^{2^{o} Jet}/P_{T}^{1^{o} Jet} vs P_{T}^{1^{o} Jet}", 100,0.,50.);
  pPtRatio_vs_EtaJleadRECO = new TProfile("PtRation_vs_EtaJleadRECO","P_{T}^{2^{o} Jet}/P_{T}^{1^{o} Jet} vs #eta^{1^{o} Jet}", 100,0.,5.);
  pPtRatio_vs_PhiJleadRECO = new TProfile("PtRation_vs_PhiJleadRECO","P_{T}^{2^{o} Jet}/P_{T}^{1^{o} Jet} vs #phi^{1^{o} Jet}", 101,-4.,4.);

  piG = acos(-1.);
  rangePhi = acos(-1.)/180*50;
}

void UEAnalysisOnRootple::EndJob()
{
  hFile->Write();
  hFile->Close();
}

Float_t UEAnalysisOnRootple::CalibrationPt(Float_t ptReco,std::string tkpt){
  if(tkpt=="900"){
    Float_t corr = 0.1122*exp(-(0.2251*ptReco))+1.086-0.0005408*ptReco;
    return  corr;
  }
  if(tkpt=="500"){
    Float_t corr = 0.1389*exp(-(0.2364*ptReco))+1.048-0.0001663*ptReco;
    return  corr;
  }
}

Float_t UEAnalysisOnRootple::CorrectionPtTrans(Float_t ptReco,std::string tkpt){
  if(tkpt=="900"){
    //    Float_t corr = 2.80452*exp(-(0.278432*ptReco))+1.30988-0.000869106*ptReco;
    Float_t corr = 1.214*exp(-(0.9637*ptReco))+1.204-0.0003461*ptReco;
    return  corr;
  }
  if(tkpt=="500"){
    //    Float_t corr = 1.18227*exp(-(0.184019*ptReco))+1.21637-0.000416840*ptReco;
    Float_t corr = 0.4174*exp(-(0.537*ptReco))+1.136-0.0001166*ptReco;
    return  corr;
  }
}

Float_t UEAnalysisOnRootple::CorrectionPtToward(Float_t ptReco,std::string tkpt){
  if(tkpt=="900"){
    /*
    Float_t arg = (ptReco-(-248.6))/-355.7;
    Float_t corr = 1.396*exp(-(0.281*ptReco))-28.13+0.06122*ptReco+37.61*exp(-0.5*arg*arg);
    */
    Float_t corr = 0.1037*exp(-(0.1382*ptReco))+1.117-0.0006322*ptReco;
    return  corr;
  }
  if(tkpt=="500"){
    /*
    Float_t arg = (ptReco-(-73.4))/-35.53;
    Float_t corr = 9.206*exp(-(0.07078*ptReco))+1.196+0.0008953*ptReco-69.16*exp(-0.5*arg*arg);
    */
    Float_t corr = 0.166*exp(-(0.1989*ptReco))+1.073-0.000245*ptReco;
    return  corr;
  }
}

Float_t UEAnalysisOnRootple::CorrectionPtAway(Float_t ptReco,std::string tkpt){
  if(tkpt=="900"){
    /*
    Float_t arg = (ptReco-(-1015))/-1235;
    Float_t corr = 3.635*exp(-(0.4059*ptReco))-53.26+0.03661*ptReco+76.71*exp(-0.5*arg*arg);
    */
    Float_t corr = 0.2707*exp(-(0.2685*ptReco))+1.169-0.000411*ptReco;
    return  corr;
  }
  if(tkpt=="500"){
    /*
    Float_t arg = (ptReco-(-35.38))/-148.1;
    Float_t corr = 1.553*exp(-(0.2515*ptReco))-0.8953+0.009215*ptReco+2.178*exp(-0.5*arg*arg);
    */
    Float_t corr = 0.2835*exp(-(0.2665*ptReco))+1.1-0.0001659*ptReco;
    return  corr;
  }
}

Float_t UEAnalysisOnRootple::CorrectionNTrans(Float_t ptReco,std::string tkpt){
  if(tkpt=="900"){
    //    Float_t corr = 2.41052*exp(-(0.268028*ptReco))+1.26675-0.000509399*ptReco;
    Float_t corr = 1.101*exp(-(0.9939*ptReco))+1.198-0.0001467*ptReco;
    return  corr;
  }
  if(tkpt=="500"){
    //    Float_t corr = 0.970339*exp(-(0.178862*ptReco))+1.19788-0.000293722*ptReco;
    Float_t corr = 0.3322*exp(-(0.445*ptReco))+1.146+0.00002659*ptReco;
    return  corr;
  }
}

Float_t UEAnalysisOnRootple::CorrectionNToward(Float_t ptReco,std::string tkpt){
  if(tkpt=="900"){
    /*
    Float_t arg = (ptReco-(-701.9))/-763.1;
    Float_t corr = 1.366*exp(-(0.288*ptReco))-29.98+0.03649*ptReco+47.78*exp(-0.5*arg*arg);
    */
    Float_t corr = 0.9264*exp(-(1.053*ptReco))+1.16-0.0005176*ptReco;
    return  corr;
  }
  if(tkpt=="500"){
    /*
    Float_t arg = (ptReco-6.429)/-7.393;
    Float_t corr = 1.155*exp(-(0.1146*ptReco))+1.208-0.0005325*ptReco-0.312*exp(-0.5*arg*arg);
    */
    Float_t corr = 0.2066*exp(-(0.3254*ptReco))+1.109-0.00006666*ptReco;
    return  corr;
  }
}

Float_t UEAnalysisOnRootple::CorrectionNAway(Float_t ptReco,std::string tkpt){
  if(tkpt=="900"){
    /*
    Float_t arg = (ptReco-(-1512))/-1517;
    Float_t corr = 3.094*exp(-(0.3886*ptReco))-44.83+0.02965*ptReco+75.99*exp(-0.5*arg*arg);
    */
    Float_t corr = 0.2663*exp(-(0.342*ptReco))+1.178-0.0004006*ptReco;
    return  corr;
  }
  if(tkpt=="500"){
    /*
    Float_t arg = (ptReco-(-170.4))/-281.4;
    Float_t corr = 1.232*exp(-(0.2355*ptReco))-7.733+0.02271*ptReco+10.74*exp(-0.5*arg*arg);
    */
    Float_t corr = 0.316*exp(-(0.3741*ptReco))+1.136-0.0002407*ptReco;
    return  corr;
  }
}
