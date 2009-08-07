#include "UEAnalysisGAM.h"
#include <vector>
#include <math.h>

typedef pair<TLorentzVector*,TLorentzVector*> AssociatedObject;

UEAnalysisGAM::UEAnalysisGAM()
{
  cout << "UEAnalysisGAM constructor " <<endl;
  piG = acos(-1.);
  rangePhi = acos(-1.)/180*50;
}

void UEAnalysisGAM::Begin(TFile * f){
  f->cd();
  
  fPtLeadingGammaMC = new TH1D("PtLeadingGamma","Pt of Leading Gamma",300,0,300);
  fPhiLeadingGammaMC = new TH1D("PhiLeadingGamma","Phi of Leading Gamma",100,-5,5);
  fEtaLeadingGammaMC = new TH1D("EtaLeadingGamma","Eta of Leading Gamma",100,-5,5);
  fdPhiGamma1JetMC = new TH1D("fdPhiGamma1Jet", "dPhi gamma and 1st Jet",100,0,180);
  fdPhiGamma2JetMC = new TH1D("fdPhiGamma2Jet", "dPhi gamma and 2nd Jet",100,0,180);
  fdPhiGamma3JetMC = new TH1D("fdPhiGamma3Jet", "dPhi gamma and 3rd Jet",100,0,180);

  fNumbMPIMC             = new TH1D("NumbMPIMC","Number of MPI",20,0.5,20.5);
  fdEtaLeadingPairMC     = new TH1D("dEtaLeadingPairMC","#Delta #eta Jet in the pair",100,-5,5);
  fdPhiLeadingPairMC     = new TH1D("dPhiLeadingPairMC","#Delta #phi Jet in the pair",40,120,200);
  fptRatioLeadingPairMC  = new TH1D("ptRatioLeadingPairMC","P_{T}^{2^{o} Jet}/P_{T}^{1^{o} Jet}",100,0,1.2);
  pPtRatio_vs_PtJleadMC  = new TProfile("PtRation_vs_PtJleadMC","P_{T}^{2^{o} Jet}/P_{T}^{1^{o} Jet} vs P_{T}^{1^{o} Jet}", 100,0.,50.);
  pPtRatio_vs_EtaJleadMC = new TProfile("PtRation_vs_EtaJleadMC","P_{T}^{2^{o} Jet}/P_{T}^{1^{o} Jet} vs #eta^{1^{o} Jet}", 100,0.,5.);
  pPtRatio_vs_PhiJleadMC = new TProfile("PtRation_vs_PhiJleadMC","P_{T}^{2^{o} Jet}/P_{T}^{1^{o} Jet} vs #phi^{1^{o} Jet}", 101,-4.,4.);

}


void UEAnalysisGAM::gammaAnalysisMC(Float_t weight,Float_t etaRegion,Float_t ptThreshold, TClonesArray& MCGamma, TClonesArray& ChargedJet)
{
  
  vector<TLorentzVector*> JetMC;
  vector<TLorentzVector*> GamMC;
  GamMC.clear();
  JetMC.clear();

  for(int j=0;j<MCGamma.GetSize();++j){
        TLorentzVector *g = (TLorentzVector*)MCGamma.At(j);
        if(fabs(g->Eta())<etaRegion){
	  GamMC.push_back(g);
	  if(GamMC.size()==1) JetMC.push_back(g);
	}
   }

  if(JetMC.size() != 0){
    for(int j=0;j<ChargedJet.GetSize();++j){
      TLorentzVector *w = (TLorentzVector*)ChargedJet.At(j);
      if(fabs(w->Eta())<etaRegion){
        JetMC.push_back(w);
      }
    }
    if(JetMC.size()>=2){
      float dPhiJet1 = fabs(JetMC[0]->Phi()-JetMC[1]->Phi());    
      if(dPhiJet1> piG) dPhiJet1 = 2*piG -dPhiJet1;
      dPhiJet1 = (180*dPhiJet1)/piG;
      fdPhiGamma1JetMC->Fill(dPhiJet1);
    }
    if(JetMC.size()>=3){
      float dPhiJet2 = fabs(JetMC[0]->Phi()-JetMC[2]->Phi());    
      if(dPhiJet2> piG) dPhiJet2 = 2*piG -dPhiJet2;
      dPhiJet2 = (180*dPhiJet2)/piG;
      fdPhiGamma2JetMC->Fill(dPhiJet2);
    }
    if(JetMC.size()>=4){
      float dPhiJet3 = fabs(JetMC[0]->Phi()-JetMC[3]->Phi());    
      if(dPhiJet3> piG) dPhiJet3 = 2*piG -dPhiJet3;
      dPhiJet3 = (180*dPhiJet3)/piG;
      fdPhiGamma3JetMC->Fill(dPhiJet3);
    }
 

    vector<AssociatedObject> assoJetMC;
    assoJetMC.clear();

    while(JetMC.size()>1){
      int oldSize = JetMC.size();
      vector<TLorentzVector*>::iterator itH = JetMC.begin();
      if((*itH)->Pt()>=ptThreshold){
	for(vector<TLorentzVector*>::iterator it=JetMC.begin();it!=JetMC.end();it++){
	  float azimuthDistanceJet = fabs( (*itH)->Phi() - (*it)->Phi() );
	  if((*it)->Pt()/(*itH)->Pt()>=0.3){
	    if( (piG - rangePhi) <  azimuthDistanceJet && azimuthDistanceJet < (piG + rangePhi)){
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
      vector<AssociatedObject>::iterator at= assoJetMC.begin();
    
      const TLorentzVector* leadingJet((*at).first);
      const TLorentzVector* secondJet((*at).second);

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
    
    fPhiLeadingGammaMC->Fill(GamMC[0]->Phi());
    fPtLeadingGammaMC->Fill(GamMC[0]->Pt());
    fEtaLeadingGammaMC->Fill(GamMC[0]->Eta());
  }

}


void UEAnalysisGAM::writeToFile(TFile * file){
  file->Write();
}
