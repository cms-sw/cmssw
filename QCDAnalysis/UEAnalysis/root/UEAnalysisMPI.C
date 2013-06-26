#include "UEAnalysisMPI.h"
#include <vector>
#include <math.h>

typedef std::pair<TLorentzVector*,TLorentzVector*> AssociatedObject;

UEAnalysisMPI::UEAnalysisMPI()
{
  std::cout << "UEAnalysisMPI constructor " <<std::endl;
  piG = acos(-1.);
  rangePhi = acos(-1.)/180*50;
}

void UEAnalysisMPI::Begin(TFile * f){
  f->cd();
  
  //MPI Analysis

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

}

void UEAnalysisMPI::mpiAnalysisMC(float weight,float etaRegion,float ptThreshold, TClonesArray* ChargedJet)
{
  std::vector<TLorentzVector*> JetMC;
  JetMC.clear();
  
  for(int j=0;j<ChargedJet->GetSize();++j){
    TLorentzVector *v = (TLorentzVector*)ChargedJet->At(j);
    if(fabs(v->Eta())<etaRegion){
      JetMC.push_back(v);
    }
  }
  
  std::vector<AssociatedObject> assoJetMC;
  assoJetMC.clear();

  while(JetMC.size()>1){
    int oldSize = JetMC.size();
    std::vector<TLorentzVector*>::iterator itH = JetMC.begin();
    if((*itH)->Pt()>=ptThreshold){
      for(std::vector<TLorentzVector*>::iterator it=JetMC.begin();it!=JetMC.end();it++){
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
}

void UEAnalysisMPI::mpiAnalysisRECO(float weight,float etaRegion,float ptThreshold,TClonesArray * TracksJet)
{
  std::vector<TLorentzVector*> JetRECO;
  JetRECO.clear();
  
  for(int j=0;j<TracksJet->GetSize();++j){
    TLorentzVector *v = (TLorentzVector*)TracksJet->At(j);
    if(fabs(v->Eta())<etaRegion){
      JetRECO.push_back(v);
    }
  }
  
  std::vector<AssociatedObject> assoJetRECO;
  assoJetRECO.clear();

  while(JetRECO.size()>1){
    int oldSize = JetRECO.size();
    std::vector<TLorentzVector*>::iterator itH = JetRECO.begin();
    if((*itH)->Pt()>=ptThreshold){
      for(std::vector<TLorentzVector*>::iterator it=JetRECO.begin();it!=JetRECO.end();it++){
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
    
    const TLorentzVector* leadingJet((*at).first);
    const TLorentzVector* secondJet((*at).second);

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

void UEAnalysisMPI::writeToFile(TFile * file){
  file->Write();
}
