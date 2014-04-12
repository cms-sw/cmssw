#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"
#include "RecoParticleFlow/PFClusterTools/interface/PFPhotonClusters.h"

#include <TMath.h>
#include <TVector2.h>
using namespace reco;
PFPhotonClusters::PFPhotonClusters(PFClusterRef PFClusterRef):
  PFClusterRef_(PFClusterRef)
{
  if(PFClusterRef_->layer()==PFLayer:: ECAL_BARREL )isEB_=true;
  else isEB_=false;
  SetSeed();
  PFCrystalCoor();
  for(int i=0; i<5; ++i)
    for(int j=0; j<5; ++j)e5x5_[i][j]=0;
  FillClusterShape();
  FillClusterWidth();
}

void PFPhotonClusters::SetSeed(){
  double PFSeedE=0;
  math::XYZVector axis;
  math::XYZVector position;
  DetId idseed;
  const std::vector< reco::PFRecHitFraction >& PFRecHits=
    PFClusterRef_->recHitFractions();
  
  for(std::vector< reco::PFRecHitFraction >::const_iterator it = PFRecHits.begin();
      it != PFRecHits.end(); ++it){
    const PFRecHitRef& RefPFRecHit = it->recHitRef();
    double frac=it->fraction();
    float E= RefPFRecHit->energy()* frac;
    if(E>PFSeedE){
      PFSeedE=E;  
      axis=RefPFRecHit->getAxisXYZ();
      position=RefPFRecHit->position();
      idseed = RefPFRecHit->detId();
    }
  }
  idseed_=idseed;
  seedPosition_=position;
  seedAxis_=axis;
}

void PFPhotonClusters::PFCrystalCoor(){
  if(PFClusterRef_->layer()==PFLayer:: ECAL_BARREL ){//is Barrel
    isEB_=true;
    EBDetId EBidSeed=EBDetId(idseed_.rawId());
    CrysIEta_=EBidSeed.ieta();
    CrysIPhi_=EBidSeed.iphi();
    double depth = PFClusterRef_->getDepthCorrection(PFClusterRef_->energy(), false, false);
    math::XYZVector center_pos = seedPosition_+depth*seedAxis_;
    //Crystal Coordinates:
    double Pi=TMath::Pi();
    float Phi=PFClusterRef_->position().phi(); 
    double Theta = -(PFClusterRef_->position().theta())+0.5* Pi;
    double PhiCentr = TVector2::Phi_mpi_pi(center_pos.phi());
    double PhiWidth = (Pi/180.);
    double PhiCry = (TVector2::Phi_mpi_pi(Phi-PhiCentr))/PhiWidth;
    double ThetaCentr = -center_pos.theta()+0.5*Pi;
    double ThetaWidth = (Pi/180.)*cos(ThetaCentr);
    
    double EtaCry = (Theta-ThetaCentr)/ThetaWidth; 
    CrysEta_=EtaCry;
    CrysPhi_=PhiCry;
    
    if(abs(CrysIEta_)==1 || abs(CrysIEta_)==2 )
      CrysIEtaCrack_=abs(CrysIEta_);
    if(abs(CrysIEta_)>2 && abs(CrysIEta_)<24)
      CrysIEtaCrack_=3;
    if(abs(CrysIEta_)==24)
      CrysIEtaCrack_=4;
    if(abs(CrysIEta_)==25)
      CrysIEtaCrack_=5;
    if(abs(CrysIEta_)==26)
      CrysIEtaCrack_=6;
    if(abs(CrysIEta_)==27)
      CrysIEtaCrack_=7;
    if(abs(CrysIEta_)>27 &&  abs(CrysIEta_)<44)
      CrysIEtaCrack_=8;
    if(abs(CrysIEta_)==44)
      CrysIEtaCrack_=9;
    if(abs(CrysIEta_)==45)
      CrysIEtaCrack_=10;
    if(abs(CrysIEta_)==46)
      CrysIEtaCrack_=11;
    if(abs(CrysIEta_)==47)
      CrysIEtaCrack_=12;
    if(abs(CrysIEta_)>47 &&  abs(CrysIEta_)<64)
      CrysIEtaCrack_=13;
    if(abs(CrysIEta_)==64)
      CrysIEtaCrack_=14;
    if(abs(CrysIEta_)==65)
	CrysIEtaCrack_=15;
    if(abs(CrysIEta_)==66)
      CrysIEtaCrack_=16;
    if(abs(CrysIEta_)==67)
      CrysIEtaCrack_=17;
    if(abs(CrysIEta_)>67 &&  abs(CrysIEta_)<84)
      CrysIEtaCrack_=18;
    if(abs(CrysIEta_)==84)
      CrysIEtaCrack_=19;
    if(abs(CrysIEta_)==85)
      CrysIEtaCrack_=20;
  }
  else{
    isEB_=false;
    EEDetId EEidSeed=EEDetId(idseed_.rawId());
    CrysIX_=EEidSeed.ix();
    CrysIY_=EEidSeed.iy();
    float X0 = 0.89; float T0 = 1.2;
    if(fabs(PFClusterRef_->eta())>1.653)T0=3.1;
    double depth = X0 * (T0 + log(PFClusterRef_->energy()));
    math::XYZVector center_pos=(seedPosition_)+depth*seedAxis_;
    double XCentr = center_pos.x();
    double YCentr = center_pos.y();
    double XWidth = 2.59;
    double YWidth = 2.59;
    
    CrysX_=(PFClusterRef_->x()-XCentr)/XWidth;
    CrysY_=(PFClusterRef_->y()-YCentr)/YWidth;
  }
}

void PFPhotonClusters::FillClusterShape(){
  const std::vector< reco::PFRecHitFraction >& PFRecHits=PFClusterRef_->recHitFractions();  
  for(std::vector< reco::PFRecHitFraction >::const_iterator it = PFRecHits.begin(); it != PFRecHits.end(); ++it){
    const PFRecHitRef& RefPFRecHit = it->recHitRef();
    DetId id=RefPFRecHit->detId();
    double frac=it->fraction();
    float E=RefPFRecHit->energy()*frac;
    if(isEB_){	
      int deta=EBDetId::distanceEta(id,idseed_);
      int dphi=EBDetId::distancePhi(id,idseed_);

      if(abs(deta)>2 ||abs(dphi)>2)continue;
      
      //f(abs(dphi)<=2 && abs(deta)<=2){
      EBDetId EBidSeed=EBDetId(idseed_.rawId());
      EBDetId EBid=EBDetId(id.rawId());
      int ind1=EBidSeed.ieta()-EBid.ieta();
      int ind2=EBidSeed.iphi()-EBid.iphi();
      if(EBidSeed.ieta() * EBid.ieta() > 0){
	ind1=EBid.ieta()-EBidSeed.ieta();
      }
      else{ //near EB+ EB-
	int shift(EBidSeed.ieta()>0 ? -1 : 1);
	ind1=EBidSeed.ieta()-EBid.ieta()+shift; 
      }

      // more tricky edges in phi. Note that distance is already <2 at this point 
      if( (EBidSeed.iphi()<5&&EBid.iphi()>355) || (EBidSeed.iphi()>355&&EBid.iphi()<5)) {
	int shift(EBidSeed.iphi()<180 ? EBDetId::MAX_IPHI:-EBDetId::MAX_IPHI) ; 
	ind2 = shift + EBidSeed.iphi() - EBid.iphi();
	//	std::cout << " Phi check " << EBidSeed.iphi() << " " <<  EBid.iphi() << " " << ind2 << std::endl;
      }

      int iEta=ind1+2;
      int iPhi=ind2+2;
      //std::cout<<"IEta, IPhi "<<iEta<<", "<<iPhi<<std::endl;
//      if(iPhi >= 5 || iPhi <0) { 
//	std::cout << "dphi "<< EBDetId::distancePhi(id,idseed_) << " iphi " << EBid.iphi() << " iphiseed " << EBidSeed.iphi() << " iPhi " << iPhi << std::endl;}
//      if(iEta >= 5 || iEta <0) { 
//	std::cout << "deta "<< EBDetId::distanceEta(id,idseed_) << " ieta " << EBid.ieta() << " ietaseed " << EBidSeed.ieta() << "ind1 " << ind1 << " iEta " << iEta << " " ;
//	ind1=ind1prime;
//	iEta=ind1+2;
//	std::cout << " new iEta " << iEta << std::endl;
//      }
//      assert(iEta < 5);
//      assert(iEta >= 0);
//      assert(iPhi < 5);
//      assert(iPhi >= 0);
      e5x5_[iEta][iPhi]=E;
    }
    else{
      int dx=EEDetId::distanceX(id,idseed_);
      int dy=EEDetId::distanceY(id,idseed_);
      if(abs(dx)>2 ||abs(dy>2))continue;
      EEDetId EEidSeed=EEDetId(idseed_.rawId());
      EEDetId EEid=EEDetId(id.rawId());
      int ind1=EEid.ix()-EEidSeed.ix();
      int ind2=EEid.iy()-EEidSeed.iy();
      int ix=ind1+2;
      int iy=ind2+2;
      //std::cout<<"IX, IY "<<ix<<", "<<iy<<std::endl;	    
//      assert(ix < 5);
//      assert(ix >= 0);
//      assert(iy < 5);
//      assert(iy >= 0);
      e5x5_[ix][iy]=E;
    }
  }
}

void PFPhotonClusters::FillClusterWidth(){
  double numeratorEtaWidth = 0.;
  double numeratorPhiWidth = 0.;
  double numeratorEtaPhiWidth = 0.;
  double ClustEta=PFClusterRef_->eta();
  double ClustPhi=PFClusterRef_->phi();
  const std::vector< reco::PFRecHitFraction >& PFRecHits=PFClusterRef_->recHitFractions();  
  for(std::vector< reco::PFRecHitFraction >::const_iterator it = PFRecHits.begin(); it != PFRecHits.end(); ++it){
    const PFRecHitRef& RefPFRecHit = it->recHitRef();  
    float E=RefPFRecHit->energy() * it->fraction();
    double dEta = RefPFRecHit->position().eta() - ClustEta;	
    double dPhi = RefPFRecHit->position().phi() - ClustPhi;
    if (dPhi > + TMath::Pi()) { dPhi = TMath::TwoPi() - dPhi; }
    if (dPhi < - TMath::Pi()) { dPhi = TMath::TwoPi() + dPhi; }
    numeratorEtaWidth += E * dEta * dEta;
    numeratorPhiWidth += E * dPhi * dPhi;
    numeratorEtaPhiWidth += E * fabs(dPhi) * fabs(dEta);
  }
  double denominator=PFClusterRef_->energy();
  sigetaeta_ = sqrt(numeratorEtaWidth / denominator);
  sigphiphi_ = sqrt(numeratorPhiWidth / denominator);
  sigetaphi_ = sqrt(numeratorEtaPhiWidth / denominator);
}
