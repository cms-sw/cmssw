#include "RecoEgamma/EgammaTools/interface/ggPFClusters.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/TruncatedPyramid.h"
#include "TMath.h"
ggPFClusters::ggPFClusters(
			   reco::Photon PFPhoton,
			   edm::Handle<EcalRecHitCollection>& EBReducedRecHits,
			   edm::Handle<EcalRecHitCollection>& EEReducedRecHits,
			   const CaloSubdetectorGeometry* geomBar,
			   const CaloSubdetectorGeometry* geomEnd
			   
			   ):
  EBReducedRecHits_(EBReducedRecHits),
  EEReducedRecHits_(EEReducedRecHits),
  geomBar_(geomBar),
  geomEnd_(geomEnd)
{
  
}


ggPFClusters::~ggPFClusters(){
  
}


vector<reco::CaloCluster> ggPFClusters::getPFClusters(reco::SuperCluster sc){ 
  vector<reco::CaloCluster> PFClust;
  reco::CaloCluster_iterator it=sc.clustersBegin();
  //get PFCluster Position from Basic Clusters, get Energy from RecHits
  for(;it!=sc.clustersEnd();++it){
    std::vector< std::pair<DetId, float> >bcCells=(*it)->hitsAndFractions();
    DetId seedXtalId = bcCells[0].first ;
    int detector = seedXtalId.subdetId(); //use Seed to check if EB or EE
    bool isEb;
    if(detector==1)isEb=true;
    else isEb=false;
    
    float ClusSum=SumPFRecHits(bcCells, isEb); //return PFCluster Energy by Matching to RecHit and using the fractions from the bcCells
    CaloCluster calo(ClusSum, (*it)->position());//store in CaloCluster (parent of PFCluster)
    for(unsigned int i=0; i<bcCells.size(); ++i){
      calo.addHitAndFraction(bcCells[i].first, bcCells[i].second);//Store DetIds and Fractions
    }
    PFClust.push_back(calo);//store in the Vector
  }
  return PFClust;
}

float ggPFClusters::SumPFRecHits(std::vector< std::pair<DetId, float> >& bcCells, bool isEB){
  float ClustSum=0;
  for(unsigned int i=0; i<bcCells.size(); ++i){//loop over the Basic Cluster Cells
    EcalRecHitCollection::const_iterator eb;
    EcalRecHitCollection::const_iterator ee;
    
    if(isEB){
      for(eb=EBReducedRecHits_->begin();eb!= EBReducedRecHits_->end();++eb){//loop over RecHits		  
	if(eb->id().rawId()==bcCells[i].first.rawId()){//match
	  float cellE=eb->energy()* bcCells[i].second;//Energy times fraction
	  ClustSum=ClustSum+cellE;
	}
      }
    }  
    else{
      for(ee=EEReducedRecHits_->begin();ee!= EEReducedRecHits_->end();++ee){//loop over RecHits		  
	if(ee->id().rawId()==bcCells[i].first.rawId()){//match
	  float cellE=ee->energy()* bcCells[i].second;//Energy times fraction
	  ClustSum=ClustSum+cellE;
	  break;
	}
      }
    }
  }    
  return ClustSum;  
}

float ggPFClusters::getPFSuperclusterOverlap(reco::CaloCluster PFClust, reco::Photon phot){
  float overlap=0;
  SuperClusterRef recoSC=phot.superCluster();
  reco::CaloCluster_iterator pit=recoSC->clustersBegin();
  
  std::vector< std::pair<DetId, float> >bcCellsPF=PFClust.hitsAndFractions();
  
  DetId seedXtalId = bcCellsPF[0].first ;
  int detector = seedXtalId.subdetId() ;
  bool isEb;
  std::vector< std::pair<DetId, float> >bcCellsreco;
  if(detector==1)isEb=true;
  else isEb=false;
  for(;pit!=recoSC->clustersEnd();++pit){//fill vector of basic Clusters from SuperCluster
    std::vector< std::pair<DetId, float> >bcCells2=(*pit)->hitsAndFractions();
    for(unsigned int h=0; h<bcCells2.size(); ++h)bcCellsreco.push_back(bcCells2[h]);
    }
  float clustOverlap=0;
  clustOverlap=PFRecHitsSCOverlap(//find overlap of a PFCluster with SuperCluster
				  bcCellsPF, 
				  bcCellsreco,
				  isEb);
  overlap=clustOverlap;
  return overlap;
}

float ggPFClusters::PFRecHitsSCOverlap(
				       std::vector< std::pair<DetId, float> >& bcCells1, 
				       std::vector< std::pair<DetId, float> >& bcCells2,
				       bool isEB){
  float OverlapSum=0;
  multimap <DetId, float> pfDID;
  multimap <DetId, float> recoDID;
  multimap<DetId, float>::iterator pfit;
  multimap<DetId, float>::iterator pit;
  vector<DetId>matched;
  vector<float>matchedfrac;
  //fill Multimap of DetId, fraction for PFCluster
  for(unsigned int i=0; i<bcCells1.size(); ++i){ 
    pfDID.insert(make_pair(bcCells1[i].first, bcCells1[i].second));    
  }
   //fill Multimap of DetId, fraction for SuperCluster
  for(unsigned int i=0; i<bcCells2.size(); ++i){ 
    recoDID.insert(make_pair(bcCells2[i].first, bcCells2[i].second));    
  }
  pit=recoDID.begin();
  pfit=pfDID.begin();
  
  for(; pfit!=pfDID.end(); ++pfit){
    pit=recoDID.find(pfit->first);//match DetId from PFCluster to Supercluster
    if(pit!=recoDID.end()){
      // cout<<"Found Match "<<endl; 
      matched.push_back(pfit->first);//store detId
      matchedfrac.push_back(pfit->second);//store fraction
    }
  }
  
  for(unsigned int m=0; m<matched.size(); ++m){ //loop over matched cells
    DetId det=matched[m];
    EcalRecHitCollection::const_iterator eb;
    EcalRecHitCollection::const_iterator ee;
    if(isEB){
      for(eb=EBReducedRecHits_->begin();eb!= EBReducedRecHits_->end();++eb){		  
	if(eb->id().rawId()==det.rawId()){
	  float cellE=eb->energy()* matchedfrac[m]; //compute overlap  
	  OverlapSum=OverlapSum+cellE;
	  break;
	}
      }
    }
    else{
      for(ee=EEReducedRecHits_->begin();ee!= EEReducedRecHits_->end();++ee){		  
	if(ee->id().rawId()==det.rawId()){
	  float cellE=ee->energy() *  matchedfrac[m];//compute overlap  
	  OverlapSum=OverlapSum+cellE;
	  break;
	}
      }
    }
    
  }
  return OverlapSum;
}

void ggPFClusters::localCoordsEB( reco::CaloCluster clus, float &etacry, float &phicry, int &ieta, int &iphi, float &thetatilt, float &phitilt){
  const math::XYZPoint position_ = clus.position(); 
  double Theta = -position_.theta()+0.5*TMath::Pi();
  double Eta = position_.eta();
  double Phi = TVector2::Phi_mpi_pi(position_.phi());
  
  //Calculate expected depth of the maximum shower from energy (like in PositionCalc::Calculate_Location()):
  // The parameters X0 and T0 are hardcoded here because these values were used to calculate the corrections:
  const float X0 = 0.89; const float T0 = 7.4;
  double depth = X0 * (T0 + log(clus.energy()));
  
  //find max energy crystal
  std::vector< std::pair<DetId, float> > crystals_vector = clus.hitsAndFractions();
  float drmin = 999.;
  EBDetId crystalseed;
  //printf("starting loop over crystals, etot = %5f:\n",clus.energy());
  for (unsigned int icry=0; icry!=crystals_vector.size(); ++icry) {    
    
    EBDetId crystal(crystals_vector[icry].first);
    
    const CaloCellGeometry* cell=geomBar_->getGeometry(crystal);
    GlobalPoint center_pos = (dynamic_cast<const TruncatedPyramid*>(cell))->getPosition(depth);
    double EtaCentr = center_pos.eta();
    double PhiCentr = TVector2::Phi_mpi_pi(center_pos.phi());
    
    float dr = reco::deltaR(Eta,Phi,EtaCentr,PhiCentr);
    if (dr<drmin) {
      drmin = dr;
      crystalseed = crystal;
    }
    
  }
  
  ieta = crystalseed.ieta();
  iphi = crystalseed.iphi();
  
  // Get center cell position from shower depth
  const CaloCellGeometry* cell=geomBar_->getGeometry(crystalseed);
  const TruncatedPyramid *cpyr = dynamic_cast<const TruncatedPyramid*>(cell);
  
  thetatilt = cpyr->getThetaAxis();
  phitilt = cpyr->getPhiAxis();
  
  GlobalPoint center_pos = cpyr->getPosition(depth);
  
  double PhiCentr = TVector2::Phi_mpi_pi(center_pos.phi());
  double PhiWidth = (TMath::Pi()/180.);
  phicry = (TVector2::Phi_mpi_pi(Phi-PhiCentr))/PhiWidth;
  //Some flips to take into account ECAL barrel symmetries:
  if (ieta<0) phicry *= -1.;  
  
  double ThetaCentr = -center_pos.theta()+0.5*TMath::Pi();
  double ThetaWidth = (TMath::Pi()/180.)*TMath::Cos(ThetaCentr);
  etacry = (Theta-ThetaCentr)/ThetaWidth;
  //cout<<"eta, phi raw w/o widths "<<(Theta-ThetaCentr)<<", "<<(TVector2::Phi_mpi_pi(Phi-PhiCentr))<<endl;
  //flip to take into account ECAL barrel symmetries:
  if (ieta<0) etacry *= -1.;  
  return;
  
}

void ggPFClusters::localCoordsEE(reco::CaloCluster clus, float &xcry, float &ycry, int &ix, int &iy, float &thetatilt, float &phitilt){
  const math::XYZPoint position_ = clus.position(); 
  //double Theta = -position_.theta()+0.5*TMath::Pi();
  double Eta = position_.eta();
  double Phi = TVector2::Phi_mpi_pi(position_.phi());
  double X = position_.x();
  double Y = position_.y();
  
  //Calculate expected depth of the maximum shower from energy (like in PositionCalc::Calculate_Location()):
  // The parameters X0 and T0 are hardcoded here because these values were used to calculate the corrections:
  const float X0 = 0.89; float T0 = 1.2;
  //different T0 value if outside of preshower coverage
  if (TMath::Abs(clus.eta())<1.653) T0 = 3.1;
  
  double depth = X0 * (T0 + log(clus.energy()));
  
  //find max energy crystal
  std::vector< std::pair<DetId, float> > crystals_vector = clus.hitsAndFractions();
  float drmin = 999.;
  EEDetId crystalseed;
  //printf("starting loop over crystals, etot = %5f:\n",bclus.energy());
  for (unsigned int icry=0; icry!=crystals_vector.size(); ++icry) {    
    
    EEDetId crystal(crystals_vector[icry].first);
        
    const CaloCellGeometry* cell=geomEnd_->getGeometry(crystal);
    GlobalPoint center_pos = (dynamic_cast<const TruncatedPyramid*>(cell))->getPosition(depth);
    double EtaCentr = center_pos.eta();
    double PhiCentr = TVector2::Phi_mpi_pi(center_pos.phi());

    float dr = reco::deltaR(Eta,Phi,EtaCentr,PhiCentr);
    if (dr<drmin) {
      drmin = dr;
      crystalseed = crystal;
    }
    
  }
  
  ix = crystalseed.ix();
  iy = crystalseed.iy();
  
  // Get center cell position from shower depth
  const CaloCellGeometry* cell=geomEnd_->getGeometry(crystalseed);
  const TruncatedPyramid *cpyr = dynamic_cast<const TruncatedPyramid*>(cell);

  thetatilt = cpyr->getThetaAxis();
  phitilt = cpyr->getPhiAxis();

  GlobalPoint center_pos = cpyr->getPosition(depth);
  
  double XCentr = center_pos.x();
  double XWidth = 2.59;
  xcry = (X-XCentr)/XWidth;
 
  
  double YCentr = center_pos.y();
  double YWidth = 2.59;
  ycry = (Y-YCentr)/YWidth;  
  //cout<<"x, y raw w/o widths "<<(X-XCentr)<<", "<<(Y-YCentr)<<endl;
}

float ggPFClusters::get5x5Element(int i, int j,
				  std::vector< std::pair<DetId, float> >& bcCells, 
				  bool isEB){
  
  Fill5x5Map(bcCells,isEB);
  if(abs(i)>2 ||abs(j)>2)return 0; //outside 5x5
  int ind1=i+2;
  int ind2=j+2;
  float E=e5x5_[ind1][ind2]; //return element from 5x5 cells 
  return E;
}

void ggPFClusters::Fill5x5Map(std::vector< std::pair<DetId, float> >& bcCells, 
			      bool isEB){
  
  for(int i=0; i<5; ++i)
    for(int j=0; j<5; ++j)e5x5_[i][j]=0;
  //cout<<"here 2 "<<endl;
  EcalRecHitCollection::const_iterator eb;
  EcalRecHitCollection::const_iterator ee;
  
  DetId idseed=FindSeed(bcCells, isEB);//return seed crystal
  
  for(unsigned int i=0; i<bcCells.size(); ++i){
    DetId id=bcCells[i].first;
    if(isEB){
      EBDetId EBidSeed=EBDetId(idseed.rawId());
      int deta=EBDetId::distanceEta(id,idseed);
      int dphi=EBDetId::distancePhi(id,idseed);
      if(abs(dphi)<=2 && abs(deta)<=2){
	for(eb=EBReducedRecHits_->begin();eb!= EBReducedRecHits_->end();++eb){
	  if(eb->id().rawId()==bcCells[i].first.rawId()){
	    EBDetId EBid=EBDetId(id.rawId());
	    int ind1=EBidSeed.ieta()-EBid.ieta();
	    int ind2=EBid.iphi()-EBidSeed.iphi();
	    if(EBidSeed.ieta() * EBid.ieta() > 0){
	      ind1=EBid.ieta()-EBidSeed.ieta();
	    }
	    else{ //near EB+ EB-
	      ind1=(1-(EBidSeed.ieta()-EBid.ieta())); 
	    }
	    int iEta=ind1+2;
	    int iPhi=ind2+2;
	    e5x5_[iEta][iPhi]=eb->energy()* bcCells[i].second;
	  }	  
	}
      }
    }
    else{
      EEDetId EEidSeed=EEDetId(idseed.rawId());
      int dx=EEDetId::distanceX(id,idseed);
      int dy=EEDetId::distanceY(id,idseed);
      if(abs(dx)<=2 && abs(dy)<=2){
	for(ee=EEReducedRecHits_->begin();ee!= EEReducedRecHits_->end();++ee){
	  if(ee->id().rawId()==bcCells[i].first.rawId()){
	    EEDetId EEid=EEDetId(id.rawId());	 
	    int ind1=EEid.ix()-EEidSeed.ix();
	    int ind2=EEid.iy()-EEidSeed.iy();
	    int ix=ind1+2;
	    int iy=ind2+2;
	    e5x5_[ix][iy]=ee->energy()* bcCells[i].second;
	  }
	}
      }   
    }
  }
}

DetId ggPFClusters::FindSeed(std::vector< std::pair<DetId, float> >& bcCells, bool isEB){
  //first find seed:
  EcalRecHitCollection::const_iterator eb;
  EcalRecHitCollection::const_iterator ee;
  DetId idseed;
  float PFSeedE=0;
  //find seed by largest energy matching
  for(unsigned int i=0; i<bcCells.size(); ++i){
    if(isEB){      
      for(eb=EBReducedRecHits_->begin();eb!= EBReducedRecHits_->end();++eb){		  
	if(eb->id().rawId()==bcCells[i].first.rawId()){
	  float cellE=eb->energy()* bcCells[i].second;
	  if(PFSeedE<cellE){		      
	    PFSeedE=cellE;
	    idseed=bcCells[i].first;
	  }
	  break;
	}
      }			
    }
    else{
      for(ee=EEReducedRecHits_->begin();ee!= EEReducedRecHits_->end();++ee){
	
	if(ee->id().rawId()==bcCells[i].first.rawId()){
	  float cellE=ee->energy()* bcCells[i].second;
	  if(PFSeedE<cellE){
	    PFSeedE=cellE;
	    idseed=bcCells[i].first;
	  }
	  break;
	}
      }
    }
  }
  return idseed;
}

std::pair<float, float> ggPFClusters::ClusterWidth(
						   vector<reco::CaloCluster>& PFClust){
  pair<float, float> widths(0,0);
  multimap<float, unsigned int>ClusterMap;
  //fill in Multimap to order by Energy (ascending order
  for(unsigned int i=0; i<PFClust.size();++i)ClusterMap.insert(make_pair(PFClust[i].energy(), i));
  //reverse iterator to get cluster with largest first 
  std::multimap<float, unsigned int>::reverse_iterator it;
  it=ClusterMap.rbegin();
  unsigned int max_c=(*it).second;
  std::vector< std::pair<DetId, float> >bcCells=PFClust[max_c].hitsAndFractions();
  EcalRecHitCollection::const_iterator eb;
  EcalRecHitCollection::const_iterator ee;
  float numeratorEtaWidth=0;
  float numeratorPhiWidth=0;
  float denominator=0;
  
  DetId seedXtalId = bcCells[0].first ;
  int detector = seedXtalId.subdetId() ;
  bool isEb;
  if(detector==1)isEb=true;
  else isEb=false;
  
  
  if(isEb){
    for(unsigned int i=0; i<bcCells.size(); ++i){
      for(eb=EBReducedRecHits_->begin();eb!= EBReducedRecHits_->end();++eb){
	if(eb->id().rawId()==bcCells[i].first.rawId()){
	  
	  double energyHit = eb->energy()*bcCells[i].second;
	  DetId id=bcCells[i].first;
	  float eta=geomBar_->getGeometry(id)->getPosition().eta();
	  float phi=geomBar_->getGeometry(id)->getPosition().phi();
	  float dEta = eta - PFClust[max_c].eta();
	  float dPhi = phi - PFClust[max_c].phi();
	  if (dPhi > + TMath::Pi()) { dPhi = TMath::TwoPi() - dPhi; }
	  if (dPhi < - TMath::Pi()) { dPhi = TMath::TwoPi() + dPhi; }
	  numeratorEtaWidth=dEta*dEta*energyHit+numeratorEtaWidth;
	  numeratorPhiWidth=dPhi*dPhi*energyHit+numeratorPhiWidth;
	  denominator=energyHit+denominator;
	  break;
	}
      }
      
    }
  }
  else{
    for(unsigned int i=0; i<bcCells.size(); ++i){
      for(ee=EEReducedRecHits_->begin();ee!= EEReducedRecHits_->end();++ee){
	if(ee->id().rawId()==bcCells[i].first.rawId()){	  
	  double energyHit = ee->energy()*bcCells[i].second;
	  DetId id=bcCells[i].first;
	  float eta=geomEnd_->getGeometry(id)->getPosition().eta();
	  float phi=geomEnd_->getGeometry(id)->getPosition().phi();
	  float dEta = eta - PFClust[max_c].eta();
	  float dPhi = phi - PFClust[max_c].phi();
	  if (dPhi > + TMath::Pi()) { dPhi = TMath::TwoPi() - dPhi; }
	  if (dPhi < - TMath::Pi()) { dPhi = TMath::TwoPi() + dPhi; }
	  numeratorEtaWidth=dEta*dEta*energyHit+numeratorEtaWidth;
	  numeratorPhiWidth=dPhi*dPhi*energyHit+numeratorPhiWidth;
	  denominator=energyHit+denominator;
	  break;	  
	}
      }   
    }
  }
  float etaWidth=sqrt(numeratorEtaWidth/denominator);
  float phiWidth=sqrt(numeratorPhiWidth/denominator);
  widths=make_pair(etaWidth, phiWidth);
  return widths;
}

double ggPFClusters::LocalEnergyCorrection(const GBRForest *ReaderLCEB, const GBRForest *ReaderLCEE,reco::CaloCluster PFClust,float beamspotZ ){
  
  //double Coor=1;
  double PFClustCorr=PFClust.energy();
  float ClusEta=PFClust.eta();
  float ClusE=PFClust.energy();
  float ClusPhi=PFClust.phi();
  
  std::vector<float>inputs;
  std::vector< std::pair<DetId, float> >bcCells=PFClust.hitsAndFractions();
  bool isEb;
  DetId seedXtalId = bcCells[0].first ;
  int detector = seedXtalId.subdetId();
  if(detector==1)isEb=true;
  else isEb=false;
  //Shower Shape Variables:
  Fill5x5Map(bcCells, isEb);
  float Eseed=get5x5Element(0, 0, bcCells, isEb);
  
  float Etop=get5x5Element(0, 1, bcCells, isEb);
  float Ebottom=get5x5Element(0, -1, bcCells, isEb);
  float Eleft=get5x5Element(-1, 0, bcCells, isEb);
  float Eright=get5x5Element(1, 0, bcCells, isEb);
  float E5x5=0; float E3x3=0;
  for(int i=-2; i<3; ++i)
    for(int j=-2; j<3; ++j){
      float e=get5x5Element(i, j, bcCells, isEb);
      if(abs(i)<2)E3x3=E3x3+e;
      E5x5=e+E5x5;
    }
  //Fill5x5Map(bcCells, isEb);
  float E3x1=e5x5_[2][2]+e5x5_[1][2]+e5x5_[3][2];
  float E1x3=e5x5_[2][2]+e5x5_[2][1]+e5x5_[2][3];
  float E1x5=e5x5_[2][2]+e5x5_[2][0]+e5x5_[2][1]+e5x5_[2][3]+e5x5_[2][4];
  
  float E2x5Top=e5x5_[0][4]+e5x5_[1][4]+e5x5_[2][4]+e5x5_[3][4]+e5x5_[4][4]
    +e5x5_[0][3]+e5x5_[1][3]+e5x5_[2][3]+e5x5_[3][3]+e5x5_[4][3];
  //add up bottom edge of 5x5 2 rows
  float E2x5Bottom=e5x5_[0][0]+e5x5_[1][0]+e5x5_[2][0]+e5x5_[3][0]+e5x5_[4][0]
    +e5x5_[0][1]+e5x5_[1][1]+e5x5_[2][1]+e5x5_[3][1]+e5x5_[4][1];
  //add up left edge of 5x5 2 rows
  float E2x5Left=e5x5_[0][1]+e5x5_[0][1]+e5x5_[0][2]+e5x5_[0][3]+e5x5_[0][4]
    +e5x5_[1][0]+e5x5_[1][1]+e5x5_[1][2]+e5x5_[1][3]+e5x5_[1][4];
  //add up right edge of 5x5 2 rows
  float E2x5Right=e5x5_[4][0]+e5x5_[4][1]+e5x5_[4][2]+e5x5_[4][3]+e5x5_[4][4]
    +e5x5_[3][0]+e5x5_[3][1]+e5x5_[3][2]+e5x5_[3][3]+e5x5_[3][4];
  //find max 2x5 from the center
  float centerstrip=e5x5_[2][2]+e5x5_[2][0]+e5x5_[2][1]+e5x5_[2][3]+e5x5_[2][4];
  float rightstrip=e5x5_[3][2]+e5x5_[3][0]+e5x5_[3][1]+e5x5_[3][3]+e5x5_[3][4];
  float leftstrip=e5x5_[1][2]+e5x5_[1][0]+e5x5_[1][1]+e5x5_[1][3]+e5x5_[1][4];
  float E2x5Max=0;
  if(rightstrip>leftstrip)E2x5Max=rightstrip+centerstrip;
  else E2x5Max=leftstrip+centerstrip;
  //get Local Coordinates
  if(isEb){
    float etacry; float phicry; int ieta; int iphi; float thetatilt; float phitilt;
    int iEtaCrack=3;
    if(abs(ieta)==1 || abs(ieta)==2 )
      iEtaCrack=abs(ieta);
    if(abs(ieta)>2 && abs(ieta)<24)
      iEtaCrack=3;
    if(abs(ieta)==24)
      iEtaCrack=4;
    if(abs(ieta)==25)
      iEtaCrack=5;
    if(abs(ieta)==26)
      iEtaCrack=6;
    if(abs(ieta)==27)
      iEtaCrack=7;
    if(abs(ieta)>27 &&  abs(ieta)<44)
      iEtaCrack=8;
    if(abs(ieta)==44)
      iEtaCrack=9;
    if(abs(ieta)==45)
      iEtaCrack=10;
    if(abs(ieta)==46)
      iEtaCrack=11;
    if(abs(ieta)==47)
      iEtaCrack=12;
    if(abs(ieta)>47 &&  abs(ieta)<64)
      iEtaCrack=13;
    if(abs(ieta)==64)
      iEtaCrack=14;
    if(abs(ieta)==65)
      iEtaCrack=15;
    if(abs(ieta)==66)
      iEtaCrack=16;
    if(abs(ieta)==67)
      iEtaCrack=17;
    if(abs(ieta)>67 &&  abs(ieta)<84)
      iEtaCrack=18;
    if(abs(ieta)==84)
      iEtaCrack=19;
    if(abs(ieta)==85)
      iEtaCrack=20;
    localCoordsEB(PFClust, etacry, phicry, ieta, iphi, thetatilt, phitilt);
    inputs.push_back(beamspotZ);
    inputs.push_back(ClusEta/fabs(ClusEta));
    inputs.push_back(fabs(ClusEta));
    inputs.push_back(fabs(ClusPhi));
    inputs.push_back(log(ClusE));
    inputs.push_back((Eseed/ClusE));
    inputs.push_back((Etop/ClusE));
    inputs.push_back((Ebottom/ClusE));
    inputs.push_back((Eleft/ClusE));
    inputs.push_back((Eright/ClusE));
    inputs.push_back(E3x3/ClusE);
    inputs.push_back(E1x3/ClusE);
    inputs.push_back(E3x1/ClusE);
    inputs.push_back(E5x5/ClusE);
    inputs.push_back(E1x5/ClusE);
    inputs.push_back(E2x5Max/ClusE);
    inputs.push_back(E2x5Top/ClusE);
    inputs.push_back(E2x5Bottom/ClusE);
    inputs.push_back(E2x5Left/ClusE);
    inputs.push_back(E2x5Right/ClusE);
    inputs.push_back(etacry);
    inputs.push_back(phicry);
    inputs.push_back(iphi%2);
    inputs.push_back(ieta%5);
    inputs.push_back(iphi%20);
    inputs.push_back(iEtaCrack);
    int size=inputs.size();
    float PFInputs[26];
    for(int i=0; i<size; ++i)PFInputs[i]=inputs[i];
    PFClustCorr= ReaderLCEB->GetResponse(PFInputs)*ClusE;
  }
  else{    
    float xcry; float ycry; int ix; int iy; float thetatilt; float phitilt;
    localCoordsEE(PFClust, xcry, ycry, ix, iy, thetatilt, phitilt);
    inputs.push_back(beamspotZ);
    inputs.push_back(ClusEta/fabs(ClusEta));
    inputs.push_back(fabs(ClusEta));
    inputs.push_back(fabs(ClusPhi));
    inputs.push_back(log(ClusE));
    inputs.push_back((Eseed/ClusE));
    inputs.push_back((Etop/ClusE));
    inputs.push_back((Ebottom/ClusE));
    inputs.push_back((Eleft/ClusE));
    inputs.push_back((Eright/ClusE));
    inputs.push_back(E3x3/ClusE);
    inputs.push_back(E1x3/ClusE);
    inputs.push_back(E3x1/ClusE);
    inputs.push_back(E5x5/ClusE);
    inputs.push_back(E1x5/ClusE);
    inputs.push_back(E2x5Max/ClusE);
    inputs.push_back(E2x5Top/ClusE);
    inputs.push_back(E2x5Bottom/ClusE);
    inputs.push_back(E2x5Left/ClusE);
    inputs.push_back(E2x5Right/ClusE);
    inputs.push_back(xcry);
    inputs.push_back(ycry);
    int size=inputs.size();
    float PFInputs[22];
    for(int i=0; i<size; ++i)PFInputs[i]=inputs[i];
    PFClustCorr= ReaderLCEE->GetResponse(PFInputs) *ClusE;
  }
  
  return PFClustCorr;
}
