#include <iostream>

#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "RecoEcal/EgammaCoreTools/interface/ClusterShapeAlgo.h"
#include "RecoCaloTools/Navigation/interface/CaloNavigator.h"
#include "Geometry/EcalAlgo/interface/EcalBarrelGeometry.h"
#include "Geometry/CaloTopology/interface/EcalBarrelTopology.h"
#include "Geometry/CaloTopology/interface/EcalEndcapTopology.h"
#include "Geometry/CaloTopology/interface/EcalPreshowerTopology.h"

#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/Math/interface/Vector3D.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CLHEP/Geometry/Transform3D.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"

ClusterShapeAlgo::ClusterShapeAlgo(const edm::ParameterSet& par) : 
  parameterSet_(par) {}

reco::ClusterShape ClusterShapeAlgo::Calculate(const reco::BasicCluster &passedCluster,
                                               const EcalRecHitCollection *hits,
                                               const CaloSubdetectorGeometry * geometry,
                                               const CaloSubdetectorTopology* topology)
{
  Calculate_TopEnergy(passedCluster,hits);
  Calculate_2ndEnergy(passedCluster,hits);
  Create_Map(hits,topology);
  Calculate_e2x2();
  Calculate_e3x2();
  Calculate_e3x3();
  Calculate_e4x4();
  Calculate_e5x5();
  Calculate_e2x5Right();
  Calculate_e2x5Left();
  Calculate_e2x5Top();
  Calculate_e2x5Bottom();
  Calculate_Covariances(passedCluster,hits,geometry);
  Calculate_BarrelBasketEnergyFraction(passedCluster,hits, Eta, geometry);
  Calculate_BarrelBasketEnergyFraction(passedCluster,hits, Phi, geometry);
  Calculate_EnergyDepTopology (passedCluster,hits,geometry,true) ;
  Calculate_lat(passedCluster);
  Calculate_ComplexZernikeMoments(passedCluster);

  return reco::ClusterShape(covEtaEta_, covEtaPhi_, covPhiPhi_, eMax_, eMaxId_,
			    e2nd_, e2ndId_, e2x2_, e3x2_, e3x3_,e4x4_, e5x5_,
			    e2x5Right_, e2x5Left_, e2x5Top_, e2x5Bottom_,
			    e3x2Ratio_, lat_, etaLat_, phiLat_, A20_, A42_,
			    energyBasketFractionEta_, energyBasketFractionPhi_);
}

void ClusterShapeAlgo::Calculate_TopEnergy(const reco::BasicCluster &passedCluster,const EcalRecHitCollection *hits)
{
  double eMax=0;
  DetId eMaxId(0);

  std::vector< std::pair<DetId, float> > clusterDetIds = passedCluster.hitsAndFractions();
  std::vector< std::pair<DetId, float> >::iterator posCurrent;

  EcalRecHit testEcalRecHit;

  for(posCurrent = clusterDetIds.begin(); posCurrent != clusterDetIds.end(); posCurrent++)
  {
    if (((*posCurrent).first != DetId(0)) && (hits->find((*posCurrent).first) != hits->end()))
    {
      EcalRecHitCollection::const_iterator itt = hits->find((*posCurrent).first);
      testEcalRecHit = *itt;

      if(testEcalRecHit.energy() * (*posCurrent).second > eMax)
      {
        eMax = testEcalRecHit.energy() * (*posCurrent).second;
        eMaxId = testEcalRecHit.id();
      }
    }
  }

  eMax_ = eMax;
  eMaxId_ = eMaxId;
}

void ClusterShapeAlgo::Calculate_2ndEnergy(const reco::BasicCluster &passedCluster,const EcalRecHitCollection *hits)
{
  double e2nd=0;
  DetId e2ndId(0);

  std::vector< std::pair<DetId, float> > clusterDetIds = passedCluster.hitsAndFractions();
  std::vector< std::pair<DetId, float> >::iterator posCurrent;

  EcalRecHit testEcalRecHit;

  for(posCurrent = clusterDetIds.begin(); posCurrent != clusterDetIds.end(); posCurrent++)
  { 
    if (( (*posCurrent).first != DetId(0)) && (hits->find( (*posCurrent).first ) != hits->end()))
    {
      EcalRecHitCollection::const_iterator itt = hits->find( (*posCurrent).first );
      testEcalRecHit = *itt;

      if(testEcalRecHit.energy() * (*posCurrent).second > e2nd && testEcalRecHit.id() != eMaxId_)
      {
        e2nd = testEcalRecHit.energy() * (*posCurrent).second;
        e2ndId = testEcalRecHit.id();
      }
    }
  }

  e2nd_ = e2nd;
  e2ndId_ = e2ndId;
}

void ClusterShapeAlgo::Create_Map(const EcalRecHitCollection *hits,const CaloSubdetectorTopology* topology)
{
  EcalRecHit tempEcalRecHit;
  CaloNavigator<DetId> posCurrent = CaloNavigator<DetId>(eMaxId_,topology );

  for(int x = 0; x < 5; x++)
    for(int y = 0; y < 5; y++)
    {
      posCurrent.home();
      posCurrent.offsetBy(-2+x,-2+y);

      if((*posCurrent != DetId(0)) && (hits->find(*posCurrent) != hits->end()))
      {
				EcalRecHitCollection::const_iterator itt = hits->find(*posCurrent);
				tempEcalRecHit = *itt;
				energyMap_[y][x] = std::make_pair(tempEcalRecHit.id(),tempEcalRecHit.energy()); 
      }
      else
				energyMap_[y][x] = std::make_pair(DetId(0), 0);  
    }
}

void ClusterShapeAlgo::Calculate_e2x2()
{
  double e2x2Max = 0;
  double e2x2Test = 0;

  int deltaX=0, deltaY=0;

  for(int corner = 0; corner < 4; corner++)
  {
    switch(corner)
    {
      case 0: deltaX = -1; deltaY = -1; break;
      case 1: deltaX = -1; deltaY =  1; break;
      case 2: deltaX =  1; deltaY = -1; break;
      case 3: deltaX =  1; deltaY =  1; break;
    }

    e2x2Test  = energyMap_[2][2].second;
    e2x2Test += energyMap_[2+deltaY][2].second;
    e2x2Test += energyMap_[2][2+deltaX].second;
    e2x2Test += energyMap_[2+deltaY][2+deltaX].second;

    if(e2x2Test > e2x2Max)
    {
			e2x2Max = e2x2Test;
			e2x2_Diagonal_X_ = 2+deltaX;
			e2x2_Diagonal_Y_ = 2+deltaY;
    }
  }

  e2x2_ = e2x2Max;

}

void ClusterShapeAlgo::Calculate_e3x2()
{
  double e3x2 = 0.0;
  double e3x2Ratio = 0.0, e3x2RatioNumerator = 0.0, e3x2RatioDenominator = 0.0;

  int e2ndX = 2, e2ndY = 2;
  int deltaY = 0, deltaX = 0;

  double nextEnergy = -999;
  int nextEneryDirection = -1;

  for(int cardinalDirection = 0; cardinalDirection < 4; cardinalDirection++)
  {
    switch(cardinalDirection)
    {
      case 0: deltaX = -1; deltaY =  0; break;
      case 1: deltaX =  1; deltaY =  0; break;
      case 2: deltaX =  0; deltaY = -1; break;
      case 3: deltaX =  0; deltaY =  1; break;
    }
   
    if(energyMap_[2+deltaY][2+deltaX].second >= nextEnergy)
    {
        nextEnergy = energyMap_[2+deltaY][2+deltaX].second;
        nextEneryDirection = cardinalDirection;
       
        e2ndX = 2+deltaX;
        e2ndY = 2+deltaY;
    }
  }
 
  switch(nextEneryDirection)
  {
    case 0: ;
    case 1: deltaX = 0; deltaY = 1; break;
    case 2: ;
    case 3: deltaX = 1; deltaY = 0; break;
  }

  for(int sign = -1; sign <= 1; sign++)
      e3x2 += (energyMap_[2+deltaY*sign][2+deltaX*sign].second + energyMap_[e2ndY+deltaY*sign][e2ndX+deltaX*sign].second);
 
  e3x2RatioNumerator   = energyMap_[e2ndY+deltaY][e2ndX+deltaX].second + energyMap_[e2ndY-deltaY][e2ndX-deltaX].second;
  e3x2RatioDenominator = 0.5 + energyMap_[2+deltaY][2+deltaX].second + energyMap_[2-deltaY][2-deltaX].second;
  e3x2Ratio = e3x2RatioNumerator / e3x2RatioDenominator;

  e3x2_ = e3x2;
  e3x2Ratio_ = e3x2Ratio;
}  

void ClusterShapeAlgo::Calculate_e3x3()
{
  double e3x3=0;

  for(int i = 1; i <= 3; i++)
    for(int j = 1; j <= 3; j++)
      e3x3 += energyMap_[j][i].second;

  e3x3_ = e3x3;

}

void ClusterShapeAlgo::Calculate_e4x4()
{
  double e4x4=0;
	
  int startX=-1, startY=-1;

	switch(e2x2_Diagonal_X_)
	{
		case 1: startX = 0; break;
		case 3: startX = 1; break;
	}

	switch(e2x2_Diagonal_Y_)
	{
		case 1: startY = 0; break;
		case 3: startY = 1; break;
	}

  for(int i = startX; i <= startX+3; i++)
 	  for(int j = startY; j <= startY+3; j++)
   	  e4x4 += energyMap_[j][i].second;

  e4x4_ = e4x4;
}

void ClusterShapeAlgo::Calculate_e5x5()
{
  double e5x5=0;

  for(int i = 0; i <= 4; i++)
    for(int j = 0; j <= 4; j++)
      e5x5 += energyMap_[j][i].second;

  e5x5_ = e5x5;

}

void ClusterShapeAlgo::Calculate_e2x5Right()
{
double e2x5R=0.0;
  for(int i = 0; i <= 4; i++){
    for(int j = 0; j <= 4; j++){
      if(j>2){e2x5R +=energyMap_[i][j].second;}
    }
  }
  e2x5Right_=e2x5R;
}

void ClusterShapeAlgo::Calculate_e2x5Left()
{
double e2x5L=0.0;
  for(int i = 0; i <= 4; i++){
    for(int j = 0; j <= 4; j++){
      if(j<2){e2x5L +=energyMap_[i][j].second;}
    }
  }
  e2x5Left_=e2x5L;
}

void ClusterShapeAlgo::Calculate_e2x5Bottom()
{
double e2x5B=0.0;
  for(int i = 0; i <= 4; i++){
    for(int j = 0; j <= 4; j++){

      if(i>2){e2x5B +=energyMap_[i][j].second;}
    }
  }
  e2x5Bottom_=e2x5B;
}

void ClusterShapeAlgo::Calculate_e2x5Top()
{
double e2x5T=0.0;
  for(int i = 0; i <= 4; i++){
    for(int j = 0; j <= 4; j++){

      if(i<2){e2x5T +=energyMap_[i][j].second;}
    }
  }
  e2x5Top_=e2x5T;
}

void ClusterShapeAlgo::Calculate_Covariances(const reco::BasicCluster &passedCluster, const EcalRecHitCollection* hits, 
					     const CaloSubdetectorGeometry* geometry)
{
  if (e5x5_ > 0.)
    {
      double w0_ = parameterSet_.getParameter<double>("W0");
      
      
      // first find energy-weighted mean position - doing it when filling the energy map might save time
      math::XYZVector meanPosition(0.0, 0.0, 0.0);
      for (int i = 0; i < 5; ++i)
	{
	  for (int j = 0; j < 5; ++j)
	    {
	      DetId id = energyMap_[i][j].first;
	      if (id != DetId(0))
		{
		  GlobalPoint positionGP = geometry->getGeometry(id)->getPosition();
		  math::XYZVector position(positionGP.x(),positionGP.y(),positionGP.z());
		  meanPosition = meanPosition + energyMap_[i][j].second * position;
		}
	    }
	}
      
      meanPosition /= e5x5_;
      
      // now we can calculate the covariances
      double numeratorEtaEta = 0;
      double numeratorEtaPhi = 0;
      double numeratorPhiPhi = 0;
      double denominator     = 0;
      
      for (int i = 0; i < 5; ++i)
	{
	  for (int j = 0; j < 5; ++j)
	    {
	      DetId id = energyMap_[i][j].first;
	      if (id != DetId(0))
		{
		  GlobalPoint position = geometry->getGeometry(id)->getPosition();
		  
		  double dPhi = position.phi() - meanPosition.phi();
		  if (dPhi > + Geom::pi()) { dPhi = Geom::twoPi() - dPhi; }
		  if (dPhi < - Geom::pi()) { dPhi = Geom::twoPi() + dPhi; }
		  
		  double dEta = position.eta() - meanPosition.eta();
		  double w = 0.;
		  if ( energyMap_[i][j].second > 0.)
		    w = std::max(0.0, w0_ + log( energyMap_[i][j].second / e5x5_));
		  
		  denominator += w;
		  numeratorEtaEta += w * dEta * dEta;
		  numeratorEtaPhi += w * dEta * dPhi;
		  numeratorPhiPhi += w * dPhi * dPhi;
		}
	    }
	}
      
      covEtaEta_ = numeratorEtaEta / denominator;
      covEtaPhi_ = numeratorEtaPhi / denominator;
      covPhiPhi_ = numeratorPhiPhi / denominator;
    }
  else 
    {
      // Warn the user if there was no energy in the cells and return zeroes.
      //       std::cout << "\ClusterShapeAlgo::Calculate_Covariances:  no energy in supplied cells.\n";
      covEtaEta_ = 0;
      covEtaPhi_ = 0;
      covPhiPhi_ = 0;
    }
}

void ClusterShapeAlgo::Calculate_BarrelBasketEnergyFraction(const reco::BasicCluster &passedCluster,
                                                            const EcalRecHitCollection *hits,
                                                            const int EtaPhi,
                                                            const CaloSubdetectorGeometry* geometry) 
{
  if(  (hits!=0) && ( ((*hits)[0]).id().subdetId() != EcalBarrel )  ) {
     //std::cout << "No basket correction for endacap!" << std::endl;
     return;
  }

  std::map<int,double> indexedBasketEnergy;
  std::vector< std::pair<DetId, float> > clusterDetIds = passedCluster.hitsAndFractions();
  const EcalBarrelGeometry* subDetGeometry = (const EcalBarrelGeometry*) geometry;

  for(std::vector< std::pair<DetId, float> >::iterator posCurrent = clusterDetIds.begin(); posCurrent != clusterDetIds.end(); posCurrent++)
  {
    int basketIndex = 999;

    if(EtaPhi == Eta) {
      int unsignedIEta = abs(EBDetId( (*posCurrent).first ).ieta());
      std::vector<int> etaBasketSize = subDetGeometry->getEtaBaskets();

      for(unsigned int i = 0; i < etaBasketSize.size(); i++) {
        unsignedIEta -= etaBasketSize[i];
        if(unsignedIEta - 1 < 0)
        {
          basketIndex = i;
          break;
        }
      }
      basketIndex = (basketIndex+1)*(EBDetId( (*posCurrent).first ).ieta() > 0 ? 1 : -1);

    } else if(EtaPhi == Phi) {
      int halfNumBasketInPhi = (EBDetId::MAX_IPHI - EBDetId::MIN_IPHI + 1)/subDetGeometry->getBasketSizeInPhi()/2;

      basketIndex = (EBDetId( (*posCurrent).first ).iphi() - 1)/subDetGeometry->getBasketSizeInPhi()
                  - (EBDetId( (clusterDetIds[0]).first ).iphi() - 1)/subDetGeometry->getBasketSizeInPhi();

      if(basketIndex >= halfNumBasketInPhi)             basketIndex -= 2*halfNumBasketInPhi;
      else if(basketIndex <  -1*halfNumBasketInPhi)     basketIndex += 2*halfNumBasketInPhi;

    } else throw(std::runtime_error("\n\nOh No! Calculate_BarrelBasketEnergyFraction called on invalid index.\n\n"));

    indexedBasketEnergy[basketIndex] += (hits->find( (*posCurrent).first ))->energy();
  }

  std::vector<double> energyFraction;
  for(std::map<int,double>::iterator posCurrent = indexedBasketEnergy.begin(); posCurrent != indexedBasketEnergy.end(); posCurrent++)
  {
    energyFraction.push_back(indexedBasketEnergy[posCurrent->first]/passedCluster.energy());
  }

  switch(EtaPhi)
  {
    case Eta: energyBasketFractionEta_ = energyFraction; break;
    case Phi: energyBasketFractionPhi_ = energyFraction; break;
  }

}

void ClusterShapeAlgo::Calculate_lat(const reco::BasicCluster &passedCluster) {

  double r,redmoment=0;
  double phiRedmoment = 0 ;
  double etaRedmoment = 0 ;
  int n,n1,n2,tmp;
  int clusterSize=energyDistribution_.size();
  if (clusterSize<3) {
    etaLat_ = 0.0 ; 
    lat_ = 0.0;
    return; 
  }
  
  n1=0; n2=1;
  if (energyDistribution_[1].deposited_energy > 
      energyDistribution_[0].deposited_energy) 
    {
      tmp=n2; n2=n1; n1=tmp;
    }
  for (int i=2; i<clusterSize; i++) {
    n=i;
    if (energyDistribution_[i].deposited_energy > 
        energyDistribution_[n1].deposited_energy) 
      {
        tmp = n2;
        n2 = n1; n1 = i; n=tmp;
      } else {
        if (energyDistribution_[i].deposited_energy > 
            energyDistribution_[n2].deposited_energy) 
          {
            tmp=n2; n2=i; n=tmp;
          }
      }

    r = energyDistribution_[n].r;
    redmoment += r*r* energyDistribution_[n].deposited_energy;
    double rphi = r * cos (energyDistribution_[n].phi) ;
    phiRedmoment += rphi * rphi * energyDistribution_[n].deposited_energy;
    double reta = r * sin (energyDistribution_[n].phi) ;
    etaRedmoment += reta * reta * energyDistribution_[n].deposited_energy;
  } 
  double e1 = energyDistribution_[n1].deposited_energy;
  double e2 = energyDistribution_[n2].deposited_energy;
  
  lat_ = redmoment/(redmoment+2.19*2.19*(e1+e2));
  phiLat_ = phiRedmoment/(phiRedmoment+2.19*2.19*(e1+e2));
  etaLat_ = etaRedmoment/(etaRedmoment+2.19*2.19*(e1+e2));
}

void ClusterShapeAlgo::Calculate_ComplexZernikeMoments(const reco::BasicCluster &passedCluster) {
  // Calculate only the moments which go into the default cluster shape
  // (moments with m>=2 are the only sensitive to azimuthal shape)
  A20_ = absZernikeMoment(passedCluster,2,0);
  A42_ = absZernikeMoment(passedCluster,4,2);
}

double ClusterShapeAlgo::absZernikeMoment(const reco::BasicCluster &passedCluster,
                                          int n, int m, double R0) {
  // 1. Check if n,m are correctly
  if ((m>n) || ((n-m)%2 != 0) || (n<0) || (m<0)) return -1;

  // 2. Check if n,R0 are within validity Range :
  // n>20 or R0<2.19cm  just makes no sense !
  if ((n>20) || (R0<=2.19)) return -1;
  if (n<=5) return fast_AbsZernikeMoment(passedCluster,n,m,R0);
  else return calc_AbsZernikeMoment(passedCluster,n,m,R0);
}

double ClusterShapeAlgo::f00(double r) { return 1; }

double ClusterShapeAlgo::f11(double r) { return r; }

double ClusterShapeAlgo::f20(double r) { return 2.0*r*r-1.0; }

double ClusterShapeAlgo::f22(double r) { return r*r; }

double ClusterShapeAlgo::f31(double r) { return 3.0*r*r*r - 2.0*r; }

double ClusterShapeAlgo::f33(double r) { return r*r*r; }

double ClusterShapeAlgo::f40(double r) { return 6.0*r*r*r*r-6.0*r*r+1.0; }

double ClusterShapeAlgo::f42(double r) { return 4.0*r*r*r*r-3.0*r*r; }

double ClusterShapeAlgo::f44(double r) { return r*r*r*r; }

double ClusterShapeAlgo::f51(double r) { return 10.0*pow(r,5)-12.0*pow(r,3)+3.0*r; }

double ClusterShapeAlgo::f53(double r) { return 5.0*pow(r,5) - 4.0*pow(r,3); }

double ClusterShapeAlgo::f55(double r) { return pow(r,5); }

double ClusterShapeAlgo::fast_AbsZernikeMoment(const reco::BasicCluster &passedCluster,
                                               int n, int m, double R0) {
  double r,ph,e,Re=0,Im=0,result;
  double TotalEnergy = passedCluster.energy();
  int index = (n/2)*(n/2)+(n/2)+m;
  int clusterSize=energyDistribution_.size();
  if(clusterSize<3) return 0.0;

  for (int i=0; i<clusterSize; i++)
    { 
      r = energyDistribution_[i].r / R0;
      if (r<1) {
        fcn_.clear();
        Calculate_Polynomials(r);
        ph = (energyDistribution_[i]).phi;
        e = energyDistribution_[i].deposited_energy;
        Re = Re + e/TotalEnergy * fcn_[index] * cos( (double) m * ph);
        Im = Im - e/TotalEnergy * fcn_[index] * sin( (double) m * ph);
      }
    }
  result = sqrt(Re*Re+Im*Im);

  return result;
}

double ClusterShapeAlgo::calc_AbsZernikeMoment(const reco::BasicCluster &passedCluster,
                                               int n, int m, double R0) {
  double r,ph,e,Re=0,Im=0,f_nm,result;
  double TotalEnergy = passedCluster.energy();
  std::vector< std::pair<DetId, float> > clusterDetIds = passedCluster.hitsAndFractions();
  int clusterSize=energyDistribution_.size();
  if(clusterSize<3) return 0.0;

  for (int i=0; i<clusterSize; i++)
    { 
      r = energyDistribution_[i].r / R0;
      if (r<1) {
        ph = (energyDistribution_[i]).phi;
        e = energyDistribution_[i].deposited_energy;
        f_nm=0;
        for (int s=0; s<=(n-m)/2; s++) {
          if (s%2==0)
            { 
              f_nm = f_nm + factorial(n-s)*pow(r,(double) (n-2*s))/(factorial(s)*factorial((n+m)/2-s)*factorial((n-m)/2-s));
            }else {
              f_nm = f_nm - factorial(n-s)*pow(r,(double) (n-2*s))/(factorial(s)*factorial((n+m)/2-s)*factorial((n-m)/2-s));
            }
        }
        Re = Re + e/TotalEnergy * f_nm * cos( (double) m*ph);
        Im = Im - e/TotalEnergy * f_nm * sin( (double) m*ph);
      }
    }
  result = sqrt(Re*Re+Im*Im);

  return result;
}

void ClusterShapeAlgo::Calculate_EnergyDepTopology (const reco::BasicCluster &passedCluster,
						    const EcalRecHitCollection *hits,
						    const CaloSubdetectorGeometry* geometry,
						    bool logW) {
  // resets the energy distribution
  energyDistribution_.clear();

  // init a map of the energy deposition centered on the
  // cluster centroid. This is for momenta calculation only.
  CLHEP::Hep3Vector clVect(passedCluster.position().x(),
                           passedCluster.position().y(),
                           passedCluster.position().z());
  CLHEP::Hep3Vector clDir(clVect);
  clDir*=1.0/clDir.mag();
  // in the transverse plane, axis perpendicular to clusterDir
  CLHEP::Hep3Vector theta_axis(clDir.y(),-clDir.x(),0.0);
  theta_axis *= 1.0/theta_axis.mag();
  CLHEP::Hep3Vector phi_axis = theta_axis.cross(clDir);

  std::vector< std::pair<DetId, float> > clusterDetIds = passedCluster.hitsAndFractions();

  EcalClusterEnergyDeposition clEdep;
  EcalRecHit testEcalRecHit;
  std::vector< std::pair<DetId, float> >::iterator posCurrent;
  // loop over crystals
  for(posCurrent=clusterDetIds.begin(); posCurrent!=clusterDetIds.end(); ++posCurrent) {
    EcalRecHitCollection::const_iterator itt = hits->find( (*posCurrent).first );
    testEcalRecHit=*itt;

    if(( (*posCurrent).first != DetId(0)) && (hits->find( (*posCurrent).first ) != hits->end())) {
      clEdep.deposited_energy = testEcalRecHit.energy();

      // if logarithmic weight is requested, apply cut on minimum energy of the recHit
      if(logW) {
        double w0_ = parameterSet_.getParameter<double>("W0");

        if ( clEdep.deposited_energy == 0 ) {
          LogDebug("ClusterShapeAlgo") << "Crystal has zero energy; skipping... ";
          continue;
        }
        double weight = std::max(0.0, w0_ + log(fabs(clEdep.deposited_energy)/passedCluster.energy()) );
        if(weight==0) {
          LogDebug("ClusterShapeAlgo") << "Crystal has insufficient energy: E = " 
                                       << clEdep.deposited_energy << " GeV; skipping... ";
          continue;
        }
        else LogDebug("ClusterShapeAlgo") << "===> got crystal. Energy = " << clEdep.deposited_energy << " GeV. ";
      }
      DetId id_ = (*posCurrent).first;
      const CaloCellGeometry *this_cell = geometry->getGeometry(id_);
      GlobalPoint cellPos = this_cell->getPosition();
      CLHEP::Hep3Vector gblPos (cellPos.x(),cellPos.y(),cellPos.z()); //surface position?
      // Evaluate the distance from the cluster centroid
      CLHEP::Hep3Vector diff = gblPos - clVect;
      // Important: for the moment calculation, only the "lateral distance" is important
      // "lateral distance" r_i = distance of the digi position from the axis Origin-Cluster Center
      // ---> subtract the projection on clDir
      CLHEP::Hep3Vector DigiVect = diff - diff.dot(clDir)*clDir;
      clEdep.r = DigiVect.mag();
      LogDebug("ClusterShapeAlgo") << "E = " << clEdep.deposited_energy
                                   << "\tdiff = " << diff.mag()
                                   << "\tr = " << clEdep.r;
      clEdep.phi = DigiVect.angle(theta_axis);
      if(DigiVect.dot(phi_axis)<0) clEdep.phi = 2*M_PI - clEdep.phi;
      energyDistribution_.push_back(clEdep);
    }
  } 
}

void ClusterShapeAlgo::Calculate_Polynomials(double rho) {
  fcn_.push_back(f00(rho));
  fcn_.push_back(f11(rho));
  fcn_.push_back(f20(rho));
  fcn_.push_back(f31(rho));
  fcn_.push_back(f22(rho));
  fcn_.push_back(f33(rho));
  fcn_.push_back(f40(rho));
  fcn_.push_back(f51(rho));
  fcn_.push_back(f42(rho));
  fcn_.push_back(f53(rho));
  fcn_.push_back(f44(rho));
  fcn_.push_back(f55(rho));
}

double ClusterShapeAlgo::factorial(int n) const {
  double res=1.0;
  for(int i=2; i<=n; i++) res*=(double) i;
  return res;
}
