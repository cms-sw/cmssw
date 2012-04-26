#include "RecoEgamma/EgammaTools/interface/ggPFESClusters.h"
#include "DataFormats/EcalDetId/interface/ESDetId.h"
#include "TMath.h"
using namespace edm;
using namespace std;
using namespace reco;
/*
class by Rishi Patel rpatel@cern.ch 
and Eleni Petrakou Eleni.Petrakou@cern.ch
*/
ggPFESClusters::ggPFESClusters(
			       edm::Handle<EcalRecHitCollection>& ESRecHits,
			       const CaloSubdetectorGeometry* geomEnd
			       ):
  ESRecHits_(ESRecHits),
  geomEnd_(geomEnd)
{
  
}

ggPFESClusters::~ggPFESClusters(){

}

vector<reco::PreshowerCluster>ggPFESClusters::getPFESClusters(
							      reco::SuperCluster sc 
							      ){
  // cout<<"SC Eta "<<sc.eta()<<endl;
  std::vector<PreshowerCluster>PFPreShowerClust;
  for(reco::CaloCluster_iterator ps=sc.preshowerClustersBegin(); ps!=sc.preshowerClustersEnd(); ++ps){
    std::vector< std::pair<DetId, float> > psCells=(*ps)->hitsAndFractions();
    float PS1E=0;
    float PS2E=0;	
    int plane=0;
    for(unsigned int s=0; s<psCells.size(); ++s){
      for(EcalRecHitCollection::const_iterator es=ESRecHits_->begin(); es!= ESRecHits_->end(); ++es){
	if(es->id().rawId()==psCells[s].first.rawId()){ 
	  ESDetId id=ESDetId(psCells[s].first.rawId());
	  plane=id.plane();
	  if(id.plane()==1)PS1E=PS1E + es->energy() * psCells[s].second;
	  if(id.plane()==2)PS2E=PS2E + es->energy() * psCells[s].second;
	  break;
	}		  
      }
    }
    //make PreShower object storing plane PSEnergy and Position
    if(plane==1){
      PreshowerCluster PS1=PreshowerCluster(PS1E,(*ps)->position(),(*ps)->hitsAndFractions(),plane);
      PFPreShowerClust.push_back(PS1);	
    }
    if(plane==2){
      PreshowerCluster PS2=PreshowerCluster(PS2E,(*ps)->position(),(*ps)->hitsAndFractions(),plane);
      PFPreShowerClust.push_back(PS2);      
    }
  }  
  return PFPreShowerClust;  
}

double ggPFESClusters::getLinkDist(reco::PreshowerCluster clusterPS, reco::CaloCluster clusterECAL){

  static double resPSpitch = 0.19/sqrt(12.);
  static double resPSlength = 6.1/sqrt(12.);

  // ECAL cluster position
  double zECAL  = clusterECAL.position().Z(); // cluster centroid position 
  double xECAL  = clusterECAL.position().X();
  double yECAL  = clusterECAL.position().Y();

  // PS cluster position 
  double zPS = clusterPS.position().Z();
  double xPS = clusterPS.position().X();
  double yPS = clusterPS.position().Y();

  // Check that zEcal and zPs have the same sign
  if (zECAL*zPS < 0.) return -1.;
  double deltaX = 0.;
  double deltaY = 0.;
  double sqr12 = std::sqrt(12.);
  switch (clusterPS.plane()) {
  case 1:
    // vertical strips, measure x with pitch precision
    deltaX = resPSpitch * sqr12;
    deltaY = resPSlength * sqr12;
    break;
  case 2:
    // horizontal strips, measure y with pitch precision
    deltaY = resPSpitch * sqr12;
    deltaX = resPSlength * sqr12;
    break;
  default:
    break;
  }

  // Get the rechits
  const std::vector< std::pair<DetId, float> > fracs = clusterECAL.hitsAndFractions();
  bool linkedbyrechit = false;
  //loop rechits
  for(unsigned int rhit = 0; rhit < fracs.size(); rhit++){
    double fraction = fracs[rhit].second;
    if(fraction < 1E-4) continue;
    if(fracs[rhit].first.null()) continue;

    //getting rechit center position
    DetId eehitid=DetId(fracs[rhit].first.rawId());
    const CaloCellGeometry *xtal = geomEnd_->getGeometry(eehitid);
    if(!xtal) continue;
    
    //getting rechit corners
    const CaloCellGeometry::CornersVec& corners_vec = xtal->getCorners();
    math::XYZPoint corners[4];
    // They have this order just for compatibility with the original function. 
    corners[0] = math::XYZPoint(corners_vec[3].x(), corners_vec[3].y(), corners_vec[3].z() );
    corners[1] = math::XYZPoint(corners_vec[2].x(), corners_vec[2].y(), corners_vec[2].z() );
    corners[2] = math::XYZPoint(corners_vec[1].x(), corners_vec[1].y(), corners_vec[1].z() );
    corners[3] = math::XYZPoint(corners_vec[0].x(), corners_vec[0].y(), corners_vec[0].z() );
    const GlobalPoint p = geomEnd_->getGeometry(eehitid)->getPosition();

    // Extrapolating from the xtal's surface (because of working with DetId's) 
    const math::XYZPoint& posxyz = math::XYZPoint(p.x()*zPS/p.z(), p.y()*zPS/p.z(), p.z()*zPS/p.z());

    double x[5];
    double y[5];
    for ( unsigned jc=0; jc<4; ++jc ) {
      // corner position projected onto the preshower
      math::XYZPoint cornerpos = corners[jc] * zPS/p.z();
      // Inflate the size by the size of the PS strips, and by 5% to include ECAL cracks.
      x[jc] = cornerpos.X() + (cornerpos.X()-posxyz.X()) * (0.05 +1.0/fabs((cornerpos.X()-posxyz.X()))*deltaX/2.);
      y[jc] = cornerpos.Y() + (cornerpos.Y()-posxyz.Y()) * (0.05 +1.0/fabs((cornerpos.Y()-posxyz.Y()))*deltaY/2.);
    }//loop corners

    //need to close the polygon in order to use the TMath::IsInside fonction from root lib
    x[4] = x[0];
    y[4] = y[0];

    //Check if the extrapolation point of the track falls within the rechit boundaries
    bool isinside = TMath::IsInside(xPS,yPS,5,x,y);
      
    if( isinside ){
      linkedbyrechit = true;
      break;
    }

  }//loop rechits
  
  if( linkedbyrechit ) {
    double dist = std::sqrt( (xECAL/1000. - xPS/1000.)*(xECAL/1000. - xPS/1000.) 
			  + (yECAL/1000. - yPS/1000.)*(yECAL/1000. - yPS/1000.) );
    return dist;
  } else { 
    return -1.;
  }
  
}
