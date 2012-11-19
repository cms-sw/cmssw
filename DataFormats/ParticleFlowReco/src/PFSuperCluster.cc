#include "DataFormats/ParticleFlowReco/interface/PFSuperCluster.h"
#include "DataFormats/GeometryVector/interface/Pi.h"


using namespace std;
using namespace reco;

PFSuperCluster::PFSuperCluster(const std::vector< reco::PFCluster >  clusters) {

  clusters_ = clusters;

//  std::vector< unsigned>  recHitDetId;
  std::vector< double>  recHitEnergy;
  std::vector< PFRecHitRef>  recHitRef;
  double superClusterEnergy = 0.0;
//  unsigned maxHitDetId = 0;
  double maxHitEnergy = 0.0;
  CaloID maxClusterCaloId;
  PFLayer::Layer maxClusterLayer=PFLayer::HCAL_BARREL1;
  double maxClusterEnergy = 0.0;
  int maxClusterColor=2;
//  cout << " Supercluster clusters: " << clusters.size() <<endl;
  for (unsigned short ic=0; ic<clusters.size();++ic) {
//    const std::vector< std::pair<DetId, float> > & hitsandfracs =
//          clusters[ic].hitsAndFractions();
    const std::vector< reco::PFRecHitFraction >& pfhitsandfracs = clusters[ic].recHitFractions();
    double clusterEnergy = clusters[ic].energy();
    superClusterEnergy+=clusterEnergy;
    if (clusterEnergy>=maxClusterEnergy) {
      maxClusterLayer=clusters[ic].layer();
      maxClusterColor=clusters[ic].color();
      maxClusterCaloId=clusters[ic].caloID();
      maxClusterEnergy=clusterEnergy;
    }
    for (unsigned ihandf=0; ihandf<pfhitsandfracs.size(); ihandf++) {
//      unsigned hitDetId = pfhitsandfracs[ihandf].recHitRef()->detId();
      double hitEnergy = pfhitsandfracs[ihandf].fraction()*clusterEnergy;
//      recHitDetId.push_back(hitDetId);
      recHitEnergy.push_back(hitEnergy);
      recHitRef.push_back(pfhitsandfracs[ihandf].recHitRef());
      if(hitEnergy>=maxHitEnergy) {
//        maxHitDetId=hitDetId;
        maxHitEnergy=hitEnergy;
      }
    }
//    delete hitsandfracs;
  }

  for (unsigned short ir=0; ir<recHitEnergy.size();++ir) {
    double fraction=1.0;
    if( superClusterEnergy >0.0) {
      fraction  = recHitEnergy[ir]/superClusterEnergy;
    }
    reco::PFRecHitFraction frac(recHitRef[ir],fraction);

    addRecHitFraction( frac ); 

//    addHitAndFraction( recHitDetId[ir], fraction );
//    cout << " Supercluster rechits detid/fraction: " << recHitDetId[ir] << " / " << fraction <<endl;
  }
//  recHitDetId.clear();
  recHitEnergy.clear();
  recHitRef.clear();

  setLayer( maxClusterLayer );
  setEnergy( superClusterEnergy );
  setCaloId( maxClusterCaloId );
  setSeed( maxHitEnergy );
  setAlgoId( CaloCluster::particleFlow );
  setColor( maxClusterColor );

  double numeratorEta = 0.0;
  double numeratorPhi = 0.0;
  double numeratorRho = 0.0;
  double denominator = 0.0;
  double posEta = 0.0;
  double posPhi = 0.0;
  double posRho = 1.0;
  double w0_ = 4.2;
  if (superClusterEnergy>0.0 && recHitFractions().size()>0) {
    const std::vector <reco::PFRecHitFraction >& pfhitsandfracs = recHitFractions();
    for (std::vector<reco::PFRecHitFraction>::const_iterator it = pfhitsandfracs.begin(); it != pfhitsandfracs.end(); ++it) {
      const reco::PFRecHitRef rechit = it->recHitRef();
      double hitEta = rechit->positionREP().Eta();
      double hitPhi = rechit->positionREP().Phi();
      if (hitPhi > + Geom::pi()) { hitPhi = Geom::twoPi() - hitPhi; }
      if (hitPhi < - Geom::pi()) { hitPhi = Geom::twoPi() - hitPhi; }
      double hitRho = rechit->positionREP().Rho();
      double hitEnergy = rechit->energy();
      const double w = std::max(0.0, w0_ + log(hitEnergy / superClusterEnergy));
      denominator += w;
      numeratorEta += w*hitEta;
      numeratorPhi += w*hitPhi;
      numeratorRho += w*hitRho;
    }
    posEta = numeratorEta/denominator;
    posPhi = numeratorPhi/denominator;
    posRho = numeratorRho/denominator;
  }

  REPPoint posEtaPhiRho(posRho,posEta,posPhi);

  math::XYZPoint superClusterPosition(posEtaPhiRho.X(),posEtaPhiRho.Y(),posEtaPhiRho.Z());

  setPosition( superClusterPosition );
  calculatePositionREP();

}

void PFSuperCluster::reset() {
  
  PFCluster::reset();
  clusters_.clear();

}

PFSuperCluster& PFSuperCluster::operator=(const PFSuperCluster& other) {

  PFCluster::operator=((PFCluster)other); 
  clusters_ = other.clusters_;

  return *this;
}


std::ostream& reco::operator<<(std::ostream& out, 
                               const PFSuperCluster& cluster) {
  
  if(!out) return out;

  const math::XYZPoint&  pos = cluster.position();
  const PFCluster::REPPoint&  posrep = cluster.positionREP();
  const std::vector< reco::PFRecHitFraction >& fracs =
    cluster.recHitFractions();

  out<<"PFSuperCluster "
     <<", clusters: "<<cluster.clusters().size()
     <<", layer: "<<cluster.layer()
     <<"\tE = "<<cluster.energy()
     <<"\tXYZ: "
     <<pos.X()<<","<<pos.Y()<<","<<pos.Z()<<" | "
     <<"\tREP: "
     <<posrep.Rho()<<","<<posrep.Eta()<<","<<posrep.Phi()<<" | "
     <<fracs.size()<<" rechits";

  for(unsigned i=0; i<fracs.size(); i++) {
    // PFRecHit is not available, print the detID
    if( !fracs[i].recHitRef().isAvailable() )
      out<<cluster.printHitAndFraction(i)<<", ";
    else
      out<<fracs[i]<<", ";
  }

  
  return out;
}
