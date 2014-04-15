#include "RecoParticleFlow/PFClusterProducer/interface/PFHcalSuperClusterInit.h"
#include "DataFormats/GeometryVector/interface/Pi.h"
#include "DataFormats/Common/interface/PtrVector.h"
#include "DataFormats/ParticleFlowReco/interface/PFSuperCluster.h"
#include "RecoParticleFlow/PFClusterProducer/interface/PFHcalSuperClusterAlgo.h"

using namespace std;
using namespace reco;

void PFHcalSuperClusterInit::initialize( PFSuperCluster & supercluster,  edm::PtrVector<reco::PFCluster> const & clusters){
  
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
    const std::vector< reco::PFRecHitFraction >& pfhitsandfracs = clusters[ic]->recHitFractions();
    double clusterEnergy = clusters[ic]->energy();
    superClusterEnergy+=clusterEnergy;
    if (clusterEnergy>=maxClusterEnergy) {
      maxClusterLayer=clusters[ic]->layer();
      maxClusterColor=clusters[ic]->color();
      maxClusterCaloId=clusters[ic]->caloID();
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

    supercluster.addRecHitFraction( frac ); 

//    addHitAndFraction( recHitDetId[ir], fraction );
//    cout << " Supercluster rechits detid/fraction: " << recHitDetId[ir] << " / " << fraction <<endl;
  }
//  recHitDetId.clear();
  recHitEnergy.clear();
  recHitRef.clear();

  supercluster.setLayer( maxClusterLayer );
  supercluster.setEnergy( superClusterEnergy );
  supercluster.setCaloId( maxClusterCaloId );
  supercluster.setSeed( maxHitEnergy );
  supercluster.setAlgoId( CaloCluster::particleFlow );
  supercluster.setColor( maxClusterColor );

  double numeratorEta = 0.0;
  double numeratorPhi = 0.0;
  double numeratorRho = 0.0;
  double denominator = 0.0;
  double posEta = 0.0;
  double posPhi = 0.0;
  double posRho = 1.0;
  double w0_ = 4.2;
  if (superClusterEnergy>0.0 && supercluster.recHitFractions().size()>0) {
    const std::vector <reco::PFRecHitFraction >& pfhitsandfracs = supercluster.recHitFractions();
    for (std::vector<reco::PFRecHitFraction>::const_iterator it = pfhitsandfracs.begin(); it != pfhitsandfracs.end(); ++it) {
      const reco::PFRecHitRef rechit = it->recHitRef();
      double hitEta = rechit->positionREP().Eta();
      double hitPhi = rechit->positionREP().Phi();
      while (hitPhi > +Geom::pi()) { hitPhi -= Geom::twoPi(); }
      while (hitPhi < -Geom::pi()) { hitPhi += Geom::twoPi(); }
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

  PFCluster::REPPoint posEtaPhiRho(posRho,posEta,posPhi);

  math::XYZPoint superClusterPosition(posEtaPhiRho.X(),posEtaPhiRho.Y(),posEtaPhiRho.Z());

  supercluster.setPosition( superClusterPosition );
  supercluster.calculatePositionREP();

 return;
}

