#include "RecoLocalCalo/HGCalRecAlgos/interface/ClusterTools.h"

#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"

using namespace hgcal;

ClusterTools::ClusterTools(const edm::ParameterSet& conf, 
                           edm::ConsumesCollector& sumes):
  eetok( sumes.consumes<HGCRecHitCollection>(conf.getParameter<edm::InputTag>("HGCEEInput")) ),
  fhtok( sumes.consumes<HGCRecHitCollection>(conf.getParameter<edm::InputTag>("HGCFHInput")) ),
  bhtok( sumes.consumes<HGCRecHitCollection>(conf.getParameter<edm::InputTag>("HGCBHInput")) ) {
}

void ClusterTools::getEvent(const edm::Event& ev) {
  rhtools_.getEvent(ev);
  edm::Handle<HGCRecHitCollection> temp;
  ev.getByToken(eetok, temp);
  eerh_ = temp.product();
  ev.getByToken(fhtok, temp);
  fhrh_ = temp.product();
  ev.getByToken(bhtok, temp);
  bhrh_ = temp.product();
}

void ClusterTools::getEventSetup(const edm::EventSetup& es) {
  rhtools_.getEventSetup(es);
}

float ClusterTools::getClusterHadronFraction(const reco::CaloCluster& clus) const {
  float energy=0.f, energyHad=0.f;
  const auto& hits = clus.hitsAndFractions();
  for( const auto& hit : hits ) {
    const auto& id = hit.first;
    const float fraction = hit.second;
    if( id.det() == DetId::Forward ) {
      switch( id.subdetId() ) {
      case HGCEE:
        energy += eerh_->find(id)->energy()*fraction;
        break;
      case HGCHEF:
        {
          const float temp = fhrh_->find(id)->energy();
          energy += temp*fraction;
          energyHad += temp*fraction;
        }
        break;
      default:
        throw cms::Exception("HGCalClusterTools")
          << " Cluster contains hits that are not from HGCal! " << std::endl;
      }
    } else if ( id.det() == DetId::Hcal && id.subdetId() == HcalEndcap ) {
      const float temp = bhrh_->find(id)->energy();
      energy += temp*fraction;
      energyHad += temp*fraction;
    } else {
      throw cms::Exception("HGCalClusterTools")
        << " Cluster contains hits that are not from HGCal! " << std::endl;
    }    
  }
  float fraction = -1.f;
  if( energy > 0.f ) {
    fraction = energyHad/energy;
  }
  return fraction;
}


math::XYZPoint ClusterTools::getMultiClusterPosition(const reco::HGCalMultiCluster& clu, double vz) const {
  if( clu.clusters().size() == 0 ) return math::XYZPoint();
  double acc_rho = 0.0;
  double acc_eta = 0.0;
  double acc_phi = 0.0;
  double totweight = 0.;
  for( const auto& ptr : clu.clusters() ) {   
    const double x = ptr->x();    
    const double y = ptr->y();
    const float point_r = std::sqrt(x*x + y*y);    
    const double point_z = ptr->z()-vz;
    const double weight = ptr->energy() * ptr->size();
    assert((y != 0. || x != 0.) && "Cluster position somehow in beampipe.");
    assert(point_z != 0. && "Layer-cluster position given as reference point.");
    acc_rho += point_r * weight;
    acc_phi += std::atan2(y,x) * weight;
    acc_eta += -1. * std::log(std::tan(0.5*std::atan2(point_r,point_z))) * weight;
    totweight += weight;
  }
  const double invweight = 1.0/totweight;
  reco::PFCluster::REPPoint temp(acc_rho*invweight,acc_eta*invweight,acc_phi*invweight);
  return math::XYZPoint(temp.x(),temp.y(),temp.z());
}

double ClusterTools::getMultiClusterEnergy(const reco::HGCalMultiCluster& clu) const {
  double acc = 0.0;
  for(const auto& ptr : clu.clusters() ) {
    acc += ptr->energy();
  }
  return acc;
}
