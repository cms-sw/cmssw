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


