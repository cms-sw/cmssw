#include "RecoLocalCalo/HGCalRecAlgos/interface/ClusterTools.h"

#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"

#include "vdt/vdtMath.h"

#include <iostream>

using namespace hgcal;
ClusterTools::ClusterTools(){
}

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
  /// *******ALL CRAP*********
  // double acc_rho = 0.0;
  // double acc_eta = 0.0;
  // double acc_phi = 0.0;
  // double totweight = 0.;
  // for( const auto& ptr : clu.clusters() ) {   
  //   const double x = ptr->x();    
  //   const double y = ptr->y();
  //   const double point_r2 = (x*x + y*y);    
  //   const double point_z = ptr->z()-vz;
  //   const double point_h = std::sqrt(point_r2 + point_z*point_z);
  //   const double weight = ptr->energy() * ptr->size();
  //   assert((y != 0. || x != 0.) && "Cluster position somehow in beampipe.");
  //   assert(point_z != 0.f && "Layer-cluster position given as reference point.");
  //   const double point_r = std::sqrt(point_r2);
  //   acc_rho += point_r * weight;
  //   acc_phi += vdt::fast_atan2(y,x) * weight;
  //   acc_eta += -1. * vdt::fast_log(point_r/(point_z + point_h)) * weight;
  //   totweight += weight;
  // }
  // const double invweight = 1.0/totweight;
  // reco::PFCluster::REPPoint temp(acc_rho*invweight,acc_eta*invweight,acc_phi*invweight);

  double acc_x = 0.0;
  double acc_y = 0.0;
  double acc_z = 0.0;
  double totweight = 0.;
  for( const auto& ptr : clu.clusters() ) {   
    const double weight = ptr->energy() * ptr->size();
    acc_x += ptr->x()*weight;    
    acc_y += ptr->y()*weight;    
    acc_z += ptr->z()*weight; 
    totweight += weight;   
  }
  acc_x /= totweight;
  acc_y /= totweight;
  acc_z /= totweight;
  float slope = sqrt(acc_x*acc_x+acc_y*acc_y)/acc_z;
  acc_x = 0.;
  acc_y = 0.;
  acc_z = 0.;
  double mcenergy = getMultiClusterEnergy(clu);
  std::vector<size_t> es = sort_by_z(clu); //sorted by increasing absolute z
  //  std::cout << " multicluster with " << es.size() << " clusters" << std::endl;
  for(unsigned int i = 0; i < es.size(); i++ ) {   
    if(clu.clusters()[es[i]]->energy()/mcenergy<.01) continue; //cutoff < 5% layer contribution
    //    std::cout << "here 0" << std::endl;
    const double weight = clu.clusters()[es[i]]->energy() * clu.clusters()[es[i]]->size();
    acc_x += clu.clusters()[es[i]]->x()*weight;    
    acc_y += clu.clusters()[es[i]]->y()*weight;    
    //    std::cout << "here 1" << std::endl;
    if(i>0)
      {
	// std::cout << " energy cluster " << i << " " << es[i] << " " << clu.clusters()[es[i]]->energy() << std::endl; 
	// std::cout << " energy cluster " << i-1 << " " << es[i-1] << " " << clu.clusters()[es[i-1]]->energy() << std::endl; 
	// std::cout << "z sum " << (clu.clusters()[es[i]]->z()*clu.clusters()[es[i]]->energy()
	// 	   +
	// 	   clu.clusters()[es[i-1]]->z()*clu.clusters()[es[i-1]]->energy())
	// 	  << std::endl;
	// std::cout << "energy sum " << (clu.clusters()[es[i]]->energy()+clu.clusters()[es[i-1]]->energy()) << std::endl;
 	// std::cout << "z weighted average cluster " << es[i] << " " 
	// 	  << (clu.clusters()[es[i]]->z()*clu.clusters()[es[i]]->energy()+
	// 	      clu.clusters()[es[i-1]]->z()*clu.clusters()[es[i-1]]->energy())/
	//   (clu.clusters()[es[i]]->energy()+clu.clusters()[es[i-1]]->energy())
	// 	  << std::endl;
	// std::cout << "z corrected average cluster " << es[i]  << " "
	// 	  << clu.clusters()[es[i]]->z()-
	//   (clu.clusters()[es[i]]->z()- clu.clusters()[es[i-1]]->z())*slope 
	// 	  << std::endl; 
	  
	acc_z += (
		  // (clu.clusters()[es[i]]->z()*clu.clusters()[es[i]]->energy()
		  //  +
		  //  clu.clusters()[es[i-1]]->z()*clu.clusters()[es[i-1]]->energy())
		  // /(clu.clusters()[es[i]]->energy()+clu.clusters()[es[i-1]]->energy())
		  clu.clusters()[es[i]]->z()-
		  (clu.clusters()[es[i]]->z()- clu.clusters()[es[i-1]]->z())*slope
		  -
		  vz
		  )*
	  weight;
	//	std::cout <<" here 1.5 " << std::endl;
      }    
    else
      acc_z += (clu.clusters()[es[i]]->z()-vz-0.5*slope)*weight;    
    //    std::cout << "here 2" << std::endl;
    totweight += weight;
  }
  //  std::cout << "here 3" << std::endl;
  acc_x /= totweight;
  acc_y /= totweight;
  acc_z /= totweight;

  //  acc_z -= 0.5*sqrt(acc_x*acc_x+acc_y*acc_y)/acc_z;
  return math::XYZPoint(acc_x,acc_y,acc_z);
}

int ClusterTools::getLayer(const DetId detid) const {
  return rhtools_.getLayerWithOffset(detid);
}

double ClusterTools::getMultiClusterEnergy(const reco::HGCalMultiCluster& clu) const {
  double acc = 0.0;
  for(const auto& ptr : clu.clusters() ) {
    acc += ptr->energy();
  }
  return acc;
}

bool  ClusterTools::getWidths(const reco::CaloCluster & clus,double & sigmaetaeta, double & sigmaphiphi, double & sigmaetaetal, double & sigmaphiphil ) const{
  
  if (getLayer(clus.hitsAndFractions()[0].first)  > 28) return false;
  const  math::XYZPoint & position(clus.position());
  unsigned nhit=clus.hitsAndFractions().size();

  sigmaetaeta=0.;
  sigmaphiphi=0.;
  sigmaetaetal=0.;
  sigmaphiphil=0.;

  double sumw=0.;
  double sumlogw=0.;

  for (unsigned int ih=0;ih<nhit;++ih) {
    const DetId & id = (clus.hitsAndFractions())[ih].first ;
    if ((clus.hitsAndFractions())[ih].second==0.) continue;

    HGCRecHitCollection::const_iterator theHit; 
    if (id.det()==DetId::Forward && id.subdetId()==HGCEE) {
      const HGCRecHit * theHit = &(*eerh_->find(id));

	GlobalPoint cellPos = rhtools_.getPosition(HGCEEDetId(id));
	double weight = theHit->energy();
	// take w0=2 To be optimized
	double logweight = std::max(0.,2 + log(theHit->energy()/clus.energy()));
	double deltaetaeta2 = (cellPos.eta()-position.eta())*(cellPos.eta()-position.eta());
	double deltaphiphi2 = (cellPos.phi()-position.phi())*(cellPos.phi()-position.phi());
	sigmaetaeta +=  deltaetaeta2* weight;
	sigmaphiphi +=  deltaphiphi2 * weight;
	sigmaetaetal +=  deltaetaeta2* logweight;
	sigmaphiphil +=  deltaphiphi2 * logweight;
	sumw += weight;
	sumlogw += logweight;
      }
    }


  //std::cout << "[HGCALShowerBasedEmIdentification::sigmaetaeta], layer " << ilayer
  //<< " position " << position << " sigmaeta " << sigmaetaeta << std::endl;

  if (sumw==0.) return false;
  
  sigmaetaeta /= sumw;
  sigmaetaeta = sqrt(sigmaetaeta);
  sigmaphiphi /= sumw;
  sigmaphiphi = sqrt(sigmaphiphi);
  
  sigmaetaetal /= sumlogw;
  sigmaetaetal = sqrt(sigmaetaetal);
  sigmaphiphil /= sumlogw;
  sigmaphiphil = sqrt(sigmaphiphil);
  return true;
}
