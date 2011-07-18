// system include files
#include <vector>
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

// Reconstruction Classes
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/TruncatedPyramid.h"
#include "Geometry/EcalAlgo/interface/EcalPreshowerGeometry.h"
#include "Geometry/CaloTopology/interface/EcalPreshowerTopology.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "DataFormats/EcalDetId/interface/ESDetId.h"
#include <fstream>
#include <sstream>

#include "RecoEcal/EgammaClusterProducers/interface/PreshowerClusterShapeProducer.h"

using namespace std;
using namespace reco;
using namespace edm;
///----

PreshowerClusterShapeProducer::PreshowerClusterShapeProducer(const ParameterSet& ps) {
  // use configuration file to setup input/output collection names
  // Parameters to identify the hit collections
  preshHitProducer_   = ps.getParameter<edm::InputTag>("preshRecHitProducer");
  endcapSClusterProducer_   = ps.getParameter<edm::InputTag>("endcapSClusterProducer");

  PreshowerClusterShapeCollectionX_ = ps.getParameter<string>("PreshowerClusterShapeCollectionX");
  PreshowerClusterShapeCollectionY_ = ps.getParameter<string>("PreshowerClusterShapeCollectionY");

  produces< reco::PreshowerClusterShapeCollection >(PreshowerClusterShapeCollectionX_);
  produces< reco::PreshowerClusterShapeCollection >(PreshowerClusterShapeCollectionY_);
  
  float preshStripECut = ps.getParameter<double>("preshStripEnergyCut");
  int preshNst = ps.getParameter<int>("preshPi0Nstrip");
  
  string debugString = ps.getParameter<string>("debugLevel");


  string tmpPath = ps.getUntrackedParameter<string>("pathToWeightFiles","RecoEcal/EgammaClusterProducers/data/");
  
  presh_pi0_algo = new EndcapPiZeroDiscriminatorAlgo(preshStripECut, preshNst, tmpPath.c_str()); 

  LogTrace("EcalClusters") << "PreshowerClusterShapeProducer:presh_pi0_algo class instantiated " ;
  
  
  nEvt_ = 0;

}


PreshowerClusterShapeProducer::~PreshowerClusterShapeProducer() {
   delete presh_pi0_algo;
}


void PreshowerClusterShapeProducer::produce(Event& evt, const EventSetup& es) {

  ostringstream ostr; // use this stream for all messages in produce

  LogTrace("EcalClusters") << "\n .......  Event " << evt.id() << " with Number = " <<  nEvt_+1 << " is analyzing ....... " ;


  Handle< EcalRecHitCollection >   pRecHits;
  Handle< SuperClusterCollection > pSuperClusters;

  // get the ECAL -> Preshower geometry and topology:
  ESHandle<CaloGeometry> geoHandle;
  es.get<CaloGeometryRecord>().get(geoHandle);
  const CaloSubdetectorGeometry *geometry = geoHandle->getSubdetectorGeometry(DetId::Ecal, EcalPreshower);
  const CaloSubdetectorGeometry *& geometry_p = geometry;


  // create an auto_ptr to a PreshowerClusterShapeCollection
  std::auto_ptr< reco::PreshowerClusterShapeCollection > ps_cl_for_pi0_disc_x(new reco::PreshowerClusterShapeCollection);
  std::auto_ptr< reco::PreshowerClusterShapeCollection > ps_cl_for_pi0_disc_y(new reco::PreshowerClusterShapeCollection);


  CaloSubdetectorTopology* topology_p=0;
  if (geometry)
      topology_p = new EcalPreshowerTopology(geoHandle);

  
  // fetch the Preshower product (RecHits)
  evt.getByLabel( preshHitProducer_, pRecHits);
  // pointer to the object in the product
  const EcalRecHitCollection* rechits = pRecHits.product(); 
  
  LogTrace("EcalClusters") << "PreshowerClusterShapeProducer: ### Total # of preshower RecHits: " << rechits->size() ;
  
  //  if ( rechits->size() <= 0 ) return;
  
  // make the map of Preshower rechits:
  map<DetId, EcalRecHit> rechits_map;
  EcalRecHitCollection::const_iterator it;
  for (it = rechits->begin(); it != rechits->end(); it++) {
     rechits_map.insert(make_pair(it->id(), *it));
  }
  
  LogTrace("EcalClusters") << "PreshowerClusterShapeProducer: ### Preshower RecHits_map of size " << rechits_map.size() <<" was created!" ;
  
  reco::PreshowerClusterShapeCollection ps_cl_x, ps_cl_y;

  //make cycle over Photon Collection
  int SC_index  = 0;
//  Handle<PhotonCollection> correctedPhotonHandle; 
//  evt.getByLabel(photonCorrCollectionProducer_, correctedPhotonCollection_ , correctedPhotonHandle);
//  const PhotonCollection corrPhoCollection = *(correctedPhotonHandle.product());
//  cout << " Photon Collection size : " << corrPhoCollection.size() << endl;

  evt.getByLabel(endcapSClusterProducer_, pSuperClusters);
  const reco::SuperClusterCollection* SClusts = pSuperClusters.product();
  LogTrace("EcalClusters") << "### Total # Endcap Superclusters: " << SClusts->size() ;

  SuperClusterCollection::const_iterator it_s;
  for ( it_s=SClusts->begin();  it_s!=SClusts->end(); it_s++ ) {

      SuperClusterRef it_super(reco::SuperClusterRef(pSuperClusters,SC_index));
      
      float SC_Et   = it_super->energy()*sin(2*atan(exp(-it_super->eta())));
      float SC_eta  = it_super->eta();
      float SC_phi  = it_super->phi();

      LogTrace("EcalClusters") << "PreshowerClusterShapeProducer: superCl_E = " << it_super->energy() << " superCl_Et = " << SC_Et << " superCl_Eta = " << SC_eta << " superCl_Phi = " << SC_phi ;

      
      if(fabs(SC_eta) >= 1.65 && fabs(SC_eta) <= 2.5) 
	{  //  Use Preshower region only
	  if (geometry)
	    {
	      const GlobalPoint pointSC(it_super->x(),it_super->y(),it_super->z()); // get the centroid of the SC
	      LogTrace("EcalClusters") << "SC centroind = " << pointSC ;
	      
	      // Get the Preshower 2-planes RecHit vectors associated with the given SC
	      
	      DetId tmp_stripX = (dynamic_cast<const EcalPreshowerGeometry*>(geometry_p))->getClosestCellInPlane(pointSC, 1);
	      DetId tmp_stripY = (dynamic_cast<const EcalPreshowerGeometry*>(geometry_p))->getClosestCellInPlane(pointSC, 2);
	      ESDetId stripX = (tmp_stripX == DetId(0)) ? ESDetId(0) : ESDetId(tmp_stripX);
	      ESDetId stripY = (tmp_stripY == DetId(0)) ? ESDetId(0) : ESDetId(tmp_stripY);
	      
	      vector<float> vout_stripE1 = presh_pi0_algo->findPreshVector(stripX, &rechits_map, topology_p);
	      vector<float> vout_stripE2 = presh_pi0_algo->findPreshVector(stripY, &rechits_map, topology_p);
	      
	      LogTrace("EcalClusters") << "PreshowerClusterShapeProducer : ES Energy vector associated to the given SC = " ;
	      for(int k1=0;k1<11;k1++) {
		LogTrace("EcalClusters") << vout_stripE1[k1] << " " ;
	      }
	      
	      for(int k1=0;k1<11;k1++) {
		LogTrace("EcalClusters")  << vout_stripE2[k1] << " " ;
	      } 
	      
	      reco::PreshowerClusterShape ps1 = reco::PreshowerClusterShape(vout_stripE1,1);
	      ps1.setSCRef(it_super);
	      ps_cl_x.push_back(ps1);
	      
	      reco::PreshowerClusterShape ps2 = reco::PreshowerClusterShape(vout_stripE2,2);
	      ps2.setSCRef(it_super);
	      ps_cl_y.push_back(ps2);
	      
	    }
	  SC_index++;
	} // end of cycle over Endcap SC       
  } 
  // put collection of PreshowerClusterShape in the Event:
  ps_cl_for_pi0_disc_x->assign(ps_cl_x.begin(), ps_cl_x.end());
  ps_cl_for_pi0_disc_y->assign(ps_cl_y.begin(), ps_cl_y.end());
  
  evt.put(ps_cl_for_pi0_disc_x, PreshowerClusterShapeCollectionX_);
  evt.put(ps_cl_for_pi0_disc_y, PreshowerClusterShapeCollectionY_);  
  LogTrace("EcalClusters") << "PreshowerClusterShapeCollection added to the event" ;
  
  if (topology_p)
    delete topology_p;

  nEvt_++;

  LogDebug("PiZeroDiscriminatorDebug") << ostr.str();
}
