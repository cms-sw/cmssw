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
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/TruncatedPyramid.h"
#include "Geometry/EcalPreshowerAlgo/interface/EcalPreshowerGeometry.h"
#include "Geometry/CaloTopology/interface/EcalPreshowerTopology.h"
#include "Geometry/CaloTopology/interface/EcalEndcapTopology.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDetId/interface/ESDetId.h"
#include <fstream>
#include <sstream>

#include "RecoEcal/EgammaClusterProducers/interface/PiZeroDiscriminatorProducer.h"
#include "DataFormats/EgammaReco/interface/ClusterPi0Discriminator.h"

// Class for Cluster Shape Algorithm
#include "DataFormats/EgammaReco/interface/ClusterShape.h"
#include "DataFormats/EgammaReco/interface/ClusterShapeFwd.h"
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"

#include "TFile.h"

///----

PiZeroDiscriminatorProducer::PiZeroDiscriminatorProducer(const edm::ParameterSet& ps) {
  // use configuration file to setup input/output collection names
  // Parameters to identify the hit collections
  endcapHitProducer_   = ps.getParameter<std::string>("endcapHitProducer");
  endcapHitCollection_ = ps.getParameter<std::string>("endcapHitCollection");

  preshHitProducer_   = ps.getParameter<std::string>("preshRecHitProducer");
  preshHitCollection_ = ps.getParameter<std::string>("preshRecHitCollection");

  float preshStripECut = ps.getParameter<double>("preshStripEnergyCut");
  int preshNst = ps.getParameter<int>("preshPi0Nstrip");

  std::map<std::string,double> providedParameters;  
  providedParameters.insert(std::make_pair("LogWeighted",ps.getParameter<bool>("coretools_logweight")));
  providedParameters.insert(std::make_pair("X0",ps.getParameter<double>("coretools_x0")));
  providedParameters.insert(std::make_pair("T0",ps.getParameter<double>("coretools_t0")));
  providedParameters.insert(std::make_pair("W0",ps.getParameter<double>("coretools_w0")));
  posCalculator_ = PositionCalc(providedParameters);
  shapeAlgo_ = ClusterShapeAlgo(posCalculator_);

  // Name of a SuperClusterCollection to make associations:
  endcapSClusterCollection_ = ps.getParameter<std::string>("endcapSClusterCollection");
  endcapSClusterProducer_   = ps.getParameter<std::string>("endcapSClusterProducer");

  // output collection
  endcapPiZeroDiscriminatorCollection_  = ps.getParameter<std::string>("endcapPiZeroDiscriminatorCollection");
  barrelPiZeroDiscriminatorCollection_  = ps.getParameter<std::string>("barrelPiZeroDiscriminatorCollection");

  std::string debugString = ps.getParameter<std::string>("debugLevel");
  
  if      (debugString == "DEBUG")   debugL_pi0 = EndcapPiZeroDiscriminatorAlgo::pDEBUG;
  else if (debugString == "INFO")    debugL_pi0 = EndcapPiZeroDiscriminatorAlgo::pINFO;
  else                               debugL_pi0 = EndcapPiZeroDiscriminatorAlgo::pERROR;

  std::string tmpPath = ps.getUntrackedParameter<std::string>("pathToWeightFiles","RecoEcal/EgammaClusterProducers/data/");
  //edm::FileInPath pathToFiles(tmpPath);

  presh_pi0_algo = new EndcapPiZeroDiscriminatorAlgo(preshStripECut, preshNst, tmpPath.c_str(), debugL_pi0); // Aris 20/7/2006 

  produces< reco::ClusterPi0DiscriminatorCollection >(endcapPiZeroDiscriminatorCollection_);

  nEvt_ = 0;

}


PiZeroDiscriminatorProducer::~PiZeroDiscriminatorProducer() {
   delete presh_pi0_algo; 
}


void PiZeroDiscriminatorProducer::produce(edm::Event& evt, const edm::EventSetup& es) {

  std::ostringstream ostr; // use this stream for all messages in produce


  if ( debugL_pi0 <= EndcapPiZeroDiscriminatorAlgo::pDEBUG ) 
       ostr << "\n .......  Event # " << nEvt_+1 
                                                         << " is analyzing ....... " << std::endl << std::endl;
  
  edm::Handle< EcalRecHitCollection >   pRecHits;
  edm::Handle< reco::SuperClusterCollection > pSuperClusters;
  
  // get the ECAL -> Preshower geometry and topology:
  edm::ESHandle<CaloGeometry> geoHandle;
  es.get<IdealGeometryRecord>().get(geoHandle);
  const CaloSubdetectorGeometry *geometry = geoHandle->getSubdetectorGeometry(DetId::Ecal, EcalPreshower);
  const CaloSubdetectorGeometry *& geometry_p = geometry;
//  CaloSubdetectorTopology *topology = new EcalPreshowerTopology(geoHandle); 
//  CaloSubdetectorTopology *& topology_p = topology;

   EcalPreshowerTopology topology(geoHandle);
   CaloSubdetectorTopology * topology_p = &topology;

 
  // fetch the Preshower product (RecHits)
  evt.getByLabel( preshHitProducer_, preshHitCollection_, pRecHits);
  // pointer to the object in the product
  const EcalRecHitCollection* rechits = pRecHits.product(); // EcalRecHitCollection hit_collection = *rhcHandle;
  if ( debugL_pi0 == EndcapPiZeroDiscriminatorAlgo::pDEBUG ) ostr << "PiZeroDiscriminatorProducer: ### Total # of preshower RecHits: " 
                                                          << rechits->size() << std::endl;
  // make the map of Preshower rechits:
  std::map<DetId, EcalRecHit> rechits_map;
  EcalRecHitCollection::const_iterator it;
  for (it = rechits->begin(); it != rechits->end(); it++) {
     rechits_map.insert(std::make_pair(it->id(), *it));   
  }
  if ( debugL_pi0 <= EndcapPiZeroDiscriminatorAlgo::pDEBUG ) ostr 
                                << "PiZeroDiscriminatorProducer: ### Preshower RecHits_map of size " 
                                << rechits_map.size() <<" was created!" << std::endl; 

  const CaloSubdetectorGeometry *geometry_pee;
  geometry_pee = geoHandle->getSubdetectorGeometry(DetId::Ecal, EcalEndcap);
  EcalEndcapTopology topology_EE(geoHandle);
  //CaloSubdetectorTopology * topology_p = &topology;

  // fetch the ECAL Endcap product (RecHits)
  edm::Handle<EcalRecHitCollection> rhcHandle;
//  evt.getByLabel("ecalRecHit", "EcalRecHitsEE", rhcHandle);
  evt.getByLabel(endcapHitProducer_, endcapHitCollection_, rhcHandle);
  if (!(rhcHandle.isValid()))
    {
      ostr << "Pi0rejection : could not get a handle on the EcalRecHitCollection!" << std::endl;
    }        
  const EcalRecHitCollection *hit_collection = rhcHandle.product();

 // fetch the product (pSuperClusters)
  evt.getByLabel(endcapSClusterProducer_, endcapSClusterCollection_, pSuperClusters);   
  const reco::SuperClusterCollection* SClusts = pSuperClusters.product();
  if ( debugL_pi0 <= EndcapPiZeroDiscriminatorAlgo::pDEBUG ) ostr <<"### Total # Endcap Superclusters: " << SClusts->size() << std::endl;
  if ( debugL_pi0 == EndcapPiZeroDiscriminatorAlgo::pDEBUG ) ostr << " Making a cycle over Superclusters ..." << std::endl; 

  reco::ClusterPi0DiscriminatorCollection new_Pi0Disc;

  //make cycle over  clusters
  reco::SuperClusterCollection::const_iterator it_super;
  for ( it_super=SClusts->begin();  it_super!=SClusts->end(); it_super++ ) { 
    
      if ( debugL_pi0 <= EndcapPiZeroDiscriminatorAlgo::pDEBUG ) ostr << " superE = " << it_super->energy() 
                                                               << " superETA = " << it_super->eta() 
       		                                               << " superPHI = " << it_super->phi() << std::endl;
       
       float nnoutput = -10.;
       if(fabs(it_super->eta()) > 1.5) { 
          const GlobalPoint pointSC(it_super->x(),it_super->y(),it_super->z()); // get the centroid of the SC
          ostr << "SC centroind = " << pointSC << std::endl;
          reco::BasicClusterRef BS_clus_Id = it_super->seed();
          double SC_seed_energy = it_super->seed()->energy();

          //ClusterShapeAlgo::Initialize(hit_collection, &geoHandle);
          reco::ClusterShape TestShape = shapeAlgo_.Calculate(*BS_clus_Id,hit_collection,geometry_pee,&topology_EE);
          double SC_seed_Shape_E1 = TestShape.eMax();
          double SC_seed_Shape_E3x3 = TestShape.e3x3();
          double SC_seed_Shape_E5x5 = TestShape.e5x5();
       
          ostr << "BC energy_max = " <<  SC_seed_energy << std::endl;
          ostr << "ClusterShape  E1_max = " <<   SC_seed_Shape_E1 << std::endl;
          ostr << "ClusterShape  E3x3_max = " <<   SC_seed_Shape_E3x3 <<  std::endl;
          ostr << "ClusterShape  E5x5_max = " <<   SC_seed_Shape_E5x5 << std::endl;

          DetId tmp_stripX = (dynamic_cast<const EcalPreshowerGeometry*>(geometry_p))->getClosestCellInPlane(pointSC, 1);
          DetId tmp_stripY = (dynamic_cast<const EcalPreshowerGeometry*>(geometry_p))->getClosestCellInPlane(pointSC, 2);
          ESDetId stripX = (tmp_stripX == DetId(0)) ? ESDetId(0) : ESDetId(tmp_stripX);
          ESDetId stripY = (tmp_stripY == DetId(0)) ? ESDetId(0) : ESDetId(tmp_stripY); 

          std::vector<float> vout_stripE1 = presh_pi0_algo->findPreshVector(stripX, &rechits_map, topology_p);
          std::vector<float> vout_stripE2 = presh_pi0_algo->findPreshVector(stripY, &rechits_map, topology_p);
//          std::vector<float> vout_stripE1;
//          std::vector<float> vout_stripE2;
//          presh_pi0_algo->findPreshVector(stripX, &rechits_map, topology_p, vout_stripE1);
//          presh_pi0_algo->findPreshVector(stripY, &rechits_map, topology_p, vout_stripE2);

          float nn_invar_presh[25];
          presh_pi0_algo->calculateNNInputVariables(vout_stripE1, vout_stripE2,
                                                 SC_seed_Shape_E1, SC_seed_Shape_E3x3, SC_seed_Shape_E5x5,
                                                 nn_invar_presh);

          float SE_et = it_super->energy()*sin(2*atan(exp(-it_super->eta())));
          ostr << "SC Et = " << SE_et << std::endl;
          nnoutput = presh_pi0_algo->GetNNOutput(SE_et, nn_invar_presh);
       }
       ostr << "PreshowerPi0NNProducer: SC NNOUTPUT = " <<  nnoutput << " and SC contains " 
                 << it_super->clustersSize() << " BCs" << std::endl;						       
						       
       reco::ClusterPi0Discriminator Pi0_Disc((double)nnoutput, 0., 0.);
       new_Pi0Disc.push_back(Pi0_Disc);

   } // end of cycle over SCs

   // put new collection of corrected super clusters to the event
   std::auto_ptr< reco::ClusterPi0DiscriminatorCollection > Pi0DiscCollection(new reco::ClusterPi0DiscriminatorCollection);
   Pi0DiscCollection->assign(new_Pi0Disc.begin(), new_Pi0Disc.end());
   evt.put(Pi0DiscCollection, endcapPiZeroDiscriminatorCollection_);
   if ( debugL_pi0 <= EndcapPiZeroDiscriminatorAlgo::pDEBUG ) ostr << "Corrected SClusters added to the event" << std::endl;

   nEvt_++;

   LogDebug("PiZeroDiscriminatorDebug") << ostr.str();


}

