#include <RecoLocalMuon/GEMSegment/plugins/GEMSegmentBuilder.h>
#include <DataFormats/MuonDetId/interface/GEMDetId.h>
#include <DataFormats/GEMRecHit/interface/GEMRecHit.h>
#include <Geometry/GEMGeometry/interface/GEMGeometry.h>
#include <Geometry/GEMGeometry/interface/GEMEtaPartition.h>
#include <RecoLocalMuon/GEMSegment/plugins/GEMSegmentAlgorithm.h>
#include <RecoLocalMuon/GEMSegment/plugins/GEMSegmentBuilderPluginFactory.h>

#include <FWCore/Utilities/interface/Exception.h>
#include <FWCore/MessageLogger/interface/MessageLogger.h> 

GEMSegmentBuilder::GEMSegmentBuilder(const edm::ParameterSet& ps) : geom_(0) {
  
  // Algo name
  algoName = ps.getParameter<std::string>("algo_name");
  
  edm::LogVerbatim("GEMSegmentBuilder")<< "GEMSegmentBuilder algorithm name: " << algoName;
  
  // SegAlgo parameter set
  segAlgoPSet = ps.getParameter<edm::ParameterSet>("algo_pset");
  
  // Ask factory to build this algorithm, giving it appropriate ParameterSet  
  algo = GEMSegmentBuilderPluginFactory::get()->create(algoName, segAlgoPSet);
  
  // Use GE21Short to have 4 rechits available in GE21?
  useGE21Short = ps.getParameter<bool>("useGE21Short");

}
GEMSegmentBuilder::~GEMSegmentBuilder() {
  delete algo;
}

void GEMSegmentBuilder::build(const GEMRecHitCollection* recHits, GEMSegmentCollection& oc) {
  	
  edm::LogVerbatim("GEMSegmentBuilder")<< "[GEMSegmentBuilder::build] Total number of rechits in this event: " << recHits->size();
  
  // Let's define the ensemble of GEM devices having the same region, chambers number (phi)
  // different eta partitions and different layers are allowed

  std::map<uint32_t, std::vector<GEMRecHit*> > ensembleRH;
 
  // Loop on the GEM rechit and select the different GEM Ensemble
  for(GEMRecHitCollection::const_iterator it2 = recHits->begin(); it2 != recHits->end(); ++it2) {
    // GEM Ensemble is defined by assigning all the GEMDetIds of the same "superchamber" 
    // (i.e. region same, chamber same) to the DetId of the first layer

    // if use of GE21Short is prohibited and gemrechit is in station 2 quit the loop
    if(!useGE21Short && it2->gemId().station()==2) continue;

    // here a reference GEMDetId is created: named "id"
    // - Ring 1 (no other rings available for GEM)
    // - Layer 1 = reference layer (effective layermask)
    // - Roll 0  = reference roll  (effective rollmask)
    // - GE2/1 Long & Short needs to be combined => station = 2,3 ==> 3
    // - Station == 1 (GE1/1) or == 3 (GE2/1)
    // this reference id serves to link all GEMEtaPartitions
    // and will also be used to determine the GEMSuperChamber 
    // to which the GEMSegment is assigned (done inside GEMSegAlgoXX)
    int station = 0; 
    if(it2->gemId().station()==1) station=1;
    else if(it2->gemId().station()==2 || it2->gemId().station()==3) station=3;
    GEMDetId id(it2->gemId().region(),1,station,0,it2->gemId().chamber(),0);
    // edm::LogVerbatim("GEMSegmentBuilder") << "GEM Reference id :: "<<id<< " = " << id.rawId();

    // retrieve vector of GEMRecHits associated to the reference id
    std::vector<GEMRecHit* > pp = ensembleRH[id.rawId()];
    // save current GEMRecHit in vector
    pp.push_back(it2->clone());
    // assign updated vector of GEMRecHits to reference id
    ensembleRH[id.rawId()]=pp;
  }

  // Loop on the entire map <ref id, vector of GEMRecHits>
  for(auto enIt=ensembleRH.begin(); enIt != ensembleRH.end(); ++enIt) {
    
    std::vector<const GEMRecHit*> gemRecHits;
    std::map<uint32_t,const GEMEtaPartition* > ens;

    std::vector<GEMRecHit* > pp = enIt->second;
    std::vector<GEMRecHit*>::iterator ppit = pp.begin();
    GEMRecHit * pphit = (*ppit);
    // !!! important !!! for GE2/1 make that the chamber is always in station 3
    int chambidstat = pphit->gemId().station(); if(chambidstat==2) chambidstat=3;
    // would layer = 0 work? actually it should ...
    GEMDetId chamberid = GEMDetId(pphit->gemId().region(), 1, chambidstat, 0, pphit->gemId().chamber(), 0); 
    const GEMSuperChamber* chamber = geom_->superChamber(chamberid);
    for(auto rechit = enIt->second.begin(); rechit != enIt->second.end(); ++rechit) {
      gemRecHits.push_back(*rechit);
      ens[(*rechit)->gemId()]=geom_->etaPartition((*rechit)->gemId());
    }    

    #ifdef EDM_ML_DEBUG // have lines below only compiled when in debug mode
    edm::LogVerbatim("GEMSegmentBuilder") << "[GEMSegmentBuilder::build] -----------------------------------------------------------------------------"; 
    edm::LogVerbatim("GEMSegmentBuilder") << "[GEMSegmentBuilder::build] found " << gemRecHits.size() << " rechits in GEM Super Chamber " << chamber->id()<<" ::"; 
    for (auto rh=gemRecHits.begin(); rh!=gemRecHits.end(); ++rh){
      auto gemid = (*rh)->gemId();
      // auto rhr = gemGeom->etaPartition(gemid);
      auto rhLP = (*rh)->localPosition();
      // auto rhGP = rhr->toGlobal(rhLP);
      // no sense to print local y because local y here is in the roll reference frame
      // in the roll reference frame the local y of a rechit is always the middle of the roll, and hence equal to 0.0
      edm::LogVerbatim("GEMSegmentBuilder") << "[RecHit :: Loc x = "<<std::showpos<<std::setw(9)<<rhLP.x() /*<<" Loc y = "<<std::showpos<<std::setw(9)<<rhLP.y()*/
					    <<" BX = "<<(*rh)->BunchX()<<" -- "<<gemid.rawId()<<" = "<<gemid<<" ]";
    }
    #endif


    GEMSegmentAlgorithm::GEMEnsemble ensemble(std::pair<const GEMSuperChamber*,      std::map<uint32_t,const GEMEtaPartition*> >(chamber,ens));
    
    edm::LogVerbatim("GEMSegmentBuilder") << "[GEMSegmentBuilder::build] run the segment reconstruction algorithm now";
    // given the superchamber select the appropriate algo... and run it
    std::vector<GEMSegment> segv = algo->run(ensemble, gemRecHits);
    GEMDetId mid(enIt->first);
    edm::LogVerbatim("GEMSegmentBuilder") << "[GEMSegmentBuilder::build] found " << segv.size() << " segments in GEM Super Chamber " << mid;
    edm::LogVerbatim("GEMSegmentBuilder") << "[GEMSegmentBuilder::build] -----------------------------------------------------------------------------"; 
    // Add the segments to master collection
    oc.put(mid, segv.begin(), segv.end());
  }
}

void GEMSegmentBuilder::setGeometry(const GEMGeometry* geom) {
  geom_ = geom;
}

