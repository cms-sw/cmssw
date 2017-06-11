#include "RecoLocalMuon/GEMSegment/plugins/GEMSegmentBuilder.h"
#include "DataFormats/MuonDetId/interface/GEMDetId.h"
#include "DataFormats/GEMRecHit/interface/GEMRecHit.h"
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"
#include "Geometry/GEMGeometry/interface/GEMEtaPartition.h"
#include "RecoLocalMuon/GEMSegment/plugins/GEMSegmentAlgorithmBase.h"
#include "RecoLocalMuon/GEMSegment/plugins/GEMSegmentBuilderPluginFactory.h"

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h" 

GEMSegmentBuilder::GEMSegmentBuilder(const edm::ParameterSet& ps) : geom_(0) {
  
  // Algo name
  algoName = ps.getParameter<std::string>("algo_name");
  
  edm::LogVerbatim("GEMSegmentBuilder")<< "GEMSegmentBuilder algorithm name: " << algoName;
  
  // SegAlgo parameter set
  segAlgoPSet = ps.getParameter<edm::ParameterSet>("algo_pset");
  
  // Ask factory to build this algorithm, giving it appropriate ParameterSet  
  algo = std::unique_ptr<GEMSegmentAlgorithmBase>(GEMSegmentBuilderPluginFactory::get()->create(algoName, segAlgoPSet));
  
}
GEMSegmentBuilder::~GEMSegmentBuilder() {}

void GEMSegmentBuilder::build(const GEMRecHitCollection* recHits, GEMSegmentCollection& oc) {
  	
  edm::LogVerbatim("GEMSegmentBuilder")<< "[GEMSegmentBuilder::build] Total number of rechits in this event: " << recHits->size();
  
  // Let's define the ensemble of GEM devices having the same region, chambers number (phi)
  // different eta partitions and different layers are allowed

  std::map<uint32_t, std::vector<GEMRecHit*> > ensembleRH;
 
  // Loop on the GEM rechit and select the different GEM Ensemble
  for(GEMRecHitCollection::const_iterator it2 = recHits->begin(); it2 != recHits->end(); ++it2) {
    // GEM Ensemble is defined by assigning all the GEMDetIds of the same "superchamber" 
    // (i.e. region same, chamber same) to the DetId of the first layer

    // here a reference GEMDetId is created: named "id"
    // - Ring 1 (no other rings available for GEM)
    // - Layer 1 = reference layer (effective layermask)
    // - Roll 0  = reference roll  (effective rollmask)
    // - Station == 1 (GE1/1) or == 2 (GE2/1)
    // this reference id serves to link all GEMEtaPartitions
    // and will also be used to determine the GEMSuperChamber 
    // to which the GEMSegment is assigned (done inside GEMSegAlgoXX)
    GEMDetId id(it2->gemId().region(),1,it2->gemId().station(),0,it2->gemId().chamber(),0);
    // save current GEMRecHit in vector associated to the reference id
    ensembleRH[id.rawId()].push_back(it2->clone());
  }

  // Loop on the entire map <ref id, vector of GEMRecHits>
  for(auto enIt=ensembleRH.begin(); enIt != ensembleRH.end(); ++enIt) {
    
    std::vector<const GEMRecHit*> gemRecHits;
    std::map<uint32_t,const GEMEtaPartition* > ens;

    // all detIds have been assigned to the to chamber
    const GEMSuperChamber* chamber = geom_->superChamber(enIt->first);
    for(auto rechit = enIt->second.begin(); rechit != enIt->second.end(); ++rechit) {
      gemRecHits.push_back(*rechit);
      ens[(*rechit)->gemId()]=geom_->etaPartition((*rechit)->gemId());
    }    

    #ifdef EDM_ML_DEBUG // have lines below only compiled when in debug mode
    LogTrace("GEMSegmentBuilder") << "[GEMSegmentBuilder::build] -----------------------------------------------------------------------------"; 
    LogTrace("GEMSegmentBuilder") << "[GEMSegmentBuilder::build] found " << gemRecHits.size() << " rechits in GEM Super Chamber " << chamber->id()<<" ::"; 
    for (auto rh=gemRecHits.begin(); rh!=gemRecHits.end(); ++rh){
      auto gemid = (*rh)->gemId();
      // auto rhr = gemGeom->etaPartition(gemid);
      auto rhLP = (*rh)->localPosition();
      // auto rhGP = rhr->toGlobal(rhLP);
      // no sense to print local y because local y here is in the roll reference frame
      // in the roll reference frame the local y of a rechit is always the middle of the roll, and hence equal to 0.0
      LogTrace("GEMSegmentBuilder") << "[RecHit :: Loc x = "<<std::showpos<<std::setw(9)<<rhLP.x() /*<<" Loc y = "<<std::showpos<<std::setw(9)<<rhLP.y()*/
					    <<" BX = "<<(*rh)->BunchX()<<" -- "<<gemid.rawId()<<" = "<<gemid<<" ]";
    }
    #endif


    GEMSegmentAlgorithmBase::GEMEnsemble ensemble(std::pair<const GEMSuperChamber*,      std::map<uint32_t,const GEMEtaPartition*> >(chamber,ens));
    
    #ifdef EDM_ML_DEBUG // have lines below only compiled when in debug mode
    LogTrace("GEMSegmentBuilder") << "[GEMSegmentBuilder::build] run the segment reconstruction algorithm now";
    #endif

    // given the superchamber select the appropriate algo... and run it
    std::vector<GEMSegment> segv = algo->run(ensemble, gemRecHits);
    #ifdef EDM_ML_DEBUG // have lines below only compiled when in debug mode
    LogTrace("GEMSegmentBuilder") << "[GEMSegmentBuilder::build] found " << segv.size() ;
    #endif

    GEMDetId mid(enIt->first);    

    #ifdef EDM_ML_DEBUG // have lines below only compiled when in debug mode
    LogTrace("GEMSegmentBuilder") << "[GEMSegmentBuilder::build] found " << segv.size() << " segments in GEM Super Chamber " << mid;
    LogTrace("GEMSegmentBuilder") << "[GEMSegmentBuilder::build] -----------------------------------------------------------------------------"; 
    #endif
    
    // Add the segments to master collection
    oc.put(mid, segv.begin(), segv.end());
  }
}

void GEMSegmentBuilder::setGeometry(const GEMGeometry* geom) {
  geom_ = geom;
}
