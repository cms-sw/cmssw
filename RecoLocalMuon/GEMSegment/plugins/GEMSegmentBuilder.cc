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
  std::string algoName = ps.getParameter<std::string>("algo_name");
  
  LogDebug("GEMSegmentBuilder")<< "GEMSegmentBuilder algorithm name: " << algoName;
  
  // SegAlgo parameter set
  edm::ParameterSet segAlgoPSet = ps.getParameter<edm::ParameterSet>("algo_pset");
  
  // Ask factory to build this algorithm, giving it appropriate ParameterSet  
  algo = GEMSegmentBuilderPluginFactory::get()->create(algoName, segAlgoPSet);
  
}
GEMSegmentBuilder::~GEMSegmentBuilder() {
  delete algo;
}

void GEMSegmentBuilder::build(const GEMRecHitCollection* recHits, GEMSegmentCollection& oc) {
  	
  LogDebug("GEMSegmentBuilder")<< "Total number of rechits in this event: " << recHits->size();
  
  // Let's define the ensemble of GEM devices having the same region, chambers number (phi), and eta partition
  // and layer run from 1 to number of layer. This is not the definition of one chamber... and indeed segments
  // could in principle run in different way... The concept of the DetLayer would be more appropriate...

  std::map<uint32_t, std::vector<GEMRecHit*> > ensembleRH;
    
  // Loop on the GEM rechit and select the different GEM Ensemble
  for(GEMRecHitCollection::const_iterator it2 = recHits->begin(); it2 != recHits->end(); ++it2) {
    // GEM Ensemble is defined by assigning all the GEMDetIds of the same "superchamber" 
    // (i.e. region same, chamber same) to the DetId of the first layer

    // here a reference GEMDetId is created: named "id"
    // - Ring 1 (no other rings available for GEM)
    // - Layer 1 = reference layer (effective layermask)
    // - Roll 0  = reference roll  (effective rollmask)
    // - GE2/1 Long & Short needs to be combined => station = 2,3 ==> 3
    // - Station == 1 (GE1/1) or == 3 (GE2/1)
    // this reference id serves to link all GEMEtaPartitions
    // and will also be used to determine the GEMChamber 
    // to which the GEMSegment is assigned (done inside GEMSegAlgoXX)
    int station = 0; 
    if(it2->gemId().station()==1) station=1;
    else if(it2->gemId().station()==2 || it2->gemId().station()==3) station=3;
    GEMDetId id(it2->gemId().region(),1,station,1,it2->gemId().chamber(),0);
    LogDebug("GEMSegmentBuilder") << "GEM Reference id :: "<<id<< " = " << id.rawId();

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

    // all detIds have been assigned to the according detId with layer 1, roll 0 
    // which is not a GEMEtaPartition, the next line will crash therefore
    // const GEMEtaPartition* firstlayer = geom_->etaPartition(enIt->first); 
    // therefore just take the GEMEtaPartition of the first rechit in the map
    std::vector<GEMRecHit* > pp = enIt->second;
    std::vector<GEMRecHit*>::iterator ppit = pp.begin();
    GEMRecHit * pphit = (*ppit);
    const GEMEtaPartition* firstlayer = geom_->etaPartition(pphit->gemId());
    for(auto rechit = enIt->second.begin(); rechit != enIt->second.end(); ++rechit) {
      gemRecHits.push_back(*rechit);
      ens[(*rechit)->gemId()]=geom_->etaPartition((*rechit)->gemId());
    }    
    GEMSegmentAlgorithm::GEMEnsemble ensemble(std::pair<const GEMEtaPartition*, std::map<uint32_t,const GEMEtaPartition*> >(firstlayer,ens));
    
    LogDebug("GEMSegmentBuilder") << "found " << gemRecHits.size() << " rechits in chamber " << firstlayer->id();
    LogDebug("GEMSegmentBuilder") << "run the segment reconstruction algorithm now";
    // given the chamber select the appropriate algo... and run it
    std::vector<GEMSegment> segv = algo->run(ensemble, gemRecHits);
    GEMDetId mid(enIt->first);
    LogDebug("GEMSegmentBuilder") << "found " << segv.size() << " segments in chamber " << mid;
    
    // Add the segments to master collection
    oc.put(mid, segv.begin(), segv.end());
  }
}

void GEMSegmentBuilder::setGeometry(const GEMGeometry* geom) {
  geom_ = geom;
}

