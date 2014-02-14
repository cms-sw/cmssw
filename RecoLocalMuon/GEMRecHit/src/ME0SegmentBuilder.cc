#include <RecoLocalMuon/GEMRecHit/src/ME0SegmentBuilder.h>
#include <DataFormats/MuonDetId/interface/ME0DetId.h>
#include <DataFormats/GEMRecHit/interface/ME0RecHit.h>
#include <Geometry/GEMGeometry/interface/ME0Geometry.h>
#include <Geometry/GEMGeometry/interface/ME0EtaPartition.h>
#include <RecoLocalMuon/GEMRecHit/src/ME0SegmentAlgorithm.h>
#include <RecoLocalMuon/GEMRecHit/src/ME0SegmentBuilderPluginFactory.h>

#include <FWCore/Utilities/interface/Exception.h>
#include <FWCore/MessageLogger/interface/MessageLogger.h> 

ME0SegmentBuilder::ME0SegmentBuilder(const edm::ParameterSet& ps) : geom_(0) {
  
  // Algo name
  std::string algoName = ps.getParameter<std::string>("algo_name");
  
  LogDebug("ME0Segment|ME0")<< "ME0SegmentBuilder algorithm name: " << algoName;
  
  // SegAlgo parameter set
  edm::ParameterSet segAlgoPSet = ps.getParameter<edm::ParameterSet>("algo_pset");
  
  // Ask factory to build this algorithm, giving it appropriate ParameterSet  
  algo = ME0SegmentBuilderPluginFactory::get()->create(algoName, segAlgoPSet);
  
}
ME0SegmentBuilder::~ME0SegmentBuilder() {
  delete algo;
}

void ME0SegmentBuilder::build(const ME0RecHitCollection* recHits, ME0SegmentCollection& oc) {
  	
  LogDebug("ME0Segment|ME0")<< "Total number of rechits in this event: " << recHits->size();
  
  // Let's define the ensemble of ME0 devices having the same region, chambers number (phi), and eta partition
  // and layer run from 1 to number of layer. This is not the definition of one chamber... and indeed segments
  // could in principle run in different way... The concept of the DetLayer would be more appropriate...

  std::map<uint32_t, std::vector<ME0RecHit*> > ensembleRH;
    
  // Loop on the ME0 rechit and select the different ME0 Ensemble
  for(ME0RecHitCollection::const_iterator it2 = recHits->begin(); it2 != recHits->end(); it2++) {        
    ME0DetId id(it2->me0Id().region(),1,it2->me0Id().chamber(),it2->me0Id().roll());
    std::vector<ME0RecHit* > pp = ensembleRH[id.rawId()];
    pp.push_back(it2->clone());
    ensembleRH[id.rawId()]=pp;
  }
  
  for(auto enIt=ensembleRH.begin(); enIt != ensembleRH.end(); ++enIt) {
    
    std::vector<const ME0RecHit*> me0RecHits;
    std::map<uint32_t,const ME0EtaPartition* > ens;

    const ME0EtaPartition* firstlayer = geom_->etaPartition(enIt->first);
    for(auto rechit = enIt->second.begin(); rechit != enIt->second.end(); rechit++) {
      me0RecHits.push_back(*rechit);
      ens[(*rechit)->me0Id()]=geom_->etaPartition((*rechit)->me0Id());
    }    
    ME0SegmentAlgorithm::ME0Ensamble ensamble(std::pair<const ME0EtaPartition*, std::map<uint32_t,const ME0EtaPartition *> >(firstlayer,ens));
    
    LogDebug("ME0Segment|ME0") << "found " << me0RecHits.size() << " rechits in chamber " << *enIt;
    
    // given the chamber select the appropriate algo... and run it
    std::vector<ME0Segment> segv = algo->run(ensamble, me0RecHits);
    ME0DetId mid(enIt->first);
    LogDebug("ME0Segment|ME0") << "found " << segv.size() << " segments in chamber " << mid;
    
    // Add the segments to master collection
    oc.put(mid, segv.begin(), segv.end());
  }
}

void ME0SegmentBuilder::setGeometry(const ME0Geometry* geom) {
  geom_ = geom;
}

