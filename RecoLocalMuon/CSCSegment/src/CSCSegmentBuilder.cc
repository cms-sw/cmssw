
/** \file CSCSegmentBuilder.cc
 *
 *
 */

#include <RecoLocalMuon/CSCSegment/src/CSCSegmentBuilder.h>
#include <Geometry/CSCGeometry/interface/CSCChamberSpecs.h>
#include <Geometry/CSCGeometry/interface/CSCLayer.h>
#include <Geometry/CSCGeometry/interface/CSCGeometry.h>
#include <DataFormats/MuonDetId/interface/CSCDetId.h>
#include <DataFormats/CSCRecHit/interface/CSCRecHit2D.h>
#include <DataFormats/CSCRecHit/interface/CSCRangeMapAccessor.h>

#include <RecoLocalMuon/CSCSegment/src/CSCSegmentAlgorithm.h>
#include <RecoLocalMuon/CSCSegment/src/CSCSegmentBuilderPluginFactory.h>

#include <FWCore/Utilities/interface/Exception.h>
#include <FWCore/MessageLogger/interface/MessageLogger.h> 

CSCSegmentBuilder::CSCSegmentBuilder(const edm::ParameterSet& ps) : geom_(0) {
    
    // The algo chosen for the segment building
    int chosenAlgo = ps.getParameter<int>("algo_type") - 1;
    
    // Find appropriate ParameterSets for each algo type
    std::vector<edm::ParameterSet> algoPSets = ps.getParameter<std::vector<edm::ParameterSet> >("algo_psets");

    // Now load the right parameter set
    // Algo name
    std::string algoName = algoPSets[chosenAlgo].getParameter<std::string>("algo_name");
        
    LogDebug("CSCSegment|CSC")<< "CSCSegmentBuilder algorithm name: " << algoName;

    // SegAlgo parameter set
    std::vector<edm::ParameterSet> segAlgoPSet = algoPSets[chosenAlgo].getParameter<std::vector<edm::ParameterSet> >("algo_psets");

    // Chamber types to handle
    std::vector<std::string> chType = algoPSets[chosenAlgo].getParameter<std::vector<std::string> >("chamber_types");
    LogDebug("CSCSegment|CSC")<< "No. of chamber types to handle: " << chType.size();

    // Algo to chamber type 
    std::vector<int> algoToType = algoPSets[chosenAlgo].getParameter<std::vector<int> >("parameters_per_chamber_type");

    // Trap if we don't have enough parameter sets or haven't assigned an algo to every type   
    if (algoToType.size() !=  chType.size()) {
        throw cms::Exception("ParameterSetError") << 
	  "#dim algosToType=" << algoToType.size() << ", #dim chType=" << chType.size() << std::endl;
    }

    // Ask factory to build this algorithm, giving it appropriate ParameterSet
            
    for (size_t j=0; j<chType.size(); ++j) {
        algoMap[chType[j]] = CSCSegmentBuilderPluginFactory::get()->
                create(algoName, segAlgoPSet[algoToType[j]-1]);
	edm::LogVerbatim("CSCSegment|CSC")<< "using algorithm #" << algoToType[j] << " for chamber type " << chType[j];
    }
}

CSCSegmentBuilder::~CSCSegmentBuilder() {
  //
  // loop on algomap and delete them
  //
  for (std::map<std::string, CSCSegmentAlgorithm*>::iterator it = algoMap.begin();it != algoMap.end(); it++){
    delete ((*it).second);
  }
}

void CSCSegmentBuilder::build(const CSCRecHit2DCollection* recHits, CSCSegmentCollection& oc) {
  	
  LogDebug("CSCSegment|CSC")<< "Total number of rechits in this event: " << recHits->size();

    std::vector<CSCDetId> chambers;
    std::vector<CSCDetId>::const_iterator chIt;
    
    for(CSCRecHit2DCollection::const_iterator it2 = recHits->begin(); it2 != recHits->end(); it2++) {
        
        bool insert = true;
        for(chIt=chambers.begin(); chIt != chambers.end(); ++chIt) 
            if (((*it2).cscDetId().chamber() == (*chIt).chamber()) &&
                ((*it2).cscDetId().station() == (*chIt).station()) &&
                ((*it2).cscDetId().ring() == (*chIt).ring()) &&
                ((*it2).cscDetId().endcap() == (*chIt).endcap()))
                insert = false;
	
        if (insert)
            chambers.push_back((*it2).cscDetId().chamberId());
    }

    for(chIt=chambers.begin(); chIt != chambers.end(); ++chIt) {

        std::vector<const CSCRecHit2D*> cscRecHits;
        const CSCChamber* chamber = geom_->chamber(*chIt);
        
        CSCRangeMapAccessor acc;
        CSCRecHit2DCollection::range range = recHits->get(acc.cscChamber(*chIt));
        
        std::vector<int> hitPerLayer(6);
        for(CSCRecHit2DCollection::const_iterator rechit = range.first; rechit != range.second; rechit++) {
            
            hitPerLayer[(*rechit).cscDetId().layer()-1]++;
            cscRecHits.push_back(&(*rechit));
        }    
        
        LogDebug("CSCSegment|CSC") << "found " << cscRecHits.size() << " rechits in chamber " << *chIt;
            
        // given the chamber select the appropriate algo... and run it
        std::vector<CSCSegment> segv = algoMap[chamber->specs()->chamberTypeName()]->run(chamber, cscRecHits);

        LogDebug("CSCSegment|CSC") << "found " << segv.size() << " segments in chamber " << *chIt;

        // Add the segments to master collection
        oc.put((*chIt), segv.begin(), segv.end());
    }
}

void CSCSegmentBuilder::setGeometry(const CSCGeometry* geom) {
	geom_ = geom;
}

