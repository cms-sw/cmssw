
/** \file CSCSegmentBuilder.cc
 *
 * $Date: 2006/05/17 14:38:44 $
 * $Revision: 1.7 $
 * \author M. Sani
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
        
    // SegAlgo parameter set
    std::vector<edm::ParameterSet> segAlgoPSet = algoPSets[chosenAlgo].getParameter<std::vector<edm::ParameterSet> >("algo_psets");
        
    // Algo to chamber type 
    std::vector<int> algoToType = algoPSets[chosenAlgo].getParameter<std::vector<int> >("parameters_per_chamber_type");

    // Trap if we don't have enough parameter sets or haven't assigned an algo to every type   
    if (algoToType.size() != 9) {
        throw cms::Exception("ParameterSetError") << 
            "#dim algosToType=" << algoToType.size() << ", # chamber types=9" << std::endl;
    }

    // Ask factory to build this algorithm, giving it appropriate ParameterSet
    std::string chType[] = {"ME1/a", "ME1/b", "ME1/2", "ME1/3",
            "ME2/1", "ME2/2", "ME3/1", "ME3/2", "ME4/1"};
            
    for (size_t j=0; j<9; j++) 		
        algoMap[chType[j]] = CSCSegmentBuilderPluginFactory::get()->
                create(algoName, segAlgoPSet[algoToType[j]-1]);
}

CSCSegmentBuilder::~CSCSegmentBuilder() {}

void CSCSegmentBuilder::build(const CSCRecHit2DCollection* recHits, CSCSegmentCollection& oc) {
  	
    LogDebug("CSC")<< "Total number of RecHits: " << recHits->size() << "\n";

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
            chambers.push_back((*it2).cscDetId());
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
        
        LogDebug("CSC") << "found " << cscRecHits.size() << " rechit in this chamber.";
            
        // given the chamber select the right algo...
        std::vector<CSCSegment> segv = algoMap[chamber->specs()->chamberTypeName()]->run(chamber, cscRecHits);

        // Add the segments to master collection !!!
        LogDebug("CSC") << "Total number of segments found: " << segv.size() <<std::endl;
		
        oc.put((*chIt), segv.begin(), segv.end());
    }
}

void CSCSegmentBuilder::setGeometry(const CSCGeometry* geom) {
	geom_ = geom;
}

