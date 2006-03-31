// This is CSCSegmentBuilder.cc

#include <RecoLocalMuon/CSCSegment/src/CSCSegmentBuilder.h>
#include <Geometry/CSCGeometry/interface/CSCChamberSpecs.h>
#include <Geometry/CSCGeometry/interface/CSCLayer.h>
#include <Geometry/CSCGeometry/interface/CSCGeometry.h>
#include <DataFormats/MuonDetId/interface/CSCDetId.h>
#include <DataFormats/CSCRecHit/interface/CSCRecHit2D.h>
#include <RecoLocalMuon/CSCSegment/src/CSCDetIdAccessor.h>

#include <RecoLocalMuon/CSCSegment/src/CSCSegmentAlgorithm.h>
#include <RecoLocalMuon/CSCSegment/src/CSCSegmentBuilderPluginFactory.h>

#include <FWCore/Utilities/interface/Exception.h>
#include <FWCore/MessageLogger/interface/MessageLogger.h> 

CSCSegmentBuilder::CSCSegmentBuilder(const edm::ParameterSet& ps) : geom_(0), 
    algos_(std::vector<CSCSegmentAlgorithm*>()) {
	
    // Receives ParameterSet percolated down from EDProducer
    // Find names of algorithms
    std::vector<std::string> algoNames = ps.getParameter<std::vector<std::string> >("algo_types");

    // Find appropriate ParameterSets
    std::vector<edm::ParameterSet> algoPSets = ps.getParameter<std::vector<edm::ParameterSet> >("algo_psets");

    // Find allocation of algorithm to chamber type
    std::vector<int> algoToType = ps.getParameter<std::vector<int> >("parameters_per_chamber_type");

    // How many chamber types do we have?
    ntypes = 9;
    LogDebug("CSC") << "no. of chamber types = " << ntypes;

    algos_.resize(ntypes*algoNames.size());
	
    // Trap if we don't have enough parameter sets or haven't assigned an algo to every type
    if (algoToType.size() != (ntypes*algoNames.size())) {
        throw cms::Exception("ParameterSetError") << 
            "#dim algosToType=" << algoToType.size() << ", # chamber types=" << ntypes 
            << ", algos=" << algoNames.size() << std::endl;
    }

    std::vector<CSCSegmentAlgorithm*> algobuf;
  	
    // Ask factory to build this algorithm, giving it appropriate ParameterSet
    for (size_t i=0; i<algoNames.size(); ++i ) {
        for (size_t j=0; j<algoToType.size(); j++) {
			
            CSCSegmentAlgorithm* pAlgo = CSCSegmentBuilderPluginFactory::get()->
                create(algoNames[i], algoPSets[algoToType[j]-1]);
            algobuf.push_back(pAlgo);
            LogDebug("CSC") << "algorithm [" << i << "] named " << algoNames[i] << " has address " << pAlgo;
        }
    }	
	
    for ( size_t i = 0; i < algobuf.size(); ++i ) {
        algos_[i] = algobuf[i]; // 
        LogDebug("CSC") << "address of algorithm for chamber type " << i << " is " << algos_[i];
    }
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

        std::vector<CSCRecHit2D> cscRecHits;
        const CSCChamber* chamber = geom_->chamber(*chIt);
       
        CSCDetIdAccessor acc;
        CSCRecHit2DCollection::range range = recHits->get(acc.cscChamber(*chIt));
        for(CSCRecHit2DCollection::const_iterator rechit = range.first; rechit != range.second; rechit++)
            cscRecHits.push_back(*rechit);
        
        LogDebug("CSC") << "found " << cscRecHits.size() << " rechit in this chamber.";
        
        std::string chType[] = {"ME1/a", "ME1/b", "ME1/1", "ME1/2",
            "ME2/1", "ME2/2", "ME3/1", "ME3/2", "ME4/1"};
        unsigned int algoNum;
        for(algoNum=0; algoNum<ntypes; algoNum++) {
		
            if (chamber->specs()->chamberTypeName() == chType[algoNum])
                break;
        }
				
        // given the chamber select the right algo...
        CSCSegmentCollection rhv = algos_[algoNum]->run(chamber, cscRecHits);
	  
        // Add the segments to master collection !!!
        LogDebug("CSC") << "Total number of segments found: " << rhv.size() <<std::endl;
		
        CSCSegmentCollection::const_iterator segmIt;
        for(segmIt = rhv.begin(); segmIt != rhv.end(); segmIt++)
            oc.push_back(*segmIt);    
    }
}

void CSCSegmentBuilder::setGeometry(const CSCGeometry* geom) {
	geom_ = geom;
}

