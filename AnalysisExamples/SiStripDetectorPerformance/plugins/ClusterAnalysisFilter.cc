#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"


#include "AnalysisExamples/SiStripDetectorPerformance/plugins/ClusterAnalysisFilter.h"

namespace cms
{
  ClusterAnalysisFilter::ClusterAnalysisFilter(const edm::ParameterSet& conf) :
    conf_(conf),
    Track_src_( conf.getParameter<edm::InputTag>( "Track_src" ) ),
    ClusterInfo_src_( conf.getParameter<edm::InputTag>( "ClusterInfo_src" ) ),
    Cluster_src_( conf.getParameter<edm::InputTag>( "Cluster_src" ) )
  {  
    produces <uint16_t>();
  }

  void ClusterAnalysisFilter::beginJob(edm::EventSetup const& es) {
    //get geom    
    es.get<TrackerDigiGeometryRecord>().get( tkgeom );
    edm::LogInfo("ClusterAnalysis") << "[ClusterAnalysis::beginJob] There are "<<tkgeom->detUnits().size() <<" detectors instantiated in the geometry" << std::endl;  
  }

  bool ClusterAnalysisFilter::filter(edm::Event & e, edm::EventSetup const& c) {

    //Get input 
    e.getByLabel( ClusterInfo_src_, dsv_SiStripClusterInfo);
    e.getByLabel( Cluster_src_, dsv_SiStripCluster);    

    bool TrackNumberSelector_Decision_;
    try{
      e.getByLabel( Track_src_, trackCollection);
      TrackNumberSelector_Decision_ =TrackNumberSelector();
    } catch ( ... ) {
      TrackNumberSelector_Decision_ = true;
    }

    bool TriggerSelector_Decision_;
    try{
      e.getByType(ltcdigis);
      TriggerSelector_Decision_=TriggerSelector();
    } catch ( ... ) {
      TriggerSelector_Decision_=true;
    }

    bool ClusterNumberSelector_Decision_=ClusterNumberSelector();
    bool ClusterInModuleSelector_Decision_=ClusterInModuleSelector();

    bool decision = 
      ( ! conf_.getParameter<edm::ParameterSet>("TrackNumberSelector").getParameter<bool>("On") || TrackNumberSelector_Decision_ )
      &&
      ( ! conf_.getParameter<edm::ParameterSet>("ClusterNumberSelector").getParameter<bool>("On") || ClusterNumberSelector_Decision_ )
      &&
      ( ! conf_.getParameter<edm::ParameterSet>("TriggerSelector").getParameter<bool>("On") || TriggerSelector_Decision_ )
      &&
      ( ! conf_.getParameter<edm::ParameterSet>("ClusterInModuleSelector").getParameter<bool>("On") || ClusterInModuleSelector_Decision_ )
      //&& SomeThingElse
      ;

    uint16_t decisionWord = 
      ( TrackNumberSelector_Decision_ & 0x1)
      |
      ((ClusterNumberSelector_Decision_ << 1) & 0x2)
      |
      ((TriggerSelector_Decision_ << 2) & 0x4)
      | 
      ((ClusterInModuleSelector_Decision_ << 3) & 0x8)
      // | ( SomeThingElse < nbits)
      ;

    std::auto_ptr< uint16_t > odecisionWord( new uint16_t(decisionWord) );
    e.put(odecisionWord);
    
    return decision;
  }

  bool ClusterAnalysisFilter::TrackNumberSelector(){
    
    const edm::ParameterSet ps = conf_.getParameter<edm::ParameterSet>("TrackNumberSelector");
    
    int32_t min=ps.getParameter<int32_t>("minNTracks");
    int32_t max=ps.getParameter<int32_t>("maxNTracks");

    return ( (int32_t) trackCollection.product()->size() >= min && (int32_t) trackCollection.product()->size() < max) ;
  }
  
  bool ClusterAnalysisFilter::ClusterNumberSelector(){
    
    const edm::ParameterSet ps = conf_.getParameter<edm::ParameterSet>("ClusterNumberSelector");
    
    int32_t min=ps.getParameter<int32_t>("minNClus");
    int32_t max=ps.getParameter<int32_t>("maxNClus");
    
    int count=0;
    edm::DetSetVector<SiStripCluster>::const_iterator DSViter=dsv_SiStripCluster->begin();
    for (; DSViter!=dsv_SiStripCluster->end();DSViter++){
      count+=DSViter->data.size();
    }  
    return (count >= min && count < max) ;
  }


  bool ClusterAnalysisFilter::TriggerSelector(){
    
    const edm::ParameterSet ps = conf_.getParameter<edm::ParameterSet>("TriggerSelector");
    
    bool word=false;
    for (std::vector<LTCDigi>::const_iterator ltc_it =
	   ltcdigis->begin(); ltc_it != ltcdigis->end(); ltc_it++){
      word =
	(*ltc_it).externTriggerMask() &
	(
	 ps.getParameter<bool>("DT") & 0x1u	
	 |				       						
	 ( ps.getParameter<bool>("CSC")   & 0x1u  ) << 1	
	 |				 	      					
	 ( ps.getParameter<bool>("RBC1")  & 0x1u  ) << 2 
	 |				 	      					
	 ( ps.getParameter<bool>("RBC2")  & 0x1u  ) << 3
	 |				 						
	 ( ps.getParameter<bool>("RPCTB") & 0x1u ) << 4
	 )
	;
    
      LogTrace("ClusterAnalysisFilter")
	<< "Filter " 	
	<< "Mask " <<  ((*ltc_it).externTriggerMask() & 0x1Fu)
	<< " DT " <<  (ps.getParameter<bool>("DT") & 0x1u)
	<< " CSC " << (( ps.getParameter<bool>("CSC") & 0x1u ) << 1 )
	<< " RBC1 " << (( ps.getParameter<bool>("RBC1") & 0x1u )  << 2 )
	<< " RBC2 " << (( ps.getParameter<bool>("RBC2") & 0x1u ) << 3 ) 
	<< " RPCTB " << (( ps.getParameter<bool>("RPCTB") & 0x1u ) << 4 ) 
	<< " word " << word
	<< std::endl;
    }
    return word;    
  }


  bool ClusterAnalysisFilter::ClusterInModuleSelector (){

    const edm::ParameterSet ps = conf_.getParameter<edm::ParameterSet>("ClusterInModuleSelector");
    const edm::ParameterSet pps = ps.getParameter<edm::ParameterSet>("ClusterConditions");
    
    std::vector<uint32_t> ModulesToLookIn_=ps.getParameter< std::vector<uint32_t> >("ModulesToLookIn");
    std::vector<uint32_t> SubDetToLookIn_=ps.getParameter< std::vector<uint32_t> >("SubDetToLookIn");
    std::vector<uint32_t> SkipModules_=ps.getParameter< std::vector<uint32_t> >("SkipModules");

    
    for (std::vector<uint32_t>::const_iterator iter=ModulesToLookIn_.begin(); iter!=ModulesToLookIn_.end();iter++){
      edm::DetSetVector<SiStripClusterInfo>::const_iterator DSViter = dsv_SiStripClusterInfo->find(*iter);
      if ( DSViter != dsv_SiStripClusterInfo->end() ){	
	for(edm::DetSet<SiStripClusterInfo>::const_iterator ClusIter = DSViter->data.begin(); ClusIter!=DSViter->data.end(); ClusIter++) 
	  if (
	      ClusIter->charge()/ClusIter->noise() > pps.getParameter<double>("minStoN") 
	      &&
	      ClusIter->charge()/ClusIter->noise() < pps.getParameter<double>("maxStoN") 
	      &&
	      ClusIter->width() > pps.getParameter<double>("minWidth") 
	      &&
	      ClusIter->width() < pps.getParameter<double>("maxWidth") 
	      ){
	    return ps.getParameter<bool>("Accept");
	  }
      }	      
    }

    for (edm::DetSetVector<SiStripClusterInfo>::const_iterator DSViter = dsv_SiStripClusterInfo->begin(); DSViter != dsv_SiStripClusterInfo->end(); DSViter++ ){	
      uint32_t detid=DSViter->id;
      if (find(SkipModules_.begin(),SkipModules_.end(),detid)!=SkipModules_.end())
	continue;
      const StripGeomDetUnit* _StripGeomDetUnit = dynamic_cast<const StripGeomDetUnit*>(tkgeom->idToDetUnit(DetId(detid)));
      for (std::vector<uint32_t>::const_iterator iter=SubDetToLookIn_.begin(); iter!=SubDetToLookIn_.end();iter++){	
	if ((uint) _StripGeomDetUnit->specificType().subDetector() == *iter){
	  for(edm::DetSet<SiStripClusterInfo>::const_iterator ClusIter = DSViter->data.begin(); ClusIter!=DSViter->data.end(); ClusIter++) 
	    if (
		ClusIter->charge()/ClusIter->noise() > pps.getParameter<double>("minStoN") 
		&&
		ClusIter->charge()/ClusIter->noise() < pps.getParameter<double>("maxStoN") 
		&&
		ClusIter->width() > pps.getParameter<double>("minWidth") 
		&&
		ClusIter->width() < pps.getParameter<double>("maxWidth") 
		){
	//       std::cout
// 		<< "\n\t\tcluster detid "       << ClusIter->geographicalId()  
// 		<< "\n\t\tcluster first strip " << ClusIter->firstStrip()
// 		<< "\n\t\tcluster charge "      << ClusIter->charge()    
// 		<< "\n\t\tcluster noise "       << ClusIter->noise()     
// 		<< "\n\t\tcluster position "    << ClusIter->position()  
// 		<< "\n\t\tcluster width "       << ClusIter->width()  
// 		<< std::endl;
	      return ps.getParameter<bool>("Accept");
	    }
	}	      
      }
    }
    
    return !ps.getParameter<bool>("Accept");
  }
  
}
