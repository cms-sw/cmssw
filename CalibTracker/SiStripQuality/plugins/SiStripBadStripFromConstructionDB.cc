// system include files
#include <memory>
#include <cstdio>
#include <string>


// user include files


#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h" 
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetType.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"


#include "CLHEP/Random/RandFlat.h"
#include "CLHEP/Random/RandGauss.h"



#include "CalibTracker/SiStripQuality/plugins/SiStripBadStripFromConstructionDB.h"
#include "CalibTracker/SiStripQuality/plugins/DefectsFromConstructionDB.h"


using namespace std;

SiStripBadStripFromConstructionDB::SiStripBadStripFromConstructionDB( const edm::ParameterSet& iConfig ):
  ConditionDBWriter<SiStripBadStrip>( iConfig),
  printdebug_(iConfig.getUntrackedParameter<bool>("printDebug",false))
{
  ext_bad_detids = iConfig.getUntrackedParameter<std::vector<uint32_t> >("ext_bad_detids");
}



void SiStripBadStripFromConstructionDB::algoBeginJob( const edm::EventSetup& iSetup ) {
  
  // Read detids and corresponding bad strips from txt file from construction DB 
  std::string cmssw_path=std::string(getenv("CMSSW_BASE"));
  std::string src_filepath=std::string("/src/CalibTracker/SiStripQuality/plugins/DetidBadChannelArcFlags.txt" );
  
    
  DefectsFromConstructionDB arcdefects((cmssw_path+src_filepath).c_str());

  v_detidallbadstrips v_BadStripsfromConDB =  arcdefects.GetBadStripsFromConstructionDB();
    
  for(v_detidallbadstrips::iterator iter=v_BadStripsfromConDB.begin(); iter<v_BadStripsfromConDB.end();iter++){
     
    for(v_channelflag::iterator it=iter->second.begin(); it<iter->second.end(); it++){
	
      badstrips.push_back((*it).first);
    }
    // fill detid with corresponding bad strips
    constdb_strips.push_back(std::pair<uint32_t, std::vector<short> >((uint32_t)((*iter).first),badstrips));
    badstrips.clear();
     
  }  
}

SiStripBadStrip *SiStripBadStripFromConstructionDB::getNewObject() {

  SiStripBadStrip* SiStripBadStrip_ = new SiStripBadStrip();
  
  // loop over detids from construction db 
  for(std::vector< std::pair<uint32_t,std::vector<short> > >::const_iterator it = constdb_strips.begin(); it != constdb_strips.end(); it++){
    std::vector<unsigned int> theSiStripVector;


    std::vector<short>::const_iterator tempiter;
    unsigned int theBadStripRange=0;

    for(std::vector<short>::const_iterator itstrip=(it->second.begin()); itstrip!=(it->second.end()); itstrip++){
      if((itstrip==it->second.begin())&&(itstrip!=(it->second.end()-1))){
	tempiter=itstrip;
	   
      }
	  
      if (itstrip!=it->second.begin()){
	if(((*itstrip)-(*(itstrip-1)))!=1){
	  // firstbadstrip               nconsecutivebadstrips   
	  theBadStripRange = SiStripBadStrip_->encode(*tempiter,(*(itstrip-1)-(*tempiter)+1)); 
		
	  edm::LogInfo("SiStripBadStripFromConstructionDB")<< "detid " << it->first << " \t"
							   << " firstBadStrip " << *tempiter << "\t "
							   << " NconsecutiveBadStrips " << (*(itstrip-1)-(*tempiter))+1  << "\t "
							   << " packed integer " << std::hex << theBadStripRange  << std::dec
							   << std::endl; 	    
	    
	  theSiStripVector.push_back(theBadStripRange);
	      
	  tempiter=itstrip;
	}
      }
	    
	    
      if(itstrip==((it->second.end())-1)){
	if(itstrip!=it->second.begin()){
	  if(((*itstrip)-(*(itstrip-1)))!=1){
	    // firstbadstrip               nconsecutivebadstrips   
	    theBadStripRange = SiStripBadStrip_->encode(*tempiter,1); 
		    
	    edm::LogInfo("SiStripBadStripFromConstructionDB")
	      << "detid " << it->first << " \t"
	      << " firstBadStrip " << *tempiter << "\t "
	      << " NconsecutiveBadStrips " << 1  << "\t "
	      << " packed integer " << std::hex << theBadStripRange  << std::dec
	      << std::endl; 	    
		    
	    theSiStripVector.push_back(theBadStripRange);
	  }else{
		    
	    // firstbadstrip               nconsecutivebadstrips   
	    theBadStripRange = SiStripBadStrip_->encode(*tempiter,(*(itstrip)-(*tempiter))+1) ; 
		
		    
	    edm::LogInfo("SiStripBadStripFromConstructionDB")
	      << "detid " << it->first << " \t"
	      << " firstBadStrip " << *tempiter << "\t "
	      << " NconsecutiveBadStrips " << (*(itstrip)-(*tempiter))+1  << "\t "
	      << " packed integer " << std::hex << theBadStripRange  << std::dec
	      << std::endl; 	    
		    
	    theSiStripVector.push_back(theBadStripRange);
	  }
	}else{
	  // firstbadstrip               nconsecutivebadstrips   
	  theBadStripRange = SiStripBadStrip_->encode(*tempiter,1);
		    		    
	  edm::LogInfo("SiStripBadStripFromConstructionDB")
	    << "detid " << it->first << " \t"
	    << " firstBadStrip " << *itstrip << "\t "
	    << " NconsecutiveBadStrips " << 1  << "\t "
	    << " packed integer " << std::hex << theBadStripRange  << std::dec
	    << std::endl; 	    
		    
	  theSiStripVector.push_back(theBadStripRange);
	}
	
      }
	      
    }
	
    // populate db  object
    SiStripBadStrip::Range range(theSiStripVector.begin(),theSiStripVector.end());
    if ( ! SiStripBadStrip_->put(it->first,range) )
      edm::LogError("SiStripBadStripFromConstructionDB")<<"[SiStripBadStripFromConstructionDB::analyze] detid already exists"<<std::endl;
  }
    
  SiStripBadStrip* obj=new SiStripBadStrip;
  return obj;
}
