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



#include "CalibTracker/SiStripQuality/plugins/SiStripBadStripFromASCIIFile.h"



using namespace std;

SiStripBadStripFromASCIIFile::SiStripBadStripFromASCIIFile( const edm::ParameterSet& iConfig ):
  ConditionDBWriter<SiStripBadStrip>( iConfig),
  printdebug_(iConfig.getUntrackedParameter<bool>("printDebug",false))
{
  fp_ = iConfig.getUntrackedParameter<edm::FileInPath>("file",edm::FileInPath("CalibTracker/SiStripQuality/data/DefectsFromConstructionDB.dat"));
 }



void SiStripBadStripFromASCIIFile::algoBeginJob( const edm::EventSetup& iSetup ) {
  p_channelflag p_chanflag;
  v_channelflag v_chanflag;
  
  unsigned int detid;
  short flag;
  short channel;
  



  // open filename from constructor
  ifstream infile((fp_.fullPath()).c_str());
  if(!infile){std::cout << "Problem while trying to open File: " << (fp_.fullPath()).c_str() << std::endl;}



  unsigned int detid_temp=0;
  bool first_detid = true;
  v_allbadstrips.clear();

  while(!infile.eof()){

    // get data from file: 
    infile >> detid >> channel >> flag;
    if(detid_temp==detid || first_detid){
      p_chanflag=p_channelflag(channel, flag);
      v_chanflag.push_back(p_chanflag);
    }
    else{
     
      v_allbadstrips.push_back(p_detidchannelflag(detid_temp,v_chanflag));
      v_chanflag.clear();
      p_chanflag=p_channelflag(channel, flag);
      v_chanflag.push_back(p_chanflag);
         }
   
    if(first_detid) first_detid = false;
    detid_temp=detid;

  }



 






    
   
     
}  


SiStripBadStrip *SiStripBadStripFromASCIIFile::getNewObject() {

  SiStripBadStrip* SiStripBadStrip_ = new SiStripBadStrip();
  
  // loop over detids from construction db 
  for(v_detidallbadstrips::const_iterator it = v_allbadstrips.begin(); it != v_allbadstrips.end(); it++){
    std::vector<unsigned int> theSiStripVector;


    int tempconsecstrips=0;
    unsigned int theBadStripRange=0;
    int count=0;
    uint32_t tempflag=0;

    for(v_channelflag::const_iterator itstrip=(it->second.begin()); itstrip!=(it->second.end()); itstrip++){
     
      if(itstrip==it->second.begin()){
	tempconsecstrips=itstrip->first;
	tempflag=itstrip->second;
      }
     
	  
      if (itstrip->first!=tempconsecstrips+count){
	
	  // firstbadstrip               nconsecutivebadstrips     flag
	  theBadStripRange = SiStripBadStrip_->encode(tempconsecstrips, count, tempflag); 
		
	  edm::LogInfo("SiStripBadStripFromASCIIFile")<< "detid " << it->first << " \t"
							   << " firstBadStrip " << tempconsecstrips << "\t "
							   << " NconsecutiveBadStrips " << count  << "\t "
	                                                   <<"flag " << tempflag << "\t"
							   << " packed integer " << std::hex << theBadStripRange  << std::dec
							   << std::endl; 	    
	    
	  theSiStripVector.push_back(theBadStripRange);
	  count=1;
	  tempconsecstrips=itstrip->first;
	  tempflag=itstrip->second;
	  	
      }else{count++;           }
	    
    
    }
      
     // firstbadstrip               nconsecutivebadstrips     flag
	  theBadStripRange = SiStripBadStrip_->encode(tempconsecstrips, count, tempflag); 
		
	  edm::LogInfo("SiStripBadStripFromASCIIFile")<< "detid " << it->first << " \t"
							   << " firstBadStrip " << tempconsecstrips << "\t "
							   << " NconsecutiveBadStrips " << count  << "\t "
	                                                   <<"flag " << tempflag << "\t"
							   << " packed integer " << std::hex << theBadStripRange  << std::dec
							   << std::endl; 	    
	    
	  theSiStripVector.push_back(theBadStripRange);
	  

	  count=0;
          
    // populate db  object
    SiStripBadStrip::Range range(theSiStripVector.begin(),theSiStripVector.end());
    if ( ! SiStripBadStrip_->put(it->first,range) )
      edm::LogError("SiStripBadStripFromASCIIFile")<<"[SiStripBadStripFromASCIIFile::analyze] detid already exists"<<std::endl;
  }
    
  SiStripBadStrip* obj=new SiStripBadStrip;
  return obj;
}
