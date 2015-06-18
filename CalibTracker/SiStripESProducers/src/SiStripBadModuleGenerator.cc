#include "CalibTracker/SiStripESProducers/interface/SiStripBadModuleGenerator.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <boost/cstdint.hpp>
#include "CalibTracker/SiStripCommon/interface/SiStripDetInfoFileReader.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "CalibFormats/SiStripObjects/interface/SiStripQuality.h"

SiStripBadModuleGenerator::SiStripBadModuleGenerator(const edm::ParameterSet& iConfig,const edm::ActivityRegistry& aReg):
  SiStripDepCondObjBuilderBase<SiStripBadStrip,TrackerTopology>::SiStripDepCondObjBuilderBase(iConfig)
{
  edm::LogInfo("SiStripBadModuleGenerator") <<  "[SiStripBadModuleGenerator::SiStripBadModuleGenerator]";
}


SiStripBadModuleGenerator::~SiStripBadModuleGenerator() { 
  edm::LogInfo("SiStripBadModuleGenerator") <<  "[SiStripBadModuleGenerator::~SiStripBadModuleGenerator]";
}


SiStripBadStrip* SiStripBadModuleGenerator::createObject(const TrackerTopology* tTopo){
    
  SiStripQuality* obj  = new SiStripQuality();

  edm::FileInPath fp_           = _pset.getUntrackedParameter<edm::FileInPath>("file",edm::FileInPath("CalibTracker/SiStripCommon/data/SiStripDetInfo.dat"));
  printdebug_             = _pset.getUntrackedParameter<bool>("printDebug",false);
  BadComponentList_            =  _pset.getUntrackedParameter<Parameters>("BadComponentList");


  SiStripDetInfoFileReader reader(fp_.fullPath());   
  const std::vector<uint32_t>& DetIds= reader.getAllDetIds();
  std::vector<uint32_t> selDetIds;
  selectDetectors(tTopo,DetIds,selDetIds);

  edm::LogInfo("SiStripQualityConfigurableFakeESSource")<<"[produce] number of selected dets to be removed " << selDetIds.size() <<std::endl;

  std::stringstream ss;  
  std::vector<uint32_t>::const_iterator iter=selDetIds.begin();
  std::vector<uint32_t>::const_iterator iterEnd=selDetIds.end();
  for(;iter!=iterEnd;++iter){
    
    SiStripQuality::InputVector theSiStripVector;
    
    unsigned short firstBadStrip=0, NconsecutiveBadStrips=reader.getNumberOfApvsAndStripLength(*iter).first * 128;
    unsigned int theBadStripRange;
    
    theBadStripRange = obj->encode(firstBadStrip,NconsecutiveBadStrips);
    
    if (printdebug_)
      ss << "detid " << *iter << " \t"
	 << " firstBadStrip " << firstBadStrip << "\t "
	 << " NconsecutiveBadStrips " << NconsecutiveBadStrips << "\t "
	 << " packed integer " << std::hex << theBadStripRange  << std::dec
	 << std::endl; 	    
    
    theSiStripVector.push_back(theBadStripRange);
    
    SiStripBadStrip::Range range(theSiStripVector.begin(),theSiStripVector.end());
    if ( ! obj->put(*iter,range) )
      edm::LogError("SiStripQualityConfigurableFakeESSource")<<"[produce] detid already exists"<<std::endl;
  }
  if (printdebug_)
    edm::LogInfo("SiStripQualityConfigurableFakeESSource") << ss.str();
  
  obj->cleanUp();
  //obj->fillBadComponents();

  std::stringstream ss1;
  if (printdebug_){
    for (std::vector<SiStripQuality::BadComponent>::const_iterator iter=obj->getBadComponentList().begin();iter!=obj->getBadComponentList().end();++iter)
      ss1 << "bad module " << iter->detid << " " << iter->BadModule <<  "\n";
    edm::LogInfo("SiStripQualityConfigurableFakeESSource") << ss1.str();
  }

//  obj_ = new SiStripBadStrip( *(dynamic_cast<SiStripBadStrip*> (obj)));
  return obj;
}


void SiStripBadModuleGenerator::selectDetectors(const TrackerTopology* tTopo, const std::vector<uint32_t>& DetIds, std::vector<uint32_t>& list){
  std::stringstream ss;

  for(Parameters::iterator iBadComponent = BadComponentList_.begin(); iBadComponent != BadComponentList_.end(); ++iBadComponent ) {
    
    if (printdebug_)
      ss << "Bad SubDet " << iBadComponent->getParameter<std::string>("SubDet") << " \t";

    SiStripDetId::SubDetector subDet=SiStripDetId::UNKNOWN;
    if (iBadComponent->getParameter<std::string>("SubDet")=="TIB")
      subDet=SiStripDetId::TIB;
    else if (iBadComponent->getParameter<std::string>("SubDet")=="TID")
      subDet=SiStripDetId::TID;
    else if (iBadComponent->getParameter<std::string>("SubDet")=="TOB")
      subDet=SiStripDetId::TOB;
    else if (iBadComponent->getParameter<std::string>("SubDet")=="TEC")
      subDet=SiStripDetId::TEC;
    std::vector<uint32_t> genericBadDetIds( iBadComponent->getUntrackedParameter<std::vector<uint32_t> >("detidList", std::vector<uint32_t>()) );
    
    bool anySubDet = true;
    if( genericBadDetIds.empty() ) anySubDet = false;

    std::cout << "genericBadDetIds.size() = " << genericBadDetIds.size() << std::endl;

    uint32_t startDet=DetId(DetId::Tracker,subDet).rawId();
    uint32_t stopDet=DetId(DetId::Tracker,subDet+1).rawId();

    if( anySubDet ) {
      startDet=DetId(DetId::Tracker,SiStripDetId::TIB).rawId();
      stopDet=DetId(DetId::Tracker,SiStripDetId::TEC+1).rawId();      
    }

    std::vector<uint32_t>::const_iterator iter=lower_bound(DetIds.begin(),DetIds.end(),startDet);
    std::vector<uint32_t>::const_iterator iterEnd=lower_bound(DetIds.begin(),DetIds.end(),stopDet);

    bool resp;
    for ( ;iter!=iterEnd;++iter){
      const DetId detectorId=DetId(*iter);
      resp=false;
      if (iBadComponent->getParameter<std::string>("SubDet")=="TIB")
	resp=isTIBDetector(tTopo,detectorId,
			   iBadComponent->getParameter<uint32_t>("layer"),
			   iBadComponent->getParameter<uint32_t>("bkw_frw"),
			   iBadComponent->getParameter<uint32_t>("int_ext"),
			   iBadComponent->getParameter<uint32_t>("ster"),
			   iBadComponent->getParameter<uint32_t>("string_"),
			   iBadComponent->getParameter<uint32_t>("detid")
			   );
      else if (iBadComponent->getParameter<std::string>("SubDet")=="TID")
	resp=isTIDDetector(tTopo,detectorId,
			   iBadComponent->getParameter<uint32_t>("side"),
			   iBadComponent->getParameter<uint32_t>("wheel"),
			   iBadComponent->getParameter<uint32_t>("ring"),
			   iBadComponent->getParameter<uint32_t>("ster"),
			   iBadComponent->getParameter<uint32_t>("detid")
			   );
      else if (iBadComponent->getParameter<std::string>("SubDet")=="TOB")
	resp=isTOBDetector(tTopo,detectorId,
			   iBadComponent->getParameter<uint32_t>("layer"),
			   iBadComponent->getParameter<uint32_t>("bkw_frw"),
			   iBadComponent->getParameter<uint32_t>("rod"),
			   iBadComponent->getParameter<uint32_t>("ster"),
			   iBadComponent->getParameter<uint32_t>("detid")
			   );
      else if (iBadComponent->getParameter<std::string>("SubDet")=="TEC")
	resp=isTECDetector(tTopo,detectorId,
			   iBadComponent->getParameter<uint32_t>("side"),
			   iBadComponent->getParameter<uint32_t>("wheel"),
			   iBadComponent->getParameter<uint32_t>("petal_bkw_frw"),
			   iBadComponent->getParameter<uint32_t>("petal"),
			   iBadComponent->getParameter<uint32_t>("ring"),
			   iBadComponent->getParameter<uint32_t>("ster"),
			   iBadComponent->getParameter<uint32_t>("detid")
			   );
      if( anySubDet ) {
	std::cout << "AnySubDet" << *iter << std::endl;
	if( std::find(genericBadDetIds.begin(), genericBadDetIds.end(), *iter) == genericBadDetIds.end() ) resp = false;
	else resp = true;
      }
   
      if(resp)
	list.push_back(*iter);      
    }
  }
  if (printdebug_)
    edm::LogInfo("SiStripBadModuleGenerator") << ss.str();
}


bool SiStripBadModuleGenerator::isTIBDetector(const TrackerTopology* tTopo,
                 const DetId & therawid,
							   uint32_t requested_layer,
							   uint32_t requested_bkw_frw,
							   uint32_t requested_int_ext,
							   uint32_t requested_string,
							   uint32_t requested_ster,
							   uint32_t requested_detid) const{
  if(  therawid.subdetId() ==  SiStripDetId::TIB ){ // check if subdetector field is a TIB, both tested numbers are int
    if( // check if TIB is from the ones requested
       (    (tTopo->tibLayer(therawid)==requested_layer) || requested_layer==0 )  // take everything if default value is 0
       &&
       ( (tTopo->tibIsZPlusSide(therawid) && requested_bkw_frw==2) || (!tTopo->tibIsZPlusSide(therawid) && requested_bkw_frw==1) || requested_bkw_frw==0)
       &&
       ( (tTopo->tibIsInternalString(therawid) && requested_int_ext==1) || (!tTopo->tibIsInternalString(therawid) && requested_int_ext==2) || requested_int_ext==0 )
       && 
       ( (tTopo->tibIsStereo(therawid) && requested_ster==1) || (tTopo->tibIsRPhi(therawid) && requested_ster==2) || requested_ster==0 )
       && 
       ( (tTopo->tibString(therawid)==requested_string) || requested_string==0 )
       &&
       ( (therawid.rawId()==requested_detid) || requested_detid==0 )
       )
      return 1;
  }
  return 0;
}

bool SiStripBadModuleGenerator::isTOBDetector(const TrackerTopology* tTopo,
                 const DetId & therawid,
							   uint32_t requested_layer,
							   uint32_t requested_bkw_frw,
							   uint32_t requested_rod,
							   uint32_t requested_ster,
							   uint32_t requested_detid) const{
  if( therawid.subdetId() ==  SiStripDetId::TOB ){ // check if subdetector field is a TOB, both tested numbers are int
    if( // check if TOB is from the ones requested
       (    (tTopo->tobLayer(therawid)==requested_layer) || requested_layer==0 )  // take everything if default value is 0
       &&
       ( (tTopo->tobIsZPlusSide(therawid) && requested_bkw_frw==2) || (!tTopo->tobIsZPlusSide(therawid) && requested_bkw_frw==1) || requested_bkw_frw==0)
       &&
       ( (tTopo->tobIsStereo(therawid) && requested_ster==1) || (tTopo->tobIsRPhi(therawid) && requested_ster==2) || requested_ster==0 )
       && 
       ( (tTopo->tobRod(therawid) ==requested_rod) || requested_rod==0 )
       &&
       ( (therawid.rawId()==requested_detid) || requested_detid==0 )
       )
      return 1;
  }
  return 0;
}


bool SiStripBadModuleGenerator::isTIDDetector(const TrackerTopology* tTopo,
                 const DetId & therawid,
							   uint32_t requested_side,
							   uint32_t requested_wheel,
							   uint32_t requested_ring,
							   uint32_t requested_ster,
							   uint32_t requested_detid) const{
  if( therawid.subdetId() ==  SiStripDetId::TID ){ // check if subdetector field is a TID, both tested numbers are int
    if( // check if TID is from the ones requested 
     (    (tTopo->tidWheel(therawid)==requested_wheel) || requested_wheel==0 )  // take everything if default value is 0
     &&
     ( (tTopo->tidIsZPlusSide(therawid) && requested_side==2) || (!tTopo->tidIsZPlusSide(therawid) && requested_side==1) || requested_side==0)
     &&
     ( (tTopo->tidIsStereo(therawid) && requested_ster==1) || (tTopo->tidIsRPhi(therawid) && requested_ster==2) || requested_ster==0 )
     &&
     ( (tTopo->tidRing(therawid) ==requested_ring) || requested_ring==0 )
     &&
     ( (therawid.rawId()==requested_detid) || requested_detid==0 )
     )
     return 1;
     }
  return 0;
}
  

bool SiStripBadModuleGenerator::isTECDetector(const TrackerTopology* tTopo,
                 const DetId & therawid,
							   uint32_t requested_side,
							   uint32_t requested_wheel,
							   uint32_t requested_petal_bkw_frw,
							   uint32_t requested_petal,			
							   uint32_t requested_ring,
							   uint32_t requested_ster,
							   uint32_t requested_detid) const{
  if( therawid.subdetId() ==  SiStripDetId::TEC ){ // check if subdetector field is a TEC, both tested numbers are int
    if( // check if TEC is from the ones requested 
       (    (tTopo->tecWheel(therawid)==requested_wheel) || requested_wheel==0 )  // take everything if default value is 0
       &&
       ( (tTopo->tecIsZPlusSide(therawid) && requested_side==2) || (!tTopo->tecIsZPlusSide(therawid) && requested_side==1) || requested_side==0)
       &&
       ( (tTopo->tecIsStereo(therawid) && requested_ster==1) || (!tTopo->tecIsStereo(therawid) && requested_ster==2) || requested_ster==0 )
       &&
       ( (tTopo->tecIsFrontPetal(therawid)&& requested_petal_bkw_frw==2) || (!tTopo->tecIsFrontPetal(therawid) && requested_petal_bkw_frw==2) || requested_petal_bkw_frw==0 )
       &&
       ( (tTopo->tecPetalNumber(therawid)==requested_petal) || requested_petal==0 )
       &&
       ( (tTopo->tecRing(therawid)==requested_ring) || requested_ring==0 )
       &&
       ( (therawid.rawId()==requested_detid) || requested_detid==0 )
       )
      return 1;
  }
  return 0;
}

