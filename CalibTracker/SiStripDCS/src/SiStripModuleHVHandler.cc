#include "CalibTracker/SiStripDCS/interface/SiStripModuleHVHandler.h"
#include "DataFormats/Provenance/interface/Timestamp.h"
#include "CoralBase/TimeStamp.h"
#include "CalibTracker/SiStripDCS/interface/SiStripCoralIface.h"

popcon::SiStripModuleHVHandler::SiStripModuleHVHandler(std::string name,std::string cstring,std::string cat, const edm::Event& evt, const edm::EventSetup& est, std::string pconnect) : popcon::PopConSourceHandler<SiStripModuleHV>(name,cstring,cat,evt,est)
{
//std::cout<<evt.time().value()<<std::endl; 
unsigned int lastsincetime=0;
 std::map<std::string, popcon::PayloadIOV> mp = getOfflineInfo();
   for(std::map<std::string, popcon::PayloadIOV>::iterator it = mp.begin(); it != mp.end();it++)
      lastsincetime=it->second.last_since;	
//       std::cout << "Tag: " << it->first << " , last object valid since " << it->second.last_since << std::endl;
if(lastsincetime>evt.time().value()){
std::cout << "This event has been o2oed "<<std::endl;
//don't initialize this handler,
} 
else{ 

//now get the last vector. 

//initialize this handler, do the online query in getNewObjects();

}



}

popcon::SiStripModuleHVHandler::~SiStripModuleHVHandler()
{
}
void popcon::SiStripModuleHVHandler::getNewObjects()
{
    //to access the information on the tags inside the offline database call:
    std::map<std::string, popcon::PayloadIOV> mp = getOfflineInfo();
    //if mp is empty, offline db is not ready   	
//    std::cout << "map is empty "<<mp.empty()<<"and it's size "<<mp.size()<<std::endl;

    if(mp.empty()) //db record doesn't exist
    {    
         unsigned int snc = 1;
         unsigned int tll = 150000000;//fixme, has to be endoftime
	 edm::Timestamp tmpCondTime(snc); 
	 //snc=tmpCondTime.beginOfTime();
	 //tll=tmpCondTime.endOfTime(); //problem since Timestamp is unsigned longlong 
         popcon::IOVPair iop = {snc,tll};
        //fixme: set up a full OFF vector and put it to DB
            std::vector<uint32_t> vec_detid;
            vec_detid.push_back(321);
    SiStripModuleHV_ = new SiStripModuleHV();
            SiStripModuleHV_->put(vec_detid);
    std::cout << "module is set up "<<std::endl;
        m_to_transfer->push_back(std::make_pair(SiStripModuleHV_,iop));
    std::cout << "vect pushed "<<std::endl;

	}

    else{
    for(std::map<std::string, popcon::PayloadIOV>::iterator it = mp.begin(); it != mp.end();it++)
       std::cout << "Tag: " << it->first << " , last object valid since " << it->second.last_since << std::endl;

            //get last vector??


        //initial Coral with lastsince till lastevent of the run. 
	  std::vector<coral::TimeStamp> vec_changeDate;
     std::vector<uint32_t> vec_dpid;
     std::vector<uint32_t> vec_actualStatus;
     //fix me, put these connection strings as cfg parameters
     //when initializing this handler.
     std::string onlineDBConnectionString="oracle://cms_pvss_tk/CMS_PVSS_TK_SLICETEST";
     //std::string onlineDBConnectionString="oracle://omds/CMS_TRACKER_DCS_CONFIGURATION"
    std::string authenticationPath= "../data/";
    SiStripCoralIface * cif = new SiStripCoralIface(onlineDBConnectionString, authenticationPath);
    coral::TimeStamp tmax(2007,3,16,16,20,13,444000000);
    coral::TimeStamp tmin(2007,3,16,13,42,0,0);

    cif->doQuery(tmin, tmax, vec_changeDate, vec_dpid, vec_actualStatus);
    delete cif;

        //produce vector and put it offline

	// to write to Offline DB
         //for each payload provide IOV information
         unsigned int snc = 15000007;
         unsigned int tll = 15000009;
         popcon::IOVPair iop = {snc,tll};
          std::vector<uint32_t> vec_detid;
          vec_detid.push_back(789);
          SiStripModuleHV_ = new SiStripModuleHV();
          SiStripModuleHV_->put(vec_detid);

            m_to_transfer->push_back(std::make_pair(SiStripModuleHV_,iop));
      }



}
