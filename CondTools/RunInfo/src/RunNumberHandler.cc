#include "CondTools/RunInfo/interface/RunNumberHandler.h"
#include "CondTools/RunInfo/interface/RunNumberRead.h"


#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CondTools/RunInfo/interface/TestBase.h"


#include<iostream>
#include<sstream>
#include<vector>

namespace {

 
RunNumberHandler::RunNumberHandler(const edm::ParameterSet& pset) :
  m_name(pset.getUntrackedParameter<std::string>("name","RunNumberHandler")),
  // m_connect(pset.getUntrackedParameter<std::string>("OnlineConn","")),
  m_authpath(pset.getUntrackedParameter<std::string>("OnlineAuthPath",".")),
  m_host(pset.getUntrackedParameter<std::string>("OnlineDBHost","cmsor1-v.cern.ch")),
  m_sid(pset.getUntrackedParameter<std::string>("OnlineDBSID","omds")),
  
  m_user(pset.getUntrackedParameter<std::string>("OnlineDBUser","CMS_RUNINFO")), 
  m_pass(pset.getUntrackedParameter<std::string>("OnlineDBPass","********")),
  m_port(pset.getUntrackedParameter<int>("OnlineDBPort",10121))
{
  m_connectionString= "oracle://cms_omds_lb/CMS_RUNINFO";
 
}

RunNumberHandler::~RunNumberHandler()
{
} 


void RunNumberHandler::getNewObjects() {
   edm::LogInfo   ("RunNumberHandler") << "------- " << m_name 
	     << " - > getNewObjects\n" << 
  //check whats already inside of database
      "got offlineInfo"<<
    tagInfo().name << ", size " << tagInfo().size 
            << ", last object valid since " 
	    << tagInfo().lastInterval.first << " token "   
            << tagInfo().lastPayloadToken << std::endl;
  

  if (tagInfo().size>0) {
    Ref payload = lastPayload();
    edm::LogInfo   ("RunNumberHandler")<<"size of last payload  "<< 
      payload->m_runnumber.size()<<std::endl;
  }

  int snc;
  
  std::cerr << "Source implementation test ::getNewObjects : enter runnumber as a first since !  \n";
  std::cin >> snc;




  std::cout<<"runnumber/first since = "<< snc <<std::endl;
  
 
 
   RunNumber  * r = new RunNumber; 
   
   // reading from omds
   RunNumberRead rn( m_connectionString, m_user, m_pass);
   std::vector<RunNumber::Item> rnarray;

   // table to be  cms_runinfo.runsession_parameter
   //column to be string_value;
   // run to be 43623 

 rnarray = rn.readData("RUNSESSION_PARAMETER", "STRING_VALUE",(int)snc );

RunNumber::Item rnfill;
  std::vector<RunNumber::Item>::iterator Iit;
  for(Iit = rnarray.begin(); Iit != rnarray.end(); Iit++)
    {
      rnfill = *(Iit);
      r->m_runnumber.push_back(rnfill);   
    }
  
   m_to_transfer.push_back(std::make_pair((RunNumber*)r,snc));
   std::ostringstream ss; 
   ss << "since =" << snc;
    
  

  m_userTextLog = ss.str()+";";


  edm::LogInfo   ("RunNumberHandler") << "------- " << m_name << " - > getNewObjects" << std::endl;


}
}




