#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CondTools/RunInfo/interface/L1TriggerScalerHandler.h"
#include "CondTools/RunInfo/interface/L1TriggerScalerRead.h"
#include<iostream>
#include<vector>

L1TriggerScalerHandler::L1TriggerScalerHandler(const edm::ParameterSet& pset) :
  m_name(pset.getUntrackedParameter<std::string>("name","L1TriggerScalerHandler")),
  // m_connect(pset.getUntrackedParameter<std::string>("OnlineConn","")),
  //m_authpath(pset.getUntrackedParameter<std::string>("OnlineAuthPath",".")),
  //m_host(pset.getUntrackedParameter<std::string>("OnlineDBHost","cmsor1-v.cern.ch")),
  // m_sid(pset.getUntrackedParameter<std::string>("OnlineDBSID","omds")),
  
  m_user(pset.getUntrackedParameter<std::string>("OnlineDBUser","CMS_RUNINFO")), 
  m_pass(pset.getUntrackedParameter<std::string>("OnlineDBPass","XXXXXXX"))
  // m_port(pset.getUntrackedParameter<int>("OnlineDBPort",10121))
{
  m_connectionString= "oracle://cms_omds_lb/CMS_RUNINFO";
 
}

L1TriggerScalerHandler::~L1TriggerScalerHandler()
{
 
}

void L1TriggerScalerHandler::getNewObjects() {
   edm::LogInfo   ("L1TriggerScalerHandler") << "------- " << m_name 
	     << " - > getNewObjects\n" << 
  //check whats already inside of database
      "got offlineInfo"<<
    tagInfo().name << ", size " << tagInfo().size 
            << ", last object valid since " 
	    << tagInfo().lastInterval.first << " token "   
            << tagInfo().lastPayloadToken << std::endl;
  

  if (tagInfo().size>0) {
    Ref payload = lastPayload();
    edm::LogInfo   ("L1TriggerScalerHandler")<<"size of last payload  "<< 
      payload->m_run.size()<<std::endl;
  }

  int snc;
  
  std::cerr << "Source implementation test ::getNewObjects : enter runnumber as a first since !  \n";
  std::cin >> snc;




  std::cout<<"runnumber/first since = "<< snc <<std::endl;
  
 
  ///... understand how  to know in advise the lumisection_number
 
   L1TriggerScaler  * r = new L1TriggerScaler(); 

   
   
   // reading from omds
   L1TriggerScalerRead rn( m_connectionString, m_user, m_pass);
   std::vector<L1TriggerScaler::Lumi> l1lumiscaler_array;

   

 l1lumiscaler_array = rn.readData((int)snc );

  L1TriggerScaler::Lumi rnfill;
  std::vector<L1TriggerScaler::Lumi>::iterator Iit;
  for(Iit = l1lumiscaler_array.begin() ; Iit != l1lumiscaler_array.end(); Iit++)
    {
      rnfill = *(Iit);
      r->m_run.push_back(rnfill);   
    }
  
   m_to_transfer.push_back(std::make_pair((L1TriggerScaler*)r,snc));
   std::ostringstream ss; 
   ss << "since =" << snc;
    
  

  m_userTextLog = ss.str()+";";


  edm::LogInfo   ("L1TriggerScalerHandler") << "------- " << m_name << " - > getNewObjects" << std::endl;

 
}




