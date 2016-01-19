#include "CondCore/CondDB/interface/Types.h"
#include "DQM/RPCMonitorClient/interface/RPCDBHandler.h"

RPCDBHandler::RPCDBHandler(const edm::ParameterSet& iConfig) : 
 m_name(iConfig.getUntrackedParameter<std::string>("name","RPCDBHandler")), 
 sinceTime(iConfig.getUntrackedParameter<unsigned>("IOVRun",0))
{}

RPCDBHandler::~RPCDBHandler(){}

void RPCDBHandler::getNewObjects()
{


  cond::Time_t myTime = sinceTime;
 
  //  std::cout << "sinceTime= " << myTime << std::endl;

  size_t n_empty_run = 0;
  if(tagInfo().size > 0  && (tagInfo().lastInterval.first+1) < myTime) {
    n_empty_run = myTime - tagInfo().lastInterval.first - 1; 
  } 

  if (n_empty_run != 0) {
    RPCDQMObject * r = new RPCDQMObject();
    m_to_transfer.push_back(std::make_pair((RPCDQMObject*) (r->Fake_RPCDQMObject()),tagInfo().lastInterval.first + 1));
  }

  m_to_transfer.push_back(std::make_pair(rpcDQMObject,myTime));
   
}

void RPCDBHandler::initObject(RPCDQMObject* fObject){
  rpcDQMObject = fObject;
}
