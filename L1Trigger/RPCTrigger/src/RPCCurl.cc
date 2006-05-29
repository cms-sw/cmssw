#include "L1Trigger/RPCTrigger/src/RPCCurl.h"
#include "L1Trigger/RPCTrigger/src/RPCDetInfo.h"




RPCCurl::RPCCurl(){ }


RPCCurl::~RPCCurl(){ }

//
// Adds detId tu the curl
// TODO: implement check if added detInfo  _does_ belong to this RPCCurl
//
bool RPCCurl::addDetId(RPCDetInfo detInfo){

  //if ( mRPCDetInfoMap.find(detInfo.mDetId)==mRPCDetInfoMap.end() ){
    mRPCDetInfoMap[detInfo.rawId()]=detInfo; 
  //}
    return true;
}

//
//
//
void RPCCurl::printContents() const{
  /*
  std::cout << "CurlId" << "(imlement_me) "
            << " No. of RPCDetInfo's " << mRPCDetInfoMap.size()
            //<< " hwPlane " <<  (mRPCDetInfoMap.begin()->second()).
            << std::endl;
  */
}

