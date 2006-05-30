/** \file RPCCurl.cc
 *
 *  $Date: 2006/05/29 12:00:00 $
 *  $Revision: 1.1 $
 *  \author Tomasz Fruboes
 */


#include "L1Trigger/RPCTrigger/src/RPCCurl.h"
#include "L1Trigger/RPCTrigger/src/RPCDetInfo.h"

RPCCurl::RPCCurl(){ }

RPCCurl::~RPCCurl(){ }

//#############################################################################
/**
*
* \brief Adds detId tu the curl
* \todo Implement check if added detInfo  _does_ belong to this RPCCurl
*
*/
//#############################################################################
bool RPCCurl::addDetId(RPCDetInfo detInfo){

  //if ( mRPCDetInfoMap.find(detInfo.mDetId)==mRPCDetInfoMap.end() ){
    mRPCDetInfoMap[detInfo.rawId()]=detInfo; 
  //}
    return true;
}

//#############################################################################
/**
*
* \brief prints the contents of a RPCurl. Commented out, as cout`s are forbidden
*
*/
//#############################################################################
void RPCCurl::printContents() const{
  /*
  std::cout << "CurlId" << "(imlement_me) "
            << " No. of RPCDetInfo's " << mRPCDetInfoMap.size()
            //<< " hwPlane " <<  (mRPCDetInfoMap.begin()->second()).
            << std::endl;
  //*/
}

