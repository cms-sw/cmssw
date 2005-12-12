/** \file
 *
 *  Implementation of RPCBXData
 *
 *  $Date: 2005/11/24 18:17:10 $
 *  $Revision: 1.1 $
 *  \author Ilaria Segoni
 */


#include <EventFilter/RPCRawToDigi/interface/RPCBXData.h>

#include <iostream>

using namespace std;


RPCBXData::RPCBXData(const unsigned int* index): 
    word_(index) {
    
    bx_= (*word_ >> BX_SHIFT )& BX_MASK ;
       
}


int RPCBXData::bx(){  
  return bx_;
}

