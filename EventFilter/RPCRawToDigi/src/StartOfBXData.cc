/** \file
 *
 *  $Date: 2005/11/07 15:42:32 $
 *  $Revision: 1.1 $
 *  \author Ilaria Segoni
 */


#include <EventFilter/RPCRawToDigi/interface/StartOfBXData.h>

StartOfBXData::StartOfBXData(const unsigned char* index): 
    word_(reinterpret_cast<const unsigned int*>(index)) {
    
    bx_= (*word_ >> BX_SHIFT )& BX_MASK ;

}


int StartOfBXData::bx(){  
  return bx_;
}

