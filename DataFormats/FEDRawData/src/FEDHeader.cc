/** \file
 *
 *  $Date: 2005/10/06 18:25:22 $
 *  $Revision: 1.3 $
 *  \author N. Amapane - CERN
 */

#include <DataFormats/FEDRawData/interface/FEDHeader.h>
#include "fed_header.h"

FEDHeader::FEDHeader(const unsigned char* header) : 
  theHeader(reinterpret_cast<const fedh_t*>(header)) {}


FEDHeader::~FEDHeader(){}


int FEDHeader::triggerType(){
  return ((theHeader->eventid & FED_EVTY_MASK) >> FED_EVTY_SHIFT);
}

int FEDHeader::lvl1ID(){
  return (theHeader->eventid & FED_LVL1_MASK);
}

int FEDHeader::bxID(){
  return ((theHeader->sourceid & FED_BXID_MASK) >> FED_BXID_SHIFT);
}

int FEDHeader::sourceID(){
  return ((theHeader->sourceid & FED_SOID_MASK) >> FED_SOID_SHIFT);
}

int FEDHeader::version(){
  return ((theHeader->sourceid & FED_VERSION_MASK) >> FED_VERSION_SHIFT);
}

bool FEDHeader::moreHeaders(){
  return ((theHeader->sourceid & FED_MORE_HEADERS)!=0);
}

void FEDHeader::set(unsigned char* header,
		    int evt_ty,	   
		    int lvl1_ID,
		    int bx_ID,
		    int source_ID,
		    int version,
		    bool H){

  // FIXME: should check that input ranges are OK!!!
  fedh_t* h = reinterpret_cast<fedh_t*>(header);
  h->eventid = 
    ( FED_HCTRLID & FED_HCTRLID_MASK) | 
    ( ( evt_ty    << FED_EVTY_SHIFT) & FED_EVTY_MASK ) | 
    ( ( lvl1_ID   << FED_LVL1_SHIFT) & FED_LVL1_MASK );

  h->sourceid =
    ( ( bx_ID     << FED_BXID_SHIFT) & FED_BXID_MASK ) |
    ( ( source_ID << FED_SOID_SHIFT) & FED_SOID_MASK ) |
    ( ( version   << FED_VERSION_SHIFT) & FED_VERSION_MASK );
  
  if (H) h->sourceid |= FED_MORE_HEADERS;
    
}

bool FEDHeader::check() {
  // ...may report with finer detail
  bool result = true;
  result &= ((theHeader->eventid & FED_HCTRLID_MASK) == FED_HCTRLID);  

  return result;
}
