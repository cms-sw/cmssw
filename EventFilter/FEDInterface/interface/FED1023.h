#ifndef EVF_FED1023_H
#define EVF_FED1023_H

#include "EventFilter/FEDInterface/interface/FEDConstants.h"

namespace evf{
  namespace fedinterface{
    
    ///////////////////////////////////////////////////////////////////////////
    //FED1023 format definition
    //Offsets go right to left - in BYTES !!!
    //LW\SB 63..............................32 31..............................0
    //00    <=======================FED HEADER ================================>
    //01    <63-44  PCId   ><43-32  RBInst#  > <31-0    RB event count         >
    //02    <63-00  Reserved						       >
    //03    <63-00  In wallclock time in RB				       >
    //04    <63-48  EPPId  ><47-32  EPinst#  > <31-0    EP event Id            >
    //05    <63-48  EP VER ><47-32 CMSSW VER > <31-0    EP event count         >
    //06    <63-00  EP History						       >
    //07    <63-00  HLT algo bits 1					       >
    //08    <63-00  HLT algo bits 2					       >
    //09    <63-00  HLT algo bits 3     				       >
    //0a    <63-00  Reserved						       >
    //0b    <63-00  Reserved						       >
    //0c    <63-00  Out wallclock time from EP				       >
    //0d    <=======================FED TRAILER ===============================>
    ////////////////////////////////////////////////////////////////////////////
    const unsigned int EVFFED_ID              = 1023;
    const unsigned int EVFFED_VERSION         = 0x0;
    const unsigned int EVFFED_TOTALSIZE       = 112; // in bytes
    const unsigned int EVFFED_LENGTH          = EVFFED_TOTALSIZE/evtn::SLINK_WORD_SIZE; // in SLINK WORDS
    const unsigned int EVFFED_PAYLOAD_OFFSET  = evtn::FED_HEADER_SIZE;
    const unsigned int EVFFED_RBEVCNT_OFFSET  = EVFFED_PAYLOAD_OFFSET;
    const unsigned int EVFFED_RBIDENT_OFFSET  = EVFFED_RBEVCNT_OFFSET + evtn::SLINK_HALFWORD_SIZE;
    const unsigned int EVFFED_RBPCIDE_MASK    = 0x000fffff;
    const unsigned int EVFFED_RBPCIDE_SHIFT   = 12;
    const unsigned int EVFFED_RBINSTA_MASK    = 0x00000fff;
    const unsigned int EVFFED_RBINSTA_SHIFT   = 0;
    const unsigned int EVFFED_RBWCTIM_OFFSET  = EVFFED_RBIDENT_OFFSET + 3 * evtn::SLINK_HALFWORD_SIZE;
    const unsigned int EVFFED_EPEVENT_OFFSET  = EVFFED_RBWCTIM_OFFSET + evtn::SLINK_WORD_SIZE;
    const unsigned int EVFFED_EPIDENT_OFFSET  = EVFFED_EPEVENT_OFFSET + evtn::SLINK_HALFWORD_SIZE;
    const unsigned int EVFFED_EPPCIDE_MASK    = 0x000fffff;
    const unsigned int EVFFED_EPPCIDE_SHIFT   = 12; 
    const unsigned int EVFFED_EPINSTA_MASK    = 0x00000fff;
    const unsigned int EVFFED_EPINSTA_SHIFT   = 0;
    const unsigned int EVFFED_EPEVTCT_OFFSET  = EVFFED_EPIDENT_OFFSET + evtn::SLINK_HALFWORD_SIZE;
    const unsigned int EVFFED_EPVERSN_OFFSET  = EVFFED_EPEVTCT_OFFSET + evtn::SLINK_HALFWORD_SIZE;
    const unsigned int EVFFED_EPVERSN_MASK    = 0xffff0000;
    const unsigned int EVFFED_EPVERSN_SHIFT   = 16;
    const unsigned int EVFFED_CMSSWVN_MASK    = 0x0000ffff;
    const unsigned int EVFFED_CMSSWVN_SHIFT   = 0;
    const unsigned int EVFFED_EPHISTO_OFFSET  = EVFFED_EPVERSN_OFFSET + evtn::SLINK_HALFWORD_SIZE;
    const unsigned int EVFFED_EPHLTA1_OFFSET  = EVFFED_EPHISTO_OFFSET + evtn::SLINK_WORD_SIZE;
    const unsigned int EVFFED_EPHLTA2_OFFSET  = EVFFED_EPHLTA1_OFFSET + evtn::SLINK_WORD_SIZE;
    const unsigned int EVFFED_EPHLTA3_OFFSET  = EVFFED_EPHLTA2_OFFSET + evtn::SLINK_WORD_SIZE;
    const unsigned int EVFFED_EPWCTIM_OFFSET  = EVFFED_EPHLTA3_OFFSET + 3 * evtn::SLINK_WORD_SIZE;
  }
}
#endif
