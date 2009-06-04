#ifndef DCCRAWDATADEFINITIONS_
#define DCCRAWDATADEFINITIONS_


enum globalFieds{

  BLOCK_UNPACKED = 0, 
  SKIP_BLOCK_UNPACKING=1, 
  STOP_EVENT_UNPACKING=2, 


  B_MASK               =  1,
  HEADERLENGTH         =  9,
  HEADERSIZE           = 72,
  EMPTYEVENTSIZE       = 32,
	
  PHYSICTRIGGER        = 1,
  CALIBRATIONTRIGGER   = 2,
  TESTTRIGGER          = 3,
  TECHNICALTRIGGER     = 4,
      
  CH_ENABLED           = 0,
  CH_DISABLED          = 1,
  CH_TIMEOUT           = 2,
  CH_HEADERERR         = 3,
  CH_LINKERR           = 5,
  CH_LENGTHERR         = 6,
  CH_SUPPRESS          = 7,
  CH_IFIFOFULL         = 8,
  CH_L1AIFIFOFULL      = 0xC,


  SRP_NREAD            = 0,
  SRP_NUMBFLAGS        = 68,
  SRP_BLOCKLENGTH      = 6,
  SRP_EB_NUMBFLAGS     = 68,
  
  BOEVALUE             = 0x5, 
  ERROR_EMPTYEVENT     = 0x1, 		
  TOWERH_SIZE          = 8, 
  TRAILER_SIZE         = 8,
  TCC_EB_NUMBTTS       = 68,
  TCCID_SMID_SHIFT_EB  = 27,
  
  NUMB_SM             = 54,
  NUMB_FE             = 68,
  NUMB_TCC            = 108,
  NUMB_XTAL           = 5,
  NUMB_STRIP          = 5,
  NUMB_PSEUDOSTRIPS   = 30, // input1 and input2 of TCC board has at most 30 PS_input (12 of which are duplicated)
  NUMB_TTS_TPG2_DUPL  = 12, //
  NUMB_TTS_TPG1       = 16, // input1 of TCC board has at most 16 TP's
  NUMB_TTS_TPG2       = 12, // input2 of TCC board has at most 12 TP's

  NUMB_SM_EE_MIN_MIN  = 1,
  NUMB_SM_EE_MIN_MAX  = 9,
  NUMB_SM_EB_MIN_MIN  = 10,
  NUMB_SM_EB_MIN_MAX  = 27,
  NUMB_SM_EB_PLU_MIN  = 28,
  NUMB_SM_EB_PLU_MAX  = 45,
  NUMB_SM_EE_PLU_MIN  = 46,
  NUMB_SM_EE_PLU_MAX  = 54,

  // two DCC have a missing interval in the CCU_id's
  SECTOR_EEM_CCU_JUMP = 8,
  SECTOR_EEP_CCU_JUMP = 53,
  MIN_CCUID_JUMP      = 18,
  MAX_CCUID_JUMP      = 24,
  
  NUMB_TCC_EE_MIN_EXT_MIN  = 19,     // outer TCC's in EE-
  NUMB_TCC_EE_MIN_EXT_MAX  = 36,
  NUMB_TCC_EE_PLU_EXT_MIN  = 73,    // outer TCC's in EE+
  NUMB_TCC_EE_PLU_EXT_MAX  = 90

};



enum headerFields{ 
          
  H_FEDID_B            = 8,
  H_FEDID_MASK         = 0xFFF,
 
  H_BX_B               = 20,
  H_BX_MASK            = 0xFFF,
      
  H_L1_B               = 32,
  H_L1_MASK            = 0xFFFFFF,

  H_TTYPE_B            = 56,
  H_TTYPE_MASK         = 0xF,    

  H_EVLENGTH_MASK      = 0xFFFFFF,
      
  H_ERRORS_B           = 24,
  H_ERRORS_MASK        = 0xFF,

  H_RNUMB_B            = 32,
  H_RNUMB_MASK         = 0xFFFFFF,

  H_RTYPE_MASK         = 0xFFFFFFFF, // bits 0.. 31 of the 3rd DCC header word

  H_DET_TTYPE_B        = 32,
  H_DET_TTYPE_MASK     = 0xFFFF,     // for bits 32.. 47 of the 3rd DCC header word

   	
  H_FOV_B              = 48,
  H_FOV_MASK           = 0xF,


  H_ORBITCOUNTER_B            = 0,
  H_ORBITCOUNTER_MASK         = 0xFFFFFFFF, // bits 0.. 31 of the 4th DCC header word

  H_SR_B               = 32,
  H_ZS_B               = 33,
  H_TZS_B              = 34,
  H_MEM_B              = 35,
        
  H_SRCHSTATUS_B       = 36,
  H_CHSTATUS_MASK      = 0xF,

  H_TCC1CHSTATUS_B     = 40, 
  H_TCC2CHSTATUS_B     = 44,
  H_TCC3CHSTATUS_B     = 48,
  H_TCC4CHSTATUS_B     = 52
     

};		


/* 1st TTC Command */
/*                 Half :       1 bits: 7                 1st Half (0), 2nd Half (1) */
/*                 TE           1 bit : 6                 Test Enable Identifier */
/*                 Type         2 bits: 5-4               Laser (00), LED (01) Test pulse (10), Pedestal (11) */
/*                 Color        2 bits: 3-2               Blue (00), Red(01), Infrared (10), Green (11) */

/* 2nd TCC Command */
/*                 DCC #:     6 bits: 5-0.              DCC 1 to 54. Zero means all DCC */


enum detailedTriggerTypeFields{ 

   H_DCCID_B            = 0,
   H_DCCID_MASK         = 0x3F,

   H_WAVEL_B            = 6,
   H_WAVEL_MASK         = 0x3,

   H_TR_TYPE_B          = 8,
   H_TR_TYPE_MASK       = 0x7,

   H_HALF_B             = 11,
   H_HALF_MASK          = 0x1

};


enum towerFields{ 
       
  TOWER_ID_MASK        = 0x7F,
  
  TOWER_NSAMP_MASK     = 0x7F,
  TOWER_NSAMP_B        = 8,  
      
  TOWER_BX_MASK        = 0xFFF,
  TOWER_BX_B           = 16,     
 
  TOWER_L1_MASK        = 0xFFF,
  TOWER_L1_B           = 32,
      
  TOWER_ADC_MASK       = 0xFFF,
  TOWER_DIGI_MASK      = 0x3FFF,
      
  TOWER_STRIPID_MASK   = 0x7,
      
  TOWER_XTALID_MASK    = 0x7,
  TOWER_XTALID_B       = 4,


  TOWER_LENGTH_MASK    = 0x1FF,
  TOWER_LENGTH_B       = 48

};	


enum tccFields{
  
   TCC_ID_MASK         = 0xFF,
   
   TCC_PS_B            = 11,
 
   TCC_BX_MASK         = 0xFFF,
   TCC_BX_B            = 16,

   TCC_L1_MASK         = 0xFFF, 
   TCC_L1_B            = 32,  

   TCC_TT_MASK         = 0x7F,
   TCC_TT_B            = 48,

   TCC_TS_MASK         = 0xF,
   TCC_TS_B            = 55

};


enum srpFields{
  
   SRP_ID_MASK         = 0xFF,
 
   SRP_BX_MASK         = 0xFFF,
   SRP_BX_B            = 16,

   SRP_L1_MASK         = 0xFFF, 
   SRP_L1_B            = 32,  

   SRP_NFLAGS_MASK     = 0x7F,
   SRP_NFLAGS_B        = 48,
	
   SRP_SRFLAG_MASK     = 0x7,
   SRP_SRVAL_MASK      = 0x3

};

enum dccFOVs{
  // MC raw data based on CMS NOTE 2005/021
  // (and raw data when FOV was unassigned, earlier than mid 2008)
  dcc_FOV_0           = 0,

  // real data since ever FOV was initialized; only 2 used >= June 09 
  dcc_FOV_1           = 1,
  dcc_FOV_2           = 2
 
};


#endif	
