// Date   : 30/05/2005
// Author : N.Almeida (LIP)

#ifndef DCCTBDATAMAPPER_HH
#define DCCTBDATAMAPPER_HH


#include <string>                //STL
#include <set>

#include "DCCDataParser.h"


/*----------------------------------------------------------*/
/* DCC DATA FIELD                                           */
/* define data fields from the ECAL raw data format         */
/* a data field has a name, a word position, a bit position */
/* and a mask (number of bits)                              */
/* Note: this class is defined inline                       */
/*----------------------------------------------------------*/
class DCCTBDataField{
public : 
  /**
     Class constructor (sets data field's characteristics)
  */
  DCCTBDataField(std::string name, uint32_t wordPosition, uint32_t bitPosition, uint32_t mask){
    name_=name; wordPosition_ = wordPosition; bitPosition_= bitPosition; mask_= mask;
  }
		
  /**
     Return and set methods for field's data
  */
  void setName(std::string namestr)        { name_.clear(); name_ = namestr; }
  std::string name()                       { return name_;                   }
  void setWordPosition(uint32_t wordpos) { wordPosition_ = wordpos;        }
  uint32_t wordPosition()                { return wordPosition_;           }
  void setBitPosition(uint32_t bitpos)   { bitPosition_ = bitpos;          }
  uint32_t bitPosition()                 { return bitPosition_;            }
  void setMask(uint32_t maskvalue)       { mask_=maskvalue;                }
  uint32_t mask()                        { return mask_;                   }

  /**
     Class destructor
  */
  ~DCCTBDataField() { };
		
protected :
  std::string name_;
  uint32_t wordPosition_;
  uint32_t bitPosition_;
  uint32_t mask_;
};



/*----------------------------------------------------------*/
/* DCC DATA FIELD COMPARATOR                                */
/* compares data fields positions                           */
/*----------------------------------------------------------*/
class DCCTBDataFieldComparator{ 
 
public : 

  /** 
      Overloads operator() returning true if DCCDataField 1 comes first then DCCDataField 2 in the DCC data block
  */ 
  bool operator()(DCCTBDataField *d1, DCCTBDataField * d2) const{
    bool value(false);
    
    if (d1->wordPosition() < d2->wordPosition()){ 
      value=true;
    }
    else if(d1->wordPosition() == d2->wordPosition()){ 
      if(d1->bitPosition() > d2->bitPosition()) {
	value=true;
      } 
    }
    
    return value;  
  } 
}; 



/*----------------------------------------------------------*/
/* DCC DATA MAPPER                                          */
/* maps the data according to ECAL raw data format specs.   */
/*----------------------------------------------------------*/
class DCCTBDataMapper{
public: 
  
  DCCTBDataMapper(DCCTBDataParser * myParser );
  ~DCCTBDataMapper();

  /**
     Build methods for raw data fields
  */
  void buildDCCFields();
  void buildTCCFields();
  void buildSRPFields();
  void buildTowerFields();
  void buildXtalFields();
  void buildTrailerFields();
  
  /**
     Return methods for raw data fields
  */
  std::set<DCCTBDataField *, DCCTBDataFieldComparator> *dccFields()        { return dccFields_;        }
  std::set<DCCTBDataField *, DCCTBDataFieldComparator> *emptyEventFields() { return emptyEventFields_; }
  std::set<DCCTBDataField *, DCCTBDataFieldComparator> *tcc68Fields()      { return tcc68Fields_;      }
  std::set<DCCTBDataField *, DCCTBDataFieldComparator> *tcc32Fields()      { return tcc32Fields_;      }
  std::set<DCCTBDataField *, DCCTBDataFieldComparator> *tcc16Fields()      { return tcc16Fields_;      }
  std::set<DCCTBDataField *, DCCTBDataFieldComparator> *srp68Fields()      { return srp68Fields_;      }
  std::set<DCCTBDataField *, DCCTBDataFieldComparator> *srp32Fields()      { return srp32Fields_;      }
  std::set<DCCTBDataField *, DCCTBDataFieldComparator> *srp16Fields()      { return srp16Fields_;      }
  std::set<DCCTBDataField *, DCCTBDataFieldComparator> *towerFields()      { return towerFields_;      }
  std::set<DCCTBDataField *, DCCTBDataFieldComparator> *xtalFields()       { return xtalFields_;       }
  std::set<DCCTBDataField *, DCCTBDataFieldComparator> *trailerFields()    { return trailerFields_;    }
  
protected:
  DCCTBDataParser * parser_;
  std::set<DCCTBDataField *, DCCTBDataFieldComparator> * dccFields_;
  std::set<DCCTBDataField *, DCCTBDataFieldComparator> * emptyEventFields_;
  std::set<DCCTBDataField *, DCCTBDataFieldComparator> * tcc68Fields_;
  std::set<DCCTBDataField *, DCCTBDataFieldComparator> * tcc32Fields_;
  std::set<DCCTBDataField *, DCCTBDataFieldComparator> * tcc16Fields_;
  
  std::set<DCCTBDataField *, DCCTBDataFieldComparator> * srp68Fields_;
  std::set<DCCTBDataField *, DCCTBDataFieldComparator> * srp32Fields_;
  std::set<DCCTBDataField *, DCCTBDataFieldComparator> * srp16Fields_;
  
  std::set<DCCTBDataField *, DCCTBDataFieldComparator> * towerFields_;
  std::set<DCCTBDataField *, DCCTBDataFieldComparator> * xtalFields_;
  std::set<DCCTBDataField *, DCCTBDataFieldComparator> * trailerFields_;
  
public: 

  //HEADER data fields (each 32 bits is separated by a space in the enum)
  enum DCCFIELDS{
    H_WPOSITION                = 0, H_BPOSITION              =  3,    H_MASK              = 0x1,
    FOV_WPOSITION              = 0, FOV_BPOSITION            =  4,    FOV_MASK            = 0xF,
    DCCID_WPOSITION            = 0, DCCID_BPOSITION          =  8,    DCCID_MASK          = 0xFFF,
    DCCBX_WPOSITION            = 0, DCCBX_BPOSITION          = 20,    DCCBX_MASK          = 0xFFF,

    DCCL1_WPOSITION            = 1, DCCL1_BPOSITION          =  0,    DCCL1_MASK          = 0xFFFFFF,
    TRIGGERTYPE_WPOSITION      = 1, TRIGGERTYPE_BPOSITION    = 24,    TRIGGERTYPE_MASK    = 0xF,
    BOE_WPOSITION              = 1, BOE_BPOSITION            = 28,    BOE_MASK            = 0xF,
    
    EVENTLENGTH_WPOSITION      = 2, EVENTLENGTH_BPOSITION     =  0,    EVENTLENGTH_MASK    = 0xFFFFFF,
    DCCERRORS_WPOSITION        = 2, DCCERRORS_BPOSITION       = 24,    DCCERRORS_MASK      = 0xFF,
    
    RNUMB_WPOSITION            = 3, RNUMB_BPOSITION           =  0,    RNUMB_MASK          = 0xFFFFFF,
    HD_WPOSITION               = 3, HD_BPOSITION              = 24,    HD_MASK             = 0xFF,
    
    RUNTYPE_WPOSITION          = 4, RUNTYPE_BPOSITION         =  0,    RUNTYPE_MASK        = 0xFFFFFFFF,
    
    DETAILEDTT_WPOSITION       = 5, DETAILEDTT_BPOSITION      =  0,    DETAILEDTT_MASK     = 0xFFFF,
    
    ORBITCOUNTER_WPOSITION     = 6, ORBITCOUNTER_BPOSITION   =  0,    ORBITCOUNTER_MASK    = 0xFFFFFFFF,
			
    SR_WPOSITION               = 7, SR_BPOSITION             =  0,    SR_MASK             = 0x1,
    ZS_WPOSITION               = 7, ZS_BPOSITION             =  1,    ZS_MASK             = 0x1,
    TZS_WPOSITION              = 7, TZS_BPOSITION            =  2,    TZS_MASK            = 0x1,
    SR_CHSTATUS_WPOSITION      = 7, SR_CHSTATUS_BPOSITION    =  4,    SR_CHSTATUS_MASK    = 0xF, 
    TCC_CHSTATUS_WPOSITION     = 7, TCC_CHSTATUS_BPOSITION   =  8,    TCC_CHSTATUS_MASK   = 0xF, 
    
    FE_CHSTATUS_WPOSITION      = 8, CHSTATUS_BPOSITION       =  0,    FE_CHSTATUS_MASK    = 0xF 
  };


  //TCC block data fields
  enum TCCFIELDS{
    TCCID_WPOSITION           =  0, TCCID_BPOSITION          =  0,    TCCID_MASK          = 0xFF,
    TCCBX_WPOSITION           =  0, TCCBX_BPOSITION          = 16,    TCCBX_MASK          = 0xFFF,
    TCCE0_WPOSITION           =  0, TCCE0_BPOSITION          = 28,    TCCE0_MASK          = 0x1,
    
    TCCL1_WPOSITION           =  1, TCCL1_BPOSITION          =  0,    TCCL1_MASK          = 0xFFF,
    TCCE1_WPOSITION           =  1, TCCE1_BPOSITION          = 12,    TCCE1_MASK          = 0x1,
    NTT_WPOSITION             =  1, NTT_BPOSITION            = 16,    NTT_MASK            = 0x7F,
    TCCTSAMP_WPOSITION        =  1, TCCTSAMP_BPOSITION       = 23,    TCCTSAMP_MASK       = 0xF,
    TCCLE0_WPOSITION          =  1, TCCLE0_BPOSITION         = 27,    TCCLE0_MASK         = 0x1,
    TCCLE1_WPOSITION          =  1, TCCLE1_BPOSITION         = 28,    TCCLE1_MASK         = 0x1,
			
    TPG_WPOSITION             =  2,  TPG_BPOSITION           =  0,    TPG_MASK            = 0x1FF,
    TTF_WPOSITION             =  2,  TTF_BPOSITION           =  9,    TTF_MASK            = 0x7
  };
		
  //SR block data fields
  enum SRPFIELDS{
    SRPID_WPOSITION           =  0, SRPID_BPOSITION          =  0,    SRPID_MASK          = 0xFF,
    SRPBX_WPOSITION           =  0, SRPBX_BPOSITION          = 16,    SRPBX_MASK          = 0xFFF,
    SRPE0_WPOSITION           =  0, SRPE0_BPOSITION          = 28,    SRPE0_MASK          = 0x1,
				
    SRPL1_WPOSITION           =  1, SRPL1_BPOSITION          =  0,    SRPL1_MASK          = 0xFFF,
    SRPE1_WPOSITION           =  1, SRPE1_BPOSITION          = 12,    SRPE1_MASK          = 0x1,
    NSRF_WPOSITION            =  1, NSRF_BPOSITION           = 16,    NSRF_MASK           = 0x7F,
    SRPLE0_WPOSITION          =  1, SRPLE0_BPOSITION         = 27,    SRPLE0_MASK         = 0x1,
    SRPLE1_WPOSITION          =  1, SRPLE1_BPOSITION         = 28,    SRPLE1_MASK         = 0x1,

    SRF_WPOSITION             =  2, SRF_BPOSITION            =  0,    SRF_MASK            = 0x3, 
    SRPBOFFSET                = 16 
  };
	
  //TOWER block data fields
  enum TOWERFIELDS{
    TOWERID_WPOSITION          = 0, TOWERID_BPOSITION          =   0,   TOWERID_MASK          = 0x7F, //FEID remask?? --> the 8th bit is in use 
    XSAMP_WPOSITION            = 0, XSAMP_BPOSITION            =   8,   XSAMP_MASK            = 0x7F,
    TOWERBX_WPOSITION          = 0, TOWERBX_BPOSITION          =  16,   TOWERBX_MASK          = 0xFFF,
    TOWERE0_WPOSITION          = 0, TOWERE0_BPOSITION          =  28,   TOWERE0_MASK          = 0x1,
    
    TOWERL1_WPOSITION          = 1, TOWERL1_BPOSITION          =  0,    TOWERL1_MASK         = 0xFFF,
    TOWERE1_WPOSITION          = 1, TOWERE1_BPOSITION          = 12,    TOWERE1_MASK         = 0x1,
    TOWERLENGTH_WPOSITION      = 1, TOWERLENGTH_BPOSITION      = 16,   TOWERLENGTH_MASK      = 0x1FF			
  };
		
  //CRYSTAL data fields
  enum XTALFIELDS{
    STRIPID_WPOSITION        = 0, STRIPID_BPOSITION        =  0, STRIPID_MASK         = 0x7,
    XTALID_WPOSITION         = 0, XTALID_BPOSITION         =  4, XTALID_MASK          = 0x7,
    M_WPOSITION              = 0, M_BPOSITION              =  8, M_MASK               = 0x1,
    SMF_WPOSITION            = 0, SMF_BPOSITION            =  9, SMF_MASK             = 0x1,
    GMF_WPOSITION            = 0, GMF_BPOSITION            = 10, GMF_MASK             = 0x1,  
    XTAL_TZS_WPOSITION       = 0, XTAL_TZS_BPOSITION       = 16, XTAL_TZS_MASK        = 0x1,  
    XTAL_GDECISION_WPOSITION = 0, XTAL_GDECISION_BPOSITION = 17, XTAL_GDECISION_MASK  = 0x1,	
    ADC_WPOSITION            = 0, ADC_BPOSITION            =  0, ADC_MASK             = 0x3FFF,
    ADCBOFFSET               = 16 
  };
	
  //TRAILER data fields
  enum TRAILERFIELDS{
    T_WPOSITION                = 0, T_BPOSITION                =   3,    T_MASK               = 0x1,
    ESTAT_WPOSITION            = 0, ESTAT_BPOSITION            =   8,    ESTAT_MASK           = 0xF,
			
    TTS_WPOSITION              = 0, TTS_BPOSITION              =   4,     TTS_MASK             = 0xF,
    
    CRC_WPOSITION              = 0, CRC_BPOSITION              =  16,    CRC_MASK             = 0xFFFF,
    TLENGTH_WPOSITION          = 1, TLENGTH_BPOSITION          =   0,    TLENGTH_MASK         = 0xFFFFFF,
    EOE_WPOSITION              = 1, EOE_BPOSITION              =  28,    EOE_MASK             = 0xF
  };
				
};

#endif
