#include <vector>
class VFATdata 
{
 private:
  uint8_t  fb1010;                   ///<1010:4 Control bits, shoud be 1010
  uint16_t fBC;                      ///<Bunch Crossing number, 12 bits
  uint8_t  fb1100;                   ///<1100:4, Control bits, shoud be 1100
  uint8_t  fEC;                      ///<Event Counter, 8 bits
  uint8_t  fFlag;                    ///<Control Flags: 4 bits, Hamming Error/AFULL/SEUlogic/SUEI2C
  uint8_t  fb1110;                   ///<1110:4 Control bits, shoud be 1110
  uint16_t fChipID;                  ///<Chip ID, 12 bits
  uint64_t flsData;                  ///<channels from 1to64 
  uint64_t fmsData;                  ///<channels from 65to128
  uint16_t fcrc;                     ///<Check Sum value, 16 bits
  uint16_t fcrc_calc;                ///<Check Sum value recalculated, 16 bits
  int      fSlotNumber;              ///<Calculated chip position
  bool     fisBlockGood;             ///<Shows if block is good (control bits, chip ID and CRC checks)
  
 public:
  VFATdata(){}
 VFATdata(const uint8_t &b1010_, 
	  const uint16_t &BC_,
	  const uint8_t &b1100_, 
	  const uint8_t &EC_,
	  const uint8_t &Flag_, 
	  const uint8_t &b1110_, 
	  const uint16_t &ChipID_, 
	  const uint64_t &lsData_, 
	  const uint64_t &msData_, 
	  const uint16_t &crc_,
	  const uint16_t &crc_calc_,
	  const int &SlotNumber_,
	  const bool &isBlockGood_) : 
  fb1010(b1010_),
    fBC(BC_),
    fb1100(b1100_),
    fEC(EC_),
    fFlag(Flag_),
    fb1110(b1110_),
    fChipID(ChipID_),
    flsData(lsData_),
    fmsData(msData_),
    fcrc(crc_),
    fcrc_calc(crc_calc_),
    fSlotNumber(SlotNumber_),
    fisBlockGood(isBlockGood_){}
  ~VFATdata(){}

  //!Read first word from the block.
  void read_fw(uint64_t word)
  {
    fb1010 = 0x0f & (word >> 60);
    fBC = 0x0fff & (word >> 48);
    fb1100 = 0x0f & (word >> 44);
    fEC = word >> 36;
    fFlag = 0x0f & (word >> 32);
    fb1110 = 0x0f & (word >> 28);
    fChipID = 0x0fff & (word >> 16);
    fmsData = 0xffff000000000000 & (word << 48);
  }
    
  //!Read second word from the block.
  void read_sw(uint64_t word)
  {
    fmsData = fmsData | (0x0000ffffffffffff & word >> 16);
    flsData = 0xffff000000000000 & (word << 48);
  }
    
  //!Read third word from the block.
  void read_tw(uint64_t word)
  {
    flsData = flsData | (0x0000ffffffffffff & word >> 16);
    fcrc = word;
  }
    
  uint8_t   b1010      (){ return fb1010;      }
  uint16_t  BC         (){ return fBC;         }
  uint8_t   b1100      (){ return fb1100;      }
  uint8_t   EC         (){ return fEC;         }
  uint8_t   Flag       (){ return fFlag;       }
  uint8_t   b1110      (){ return fb1110;      }
  uint16_t  ChipID     (){ return fChipID;     }
  uint64_t  lsData     (){ return flsData;     }
  uint64_t  msData     (){ return fmsData;     }
  uint16_t  crc        (){ return fcrc;        }
  uint16_t  crc_calc   (){ return fcrc_calc;   }
  int       SlotNumber (){ return fSlotNumber; }
  bool      isBlockGood(){ return fisBlockGood;}

};

//!A class for GEB data
/**
   The number after the ":" indicates how many bits a certain item consists of. 
*/
class GEBdata
{
 private:
  std::vector<VFATdata> vfatd;     ///<Vector of VFAT data
  std::vector<uint8_t> v_GEBflags; ///<Vector for thirteen flags in GEM Chamber Header

  //GEM chamber header

  //!Zero Suppression Flags:24  (8 zeroes):8
  /**Bitmask indicating if certain VFAT blocks have been zero suppressed*/
  uint32_t m_ZeroSup;
  //!Input ID:5    000:3
  /**GLIB input ID (starting at 0)*/
  uint8_t m_InputID;   
  //!VFAT word count:12   0000:4
  /**Size of VFAT payload in 64 bit words*/
  uint16_t m_Vwh;
  //!Thirteen Flags, only one bit each
  /** 
      000:3    EvtFIFO full:1    InFIFO full:1   L1AFIFO full:1    Even size overflow:1    EvtFIFO near full:1   InFIFO near full:1    
      L1AFIFO near full:1    Event size warn:1   No VFAT marker:1    OOS GLIB VFAT:1   OOS GLIB OH:1   
      BX mismatch GLIB VFAT:1    BX mismatch GLIB OH:1
  */
  uint16_t m_ErrorC;

  //GEM chamber trailer
		
  //!OH CRC:16
  /**CRC of OH data (currently not available)*/
  uint16_t m_OHCRC;     
  //!0000:4   VFAT word count:12   
  /**Same as in header. This one actually counts the number of valid words that were sent to AMC13; the one in header is what we expected to send to AMC13*/
  uint16_t m_Vwt;      
  //!(7 0's):7    InFIFO underflow:1   
  /**Input status (critical): Input FIFO underflow occured while sending this event*/
  uint8_t m_InFu;    
  //!(7 0's):7    Stuck data:1    
  /**Input status (warning): Data in InFIFO or EvtFIFO when L1A FIFO was empty. Only resets with resync or reset*/
  uint8_t m_Stuckd; 

 public:
  GEBdata(){};
 GEBdata(const uint32_t &ZeroSup_, 
	 const uint8_t &InputID_, 
	 const uint16_t &Vwh_, 
	 const uint16_t &ErrorC_, 
	 const uint16_t &OHCRC_, 
	 const uint16_t &Vwt_,
	 const uint8_t &InFu_,
	 const uint8_t &Stuckd_) : 
  m_ZeroSup(ZeroSup_),                                
    m_InputID(InputID_),
    m_Vwh(Vwh_),                                  
    m_ErrorC(ErrorC_),
    m_OHCRC(OHCRC_),                                    
    m_Vwt(Vwt_),                             
    m_InFu(InFu_),                                   
    m_Stuckd(Stuckd_){}         
  ~GEBdata(){vfatd.clear();}

  // need to include all the flags
  //!Reads the word for the GEM Chamber Header. Puts the thirteen flags in a vector.
  /**
     Fills the Zero Suppression, GLIB Input ID, VFAT word count, and Thirteen Flags.
  */
  void setChamberHeader(uint64_t word)
  {
    m_ZeroSup = 0x00ffffff & (word >> 40);        /*!<Zero Suppression*/
    m_InputID = 0b00011111 & (word >> 35);        /*!<GLIB Input ID*/
    m_Vwh = 0x0fff & (word >> 23);                /*!<VFAT word count*/
    m_ErrorC = 0b0001111111111111 & (word);    /*!<Thirteen Flags*/
    for(int i=0; i<13; ++i)
      {
        v_GEBflags.push_back(0x01 & (m_ErrorC >> i));
      }
  }

  //return specific flags
  //!Returns one of thirteen flags from GEM chamber header.
  /**
     Argument must be between 0 and 12. The flags corresponding to a given argument are shown.
     12->EvtFIFO full    11->InFIFO full    10->L1AFIFO full   9->Even size overflow    8->EvtFIFO near full   5->InFIFO near full    
     6->L1AFIFO near full    5->Event size warn   4->No VFAT marker    3->OOS GLIB VFAT   2->OOS GLIB OH 
     1->BX mismatch GLIB VFAT    0->BX mismatch GLIB OH
  */
  uint8_t GEBflag(int c)
  {
    return v_GEBflags[c];
  }
    
  // need to include all the flags
  //!Reads the word for GEM Chamber Trailer
  /**
     Fills the OH CRC, VFAT word count, InFIFO underflow, and Stuck data.
  */
  void setChamberTrailer(uint64_t word)
  {
    m_OHCRC = word >> 48;           /*!<OH CRC*/
    m_Vwt = 0x0fff & (word >> 36);  /*!<VFAT word count*/
    m_InFu = 0x0f & (word >> 35);   /*!<InFIFO underflow*/
    m_Stuckd = 0x01 & (word >> 34); /*!<Stuck data*/
  }

  uint32_t ZeroSup()  {return m_ZeroSup;}   ///<Returns Zero Suppression flags
  uint8_t  InputID()  {return m_InputID;}   ///<Returns GLIB input ID
  uint16_t Vwh()      {return m_Vwh;}       ///<Returns VFAT word count (size of VFAT payload)
  uint16_t ErrorC()   {return m_ErrorC;}    ///<Returns thirteen flags in GEM Chamber Header

  uint16_t OHCRC()    {return m_OHCRC;}     ///<Returns OH CRC 
  uint16_t Vwt()      {return m_Vwt;}       ///<Returns VFAT word count
  uint8_t  InFu()     {return m_InFu;}      ///<Returns InFIFO underflow flag
  uint8_t  Stuckd()   {return m_Stuckd;}    ///<Returns Stuck data flag

  //!Adds VFAT data to the vector
  void v_add(VFATdata v){vfatd.push_back(v);}
  //!Returns the vector of FVAT data
  std::vector<VFATdata> vfats(){return vfatd;}  
};

//!A class for AMC data
/**
   The number after the ":" indicates how many bits a certain item consists of.
*/
class AMCdata
{
 private:
  std::vector<GEBdata> gebd;    ///<Vector of GEB data

  //AMC header #1	

  //!0000:4   AMC#:4
  /**Slot number of AMC(GLIB/MP7/EC7, etc.)*/
  uint8_t  m_AMCnum;        
  //!(8 zeroes):8    L1A ID:24    
  /**Basically like event number, but reset by resync*/
  uint32_t m_L1A;          
  //!0000:4   BX ID:12         
  /**Bunch crossing ID*/
  uint16_t m_BX;    
  //!(12 zeroes):12    Data length:20   
  /**Overall size of this FED event fragment in 64bit words (including headers and trailers)*/
  uint32_t m_Dlength;

  //AMC header #2

  uint8_t m_FV;           ///<0000:4    Format Version:4    
  //!0000:4   Run Type:4   
  /**Current version = 0x0;  Could be used to encode run types like physics, cosmics, threshold scan, etc.*/
  uint8_t m_Rtype;
  uint8_t m_Param1;       ///<Run param1:8 
  uint8_t m_Param2;       ///<Run param2:8
  uint8_t m_Param3;       ///<Run param3:8 
  uint16_t m_Onum;        ///<Orbit number:16 
  //!Board ID:16
  /**This is currently filled with 8bit long GLIB serial number*/
  uint16_t m_BID;

  //GEM event header

  //!(8 zeroes):8   GEM DAV list:24    
  /**Bitmask indicating which inputs/chambers have data*/
  uint32_t m_GEMDAV; 
  //!(30 zeroes):30    Buffer Status:34  
  /**Bitmask indicating buffer error in given inputs*/
  uint64_t m_Bstatus;  
  //!000:3   GEM DAV count:5    
  /**Number of chamber blocks*/
  uint8_t  m_GDcount;   
  //!0000:4    TTS state:4    
  /**Debug: GLIB TTS state at the moment when this event was built*/
  uint8_t  m_Tstate;

  //GEM event trailer

  //!(8 zeroes):8    Chamber timeout:24   
  /**Bitmask indicating if GLIB did not recieve data from particular input for this L1A in X amount of GTX clock cycles*/
  uint32_t m_ChamT;     
  //!(7 zeroes):7   OOS GLIB:1   
  /**GLIB is out-of-sync (critical): L1A ID is different for different chambers in this event.*/
  uint8_t  m_OOSG;

  //AMC_trailer

  uint32_t m_CRC;
  uint8_t m_L1AT;
  uint32_t m_DlengthT;
	
 public:
  //!Constructor for the class
  AMCdata(){};
 AMCdata(const uint8_t &AMCnum_, 
	 const uint32_t &L1A_,
	 const uint16_t &BX_, 
	 const uint32_t &Dlength_,
	 const uint8_t &FV_,
	 const uint8_t &Rtype_, 
	 const uint8_t &Param1_, 
	 const uint8_t &Param2_, 
	 const uint8_t &Param3_, 
	 const uint16_t &Onum_, 
	 const uint16_t &BID_,
	 const uint32_t &GEMDAV_, 
	 const uint64_t &Bstatus_,
	 const uint8_t &GDcount_, 
	 const uint8_t &Tstate_,
	 const uint32_t &ChamT_,
	 const uint8_t OOSG_) :
  m_AMCnum(AMCnum_), 
    m_L1A(L1A_),
    m_BX(BX_),                                 
    m_Dlength(Dlength_),
    m_FV(FV_),
    m_Rtype(Rtype_),                                
    m_Param1(Param1_),
    m_Param2(Param2_),                                  
    m_Param3(Param3_),
    m_Onum(Onum_),                                    
    m_BID(BID_),
    m_GEMDAV(GEMDAV_), 
    m_Bstatus(Bstatus_),
    m_GDcount(GDcount_),                                 
    m_Tstate(Tstate_),                             
    m_ChamT(ChamT_),                                   
    m_OOSG(OOSG_){}
  //!Destructor for the class
  ~AMCdata(){gebd.clear();}

  //!Reads the word for AMC Header 
  /**
     Fills the AMC number, L1A ID, BX ID, and Data Length
  */
  void setAMCheader1(uint64_t word)
  {
    m_AMCnum = 0x0f & (word >> 56);     /*!<AMC number*/
    m_L1A = 0x00ffffff & (word >> 32);  /*!<L1A ID */
    m_BX = 0x0fff & (word >> 20);       /*!<BX ID */
    m_Dlength = 0x000fffff & word;      /*!<Data Length */
  }
    
  //!Reads the word for the AMC Header 2
  /**
     Fills the Format Version, Run Type, Run Param 1, Run Param 2, Run Param 3, Orbit Number, and Board ID
  */
  void setAMCheader2(uint64_t word)
  {
    m_FV = 0x0f & (word >> 60);     /*!<Format Version */
    m_Rtype = 0x0f & (word >> 56);  /*!<Run Type */
    m_Param1 = word >> 48;          /*!<Run Param 1 */
    m_Param2 = word >> 40;          /*!<Run Param 2 */
    m_Param3 = word >> 32;          /*!<Run Param 3 */
    m_Onum = word >> 16;            /*!<Orbit Number */
    m_BID = word;                   /*!Board ID */
  }
    
  //!Reads the word for the GEM Event Header
  /**
     Fills the GEM DAV list, Buffer Status, GEM DAV count, and TTS state.
  */
  void setGEMeventHeader(uint64_t word)
  {
    m_GEMDAV = 0x00ffffff & (word >> 40);   /*!<GEM DAV list*/
    m_Bstatus = 0x00ffffff & (word >> 16);  /*!<Buffer Status*/
    m_GDcount = 0b00011111 & (word >> 11);  /*!<GEM DAV count*/
    m_Tstate = 0b00000111 & word;           /*!<TTS state*/
  }

  //!Reads the word for the GEM Event Trailer
  /**
     Fills the Chamber Timeout and OOS GLIB.
  */
  void setGEMeventTrailer(uint64_t word)
  {
    m_ChamT = 0x00ffffff & (word >> 40);  /*!<Chamber Timeout*/
    m_OOSG = 0b00000001 & (word >> 39);   /*!<OOS GLIB*/
  }

  //!Reads the word for the AMC Trailer
  void setAMCTrailer(uint64_t word)
  {
    m_CRC = word >> 32;
    m_L1AT = word >> 24;
    m_DlengthT = 0x000fffff & word;
  }

  uint8_t  AMCnum()  {return m_AMCnum;}   ///<Returns AMC number
  uint32_t L1A()     {return m_L1A;}      ///<Returns L1A number
  uint16_t BX()      {return m_BX;}       ///<Returns Bunch Crossing ID
  uint32_t Dlength() {return m_Dlength;}  ///<Returns Data Length (Overall size of FED event fragment)

  uint8_t  FV()      {return m_FV;}       ///<Returns Format Version
  uint8_t  Rtype()   {return m_Rtype;}    ///<Returns Run Type
  uint8_t  Param1()  {return m_Param1;}   
  uint8_t  Param2()  {return m_Param2;}
  uint8_t  Param3()  {return m_Param3;}
  uint16_t Onum()    {return m_Onum;}     ///<Returns Orbit number
  uint16_t BID()     {return m_BID;}      ///<Returns Board ID

  uint32_t GEMDAV ()  {return m_GEMDAV;}        ///<Returns GEM DAV list (which chambers have data)
  uint64_t Bstatus()  {return m_Bstatus;}       ///<Returns Buffer status
  int  GDcount()  {return unsigned(m_GDcount);} ///<Returns GEM DAV count (number of chamber blocks)
  uint8_t  Tstate()   {return m_Tstate;}        ///<Returns TTS state

  uint32_t ChamT()    {return m_ChamT;}   ///<Return Chamber Timeout 
  uint8_t  OOSG()     {return m_OOSG;}    ///<Return OOS GLIB (if GLIB is out of sync)

  uint32_t CRC()    {return m_CRC;}
  uint8_t L1AT()    {return m_L1AT;}
  uint32_t DlengthT()    {return m_DlengthT;}

  //!Adds GEB data to vector
  void g_add(GEBdata g){gebd.push_back(g);}
  //!Returns a vector of GEB data
  std::vector<GEBdata> gebs(){return gebd;}
};

class AMC13Event
{
 private:
  // CDF Header
  uint8_t m_cb5; // control bit, should be 0x5 bits 60-63
  uint8_t m_Evt_ty;
  uint32_t m_LV1_id;
  uint16_t m_BX_id;
  uint16_t m_Source_id;
  // AMC13 header
  uint8_t m_CalTyp;
  uint8_t m_nAMC;
  uint32_t m_OrN;
  uint8_t m_cb0; // control bit, should be 0b0000
  // AMC headers
  std::vector<uint32_t> m_AMC_size;
  std::vector<uint8_t> m_Blk_No;
  std::vector<uint8_t> m_AMC_No;
  std::vector<uint16_t> m_BoardID;
  // AMCs payload
  std::vector<AMCdata> m_amcs;
  //AMC13 trailer
  uint32_t m_CRC_amc13;
  uint8_t m_Blk_NoT;
  uint8_t m_LV1_idT;
  uint16_t m_BX_idT;
  //CDF trailer
  uint8_t m_cbA; // control bit, should be 0xA bits 60-63
  uint32_t m_EvtLength;
  uint16_t m_CRC_cdf;

 public:

  AMC13Event(){}
  ~AMC13Event(){m_AMC_size.clear(); m_Blk_No.clear(); m_AMC_No.clear(); m_BoardID.clear(); m_amcs.clear();}

  int nAMC(){return unsigned(m_nAMC);}
  int LV1_id(){return unsigned(m_LV1_id);}
  std::vector<AMCdata> amcs(){return m_amcs;}
  //*** Set the CDF header. Not full header implemented yet. Doc:http://ohm.bu.edu/~hazen/CMS/AMC13/AMC13DataFormatDrawingv3.pdf
  void setCDFHeader(uint64_t word)
  {
    m_cb5 = 0x0f & (word >> 60);
    m_Evt_ty = 0x0f & (word >> 56);
    m_LV1_id = 0x00ffffff & (word >> 32);
    m_BX_id = 0x0fff & (word >> 20);
    m_Source_id = 0x0fff & (word >> 8);
  }
  //!Sets the AMC13 header
  /**
     Fills m_CalTyp, m_nAMC, m_OrN, and m_cb0
  */
  void setAMC13header(uint64_t word)
  {
    m_CalTyp = 0x0f & (word >> 56);
    m_nAMC = 0x0f & (word >> 52);
    m_OrN = word >> 4;
    m_cb0 = 0x0f & word;
  }
  //!Adds to various vectors
  /**
     Adds to m_AMC_size, m_Blk_No, m_AMC_No, and m_BoardID.
  */
  void addAMCheader(uint64_t word)
  {
    m_AMC_size.push_back(0x00ffffff&(word>>32));
    m_Blk_No.push_back(0xff&(word>>20));
    m_AMC_No.push_back(0x0f&(word>>16));
    m_BoardID.push_back(0xffff&word);
  }
  //!Adds to m_amcs vector
  void addAMCpayload(AMCdata a){m_amcs.push_back(a);}
  //!Sets the AMC13 trailer
  /**
     Fills m_CRC_amc13, m_Blk_NoT, m_LV1_idT, and m_BX_idT
  */
  void setAMC13trailer(uint64_t word)
  {
    m_CRC_amc13 = word >> 32;
    m_Blk_NoT = 0xff & (word >> 20);
    m_LV1_idT = 0xff & (word >> 12);
    m_BX_idT = 0x0fff & word;
  }
  //!Sets CDF Trailer
  /**
     Fills m_cbA, m_EvtLength, and m_CRC_cdf.
  */
  void setCDFTrailer(uint64_t word)
  {
    m_cbA = 0x0f & (word >> 60);
    m_EvtLength = 0x00ffffff & (word >> 32);
    m_CRC_cdf = 0xffff & (word >> 16);
  }

};

