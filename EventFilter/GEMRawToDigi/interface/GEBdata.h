#ifndef GEM_GEBdata
#define GEM_GEBdata

#include <vector>
#include "VFATdata.h"

//!A class for GEB data
/**
   The number after the ":" indicates how many bits a certain item consists of. 
*/
namespace gem {
  class GEBdata
  {
  public:
    GEBdata(){};
    /* GEBdata(const uint32_t &ZeroSup_,  */
    /* 	  const uint8_t &InputID_,  */
    /* 	  const uint16_t &Vwh_,  */
    /* 	  const uint16_t &ErrorC_,  */
    /* 	  const uint16_t &OHCRC_,  */
    /* 	  const uint16_t &Vwt_, */
    /* 	  const uint8_t &InFu_, */
    /* 	  const uint8_t &Stuckd_) :  */
    /*   m_ZeroSup(ZeroSup_),                                 */
    /*     m_InputID(InputID_), */
    /*     m_Vwh(Vwh_),                                   */
    /*     m_ErrorC(ErrorC_), */
    /*     m_OHCRC(OHCRC_),                                     */
    /*     m_Vwt(Vwt_),                              */
    /*     m_InFu(InFu_),                                    */
    /*     m_Stuckd(Stuckd_){}          */
    /*   ~GEBdata(){vfatd.clear();} */

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
    uint64_t getChamberHeader()
    {
      return
	(static_cast<uint64_t>(m_ZeroSup & 0x00ffffff) <<  40) |
	(static_cast<uint64_t>(m_InputID & 0b00011111) <<  35) |
	(static_cast<uint64_t>(m_Vwh & 0x0fff) <<  23) |
	(static_cast<uint64_t>(m_ErrorC & 0b0001111111111111));
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
    std::vector<uint8_t> GEBflag()
    {
      return v_GEBflags;
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
    uint64_t getChamberTrailer()
    {
      return
	(static_cast<uint64_t>(m_OHCRC) <<  48) |
	(static_cast<uint64_t>(m_Vwt & 0x0fff) <<  36) |
	(static_cast<uint64_t>(m_InFu & 0x0f) <<  35) |
	(static_cast<uint64_t>(m_Stuckd & 0x01) << 34);
    }

    void setVwh(uint16_t n){m_Vwh = n;}       ///<Returns VFAT word count (size of VFAT payload)
    void setInputID(uint8_t n){m_InputID = n;}///<Returns GLIB input ID

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

  };
}
#endif
