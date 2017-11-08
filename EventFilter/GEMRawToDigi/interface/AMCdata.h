#ifndef GEM_AMCdata
#define GEM_AMCdata

#include <vector>
#include "GEBdata.h"

//!A class for AMC data
/**
   The number after the ":" indicates how many bits a certain item consists of.
*/
namespace gem {
  class AMCdata
  {
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
	
  };
}
#endif
