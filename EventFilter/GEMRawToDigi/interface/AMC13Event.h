#ifndef GEM_AMC13Event
#define GEM_AMC13Event

#include <vector>
#include "AMCdata.h"

namespace gem {
  class AMC13Event
  {
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

  };
}
#endif
