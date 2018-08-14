#ifndef EventFilter_GEMRawToDigi_AMC13Event_h
#define EventFilter_GEMRawToDigi_AMC13Event_h
#include <vector>
#include "AMCdata.h"

namespace gem {
  class AMC13Event
  {
  public:
    AMC13Event(){}
    ~AMC13Event(){m_amcHeaders.clear(); m_amcs.clear();}

    int nAMC() const {return unsigned(m_nAMC);}
    int bx_id() const {return unsigned(m_BX_id);}
    int lv1_id() const {return unsigned(m_LV1_id);}
    int source_id() const {return unsigned(m_Source_id);}
    //const std::vector<AMCdata> * amcs() const {return &m_amcs;}

    uint64_t getCDFHeader() const;
    void setCDFHeader(uint8_t cb5, uint8_t Evt_ty, uint32_t LV1_id, uint16_t BX_id, uint16_t Source_id);
    void setCDFHeader(uint64_t word);

    uint64_t getAMC13header() const;
    void setAMC13header(uint8_t CalTyp, uint8_t nAMC, uint32_t OrN, uint8_t cb0);
    void setAMC13header(uint64_t word);
    
    const std::vector<uint64_t> * getAMCheaders() const {return &m_amcHeaders;}
    void addAMCheader(uint64_t word);
    void addAMCheader(uint32_t AMC_size, uint8_t Blk_No, uint8_t AMC_No, uint16_t BoardID);

    const std::vector<AMCdata> * getAMCpayloads() const {return &m_amcs;}   
    void addAMCpayload(const AMCdata& a){m_amcs.push_back(a);}
    
    uint64_t getAMC13trailer() const;
    void setAMC13trailer(uint32_t CRC_amc13, uint8_t Blk_NoT, uint8_t LV1_idT, uint16_t BX_idT);
    void setAMC13trailer(uint64_t word);

    uint64_t getCDFTrailer() const;
    void setCDFTrailer(uint8_t cbA, uint32_t EvtLength, uint16_t CRC_cdf);      
    void setCDFTrailer(uint64_t word);
    
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
    std::vector<uint64_t> m_amcHeaders;    
    /* std::vector<uint32_t> m_AMC_size; */
    /* std::vector<uint8_t> m_Blk_No; */
    /* std::vector<uint8_t> m_AMC_No; */
    /* std::vector<uint16_t> m_BoardID; */
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
