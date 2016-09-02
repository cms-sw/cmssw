/****************************************************************************
*
* This is a part of TOTEM offline software.
* Authors:
*   Jan Kašpar (jan.kaspar@gmail.com)
*
****************************************************************************/

#ifndef DataFormats_CTPPSDigi_TotemFEDInfo
#define DataFormats_CTPPSDigi_TotemFEDInfo

/**
 * \brief OptoRx headers and footers.
 **/
class TotemFEDInfo
{
  public:
    TotemFEDInfo(int _id=0) : fedId(_id), header(0), orbitCounter(0), footer(0)
    {
    }

    void setFEDId(int _f) { fedId = _f; }
    int getFEDId() const { return fedId; }

    void setHeader(uint64_t _h) { header = _h; }
    uint8_t getBOE() const { return (header >> 60) & 0xF; }
    uint32_t getLV1() const { return (header >> 32) & 0xFFFFFF; }
    uint16_t getBX() const { return (header >> 20) & 0xFFF; }
    uint16_t getOptoRxId() const { return (header >> 8) & 0xFFF; }
    uint8_t getFOV() const { return (header >> 4) & 0xF; }
    uint8_t getH0() const { return (header >> 0) & 0xF; }

    void setOrbitCounter(uint32_t _oc) { orbitCounter = _oc; }
    uint32_t getOrbitCounter() const { return orbitCounter; }
  
    void setFooter(uint64_t _f) { footer = _f; }
    uint8_t getEOE() const { return (footer >> 60) & 0xF; }
    uint16_t getFSize() const { return (footer >> 32) & 0x3FF; }
    uint8_t getF0() const { return (footer >> 0) & 0xF; }

  private:
    /// Id from FEDRawDataCollection.
    int fedId;

    /// Data from OptoRx headers and footer.
    uint64_t header;
    uint32_t orbitCounter;
    uint64_t footer;
};

#endif
