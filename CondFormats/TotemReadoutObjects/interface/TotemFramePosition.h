/****************************************************************************
*
* This is a part of the TOTEM offline software.
* Authors: 
*   Jan Kašpar (jan.kaspar@gmail.com) 
*
****************************************************************************/

#ifndef CondFormats_TotemReadoutObjects_TotemFramePosition
#define CondFormats_TotemReadoutObjects_TotemFramePosition

#include <iostream>
#include <string>

/**
 * Uniquely identifies the DAQ channel through which a VFAT frame has been received.
 * 
 * The internal representation has the following structure:
 * \verbatim
 * |                               32 bits raw position                                 |
 * | 12 bits | 2 bits |  3 bits   |  5 bits   |  2 bits   | 4 bits |       4 bits       |
 * |  empty  | empty  | SubSystem | TOTFED ID | OptoRx ID | GOH ID | index within fiber |
 * |         |   (this part is encoded in OptoRx header)  |                             |
 * \endverbatim
 * According to the convention SubSystemId goes from 1 to 6, TOTFEDId from 1 to 21 and OptoRx from 1 to 3.
 **/
class TotemFramePosition
{
  public:
    /// the official enumeration of DAQ subsystems
    enum SubSystemType {ssNone=0, ssT1=1, ssT2=2, ssRP=3, ssTrigger=4, ssTTC=5, ssFEC=6};

    static const unsigned int offsetIdxInFiber = 0, maskIdxInFiber = 0xF;
    static const unsigned int offsetGOHId = 4, maskGOHId = 0xF;
    static const unsigned int offsetOptoRxId = 8, maskOptoRxId = 0x3;
    static const unsigned int offsetTOTFEDId = 10, maskTOTFEDId = 0x1F;
    static const unsigned int offsetSubSystemId = 15, maskSubSystemId = 0x7;

    /// the preferred constructor
    TotemFramePosition(unsigned short SubSystemId, unsigned short TOTFEDId, unsigned short OptoRxId, unsigned short GOHId, unsigned short IdxInFiber) :
      rawPosition(IdxInFiber<<offsetIdxInFiber | GOHId<<offsetGOHId | OptoRxId<<offsetOptoRxId | TOTFEDId<<offsetTOTFEDId | SubSystemId<<offsetSubSystemId)
    {
    }

    /// don't use this constructor unless you have a good reason
    TotemFramePosition(unsigned int pos = 0) : rawPosition(pos)
    {
    }

    ~TotemFramePosition()
    {
    }

    unsigned short getSubSystemId() const { return (rawPosition >> offsetSubSystemId) & maskSubSystemId; }
    unsigned short getTOTFEDId() const    { return (rawPosition >> offsetTOTFEDId) & maskTOTFEDId;}
    unsigned short getOptoRxId() const    { return (rawPosition >> offsetOptoRxId) & maskOptoRxId; }
    unsigned short getGOHId() const       { return (rawPosition >> offsetGOHId) & maskGOHId; }
    unsigned short getIdxInFiber() const  { return (rawPosition >> offsetIdxInFiber) & maskIdxInFiber; }
    
    void setSubSystemId(unsigned short v)
    { v &= maskSubSystemId; rawPosition &= 0xFFFFFFFF - (maskSubSystemId << offsetSubSystemId); rawPosition |= (v << offsetSubSystemId); }

    void setTOTFEDId(unsigned short v)
    { v &= maskTOTFEDId; rawPosition &= 0xFFFFFFFF - (maskTOTFEDId << offsetTOTFEDId); rawPosition |= (v << offsetTOTFEDId); }

    void setOptoRxId(unsigned short v)
    { v &= maskOptoRxId; rawPosition &= 0xFFFFFFFF - (maskOptoRxId << offsetOptoRxId); rawPosition |= (v << offsetOptoRxId); }

    void setGOHId(unsigned short v)
    { v &= maskGOHId; rawPosition &= 0xFFFFFFFF - (maskGOHId << offsetGOHId); rawPosition |= (v << offsetGOHId); }

    void setIdxInFiber(unsigned short v)
    { v &= maskIdxInFiber; rawPosition &= 0xFFFFFFFF - (maskIdxInFiber << offsetIdxInFiber); rawPosition |= (v << offsetIdxInFiber); }

    void setAllIDs(unsigned short SubSystemId, unsigned short TOTFEDId, unsigned short OptoRxId, unsigned short GOHId, unsigned short IdxInFiber)
    {
      rawPosition = (IdxInFiber<<offsetIdxInFiber | GOHId<<offsetGOHId | OptoRxId<<offsetOptoRxId
        | TOTFEDId<<offsetTOTFEDId | SubSystemId<<offsetSubSystemId);
    }

    /// don't use this method unless you have a good reason
    unsigned int getRawPosition() const
    {
      return rawPosition;
    }

  public:
    bool operator < (const TotemFramePosition &pos) const
    {
      return (rawPosition < pos.rawPosition);
    }

    bool operator == (const TotemFramePosition &pos) const
    {
      return (rawPosition == pos.rawPosition);
    }

    /// Condensed representation of the DAQ channel.
    /// prints 5-digit hex number, the digits correspond to SubSystem, TOTFED ID, OptoRx ID, 
    /// GOH ID, index within fiber in this order
    friend std::ostream& operator << (std::ostream& s, const TotemFramePosition &fp);
    
    /// XML sub-system tags
    static const std::string tagSSNone; 
    static const std::string tagSSTrigger; 
    static const std::string tagSST1; 
    static const std::string tagSST2;
    static const std::string tagSSRP;
    static const std::string tagSSTTC;
    static const std::string tagSSFEC;

    /// prints XML formatted DAQ channel to stdout
    void printXML();

    /// Sets attribute with XML name 'attribute' and value 'value'.
    /// Also turns on attribute presents bit in the flag parameter
    /// returns 0 if the attribute is known, non-zero value else
    unsigned char setXMLAttribute(const std::string &attribute, const std::string &value, unsigned char &flag);

    /// returns true if all attributes have been set
    static bool checkXMLAttributeFlag(unsigned char flag)
    {
      return ((flag == 0x1f) | (flag == 0x20) | (flag == 0x40));
    }

    unsigned short getFullOptoRxId() const
    {
      return (rawPosition >> 8) & 0xFFF;
    }

  protected:
    unsigned int rawPosition;
};

#endif
