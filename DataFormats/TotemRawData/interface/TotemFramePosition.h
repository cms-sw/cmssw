/****************************************************************************
*
* This is a part of the TOTEM testbeam/monitoring software.
* This is a part of the TOTEM offline software.
* Authors: 
*   Jan Kašpar (jan.kaspar@gmail.com) 
*
****************************************************************************/


#ifndef _TotemFramePosition_h_
#define _TotemFramePosition_h_

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

    /// the preferred constructor
    TotemFramePosition(unsigned short SubSystemId, unsigned short TOTFEDId, unsigned short OptoRxId, unsigned short GOHId, unsigned short IdxInFiber) :
      rawPosition(IdxInFiber | GOHId<<4 | OptoRxId<<8 | TOTFEDId<<10 | SubSystemId<<15) {}

    /// don't use this constructor unless you have a good reason
    TotemFramePosition(unsigned int pos = 0) : rawPosition(pos) {}

    ~TotemFramePosition() {}

    unsigned short GetSubSystemId() const { return (rawPosition>>15)&0x7; }
    unsigned short GetTOTFEDId() const    { return (rawPosition>>10)&0x1F;}
    unsigned short GetOptoRxId() const    { return (rawPosition>> 8)&0x3; }
    unsigned short GetGOHId() const       { return (rawPosition>> 4)&0xF; }
    unsigned short GetIdxInFiber() const  { return (rawPosition>> 0)&0xF; }
    
    void SetSubSystemId(unsigned short v)
     { v &= 0x7; rawPosition &= 0xFFFC7FFF; rawPosition |= (v << 15); }

    void SetTOTFEDId(unsigned short v)
     { v &= 0x1F; rawPosition &= 0xFFFF83FF; rawPosition |= (v << 10); }

    void SetOptoRxId(unsigned short v)
     { v &= 0x3; rawPosition &= 0xFFFFFCFF; rawPosition |= (v << 8); }

    void SetGOHId(unsigned short v)
     { v &= 0xF; rawPosition &= 0xFFFFFF0F; rawPosition |= (v << 4); }

    void SetIdxInFiber(unsigned short v)
     { v &= 0xF; rawPosition &= 0xFFFFFFF0; rawPosition |= (v << 0); }

    void SetAllIDs(unsigned short SubSystemId, unsigned short TOTFEDId, unsigned short OptoRxId, unsigned short GOHId, unsigned short IdxInFiber)
      { rawPosition = (IdxInFiber | GOHId<<4 | OptoRxId<<8 | TOTFEDId<<10 | SubSystemId<<15); }

    /// don't use this method unless you have a good reason
    unsigned int GetRawPosition() const
      { return rawPosition; }

  public:
    bool operator < (const TotemFramePosition &pos) const
      { return (rawPosition < pos.rawPosition); }

    bool operator == (const TotemFramePosition &pos) const
      { return (rawPosition == pos.rawPosition); }

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
    void PrintXML();

    /// Sets attribute with XML name 'attribute' and value 'value'.
    /// Also turns on attribute presents bit in the flag parameter
    /// returns 0 if the attribute is known, non-zero value else
    unsigned char SetXMLAttribute(const std::string &attribute, const std::string &value, unsigned char &flag);

    /// returns true if all attributes have been set
    static bool CheckXMLAttributeFlag(unsigned char flag)
      { return ((flag == 0x1f) | (flag == 0x20) | (flag == 0x40)); } 

    unsigned short GetFullOptoRxId() const
      { return (rawPosition >> 8) & 0xFFF; }

  protected:
    unsigned int rawPosition;
};

#endif
