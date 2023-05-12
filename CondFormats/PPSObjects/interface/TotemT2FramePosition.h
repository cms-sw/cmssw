/****************************************************************************
*
* This is a part of the TOTEM offline software.
* Authors: 
*   Jan Kašpar (jan.kaspar@gmail.com) 
*
****************************************************************************/

#ifndef CondFormats_PPSObjects_TotemT2FramePosition
#define CondFormats_PPSObjects_TotemT2FramePosition
#include "CondFormats/PPSObjects/interface/TotemFramePosition.h"

#include <iostream>
#include <string>

/**
 * Uniquely identifies the DAQ channel through which a VFAT frame has been received.
 * 
 * The internal representation has the following structure:
 * \verbatim
 * |                   32 bits raw position                    |
 * | 12 bits | 2 bits     |  10 bits | 4 bits |       4 bits       |
 * |  empty  | T2 payload |  FED ID  | GOH ID | index within fiber |
 * \endverbatim
 *
 **/
class TotemT2FramePosition : public TotemFramePosition {
public:
  static const unsigned int offsetPayload = 18, maskPayload = 0x3;

  /// the preferred constructor
  TotemT2FramePosition(unsigned short SubSystemId,
                       unsigned short TOTFEDId,
                       unsigned short OptoRxId,
                       unsigned short GOHId,
                       unsigned short IdxInFiber,
                       unsigned short payload)
      : rawPosition(IdxInFiber << offsetIdxInFiber | GOHId << offsetGOHId | OptoRxId << offsetOptoRxId |
                    TOTFEDId << offsetTOTFEDId | SubSystemId << offsetSubSystemId | (payload + 1) << offsetPayload) {}

  /// don't use this constructor unless you have a good reason
  TotemT2FramePosition(unsigned int pos = 0) : rawPosition(pos) {}

  ~TotemT2FramePosition() {}

  /// recomended getters and setters

  unsigned short getFEDId() const { return (rawPosition >> offsetFEDId) & maskFEDId; }
  unsigned short getPayload() const { return (((rawPosition >> offsetPayload) & maskPayload) - 1); }

  void setFEDId(unsigned short v) {
    v &= maskFEDId;
    rawPosition &= 0xFFFFFFFF - (maskFEDId << offsetFEDId);
    rawPosition |= (v << offsetFEDId);
  }
  void setPayload(unsigned short v) {
    unsigned short av = (v + 1) & maskPayload;
    rawPosition &= 0xFFFFFFFF - (maskPayload << offsetPayload);
    rawPosition |= (av << offsetPayload);
  }

  unsigned short getGOHId() const { return (rawPosition >> offsetGOHId) & maskGOHId; }

  void setGOHId(unsigned short v) {
    v &= maskGOHId;
    rawPosition &= 0xFFFFFFFF - (maskGOHId << offsetGOHId);
    rawPosition |= (v << offsetGOHId);
  }

  unsigned short getIdxInFiber() const { return (rawPosition >> offsetIdxInFiber) & maskIdxInFiber; }

  void setIdxInFiber(unsigned short v) {
    v &= maskIdxInFiber;
    rawPosition &= 0xFFFFFFFF - (maskIdxInFiber << offsetIdxInFiber);
    rawPosition |= (v << offsetIdxInFiber);
  }

  /// the getters and setters below are deprecated

  /// don't use this method unless you have a good reason
  unsigned int getRawPosition() const { return rawPosition; }

  bool operator<(const TotemT2FramePosition &pos) const { return (rawPosition < pos.rawPosition); }
  bool operator<(const TotemFramePosition &pos) const { return (rawPosition < pos.getRawPosition()); }

  bool operator==(const TotemT2FramePosition &pos) const { return (rawPosition == pos.rawPosition); }
  bool operator==(const TotemFramePosition &pos) const { return (rawPosition == pos.getRawPosition()); }

  /// Condensed representation of the DAQ channel.
  /// prints 5-digit hex number, the digits correspond to SubSystem, TOTFED ID, OptoRx ID,
  /// GOH ID, index within fiber in this order
  friend std::ostream &operator<<(std::ostream &s, const TotemT2FramePosition &fp);

  /// prints XML formatted DAQ channel to stdout
  void printXML();

  /// Sets attribute with XML name 'attribute' and value 'value'.
  /// Also turns on attribute presents bit in the flag parameter
  /// returns 0 if the attribute is known, non-zero value else
  unsigned char setXMLAttribute(const std::string &attribute, const std::string &value, unsigned char &flag);

  /// returns true if all attributes have been set
  static bool checkXMLAttributeFlag(unsigned char flag) { return (flag == 0x3f); }

protected:
  unsigned int rawPosition;
};

#endif
