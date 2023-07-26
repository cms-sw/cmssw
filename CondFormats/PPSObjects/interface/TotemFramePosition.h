/****************************************************************************
*
* This is a part of the TOTEM offline software.
* Authors: 
*   Jan Kašpar (jan.kaspar@gmail.com) 
*
****************************************************************************/

#ifndef CondFormats_PPSObjects_TotemFramePosition
#define CondFormats_PPSObjects_TotemFramePosition

#include <iostream>
#include <string>
#include "CondFormats/Serialization/interface/Serializable.h"

/**
 * Uniquely identifies the DAQ channel through which a VFAT frame has been received.
 * 
 * The internal representation has the following structure:
 * \verbatim
 * |                   32 bits raw position                    |
 * | 12 bits | 2 bits |  10 bits | 4 bits |       4 bits       |
 * |  empty  | empty  |  FED ID  | GOH ID | index within fiber |
 * \endverbatim
 *
 * In the old (TOTEM only) scheme, the FED ID was further split
 * \verbatim
 * |  3 bits   |  5 bits   |  2 bits   |
 * | SubSystem | TOTFED ID | OptoRx ID |
 * \endverbatim
 * IMPORTANT: This splitting is only supported for backward compatibility and should not be used anymore.
 **/
class TotemFramePosition {
public:
  static const unsigned int offsetIdxInFiber = 0, maskIdxInFiber = 0xF;
  static const unsigned int offsetGOHId = 4, maskGOHId = 0xF;
  static const unsigned int offsetFEDId = 8, maskFEDId = 0x3FF;

  static const unsigned int offsetOptoRxId = 8, maskOptoRxId = 0x3;
  static const unsigned int offsetTOTFEDId = 10, maskTOTFEDId = 0x1F;
  static const unsigned int offsetSubSystemId = 15, maskSubSystemId = 0x7;

  /// the preferred constructor
  TotemFramePosition(unsigned short SubSystemId,
                     unsigned short TOTFEDId,
                     unsigned short OptoRxId,
                     unsigned short GOHId,
                     unsigned short IdxInFiber)
      : rawPosition(IdxInFiber << offsetIdxInFiber | GOHId << offsetGOHId | OptoRxId << offsetOptoRxId |
                    TOTFEDId << offsetTOTFEDId | SubSystemId << offsetSubSystemId) {}

  /// don't use this constructor unless you have a good reason
  TotemFramePosition(unsigned int pos = 0) : rawPosition(pos) {}

  ~TotemFramePosition() {}

  /// recomended getters and setters

  unsigned short getFEDId() const { return (rawPosition >> offsetFEDId) & maskFEDId; }

  void setFEDId(unsigned short v) {
    v &= maskFEDId;
    rawPosition &= 0xFFFFFFFF - (maskFEDId << offsetFEDId);
    rawPosition |= (v << offsetFEDId);
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

  unsigned short getSubSystemId() const { return (rawPosition >> offsetSubSystemId) & maskSubSystemId; }

  void setSubSystemId(unsigned short v) {
    v &= maskSubSystemId;
    rawPosition &= 0xFFFFFFFF - (maskSubSystemId << offsetSubSystemId);
    rawPosition |= (v << offsetSubSystemId);
  }

  unsigned short getTOTFEDId() const { return (rawPosition >> offsetTOTFEDId) & maskTOTFEDId; }

  void setTOTFEDId(unsigned short v) {
    v &= maskTOTFEDId;
    rawPosition &= 0xFFFFFFFF - (maskTOTFEDId << offsetTOTFEDId);
    rawPosition |= (v << offsetTOTFEDId);
  }

  unsigned short getOptoRxId() const { return (rawPosition >> offsetOptoRxId) & maskOptoRxId; }

  void setOptoRxId(unsigned short v) {
    v &= maskOptoRxId;
    rawPosition &= 0xFFFFFFFF - (maskOptoRxId << offsetOptoRxId);
    rawPosition |= (v << offsetOptoRxId);
  }

  /// don't use this method unless you have a good reason
  unsigned int getRawPosition() const { return rawPosition; }

  bool operator<(const TotemFramePosition &pos) const { return (rawPosition < pos.rawPosition); }

  bool operator==(const TotemFramePosition &pos) const { return (rawPosition == pos.rawPosition); }

  /// Condensed representation of the DAQ channel.
  /// prints 5-digit hex number, the digits correspond to SubSystem, TOTFED ID, OptoRx ID,
  /// GOH ID, index within fiber in this order
  friend std::ostream &operator<<(std::ostream &s, const TotemFramePosition &fp);

  /// prints XML formatted DAQ channel to stdout
  void printXML();

  /// Sets attribute with XML name 'attribute' and value 'value'.
  /// Also turns on attribute presents bit in the flag parameter
  /// returns 0 if the attribute is known, non-zero value else
  unsigned char setXMLAttribute(const std::string &attribute, const std::string &value, unsigned char &flag);

  /// returns true if all attributes have been set
  static bool checkXMLAttributeFlag(unsigned char flag) { return (flag == 0x1f); }

protected:
  unsigned int rawPosition;

  COND_SERIALIZABLE;
};

#endif
