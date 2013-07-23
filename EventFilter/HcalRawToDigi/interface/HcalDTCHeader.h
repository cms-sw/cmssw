/* -*- C++ -*- */
#ifndef HcalDTCHeader_H
#define HcalDTCHeader_H

#include <iostream>
#include <stdint.h>
#include "DataFormats/HcalDigi/interface/HcalCalibrationEventTypes.h"
class HcalHTRData;

/**  \class HcalDTCHeader
 *
 *  Interpretive class for the header of a FED-format data block
 *  from the DTC -- the uTCA DAQ and timing card also called an AMC13
 *
 *  $Date: 2009/09/23 18:29:44 $
 *  $Revision: 1.9 $
 *  \author J. Mans - UMD
 */

class HcalDTCHeader {
 public:
  static const int SLOT_COUNT;
  static const int MINIMUM_SLOT;
  static const int MAXIMUM_SLOT;

  HcalDTCHeader();

  /** Determine the expected total length of this packet in bytes*/
  unsigned int getTotalLengthBytes() const; 

  //// The First Common Data Format Slink64 word. 
  /** get the bit indicating that another CDF header Slink64 word follows the first one.*/
  inline bool thereIsASecondCDFHeaderWord() const {return ((commondataformat0>>3) & 0x0001); }
  /** Get the Format Version of the Common Data Format */
  inline short getCDFversionNumber() const {return ((commondataformat0>>4) & 0x0F); }
  /** get the source id from the CDF header */
  inline int getSourceId() const { return (commondataformat0>>8)&0xFFF; }
  /** get the bunch id from the CDF header */
  inline int getBunchId() const { return (commondataformat0>>20)&0xFFF; }
  /** get the Event Number from the CDF header */
  inline unsigned long getDTCEventNumber() const { return (commondataformat1 & 0x00FFFFFF); }
  /** Get the Event Type value (2007.11.03 - Not defined, but should stay consistent among events.) */
  inline unsigned short getCDFEventType() const { return ( (commondataformat1>>24) & 0x0F ); }
  /** Get the inviolable '5' in the highest 4 bits of the CDF header.*/
  inline unsigned short BOEshouldBe5Always() const { return ( (commondataformat1>>28) & 0x0F ); }

  //// The Second Common Data Format Slink64 word. 
  /** Check the third bit of second Slink64 CDF word */
  inline bool thereIsAThirdCDFHeaderWord() const {return ((commondataformat2>>3) & 0x0001); }
  /** Get the Orbit Number from the CDF. */
  inline unsigned int getOrbitNumber() const { return ( ((commondataformat3 && 0xF) << 28) + ( commondataformat2>>4) ); }
  /** get the (undefined) 'Reserved' part of the second Slink64 CDF word */
  inline unsigned int getSlink64ReservedBits() const { return (  (commondataformat3>>4)&0x00FFFFFF ); }
  /** Get the Beginning Of Event bits.  If it's not the first or last CDF Slink64 word, the high 4 bits must be zero.*/
  inline short BOEshouldBeZeroAlways() const { return ( (commondataformat3>>28) & 0x0F); }

  /** Get the Calibration Type*/
  inline bool isCalibType() const { return ( 0 != getCalibType ());}
  inline HcalCalibrationEventType  getCalibType() const { return HcalCalibrationEventType ((commondataformat3>>24)&0x0000000F);}

  //// The 64-bit DTC Header 
  inline short getDTCDataFormatVersion() const { return (dcch0 & 0xFF); }
  inline int nSlotWords() const { return ((dcch0>>20)& 0x0000000F); }
  inline int nDTCWords() const { return ((dcch0>>8)& 0x00000FFF); }
  
  /** Get a given slot summary from the DTC Header **/
  inline int getSlotSummary(int nslot) const { return slotInfo[nslot]; }

  /** Load the given decoder with the pointer and length from this slot 
      Returns 0 on success
      Returns -1 if slot points to data area beyond validSize
   */
  int getSlotData(int nslot, HcalHTRData& decodeTool, int validSize) const;


  /** Get the size (in 16-bit words) of the data from this slot */
  inline unsigned int getSlotDataLength(int nslot) const { return (nslot<1 || nslot>12)?(0):(slotInfo[nslot-1]&0xfff); }

  /** \brief Read the "ENABLED" bit for this slot */
  inline bool getSlotEnabled(unsigned int nslot) const { return (nslot<1 || nslot>12)?(false):(slotInfo[nslot-1]&0x8000); }
  /** \brief Read the "PRESENT" bit for this slot */
  inline bool getSlotPresent(unsigned int nslot) const { return (nslot<1 || nslot>12)?(false):(slotInfo[nslot-1]&0x4000); }
  /** \brief Read the "VALID" bit for this slot; TTC EvN matched HTR EvN */
  inline bool getSlotValid(unsigned int nslot) const { return (nslot<1 || nslot>12)?(false):(slotInfo[nslot-1]&0x2000); }
  /** \brief Read the "CRC-Mismatch" bit for this slot */
  inline bool getSlotCRCError(unsigned int nslot) const { return (nslot<1 || nslot>12)?(false):(slotInfo[nslot-1]&0x1000); }

  /* (for packing only) */
  /** \brief Add the given HcalHTRData as the given slot's data.  This should be done in increasing slot order!
      \param slot_id 
      \param slot_data 
      \param valid flag
      \param LRB_error_word
  */
  void copySlotData(unsigned int slot_id, const HcalHTRData& data, bool valid=true);

  /** clear the contents of this header */
  void clear();
  /** setup the header */
  void setHeader(int sourceid, int bcn, int l1aN, int orbN);

 private:
  // CURRENTLY VALID FOR LITTLE-ENDIAN (LINUX/x86) ONLY
  uint32_t commondataformat0;
  uint32_t commondataformat1;
  uint32_t commondataformat2;
  uint32_t commondataformat3;
  uint32_t dcch0;
  uint32_t dcch1;
  uint16_t slotInfo[12];

};

std::ostream& operator<<(std::ostream&, const HcalDTCHeader& head);

#endif
