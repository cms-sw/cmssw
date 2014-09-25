/* -*- C++ -*- */
#ifndef HcalDCCHeader_H
#define HcalDCCHeader_H

#include <iostream>
#include "DataFormats/HcalDigi/interface/HcalCalibrationEventTypes.h"
class HcalHTRData;

/**  \class HcalDCCHeader
 *
 *  Interpretive class for an HcalDCCHeader
 *   
 *
 *  \author J. Mans - UMD
 */

class HcalDCCHeader {
 public:
  static const int SPIGOT_COUNT;

  HcalDCCHeader();

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
  inline unsigned long getDCCEventNumber() const { return (commondataformat1 & 0x00FFFFFF); }
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

  //// The 64-bit DCC Header 
  inline short getDCCDataFormatVersion() const { return (dcch0 & 0xFF); }
  inline int getAcceptTimeTTS() const { return ((dcch0>>8)& 0x0000000F); }
  inline int getByte1Zeroes() const {  return ((dcch0>>12)& 0x00000003); }
  inline int getHTRStatusBits () const { return ((dcch0>>14)& 0x00007FFF); }
  inline int getByte3Zeroes() const {  return ((dcch0>>29)& 0x00000007); }
  inline int getDCCStatus() const {return (dcch1 & 0x000003FF);}
  inline int getByte567Zeroes() const {  return (dcch1 & 0xFF00FC00); }

  /** Get the value flagging a spigot's summary of error flags. */
  inline bool getSpigotErrorFlag(int nspigot) const { 
    return (( dcch0>>(14+nspigot) )&0x0001);  }

  /** Get the status of these error counters in the DCC motherboard. **/
  inline bool SawTTS_OFW()        const { return ((getDCCStatus()>>0) & 0x00000001);}
  inline bool SawTTS_BSY()        const { return ((getDCCStatus()>>1) & 0x00000001);}
  inline bool SawTTS_SYN()        const { return ((getDCCStatus()>>2) & 0x00000001);}
  inline bool SawL1A_EvN_MxMx()   const { return ((getDCCStatus()>>3) & 0x00000001);}
  inline bool SawL1A_BcN_MxMx()   const { return ((getDCCStatus()>>4) & 0x00000001);}
  inline bool SawCT_EvN_MxMx()    const { return ((getDCCStatus()>>5) & 0x00000001);}
  inline bool SawCT_BcN_MxMx()    const { return ((getDCCStatus()>>6) & 0x00000001);}
  inline bool SawOrbitLengthErr() const { return ((getDCCStatus()>>7) & 0x00000001);}
  inline bool SawTTC_SingErr()    const { return ((getDCCStatus()>>8) & 0x00000001);}
  inline bool SawTTC_DoubErr()    const { return ((getDCCStatus()>>9) & 0x00000001);}

  /** Get a given spigot summary from the DCC Header **/
  inline int getSpigotSummary(int nspigot) const { return spigotInfo[nspigot]; }

  /** Load the given decoder with the pointer and length from this spigot 
      Returns 0 on success
      Returns -1 if spigot points to data area beyond validSize
   */
  int getSpigotData(int nspigot, HcalHTRData& decodeTool, int validSize) const;

  /** Old (unsafe) getSpigotData.  All calls should be replaced by the new call.
      This call will generate WARN log messages.
   */
  void getSpigotData(int nspigot, HcalHTRData& decodeTool) const;

  /** Get the size (in 32-bit words) of the data from this spigot */
  inline unsigned int getSpigotDataLength(int nspigot) const { return (nspigot>=15)?(0):(spigotInfo[nspigot]&0x3ff); }

  /** \brief Read the "ENABLED" bit for this spigot */
  inline bool getSpigotEnabled(unsigned int nspigot) const { return (nspigot>=15)?(false):(spigotInfo[nspigot]&0x8000); }
  /** \brief Read the "PRESENT" bit for this spigot */
  inline bool getSpigotPresent(unsigned int nspigot) const { return (nspigot>=15)?(false):(spigotInfo[nspigot]&0x4000); }
  /** \brief Read the "BxID FAILS TO MATCH WITH DCC" bit for this spigot */
  inline bool getBxMismatchWithDCC(unsigned int nspigot) const { return (nspigot>=15)?(false):(spigotInfo[nspigot]&0x2000); }
  /** \brief Read the "VALID" bit for this spigot; TTC EvN matched HTR EvN */
  inline bool getSpigotValid(unsigned int nspigot) const { return (nspigot>=15)?(false):(spigotInfo[nspigot]&0x1000); }
  /** \brief Read the "TRUNCATED" bit for this spigot; LRB truncated data (took too long) */
  inline bool getSpigotDataTruncated(unsigned int nspigot) const { return (nspigot>=15)?(false):(spigotInfo[nspigot]&0x0800); }
  /** \brief Read the "CRC-Mismatch" bit for this spigot */
  inline bool getSpigotCRCError(unsigned int nspigot) const { return (nspigot>=15)?(false):(spigotInfo[nspigot]&0x0400); }
  /** \brief Access the HTR error bits (decoding tbd) */
  inline unsigned char getSpigotErrorBits(unsigned int nspigot) const { return (nspigot>=15)?(0):((unsigned char)(spigotInfo[nspigot]>>24)); }
  /** \brief Access the Link Receiver Board error bits (decoding tbd) */
  inline unsigned char getLRBErrorBits(unsigned int nspigot) const { return (nspigot>=15)?(0):((unsigned char)(spigotInfo[nspigot]>>16)); }

  /* (for packing only) */
  /** \brief Add the given HcalHTRData as the given spigot's data.  This should be done in increasing spigot order!
      \param spigot_id 
      \param spigot_data 
      \param valid flag
      \param LRB_error_word
  */
  void copySpigotData(unsigned int spigot_id, const HcalHTRData& data, bool valid=true, unsigned char LRB_error_word=0);

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
  uint32_t spigotInfo[18];   //The last three of these 32bit words should always be zero!

};

std::ostream& operator<<(std::ostream&, const HcalDCCHeader& head);

#endif
