/* -*- C++ -*- */
#ifndef HcalDCCHeader_H
#define HcalDCCHeader_H

#include <iostream>
class HcalHTRData;

/**  \class HcalDCCHeader
 *
 *  Interpretive class for an HcalDCCHeader
 *   
 *
 *  $Date: 2005/06/06 19:29:29 $
 *  $Revision: 1.1 $
 *  \author J. Mans - UMD
 */

class HcalDCCHeader {
 public:
  static const int SPIGOT_COUNT;

  HcalDCCHeader();

  /** Determine the expected total length of this packet in bytes*/
  unsigned int getTotalLengthBytes() const; 

  /** Load the given decoder with the pointer and length from this spigot */
  void getSpigotData(int nspigot, HcalHTRData& decodeTool) const;

  /** Get the size (in 32-bit words) of the data from this spigot */
  inline unsigned int getSpigotDataLength(int nspigot) const { return (nspigot>=15)?(0):(spigotInfo[nspigot]&0x3ff); }

  /** \brief Read the "ENABLED" bit for this spigot */
  inline bool getSpigotEnabled(unsigned int nspigot) const { return (nspigot>=15)?(false):(spigotInfo[nspigot]&0x8000); }
  /** \brief Read the "PRESENT" bit for this spigot */
  inline bool getSpigotPresent(unsigned int nspigot) const { return (nspigot>=15)?(false):(spigotInfo[nspigot]&0x4000); }
  /** \brief Read the "VALID" bit for this spigot */
  inline bool getSpigotValid(unsigned int nspigot) const { return (nspigot>=15)?(false):(spigotInfo[nspigot]&0x2000); }
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

  /** get the source id from the CDF header */
  inline int getSourceId() const { return (commondataformat0>>8)&0xFFF; }
  /** get the bunch id from the CDF header */
  inline int getBunchId() const { return (commondataformat0>>20)&0xFFF; }

 private:
  // CURRENTLY VALID FOR LITTLE-ENDIAN (LINUX/x86) ONLY
  unsigned int commondataformat0;
  unsigned int commondataformat1;
  unsigned int commondataformat2;
  unsigned int commondataformat3;
  unsigned int dcch0;
  unsigned int dcch1;
  unsigned int spigotInfo[18];
};

std::ostream& operator<<(std::ostream&, const HcalDCCHeader& head);

#endif
