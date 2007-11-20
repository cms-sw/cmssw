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
 *  $Date: 2007/02/19 04:05:40 $
 *  $Revision: 1.2 $
 *  \author J. Mans - UMD
 */

class HcalDCCHeader {
 public:
  static const int SPIGOT_COUNT;

  HcalDCCHeader();

  /** printf the data headers. */
  inline void printCDFheaders() const {
    printf("%08x %08x \n", commondataformat1, commondataformat0);
    if (thereIsASecondCDFHeaderWord()) {
      printf("%08x %08x \n", commondataformat3, commondataformat2);
    }
  }
  inline void printDCCheader() const {
    printf("%08x %08x \n", dcch0, dcch1);
  }
  inline void printHTRsummaries() const {
    for (int i=0; i<18; i++) {
      printf("%08x ", spigotInfo[i]);
      if ((i%2) ==0) printf("\n");
    }
  }

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
  inline unsigned int getOrbitNumber() const { return (commondataformat2>>4); }
  /** get the (undefined) 'Reserved' part of the second Slink64 CDF word */
  inline unsigned int getSlink64ReservedBits() const { return (  (commondataformat3>>4)&0x00FFFFFF ); }
  /** Get the Beginning Of Event bits.  If it's not the first or last CDF Slink64 word, the high 4 bits must be zero.*/
  inline short BOEshouldBeZeroAlways() const { return ( (commondataformat2>>28) & 0x0F); }

  //// The 64-bit DCC Header 
  /** get the DCC Data Format Version. (0x02 in 2005) */
  inline short getDCCDataFormatVersion() const { return (dcch0 & 0xFF); }
  /** get the undefined bits in the DCC header. Should never be changing. */
  inline short getDCCHeaderSchmutz() const { return ( (dcch0>>8)&0x3F); }
  /** Get the value flagging a spigot's summary of error flags. */
  inline bool getSpigotErrorFlag(int nspigot) const { 
    return (( dcch0>>(14+nspigot) )&0x0001);
  }
  /** get the high three zeros of the DCC header's low word. */
  inline short getDCCHeaderZeros() const { return  ( (dcch0>>29)&0x007); }
  /** Test the non-zero-ness of this type of DCC error. More informative labels coming soon. */
  inline bool isThisDCCErrorCounterNonZero(unsigned int countnum) const {
    return ( (dcch1>>countnum) & 0x00000001);
  }
  inline unsigned int getDCCErrorWord() const { return dcch1;}

  /** Load the given decoder with the pointer and length from this spigot */
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
  unsigned int commondataformat0;
  unsigned int commondataformat1;
  unsigned int commondataformat2;
  unsigned int commondataformat3;
  unsigned int dcch0;
  unsigned int dcch1;
  unsigned int spigotInfo[18];   //The last three of these 32bit words should always be zero!

};

std::ostream& operator<<(std::ostream&, const HcalDCCHeader& head);

#endif
