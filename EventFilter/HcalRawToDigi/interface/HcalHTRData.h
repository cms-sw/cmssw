/* -*- C++ -*- */
#ifndef HcalHTRData_H
#define HcalHTRData_H

#include <stdint.h>

/**  \class HcalHTRData
 *
 *  Interpretive class for HcalHTRData
 *  Since this class requires external specification of the length of the data, it is implemented
 *  as an interpreter, rather than a cast-able header class.
 *
 *  $Date: 2012/06/04 11:26:21 $
 *  $Revision: 1.18 $
 *  \author J. Mans - UMD
 */

class HcalHTRData {
 public:
  static const int CHANNELS_PER_SPIGOT         = 24;
  static const int MAXIMUM_SAMPLES_PER_CHANNEL = 20;
  static const int FORMAT_VERSION_COMPACT_DATA =  6;
  
  HcalHTRData();
  ~HcalHTRData() { if (m_ownData!=0) delete [] m_ownData; }
  HcalHTRData(int version_to_create);
  HcalHTRData(const unsigned short* data, int length);
  HcalHTRData(const HcalHTRData&);
  
  HcalHTRData& operator=(const HcalHTRData&);
  void allocate(int version_to_create=0);
  void adoptData(const unsigned short* data, int length);
  
  /** \brief Get the version number of this event */
  inline int getFormatVersion() const { return m_formatVersion; }
  
  /** \brief Get a pointer to the raw data */
  inline const unsigned short* getRawData() const { return m_rawConst; }
  
  /** \brief Get the length of the raw data */
  inline const int getRawLength() const { return m_rawLength; }
  
  /** \brief Check for a good event
      Requires a minimum length, matching wordcount and length, not an
      empty event */
  bool check() const;

  bool isEmptyEvent() const;
  bool isOverflowWarning() const;
  bool isBusy() const;

  
  /** \brief Obtain the starting and ending pointers for external
     unpacking of the data
      \param daq_first Pointer to a pointer to the start of the DAQ data
      \param daq_last Pointer to a pointer to the end of the DAQ data
      \param tp_first Pointer to a pointer to the start of the TP data
      \param tp_last Pointer to a pointer to the end of the TP data
  */
  void dataPointers(const unsigned short** daq_first,
		    const unsigned short** daq_last,
		    const unsigned short** tp_first,
		    const unsigned short** tp_last) const;
  
  
  /** \brief Unpack the HTR data into TP and DAQ data sorted by channel
      \param daq_lengths unsigned char[24] of lengths.  High bit set
      indicates error with this channel
      \param daq_samples unsigned short [24*20] of data
      \param tp_lengths  unsigned char[24] of lengths
      \param tp_samples  unsigned short [24*20] of data
  */
  void unpack(unsigned char* daq_lengths, unsigned short* daq_samples,
	      unsigned char* tp_lengths, unsigned short* tp_samples) const;
  
  /** \brief Unpack special histogramming mode data
      \param fiber
      \param fiberchan
      \param capid Capacitor id for which to extract a histogram
      \param histogram unsigned int[32] into which the data should be
      deposited
  */
  bool unpackHistogram(int fiber, int fiberchan, int capid, unsigned
		       short* histogram) const;

  /** \brief Unpack a per-channel header word (compact format)
   */
  static bool unpack_per_channel_header(unsigned short, int& flav, int& error_flags, int& capid0, int& channelid);
  
  /** \brief check top bit to see if this is a compact format channel header word
   */
  static bool is_channel_header(unsigned short value) { return (value&0x8000)!=0; }

  /** \brief Unpack the HTR data into TP and DAQ data sorted by channel
      \param daq_lengths unsigned char[24] of lengths
      \param daq_samples unsigned short [24*20] of data
      \param tp_lengths  unsigned char[24] of lengths
      \param tp_samples  unsigned short [24*20] of data
  */
  void pack(unsigned char* daq_lengths, unsigned short* daq_samples,
	    unsigned char* tp_lengths, unsigned short* tp_samples, bool
	    do_capid=false);
  /** \brief pack header and trailer (call _after_ pack)*/
  void packHeaderTrailer(int L1Anumber, int bcn, int submodule, int
			 orbitn, int pipeline, int ndd, int nps, int firmwareRev=0);

  /** \brief pack trailer with Mark and Pass bits */
  void packUnsuppressed(const bool* mp);
  
  
  /** \brief Get the HTR event number */
  inline unsigned int getL1ANumber() const { 
    return (m_rawConst[0]&0xFF)+(m_rawConst[1]<<8); 
  }
  /** \brief Get the HTR bunch number */
  inline unsigned int getBunchNumber() const { 
    return (m_rawConst[4]&0xFFF); 
  }
  /** \brief Get the HTR orbit number */
  unsigned int getOrbitNumber() const;
  /** \brief Get the HTR submodule number */
  unsigned int getSubmodule() const;
  /** \brief HcalElectronicsId-style HTR slot */
  // get the htr slot
  unsigned int htrSlot() const;
  /** \brief HcalElectronicsId-style HTR top/bottom (1=top/0=bottom) */
  // get the htr top/bottom (1=top/0=bottom)
  unsigned int htrTopBottom() const;
  /** \brief HcalElectronicsId-style VME crate number */
  // get the readout VME crate number
  unsigned int readoutVMECrateId() const;
  /** \brief Is this event a calibration-stream event? */
  bool isCalibrationStream() const;
  /** \brief Is this event an unsuppresed event? */
  bool isUnsuppressed() const;
  /** \brief Was this channel passed as part of Mark&Pass ZS?*/
  bool wasMarkAndPassZS(int fiber, int fiberchan) const;
  /** \brief Was this channel passed as part of Mark&Pass ZS?*/
  bool wasMarkAndPassZSTP(int slb, int slbchan) const;
  /** \brief ZS Bunch Mask (if available) */
  uint32_t zsBunchMask() const;
  
  /** \brief Is this event a pattern-ram event? */
  bool isPatternRAMEvent() const;
  /** \brief Is this event a histogram event? (do not call standard
      unpack in this case!!!!!) */
  bool isHistogramEvent() const;
  /** \brief Get the fiber numbers for the data present in this event
      (only in histogram mode!) */
  void getHistogramFibers(int& a, int& b) const;
  /** \brief Get the pipeline length used for this event */
  unsigned int getPipelineLength() const;
  /** \brief Get the HTR firmware version */
  unsigned int getFirmwareRevision() const;
  /** \brief Get the HTR firmware flavor */
  int getFirmwareFlavor() const;
  /** \brief Get the errors word */
  inline unsigned int getErrorsWord() const { 
    return m_rawConst[2]&0xFFFF; }
  /** \brief Get the total number of precision data 16-bit words */
  int getNPrecisionWords() const;
  /** \brief Get the number of daq data samples per channel when not zero-suppressed */
  int getNDD() const;
  /** \brief Get the number of trigger data samples when not
      zero-suppressed (not available after FW 4)*/
  int getNTP() const;
  /** \brief Get the number of presamples in daq data */
  int getNPS() const;

/** \brief Get DLLunlock bits */
  inline unsigned int getDLLunlock() const { 
    return (m_rawConst[5]>>1)&0x3; }

 /** \brief Get TTCready bit */
  inline unsigned int getTTCready() const { 
    return m_rawConst[5]&0x1; }

 /** \brief Get the BCN of the Fiber Orbit Messages */
  inline unsigned int getFibOrbMsgBCN(int fiber) const {
    return (m_formatVersion==-1 || fiber<1 || fiber>8)?(0):(m_rawConst[m_rawLength-12+(fiber-1)]&0xFFF);
  }

 /** \brief Get the BCN of the Fiber Orbit Messages */
  inline unsigned int getFib1OrbMsgBCN() const {
  return (m_formatVersion==-1)?(0):(m_rawConst[m_rawLength-12]&0xFFF);
}
  inline unsigned int getFib2OrbMsgBCN() const {
  return (m_formatVersion==-1)?(0):(m_rawConst[m_rawLength-11]&0xFFF);
}

  inline unsigned int getFib3OrbMsgBCN() const {
  return (m_formatVersion==-1)?(0):(m_rawConst[m_rawLength-10]&0xFFF);
}

  inline unsigned int getFib4OrbMsgBCN() const {
  return (m_formatVersion==-1)?(0):(m_rawConst[m_rawLength-9]&0xFFF);
}

  inline unsigned int getFib5OrbMsgBCN() const {
  return (m_formatVersion==-1)?(0):(m_rawConst[m_rawLength-8]&0xFFF);
}

  inline unsigned int getFib6OrbMsgBCN() const {
  return (m_formatVersion==-1)?(0):(m_rawConst[m_rawLength-7]&0xFFF);
}

  inline unsigned int getFib7OrbMsgBCN() const {
  return (m_formatVersion==-1)?(0):(m_rawConst[m_rawLength-6]&0xFFF);
}

  inline unsigned int getFib8OrbMsgBCN() const {
  return (m_formatVersion==-1)?(0):(m_rawConst[m_rawLength-5]&0xFFF);
}



 /** \brief Get the HTR Ext Header words*/
  inline unsigned int getExtHdr1() const { 
    return (m_rawConst[0]);}
  inline unsigned int getExtHdr2() const { 
    return (m_rawConst[1]); }
  inline unsigned int getExtHdr3() const { 
    return (m_rawConst[2]);} 
  inline unsigned int getExtHdr4() const { 
    return (m_rawConst[3]); }
  inline unsigned int getExtHdr5() const { 
    return (m_rawConst[4]);} 
  inline unsigned int getExtHdr6() const { 
    return (m_rawConst[5]);} 
  inline unsigned int getExtHdr7() const { 
    return (m_rawConst[6]);} 
  inline unsigned int getExtHdr8() const { 
    return (m_rawConst[7]);}  

  /* unsigned int getFib1OrbMsgBCN() const;
  unsigned int getFib2OrbMsgBCN() const;
  unsigned int getFib3OrbMsgBCN() const;
  unsigned int getFib4OrbMsgBCN() const;
  unsigned int getFib5OrbMsgBCN() const;
  unsigned int getFib6OrbMsgBCN() const;
  unsigned int getFib7OrbMsgBCN() const;
  unsigned int getFib8OrbMsgBCN() const;
  */

  /** \brief Was there an error on the given fiber for this event (only
      in histogram mode!) */
  bool wasHistogramError(int ifiber) const;
  
protected:
  void determineSectionLengths(int& tpWords, int& daqWords, int&
			       headerWords, int& trailerWords) const;
  void determineStaticLengths(int& headerWords, int& trailerWords) const;
  int m_formatVersion;
  int m_rawLength;
  const unsigned short* m_rawConst;
  unsigned short* m_ownData;
};

#endif

