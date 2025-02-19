/* -*- C++ -*- */
#ifndef CastorMergerData_H
#define CastorMergerData_H

/**  \class CastorMergerData
 *
 *  Interpretive class for CastorMergerData
 *  Since this class requires external specification of the length of the data, it is implemented
 *  as an interpreter, rather than a cast-able header class.
 *
 *  $Date: 2009/02/20 17:46:27 $
 *  $Revision: 1.1 $
 *  \author A. Campbell - DESY
 */

class CastorMergerData {
 public:
  
  CastorMergerData();
  ~CastorMergerData() { if (m_ownData!=0) delete [] m_ownData; }
  CastorMergerData(int version_to_create);
  CastorMergerData(const unsigned short* data, int length);
  CastorMergerData(const CastorMergerData&);
  
  CastorMergerData& operator=(const CastorMergerData&);
  void allocate(int version_to_create=0);
  void adoptData(const unsigned short* data, int length);
  /** \brief Get the Merger firmware version */
  unsigned int getFirmwareRevision() const;

  /** \brief Get the errors word */
  inline unsigned int getErrorsWord() const { 
    return m_rawConst[2]&0xFFFF; }
 
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
    
  
  /** \brief Unpack the HTR data into TP and DAQ data sorted by channel
      \param daq_lengths unsigned char[24] of lengths.  High bit set
      indicates error with this channel
      \param daq_samples unsigned short [24*20] of data
      \param tp_lengths  unsigned char[24] of lengths
      \param tp_samples  unsigned short [24*20] of data
  */
  void unpack(
	      unsigned char* tp_lengths, unsigned short* tp_samples) const;
  
  
  /** \brief Unpack the HTR data into TP and DAQ data sorted by channel
      \param daq_lengths unsigned char[24] of lengths
      \param daq_samples unsigned short [24*20] of data
      \param tp_lengths  unsigned char[24] of lengths
      \param tp_samples  unsigned short [24*20] of data
  */
  void pack(
	    unsigned char* tp_lengths, unsigned short* tp_samples );
  /** \brief pack header and trailer (call _after_ pack)*/
  void packHeaderTrailer(int L1Anumber, int bcn, int submodule, int
			 orbitn, int pipeline, int ndd, int nps, int firmwareRev=0);
  
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


 /** \brief Get the Merger Ext Header words*/
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


  
protected:
  void determineSectionLengths(int& tpWords, int& headerWords, int& trailerWords) const;
  void determineStaticLengths(int& headerWords, int& trailerWords) const;

  int m_formatVersion;
  int m_rawLength;
  const unsigned short* m_rawConst; // pointer to actual raw data
  unsigned short* m_ownData;      // local block in raw data format
  unsigned short* m_unpackedData; // local data in usable format
  
};

#endif

