// -*- Mode: C++; c-basic-offset: 2; indent-tabs-mode: t; tab-width: 8; -*-
//$Id: MatacqRawEvent.h,v 1.2 2012/06/11 08:57:01 davidlt Exp $

#ifndef MATACQRAWEVENT_H
#define MATACQRAWEVENT_H

#include <inttypes.h>
#include <vector>

#if 0 //replace 1 by 0 to remove XDAQ dependency. In this case it is assumed the
      //machine is little endian.
#include "i2o/utils/endian.h" //from XDAQ
#define UINT32_FROM_LE i2odecodel
#define UINT16_FROM_LE i2odecodes
#define INT16_FROM_LE i2odecodes

#else //assuming little endianness of the machine

#define UINT32_FROM_LE
#define UINT16_FROM_LE
#define INT16_FROM_LE

#endif

#include <sys/time.h> //for timeval definition

/** Wrapper for matacq raw event fragments. This class provides the
 * method to interpret the data. 
 */
class MatacqRawEvent{
  //typedefs, enums and static constants
public:
  enum matacqError_t {
    /** Event length is specified both in the data header and the trailer. This
     * flags indicates an inconsitency between the two indications.
     */
    errorLengthConsistency = 1<<0,
    /** Error in data length.
     */
    errorLength = 1<<1,
    /** Wrong Begin of event flag
     */
    errorWrongBoe = 1<<2
  };

  /* The following types are little-endian encoded types. Use of these types
   * for the I20 data block should offer portability to big-endian platforms.
   */
  //@{
  struct uint32le_t{
    uint32_t littleEndianInt;
    operator uint32_t() const{
      return UINT32_FROM_LE(littleEndianInt);
    }
  };
  
  struct uint16le_t{
    uint16_t littleEndianInt;
    operator uint16_t() const{
      return UINT16_FROM_LE(littleEndianInt);
    }
  };
  
  struct int16le_t{
    int16_t littleEndianInt;
    operator int16_t() const{
      return INT16_FROM_LE(littleEndianInt);
    }
  };
  //@}
  
  struct ChannelData{
    /** Matacq channel number. From 0 to 3.
     */
    int chId;
    /** Number of samples in following array
     */
    int nSamples;
    /** ADC count of time samples. Array of size nSamples
     */
    const int16le_t* samples;
  };

private:  
  /** Matacq header data structure
   */
//   struct matacqHeader_t{		 
//     uint16_t version;	 
//     unsigned char freqGHz;	 
//     unsigned char channelCount; 
//     uint32_t orbitId;
//     unsigned char trigRec;
//     uint16_t postTrig;
//     uint16_t vernier[4];
//     uint32_t timeStamp;
//   };                                          

  /** Specification of DAQ header field.
   */
  struct field32spec_t{
    int offset;
    unsigned int mask;
  };
  
  //@{  
  /** DAQ header field specifications.
   */
  static const field32spec_t fov32;
  static const field32spec_t fedId32;
  static const field32spec_t bxId32;
  static const field32spec_t lv132;
  static const field32spec_t triggerType32;
  static const field32spec_t boeType32;
  static const field32spec_t dccLen32;
  static const field32spec_t dccErrors32;
  static const field32spec_t runNum32;
  static const field32spec_t h1Marker32;
  //@}
  
  //@{
  /** Matacq header field specifications.
   */
  static const field32spec_t formatVersion32;
  static const field32spec_t freqGHz32;
  static const field32spec_t channelCount32;
  static const field32spec_t timeStamp32;
  static const field32spec_t tTrigPs32;
  static const field32spec_t orbitId32;
  static const field32spec_t trigRec32;
  static const field32spec_t postTrig32;
  static const field32spec_t vernier0_32;
  static const field32spec_t vernier1_32;
  static const field32spec_t vernier2_32;
  static const field32spec_t vernier3_32;
  static const field32spec_t timeStampMicroSec32;
  
  static const field32spec_t laserPower32;
  static const field32spec_t attenuation_dB32;
  static const field32spec_t emtcPhase32;
  static const field32spec_t emtcDelay32;
  static const field32spec_t delayA32;
  static const field32spec_t dccId32;
  static const field32spec_t color32;
  static const field32spec_t trigType32;
  static const field32spec_t side32;

  //@}
  
  //constructors
public:
  /** Constuctor.
   * @param dataBuffer pointer to the raw data. Beware the data are not copied,
   * therefore the data must be kept valid during the lifetime of the
   * constructed object. pData must be aligned at least on 32-bit words.
   * @param bufferSize size of the buffer pointed by dataBuffer and containing
   * the data. The data themselves are allowed to be smaller than the buffer.
   * @throw std::exception if the data cannot be decoded due to data corruption
   * or truncation.
   */
  MatacqRawEvent(const unsigned char* dataBuffer, size_t bufferSize): vernier(std::vector<int>(4)){
    setRawData(dataBuffer, bufferSize);
  }
  //methods
public:
  /** Gets the Fed event fragment data format (FOV) field content.
   * Currently the FOV is not used for MATACQ.
   * Note that matacq data format has its own internal version. See
   * #getMatacqDataFormatVersion()
   * @return FOV
   */
  int getFov() const { return read32(daqHeader, fov32);}
  
  /** Gets the FED ID field contents. Should be 655.
   * @return FED ID
   */
  int getFedId() const { return read32(daqHeader, fedId32);}

  /** Gets the bunch crossing id field contents.
   * @return BX id
   */
  int getBxId() const { return read32(daqHeader, bxId32);}

  /** Gets the LV1 field contents.
   * @return LV1 id
   */
  unsigned getEventId() const { return read32(daqHeader, lv132);}

  /** Gets the trigger type field contents.
   * @return trigger type
   */
  int getTriggerType() const { return read32(daqHeader, triggerType32);}

  /** Gets the beging of event field contents (BOE). Must be 0x5.
   * @return BOE
   */
  int getBoe() const { return read32(daqHeader, boeType32);}

  /** Gets the event length specifies in the "a la DCC" header.
   * @return event length
   */
  unsigned getDccLen() const { return read32(daqHeader, dccLen32);}

   /** Gets the event length specifies in the DCC-type header of a matacq event.
    * @param data buffer. Needs to contains at least the 3 first 32-bit words
    * of the event.
    * @param buffer size
    * @return event length, 0xFFFFFFFF if failed to retrieve dcc length
    */
   static unsigned getDccLen(unsigned char* data, size_t size){
     if(size<(unsigned)(dccLen32.offset+1)*4) return (unsigned)-1;
     return read32((uint32le_t*) data, dccLen32);
   }

   /** Gets the orbit id from the header of a matacq event. Data format
    * of the event must be >=3.
    * @param data buffer. Needs to contains at least the 8 first 32-bit words
    * of the event.
    * @param buffer size
    * @return event length, 0xFFFFFFFF if failed to retrieve dcc length
    */
   static unsigned getOrbitId(unsigned char* data, size_t size){
     if(size<(unsigned)(orbitId32.offset+1)*8) return (unsigned)-1;
     return read32((uint32le_t*) data, orbitId32);
   }

   /** Gets the run number from the  header of a matacq event.
    * @param data buffer. Needs to contains at least the 4 first 32-bit words
    * of the event.
    * @param buffer size
    * @return event length, 0xFFFFFFFF if failed to retrieve dcc length
    */
   static unsigned getRunNum(unsigned char* data, size_t size){
     if(size<(unsigned)(runNum32.offset+1)*8) return (unsigned)-1;
     return read32((uint32le_t*) data, runNum32);
   }

  
  /** Gets the event length specifies in the DAQ trailer
   * @return event length
   */
  unsigned getDaqLen() const { return fragLen;}


  /** Gets the contents of the DCC error field. Currently Not used for Matacq.
   * @return dcc error
   */
  int getDccErrors() const { return read32(daqHeader, dccErrors32);}

  /** Gets the run number field contents.
   * @return run number
   */
  unsigned getRunNum() const { return read32(daqHeader, runNum32);}

  /** Gets the header marker field contents. Must be 1
   * @return H1 header marker
   */
  int getH1Marker() const { return read32(daqHeader, h1Marker32);}
  

  /** Gets the matcq data format version
   * @return data version
   */
  int getMatacqDataFormatVersion() const { return matacqDataFormatVersion;}

  /** Gets the raw data status. Bitwise OR of the error flags
   * defined by matcqError_t
   * @return status
   */
  int32_t getStatus() const { return error;}

  /** Gets the matacq sampling frequency field contents.
   * @return sampling frequency in GHz: 1 or 2
   */
  int getFreqGHz() const { return /*matacqHeader->*/freqGHz;}

  /** Gets the matacq channel count field contents.
   * @return number of channels
   */
  int getChannelCount() const { return /*matacqHeader->*/channelCount;}

  /** Gets the matacq channel data. Beware that no copy is done and that
   * the returned data will be invalidated if the data contains in the buffer
   * is modified (see constructor and #setRawData().
   * @return matacq channel data.
   */
  const std::vector<ChannelData>& getChannelData() const{
    return channelData;
  }

  /** Gets the data length in number of 64-bit words computed by the data
   * parser.
   * @return event length
   */
  int getParsedLen() {  return parsedLen; }
  
  /** Gets the matacq data timestamp field contents: 
   * @return acquisition date of the data expressed in number of "elapsed"
   * second since the EPOCH as defined in POSIX.1. See time()  standard c
   * function.
   */
  time_t getTimeStamp() const { return /*matacqHeader->*/timeStamp.tv_sec; }

  /** Gets the matacq data timestamp with fine granularity (89.1us) 
   * @return acquisition date of the data expressed in number of "elapsed"
   * second and microseconds since the EPOCH as defined in POSIX.1.
   * See time() standard c function and gettimeofday UNIX function.
   */
  void getTimeStamp(struct timeval& t) const { t = timeStamp; }

  /** Gets the Matacq trigger time.
   * @return (t_trig-t_0) in ps, with t_0 the time of the first sample and
   * t_trig the trigger time.
   */
  int getTTrigPs() const { return tTrigPs; }

  /** Gets the LHC orbit ID of the event
   * Available only for Matacq data format version >=3 and for P5 data.
   * @return the LHC orbit ID
   */
  uint32_t getOrbitId() const { return orbitId; }

  /** Gets the Trig Rec value (see Matacq documentation)
   * Available only for Matacq data format version >=3.
   * @return the Trig Rec value
   */
  int getTrigRec() const { return trigRec; }

  /** Posttrig value (see Matacq documentation).
   * Available only for Matacq data format version >=3.
   */
  int getPostTrig() const { return postTrig; }

  /** Vernier values (see Matacq documentation)
   * Available only for Matacq data format version >=3.
   */
  std::vector<int> getVernier() const { return vernier; }

  /** "Delay A" setting of laser delay box in ns.
   */
  int getDelayA() const { return delayA; }

  /**  WTE-to-Laser delay of EMTC in LHC clock unit.
   */
  int getEmtcDelay() const { return emtcDelay; }
  
  /** EMTC laser phase in 1/8th LHC clock unit.
   */
  int getEmtcPhase() const { return emtcPhase; }

  /**  Logarithmic attenuator setting in -10dB unit. Between 0 and
   *  5*(-10dB), 0xF if unknown.
   */
  int getAttenuation_dB() const { return attenuation_dB; }

  /** Laser power in percents (set with the linear attenuator).
   */
  int getLaserPower() const { return laserPower; }
  
private:
  /** Help function to decode header content.
   * @param data pointer
   * @param spec specification of the data field to read
   * @param ovfTrans switch of overflow translation. If true the
   * MSB of the data field is interpreted as an overflow bit. If
   * it is set, then -1 is returned.
   * @return content of data field specified by 'spec'
   */
  static int read32(uint32le_t* pData, field32spec_t spec, bool ovfTrans = false);
  
//   /** Help function to get the maximum value of a data field
//    * @param spec32 data field specification
//    * @return maximum value
//    */
//   int max32(field32spec_t spec32) const;

  /** Changes the raw data pointer and updates accordingly this object. 
   * @param buffer new pointer to the data buffer. Must be aligned at least
   * on 32-bit words.
   * @param size of the data buffer.
   * @throw std::exception if the data cannot be decoded due to data corruption
   * or truncation.
   */
  void setRawData(const unsigned char* buffer, size_t bufferSize);

  //fields
private:
  /** Begin Of Event marker
   */
  int boe;

  /** Bunch crossing Id 
   */
  int bxId;

  /** Number of matacq channels in the data.
   */
  int channelCount;
  
  /** Channel samples
   */
  std::vector<ChannelData> channelData;

  /** Pointer to the standard CMS DAQ header
   */
  uint32le_t* daqHeader;

  /** DCC error field content.
   */
  int dccErrors;

  /** Event length specified in 'DCC' header
   */
  unsigned dccLen;

  /** Event id. Actually LV1 ID.
   */
  unsigned eventId;

  /** Error code or 0 if no error.
   */
  int32_t error;

  /** FED ID
   */
  int fedId;

  /** FED data format version
   */
  int fov;

  /** event fragment length as read in the std DAQ trailer. In 64-bit words
   */
  int fragLen;

  /** MATACQ sampling frequency in GHz
   */
  int freqGHz;

  /** header marker
   */
  int h1Marker;

  /**Matacq header:
   */
  //  matacqHeader_t* matacqHeader;
  
  /** MATACQ data format internal version
   */
  int matacqDataFormatVersion;

  /** event lenght computed by the raw data parser
   */
  int parsedLen;

  /** Pointer to MATACQ samples block
   */
  uint16le_t* pSamples;

  /** Run number
   */
  unsigned runNum;

  /** Matacq acquisition time stamp
   */
  struct timeval timeStamp;

  /** MATACQ trigger time position in ps
   */
  int tTrigPs;
  
  /** Trigger type
   */
  int triggerType;

  /* LHC orbit ID
   */
  uint32_t orbitId;

  /** Trig Rec value (see Matacq documentation)
   */
  int trigRec;

  /** Posttrig value (see Matacq documentation)
   */
  int postTrig;

  /** Vernier values (see Matacq documentation)
   */
  std::vector<int> vernier;

  /** "Delay A" setting of laser delay box in ns.
   */
  int delayA;

  /**  WTE-to-Laser delay of EMTC in LHC clock unit.
   */
  int emtcDelay;    
  
  /** EMTC laser phase in 1/8th LHC clock unit.
   */
  int emtcPhase;

  /**  Logarithmic attenuator setting in -10dB unit. Between 0 and
   *  5*(-10dB), 0xF if unknown.
   */
  int attenuation_dB;

  /** Laser power in percents (set with the linear attenuator).
   */
  int laserPower;
};

#endif //MATACQRAWEVENT_H not defined
