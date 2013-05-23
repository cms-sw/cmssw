#ifndef CSCDCCFormatStatusDigi_CSCDCCFormatStatusDigi_h
#define CSCDCCFormatStatusDigi_CSCDCCFormatStatusDigi_h


/*
 * =====================================================================================
 *
 *       Filename:  CSCDCCFormatStatusDigi.h
 *
 *    Description:  CSC DCC Format error, status and payload flags for a single DCC
 *
 *        Version:  1.0
 *        Created:  02/12/2009 03:22:34 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Valdas Rapsevicius (VR), valdas.rapsevicius@cern.ch,
 *                  Victor Barashko (VB), victor.barashko@cern.ch
 *        Company:  CERN, CH
 *
 * =====================================================================================
 */


#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include <set>
#include <map>
#include <algorithm>
#include <iosfwd>


/** DCC identifier type */
typedef int32_t DCCIdType;

/** DDU identifier type */
typedef int16_t DDUIdType;

/** CSC identifier type */
typedef int32_t CSCIdType;

/** Examiner status and mask type */
typedef uint32_t ExaminerMaskType;
typedef uint32_t ExaminerStatusType;

/** Format Error individual named flags */
enum FormatErrorFlag {
  ANY_ERRORS                                          = 0,
  DDU_TRAILER_MISSING                                 = 1,
  DDU_HEADER_MISSING                                  = 2,
  DDU_CRC_ERROR                                       = 3,
  DDU_WORD_COUNT_ERROR                                = 4,
  DMB_TRAILER_MISSING                                 = 5,
  DMB_HEADER_MISSING                                  = 6,
  ALCT_TRAILER_MISSING                                = 7,
  ALCT_HEADER_MISSING                                 = 8,
  ALCT_WORD_COUNT_ERROR                               = 9,
  ALCT_CRC_ERROR                                      = 10,
  ALCT_TRAILER_BIT_ERROR                              = 11,
  TMB_TRAILER_MISSING                                 = 12,
  TMB_HEADER_MISSING                                  = 13,
  TMB_WORD_COUNT_ERROR                                = 14,
  TMB_CRC_ERROR                                       = 15,
  CFEB_WORD_COUNT_PER_SAMPLE_ERROR                    = 16,
  CFEB_SAMPLE_COUNT_ERROR                             = 17,
  CFEB_CRC_ERROR                                      = 18,
  DDU_EVENT_SIZE_LIMIT_ERROR                          = 19,
  C_WORDS                                             = 20,
  ALCT_DAV_ERROR                                      = 21,
  TMB_DAV_ERROR                                       = 22,
  CFEB_DAV_ERROR                                      = 23,
  DMB_ACTIVE_ERROR                                    = 24,
  DCC_TRAILER_MISSING                                 = 25,
  DCC_HEADER_MISSING                                  = 26,
  DMB_DAV_VS_DMB_ACTIVE_MISMATCH_ERROR                = 27,
  EXTRA_WORDS_BETWEEN_DDU_HEADER_AND_FIRST_DMB_HEADER = 28
};

/** CSC Payload individual named flags */
enum CSCPayloadFlag {
  CFEB1_ACTIVE = 0,
  CFEB2_ACTIVE = 1,
  CFEB3_ACTIVE = 2,
  CFEB4_ACTIVE = 3,
  CFEB5_ACTIVE = 4,
  ALCT_DAV     = 5,
  TMB_DAV      = 6,
  CFEB1_DAV    = 7,
  CFEB2_DAV    = 8,
  CFEB3_DAV    = 9,
  CFEB4_DAV    = 10,
  CFEB5_DAV    = 11
};

/** CSC Status individual named flags */
enum CSCStatusFlag {
  ALCT_FIFO_FULL           = 0,
  TMB_FIFO_FULL            = 1,
  CFEB1_FIFO_FULL          = 2,
  CFEB2_FIFO_FULL          = 3,
  CFEB3_FIFO_FULL          = 4,
  CFEB4_FIFO_FULL          = 5,
  CFEB5_FIFO_FULL          = 6,
  ALCT_START_TIMEOUT       = 7,
  TMB_START_TIMEOUT        = 8,
  CFEB1_START_TIMEOUT      = 9,
  CFEB2_START_TIMEOUT      = 10,
  CFEB3_START_TIMEOUT      = 11,
  CFEB4_START_TIMEOUT      = 12,
  CFEB5_START_TIMEOUT      = 13,
  ALCT_END_TIMEOUT         = 14,
  TMB_END_TIMEOUT          = 15,
  CFEB1_END_TIMEOUT        = 16,
  CFEB2_END_TIMEOUT        = 17,
  CFEB3_END_TIMEOUT        = 18,
  CFEB4_END_TIMEOUT        = 19,
  CFEB5_END_TIMEOUT        = 20,
  CFEB_ACTIVE_DAV_MISMATCH = 21,
  B_WORDS_FOUND            = 22
};
    
/**
 * @brief  Map iterator template.
 * @param  it Iterator from 0 to ... (auto inc).
 * @param  key Next key to return.
 * @param  m Map to iterate.
 * @return true if key found, false - otherwise
 */
template <class TKey, class TVal>
bool nextInMap(uint32_t& it, TKey& key, const std::map<TKey, TVal>& m) { 
  uint32_t c = 0;
  typename std::map<TKey, TVal>::const_iterator itr = m.begin();
  while (itr != m.end()) {
    if (c == it) {
      it++;
      key = itr->first;
      return true;
    }
    itr++;
    c++;
  }
  return false;
}

/**
 * @brief  List of Map keys template
 * @param  m Map to iterate.
 * @return std::set ok Keys
 */
template <class TKey, class TVal>
std::set<TKey> getKeysList(const std::map<TKey, TVal>& m)
{
  std::set<TKey> keys;
  typename std::map<TKey, TVal>::const_iterator itr;
  for (itr  = m.begin(); itr != m.end(); ++itr) {
     keys.insert(itr->first);
  }
  return keys;
}

/**
 * @class CSCDCCFormatStatusDigi
 * @brief CSC Format Status Object
 */
class CSCDCCFormatStatusDigi {

 private:

  /**
   * Internal mask storage variables and containers.
   */

    
  /** DCC Examiner mask used */
  ExaminerMaskType fDCC_MASK;

  /** CSC Examiner mask used */
  ExaminerMaskType fCSC_MASK;

  /** FED/DCC Id */
  DCCIdType DCCId;

  /** DCC Level summary errors */
  ExaminerStatusType fDDU_SUMMARY_ERRORS;

  std::map<DDUIdType, ExaminerStatusType> mDDU_ERRORS;
  std::map<CSCIdType, ExaminerStatusType> mCSC_ERRORS;
  std::map<CSCIdType, ExaminerStatusType> mCSC_PAYLOADS;
  std::map<CSCIdType, ExaminerStatusType> mCSC_STATUS;

 protected:

  /// Make CSCIdType from Crate and DMB IDs
  CSCIdType makeCSCId(const uint16_t crateId, const uint16_t dmbId) const
    { return ( (CSCIdType(crateId&0xFF)<<4) | (dmbId&0xF) ); }

  /// Init internal data stuctures    
  void init() {
    fDDU_SUMMARY_ERRORS = 0;
    fCSC_MASK = 0;
    fDCC_MASK = 0;
    mDDU_ERRORS.clear();
    mCSC_ERRORS.clear();
    mCSC_PAYLOADS.clear();
    mCSC_STATUS.clear();
  }

 public:

  /**
   * @brief  Constructor
   * @param  fDCC_MASK_ DCC Examiner mask used (for information purposes).
   * @param  fCSC_MASK_ Examiner mask per chamber
   * @param  fDDU_SUMMARY_ERRORS_ Cumulative DDUs errors status
   * @param  mDDU_ERRORS_ List of errors per DDU
   * @param  mCSC_ERRORS_ List of errors per CSC
   * @param  mCSC_PAYLOADS_ List of payloads per CSC
   * @param  mCSC_STATUS_ List of statuses per CSC
   */
  CSCDCCFormatStatusDigi(const DCCIdType DCCId_, 
			 const ExaminerMaskType fDCC_MASK_,
			 const ExaminerMaskType fCSC_MASK_,
			 const ExaminerStatusType fDDU_SUMMARY_ERRORS_,
			 const std::map<DDUIdType, ExaminerStatusType>& mDDU_ERRORS_,
			 const std::map<CSCIdType, ExaminerStatusType>& mCSC_ERRORS_,
			 const std::map<CSCIdType, ExaminerStatusType>& mCSC_PAYLOADS_,
			 const std::map<CSCIdType, ExaminerStatusType>& mCSC_STATUS_): DCCId(DCCId_) 
    {
      init();
      setDCCExaminerInfo(fDCC_MASK_, fCSC_MASK_, fDDU_SUMMARY_ERRORS_, mDDU_ERRORS_, mCSC_ERRORS_, mCSC_PAYLOADS_, mCSC_STATUS_);
    }


  CSCDCCFormatStatusDigi(const DCCIdType DCCId_): DCCId(DCCId_) {init();}


  /// Default constructor.
  CSCDCCFormatStatusDigi (): DCCId(0) {init();}


  /// Fill internal data structures using Examiner object 
  void setDCCExaminerInfo(const ExaminerMaskType fDCC_MASK_,
                         const ExaminerMaskType fCSC_MASK_,
                         const ExaminerStatusType fDDU_SUMMARY_ERRORS_,
                         const std::map<DDUIdType, ExaminerStatusType>& mDDU_ERRORS_,
                         const std::map<CSCIdType, ExaminerStatusType>& mCSC_ERRORS_,
                         const std::map<CSCIdType, ExaminerStatusType>& mCSC_PAYLOADS_, 
                         const std::map<CSCIdType, ExaminerStatusType>& mCSC_STATUS_);

#ifdef DEBUG
  /**
   * Manipulate internal data structures for debug purposes
   */
  void setDCCId(DCCIdType id) { DCCId = id; }
  void setDCCMask(ExaminerMaskType mask) { fDCC_MASK = mask; }
  void setCSCMask(ExaminerMaskType mask) { fCSC_MASK = mask; }
  void setDDUSummaryErrors(ExaminerStatusType status) { fDDU_SUMMARY_ERRORS = status; }
  void setDDUErrors(DDUIdType DDUId, ExaminerStatusType status )  {
    std::map<DDUIdType,ExaminerStatusType>::const_iterator item = mDDU_ERRORS.find(DDUId);
    if( item != mDDU_ERRORS.end() ) mDDU_ERRORS[DDUId] = status; else mDDU_ERRORS.insert(std::make_pair(DDUId, status));
  }
  void setCSCErrors(CSCIdType CSCId, ExaminerStatusType status )  {
    std::map<CSCIdType,ExaminerStatusType>::const_iterator item = mCSC_ERRORS.find(CSCId);
    if( item != mCSC_ERRORS.end() ) mCSC_ERRORS[CSCId] = status; else mCSC_ERRORS.insert(std::make_pair(CSCId, status));
  }
  void setCSCPayload(CSCIdType CSCId, ExaminerStatusType status )  {
    std::map<CSCIdType,ExaminerStatusType>::const_iterator item = mCSC_PAYLOADS.find(CSCId);
    if( item != mCSC_PAYLOADS.end() ) mCSC_PAYLOADS[CSCId] = status; else mCSC_PAYLOADS.insert(std::make_pair(CSCId, status));
  }
  void setCSCStatus(CSCIdType CSCId, ExaminerStatusType status )  {
    std::map<CSCIdType,ExaminerStatusType>::const_iterator item = mCSC_STATUS.find(CSCId);
    if( item != mCSC_STATUS.end() ) mCSC_STATUS[CSCId] = status; else mCSC_STATUS.insert(std::make_pair(CSCId, status));
  }

#endif

  /**
   * Get lists of DDUs and CSCs 
   * Loop iterators for CSCs
   */

  std::set<DDUIdType> getListOfDDUs() const {
    return getKeysList(mDDU_ERRORS);
  }

  std::set<CSCIdType> getListOfCSCs() const {
    return getKeysList(mCSC_PAYLOADS);
  }
    
  std::set<CSCIdType> getListOfCSCsWithErrors() const {
    return getKeysList(mCSC_ERRORS);
  }

  /**
   * @brief  CSC with error iteration procedure.  
   * Usage:
   *   unsigned int i = 0;
   *   CSCIdType cscId;
   *   while (c.nextCSCWithError(i, cscId)) {
   *     // do stuff
   *   }
   * @param  iterator Integer iterator (incremented automatically)
   * @param  CSCId CSC id to return
   * @return true if CSC id found and returned, false - otherwise
   */
 
  bool nextCSCWithError(uint32_t& iterator, CSCIdType& CSCId) const {
    return nextInMap(iterator, CSCId, mCSC_ERRORS);
  }

  /**
   * @brief  CSC with status iteration procedure.  
   * @see    bool nextCSCWithError(uint32_t&, CSCIdType&) const
   * @param  iterator Integer iterator (incremented automatically)
   * @param  CSCId CSC id to return
   * @return true if CSC id found and returned, false - otherwise
   */
  bool nextCSCWithStatus(uint32_t& iterator, CSCIdType& CSCId) const {
    return nextInMap(iterator, CSCId, mCSC_STATUS);
  }

  /**
   * @brief  CSC with payload iteration procedure.  
   * @see    bool nextCSCWithError(uint32_t&, CSCIdType&) const
   * @param  iterator Integer iterator (incremented automatically)
   * @param  CSCId CSC id to return
   * @return true if CSC id found and returned, false - otherwise
   */
  bool nextCSCWithPayload(uint32_t& iterator, CSCIdType& CSCId) const {
    return nextInMap(iterator, CSCId, mCSC_PAYLOADS);
  }

  /**
   * Getters for complete mask by using internal identifiers.
   * Mostly to be used by examiner and old/current code.
   */

    
  /**
   * Return DCC/DDU level Error Status
   */

  ExaminerStatusType getDDUSummaryErrors() const { return fDDU_SUMMARY_ERRORS; }

  ExaminerStatusType getDDUErrors(const DDUIdType DDUId) const {
    std::map<DDUIdType,ExaminerStatusType>::const_iterator item = mDDU_ERRORS.find(DDUId);
    if( item != mDDU_ERRORS.end() ) return item->second; else return 0;
  }


  ExaminerStatusType getCSCErrors(const CSCIdType CSCId) const {
    std::map<CSCIdType,ExaminerStatusType>::const_iterator item = mCSC_ERRORS.find(CSCId);
    if( item != mCSC_ERRORS.end() ) return item->second; else return 0;
  } 

  ExaminerStatusType getCSCErrors(const uint16_t crateId, const uint16_t dmbId) const
    { return getCSCErrors( makeCSCId(crateId, dmbId) ); }


  ExaminerStatusType getCSCPayload(const CSCIdType CSCId) const {
    std::map<CSCIdType,ExaminerStatusType>::const_iterator item = mCSC_PAYLOADS.find(CSCId);
    if( item != mCSC_PAYLOADS.end() ) return item->second; else return 0;
  }

  ExaminerStatusType getCSCPayload(const uint16_t crateId, const uint16_t dmbId) const
    { return getCSCPayload( makeCSCId(crateId, dmbId) ); }


  ExaminerStatusType getCSCStatus(const CSCIdType CSCId) const {
    std::map<CSCIdType,ExaminerStatusType>::const_iterator item = mCSC_STATUS.find(CSCId);
    if( item != mCSC_STATUS.end() ) return item->second; else return 0;
  }

  ExaminerStatusType getCSCStatus(const uint16_t crateId, const uint16_t dmbId) const
    { return getCSCStatus( makeCSCId(crateId, dmbId) ); }


  /* 
   * Return FED/DCC Id
   */
  DCCIdType getDCCId() const { return DCCId; }

  /**
   * Return DCC/DDU level Errors Mask 
   */
  ExaminerMaskType getDCCMask() const { return fDCC_MASK; }

  /**
   * Return CSC level Errors Mask 
   */
  ExaminerMaskType getCSCMask() const { return fCSC_MASK; }
    
  /**
   * Flag Getters for individual named masks.
   */

  bool getDDUSummaryFlag(const FormatErrorFlag flag) const 
    { return ( (fDDU_SUMMARY_ERRORS & ExaminerStatusType(0x1<<flag) ) != 0); }
  bool getDDUErrorFlag(const DDUIdType DDUId, const FormatErrorFlag flag) const
    { return ( (getDDUErrors(DDUId) & ExaminerStatusType(0x1<<flag) ) != 0); }

  bool getCSCErrorFlag(const CSCIdType CSCId, const FormatErrorFlag flag) const
    { return ( (getCSCErrors(CSCId) & ExaminerStatusType(0x1<<flag) ) != 0); }

  bool getCSCErrorFlag(const uint16_t crateId, const uint16_t dmbId, const FormatErrorFlag flag) const
    { return ( (getCSCErrors(crateId, dmbId) & ExaminerStatusType(0x1<<flag) ) != 0); }

  bool getCSCPayloadFlag(const CSCIdType CSCId, const CSCPayloadFlag flag) const
    { return ( (getCSCPayload(CSCId) & ExaminerStatusType(0x1<<flag) ) != 0); }

  bool getCSCPayloadFlag(const uint16_t crateId, const uint16_t dmbId, const CSCPayloadFlag flag) const
    { return ( (getCSCPayload(crateId,dmbId) & ExaminerStatusType(0x1<<flag) ) != 0); }

  bool getCSCStatusFlag(const CSCIdType CSCId, const CSCStatusFlag flag) const
    { return ( (getCSCStatus(CSCId) & ExaminerStatusType(0x1<<flag) ) != 0); }

  bool getCSCStatusFlag(const uint16_t crateId, const uint16_t dmbId, const CSCStatusFlag flag) const
    { return ( (getCSCStatus(crateId, dmbId) & ExaminerStatusType(0x1<<flag) ) != 0); }

  void print() const;
 

};


std::ostream & operator<<(std::ostream & o, const CSCDCCFormatStatusDigi& digi);

#endif
