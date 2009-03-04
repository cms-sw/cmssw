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
#include "EventFilter/CSCRawToDigi/interface/CSCDCCExaminer.h"
#include <set>
#include <map>
#include <algorithm>
#include <iosfwd>

    
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
   * @param  Examiner  CSCDCCExaminer object
   * @param  fDCC_MASK_ DCC Examiner mask used (for information purposes).
   */
  CSCDCCFormatStatusDigi(const DCCIdType DCCId_, 
			 const CSCDCCExaminer* Examiner, 
			 const ExaminerMaskType fDCC_MASK_): DCCId(DCCId_) 
    {
      init();
      setDCCExaminerInfo(Examiner, fDCC_MASK_);
    }


  CSCDCCFormatStatusDigi(const DCCIdType DCCId_): DCCId(DCCId_) {init();}


  /// Default constructor.
  CSCDCCFormatStatusDigi (): DCCId(0) {init();}


  /// Fill internal data structures using Examiner object 
  void setDCCExaminerInfo(const CSCDCCExaminer* Examiner, const ExaminerMaskType fDCC_MASK_);

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
