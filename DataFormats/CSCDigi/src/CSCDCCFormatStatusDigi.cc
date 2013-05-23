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

#include "DataFormats/CSCDigi/interface/CSCDCCFormatStatusDigi.h"

#include <iostream>
#include <iomanip>

 /**
   * @brief  setExaminerInfo 
   * @param  fDCC_MASK_ DCC Examiner mask used (for information purposes).
   * @param  fCSC_MASK_ Examiner mask per chamber
   * @param  fDDU_SUMMARY_ERRORS_ Cumulative DDUs errors status
   * @param  mDDU_ERRORS_ List of errors per DDU
   * @param  mCSC_ERRORS_ List of errors per CSC
   * @param  mCSC_PAYLOADS_ List of payloads per CSC
   * @param  mCSC_STATUS_ List of statuses per CSC
   */
void CSCDCCFormatStatusDigi::setDCCExaminerInfo(const ExaminerMaskType fDCC_MASK_,
                         const ExaminerMaskType fCSC_MASK_,
                         const ExaminerStatusType fDDU_SUMMARY_ERRORS_,
                         const std::map<DDUIdType, ExaminerStatusType>& mDDU_ERRORS_,
                         const std::map<CSCIdType, ExaminerStatusType>& mCSC_ERRORS_,
                         const std::map<CSCIdType, ExaminerStatusType>& mCSC_PAYLOADS_,
                         const std::map<CSCIdType, ExaminerStatusType>& mCSC_STATUS_)
{
  fDCC_MASK = fDCC_MASK_;
  fCSC_MASK = fCSC_MASK_;
  fDDU_SUMMARY_ERRORS = fDDU_SUMMARY_ERRORS_;
  mDDU_ERRORS = mDDU_ERRORS_;
  mCSC_ERRORS = mCSC_ERRORS_;
  mCSC_PAYLOADS = mCSC_PAYLOADS_;
  mCSC_STATUS = mCSC_STATUS_;
}


            /// Debug
void CSCDCCFormatStatusDigi::print() const {

   std::cout << "CSCDCCFormatStatusDigi: DCC=" << std::dec << getDCCId() 
	<< " DCCMask=0x" << std::hex << std::setw(8) << std::setfill('0') << getDCCMask()
	<< " CSCMask=0x" << std::hex << std::setw(8) << std::setfill('0') << getCSCMask()
	<< " DCCErrors=0x" << std::hex << std::setw(8) << std::setfill('0') << getDDUSummaryErrors() 
	<< std::dec << "\n";
   std::set<DDUIdType> ddu_list = getListOfDDUs();
   for (std::set<DDUIdType>::iterator itr=ddu_list.begin(); itr != ddu_list.end(); ++itr) {
   	std::cout << "DDU_" << std::dec << ((*itr)&0xFF) 
	<< " Errors=0x" << std::hex << std::setw(8) << std::setfill('0') << getDDUErrors(*itr) << "\n"; 
   }
   std::set<CSCIdType> csc_list = getListOfCSCs();
   for (std::set<CSCIdType>::iterator itr=csc_list.begin(); itr != csc_list.end(); ++itr) {
	
        std::cout << "CSC_" << std::dec << (((*itr)>>4)&0xFF) << "_" << ((*itr)&0xF) 
		<< " Errors=0x" << std::hex << std::setw(8) << std::setfill('0') << getCSCErrors(*itr) 
		<< " Payload=0x" << std::setw(8) << std::setfill('0') << getCSCPayload(*itr) 
		<< " Status=0x" << std::setw(8) << std::setfill('0') << getCSCStatus(*itr) << "\n";
   }

}

std::ostream & operator<<(std::ostream & o, const CSCDCCFormatStatusDigi& digi) {
   o << "CSCDCCFormatStatusDigi: DCC=" << std::dec << digi.getDCCId() 
        << " DCCMask=0x" << std::hex << std::setw(8) << std::setfill('0') << digi.getDCCMask()
        << " CSCMask=0x" << std::hex << std::setw(8) << std::setfill('0') << digi.getCSCMask()
        << " DCCErrors=0x" << std::hex << std::setw(8) << std::setfill('0') << digi.getDDUSummaryErrors() 
        << std::dec << "\n";
   std::set<DDUIdType> ddu_list = digi.getListOfDDUs();
   for (std::set<DDUIdType>::iterator itr=ddu_list.begin(); itr != ddu_list.end(); ++itr) {
        o << "DDU_" << std::dec << ((*itr)&0xFF) 
        << " Errors=0x" << std::hex << std::setw(8) << std::setfill('0') << digi.getDDUErrors(*itr) << "\n"; 
   }
   std::set<CSCIdType> csc_list = digi.getListOfCSCs();
   for (std::set<CSCIdType>::iterator itr=csc_list.begin(); itr != csc_list.end(); ++itr) {
        
        o << "CSC_" << std::dec << (((*itr)>>4)&0xFF) << "_" << ((*itr)&0xF) 
                << " Errors=0x" << std::hex << std::setw(8) << std::setfill('0') << digi.getCSCErrors(*itr) 
                << " Payload=0x" << std::setw(8) << std::setfill('0') << digi. getCSCPayload(*itr) 
                << " Status=0x" << std::setw(8) << std::setfill('0') << digi.getCSCStatus(*itr) << "\n";
   }
  return o;
}

