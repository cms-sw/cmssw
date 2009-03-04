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
     * @brief  Constructor
     * @param  Examiner_  CSCDCCExaminer object
     * @param  fDCC_MASK_ DCC Examiner mask used (for information purposes).
     */

void CSCDCCFormatStatusDigi::setDCCExaminerInfo(
	const CSCDCCExaminer* Examiner, 
	const ExaminerMaskType fDCC_MASK_
) {

   fDCC_MASK = fDCC_MASK_;
   fDDU_SUMMARY_ERRORS = 0;
   mDDU_ERRORS.clear();
   mCSC_ERRORS.clear();
   mCSC_PAYLOADS.clear();
   mCSC_STATUS.clear();

   if (Examiner != NULL) {

     fCSC_MASK = Examiner->getMask(); // Get CSC Unpacking Mask
     fDDU_SUMMARY_ERRORS = Examiner->errors(); // Summary Errors per DCC/DDU
     mDDU_ERRORS = Examiner->errorsDetailedDDU();
     mCSC_ERRORS = Examiner->errorsDetailed();
     mCSC_PAYLOADS = Examiner->payloadDetailed();
     mCSC_STATUS = Examiner->statusDetailed();

   }

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

