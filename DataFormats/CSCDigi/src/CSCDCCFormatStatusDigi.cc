/** \file
 * 
 *  $Date: 2008/02/12 17:40:33 $
 *  $Revision: 1.3 $
 *
 * \author N.Terentiev, CMU
 */
#include "DataFormats/CSCDigi/interface/CSCDCCFormatStatusDigi.h"

#include <iostream>

            /// Debug
void CSCDCCFormatStatusDigi::print() const {

 std::cout<<"DCC DDU ExamInfo UnpackInfo Error " <<getDCCNmb()<<" "
     <<getDDUNmb()<<" "<<getCSCDCCExaminerInfo()<<" "
           << getUnpackerInfo()<<" "<<getErInfo()<<"\n";
  for (size_t i = 0; i<getCSCId().size(); ++i ){
    std::cout<<" "<<(getCSCId())[i]; }
    std::cout<<"\n";
}


std::ostream & operator<<(std::ostream & o, const CSCDCCFormatStatusDigi& digi) {
  o << " " <<digi.getDCCNmb()<<" "<<digi.getDDUNmb()<<" "
           <<digi.getCSCDCCExaminerInfo()<<" "
           << digi.getUnpackerInfo()<<" "<<digi.getErInfo()<<"\n";
  for (size_t i = 0; i<digi.getCSCId().size(); ++i ){
    o <<" " <<(digi.getCSCId())[i]; }
  o<<"\n";

  return o;
}

