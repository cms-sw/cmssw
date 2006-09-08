/** \file
 * 
 *  $Date:$
 *  $Revision:$
 *
 * \author N.Terentiev, CMU
 */
#include <DataFormats/CSCDigi/interface/CSCDCCFormatStatusDigi.h>
#include <iostream>
#include <bitset>
#include <vector>
#include <boost/cstdint.hpp>

using namespace std;

            /// Debug
void CSCDCCFormatStatusDigi::print() const {

 std::cout<<"DCC DDU ExamInfo UnpackInfo Error " <<getDCCNmb()<<" "
     <<getDDUNmb()<<" "<<getCSCDCCExaminerInfo()<<" "
           << getUnpackerInfo()<<" "<<getErInfo()<<"\n";
  for (size_t i = 0; i<getCSCId().size(); ++i ){
    std::cout<<" "<<(getCSCId())[i]; }
    std::cout<<"\n";
}
