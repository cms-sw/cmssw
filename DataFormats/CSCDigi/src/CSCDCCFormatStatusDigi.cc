/** \file
 * 
 *  $Date: 2007/07/23 12:08:20 $
 *  $Revision: 1.2 $
 *
 * \author N.Terentiev, CMU
 */
#include <DataFormats/CSCDigi/interface/CSCDCCFormatStatusDigi.h>

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
