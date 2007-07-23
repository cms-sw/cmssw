/** \file
 * 
 *  $Date: 2006/09/08 15:40:01 $
 *  $Revision: 1.1 $
 *
 * \author N.Terentiev, CMU
 */
#include <DataFormats/CSCDigi/interface/CSCDCCFormatStatusDigi.h>
#include <bitset>

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
