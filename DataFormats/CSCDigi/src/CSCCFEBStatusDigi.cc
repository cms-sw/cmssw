/** \file
 * 
 *  $Date: 2006/09/06 14:04:49 $
 *  $Revision: 1.1 $
 *
 * \author N.Terentiev, CMU
 */
#include <DataFormats/CSCDigi/interface/CSCCFEBStatusDigi.h>
#include <iostream>
#include <bitset>
#include <vector>
#include <boost/cstdint.hpp>


using namespace std;

            /// Shift and select
int CSCCFEBStatusDigi::ShiftSel(int nmb,int nshift,int nsel) const {
    int tmp=nmb;
    tmp=tmp>>nshift;
    return tmp= tmp & nsel;
}
            /// Get SCA Full Condition
std::vector<int> CSCCFEBStatusDigi::getSCAFullCond() const {
    std::vector<int> vec(4,0);
    vec[0]=ShiftSel(SCAFullCond_,0,15);  // 4-bit FIFO1 word count
    vec[1]=ShiftSel(SCAFullCond_,4,15);  // 4-bit Block Number if Error Code=1
                                         // (CFEB: SCA Capacitors Full)
                                         // 4-bit FIFO3 word count if Error Code=2
                                         // (CFEB: FPGA FIFO full)
    vec[2]=ShiftSel(SCAFullCond_,9,7);   // Error Code
    vec[3]=ShiftSel(SCAFullCond_,12,15); // DDU Code, should be 0xB
    return vec;
}
            /// Get TS_FLAG bit from SCA Controller data  per each time slice
std::vector<int> CSCCFEBStatusDigi::getTS_FLAG() const {
    std::vector<int> vec(SCACWord_.size(),0);
    int nmb;
    for(unsigned int i=0;i<vec.size();i++) {
      nmb=SCACWord_[i];
      vec[i]=ShiftSel(nmb,15,1);
    }
    return vec;
}

            /// Get SCA_FULL bit from SCA Controller data  per each time slice
std::vector<int> CSCCFEBStatusDigi::getSCA_FULL() const {
    std::vector<int> vec(SCACWord_.size(),0);
    int nmb;
    for(unsigned int i=0;i<vec.size();i++) {
      nmb=SCACWord_[i];
      vec[i]=ShiftSel(nmb,14,1);
    }
    return vec;
}

            /// Get LCT_PHASE bit from SCA Controller data  per each time slice
std::vector<int> CSCCFEBStatusDigi::getLCT_PHASE() const {
    std::vector<int> vec(SCACWord_.size(),0);
    int nmb;
    for(unsigned int i=0;i<vec.size();i++) {
      nmb=SCACWord_[i];
      vec[i]=ShiftSel(nmb,13,1);
    }
    return vec;
}

            /// Get L1A_PHASE bit from SCA Controller data  per each time slice
std::vector<int> CSCCFEBStatusDigi::getL1A_PHASE() const {
    std::vector<int> vec(SCACWord_.size(),0);
    int nmb;
    for(unsigned int i=0;i<vec.size();i++) {
      nmb=SCACWord_[i];
      vec[i]=ShiftSel(nmb,12,1);
    }
    return vec;
}
 
            /// Get SCA_BLK 4 bit word from SCA Controller data  per each time slice
std::vector<int> CSCCFEBStatusDigi::getSCA_BLK() const {
    std::vector<int> vec(SCACWord_.size(),0);
    int nmb;
    for(unsigned int i=0;i<vec.size();i++) {
      nmb=SCACWord_[i];
      vec[i]=ShiftSel(nmb,8,15);
    }
    return vec;
}

            /// Get TRIG_TIME 8 bit word from SCA Controller data  per each time  slice
std::vector<int> CSCCFEBStatusDigi::getTRIG_TIME() const {
    std::vector<int> vec(SCACWord_.size(),0);
    int nmb;
    for(unsigned int i=0;i<vec.size();i++) {
      nmb=SCACWord_[i];
      vec[i]=ShiftSel(nmb,0,255);
    }
    return vec;
}

            /// Debug
void CSCCFEBStatusDigi::print() const {
    cout << "CSC CFEB # : " << getCFEBNmb() <<" "<<getL1AOverlap()<<"\n";
    for (size_t i = 0; i<4; ++i ){
        cout <<" " <<(getSCAFullCond())[i]; }
    cout<<"\n";
    for (size_t i = 0; i<getCRC().size(); ++i ){
        cout <<" " <<(getCRC())[i]; }
    cout<<"\n";
    for (size_t i = 0; i<getTS_FLAG().size(); ++i ){
        cout <<" " <<(getTS_FLAG())[i]; }
    cout<<"\n";
    for (size_t i = 0; i<getSCA_FULL().size(); ++i ){
        cout <<" " <<(getSCA_FULL())[i]; }
    cout<<"\n";
    for (size_t i = 0; i<getLCT_PHASE().size(); ++i ){
        cout <<" " <<(getLCT_PHASE())[i]; }
    cout<<"\n";
    for (size_t i = 0; i<getL1A_PHASE().size(); ++i ){
        cout <<" " <<(getL1A_PHASE())[i]; }
    cout<<"\n";
    for (size_t i = 0; i<getSCA_BLK().size(); ++i ){
        cout <<" " <<(getSCA_BLK())[i]; }
    cout<<"\n";
    for (size_t i = 0; i<getTRIG_TIME().size(); ++i ){
        cout <<" " <<(getTRIG_TIME())[i]; }
    cout<<"\n";
}
