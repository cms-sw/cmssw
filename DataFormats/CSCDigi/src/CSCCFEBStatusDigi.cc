/** \file
 * 
 *  $Date: 2010/05/13 19:13:59 $
 *  $Revision: 1.8 $
 *
 * \author N.Terentiev, CMU
 */
#include <DataFormats/CSCDigi/interface/CSCCFEBStatusDigi.h>

#include <iostream>
#include <stdint.h>

            /// Shift and select
int CSCCFEBStatusDigi::ShiftSel(int nmb,int nshift,int nsel) const {
    int tmp=nmb;
    tmp=tmp>>nshift;
    return tmp= tmp & nsel;
}
            /// Get SCA Full Condition
std::vector<uint16_t> CSCCFEBStatusDigi::getSCAFullCond() const {
  /*    std::vector<int> vec(4,0);
    vec[0]=ShiftSel(SCAFullCond_,0,15);  // 4-bit FIFO1 word count
    vec[1]=ShiftSel(SCAFullCond_,4,15);  // 4-bit Block Number if Error Code=1
                                         // (CFEB: SCA Capacitors Full)
                                         // 4-bit FIFO3 word count if Error Code=2
                                         // (CFEB: FPGA FIFO full)
    vec[2]=ShiftSel(SCAFullCond_,9,7);   // Error Code
    vec[3]=ShiftSel(SCAFullCond_,12,15); // DDU Code, should be 0xB
    return vec;*/
  return bWords_;
}
            /// Get TS_FLAG bit from SCA Controller data  per each time slice
std::vector<int> CSCCFEBStatusDigi::getTS_FLAG() const {
    std::vector<int> vec(contrWords_.size(),0);
    int nmb;
    for(unsigned int i=0;i<vec.size();i++) {
      nmb=contrWords_[i];
      vec[i]=ShiftSel(nmb,15,1);
    }
    return vec;
}

            /// Get SCA_FULL bit from SCA Controller data  per each time slice
std::vector<int> CSCCFEBStatusDigi::getSCA_FULL() const {
    std::vector<int> vec(contrWords_.size(),0);
    int nmb;
    for(unsigned int i=0;i<vec.size();i++) {
      nmb=contrWords_[i];
      vec[i]=ShiftSel(nmb,14,1);
    }
    return vec;
}

            /// Get LCT_PHASE bit from SCA Controller data  per each time slice
std::vector<int> CSCCFEBStatusDigi::getLCT_PHASE() const {
    std::vector<int> vec(contrWords_.size(),0);
    int nmb;
    for(unsigned int i=0;i<vec.size();i++) {
      nmb=contrWords_[i];
      vec[i]=ShiftSel(nmb,13,1);
    }
    return vec;
}

            /// Get L1A_PHASE bit from SCA Controller data  per each time slice
std::vector<int> CSCCFEBStatusDigi::getL1A_PHASE() const {
    std::vector<int> vec(contrWords_.size(),0);
    int nmb;
    for(unsigned int i=0;i<vec.size();i++) {
      nmb=contrWords_[i];
      vec[i]=ShiftSel(nmb,12,1);
    }
    return vec;
}
 
            /// Get SCA_BLK 4 bit word from SCA Controller data  per each time slice
std::vector<int> CSCCFEBStatusDigi::getSCA_BLK() const {
    std::vector<int> vec(contrWords_.size(),0);
    int nmb;
    for(unsigned int i=0;i<vec.size();i++) {
      nmb=contrWords_[i];
      vec[i]=ShiftSel(nmb,8,15);
    }
    return vec;
}

            /// Get TRIG_TIME 8 bit word from SCA Controller data  per each time  slice
std::vector<int> CSCCFEBStatusDigi::getTRIG_TIME() const {
    std::vector<int> vec(contrWords_.size(),0);
    int nmb;
    for(unsigned int i=0;i<vec.size();i++) {
      nmb=contrWords_[i];
      vec[i]=ShiftSel(nmb,0,255);
    }
    return vec;
}

            /// Debug
void CSCCFEBStatusDigi::print() const {
    std::cout << "CSC CFEB # : " << getCFEBNmb() <<"\n";
    std::cout << " SCAFullCond: ";
    if(getSCAFullCond().size()!=0){
    for (size_t i = 0; i<4; ++i ){
        std::cout <<" " <<(getSCAFullCond())[i]; }
	}
    else {
    std::cout << " " <<"BWORD is not valied";
    }	
    std::cout << "\n";
    std::cout << " CRC: ";
    for (size_t i = 0; i<getCRC().size(); ++i ){
        std::cout <<" " <<(getCRC())[i]; }
    std::cout<<"\n";
    std::cout << " TS_FLAG: ";
    for (size_t i = 0; i<getTS_FLAG().size(); ++i ){
        std::cout <<" " <<(getTS_FLAG())[i]; }
    std::cout<<"\n";
    std::cout << " SCA_FULL: ";
    for (size_t i = 0; i<getSCA_FULL().size(); ++i ){
        std::cout <<" " <<(getSCA_FULL())[i]; }
    std::cout<<"\n";
    std::cout << " LCT_PHASE: ";
    for (size_t i = 0; i<getLCT_PHASE().size(); ++i ){
        std::cout <<" " <<(getLCT_PHASE())[i]; }
    std::cout<<"\n";
    std::cout << " L1A_PHASE: ";
    for (size_t i = 0; i<getL1A_PHASE().size(); ++i ){
        std::cout <<" " <<(getL1A_PHASE())[i]; }
    std::cout<<"\n";
    std::cout << " SCA_BLK: ";
    for (size_t i = 0; i<getSCA_BLK().size(); ++i ){
        std::cout <<" " <<(getSCA_BLK())[i]; }
    std::cout<<"\n";
    std::cout << " TRIG_TIME: ";
    for (size_t i = 0; i<getTRIG_TIME().size(); ++i ){
        std::cout <<" " <<(getTRIG_TIME())[i]; }
    std::cout<<"\n";
}

std::ostream & operator<<(std::ostream & o, const CSCCFEBStatusDigi& digi) {
  o << " " << digi.getCFEBNmb()<<"\n";
  for (size_t i = 0; i<4; ++i ){
        o <<" " <<(digi.getSCAFullCond())[i]; }
  o <<"\n";
  for (size_t i = 0; i<digi.getCRC().size(); ++i ){
    o <<" " <<(digi.getCRC())[i]; }
  o<<"\n";
  for (size_t i = 0; i<digi.getTS_FLAG().size(); ++i ){
    o <<" " <<(digi.getTS_FLAG())[i]; }
  o<<"\n";
  for (size_t i = 0; i<digi.getSCA_FULL().size(); ++i ){
    o <<" " <<(digi.getSCA_FULL())[i]; }
  o<<"\n";
  for (size_t i = 0; i<digi.getLCT_PHASE().size(); ++i ){
    o <<" " <<(digi.getLCT_PHASE())[i]; }
  o<<"\n";
  for (size_t i = 0; i<digi.getL1A_PHASE().size(); ++i ){
    o <<" " <<(digi.getL1A_PHASE())[i]; }
  o<<"\n";
  for (size_t i = 0; i<digi.getSCA_BLK().size(); ++i ){
    o <<" " <<(digi.getSCA_BLK())[i]; }
  o<<"\n";
  for (size_t i = 0; i<digi.getTRIG_TIME().size(); ++i ){
    o <<" " <<(digi.getTRIG_TIME())[i]; }
  o<<"\n";

  return o;
}

