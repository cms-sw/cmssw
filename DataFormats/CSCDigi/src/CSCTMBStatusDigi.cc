/** \class CSCTMBStatusDigi
 * 
 *  Digi for CSC TMB info available in DDU
 *
 *  $Date:$
 *  $Revision:$
 *
 */
#include <DataFormats/CSCDigi/interface/CSCTMBStatusDigi.h>
#include <iostream>
#include <bitset>
#include <boost/cstdint.hpp>

using namespace std;

            /// Shift and select
int CSCTMBStatusDigi::ShiftSel(int nmb,int nshift,int nsel) const {
    int tmp=nmb;
    tmp=tmp>>nshift;
    return tmp= tmp & nsel;
}

           /// Get the  DAV 3 bits 
           /// TMB_DAV(1) from 3 locations in DMB header words in DDU
           /// TMB_DAV(1)=1 TMB data exists 
int CSCTMBStatusDigi::getDAV() const {
    return ShiftSel(tmbdmbhdtr_,0,7);
}

           /// Get the  HALF 1 bit
           /// TMB_HALF(1) from DMB trailer word in DDU
           /// TMB_HALF(1)=0 TMB FIFO on the DMB is more than half-full
           /// TMB_HALF(1)=1 TMB FIFO on the DMB is less than half-full
int CSCTMBStatusDigi::getHALF() const {
    return ShiftSel(tmbdmbhdtr_,3,1);
}

           /// Get the  EMPTY 1 bit
           /// TMB_EMPTY(1) from DMB trailer word in DDU
           /// TMB_EMPTY(1)=1 TMB FIFO on the DMB is empty
int CSCTMBStatusDigi::getEMPTY() const {
    return ShiftSel(tmbdmbhdtr_,4,1);
}

           /// Get the Start_Timeout 1 bit
           /// TMB_Start_Timeout(1) from DMB trailer word in DDU
           /// TMB_Start_Timeout(1)=? start of TMB data was not detected within the time-out period 
int CSCTMBStatusDigi::getStart_Timeout() const {
    return ShiftSel(tmbdmbhdtr_,5,1);
}

           /// Get the End_Timeout 1 bit
           /// TMB_End_Timeout(1) from DMB trailer word in DDU
           /// TMB_End_Timeout(1)=? end of TMB data was not detected within the time-out period
int CSCTMBStatusDigi::getEnd_Timeout() const {
    return ShiftSel(tmbdmbhdtr_,6,1);
}

           /// Get the FULL 1 bit
           /// TMB_FULL(1)  from DMB trailer word in DDU
           /// TMB_FULL(1)=1 TMB FIFO on the DMB is full
int CSCTMBStatusDigi::getFULL() const {
    return ShiftSel(tmbdmbhdtr_,7,1);
}

           /// Debug
void CSCTMBStatusDigi::print() const {
    cout << "CSC TMB # : " << getCscId() <<" "<<getBoardId()<<"\n";
    cout << getDAV()<<" "<< getHALF()<<" "<< getEMPTY()<<" "
         << getStart_Timeout()<<" "<< getEnd_Timeout()<<" "
         << getFULL()<<"\n";
    cout << getBXNCntL1A()<<" "<< getBXNCntPretr()<<"\n";
    cout << getNmbTbinsPretr()<<"\n";
}
