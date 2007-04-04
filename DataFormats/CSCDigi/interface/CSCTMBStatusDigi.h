#ifndef CSCTMBStatusDigi_CSCTMBStatusDigi_h
#define CSCTMBStatusDigi_CSCTMBStatusDigi_h

/** \class CSCTMBStatusDigi
 *
 *  Digi for CSC TMB info available in DDU
 *  
 *  $Date:$
 *  $Revision:$
 *
 */

#include <vector>
#include <boost/cstdint.hpp>

class CSCTMBStatusDigi{

public:

            /// Constructor for all variables 
  CSCTMBStatusDigi (int tmbdmbhdtr, int boardid, int cscid,
                    int bxncntL1arv,int bxncntpretrig, 
                    int  nmbtbinpretrig) {

                    tmbdmbhdtr_     = tmbdmbhdtr;
                    boardid_        = boardid;
                    cscid_          = cscid;
		    bxncntL1arv_    =  bxncntL1arv;
                    bxncntpretrig_  =  bxncntpretrig;
                    nmbtbinpretrig_ = nmbtbinpretrig;
  }

            /// Default constructor.
  CSCTMBStatusDigi () {}

            /// Shift and select
  int ShiftSel(int nmb,int nshift,int nsel) const;

            /// Get the  DAV 3 bits
  int getDAV() const;

           /// Get the  HALF 1 bit
  int getHALF() const;

           /// Get the  EMPTY 1 bit
  int getEMPTY() const;

           /// Get the  Start_Timeout 1 bit
  int getStart_Timeout() const;

           /// Get the  End_Timeout 1 bit
  int getEnd_Timeout() const;

          /// Get the   FULL 1 bit
  int getFULL() const;

          /// Get the  Board ID
  int getBoardId() const {return boardid_;}

          /// Get the  CSC ID
  int getCscId() const {return cscid_;}

          /// Get the  BXN Counter at L1A arrival (12 bits)
  int getBXNCntL1A() const {return bxncntL1arv_;}

          /// Get the  BXN Counter at pre-trigger (12 bits)
  int getBXNCntPretr() const {return bxncntpretrig_;}
        
          /// Get the  # Tbins before pre-trigger (5 bits)
  int getNmbTbinsPretr() const {return nmbtbinpretrig_;}           

            /// Print content of digi
  void print() const;

private:

  uint16_t tmbdmbhdtr_;
  uint16_t boardid_;
  uint16_t cscid_;
  uint16_t bxncntL1arv_;
  uint16_t bxncntpretrig_;
  uint16_t nmbtbinpretrig_;
};

#include<iostream>
            /// needed by COBRA
inline std::ostream & operator<<(std::ostream & o, const CSCTMBStatusDigi& digi) {
  o << " " << digi.getDAV()<<" "<< digi.getHALF()<<" "<< digi.getEMPTY()<<" "
           << digi.getStart_Timeout()<<" "<< digi.getEnd_Timeout()<<" "
           << digi.getFULL()<<"\n";
  o<<"\n";
  
  o << " " << digi.getBoardId()<<" "<< digi.getCscId()<<"\n";
  o<<"\n";

  o << " " << digi.getBXNCntL1A()<<" "<< digi.getBXNCntPretr()<<"\n";
  o<<"\n";

  o << " " << digi.getNmbTbinsPretr()<<"\n";
  o<<"\n";

  return o;
}

#endif
