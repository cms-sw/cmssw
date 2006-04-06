/**\class CSCCLCTDigi
 *
 * Digi for CLCT trigger primitives.
 *
 * $Date: 2006/04/06 04:10:22 $
 * $Revision: 1.4 $
 *
 * \author N. Terentiev, CMU
 */

#include <DataFormats/CSCDigi/interface/CSCCLCTDigi.h>

#include <iostream>
#include <iomanip>

using namespace std;

  /// Constructors
  /// For DQM
CSCCLCTDigi::CSCCLCTDigi (int valid, int quality, int patshape, int striptype, int bend,  int strip, int cfeb, int bx) {
  valid_     = valid;
  quality_   = quality;
  patshape_  = patshape;
  striptype_ = striptype; 
  bend_      = bend;
  strip_     = strip;
  cfeb_      = cfeb;
  bx_        = bx; 
  trknmb_    = 0;
}

CSCCLCTDigi::CSCCLCTDigi (int valid, int quality, int patshape, int striptype,int bend,  int strip, int cfeb, int bx, int trknmb) {
  valid_     = valid;
  quality_   = quality;
  patshape_  = patshape;
  striptype_ = striptype;
  bend_      = bend;
  strip_     = strip;
  cfeb_      = cfeb;
  bx_        = bx;
  trknmb_    = trknmb;
}

      /// Default
CSCCLCTDigi::CSCCLCTDigi (){
  valid_     = 0;
  quality_   = 0;
  patshape_  = 0;
  striptype_ = 0;
  bend_      = 0;
  strip_     = 0;
  cfeb_      = 0;
  bx_        = 0;
  trknmb_    = 0;
}

  /// Debug

void CSCCLCTDigi::print() const {

  char stripType = (getStripType() == 0) ? 'D' : 'H';
  char bend      = (getBend()      == 0) ? 'L' : 'R';

  std::cout <<" CSC CLCT Valid: "          <<setw(1)<<isValid()
       <<" CSC CLCT Quality: "        <<setw(1)<<getQuality()
       <<" CSC CLCT Pattern shape "   <<setw(1)<<getPattern()
       <<" CSC CLCT Strip type: "     <<setw(1)<<stripType
       <<" CSC CLCT Bend: "           <<setw(1)<<bend
       <<" CSC CLCT Strip: "          <<setw(2)<<getStrip()
       <<" CSC CLCT Key Strip: "      <<setw(3)<<getKeyStrip()
       <<" CSC CLCT Key CFEB ID: "    <<setw(1)<<getCFEB()
       <<" CSC CLCT BX: "             <<setw(1)<<getBX()
       <<" CSC CLCT track number: "   <<setw(1)<<getTrknmb()<<std::endl;
}
