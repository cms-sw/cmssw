#ifndef CSCCFEBStatusDigi_CSCCFEBStatusDigi_h
#define CSCCFEBStatusDigi_CSCCFEBStatusDigi_h

/** \class CSCCFEBStatusDigi
 *
 * Digi for CSC CFEB status.
 *  
 *  $Date: 2013/04/22 22:39:23 $
 *  $Revision: 1.8 $
 *
 * \author N. Terentiev, CMU
 *
 */

#include <vector>
#include <iosfwd>
#include <stdint.h>

class CSCCFEBStatusDigi{

public:

            /// Construct from the CFEB number (1-5).
  CSCCFEBStatusDigi (int cfebnmb) {cfebnmb_ = cfebnmb;}


  /// Constructor for all variables
  CSCCFEBStatusDigi (int cfebnmb, const std::vector<uint16_t>& crcWords, 
		     const std::vector<uint16_t>& contrWords,  const std::vector<uint16_t>& bWords) {
    cfebnmb_ = cfebnmb;
    crcWords_ = crcWords;
    contrWords_ = contrWords;
    bWords_ = bWords;
}


            /// Default construction.
  CSCCFEBStatusDigi () {}

            /// Set CRC vector 
  void setCRC (const std::vector<uint16_t>& crc) {crcWords_ = crc;}

            /// Set SCAC (SCA Controller) vector
  void setSCAC (const std::vector<uint16_t>& scac) {contrWords_ =  scac;}

            /// Get the  CFEB number
  int getCFEBNmb() const {return cfebnmb_;}

            /// Get SCA Full Condition 
  std::vector<uint16_t> getSCAFullCond() const;

            /// Get CRC per each time sample
  std::vector<uint16_t> getCRC() const {return crcWords_;}

            /// Shift and select
  int ShiftSel(int nmb,int nshift,int nsel) const;

            /// Get TS_FLAG bit from SCA Controller data  per each time slice
  std::vector<int> getTS_FLAG() const;

            /// Get SCA_FULL bit from SCA Controller data  per each time slice
  std::vector<int> getSCA_FULL() const;

            /// Get LCT_PHASE bit from SCA Controller data  per each time slice
  std::vector<int> getLCT_PHASE() const;

            /// Get L1A_PHASE bit from SCA Controller data  per each time slice
  std::vector<int> getL1A_PHASE() const;

            /// Get SCA_BLK 4 bit word from SCA Controller data  per each time slice
  std::vector<int> getSCA_BLK() const;

            /// Get TRIG_TIME 8 bit word from SCA Controller data  per each time slice
  std::vector<int> getTRIG_TIME() const;

            /// Print content of digi
  void print() const;

private:
  
  uint16_t cfebnmb_;
  std::vector<uint16_t> crcWords_;
  std::vector<uint16_t> contrWords_;
  std::vector<uint16_t> bWords_;

};

std::ostream & operator<<(std::ostream & o, const CSCCFEBStatusDigi& digi);

#endif
