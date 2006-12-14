#ifndef CSCCFEBStatusDigi_CSCCFEBStatusDigi_h
#define CSCCFEBStatusDigi_CSCCFEBStatusDigi_h

/** \class CSCCFEBStatusDigi
 *
 * Digi for CSC CFEB status.
 *  
 *  $Date: 2006/12/05 22:47:01 $
 *  $Revision: 1.3 $
 *
 * \author N. Terentiev, CMU
 *
 */

#include <vector>
#include <boost/cstdint.hpp>

class CSCCFEBStatusDigi{

public:

            /// Construct from the CFEB number (1-5).
  CSCCFEBStatusDigi (int cfebnmb) {cfebnmb_ = cfebnmb;}


  /// Constructor for all variables
  CSCCFEBStatusDigi (int cfebnmb, std::vector<uint16_t> crcWords, 
		     std::vector<uint16_t> contrWords,  std::vector<uint16_t> bWords) {
    cfebnmb_ = cfebnmb;
    crcWords_ = crcWords;
    contrWords_ = contrWords;
    bWords_ = bWords;
}


            /// Default construction.
  CSCCFEBStatusDigi () {}

            /// Set CRC vector 
  void setCRC (std::vector<uint16_t> crc) {crcWords_ = crc;}

            /// Set SCAC (SCA Controller) vector
  void setSCAC (std::vector<uint16_t> scac) {contrWords_ =  scac;}

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

#include<iostream>
            /// needed by COBRA
inline std::ostream & operator<<(std::ostream & o, const CSCCFEBStatusDigi& digi) {
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

#endif
