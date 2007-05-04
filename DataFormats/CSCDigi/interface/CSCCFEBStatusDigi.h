#ifndef CSCCFEBStatusDigi_CSCCFEBStatusDigi_h
#define CSCCFEBStatusDigi_CSCCFEBStatusDigi_h

/** \class CSCCFEBStatusDigi
 *
 * Digi for CSC CFEB status.
 *  
 *  $Date:$
 *  $Revision:$
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

            /// Default construction.
  CSCCFEBStatusDigi () {}

            /// Set the word with L1A overlap bits
  void setL1AOverlap(int overlap) {L1AOverlap_ = overlap;}
                                
            /// Set the word SCA Full condition
  void setSCAFull(int scacapfull) {SCACapFull_ = scacapfull;}

            /// Set the word FPGA FIFO Full condition
  void setFPGAFIFOFull(int fpgafifafull) { FPGAFIFOFull_ = fpgafifafull;}
                                                
            /// Set CRC vector 
  void setCRC (std::vector<uint16_t> crc) {CRCWord_ = crc;}

            /// Set SCAC (SCA Controller) vector
  void setSCAC (std::vector<uint16_t> scac) {SCACWord_ =  scac;}

            /// Get the  CFEB number
  int getCFEBNmb() const {return cfebnmb_;}

            /// Get the  L1A overlap bits word
  int getL1AOverlap() const {return L1AOverlap_;} 

            /// Get the  SCA capacitors full condition (bits 8:0) 
  int getSCACapFull() const {return SCACapFull_;}

            /// Get the  FPGA FIFO full condition (bits 8:0)
  int getFPGAFIFOFull() const {return FPGAFIFOFull_;}

            /// Get CRC per each time sample
  std::vector<uint16_t> getCRC() const {return CRCWord_;}

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
  friend class testCSCCFEBStatusDigis;
  
  uint16_t cfebnmb_;
  uint16_t L1AOverlap_;
  uint16_t SCACapFull_;
  uint16_t FPGAFIFOFull_;
  std::vector<uint16_t> CRCWord_;
  std::vector<uint16_t> SCACWord_;
};

#include<iostream>
            /// needed by COBRA
inline std::ostream & operator<<(std::ostream & o, const CSCCFEBStatusDigi& digi) {
  o << " " << digi.getCFEBNmb()<<" "<<digi.getL1AOverlap()<<" "
           << digi.getSCACapFull()<<" "<<digi.getFPGAFIFOFull()<<"\n";
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
