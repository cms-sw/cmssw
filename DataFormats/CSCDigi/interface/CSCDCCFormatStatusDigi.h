#ifndef CSCDCCFormatStatusDigi_CSCDCCFormatStatusDigi_h
#define CSCDCCFormatStatusDigi_CSCDCCFormatStatusDigi_h

/** \class CSCDCCFormatStatusDigi
 *
 * Digi for CSC DCC/DDU Format status.
 *  
 *  $Date: 2007/07/23 12:08:19 $
 *  $Revision: 1.3 $
 *
 * \author N. Terentiev, CMU
 *
 */

#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include <vector>
#include <iosfwd>

class CSCDCCFormatStatusDigi{

public:
            /// Construct from the DCC and DDU numbers
  CSCDCCFormatStatusDigi (int dccnmb, int ddunmb) {
                          dccnmb_ = dccnmb; ddunmb_=ddunmb;}

            /// Default construction.
  CSCDCCFormatStatusDigi () {}

            /// Set the word with CSCDCCExaminer info
  void setCSCDCCExaminerInfo(unsigned int examinfo) {examinfo_ = examinfo;}
                                
            /// Set the word with unpacker info
  void setUnpackerInfo(unsigned int unpackinfo) {unpackinfo_=unpackinfo;}

            /// Set the word with fatal errors info
  void setErInfo(unsigned int erinfo) {erinfo_ = erinfo;}
                                                
            /// Set CSC Id vector 
  void setCSCId (std::vector<CSCDetId> cscid) {cscid_ = cscid;}


            /// Get the  DCC number
  int getDCCNmb() const {return dccnmb_;}

            /// Get the  DDU number
  int getDDUNmb() const {return ddunmb_;}

            /// Get the word with CSCDCCExaminer info
  unsigned int getCSCDCCExaminerInfo() const {return examinfo_;}

           /// Get the word with unpacker info
  unsigned int getUnpackerInfo() const {return unpackinfo_;}

           /// Get the word with fatal errors info
  unsigned int getErInfo() const {return erinfo_;}

           /// Get CSC Id vector 
  std::vector<CSCDetId> getCSCId () const {return cscid_;}

            /// Print content of digi
  void print() const;

private:
  
  uint16_t dccnmb_;
  uint16_t ddunmb_;
  uint32_t examinfo_;
  uint32_t unpackinfo_;
  uint32_t erinfo_;
  std::vector<CSCDetId> cscid_;
};

std::ostream & operator<<(std::ostream & o, const CSCDCCFormatStatusDigi& digi);

#endif
