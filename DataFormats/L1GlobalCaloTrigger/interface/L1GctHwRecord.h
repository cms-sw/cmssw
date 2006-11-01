
#ifndef L1GCTHWRECORD_H
#define L1GCTHWRECORD_H

/*! \class L1GctHwRecord
 * \brief Container for non-trigger data in GCT event record
 *
 * Trigger data is stored in the following formats :
 *   L1CaloRegion
 *   L1CaloEmCand
 *   L1GctEmCand
 *   L1GctJetCand
 *   L1GctEtTotal
 *   L1GctEtHad
 *   L1GctEtMiss
 * 
 * This object contains all other data recorded
 * by the GCT.
 *
 */

/*! \author Jim Brooke
 *  \date Nov 2006
 */

#include <vector>


class L1GctHwRecord {

 public:

  L1GctHwRecord();
  ~L1GctHwRecord();

  // add a header
  void addHeader(unsigned h) { m_headers.push_back(h); }

  // get header data
  unsigned header(int i) const { return m_headers.at(i); }
  unsigned headerBlockId(int i) const { return m_headers.at(i) & 0xff ; }
  unsigned headerNBx(int i) const { return (m_headers.at(i) >> 8) & 0xf ; }
  unsigned headerBcId(int i)  const { return (m_headers.at(i) >> 12) & 0xfff ; }
  unsigned headerEvtId(int i) const { return (m_headers.at(i) >> 24) & 0xff ; }

 private:

  std::vector<unsigned> m_headers;

};

#endif
