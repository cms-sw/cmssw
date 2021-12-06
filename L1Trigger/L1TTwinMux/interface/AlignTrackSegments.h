//-------------------------------------------------
//
//   Class: AlignTrackSegments
//
//   AlignTrackSegments
//
//
//   Author :
//   G. Flouris               U Ioannina    Mar. 2015
//   mod.: G Karathanasis
//--------------------------------------------------

#ifndef L1T_TwinMux_AlignTrackSegments_H
#define L1T_TwinMux_AlignTrackSegments_H

#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhContainer.h"

class AlignTrackSegments {
public:
  AlignTrackSegments(L1MuDTChambPhContainer inphiDigis);
  ~AlignTrackSegments(){};

  void run();

  ///Return Output PhContainer
  const L1MuDTChambPhContainer& getDTContainer() { return m_dt_tsshifted; }

private:
  ///Output PhContainer
  L1MuDTChambPhContainer m_dt_tsshifted;
  L1MuDTChambPhContainer m_phiDigis;
};
#endif
