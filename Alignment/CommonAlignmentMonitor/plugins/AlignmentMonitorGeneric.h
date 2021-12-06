#ifndef Alignment_CommonAlignmentMonitor_AlignmentMonitorGeneric_H
#define Alignment_CommonAlignmentMonitor_AlignmentMonitorGeneric_H

// Package:     CommonAlignmentMonitor
// Class  :     AlignmentMonitorGeneric
//
// Produce histograms generic to all alignment algorithms.
//
// Histograms defined:
//   pull of x hit residuals for positively charged tracks on each alignable
//   pull of x hit residuals for negatively charged tracks on each alignable
//   pull of y hit residuals for positively charged tracks on each alignable
//   pull of y hit residuals for negatively charged tracks on each alignable
//         pt for all tracks
//        eta for all tracks
//        phi for all tracks
//         d0 for all tracks
//         dz for all tracks
//   chi2/dof for all tracks
//
// Original Author:  Jim Pivarski
//         Created:  Thu Mar 29 13:59:56 CDT 2007
// $Id: AlignmentMonitorGeneric.h,v 1.3 2007/12/04 23:29:26 ratnik Exp $

#include "Alignment/CommonAlignmentMonitor/interface/AlignmentMonitorBase.h"
#include "TH1.h"

class AlignmentMonitorGeneric : public AlignmentMonitorBase {
  typedef std::vector<TH1F*> Hist1Ds;

public:
  AlignmentMonitorGeneric(const edm::ParameterSet&, edm::ConsumesCollector iC);

  void book() override;

  void event(const edm::Event&, const edm::EventSetup&, const ConstTrajTrackPairCollection&) override;

private:
  static const unsigned int nBin_ = 50;

  Hist1Ds m_trkHists;  // track parameters histograms

  std::map<const Alignable*, Hist1Ds> m_resHists;  // hit residuals histograms
};

#endif
