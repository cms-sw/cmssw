#ifndef Alignment_CommonAlignmentMonitor_AlignmentMonitorGeneric_H
#define Alignment_CommonAlignmentMonitor_AlignmentMonitorGeneric_H

// Package:     CommonAlignmentMonitor
// Class  :     AlignmentMonitorGeneric
// 
// Produce histograms generic to all alignment algorithms.
//
// Histograms defined:
//   hit residuals (x, y, z)
//
// Original Author:  Jim Pivarski
//         Created:  Thu Mar 29 13:59:56 CDT 2007
// $Id: AlignmentMonitorGeneric.cc,v 1.1 2007/05/09 07:06:33 fronga Exp $

#include "Alignment/CommonAlignmentMonitor/interface/AlignmentMonitorBase.h"

class TH3F;

class AlignmentMonitorGeneric:
  public AlignmentMonitorBase
{
  public:

  AlignmentMonitorGeneric(
			  const edm::ParameterSet&
			  );

  virtual void book();

  virtual void event(
		     const edm::EventSetup&,
		     const ConstTrajTrackPairCollection&
		     );

  virtual void afterAlignment(
			      const edm::EventSetup&
			      ) {}

  private:

  std::map<const Alignable*, TH3F*> m_residuals;
};

#endif
