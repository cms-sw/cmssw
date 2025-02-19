/*
L2 Tau Trigger Isolation algorithms

Author: Michail Bachtis
University of Wisconsin-Madison
e-mail: bachtis@hep.wisc.edu
*/

#ifndef L2TAUISOLATIONALGS_H
#define L2TAUISOLATIONALGS_H

#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/Math/interface/LorentzVectorFwd.h"
#include <vector>
#include "DataFormats/TauReco/interface/L2TauIsolationInfo.h"
#include "DataFormats/JetReco/interface/CaloJet.h"

class L2TauIsolationAlgs
{
 public:
  L2TauIsolationAlgs();
  ~L2TauIsolationAlgs();

  double isolatedEt(const math::PtEtaPhiELorentzVectorCollection& ,const math::XYZVector&,double innerCone, double outerCone) const;
  int nClustersAnnulus(const math::PtEtaPhiELorentzVectorCollection& ,const math::XYZVector&,double innerCone, double outerCone) const;

  std::vector<double> clusterShape(const math::PtEtaPhiELorentzVectorCollection& ,const math::XYZVector&,double innerCone,double outerCone) const;

};

#endif

