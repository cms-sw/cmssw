
#ifndef _LASCONSTANTS_H
#define _LASCONSTANTS_H

#include <vector>
#include <iostream>

#include "FWCore/ParameterSet/interface/ParameterSet.h"

class LASConstants {
public:
  LASConstants();
  LASConstants(std::vector<edm::ParameterSet> const&);
  ~LASConstants();

  double GetEndcapBsKink(unsigned int det, unsigned int ring, unsigned int beam) const;
  double GetAlignmentTubeBsKink(unsigned int beam) const;

  double GetTecRadius(unsigned int ring) const;
  double GetAtRadius(void) const;

  double GetTecZPosition(unsigned int det, unsigned int disk) const;
  double GetTibZPosition(unsigned int pos) const;
  double GetTobZPosition(unsigned int pos) const;
  double GetTecBsZPosition(unsigned int det) const;
  double GetAtBsZPosition(void) const;

private:
  void InitContainers(void);
  void FillBsKinks(edm::ParameterSet const&);
  void FillRadii(edm::ParameterSet const&);
  void FillZPositions(edm::ParameterSet const&);

  std::vector<std::vector<std::vector<double> > > endcapBsKinks;  // outer to inner: det, ring, beam
  std::vector<double> alignmentTubeBsKinks;                       // 8 beams

  std::vector<double> tecRadii;
  double atRadius;

  std::vector<double> tecZPositions;
  std::vector<double> tibZPositions;
  std::vector<double> tobZPositions;
  double tecBsZPosition;
  double atZPosition;
};

#endif
