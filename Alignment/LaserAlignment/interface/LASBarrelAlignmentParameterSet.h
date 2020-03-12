

#ifndef __LASBARRELALIGNMENTPARAMETERSET_H
#define __LASBARRELALIGNMENTPARAMETERSET_H

#include <vector>
#include <iostream>
#include <iomanip>
#include <utility>

#include <FWCore/Utilities/interface/Exception.h>

///
/// container for storing the alignment parameters
/// calculated by class LASBarrelAlgorithm
///
/// structure:
/// * for each of the 6 subdetectors (TEC+-, TIB+-, TOB+-) there's a vector
/// * each of these contains two vector<pair<>>, one for each endface in the subdet
/// * each of those vector has three pair<> entries, one for each parameter (rot, deltaX, deltaY)
/// * each entry is a pair containing <parameter,error>
///
class LASBarrelAlignmentParameterSet {
public:
  LASBarrelAlignmentParameterSet();
  std::pair<double, double>& GetParameter(int aSubdetector, int aDisk, int aParameter);
  std::pair<double, double>& GetBeamParameter(int aBeam, int aParameter);
  void Print(void);

private:
  void Init(void);

  std::vector<std::vector<std::pair<double, double> > > tecPlusParameters;
  std::vector<std::vector<std::pair<double, double> > > tecMinusParameters;
  std::vector<std::vector<std::pair<double, double> > > tibPlusParameters;
  std::vector<std::vector<std::pair<double, double> > > tibMinusParameters;
  std::vector<std::vector<std::pair<double, double> > > tobPlusParameters;
  std::vector<std::vector<std::pair<double, double> > > tobMinusParameters;

  std::vector<std::vector<std::pair<double, double> > > beamParameters;
};

#endif
