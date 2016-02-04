
#ifndef __LASENDCAPALIGNMENTPARAMETERSET_H
#define __LASENDCAPALIGNMENTPARAMETERSET_H



#include <vector>
#include <utility>
#include <iostream>
#include <iomanip>

#include <FWCore/Utilities/interface/Exception.h>

///
/// container for storing the alignment parameters
/// calculated by class LASEndcapAlgorithm
///
/// structure:
/// * for each of the 2 subdetectors (TEC+, TEC-) there's a vector 
/// * each of these contains nine vector<pair<>>, one for each disk
/// * each of those vector has three pair<> entries, one for each parameter (rot, deltaX, deltaY)
/// * each entry is a pair containing <parameter,error>
///
/// TODO:
///   * implement beam & global parameters
///
class LASEndcapAlignmentParameterSet {

 public:
  LASEndcapAlignmentParameterSet();
  std::pair<double,double>& GetDiskParameter( int aSubdetector, int aDisk, int aParameter );
  std::pair<double,double>& GetGlobalParameter( int aSubdetector, int aParameter );
  std::pair<double,double>& GetBeamParameter( int aSubdetector, int aRing, int aBeam, int aParameter );
  void Print( void );

 private:
  void Init( void );

  std::vector<std::vector<std::pair<double,double> > > tecPlusDiskParameters;
  std::vector<std::vector<std::pair<double,double> > > tecMinusDiskParameters;
  std::vector<std::pair<double,double> > tecPlusGlobalParameters;
  std::vector<std::pair<double,double> > tecMinusGlobalParameters;
  std::vector<std::vector<std::vector<std::pair<double,double> > > > tecPlusBeamParameters;
  std::vector<std::vector<std::vector<std::pair<double,double> > > > tecMinusBeamParameters;

};


#endif
