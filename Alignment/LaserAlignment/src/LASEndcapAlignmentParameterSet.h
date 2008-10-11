
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
  std::pair<double,double>& GetParameter( int aSubdetector, int aDisk, int aParameter );
  void Dump( void );

 private:
  void Init( void );

  std::vector<std::vector<std::pair<double,double> > > tecPlusParameters;
  std::vector<std::vector<std::pair<double,double> > > tecMinusParameters;

};


#endif
