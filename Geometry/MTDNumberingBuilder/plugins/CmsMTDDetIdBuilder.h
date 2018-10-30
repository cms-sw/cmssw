#ifndef Geometry_MTDNumberingBuilder_CmsMTDDetIdBuilder_H
#define Geometry_MTDNumberingBuilder_CmsMTDDetIdBuilder_H

# include "FWCore/ParameterSet/interface/types.h"
# include <ostream>
#include <vector>
#include <array>

class GeometricTimingDet;

/**
 * Class to build a geographicalId.
 */

class CmsMTDDetIdBuilder
{
public:
  CmsMTDDetIdBuilder(std::vector<int> detidShifts );
  GeometricTimingDet* buildId( GeometricTimingDet *det );  
protected:
  void iterate( GeometricTimingDet *det, int level, unsigned int ID );
  
private:

  static const unsigned int nSubDet=6;
  static const int maxLevels=6;

  // This is the map between detid and navtype to restore backward compatibility between 12* and 13* series
  std::map< std::string , uint32_t > m_mapNavTypeToDetId;
  std::array<int,nSubDet*maxLevels> m_detidshifts; 
};

#endif
