#ifndef Geometry_MTDNumberingBuilder_CmsMTDPixelPhase2EndcapBuilder_H
# define Geometry_MTDNumberingBuilder_CmsMTDPixelPhase2EndcapBuilder_H

# include "Geometry/MTDNumberingBuilder/plugins/CmsMTDLevelBuilder.h"
# include "FWCore/ParameterSet/interface/types.h"
# include <string>

/**
 * Class which builds the ETL
 */
class CmsMTDEndcapBuilder : public CmsMTDLevelBuilder
{
public:
  CmsMTDEndcapBuilder();
  
private:
  void sortNS( DDFilteredView& , GeometricTimingDet* ) override;
  void buildComponent( DDFilteredView& , GeometricTimingDet*, std::string ) override;
};

#endif
