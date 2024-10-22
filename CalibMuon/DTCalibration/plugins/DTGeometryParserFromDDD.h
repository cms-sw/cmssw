#ifndef DTGeometryParserFromDDD_H
#define DTGeometryParserFromDDD_H

/** \class DTGeometryParserFromDDD
 *  Class which read the geometry from DDD to provide a map between 
 *  layerId and pairs with first wire number, total number of wires.
 *  \author S. Bolognesi - INFN Torino
 */
#include "DetectorDescription/Core/interface/DDFilter.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "Geometry/MuonNumbering/interface/MuonGeometryNumbering.h"
#include "Geometry/MuonNumbering/interface/MuonBaseNumber.h"
#include "Geometry/MuonNumbering/interface/MuonGeometryConstants.h"
#include "Geometry/MuonNumbering/interface/DTNumberingScheme.h"

#include <map>

class DTLayerId;

class DTGeometryParserFromDDD {
public:
  /// Constructor
  DTGeometryParserFromDDD(const DDCompactView* cview,
                          const MuonGeometryConstants& muonConstants,
                          std::map<DTLayerId, std::pair<unsigned int, unsigned int> >& theLayerIdWiresMap);

  /// Destructor
  ~DTGeometryParserFromDDD();

protected:
private:
  //Parse the DDD
  void parseGeometry(DDFilteredView& fv,
                     const MuonGeometryConstants& muonConstants,
                     std::map<DTLayerId, std::pair<unsigned int, unsigned int> >& theLayerIdWiresMap);
  //Fill the map
  void buildLayer(DDFilteredView& fv,
                  const MuonGeometryConstants& muonConstants,
                  std::map<DTLayerId, std::pair<unsigned int, unsigned int> >& theLayerIdWiresMap);
};
#endif
