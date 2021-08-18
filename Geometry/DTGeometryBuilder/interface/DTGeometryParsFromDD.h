#ifndef DTGeometryBuilder_DTGeometryParsFromDD_h
#define DTGeometryBuilder_DTGeometryParsFromDD_h
/** \class DTGeometryParsFromDD
 *
 *  Build the RPCGeometry from the DDD and DD4Hep description
 *  
 *  DD4hep part added to the original old file (DD version) made by Stefano Lacaprara (INFN LNL)
 *  \author:  Sergio Lo Meo (sergio.lo.meo@cern.ch) 
 *  Created:  Tue, 26 Jan 2021 
 *
 */
#include "DataFormats/GeometrySurface/interface/BoundPlane.h"
#include <vector>

class DTGeometry;
class DDCompactView;
class DDFilteredView;
namespace cms {  // DD4Hep
  class DDFilteredView;
  class DDCompactView;
}  // namespace cms
class DTChamber;
class DTSuperLayer;
class DTLayer;
class Bounds;
class MuonGeometryConstants;
class RecoIdealGeometry;

class DTGeometryParsFromDD {
public:
  /// Constructor
  DTGeometryParsFromDD();

  /// Destructor
  virtual ~DTGeometryParsFromDD();

  // DD
  void build(const DDCompactView* cview, const MuonGeometryConstants& muonConstants, RecoIdealGeometry& rig);

  // DD4Hep
  void build(const cms::DDCompactView* cview, const MuonGeometryConstants& muonConstants, RecoIdealGeometry& rgeo);

  enum DTDetTag { DTChamberTag, DTSuperLayerTag, DTLayerTag };

private:
  // DD
  /// create the chamber
  void insertChamber(DDFilteredView& fv,
                     const std::string& type,
                     const MuonGeometryConstants& muonConstants,
                     RecoIdealGeometry& rig) const;

  /// create the SL
  void insertSuperLayer(DDFilteredView& fv,
                        const std::string& type,
                        const MuonGeometryConstants& muonConstants,
                        RecoIdealGeometry& rig) const;

  /// create the layer
  void insertLayer(DDFilteredView& fv,
                   const std::string& type,
                   const MuonGeometryConstants& muonConstants,
                   RecoIdealGeometry& rig) const;

  /// get parameter also for boolean solid.
  std::vector<double> extractParameters(DDFilteredView& fv) const;

  typedef std::pair<std::vector<double>, std::vector<double> > PosRotPair;

  PosRotPair plane(const DDFilteredView& fv) const;

  void buildGeometry(DDFilteredView& fv, const MuonGeometryConstants& muonConstants, RecoIdealGeometry& rig) const;

  // DD4Hep

  void buildGeometry(cms::DDFilteredView& fv, const MuonGeometryConstants& muonConstants, RecoIdealGeometry& rig) const;

  /// create the chamber
  void insertChamber(cms::DDFilteredView& fv, const MuonGeometryConstants& muonConstants, RecoIdealGeometry& rig) const;

  /// create the SL
  void insertSuperLayer(cms::DDFilteredView& fv,
                        const MuonGeometryConstants& muonConstants,
                        RecoIdealGeometry& rig) const;

  /// create the layer
  void insertLayer(cms::DDFilteredView& fv, const MuonGeometryConstants& muonConstants, RecoIdealGeometry& rig) const;

  PosRotPair plane(const cms::DDFilteredView& fv) const;
};
#endif
