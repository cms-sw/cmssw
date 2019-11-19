#ifndef HGCalCommonData_HGCalGeomParameters_h
#define HGCalCommonData_HGCalGeomParameters_h

/** \class HGCalGeomParameters
 *
 * this class extracts some geometry constants from CompactView
 * to be used by Reco Geometry/Topology
 *
 *  $Date: 2015/06/25 00:06:50 $
 * \author Sunanda Banerjee, Fermilab <sunanda.banerjee@cern.ch>
 * \author Lindsey Gray, Fermilab <lagray@fnal.gov> (for fixes)
 *
 */

#include <string>
#include <utility>
#include <vector>

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DetectorDescription/Core/interface/DDsvalues.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/DDCMS/interface/DDCompactView.h"
#include "DetectorDescription/DDCMS/interface/DDFilteredView.h"
#include "Geometry/HGCalCommonData/interface/HGCalGeometryMode.h"
#include "Geometry/HGCalCommonData/interface/HGCalParameters.h"
#include "Geometry/HGCalCommonData/interface/HGCalGeomTools.h"

class HGCalGeomParameters {
public:
  HGCalGeomParameters();
  ~HGCalGeomParameters();
  void loadGeometryHexagon(const DDFilteredView&,
                           HGCalParameters&,
                           const std::string&,
                           const DDCompactView*,
                           const std::string&,
                           const std::string&,
                           HGCalGeometryMode::WaferMode);
  void loadGeometryHexagon(const cms::DDFilteredView&,
                           HGCalParameters&,
                           const std::string&,
                           const cms::DDCompactView*,
                           const std::string&,
                           const std::string&,
                           HGCalGeometryMode::WaferMode);
  void loadGeometryHexagon8(const DDFilteredView&, HGCalParameters&, int);
  void loadGeometryHexagon8(const cms::DDFilteredView&, HGCalParameters&, int);
  void loadSpecParsHexagon(
      const DDFilteredView&, HGCalParameters&, const DDCompactView*, const std::string&, const std::string&);
  void loadSpecParsHexagon(const cms::DDFilteredView&,
                           HGCalParameters&,
                           const cms::DDCompactView*,
                           const std::string&,
                           const std::string&,
                           const std::string&,
                           const std::string&);
  void loadSpecParsHexagon8(const DDFilteredView&, HGCalParameters&);
  void loadSpecParsHexagon8(const cms::DDFilteredView&, const cms::DDVectorsMap&, HGCalParameters&, const std::string&);
  void loadSpecParsTrapezoid(const DDFilteredView&, HGCalParameters&);
  void loadSpecParsTrapezoid(const cms::DDFilteredView&, const cms::DDVectorsMap&, HGCalParameters&, const std::string&);
  void loadWaferHexagon(HGCalParameters& php);
  void loadWaferHexagon8(HGCalParameters& php);
  void loadCellParsHexagon(const DDCompactView* cpv, HGCalParameters& php);
  void loadCellParsHexagon(const cms::DDVectorsMap&, HGCalParameters& php);
  void loadCellParsHexagon(const HGCalParameters& php);
  void loadCellTrapezoid(HGCalParameters& php);

  struct layerParameters {
    double rmin, rmax, zpos;
    layerParameters(double rin = 0, double rout = 0, double zp = 0) : rmin(rin), rmax(rout), zpos(zp) {}
  };
  struct cellParameters {
    bool half;
    int wafer;
    GlobalPoint xyz;
    cellParameters(bool h = false, int w = 0, GlobalPoint p = GlobalPoint(0, 0, 0))
        : half(h), wafer(w), xyz(std::move(p)) {}
  };

private:
  void loadGeometryHexagon(const std::map<int, HGCalGeomParameters::layerParameters>& layers,
                           std::vector<HGCalParameters::hgtrform>& trforms,
                           std::vector<bool>& trformUse,
                           const std::unordered_map<int32_t, int32_t>& copies,
                           const HGCalParameters::layer_map& copiesInLayers,
                           const std::vector<int32_t>& wafer2copy,
                           const std::vector<HGCalGeomParameters::cellParameters>& wafers,
                           const std::map<int, int>& wafertype,
                           const std::map<int, HGCalGeomParameters::cellParameters>& cellsf,
                           const std::map<int, HGCalGeomParameters::cellParameters>& cellsc,
                           HGCalParameters& php);
  void loadGeometryHexagon8(const std::map<int, HGCalGeomParameters::layerParameters>& layers,
                            std::map<std::pair<int, int>, HGCalParameters::hgtrform>& trforms,
                            const int& firstLayer,
                            HGCalParameters& php);
  void loadSpecParsHexagon(const HGCalParameters& php);
  void loadSpecParsHexagon8(const HGCalParameters& php);
  void loadSpecParsTrapezoid(const HGCalParameters& php);
  std::vector<double> getDDDArray(const std::string&, const DDsvalues_type&, const int);
  std::pair<double, double> cellPosition(const std::vector<cellParameters>& wafers,
                                         std::vector<cellParameters>::const_iterator& itrf,
                                         int wafer,
                                         double xx,
                                         double yy);
  void rescale(std::vector<double>&, const double s);
  HGCalGeomTools geomTools_;
  const double sqrt3_;
  double waferSize_;
};

#endif
