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

  void loadGeometryHexagon(const DDFilteredView& _fv,
                           HGCalParameters& php,
                           const std::string& sdTag1,
                           const DDCompactView* cpv,
                           const std::string& sdTag2,
                           const std::string& sdTag3,
                           HGCalGeometryMode::WaferMode mode);
  void loadGeometryHexagon(const cms::DDCompactView* cpv,
                           HGCalParameters& php,
                           const std::string& sdTag1,
                           const std::string& sdTag2,
                           const std::string& sdTag3,
                           HGCalGeometryMode::WaferMode mode);
  void loadGeometryHexagon8(const DDFilteredView& _fv, HGCalParameters& php, int firstLayer);
  void loadGeometryHexagon8(const cms::DDCompactView* cpv,
                            HGCalParameters& php,
                            const std::string& sdTag1,
                            int firstLayer);
  void loadGeometryHexagonModule(const DDCompactView* cpv,
                                 HGCalParameters& php,
                                 const std::string& sdTag1,
                                 const std::string& sdTag2,
                                 int firstLayer);
  void loadGeometryHexagonModule(const cms::DDCompactView* cpv,
                                 HGCalParameters& php,
                                 const std::string& sdTag1,
                                 const std::string& sdTag2,
                                 int firstLayer);
  void loadSpecParsHexagon(const DDFilteredView& fv,
                           HGCalParameters& php,
                           const DDCompactView* cpv,
                           const std::string& sdTag1,
                           const std::string& sdTag2);
  void loadSpecParsHexagon(const cms::DDFilteredView& fv,
                           HGCalParameters& php,
                           const std::string& sdTag1,
                           const std::string& sdTag2,
                           const std::string& sdTag3,
                           const std::string& sdTag4);
  void loadSpecParsHexagon8(const DDFilteredView& fv, HGCalParameters& php);
  void loadSpecParsHexagon8(const cms::DDFilteredView& fv,
                            const cms::DDVectorsMap& vmap,
                            HGCalParameters& php,
                            const std::string& sdTag1);
  void loadSpecParsTrapezoid(const DDFilteredView& fv, HGCalParameters& php);
  void loadSpecParsTrapezoid(const cms::DDFilteredView& fv,
                             const cms::DDVectorsMap& vmap,
                             HGCalParameters& php,
                             const std::string& sdTag1);
  void loadWaferHexagon(HGCalParameters& php);
  void loadWaferHexagon8(HGCalParameters& php);
  void loadCellParsHexagon(const DDCompactView* cpv, HGCalParameters& php);
  void loadCellParsHexagon(const cms::DDVectorsMap& vmap, HGCalParameters& php);
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
  void loadSpecParsHexagon8(HGCalParameters& php);
  void loadSpecParsHexagon8(HGCalParameters& php,
                            const std::vector<int>& layerType,
                            const std::vector<int>& waferIndex,
                            const std::vector<int>& waferProperties);
  void loadSpecParsTrapezoid(HGCalParameters& php);
  void loadSpecParsTrapezoid(HGCalParameters& php,
                             const std::vector<int>& tileIndex,
                             const std::vector<int>& tileProperty,
                             const std::vector<int>& tileHEX1,
                             const std::vector<int>& tileHEX2,
                             const std::vector<int>& tileHEX3,
                             const std::vector<int>& tileHEX4,
                             const std::vector<double>& tileRMin,
                             const std::vector<double>& tileRMax,
                             const std::vector<int>& tileRingMin,
                             const std::vector<int>& tileRingMax);
  std::vector<double> getDDDArray(const std::string& str, const DDsvalues_type& sv, const int nmin);
  std::pair<double, double> cellPosition(const std::vector<cellParameters>& wafers,
                                         std::vector<cellParameters>::const_iterator& itrf,
                                         int wafer,
                                         double xx,
                                         double yy);
  void rescale(std::vector<double>&, const double s);
  void resetZero(std::vector<double>&);

  constexpr static double tan30deg_ = 0.5773502693;
  constexpr static int siliconFileEE = 2;
  constexpr static int siliconFileHE = 3;
  constexpr static int scintillatorFile = 4;
  HGCalGeomTools geomTools_;
  const double sqrt3_;
  double waferSize_;
};

#endif
