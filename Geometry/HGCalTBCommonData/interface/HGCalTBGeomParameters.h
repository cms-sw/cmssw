#ifndef HGCalCommonData_HGCalTBGeomParameters_h
#define HGCalCommonData_HGCalTBGeomParameters_h

/** \class HGCalTBGeomParameters
 *
 * this class extracts some geometry constants from CompactView
 * to be used by Reco Geometry/Topology
 *
 *  $Date: 2022/12/31 00:06:50 $
 * \author Sunanda Banerjee, Fermilab <sunanda.banerjee@cern.ch>
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
#include "Geometry/HGCalTBCommonData/interface/HGCalTBParameters.h"
#include "Geometry/HGCalCommonData/interface/HGCalGeomTools.h"

class HGCalTBGeomParameters {
public:
  HGCalTBGeomParameters();
  ~HGCalTBGeomParameters() = default;

  void loadGeometryHexagon(const DDFilteredView& _fv,
                           HGCalTBParameters& php,
                           const std::string& sdTag1,
                           const DDCompactView* cpv,
                           const std::string& sdTag2,
                           const std::string& sdTag3,
                           HGCalGeometryMode::WaferMode mode);
  void loadGeometryHexagon(const cms::DDCompactView* cpv,
                           HGCalTBParameters& php,
                           const std::string& sdTag1,
                           const std::string& sdTag2,
                           const std::string& sdTag3,
                           HGCalGeometryMode::WaferMode mode);
  void loadSpecParsHexagon(const DDFilteredView& fv,
                           HGCalTBParameters& php,
                           const DDCompactView* cpv,
                           const std::string& sdTag1,
                           const std::string& sdTag2);
  void loadSpecParsHexagon(const cms::DDFilteredView& fv,
                           HGCalTBParameters& php,
                           const std::string& sdTag1,
                           const std::string& sdTag2,
                           const std::string& sdTag3,
                           const std::string& sdTag4);
  void loadWaferHexagon(HGCalTBParameters& php);
  void loadCellParsHexagon(const DDCompactView* cpv, HGCalTBParameters& php);
  void loadCellParsHexagon(const cms::DDVectorsMap& vmap, HGCalTBParameters& php);
  void loadCellParsHexagon(const HGCalTBParameters& php);

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
  void loadGeometryHexagon(const std::map<int, HGCalTBGeomParameters::layerParameters>& layers,
                           std::vector<HGCalTBParameters::hgtrform>& trforms,
                           std::vector<bool>& trformUse,
                           const std::unordered_map<int32_t, int32_t>& copies,
                           const HGCalTBParameters::layer_map& copiesInLayers,
                           const std::vector<int32_t>& wafer2copy,
                           const std::vector<HGCalTBGeomParameters::cellParameters>& wafers,
                           const std::map<int, int>& wafertype,
                           const std::map<int, HGCalTBGeomParameters::cellParameters>& cellsf,
                           const std::map<int, HGCalTBGeomParameters::cellParameters>& cellsc,
                           HGCalTBParameters& php);
  void loadSpecParsHexagon(const HGCalTBParameters& php);
  std::vector<double> getDDDArray(const std::string& str, const DDsvalues_type& sv, const int nmin);
  std::pair<double, double> cellPosition(const std::vector<cellParameters>& wafers,
                                         std::vector<cellParameters>::const_iterator& itrf,
                                         int wafer,
                                         double xx,
                                         double yy);
  void rescale(std::vector<double>&, const double s);
  void resetZero(std::vector<double>&);

  constexpr static double tan30deg_ = 0.5773502693;
  HGCalGeomTools geomTools_;
  const double sqrt3_;
  double waferSize_;
};

#endif
