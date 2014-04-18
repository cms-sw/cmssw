#ifndef CSCTriggerPrimitives_CSCGEMTriggerGeometryHelpers_h
#define CSCTriggerPrimitives_CSCGEMTriggerGeometryHelpers_h

#include <DataFormats/MuonDetId/interface/CSCTriggerNumbering.h>
#include <Geometry/GEMGeometry/interface/GEMGeometry.h>
#include <Geometry/GEMGeometry/interface/GEMEtaPartitionSpecs.h>
#include <L1Trigger/CSCCommonTrigger/interface/CSCTriggerGeometry.h>
#include <DataFormats/Math/interface/deltaPhi.h>
#include <DataFormats/Math/interface/normalizedPhi.h>

namespace cscgemtriggeom {

static const double lut_wg_etaMin_etaMax_me11_odd[48][3] = {
{0, 2.44005, 2.44688},
{1, 2.38863, 2.45035},
{2, 2.32742, 2.43077},
{3, 2.30064, 2.40389},
{4, 2.2746, 2.37775},
{5, 2.24925, 2.35231},
{6, 2.22458, 2.32754},
{7, 2.20054, 2.30339},
{8, 2.1771, 2.27985},
{9, 2.15425, 2.25689},
{10, 2.13194, 2.23447},
{11, 2.11016, 2.21258},
{12, 2.08889, 2.19119},
{13, 2.06809, 2.17028},
{14, 2.04777, 2.14984},
{15, 2.02788, 2.12983},
{16, 2.00843, 2.11025},
{17, 1.98938, 2.09108},
{18, 1.97073, 2.0723},
{19, 1.95246, 2.0539},
{20, 1.93456, 2.03587},
{21, 1.91701, 2.01818},
{22, 1.8998, 2.00084},
{23, 1.88293, 1.98382},
{24, 1.86637, 1.96712},
{25, 1.85012, 1.95073},
{26, 1.83417, 1.93463},
{27, 1.8185, 1.91882},
{28, 1.80312, 1.90329},
{29, 1.788, 1.88803},
{30, 1.77315, 1.87302},
{31, 1.75855, 1.85827},
{32, 1.74421, 1.84377},
{33, 1.7301, 1.8295},
{34, 1.71622, 1.81547},
{35, 1.70257, 1.80166},
{36, 1.68914, 1.78807},
{37, 1.67592, 1.77469},
{38, 1.66292, 1.76151},
{39, 1.65011, 1.74854},
{40, 1.63751, 1.73577},
{41, 1.62509, 1.72319},
{42, 1.61287, 1.71079},
{43, 1.60082, 1.69857},
{44, 1.59924, 1.68654},
{45, 1.6006, 1.67467},
{46, 1.60151, 1.66297},
{47, 1.60198, 1.65144} };

static const double lut_wg_etaMin_etaMax_me11_even[48][3] = {
{0, 2.3917, 2.39853},
{1, 2.34037, 2.40199},
{2, 2.27928, 2.38244},
{3, 2.25254, 2.35561},
{4, 2.22655, 2.32951},
{5, 2.20127, 2.30412},
{6, 2.17665, 2.27939},
{7, 2.15267, 2.25529},
{8, 2.12929, 2.2318},
{9, 2.1065, 2.20889},
{10, 2.08425, 2.18652},
{11, 2.06253, 2.16468},
{12, 2.04132, 2.14334},
{13, 2.0206, 2.12249},
{14, 2.00033, 2.1021},
{15, 1.98052, 2.08215},
{16, 1.96113, 2.06262},
{17, 1.94215, 2.04351},
{18, 1.92357, 2.02479},
{19, 1.90538, 2.00645},
{20, 1.88755, 1.98847},
{21, 1.87007, 1.97085},
{22, 1.85294, 1.95357},
{23, 1.83614, 1.93662},
{24, 1.81965, 1.91998},
{25, 1.80348, 1.90365},
{26, 1.78761, 1.88762},
{27, 1.77202, 1.87187},
{28, 1.75672, 1.85641},
{29, 1.74168, 1.84121},
{30, 1.72691, 1.82628},
{31, 1.7124, 1.8116},
{32, 1.69813, 1.79716},
{33, 1.68411, 1.78297},
{34, 1.67032, 1.769},
{35, 1.65675, 1.75526},
{36, 1.64341, 1.74174},
{37, 1.63028, 1.72844},
{38, 1.61736, 1.71534},
{39, 1.60465, 1.70245},
{40, 1.59213, 1.68975},
{41, 1.57981, 1.67724},
{42, 1.56767, 1.66492},
{43, 1.55572, 1.65278},
{44, 1.55414, 1.64082},
{45, 1.55549, 1.62903},
{46, 1.5564, 1.61742},
{47, 1.55686, 1.60596} };

static const double lut_wg_eta_me21_even[112][2] = {};

int cscHalfStripToGEMStripME11(int cscHalfStrip, bool isEven);
int cscHalfStripToGEMPadME11(int cscHalfStrip, bool isEven);
int cscStripToGEMStripME11(int cscStrip, bool isEven);
int cscStripToGEMPadME11(int cscStrip, bool isEven);

int gemStripToCSCHalfStripME11(int gemStrip, bool isEven);
int gemStripToCSCStripME11(int gemStrip, bool isEven);
int gemPadToCSCHalfStripME11(int gemPad, bool isEven);
int gemPadToCSCStripME11(int gemPad, bool isEven);

int cscHalfStripToGEMStripME21(int cscHalfStrip, bool isEven);
int cscHalfStripToGEMPadME21(int cscHalfStrip, bool isEven);
int cscStripToGEMStripME21(int cscStrip, bool isEven);
int cscStripToGEMPadME21(int cscStrip, bool isEven);

int gemStripToCSCHalfStripME21(int gemStrip, bool isEven);
int gemStripToCSCStripME21(int gemStrip, bool isEven);
int gemPadToCSCHalfStripME21(int gemPad, bool isEven);
int gemPadToCSCStripME21(int gemPad, bool isEven);

}

#endif

/*
  // loop on all wiregroups to create a LUT <WG,rollMin,rollMax>
  int numberOfWG(cscChamber->layer(1)->geometry()->numberOfWireGroups());
  std::cout <<"detId " << cscChamber->id() << std::endl;
  for (int i = 0; i< numberOfWG; ++i){
    // find low-eta of WG
    auto length(cscChamber->layer(1)->geometry()->lengthOfWireGroup(i));
//     auto gp(cscChamber->layer(1)->centerOfWireGroup(i));
    auto lpc(cscChamber->layer(1)->geometry()->localCenterOfWireGroup(i));
    auto wireEnds(cscChamber->layer(1)->geometry()->wireTopology()->wireEnds(i));
    auto gpMin(cscChamber->layer(1)->toGlobal(wireEnds.first));
    auto gpMax(cscChamber->layer(1)->toGlobal(wireEnds.second));
    auto etaMin(gpMin.eta());
    auto etaMax(gpMax.eta());
    if (etaMax < etaMin)
      std::swap(etaMin,etaMax);
    //print the eta min and eta max
    //    std::cout << i << " " << etaMin << " " << etaMax << std::endl;
    auto x1(lpc.x() + cos(cscChamber->layer(1)->geometry()->wireAngle())*length/2.);
    auto x2(lpc.x() - cos(cscChamber->layer(1)->geometry()->wireAngle())*length/2.);
    auto z(lpc.z());
    auto y1(cscChamber->layer(1)->geometry()->yOfWireGroup(i,x1));
    auto y2(cscChamber->layer(1)->geometry()->yOfWireGroup(i,x2));
    auto lp1(LocalPoint(x1,y1,z));
    auto lp2(LocalPoint(x2,y2,z));
    auto gp1(cscChamber->layer(1)->toGlobal(lp1));
    auto gp2(cscChamber->layer(1)->toGlobal(lp2));
    auto eta1(gp1.eta());
    auto eta2(gp2.eta());
    if (eta1 < eta2)
      std::swap(eta1,eta2);
    std::cout << "{" << i << ", " << eta1 << ", " << eta2 << "},"<< std::endl;
    
    
//     Std ::cout << "WG "<< i << std::endl;
//    wireGroupGEMRollMap_[i] = assignGEMRoll(gp.eta());
  }

//   // print-out
//   for(auto it = wireGroupGEMRollMap_.begin(); it != wireGroupGEMRollMap_.end(); it++) {
//     std::cout << "WG "<< it->first << " GEM pad " << it->second << std::endl;
//   }
*/
