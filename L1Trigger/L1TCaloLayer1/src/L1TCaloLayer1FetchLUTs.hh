#ifndef L1TCaloLayer1FetchLUTs_hh
#define L1TCaloLayer1FetchLUTs_hh

#include "UCTGeometry.hh"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include <vector>
#include <array>

// External function declaration

class HcalTrigTowerGeometry;
class CaloGeometryRecord;
namespace l1t {
  class CaloParams;
}
class L1TCaloParamsRcd;
class CaloTPGTranscoder;
class CaloTPGRecord;

struct L1TCaloLayer1FetchLUTsTokens {
  template <typename T>
  L1TCaloLayer1FetchLUTsTokens(T &&i1, T &&i2, T &&i3) : geom_(i1), params_(i2), decoder_(i3) {}
  edm::ESGetToken<HcalTrigTowerGeometry, CaloGeometryRecord> geom_;
  edm::ESGetToken<l1t::CaloParams, L1TCaloParamsRcd> params_;
  edm::ESGetToken<CaloTPGTranscoder, CaloTPGRecord> decoder_;
};

bool L1TCaloLayer1FetchLUTs(
    const L1TCaloLayer1FetchLUTsTokens &iTokens,
    const edm::EventSetup &iSetup,
    std::vector<std::array<std::array<std::array<uint32_t, l1tcalo::nEtBins>, l1tcalo::nCalSideBins>,
                           l1tcalo::nCalEtaBins> > &eLUT,
    std::vector<std::array<std::array<std::array<uint32_t, l1tcalo::nEtBins>, l1tcalo::nCalSideBins>,
                           l1tcalo::nCalEtaBins> > &hLUT,
    std::vector<std::array<std::array<uint32_t, l1tcalo::nEtBins>, l1tcalo::nHfEtaBins> > &hfLUT,
    std::vector<unsigned long long int> &hcalFBLUT,
    std::vector<unsigned int> &ePhiMap,
    std::vector<unsigned int> &hPhiMap,
    std::vector<unsigned int> &hfPhiMap,
    bool useLSB = true,
    bool useCalib = true,
    bool useECALLUT = true,
    bool useHCALLUT = true,
    bool useHFLUT = true,
    bool useHCALFBLUT = true,
    int fwVersion = 0);

#endif
