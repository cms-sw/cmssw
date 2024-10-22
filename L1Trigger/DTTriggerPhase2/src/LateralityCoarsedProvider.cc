#include "L1Trigger/DTTriggerPhase2/interface/LateralityCoarsedProvider.h"
#include <cmath>
#include <memory>

using namespace edm;
using namespace std;
using namespace cmsdt;
// ============================================================================
// Constructors and destructor
// ============================================================================
LateralityCoarsedProvider::LateralityCoarsedProvider(const ParameterSet &pset, edm::ConsumesCollector &iC)
    : LateralityProvider(pset, iC),
      debug_(pset.getUntrackedParameter<bool>("debug")),
      laterality_filename_(pset.getParameter<edm::FileInPath>("laterality_filename")) {
  if (debug_)
    LogDebug("LateralityCoarsedProvider") << "LateralityCoarsedProvider: constructor";

  fill_lat_combinations();
}

LateralityCoarsedProvider::~LateralityCoarsedProvider() {
  if (debug_)
    LogDebug("LateralityCoarsedProvider") << "LateralityCoarsedProvider: destructor";
}

// ============================================================================
// Main methods (initialise, run, finish)
// ============================================================================
void LateralityCoarsedProvider::initialise(const edm::EventSetup &iEventSetup) {
  if (debug_)
    LogDebug("LateralityCoarsedProvider") << "LateralityCoarsedProvider::initialiase";
}

void LateralityCoarsedProvider::run(edm::Event &iEvent,
                                    const edm::EventSetup &iEventSetup,
                                    MuonPathPtrs &muonpaths,
                                    std::vector<lat_vector> &lateralities) {
  if (debug_)
    LogDebug("LateralityCoarsedProvider") << "LateralityCoarsedProvider: run";

  // fit per SL (need to allow for multiple outputs for a single mpath)
  for (auto &muonpath : muonpaths) {
    analyze(muonpath, lateralities);
  }
}

void LateralityCoarsedProvider::finish() {
  if (debug_)
    LogDebug("LateralityCoarsedProvider") << "LateralityCoarsedProvider: finish";
};

//------------------------------------------------------------------
//--- Metodos privados
//------------------------------------------------------------------

void LateralityCoarsedProvider::analyze(MuonPathPtr &inMPath, std::vector<lat_vector> &lateralities) {
  if (debug_)
    LogDebug("LateralityCoarsedProvider") << "DTp2:analyze \t\t\t\t starts";

  auto coarsified_times = coarsify_times(inMPath);

  for (auto &lat_combination : lat_combinations) {
    if (inMPath->missingLayer() == lat_combination.missing_layer &&
        inMPath->cellLayout()[0] == lat_combination.cellLayout[0] &&
        inMPath->cellLayout()[1] == lat_combination.cellLayout[1] &&
        inMPath->cellLayout()[2] == lat_combination.cellLayout[2] &&
        inMPath->cellLayout()[3] == lat_combination.cellLayout[3] &&
        coarsified_times[0] == lat_combination.coarsed_times[0] &&
        coarsified_times[1] == lat_combination.coarsed_times[1] &&
        coarsified_times[2] == lat_combination.coarsed_times[2] &&
        coarsified_times[3] == lat_combination.coarsed_times[3]) {
      lateralities.push_back(lat_combination.latcombs);
      return;
    }
  }
  lateralities.push_back(LAT_VECTOR_NULL);
  return;
}

std::vector<short> LateralityCoarsedProvider::coarsify_times(MuonPathPtr &inMPath) {
  int max_time = -999;
  // obtain the maximum time to do the coarsification
  for (int layer = 0; layer < cmsdt::NUM_LAYERS; layer++) {
    if (inMPath->missingLayer() == layer)
      continue;
    if (inMPath->primitive(layer)->tdcTimeStamp() > max_time)
      max_time = inMPath->primitive(layer)->tdcTimeStamp();
  }

  // do the coarsification
  std::vector<short> coarsified_times;
  for (int layer = 0; layer < cmsdt::NUM_LAYERS; layer++) {
    if (inMPath->missingLayer() == layer) {
      coarsified_times.push_back(-1);
      continue;
    }
    auto coarsified_time = max_time - inMPath->primitive(layer)->tdcTimeStamp();
    // transform into tdc counts
    coarsified_time = (int)round(((float)TIME_TO_TDC_COUNTS / (float)LHC_CLK_FREQ) * coarsified_time);
    // keep the LAT_MSB_BITS
    coarsified_time = coarsified_time >> (LAT_TOTAL_BITS - LAT_MSB_BITS);

    if (inMPath->missingLayer() == -1) {  // 4-hit candidates
      if (coarsified_time <= LAT_P0_4H)
        coarsified_times.push_back(0);
      else if (coarsified_time <= LAT_P1_4H)
        coarsified_times.push_back(1);
      else if (coarsified_time <= LAT_P2_4H)
        coarsified_times.push_back(2);
      else
        coarsified_times.push_back(3);
    } else {  // 3-hit candidates
      if (coarsified_time <= LAT_P0_3H)
        coarsified_times.push_back(0);
      else if (coarsified_time <= LAT_P1_3H)
        coarsified_times.push_back(1);
      else if (coarsified_time <= LAT_P2_3H)
        coarsified_times.push_back(2);
      else
        coarsified_times.push_back(3);
    }
  }
  return coarsified_times;
}

void LateralityCoarsedProvider::fill_lat_combinations() {
  std::ifstream latFile(laterality_filename_.fullPath());  // Open file
  if (latFile.fail()) {
    throw cms::Exception("Missing Input File")
        << "LateralityCoarsedProvider::fill_lat_combinations() -  Cannot find " << laterality_filename_.fullPath();
    return;
  }

  std::string line;

  short line_counter = 0;  // Line counter

  // Bit masks for every parameter
  int _12bitMask = 0xFFF;   // 12 bits
  int _layoutMask = 0xE00;  // 3 bits
  int _is4HitMask = 0x100;  // 1 bit
  int _coarsedMask = 0xFF;  // 8 bits
  int _layerMask = 0xC0;    // 2 bits

  while (std::getline(latFile, line)) {
    if (line == "000000000000") {
      line_counter++;
      continue;
    }  //skip zeros

    if (line.size() == 12) {
      std::vector<std::vector<short>> transformedVector = convertString(line);
      latcomb lat0 = {
          transformedVector[0][0], transformedVector[0][1], transformedVector[0][2], transformedVector[0][3]};
      latcomb lat1 = {
          transformedVector[1][0], transformedVector[1][1], transformedVector[1][2], transformedVector[1][3]};
      latcomb lat2 = {
          transformedVector[2][0], transformedVector[2][1], transformedVector[2][2], transformedVector[2][3]};

      //Transforming line number to binary
      short address = line_counter & _12bitMask;  // 12 bits

      short layout =
          (address & _layoutMask) >> 9;  //Doing AND and displacing 9 bits to the right to obtain 3 bits of layout
      short is4Hit = (address & _is4HitMask) >> 8;
      short coarsed = address & _coarsedMask;

      short bit1Layout = (layout & (1));
      short bit2Layout = (layout & (1 << 1)) >> 1;
      short bit3Layout = (layout & (1 << 2)) >> 2;

      //Logic implementation
      short missingLayer = -1;
      short layout_comb[NUM_LAYERS] = {bit3Layout, bit2Layout, bit1Layout, -1};
      short coarsedTimes[NUM_LAYERS] = {0, 0, 0, 0};

      if (is4Hit != 1) {  //3 hit case
        missingLayer =
            (coarsed & _layerMask) >> 6;  //Missing layer is given by the two most significative bits of coarsed vector
        coarsedTimes[missingLayer] = -1;  //Missing layer set to -1
      }

      // Filling coarsedTimes vector without the missing layer
      if (missingLayer != -1) {
        switch (missingLayer) {
          case 0:
            coarsedTimes[1] = (coarsed & 0x30) >> 4;
            coarsedTimes[2] = (coarsed & 0x0C) >> 2;
            coarsedTimes[3] = coarsed & 0x03;
            lat0 = {-1, transformedVector[0][1], transformedVector[0][2], transformedVector[0][3]};
            lat1 = {-1, transformedVector[1][1], transformedVector[1][2], transformedVector[1][3]};
            lat2 = {-1, transformedVector[2][1], transformedVector[2][2], transformedVector[2][3]};
            break;
          case 1:
            coarsedTimes[0] = (coarsed & 0x30) >> 4;
            coarsedTimes[2] = (coarsed & 0x0C) >> 2;
            coarsedTimes[3] = coarsed & 0x03;
            lat0 = {transformedVector[0][0], -1, transformedVector[0][2], transformedVector[0][3]};
            lat1 = {transformedVector[1][0], -1, transformedVector[1][2], transformedVector[1][3]};
            lat2 = {transformedVector[2][0], -1, transformedVector[2][2], transformedVector[2][3]};
            break;
          case 2:
            coarsedTimes[0] = (coarsed & 0x30) >> 4;
            coarsedTimes[1] = (coarsed & 0x0C) >> 2;
            coarsedTimes[3] = coarsed & 0x03;
            lat0 = {transformedVector[0][0], transformedVector[0][1], -1, transformedVector[0][3]};
            lat1 = {transformedVector[1][0], transformedVector[1][1], -1, transformedVector[1][3]};
            lat2 = {transformedVector[2][0], transformedVector[2][1], -1, transformedVector[2][3]};
            break;
          case 3:
            coarsedTimes[0] = (coarsed & 0x30) >> 4;
            coarsedTimes[1] = (coarsed & 0x0C) >> 2;
            coarsedTimes[2] = coarsed & 0x03;
            lat0 = {transformedVector[0][0], transformedVector[0][1], transformedVector[0][2], -1};
            lat1 = {transformedVector[1][0], transformedVector[1][1], transformedVector[1][2], -1};
            lat2 = {transformedVector[2][0], transformedVector[2][1], transformedVector[2][2], -1};
            break;

          default:
            break;
        }

      } else {  //4 hit case
        coarsedTimes[0] = (coarsed & 0xC0) >> 6;
        coarsedTimes[1] = (coarsed & 0x30) >> 4;
        coarsedTimes[2] = (coarsed & 0x0C) >> 2;
        coarsedTimes[3] = coarsed & 0x03;
      }

      lat_coarsed_combination lat_temp = {missingLayer,
                                          {layout_comb[0], layout_comb[1], layout_comb[2], layout_comb[3]},
                                          {coarsedTimes[0], coarsedTimes[1], coarsedTimes[2], coarsedTimes[3]},
                                          {lat0, lat1, lat2}};
      lat_combinations.push_back(lat_temp);

    } else {  //size different from 12
      std::cerr << "Error: line " << line_counter << " does not contain 12 bits." << std::endl;
    }
    line_counter++;
  };

  //closing lateralities file
  latFile.close();
};

// Function to convert a 12 bit string in a a vector of 4 bit vectors
std::vector<std::vector<short>> LateralityCoarsedProvider::convertString(std::string chain) {
  std::vector<std::vector<short>> result;

  for (size_t i = 0; i < chain.size(); i += 4) {
    std::vector<short> group;
    for (size_t j = 0; j < 4; j++) {
      group.push_back(chain[i + j] - '0');  // Convert the character to integer
    }
    result.push_back(group);
  }

  return result;
}
