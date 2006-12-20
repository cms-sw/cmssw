#include <DataFormats/L1CSCTrackFinder/interface/CSCTFConstants.h>

const double CSCTFConstants::RAD_PER_DEGREE = M_PI/180.;
const double CSCTFConstants::SECTOR1_CENT_DEG = 45;
const double CSCTFConstants::SECTOR1_CENT_RAD = CSCTFConstants::SECTOR1_CENT_DEG * CSCTFConstants::RAD_PER_DEGREE;
const double CSCTFConstants::SECTOR_DEG = 62.;
const double CSCTFConstants::SECTOR_RAD = CSCTFConstants::SECTOR_DEG*CSCTFConstants::RAD_PER_DEGREE; // radians
const double CSCTFConstants::minEta = .9;
const double CSCTFConstants::maxEta = 2.5;
