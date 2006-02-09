#ifndef CALIBFORMATS_SISTRIPOBJECTS_SISTRIPSTRUCTURE_H
#define CALIBFORMATS_SISTRIPOBJECTS_SISTRIPSTRUCTURE_H
#include "CondFormats/SiStripObjects/interface/SiStripReadoutCabling.h"
//#include "DataFormats/DetId/interface/DetId.h"
//#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
//#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
//#include "DataFormats/SiStripDetId/interface/TIDDetId.h"
//#include "DataFormats/SiStripDetId/interface/TOBDetId.h"
//#include "DataFormats/SiStripDetId/interface/TECDetId.h"
#include <boost/cstdint.hpp>
#include <vector>
//#include <map>

class SiStripStructure {

public:

  SiStripStructure();

  SiStripStructure(const SiStripReadoutCabling *);

  ~SiStripStructure();

  const std::vector<uint32_t> & getActiveDetectorsRawIds() const;

// commented out following methods as they are not anymore used. this functionality
// has been moved to the class: DataFormats/SiStripDetId - SiStripSubStructure
//  const std::vector<uint32_t> & getTIBDetectors(uint32_t layer = 0,
//                                                uint32_t str_fw_bw = 0,
//                                                uint32_t str_int_ext = 0,
//                                                uint32_t str = 0,
//                                                uint32_t det = 0,
//                                                uint32_t ster = 0) const;
//
//  const std::vector<uint32_t> & getTIDDetectors(uint32_t side = 0,
//                                                uint32_t wheel = 0,
//                                                uint32_t ring = 0,
//                                                uint32_t det_fw_bw = 0,
//                                                uint32_t det = 0,
//                                                uint32_t ster = 0) const;
//
//  const std::vector<uint32_t> & getTOBDetectors(uint32_t layer = 0,
//					        uint32_t rod_fw_bw = 0,
//					        uint32_t rod = 0,
//					        uint32_t det = 0,
//					        uint32_t ster = 0 ) const;
//
//  const std::vector<uint32_t> & getTECDetectors(uint32_t side = 0,
//                                                uint32_t wheel = 0,
//                                                uint32_t petal_fw_bw = 0,
//                                                uint32_t petal = 0,
//                                                uint32_t ring = 0,
//                                                uint32_t det_fw_bw = 0,
//                                                uint32_t det = 0,
//                                                uint32_t ster = 0) const;

  void debug() const; // write debug for this similar to the class from Giacomo

private:

  std::vector<uint32_t> theActiveDetectors; // raw ids for active detectors, probably no point looking to non active during monitoring

};

#endif
