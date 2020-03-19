#ifndef DQM_SiStripCommissioningSummary_ViewTranslator_H
#define DQM_SiStripCommissioningSummary_ViewTranslator_H

#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"
#include <vector>
#include <string>
#include <map>
#include <cstdint>

/**
   \class ViewTranslator
   \author P.Kalavase, R.Bainbridge
   
   Utility class that maps Fed and Det keys with the Fec Keys, saves
   them in a root file and also extracts fec keys if given a fed/det
   key
*/
class ViewTranslator {
public:
  /** Mapping type between different keys. */
  typedef std::map<uint32_t, uint32_t> Mapping;

  // -------------------- Build maps --------------------

  /** Build maps from FED cabling object */
  static void buildMaps(const SiStripFedCabling&, Mapping& det_to_fec, Mapping& fed_to_fec);

  /** Build "reduced" map based on "masked" FED key */
  uint32_t fedToFec(const uint32_t& fed_key_mask, const Mapping& input, Mapping& output);

  /** Build "reduced" map based on "masked" DET key */
  uint32_t detToFec(const uint32_t& det_key_mask, const Mapping& input, Mapping& output);

  // -------------------- Root IO --------------------

  //arguments are the root file name, the uint32_t det key to be unpacked, and the map to be filled with
  //the matching complete fed and fec keys
  void detToFec(std::string root_filename, uint32_t& det_key_mask, Mapping& det_to_fec) { ; }

  //arguments are the root file name, the uint32_t fed key to be unpacked, and the map to be filled with
  //the matching complete fed and fec keys
  void fedToFec(std::string root_filename, uint32_t& fed_key_mask, Mapping& fed_to_fec) { ; }

  //arguments are the root file name, the det to fec translation map and the fed to fec translational
  //maps (from makemaps)
  void writeMapsToFile(std::string root_filename, Mapping& det_to_fec, Mapping& fed_to_fec);

private:
  /** Root filename. */
  std::string rootfile_;

  // Some constants
  static const uint16_t fedIdMask_ = 0xFFF;
  static const uint16_t fedFeMask_ = 0xF;
  static const uint16_t fedChMask_ = 0xFF;
};

#endif  // DQM_SiStripCommissioningSummary_ViewTranslator_H
