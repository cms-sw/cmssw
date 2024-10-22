#include "RecoTBCalo/HcalTBObjectUnpacker/interface/HcalTBSlowDataUnpacker.h"
#include "TBDataFormats/HcalTBObjects/interface/HcalTBRunData.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include <iostream>
#include <string>
#include <map>
#include "FWCore/Utilities/interface/Exception.h"

using namespace std;

// structure for Slow Data
struct xdaqSlowDataFormat {
  uint32_t cdfHeader[4];
  uint16_t n_doubles;
  uint16_t n_strings;
  uint16_t key_length;
  uint16_t string_value_length;
  char start_of_data;  // see below
  // char[n_doubles*key_length] doubles names
  // double[n_doubles] doubles values
  // char[n_strings*key_length] strings names
  // char[n_strings*string_value_length] strings values
  // xdaqCommonDataFormatTrailer
};

namespace hcaltb {

  void HcalTBSlowDataUnpacker::unpackMaps(const FEDRawData &raw,
                                          std::map<std::string, std::string> &strings,
                                          std::map<std::string, double> &numerics) const {
    if (raw.size() < 3 * 8) {
      throw cms::Exception("Missing Data") << "No data in the slow data block";
    }

    const struct xdaqSlowDataFormat *sd = (const struct xdaqSlowDataFormat *)(raw.data());

#ifdef DEBUG
    cout << "#doubles = " << sd->n_doubles << endl;
    ;
    cout << "#strings = " << sd->n_strings << endl;
    cout << "key_length = " << sd->key_length << endl;
    cout << "string_value_length = " << sd->string_value_length << endl;
#endif

    // List of doubles:

    const char *keyptr = &sd->start_of_data;
    const double *valptr = (const double *)(&sd->start_of_data + sd->n_doubles * sd->key_length);

    for (int i = 0; i < sd->n_doubles; i++) {
#ifdef DEBUG
      cout << keyptr << " = " << *valptr << endl;
#endif
      numerics[keyptr] = *valptr;
      keyptr += sd->key_length;
      valptr++;
    }

    // List of strings:

    keyptr = (const char *)valptr;
    const char *strptr = (keyptr + sd->n_strings * sd->key_length);

    for (int i = 0; i < sd->n_strings; i++) {
#ifdef DEBUG
      cout << keyptr << " = " << strptr << endl;
#endif
      strings[keyptr] = strptr;
      keyptr += sd->key_length;
      strptr += sd->string_value_length;
    }
  }

  void HcalTBSlowDataUnpacker::unpack(const FEDRawData &raw, HcalTBRunData &htbrd, HcalTBEventPosition &htbep) const {
    map<string, double> sd_dblmap;
    map<string, string> sd_strmap;

    unpackMaps(raw, sd_strmap, sd_dblmap);

    // Now fill the input objects:
    htbrd.setRunData(sd_strmap["RunType"].c_str(), sd_strmap["Beam.Mode"].c_str(), sd_dblmap["Beam.Energy"]);

    htbep.setHFtableCoords(sd_dblmap["HFTable.X"], sd_dblmap["HFTable.Y"], sd_dblmap["HFTable.V"]);

    htbep.setHBHEtableCoords(sd_dblmap["Table.Eta"], sd_dblmap["Table.Phi"]);
  }
}  // namespace hcaltb
