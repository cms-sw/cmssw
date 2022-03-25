#include "L1Trigger/L1TTrackMatch/interface/L1TkEtMissEmuAlgo.h"

using namespace std;

namespace l1tmetemu {
  std::vector<global_phi_t> generateCosLUT(unsigned int size) {  // Fill cosine LUT with integer values
    float phi = 0;
    std::vector<global_phi_t> cosLUT;
    for (unsigned int LUT_idx = 0; LUT_idx < size; LUT_idx++) {
      cosLUT.push_back((global_phi_t)(floor(cos(phi) * (kGlobalPhiBins - 1))));
      phi += l1tmetemu::kStepPhi;
    }
    cosLUT.push_back((global_phi_t)(0));  //Prevent overflow in last bin
    return cosLUT;
  }

  // Generate Eta LUT for track to vertex association
  std::vector<eta_t> generateEtaRegionLUT(vector<double> EtaRegions) {
    std::vector<eta_t> LUT;
    for (unsigned int q = 0; q < EtaRegions.size(); q++) {
      LUT.push_back(digitizeSignedValue<eta_t>(EtaRegions[q], l1tmetemu::kInternalEtaWidth, l1tmetemu::kStepEta));
      //std::cout << LUT[q] << "|" <<  EtaRegions[q] << std::endl;
    }
    return LUT;
  }

  std::vector<z_t> generateDeltaZLUT(vector<double> DeltaZBins) {
    std::vector<z_t> LUT;
    for (unsigned int q = 0; q < DeltaZBins.size(); q++) {
      LUT.push_back(digitizeSignedValue<z_t>(DeltaZBins[q], l1tmetemu::kInternalVTXWidth, l1tmetemu::kStepZ0));
      //std::cout << LUT[q] << "|" <<  DeltaZBins[q] << std::endl;
    }
    return LUT;
  }

  int unpackSignedValue(unsigned int bits, unsigned int nBits) {
    int isign = 1;
    unsigned int digitized_maximum = (1 << nBits) - 1;
    if (bits & (1 << (nBits - 1))) {  // check the sign
      isign = -1;
      bits = (1 << (nBits + 1)) - bits;  // if negative, flip everything for two's complement encoding
    }
    return (int(bits & digitized_maximum)) * isign;
  }

  unsigned int transformSignedValue(unsigned int bits, unsigned int oldnBits, unsigned int newnBits) {
    int isign = 1;
    unsigned int olddigitized_maximum = (1 << oldnBits) - 1;
    unsigned int newdigitized_maximum = (1 << newnBits) - 1;
    if (bits & (1 << (oldnBits - 1))) {  // check the sign
      isign = -1;
      bits = (1 << (oldnBits + 1)) - bits;  // if negative, flip everything for two's complement encoding
    }
    unsigned int temp = (int(bits & olddigitized_maximum));

    temp = round(temp / (1 << (oldnBits - newnBits)));

    if (temp > newdigitized_maximum)
      temp = newdigitized_maximum;
    if (isign < 0)
      temp = (1 << newnBits) - 1 - temp;  // two's complement encoding

    return temp;
  }

}  // namespace l1tmetemu
