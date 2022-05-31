#include "L1Trigger/Phase2L1ParticleFlow/interface/egamma/l2egencoder_ref.h"

using namespace l1ct;

#ifdef CMSSW_GIT_HASH

#include "FWCore/ParameterSet/interface/ParameterSet.h"

l1ct::L2EgEncoderEmulator::L2EgEncoderEmulator(const edm::ParameterSet& pset)
    : L2EgEncoderEmulator(pset.getParameter<uint32_t>("nTKELE_OUT"), pset.getParameter<uint32_t>("nTKPHO_OUT")) {}

#endif

void L2EgEncoderEmulator::toFirmware(const std::vector<ap_uint<64>>& encoded_in, ap_uint<64> encoded_fw[]) const {
  for (unsigned int i = 0; i < nEncodedWords_; i++) {
    encoded_fw[i] = (i < encoded_in.size()) ? encoded_in[i] : ap_uint<64>(0);
  }
}
