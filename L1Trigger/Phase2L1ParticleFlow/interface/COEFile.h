#ifndef L1Trigger_Phase2L1ParticleFlow_CoeFile_h
#define L1Trigger_Phase2L1ParticleFlow_CoeFile_h

// system include files
#include <vector>
#include <string>
#include <numeric>
#include <cstdio>
#include <boost/dynamic_bitset.hpp>
#include <boost/multiprecision/cpp_int.hpp>

// user include files
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/L1TParticleFlow/interface/PFCandidate.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/DiscretePFInputs.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/Region.h"

namespace l1tpf_impl {
  class COEFile {
  public:
    COEFile(const edm::ParameterSet&);
    ~COEFile();

    void close() { fclose(file); }
    template <typename T>
    bool getBit(T value, unsigned bit) {
      return (value >> bit) & 1;
    }
    bool is_open() { return (file != nullptr); }
    void writeHeaderToFile();
    void writeTracksToFile(const std::vector<Region>& regions, bool print = false);

  protected:
    FILE* file;
    std::string coeFileName, bset_string_;
    unsigned int ntracksmax, phiSlices;
    static constexpr unsigned int tracksize = 96;
    boost::dynamic_bitset<> bset_;
    const std::vector<uint32_t> track_word_block_sizes = {14, 1, 12, 16, 12, 13, 4, 3, 7, 14};
    int debug_;
  };
}  // namespace l1tpf_impl

#endif
