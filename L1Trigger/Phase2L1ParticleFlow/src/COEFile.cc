#include "L1Trigger/Phase2L1ParticleFlow/interface/COEFile.h"

using namespace l1tpf_impl;

COEFile::COEFile(const edm::ParameterSet& iConfig)
    : file(nullptr),
      coeFileName(iConfig.getUntrackedParameter<std::string>("coeFileName", "")),
      bset_string_(""),
      ntracksmax(iConfig.getUntrackedParameter<unsigned int>("ntracksmax")),
      phiSlices(iConfig.getParameter<std::vector<edm::ParameterSet>>("regions")[0].getParameter<uint32_t>("phiSlices")),
      debug_(iConfig.getUntrackedParameter<int>("debug", 0)) {
  file = fopen(coeFileName.c_str(), "w");
  writeHeaderToFile();
  bset_.resize(tracksize);
}

COEFile::~COEFile() {}

void COEFile::writeHeaderToFile() {
  char depth_width[256];
  snprintf(depth_width,
           255,
           "; of depth=%i, and width=%i. In this case, values are specified\n",
           ntracksmax,
           tracksize * phiSlices);
  std::vector<std::string> vheader = {"; Sample memory initialization file for Dual Port Block Memory,\n",
                                      "; v3.0 or later.\n",
                                      "; Board: VCU118\n",
                                      "; tmux: 1\n",
                                      ";\n",
                                      "; This .COE file specifies the contents for a block memory\n",
                                      std::string(depth_width),
                                      "; in binary format.\n",
                                      "memory_initialization_radix=2;\n",
                                      "memory_initialization_vector=\n"};
  for (uint32_t i = 0; i < vheader.size(); ++i)
    fprintf(file, "%s", vheader[i].c_str());
}

void COEFile::writeTracksToFile(const std::vector<Region>& regions, bool print) {
  PropagatedTrack current_track;
  bool has_track = false;
  for (unsigned int irow = 0; irow < ntracksmax; irow++) {
    for (unsigned int icol = 0; icol < regions.size(); icol++) {
      if (regions[icol].track.size() <= irow)
        has_track = false;
      else
        has_track = true;

      if (has_track) {
        // select the track that will be converted to a bit string
        current_track = regions[icol].track[irow];

        // convert the values in a PropogatedTrack to a 96-bit track word
        for (unsigned int iblock = 0; iblock < track_word_block_sizes.size(); iblock++) {
          for (unsigned int ibit = 0; ibit < track_word_block_sizes[iblock]; ibit++) {
            int offset = std::accumulate(track_word_block_sizes.begin(), track_word_block_sizes.begin() + iblock, 0);
            switch (iblock) {
              case 0:
                bset_.set(ibit + offset, getBit<uint16_t>(current_track.hwPt, ibit));
                break;
              case 1:
                bset_.set(ibit + offset, current_track.hwCharge);
                break;
              case 2:
                bset_.set(ibit + offset, getBit<uint16_t>(current_track.hwVtxPhi, ibit));
                break;
              case 3:
                bset_.set(ibit + offset, getBit<uint16_t>(current_track.hwVtxEta, ibit));
                break;
              case 4:
                bset_.set(ibit + offset, getBit<uint16_t>(current_track.hwZ0, ibit));
                break;
              case 5:
                bset_.set(ibit + offset, false);
                break;
              case 6:
                bset_.set(ibit + offset, getBit<uint16_t>(current_track.hwChi2, ibit));
                break;
              case 7:
                bset_.set(ibit + offset, false);
                break;
              case 8:
                bset_.set(ibit + offset, getBit<uint16_t>(current_track.hwStubs, ibit));
                break;
              case 9:
                bset_.set(ibit + offset, false);
                break;
            }
          }
        }

        // print the track word to the COE file
        boost::to_string(bset_, bset_string_);
        fprintf(file, "%s", bset_string_.c_str());

        // print some debugging information
        if (debug_ && print && irow == 0 && icol == 0) {
          printf("region: eta=[%f,%f] phi=%f+/-%f\n",
                 regions[icol].etaMin,
                 regions[icol].etaMax,
                 regions[icol].phiCenter,
                 regions[icol].phiHalfWidth);
          printf("l1t::PFTrack (pT,eta,phi) [float] = (%f,%f,%f)\n",
                 current_track.src->p4().Pt(),
                 current_track.src->p4().Eta(),
                 current_track.src->p4().Phi());
          printf("l1t::PFTrack (pT,eta,phi) [int] = (%i,%i,%i)\n",
                 current_track.src->hwPt(),
                 current_track.src->hwEta(),
                 current_track.src->hwPhi());
          printf("l1tpf_impl::PropagatedTrack (1/pT,eta,phi) [int,10] = (%i,%i,%i)\n",
                 current_track.hwPt,
                 current_track.hwVtxEta,
                 current_track.hwVtxPhi);
          printf("l1tpf_impl::PropagatedTrack (1/pT,eta,phi) [int,2] = (%s,%s,%s)\n",
                 std::bitset<16>(current_track.hwPt).to_string().c_str(),
                 std::bitset<32>(current_track.hwVtxEta).to_string().c_str(),
                 std::bitset<32>(current_track.hwVtxPhi).to_string().c_str());
          printf("bitset = %s\n", bset_string_.c_str());
        }
      } else {
        bset_.reset();
        boost::to_string(bset_, bset_string_);
        fprintf(file, "%s", bset_string_.c_str());
      }
    }
    fprintf(file, (irow == ntracksmax - 1) ? ";\n" : ",\n");
  }
}
