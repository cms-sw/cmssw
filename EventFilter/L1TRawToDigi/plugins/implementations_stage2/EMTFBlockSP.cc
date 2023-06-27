// Code to unpack the "SP Output Data Record"

#include "EventFilter/L1TRawToDigi/plugins/UnpackerFactory.h"

#include "EMTFCollections.h"
#include "EMTFUnpackerTools.h"

// This is the "header" - no EMTFBlockSP.h file is needed
namespace l1t {
  namespace stage2 {
    namespace emtf {

      class SPBlockUnpacker : public Unpacker {  // "SPBlockUnpacker" inherits from "Unpacker"
      public:
        virtual int checkFormat(const Block& block);
        bool unpack(const Block& block,
                    UnpackerCollections* coll) override;  // Apparently it's always good to use override in C++
        // virtual bool packBlock(const Block& block, UnpackerCollections *coll) override;
      };

      // class SPBlockPacker : public Packer { // "SPBlockPacker" inherits from "Packer"
      // public:
      // 	virtual bool unpack(const Block& block, UnpackerCollections *coll) override; // Apparently it's always good to use override in C++
      // };

    }  // namespace emtf
  }    // namespace stage2
}  // namespace l1t

namespace l1t {
  namespace stage2 {
    namespace emtf {

      int SPBlockUnpacker::checkFormat(const Block& block) {
        auto payload = block.payload();
        int errors = 0;

        // Check the number of 16-bit words
        if (payload.size() != 8) {
          errors += 1;
          edm::LogError("L1T|EMTF") << "Payload size in 'SP Output Data Record' is different than expected";
        }

        // Check that each word is 16 bits
        for (unsigned int i = 0; i < 8; i++) {
          if (GetHexBits(payload[i], 16, 31) != 0) {
            errors += 1;
            edm::LogError("L1T|EMTF") << "Payload[" << i << "] has more than 16 bits in 'SP Output Data Record'";
          }
        }

        uint16_t SP1a = payload[0];
        uint16_t SP1b = payload[1];
        uint16_t SP1c = payload[2];
        uint16_t SP1d = payload[3];
        uint16_t SP2a = payload[4];
        uint16_t SP2b = payload[5];
        uint16_t SP2c = payload[6];
        uint16_t SP2d = payload[7];

        // Check Format
        if (GetHexBits(SP1a, 15, 15) != 1) {
          errors += 1;
          edm::LogError("L1T|EMTF") << "Format identifier bits in SP1a are incorrect";
        }
        if (GetHexBits(SP1b, 15, 15) != 0) {
          errors += 1;
          edm::LogError("L1T|EMTF") << "Format identifier bits in SP1b are incorrect";
        }
        if (GetHexBits(SP1c, 15, 15) != 1) {
          errors += 1;
          edm::LogError("L1T|EMTF") << "Format identifier bits in SP1c are incorrect";
        }
        if (GetHexBits(SP1d, 15, 15) != 0) {
          errors += 1;
          edm::LogError("L1T|EMTF") << "Format identifier bits in SP1d are incorrect";
        }
        if (GetHexBits(SP2a, 15, 15) != 0) {
          errors += 1;
          edm::LogError("L1T|EMTF") << "Format identifier bits in SP2a are incorrect";
        }
        if (GetHexBits(SP2b, 15, 15) != 1) {
          errors += 1;
          edm::LogError("L1T|EMTF") << "Format identifier bits in SP2b are incorrect";
        }
        if (GetHexBits(SP2c, 15, 15) != 1) {
          errors += 1;
          edm::LogError("L1T|EMTF") << "Format identifier bits in SP2c are incorrect";
        }
        if (GetHexBits(SP2d, 15, 15) != 0) {
          errors += 1;
          edm::LogError("L1T|EMTF") << "Format identifier bits in SP2d are incorrect";
        }

        return errors;
      }

      // Converts CSC_ID, sector, subsector, and neighbor
      std::vector<int> convert_SP_location(int _csc_ID, int _sector, int _subsector, int _station) {
        int new_sector = _sector;
        if (_station == 1) {
          if (_csc_ID < 0) {
            int arr[] = {_csc_ID, -99, -99, -99};
            std::vector<int> vec(arr, arr + 4);
            return vec;
          } else if (_csc_ID == 0) {
            int arr[] = {-1, -1, -1, -1};
            std::vector<int> vec(arr, arr + 4);
            return vec;
          } else if (_csc_ID <= 9) {
            int arr[] = {_csc_ID, new_sector, _subsector + 1, 0};
            std::vector<int> vec(arr, arr + 4);
            return vec;
          } else
            new_sector = (_sector != 1) ? _sector - 1 : 6;

          if (_csc_ID == 10) {
            int arr[] = {3, new_sector, 2, 1};
            std::vector<int> vec(arr, arr + 4);
            return vec;
          } else if (_csc_ID == 11) {
            int arr[] = {6, new_sector, 2, 1};
            std::vector<int> vec(arr, arr + 4);
            return vec;
          } else if (_csc_ID == 12) {
            int arr[] = {9, new_sector, 2, 1};
            std::vector<int> vec(arr, arr + 4);
            return vec;
          } else {
            int arr[] = {_csc_ID, -99, -99, -99};
            std::vector<int> vec(arr, arr + 4);
            return vec;
          }
        } else if (_station == 2 || _station == 3 || _station == 4) {
          if (_csc_ID < 0) {
            int arr[] = {_csc_ID, -99, -99, -99};
            std::vector<int> vec(arr, arr + 4);
            return vec;
          } else if (_csc_ID == 0) {
            int arr[] = {-1, -1, -1, -1};
            std::vector<int> vec(arr, arr + 4);
            return vec;
          } else if (_csc_ID <= 9) {
            int arr[] = {_csc_ID, new_sector, -1, 0};
            std::vector<int> vec(arr, arr + 4);
            return vec;
          } else
            new_sector = (_sector != 1) ? _sector - 1 : 6;

          if (_csc_ID == 10) {
            int arr[] = {3, new_sector, -1, 1};
            std::vector<int> vec(arr, arr + 4);
            return vec;
          } else if (_csc_ID == 11) {
            int arr[] = {9, new_sector, -1, 1};
            std::vector<int> vec(arr, arr + 4);
            return vec;
          } else {
            int arr[] = {_csc_ID, -99, -99, -99};
            std::vector<int> vec(arr, arr + 4);
            return vec;
          }
        } else {
          int arr[] = {-99, -99, -99, -99};
          std::vector<int> vec(arr, arr + 4);
          return vec;
        }
      }

      bool SPBlockUnpacker::unpack(const Block& block, UnpackerCollections* coll) {
        // std::cout << "Inside EMTFBlockSP.cc: unpack" << std::endl;
        // LogDebug("L1T|EMTF") << "Inside EMTFBlockSP.cc: unpack"; // Why doesn't this work? - AWB 09.04.16

        // Get the payload for this block, made up of 16-bit words (0xffff)
        // Format defined in MTF7Payload::getBlock() in src/Block.cc
        // payload[0] = bits 0-15, payload[1] = 16-31, payload[3] = 32-47, etc.
        auto payload = block.payload();

        // FW version is computed as (Year - 2000)*2^9 + Month*2^5 + Day (see Block.cc and EMTFBlockTrailers.cc)
        bool useNNBits_ = getAlgoVersion() >= 11098;   // FW versions >= 26.10.2021
        bool useMUSBits_ = getAlgoVersion() >= 11306;  // FW versions >= 10.01.2022
        bool reducedDAQWindow =
            (getAlgoVersion() >=
             11656);  // Firmware from 08.12.22 which is used as a flag for new reduced readout window - EY 01.03.23

        static constexpr int looseShower_ = 1;
        static constexpr int nominalShower_ = 2;
        static constexpr int tightShower_ = 4;

        // Check Format of Payload
        l1t::emtf::SP SP_;
        for (int err = 0; err < checkFormat(block); err++)
          SP_.add_format_error();

        // Assign payload to 16-bit words
        uint16_t SP1a = payload[0];
        uint16_t SP1b = payload[1];
        uint16_t SP1c = payload[2];
        uint16_t SP1d = payload[3];
        uint16_t SP2a = payload[4];
        uint16_t SP2b = payload[5];
        uint16_t SP2c = payload[6];
        uint16_t SP2d = payload[7];

        // res is a pointer to a collection of EMTFDaqOut class objects
        // There is one EMTFDaqOut for each MTF7 (60 deg. sector) in the event
        EMTFDaqOutCollection* res;
        res = static_cast<EMTFCollections*>(coll)->getEMTFDaqOuts();
        int iOut = res->size() - 1;
        std::vector<int> conv_vals_SP;
        std::vector<int> conv_vals_pT_LUT;

        EMTFHitCollection* res_hit;
        res_hit = static_cast<EMTFCollections*>(coll)->getEMTFHits();

        EMTFTrackCollection* res_track;
        res_track = static_cast<EMTFCollections*>(coll)->getEMTFTracks();
        EMTFTrack Track_;

        RegionalMuonCandBxCollection* res_cand;
        res_cand = static_cast<EMTFCollections*>(coll)->getRegionalMuonCands();
        RegionalMuonCand mu_(0, 0, 0, 0, 0, 0, 0, tftype::emtf_pos);

        RegionalMuonShowerBxCollection* res_shower;
        res_shower = static_cast<EMTFCollections*>(coll)->getRegionalMuonShowers();
        RegionalMuonShower muShower_(false, false, false, false, false, false);

        ///////////////////////////////////
        // Unpack the SP Output Data Record
        ///////////////////////////////////

        SP_.set_phi_full(GetHexBits(SP1a, 0, 12));
        SP_.set_c(GetHexBits(SP1a, 13, 13));
        SP_.set_hl(GetHexBits(SP1a, 14, 14));

        SP_.set_phi_GMT(TwosCompl(8, GetHexBits(SP1b, 0, 7)));
        SP_.set_quality_GMT(GetHexBits(SP1b, 8, 11));
        SP_.set_bc0(GetHexBits(SP1b, 12, 12));
        SP_.set_vc(GetHexBits(SP1b, 14, 14));

        SP_.set_eta_GMT(TwosCompl(9, GetHexBits(SP1c, 0, 8)));
        SP_.set_mode(GetHexBits(SP1c, 9, 12));

        if (useMUSBits_) {
          SP_.set_mus(GetHexBits(SP1b, 13, 13, SP1c, 13, 14));
        } else {
          SP_.set_se(GetHexBits(SP1b, 13, 13));
          SP_.set_bx(GetHexBits(SP1c, 13, 14));
        }

        SP_.set_pt_GMT(GetHexBits(SP1d, 0, 8));
        SP_.set_me1_stub_num(GetHexBits(SP1d, 9, 9));
        SP_.set_me1_CSC_ID(GetHexBits(SP1d, 10, 13));
        SP_.set_me1_subsector(GetHexBits(SP1d, 14, 14));

        SP_.set_me2_stub_num(GetHexBits(SP2a, 0, 0));
        SP_.set_me2_CSC_ID(GetHexBits(SP2a, 1, 4));
        SP_.set_me3_stub_num(GetHexBits(SP2a, 5, 5));
        SP_.set_me3_CSC_ID(GetHexBits(SP2a, 6, 9));
        SP_.set_me4_stub_num(GetHexBits(SP2a, 10, 10));
        SP_.set_me4_CSC_ID(GetHexBits(SP2a, 11, 14));

        SP_.set_me1_delay(GetHexBits(SP2b, 0, 2));
        SP_.set_me2_delay(GetHexBits(SP2b, 3, 5));
        SP_.set_me3_delay(GetHexBits(SP2b, 6, 8));
        SP_.set_me4_delay(GetHexBits(SP2b, 9, 11));
        if (reducedDAQWindow)  // reduced DAQ window is used only after run3 DAQ format
          SP_.set_tbin(GetHexBits(SP2b, 12, 14) + 1);
        else
          SP_.set_tbin(GetHexBits(SP2b, 12, 14));

        if (useNNBits_) {
          SP_.set_pt_dxy_GMT(GetHexBits(SP2c, 0, 7));
          SP_.set_dxy_GMT(GetHexBits(SP2c, 8, 10));
          SP_.set_nn_pt_valid(GetHexBits(SP2c, 11, 11));
        } else {
          SP_.set_pt_LUT_addr(GetHexBits(SP2c, 0, 14, SP2d, 0, 14));
        }

        // SP_.set_dataword     ( uint64_t dataword );

        ImportSP(Track_, SP_, (res->at(iOut)).PtrEventHeader()->Endcap(), (res->at(iOut)).PtrEventHeader()->Sector());
        // Track_.ImportPtLUT( Track_.Mode(), Track_.Pt_LUT_addr() );  // Deprecated ... replace? - AWB 15.03.17

        if (!(res->at(iOut)).PtrSPCollection()->empty())
          if (SP_.TBIN() == (res->at(iOut)).PtrSPCollection()->at((res->at(iOut)).PtrSPCollection()->size() - 1).TBIN())
            Track_.set_track_num((res->at(iOut)).PtrSPCollection()->size());
          else
            Track_.set_track_num(0);
        else
          Track_.set_track_num(0);

        // For single-LCT tracks, "Track_num" = 2 (last in collection)
        if (SP_.Quality_GMT() == 0)
          Track_.set_track_num(2);

        mu_.setHwSign(SP_.C());
        mu_.setHwSignValid(SP_.VC());
        mu_.setHwQual(SP_.Quality_GMT());
        mu_.setHwEta(SP_.Eta_GMT());
        mu_.setHwPhi(SP_.Phi_GMT());
        mu_.setHwPt(SP_.Pt_GMT());
        if (useNNBits_) {
          mu_.setHwPtUnconstrained(SP_.Pt_dxy_GMT());
          mu_.setHwDXY(SP_.Dxy_GMT());
        }
        mu_.setTFIdentifiers(Track_.Sector() - 1, (Track_.Endcap() == 1) ? emtf_pos : emtf_neg);
        mu_.setTrackSubAddress(RegionalMuonCand::kTrkNum, Track_.Track_num());
        // Truncated to 11 bits and offset by 25 from global event BX in EMTF firmware
        int EMTF_kBX = ((res->at(iOut)).PtrEventHeader()->L1A_BXN() % 2048) - 25 + Track_.BX();
        if (EMTF_kBX < 0)
          EMTF_kBX += 2048;
        mu_.setTrackSubAddress(RegionalMuonCand::kBX, EMTF_kBX);
        // mu_.set_dataword   ( SP_.Dataword() );
        // Track_.set_GMT(mu_);

        // Set Regional Muon Showers
        if (useMUSBits_) {
          muShower_.setTFIdentifiers(Track_.Sector() - 1, (Track_.Endcap() == 1) ? emtf_pos : emtf_neg);
          muShower_.setOneLooseInTime(SP_.MUS() >= looseShower_ ? true : false);
          muShower_.setOneNominalInTime(SP_.MUS() >= nominalShower_ ? true : false);
          muShower_.setOneTightInTime(SP_.MUS() >= tightShower_ ? true : false);
        }

        ///////////////////////
        // Match hits to tracks
        ///////////////////////

        // Find the track delay
        int nDelay[3] = {0, 0, 0};  // Number of hits in the track with delay 0, 1, or 2
        if (Track_.Mode() >= 8)
          nDelay[SP_.ME1_delay()] += 1;
        if ((Track_.Mode() % 8) >= 4)
          nDelay[SP_.ME2_delay()] += 1;
        if ((Track_.Mode() % 4) >= 2)
          nDelay[SP_.ME3_delay()] += 1;
        if ((Track_.Mode() % 2) == 1)
          nDelay[SP_.ME4_delay()] += 1;

        int trk_delay = -99;
        // Assume 2nd-earliest LCT configuration
        if (nDelay[2] >= 2)
          trk_delay = 2;
        else if (nDelay[2] + nDelay[1] >= 2)
          trk_delay = 1;
        else if (nDelay[2] + nDelay[1] + nDelay[0] >= 2)
          trk_delay = 0;

        // // For earliest LCT configuration
        // if      (nDelay[2]                         >= 1) trk_delay = 2;
        // else if (nDelay[2] + nDelay[1]             >= 1) trk_delay = 1;
        // else if (nDelay[2] + nDelay[1] + nDelay[0] >= 1) trk_delay = 0;

        // Reverse 'rotate by 2' to get CPPF subsector number
        auto get_subsector_rpc_cppf = [](int subsector_rpc) { return ((subsector_rpc + 3) % 6) + 1; };

        std::array<int, 4> St_hits{{0, 0, 0, 0}};  // Number of matched hits in each station

        for (auto const& Hit : *res_hit) {
          if (Track_.Mode() == 1)
            continue;  // Special case dealt with later
          if (Hit.Endcap() != Track_.Endcap())
            continue;

          int hit_delay = -99;
          if (Hit.Station() == 1)
            hit_delay = SP_.ME1_delay();
          else if (Hit.Station() == 2)
            hit_delay = SP_.ME2_delay();
          else if (Hit.Station() == 3)
            hit_delay = SP_.ME3_delay();
          else if (Hit.Station() == 4)
            hit_delay = SP_.ME4_delay();

          // Require exact matching according to TBIN and delays
          if (Hit.BX() + 3 + hit_delay != SP_.TBIN() + trk_delay)
            continue;

          // Match hit in station 1
          conv_vals_SP =
              convert_SP_location(SP_.ME1_CSC_ID(), (res->at(iOut)).PtrEventHeader()->Sector(), SP_.ME1_subsector(), 1);

          if (Hit.Station() == 1 && Hit.Sector() == conv_vals_SP.at(1) && Hit.Neighbor() == conv_vals_SP.at(3) &&
              Hit.Stub_num() == SP_.ME1_stub_num()) {
            if (Hit.Is_CSC() == 1 && (Hit.CSC_ID() != conv_vals_SP.at(0) || Hit.Subsector() != conv_vals_SP.at(2)))
              continue;

            int tmp_subsector = get_subsector_rpc_cppf(Hit.Subsector_RPC());
            int RPC_subsector = ((tmp_subsector - 1) / 3) + 1;  // Map RPC subsector to equivalent CSC subsector
            int RPC_CSC_ID = ((tmp_subsector - 1) % 3) + 4;     // Map RPC subsector and ring to equivalent CSC ID

            if (Hit.Is_RPC() == 1 && (RPC_CSC_ID != conv_vals_SP.at(0) || RPC_subsector != conv_vals_SP.at(2)))
              continue;

            if (St_hits.at(0) == 0) {  // Only add the first matched hit to the track
              Track_.push_Hit((Hit));
              mu_.setTrackSubAddress(RegionalMuonCand::kME1Seg, SP_.ME1_stub_num());
              mu_.setTrackSubAddress(
                  RegionalMuonCand::kME1Ch,
                  L1TMuonEndCap::calc_uGMT_chamber(conv_vals_SP.at(0), conv_vals_SP.at(2), conv_vals_SP.at(3), 1));
            }
            St_hits.at(0) += 1;  // Count the total number of matches for debugging purposes
          }                      // End conditional: if ( Hit.Station() == 1

          // Match hit in station 2
          conv_vals_SP = convert_SP_location(SP_.ME2_CSC_ID(), (res->at(iOut)).PtrEventHeader()->Sector(), -99, 2);

          if (Hit.Station() == 2 && Hit.Sector() == conv_vals_SP.at(1) && Hit.Neighbor() == conv_vals_SP.at(3) &&
              Hit.Stub_num() == SP_.ME2_stub_num()) {
            if (Hit.Is_CSC() == 1 && Hit.CSC_ID() != conv_vals_SP.at(0))
              continue;

            int tmp_subsector = get_subsector_rpc_cppf(Hit.Subsector_RPC());
            if (Hit.Is_RPC() == 1 && tmp_subsector + 3 != conv_vals_SP.at(0))
              continue;

            if (St_hits.at(1) == 0) {
              Track_.push_Hit((Hit));
              mu_.setTrackSubAddress(RegionalMuonCand::kME2Seg, SP_.ME2_stub_num());
              mu_.setTrackSubAddress(
                  RegionalMuonCand::kME2Ch,
                  L1TMuonEndCap::calc_uGMT_chamber(conv_vals_SP.at(0), conv_vals_SP.at(2), conv_vals_SP.at(3), 2));
            }
            St_hits.at(1) += 1;
          }  // End conditional: if ( Hit.Station() == 2

          // Match hit in station 3
          conv_vals_SP = convert_SP_location(SP_.ME3_CSC_ID(), (res->at(iOut)).PtrEventHeader()->Sector(), -99, 3);

          if (Hit.Station() == 3 && Hit.Sector() == conv_vals_SP.at(1) && Hit.Neighbor() == conv_vals_SP.at(3) &&
              Hit.Stub_num() == SP_.ME3_stub_num()) {
            if (Hit.Is_CSC() == 1 && Hit.CSC_ID() != conv_vals_SP.at(0))
              continue;

            int tmp_subsector = get_subsector_rpc_cppf(Hit.Subsector_RPC());
            if (Hit.Is_RPC() == 1 && tmp_subsector + 3 != conv_vals_SP.at(0))
              continue;

            if (St_hits.at(2) == 0) {
              Track_.push_Hit((Hit));
              mu_.setTrackSubAddress(RegionalMuonCand::kME3Seg, SP_.ME3_stub_num());
              mu_.setTrackSubAddress(
                  RegionalMuonCand::kME3Ch,
                  L1TMuonEndCap::calc_uGMT_chamber(conv_vals_SP.at(0), conv_vals_SP.at(2), conv_vals_SP.at(3), 3));
            }
            St_hits.at(2) += 1;
          }  // End conditional: if ( Hit.Station() == 3

          // Match hit in station 4
          conv_vals_SP = convert_SP_location(SP_.ME4_CSC_ID(), (res->at(iOut)).PtrEventHeader()->Sector(), -99, 4);

          if (Hit.Station() == 4 && Hit.Sector() == conv_vals_SP.at(1) && Hit.Neighbor() == conv_vals_SP.at(3) &&
              Hit.Stub_num() == SP_.ME4_stub_num()) {
            if (Hit.Is_CSC() == 1 && Hit.CSC_ID() != conv_vals_SP.at(0))
              continue;

            int tmp_subsector = get_subsector_rpc_cppf(Hit.Subsector_RPC());
            if (Hit.Is_RPC() == 1 && tmp_subsector + 3 != conv_vals_SP.at(0))
              continue;

            if (St_hits.at(3) == 0) {
              Track_.push_Hit((Hit));
              mu_.setTrackSubAddress(RegionalMuonCand::kME4Seg, SP_.ME4_stub_num());
              mu_.setTrackSubAddress(
                  RegionalMuonCand::kME4Ch,
                  L1TMuonEndCap::calc_uGMT_chamber(conv_vals_SP.at(0), conv_vals_SP.at(2), conv_vals_SP.at(3), 4));
            }
            St_hits.at(3) += 1;
          }  // End conditional: if ( Hit.Station() == 4

        }  // End loop: for (auto const & Hit : *res_hit)

        // Special configuration for single-stub tracks from ME1/1
        if (Track_.Mode() == 1) {
          // Infer ME1/1 chamber based on track phi
          int chamber_min = ((Track_.GMT_phi() - 17) / 16) + Track_.Sector() * 6 - 3;
          int chamber_max = ((Track_.GMT_phi() + 1) / 16) + Track_.Sector() * 6 - 3;
          for (int iChamb = chamber_max; iChamb >= chamber_min; iChamb--) {
            int chamber = (iChamb < 37 ? iChamb : (iChamb % 36));

            for (auto const& Hit : *res_hit) {
              if (Hit.Sector_idx() != Track_.Sector_idx())
                continue;
              if (Hit.BX() != Track_.BX())
                continue;
              if (Hit.Chamber() != chamber)
                continue;
              if (Hit.Is_CSC() != 1)
                continue;
              if (Hit.Station() != 1)
                continue;
              if ((Hit.Ring() % 3) != 1)
                continue;
              if (Hit.Neighbor() == 1)
                continue;

              // Don't use LCTs that were already used in a multi-station track
              bool hit_already_used = false;
              for (auto const& Trk : *res_track) {
                if (Trk.Sector_idx() != Track_.Sector_idx())
                  continue;
                if (Trk.NumHits() < 1)
                  continue;

                if (Trk.Hits().at(0).Station() == 1 && Trk.Hits().at(0).Chamber() == chamber &&
                    Trk.Hits().at(0).BX() == Hit.BX() && Trk.Hits().at(0).Ring() == Hit.Ring() &&
                    Trk.Hits().at(0).Strip() == Hit.Strip() && Trk.Hits().at(0).Wire() == Hit.Wire()) {
                  hit_already_used = true;
                  break;
                }
              }  // End loop: for (auto const & Trk : *res_track)

              if (!hit_already_used) {
                Track_.push_Hit((Hit));
                break;
              }
            }  // End loop: for (auto const & Hit : *res_hit)
            if (Track_.NumHits() > 0)
              break;
          }  // End loop: for (int iChamb = chamber_max; iChamb >= chamber_min; iChamb--)

          // if (Track_.NumHits() != 1) {
          //   std::cout << "\n\n***********************************************************" << std::endl;
          //   std::cout << "Bug in unpacked EMTF event! Mode " << Track_.Mode() << " track in sector " << Track_.Sector()*Track_.Endcap()
          // 	      << ", BX " << Track_.BX() << ", GMT phi " << Track_.GMT_phi() << ", GMT eta " << Track_.GMT_eta()
          // 	      << " should have found an LCT between chamber " << chamber_min << " and " << chamber_max << std::endl;
          //   std::cout << "All available LCTs as follows:" << std::endl;
          //   for (auto const & Hit : *res_hit) {
          //     std::cout << "Hit: Is CSC = " << Hit.Is_CSC() << ", CSC ID = " << Hit.CSC_ID()
          // 		<< ", sector = " << Hit.Sector()*Hit.Endcap() << ", sub = " << Hit.Subsector()
          // 		<< ", neighbor = " << Hit.Neighbor() << ", station = " << Hit.Station()
          // 		<< ", ring = " << Hit.Ring() << ", chamber = " << Hit.Chamber()
          // 		<< ", stub = " << Hit.Stub_num() << ", BX = " << Hit.BX() << std::endl;
          //   std::cout << "All other tracks are as follows:" << std::endl;
          //   for (auto Trk = res_hit->begin(); Trk != res_hit->end(); ++Trk) {
          //     std::cout << "Track: mode " << Trk.Mode() << " track in sector " << Trk.Sector()*Trk.Endcap()
          // 		<< ", BX " << Trk.BX() << ", GMT phi " << Trk.GMT_phi() << ", GMT eta " << Trk.GMT_eta() << std::endl;
          //   }
          //   std::cout << "***********************************************************\n\n" << std::endl;
          // } // End conditional: if (Track_.NumHits() != 1)

        }  // End conditional: if (Track_.Mode() == 1)

        // if ( Track_.Mode() != St_hits.at(0)*8 + St_hits.at(1)*4 + St_hits.at(2)*2 + St_hits.at(3) && Track_.BX() == 0) {
        //   std::cout << "\n\n***********************************************************" << std::endl;
        //   std::cout << "Bug in unpacked EMTF event! Mode " << Track_.Mode() << " track in sector " << Track_.Sector()*Track_.Endcap()
        // 	    << ", BX " << Track_.BX() << " (delay = " << trk_delay << ") with (" << St_hits.at(0) << ", " << St_hits.at(1)
        // 	    << ", " << St_hits.at(2) << ", " << St_hits.at(3) << ") hits in stations (1, 2, 3, 4)" << std::endl;

        //   std::cout << "\nME1_stub_num = " << SP_.ME1_stub_num() << ", ME1_delay = " <<  SP_.ME1_delay()
        // 	    << ", ME1_CSC_ID = " << SP_.ME1_CSC_ID() <<  ", ME1_subsector = " << SP_.ME1_subsector() << std::endl;
        //   std::cout << "ME2_stub_num = " << SP_.ME2_stub_num() << ", ME2_delay = " <<  SP_.ME2_delay()
        // 	    << ", ME2_CSC_ID = " << SP_.ME2_CSC_ID() << std::endl;
        //   std::cout << "ME3_stub_num = " << SP_.ME3_stub_num() << ", ME3_delay = " <<  SP_.ME3_delay()
        // 	    << ", ME3_CSC_ID = " << SP_.ME3_CSC_ID() << std::endl;
        //   std::cout << "ME4_stub_num = " << SP_.ME4_stub_num() << ", ME4_delay = " <<  SP_.ME4_delay()
        // 	    << ", ME4_CSC_ID = " << SP_.ME4_CSC_ID() << std::endl;

        //   conv_vals_SP = convert_SP_location( SP_.ME1_CSC_ID(), (res->at(iOut)).PtrEventHeader()->Sector(), SP_.ME1_subsector(), 1 );
        //   std::cout << "\nConverted ME1 CSC ID = " << conv_vals_SP.at(0) << ", sector = " << conv_vals_SP.at(1)
        // 	    << ", subsector = " << conv_vals_SP.at(2) << ", neighbor = " << conv_vals_SP.at(3) << std::endl;
        //   conv_vals_SP = convert_SP_location( SP_.ME2_CSC_ID(), (res->at(iOut)).PtrEventHeader()->Sector(), -99, 2 );
        //   std::cout << "Converted ME2 CSC ID = " << conv_vals_SP.at(0) << ", sector = " << conv_vals_SP.at(1)
        // 	    << ", subsector = " << conv_vals_SP.at(2) << ", neighbor = " << conv_vals_SP.at(3) << std::endl;
        //   conv_vals_SP = convert_SP_location( SP_.ME3_CSC_ID(), (res->at(iOut)).PtrEventHeader()->Sector(), -99, 3 );
        //   std::cout << "Converted ME3 CSC ID = " << conv_vals_SP.at(0) << ", sector = " << conv_vals_SP.at(1)
        // 	    << ", subsector = " << conv_vals_SP.at(2) << ", neighbor = " << conv_vals_SP.at(3) << std::endl;
        //   conv_vals_SP = convert_SP_location( SP_.ME4_CSC_ID(), (res->at(iOut)).PtrEventHeader()->Sector(), -99, 4 );
        //   std::cout << "Converted ME4 CSC ID = " << conv_vals_SP.at(0) << ", sector = " << conv_vals_SP.at(1)
        // 	    << ", subsector = " << conv_vals_SP.at(2) << ", neighbor = " << conv_vals_SP.at(3) << "\n" << std::endl;

        //   for (auto const & Hit : *res_hit)
        //     std::cout << "Hit: Is CSC = " << Hit.Is_CSC() << ", CSC ID = " << Hit.CSC_ID()
        // 	      << ", sector = " << Hit.Sector() << ", sub = " << Hit.Subsector()
        // 	      << ", neighbor = " << Hit.Neighbor() << ", station = " << Hit.Station()
        // 	      << ", ring = " << Hit.Ring() << ", chamber = " << Hit.Chamber()
        // 	      << ", stub = " << Hit.Stub_num() << ", BX = " << Hit.BX() << std::endl;

        //   // int iHit = 0;
        //   // for (auto const & Hit : *res_hit) {
        //   //   if (iHit == 0) Hit.PrintSimulatorHeader();
        //   //   Hit.PrintForSimulator();
        //   //   iHit += 1;
        //   // }
        //   std::cout << "***********************************************************\n\n" << std::endl;
        // }

        // Reject tracks with out-of-range BX values. This needs to be adjusted if we increase l1a_window parameter in EMTF config - EY 03.08.2022
        if (Track_.BX() > 3 or Track_.BX() < -3) {
          edm::LogWarning("L1T|EMTF") << "EMTF unpacked track with out-of-range BX! BX: " << Track_.BX()
                                      << " endcap: " << (Track_.Endcap() == 1 ? 1 : 2) << " sector: " << Track_.Sector()
                                      << " address: " << Track_.PtLUT().address << " mode: " << Track_.Mode()
                                      << " eta: " << (Track_.GMT_eta() >= 0 ? Track_.GMT_eta() : Track_.GMT_eta() + 512)
                                      << " phi: " << Track_.GMT_phi() << " charge: " << Track_.GMT_charge()
                                      << " qual: " << Track_.GMT_quality() << " pt: " << Track_.Pt()
                                      << " pt_dxy: " << Track_.Pt_dxy() << std::endl;
          return true;
        }

        (res->at(iOut)).push_SP(SP_);

        res_track->push_back(Track_);

        if (Track_.Mode() != 0) {  // Mode == 0 means no track was found (only muon shower)
          // TBIN_num can range from 0 through 7, i.e. BX = -3 through +4. - AWB 04.04.16
          res_cand->setBXRange(-3, 4);
          res_cand->push_back(SP_.TBIN() - 3, mu_);
        }

        res_shower->setBXRange(-3, 4);
        res_shower->push_back(SP_.TBIN() - 3, muShower_);

        // Finished with unpacking one SP Output Data Record
        return true;

      }  // End bool SPBlockUnpacker::unpack

      // bool SPBlockPacker::pack(const Block& block, UnpackerCollections *coll) {
      // 	std::cout << "Inside SPBlockPacker::pack" << std::endl;
      // 	return true;
      // } // End bool SPBlockPacker::pack

    }  // End namespace emtf
  }    // End namespace stage2
}  // End namespace l1t

DEFINE_L1T_UNPACKER(l1t::stage2::emtf::SPBlockUnpacker);
// DEFINE_L1T_PACKER(l1t::stage2::SPBlockPacker);
