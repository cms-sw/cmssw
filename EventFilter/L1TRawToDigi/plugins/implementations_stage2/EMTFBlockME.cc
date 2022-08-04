// Code to unpack the "ME Data Record"

#include "EventFilter/L1TRawToDigi/plugins/UnpackerFactory.h"

#include "EMTFCollections.h"
#include "EMTFUnpackerTools.h"

// This is the "header" - no EMTFBlockME.h file is needed
namespace l1t {
  namespace stage2 {
    namespace emtf {

      class MEBlockUnpacker : public Unpacker {  // "MEBlockUnpacker" inherits from "Unpacker"
      public:
        virtual int checkFormat(const Block& block);
        bool unpack(const Block& block,
                    UnpackerCollections* coll) override;  // Apparently it's always good to use override in C++
        // virtual bool packBlock(const Block& block, UnpackerCollections *coll) override;
      };

      // class MEBlockPacker : public Packer { // "MEBlockPacker" inherits from "Packer"
      // public:
      // 	virtual bool unpack(const Block& block, UnpackerCollections *coll) override; // Apparently it's always good to use override in C++
      // };

    }  // namespace emtf
  }    // namespace stage2
}  // namespace l1t

namespace l1t {
  namespace stage2 {
    namespace emtf {

      int MEBlockUnpacker::checkFormat(const Block& block) {
        auto payload = block.payload();
        int errors = 0;

        // Check the number of 16-bit words
        if (payload.size() != 4) {
          errors += 1;
          edm::LogError("L1T|EMTF") << "Payload size in 'ME Data Record' is different than expected";
        }

        // Check that each word is 16 bits
        for (unsigned int i = 0; i < 4; i++) {
          if (GetHexBits(payload[i], 16, 31) != 0) {
            errors += 1;
            edm::LogError("L1T|EMTF") << "Payload[" << i << "] has more than 16 bits in 'ME Data Record'";
          }
        }

        uint16_t MEa = payload[0];
        uint16_t MEb = payload[1];
        uint16_t MEc = payload[2];
        uint16_t MEd = payload[3];

        //Check Format
        if (GetHexBits(MEa, 15, 15) != 1) {
          errors += 1;
          edm::LogError("L1T|EMTF") << "Format identifier bits in MEa are incorrect";
        }
        if (GetHexBits(MEb, 15, 15) != 1) {
          errors += 1;
          edm::LogError("L1T|EMTF") << "Format identifier bits in MEb are incorrect";
        }
        if (GetHexBits(MEc, 15, 15) != 0) {
          errors += 1;
          edm::LogError("L1T|EMTF") << "Format identifier bits in MEc are incorrect";
        }
        if (GetHexBits(MEd, 15, 15) != 0) {
          errors += 1;
          edm::LogError("L1T|EMTF") << "Format identifier bits in MEd are incorrect";
        }

        return errors;
      }

      // Converts station, CSC_ID, sector, subsector, and neighbor from the ME output
      std::vector<int> convert_ME_location(int _station, int _csc_ID, int _sector, bool _csc_ID_shift = false) {
        int new_sector = _sector;
        int new_csc_ID = _csc_ID;
        if (_csc_ID_shift)
          new_csc_ID += 1;  // Before FW update on 05.05.16, shift by +1 from 0,1,2... convention to 1,2,3...
        if (_station == 0) {
          int arr[] = {1, new_csc_ID, new_sector, 1, 0};
          std::vector<int> vec(arr, arr + 5);
          return vec;
        } else if (_station == 1) {
          int arr[] = {1, new_csc_ID, new_sector, 2, 0};
          std::vector<int> vec(arr, arr + 5);
          return vec;
        } else if (_station <= 4) {
          int arr[] = {_station, new_csc_ID, new_sector, -1, 0};
          std::vector<int> vec(arr, arr + 5);
          return vec;
        } else if (_station == 5) {
          new_sector = (_sector != 1) ? _sector - 1 : 6;  // Indicates neighbor chamber, don't return yet
        } else {
          int arr[] = {_station, _csc_ID, _sector, -99, -99};
          std::vector<int> vec(arr, arr + 5);
          return vec;
        }

        // Mapping for chambers from neighboring sector
        if (new_csc_ID == 1) {
          int arr[] = {1, 3, new_sector, 2, 1};
          std::vector<int> vec(arr, arr + 5);
          return vec;
        } else if (new_csc_ID == 2) {
          int arr[] = {1, 6, new_sector, 2, 1};
          std::vector<int> vec(arr, arr + 5);
          return vec;
        } else if (new_csc_ID == 3) {
          int arr[] = {1, 9, new_sector, 2, 1};
          std::vector<int> vec(arr, arr + 5);
          return vec;
        } else if (new_csc_ID == 4) {
          int arr[] = {2, 3, new_sector, -1, 1};
          std::vector<int> vec(arr, arr + 5);
          return vec;
        } else if (new_csc_ID == 5) {
          int arr[] = {2, 9, new_sector, -1, 1};
          std::vector<int> vec(arr, arr + 5);
          return vec;
        } else if (new_csc_ID == 6) {
          int arr[] = {3, 3, new_sector, -1, 1};
          std::vector<int> vec(arr, arr + 5);
          return vec;
        } else if (new_csc_ID == 7) {
          int arr[] = {3, 9, new_sector, -1, 1};
          std::vector<int> vec(arr, arr + 5);
          return vec;
        } else if (new_csc_ID == 8) {
          int arr[] = {4, 3, new_sector, -1, 1};
          std::vector<int> vec(arr, arr + 5);
          return vec;
        } else if (new_csc_ID == 9) {
          int arr[] = {4, 9, new_sector, -1, 1};
          std::vector<int> vec(arr, arr + 5);
          return vec;
        } else {
          int arr[] = {_station, _csc_ID, _sector, -99, -99};
          std::vector<int> vec(arr, arr + 5);
          return vec;
        }
      }

      bool MEBlockUnpacker::unpack(const Block& block, UnpackerCollections* coll) {
        // std::cout << "Inside EMTFBlockME.cc: unpack" << std::endl;

        // Get the payload for this block, made up of 16-bit words (0xffff)
        // Format defined in MTF7Payload::getBlock() in src/Block.cc
        // payload[0] = bits 0-15, payload[1] = 16-31, payload[3] = 32-47, etc.
        auto payload = block.payload();

        // Assign payload to 16-bit words
        uint16_t MEa = payload[0];
        uint16_t MEb = payload[1];
        uint16_t MEc = payload[2];
        uint16_t MEd = payload[3];

        // Check Format of Payload
        l1t::emtf::ME ME_;
        for (int err = 0; err < checkFormat(block); err++)
          ME_.add_format_error();

        // res is a pointer to a collection of EMTFDaqOut class objects
        // There is one EMTFDaqOut for each MTF7 (60 deg. sector) in the event
        EMTFDaqOutCollection* res;
        res = static_cast<EMTFCollections*>(coll)->getEMTFDaqOuts();
        int iOut = res->size() - 1;

        EMTFHitCollection* res_hit;
        res_hit = static_cast<EMTFCollections*>(coll)->getEMTFHits();
        EMTFHit Hit_;

        CSCCorrelatedLCTDigiCollection* res_LCT;
        res_LCT = static_cast<EMTFCollections*>(coll)->getEMTFLCTs();

        CSCShowerDigiCollection* res_shower;
        res_shower = static_cast<EMTFCollections*>(coll)->getEMTFCSCShowers();

        ////////////////////////////
        // Unpack the ME Data Record
        ////////////////////////////

        // Run 3 has a different EMTF DAQ output format
        // Computed as (Year - 2000)*2^9 + Month*2^5 + Day (see Block.cc and EMTFBlockTrailers.cc)
        bool run3_DAQ_format =
            (getAlgoVersion() >= 11460);  // Firmware from 04.06.22 which enabled new Run 3 DAQ format - EY 04.07.22

        // Set fields assuming Run 2 format. Modify for Run 3 later
        ME_.set_clct_pattern(GetHexBits(MEa, 0, 3));
        ME_.set_quality(GetHexBits(MEa, 4, 7));
        ME_.set_wire(GetHexBits(MEa, 8, 14));

        ME_.set_strip(GetHexBits(MEb, 0, 7));
        ME_.set_csc_ID(GetHexBits(MEb, 8, 11));
        ME_.set_lr(GetHexBits(MEb, 12, 12));
        ME_.set_bxe(GetHexBits(MEb, 13, 13));
        ME_.set_bc0(GetHexBits(MEb, 14, 14));

        ME_.set_me_bxn(GetHexBits(MEc, 0, 11));
        ME_.set_nit(GetHexBits(MEc, 12, 12));
        ME_.set_cik(GetHexBits(MEc, 13, 13));
        ME_.set_afff(GetHexBits(MEc, 14, 14));

        ME_.set_tbin(GetHexBits(MEd, 0, 2));
        ME_.set_vp(GetHexBits(MEd, 3, 3));
        ME_.set_station(GetHexBits(MEd, 4, 6));
        ME_.set_af(GetHexBits(MEd, 7, 7));
        ME_.set_epc(GetHexBits(MEd, 8, 11));
        ME_.set_sm(GetHexBits(MEd, 12, 12));
        ME_.set_se(GetHexBits(MEd, 13, 13));
        ME_.set_afef(GetHexBits(MEd, 14, 14));

        // ME_.set_dataword     ( uint64_t dataword);

        // Convert specially-encoded ME quantities
        bool csc_ID_shift = (getAlgoVersion() <=
                             8348);  // For FW versions <= 28.04.2016, shift by +1 from 0,1,2... convention to 1,2,3...
        // Computed as (Year - 2000)*2^9 + Month*2^5 + Day (see Block.cc and EMTFBlockTrailers.cc)
        std::vector<int> conv_vals =
            convert_ME_location(ME_.Station(), ME_.CSC_ID(), (res->at(iOut)).PtrEventHeader()->Sector(), csc_ID_shift);

        Hit_.set_station(conv_vals.at(0));
        Hit_.set_csc_ID(conv_vals.at(1));
        Hit_.set_sector(conv_vals.at(2));
        Hit_.set_subsector(conv_vals.at(3));
        Hit_.set_neighbor(conv_vals.at(4));
        Hit_.set_ring(L1TMuonEndCap::calc_ring(Hit_.Station(), Hit_.CSC_ID(), ME_.Strip()));

        if (Hit_.Station() < 1 || Hit_.Station() > 4)
          edm::LogWarning("L1T|EMTF") << "EMTF unpacked LCT station = " << Hit_.Station()
                                      << ", outside proper [1, 4] range" << std::endl;
        if (Hit_.CSC_ID() < 1 || Hit_.CSC_ID() > 9)
          edm::LogWarning("L1T|EMTF") << "EMTF unpacked LCT CSC ID = " << Hit_.CSC_ID()
                                      << ", outside proper [1, 9] range" << std::endl;
        if (Hit_.Sector() < 1 || Hit_.Sector() > 6)
          edm::LogWarning("L1T|EMTF") << "EMTF unpacked LCT sector = " << Hit_.Sector()
                                      << ", outside proper [1, 6] range" << std::endl;

        // Modifications for Run 3 format - EY 04.07.22
        bool isOTMB = (Hit_.Ring() == 1 or
                       Hit_.Ring() == 4);  // Data format is different between OTMBs (MEX/1) and TMBs (MEX/2-3)

        bool isRun3 =
            isOTMB and run3_DAQ_format;  // in Run3 DAQ format, OTMB TPs are Run 3 CSC TPs with CCLUT algorithm

        if (run3_DAQ_format) {
          ME_.set_quality(GetHexBits(MEa, 4, 6));
          ME_.set_quarter_strip(GetHexBits(MEa, 7, 7));

          ME_.set_frame(GetHexBits(MEc, 12, 12));

          ME_.set_eighth_strip(GetHexBits(MEd, 13, 13));

          if (isOTMB) {  // Derive Run 2 pattern ID from Run 3 slope for OTMBs

            ME_.set_slope(GetHexBits(MEd, 8, 11));

            // convert Run-3 slope to Run-2 pattern for CSC TPs coming from MEX/1 chambers
            // where the CCLUT algorithm is enabled
            const unsigned slopeList[32] = {10, 10, 10, 8, 8, 8, 6, 6, 6, 4, 4, 4, 2, 2, 2, 2,
                                            10, 10, 10, 9, 9, 9, 7, 7, 7, 5, 5, 5, 3, 3, 3, 3};

            // this LUT follows the same convention as in CSCPatternBank.cc
            unsigned slope_and_sign(ME_.Slope());
            if (ME_.LR() == 1) {
              slope_and_sign += 16;
            }
            unsigned run2_converted_PID = slopeList[slope_and_sign];

            ME_.set_clct_pattern(run2_converted_PID);

          } else {  // Use Run 2 pattern directly for TMBs
            ME_.set_clct_pattern(GetHexBits(MEd, 8, 11));
          }

          // Frame 1 has HMT related information
          if (ME_.Frame() == 1) {
            // Run 3 pattern is unused for now. Needs to be combined with rest of the word in Frame 0 - EY 04.07.22
            ME_.set_run3_pattern(GetHexBits(MEa, 0, 0));

            // HMT[1] is in MEa, but HMT[0] is in MEb. These encode in time showers - EY 04.07.22
            ME_.set_hmt_inTime(GetHexBits(MEb, 13, 13, MEa, 1, 1));

            // HMT[3:2] encodes out-of-time showers which are not used for now
            ME_.set_hmt_outOfTime(GetHexBits(MEa, 2, 3));

            ME_.set_hmv(GetHexBits(MEd, 7, 7));
          } else {
            ME_.set_run3_pattern(GetHexBits(MEa, 0, 3));

            ME_.set_bxe(GetHexBits(MEb, 13, 13));

            ME_.set_af(GetHexBits(MEd, 7, 7));
          }
        }

        // Fill the EMTFHit
        ImportME(Hit_, ME_, (res->at(iOut)).PtrEventHeader()->Endcap(), (res->at(iOut)).PtrEventHeader()->Sector());

        // Fill the CSCShowerDigi
        CSCShowerDigi Shower_(ME_.HMT_inTime() == -99 ? 0 : ME_.HMT_inTime(),
                              ME_.HMT_outOfTime() == -99 ? 0 : ME_.HMT_outOfTime(),
                              Hit_.CSC_DetId());

        // Set the stub number for this hit
        // Each chamber can send up to 2 stubs per BX
        ME_.set_stub_num(0);
        Hit_.set_stub_num(0);
        // See if matching hit is already in event record: exact duplicate, or from neighboring sector
        bool exact_duplicate = false;
        bool neighbor_duplicate = false;
        for (auto const& iHit : *res_hit) {
          if (iHit.Is_CSC() == 1 && Hit_.BX() == iHit.BX() && Hit_.Endcap() == iHit.Endcap() &&
              Hit_.Station() == iHit.Station() && Hit_.Chamber() == iHit.Chamber() &&
              (Hit_.Ring() % 3) == (iHit.Ring() % 3)) {  // ME1/1a and ME1/1b (rings "4" and 1) are the same chamber

            if (Hit_.Ring() == iHit.Ring() && Hit_.Strip() == iHit.Strip() && Hit_.Wire() == iHit.Wire()) {
              exact_duplicate = (Hit_.Neighbor() == iHit.Neighbor());
              neighbor_duplicate = (Hit_.Neighbor() != iHit.Neighbor());
            } else if (Hit_.Neighbor() == iHit.Neighbor()) {
              ME_.set_stub_num(ME_.Stub_num() + 1);
              Hit_.set_stub_num(Hit_.Stub_num() + 1);
            }
          }
        }  // End loop: for (auto const & iHit : *res_hit)

        if (exact_duplicate)
          edm::LogWarning("L1T|EMTF") << "EMTF unpacked duplicate LCTs: BX " << Hit_.BX() << ", endcap "
                                      << Hit_.Endcap() << ", station " << Hit_.Station() << ", sector " << Hit_.Sector()
                                      << ", neighbor " << Hit_.Neighbor() << ", ring " << Hit_.Ring() << ", chamber "
                                      << Hit_.Chamber() << ", strip " << Hit_.Strip() << ", wire " << Hit_.Wire()
                                      << std::endl;

        (res->at(iOut)).push_ME(ME_);
        if (!exact_duplicate && Hit_.Valid() == 1)
          res_hit->push_back(Hit_);
        if (!exact_duplicate && !neighbor_duplicate &&
            Hit_.Valid() == 1)  // Don't write duplicate LCTs from adjacent sectors
          res_LCT->insertDigi(Hit_.CSC_DetId(), Hit_.CreateCSCCorrelatedLCTDigi(isRun3));
        if (ME_.HMV() == 1) {  // Only write when HMT valid bit is set to 1
          res_shower->insertDigi(Hit_.CSC_DetId(), Shower_);
        }
        // Finished with unpacking one ME Data Record
        return true;

      }  // End bool MEBlockUnpacker::unpack

      // bool MEBlockPacker::pack(const Block& block, UnpackerCollections *coll) {
      // 	std::cout << "Inside MEBlockPacker::pack" << std::endl;
      // 	return true;
      // } // End bool MEBlockPacker::pack

    }  // End namespace emtf
  }    // End namespace stage2
}  // End namespace l1t

DEFINE_L1T_UNPACKER(l1t::stage2::emtf::MEBlockUnpacker);
// DEFINE_L1T_PACKER(l1t::stage2::MEBlockPacker);
