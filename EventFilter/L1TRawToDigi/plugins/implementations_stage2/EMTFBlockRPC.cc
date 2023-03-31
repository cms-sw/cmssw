// Code to unpack the "RPC Data Record"

#include "EventFilter/L1TRawToDigi/plugins/UnpackerFactory.h"

#include "EMTFCollections.h"
#include "EMTFUnpackerTools.h"

// This is the "header" - no EMTFBlockRPC.h file is needed
namespace l1t {
  namespace stage2 {
    namespace emtf {

      class RPCBlockUnpacker : public Unpacker {  // "RPCBlockUnpacker" inherits from "Unpacker"
      public:
        virtual int checkFormat(const Block& block);
        // virtual bool checkFormat() override; // Return "false" if block format does not match expected format
        bool unpack(const Block& block,
                    UnpackerCollections* coll) override;  // Apparently it's always good to use override in C++
        // virtual bool packBlock(const Block& block, UnpackerCollections *coll) override;
      };

      // class RPCBlockPacker : public Packer { // "RPCBlockPacker" inherits from "Packer"
      // public:
      // 	virtual bool unpack(const Block& block, UnpackerCollections *coll) override; // Apparently it's always good to use override in C++
      // };

    }  // namespace emtf
  }    // namespace stage2
}  // namespace l1t

namespace l1t {
  namespace stage2 {
    namespace emtf {

      int RPCBlockUnpacker::checkFormat(const Block& block) {
        auto payload = block.payload();
        int errors = 0;

        // Check the number of 16-bit words
        if (payload.size() != 4) {
          errors += 1;
          edm::LogError("L1T|EMTF") << "Payload size in 'RPC Data Record' is different than expected";
        }

        // Check that each word is 16 bits
        for (unsigned int i = 0; i < 4; i++) {
          if (GetHexBits(payload[i], 16, 31) != 0) {
            errors += 1;
            edm::LogError("L1T|EMTF") << "Payload[" << i << "] has more than 16 bits in 'RPC Data Record'";
          }
        }

        uint16_t RPCa = payload[0];
        uint16_t RPCb = payload[1];
        uint16_t RPCc = payload[2];
        uint16_t RPCd = payload[3];

        // Check Format
        if (GetHexBits(RPCa, 15, 15) != 0) {
          errors += 1;
          edm::LogError("L1T|EMTF") << "Format identifier bits in RPCa are incorrect";
        }
        if (GetHexBits(RPCb, 15, 15) != 0) {
          errors += 1;
          edm::LogError("L1T|EMTF") << "Format identifier bits in RPCb are incorrect";
        }
        if (GetHexBits(RPCc, 15, 15) != 1) {
          errors += 1;
          edm::LogError("L1T|EMTF") << "Format identifier bits in RPCc are incorrect";
        }
        if (GetHexBits(RPCd, 15, 15) != 0) {
          errors += 1;
          edm::LogError("L1T|EMTF") << "Format identifier bits in RPCd are incorrect";
        }

        return errors;
      }

      // Converts station, ring, sector, subsector, neighbor, and segment from the RPC output
      void convert_RPC_location(int& station,
                                int& ring,
                                int& sector,
                                int& subsector,
                                int& neighbor,
                                int& segment,
                                const int evt_sector,
                                const int frame,
                                const int word,
                                const int link) {
        station = -99;
        ring = -99;
        sector = -99;
        subsector = -99;
        neighbor = -99;
        segment = -99;

        // "link" is the "link index" field (0 - 6) in the EMTF DAQ document, not "link number" (1 - 7)
        // Neighbor indicated by link == 0
        sector = (link != 0 ? evt_sector : (evt_sector == 1 ? 6 : evt_sector - 1));
        subsector = (link != 0 ? link : 6);
        neighbor = (link == 0 ? 1 : 0);
        segment = (word % 2);

        if (frame == 0) {
          station = (word < 2 ? 1 : 2);
          ring = 2;
        } else if (frame == 1) {
          station = 3;
          ring = (word < 2 ? 2 : 3);
        } else if (frame == 2) {
          station = 4;
          ring = (word < 2 ? 2 : 3);
        }
      }  // End function: void convert_RPC_location()

      bool RPCBlockUnpacker::unpack(const Block& block, UnpackerCollections* coll) {
        // std::cout << "Inside EMTFBlockRPC.cc: unpack" << std::endl;

        // Get the payload for this block, made up of 16-bit words (0xffff)
        // Format defined in MTF7Payload::getBlock() in src/Block.cc
        // payload[0] = bits 0-15, payload[1] = 16-31, payload[3] = 32-47, etc.
        auto payload = block.payload();

        // Run 3 has a different EMTF DAQ output format since August 26th
        // Computed as (Year - 2000)*2^9 + Month*2^5 + Day (see Block.cc and EMTFBlockTrailers.cc)
        bool run3_DAQ_format =
            (getAlgoVersion() >=
             11546);  // Firmware from 26.08.22 which enabled new Run 3 DAQ format for RPCs - EY 13.09.22
        bool reducedDAQWindow =
            (getAlgoVersion() >=
             11656);  // Firmware from 08.12.22 which is used as a flag for new reduced readout window - EY 01.03.23

        int nTPs = run3_DAQ_format ? 2 : 1;

        // Check Format of Payload
        l1t::emtf::RPC RPC_;
        for (int err = 0; err < checkFormat(block); err++)
          RPC_.add_format_error();

        // Assign payload to 16-bit words
        uint16_t RPCa = payload[0];
        uint16_t RPCb = payload[1];
        uint16_t RPCc = payload[2];
        uint16_t RPCd = payload[3];

        // If there are 2 TPs in the block we fill them 1 by 1
        for (int i = 1; i <= nTPs; i++) {
          // res is a pointer to a collection of EMTFDaqOut class objects
          // There is one EMTFDaqOut for each MTF7 (60 deg. sector) in the event
          EMTFDaqOutCollection* res;
          res = static_cast<EMTFCollections*>(coll)->getEMTFDaqOuts();
          int iOut = res->size() - 1;

          EMTFHitCollection* res_hit;
          res_hit = static_cast<EMTFCollections*>(coll)->getEMTFHits();
          EMTFHit Hit_;

          CPPFDigiCollection* res_CPPF;
          res_CPPF = static_cast<EMTFCollections*>(coll)->getEMTFCPPFs();

          ////////////////////////////
          // Unpack the RPC Data Record
          ////////////////////////////

          if (run3_DAQ_format) {  // Run 3 DAQ format has 2 TPs per block
            if (i == 1) {
              RPC_.set_phi(GetHexBits(RPCa, 0, 10));
              RPC_.set_word(GetHexBits(RPCa, 11, 12));
              RPC_.set_frame(GetHexBits(RPCa, 13, 14));

              if (reducedDAQWindow)  // reduced DAQ window is used only after run3 DAQ format
                RPC_.set_tbin(GetHexBits(RPCb, 0, 2) + 1);
              else
                RPC_.set_tbin(GetHexBits(RPCb, 0, 2));
              RPC_.set_vp(GetHexBits(RPCb, 3, 3));
              RPC_.set_theta(GetHexBits(RPCb, 4, 8));
              RPC_.set_bc0(GetHexBits(RPCb, 9, 9));
              RPC_.set_link(GetHexBits(RPCb, 12, 14));  // Link index (0 - 6); link number runs 1 - 7
            } else if (i == 2) {
              RPC_.set_phi(GetHexBits(RPCc, 0, 10));
              RPC_.set_word(GetHexBits(RPCc, 11, 12));
              RPC_.set_frame(GetHexBits(RPCc, 13, 14));

              if (reducedDAQWindow)  // reduced DAQ window is used only after run3 DAQ format
                RPC_.set_tbin(GetHexBits(RPCd, 0, 2) + 1);
              else
                RPC_.set_tbin(GetHexBits(RPCd, 0, 2));
              RPC_.set_vp(GetHexBits(RPCd, 3, 3));
              RPC_.set_theta(GetHexBits(RPCd, 4, 8));
              RPC_.set_bc0(GetHexBits(RPCd, 9, 9));
              RPC_.set_link(GetHexBits(RPCd, 12, 14));  // Link index (0 - 6); link number runs 1 - 7
            }
          } else {  // Run 2 DAQ format
            RPC_.set_phi(GetHexBits(RPCa, 0, 10));

            RPC_.set_theta(GetHexBits(RPCb, 0, 4));
            RPC_.set_word(GetHexBits(RPCb, 8, 9));
            RPC_.set_frame(GetHexBits(RPCb, 10, 11));
            RPC_.set_link(GetHexBits(RPCb, 12, 14));  // Link index (0 - 6); link number runs 1 - 7

            RPC_.set_rpc_bxn(GetHexBits(RPCc, 0, 11));
            RPC_.set_bc0(GetHexBits(RPCc, 14, 14));

            RPC_.set_tbin(GetHexBits(RPCd, 0, 2));
            RPC_.set_vp(GetHexBits(RPCd, 3, 3));

            // RPC_.set_dataword            ( uint64_t dataword);
          }

          // Convert specially-encoded RPC quantities
          int _station, _ring, _sector, _subsector, _neighbor, _segment;
          convert_RPC_location(_station,
                               _ring,
                               _sector,
                               _subsector,
                               _neighbor,
                               _segment,
                               (res->at(iOut)).PtrEventHeader()->Sector(),
                               RPC_.Frame(),
                               RPC_.Word(),
                               RPC_.Link());

          // Rotate by 20 deg to match RPC convention in CMSSW
          int _sector_rpc = (_subsector < 5) ? _sector : (_sector % 6) + 1;
          // Rotate by 2 to match RPC convention in CMSSW (RPCDetId.h)
          int _subsector_rpc = ((_subsector + 1) % 6) + 1;
          // Define chamber number
          int _chamber = (_sector_rpc - 1) * 6 + _subsector_rpc;
          // Define CSC-like subsector
          int _subsector_csc = (_station != 1) ? 0 : ((_chamber % 6 > 2) ? 1 : 2);

          Hit_.set_station(_station);
          Hit_.set_ring(_ring);
          Hit_.set_sector(_sector);
          Hit_.set_subsector(_subsector_csc);
          Hit_.set_sector_RPC(_sector_rpc);
          Hit_.set_subsector_RPC(_subsector_rpc);
          Hit_.set_chamber(_chamber);
          Hit_.set_neighbor(_neighbor);
          Hit_.set_pc_segment(_segment);
          Hit_.set_fs_segment(_segment);
          Hit_.set_bt_segment(_segment);

          // Fill the EMTFHit
          ImportRPC(Hit_, RPC_, (res->at(iOut)).PtrEventHeader()->Endcap(), (res->at(iOut)).PtrEventHeader()->Sector());

          // Set the stub number for this hit
          // Each chamber can send up to 2 stubs per BX
          // Also count stubs in corresponding CSC chamber; RPC hit counting is on top of LCT counting
          Hit_.set_stub_num(0);
          // See if matching hit is already in event record
          bool exact_duplicate = false;
          for (auto const& iHit : *res_hit) {
            if (Hit_.BX() == iHit.BX() && Hit_.Endcap() == iHit.Endcap() && Hit_.Station() == iHit.Station() &&
                Hit_.Chamber() == iHit.Chamber()) {
              if ((iHit.Is_CSC() == 1 && iHit.Ring() == 2) ||
                  (iHit.Is_RPC() == 1)) {  // RPC rings 2 and 3 both map to CSC ring 2
                if (Hit_.Neighbor() == iHit.Neighbor()) {
                  Hit_.set_stub_num(Hit_.Stub_num() + 1);
                  if (iHit.Is_RPC() == 1 && iHit.Ring() == Hit_.Ring() && iHit.Theta_fp() == Hit_.Theta_fp() &&
                      iHit.Phi_fp() == Hit_.Phi_fp()) {
                    exact_duplicate = true;
                  }
                }
              }
            }
          }  // End loop: for (auto const & iHit : *res_hit)

          // Reject TPs with out-of-range BX values. This needs to be adjusted if we increase l1a_window parameter in EMTF config - EY 03.08.2022
          if (Hit_.BX() > 3 or Hit_.BX() < -3) {
            edm::LogWarning("L1T|EMTF") << "EMTF unpacked CPPF digis with out-of-range BX! BX " << Hit_.BX()
                                        << ", endcap " << Hit_.Endcap() << ", station " << Hit_.Station() << ", sector "
                                        << Hit_.Sector() << ", neighbor " << Hit_.Neighbor() << ", ring " << Hit_.Ring()
                                        << ", chamber " << Hit_.Chamber() << ", theta " << Hit_.Theta_fp() / 4
                                        << ", phi " << Hit_.Phi_fp() / 4 << std::endl;
            return true;
          }

          if (exact_duplicate)
            edm::LogWarning("L1T|EMTF") << "EMTF unpacked duplicate CPPF digis: BX " << Hit_.BX() << ", endcap "
                                        << Hit_.Endcap() << ", station " << Hit_.Station() << ", sector "
                                        << Hit_.Sector() << ", neighbor " << Hit_.Neighbor() << ", ring " << Hit_.Ring()
                                        << ", chamber " << Hit_.Chamber() << ", theta " << Hit_.Theta_fp() / 4
                                        << ", phi " << Hit_.Phi_fp() / 4 << std::endl;

          (res->at(iOut)).push_RPC(RPC_);
          if (!exact_duplicate and Hit_.Valid())
            res_hit->push_back(Hit_);
          if (!exact_duplicate and Hit_.Valid())
            res_CPPF->push_back(Hit_.CreateCPPFDigi());
        }

        // Finished with unpacking one RPC Data Record
        return true;

      }  // End bool RPCBlockUnpacker::unpack

      // bool RPCBlockPacker::pack(const Block& block, UnpackerCollections *coll) {
      // 	std::cout << "Inside RPCBlockPacker::pack" << std::endl;
      // 	return true;
      // } // End bool RPCBlockPacker::pack

    }  // End namespace emtf
  }    // End namespace stage2
}  // End namespace l1t

DEFINE_L1T_UNPACKER(l1t::stage2::emtf::RPCBlockUnpacker);
// DEFINE_L1T_PACKER(l1t::stage2::RPCBlockPacker);
