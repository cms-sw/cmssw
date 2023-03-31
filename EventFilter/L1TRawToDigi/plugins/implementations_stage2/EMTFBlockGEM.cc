// Code to unpack the "GEM Data Record"

#include "EventFilter/L1TRawToDigi/plugins/UnpackerFactory.h"

#include "EMTFCollections.h"
#include "EMTFUnpackerTools.h"

namespace l1t {
  namespace stage2 {
    namespace emtf {

      class GEMBlockUnpacker : public Unpacker {
      public:
        virtual int checkFormat(const Block& block);
        // virtual bool checkFormat() override; // Return "false" if block format does not match expected format
        bool unpack(const Block& block, UnpackerCollections* coll) override;
        // virtual bool packBlock(const Block& block, UnpackerCollections *coll) override;
      };

      // class GEMBlockPacker : public Packer {
      // public:
      // 	virtual bool unpack(const Block& block, UnpackerCollections *coll) override;
      // };

    }  // namespace emtf
  }    // namespace stage2
}  // namespace l1t

namespace l1t {
  namespace stage2 {
    namespace emtf {

      int GEMBlockUnpacker::checkFormat(const Block& block) {
        auto payload = block.payload();
        int errors = 0;

        // Check the number of 16-bit words
        if (payload.size() != 4) {
          errors += 1;
          edm::LogError("L1T|EMTF") << "Payload size in 'GEM Data Record' is different than expected";
        }

        // Check that each word is 16 bits
        for (size_t i = 0; i < 4; ++i) {
          if (GetHexBits(payload[i], 16, 31) != 0) {
            errors += 1;
            edm::LogError("L1T|EMTF") << "Payload[" << i << "] has more than 16 bits in 'GEM Data Record'";
          }
        }

        uint16_t GEMa = payload[0];
        uint16_t GEMb = payload[1];
        uint16_t GEMc = payload[2];
        uint16_t GEMd = payload[3];

        // Check Format
        if (GetHexBits(GEMa, 15, 15) != 1) {
          errors += 1;
          edm::LogError("L1T|EMTF") << "Format identifier bits in GEMa are incorrect";
        }
        if (GetHexBits(GEMb, 15, 15) != 1) {
          errors += 1;
          edm::LogError("L1T|EMTF") << "Format identifier bits in GEMb are incorrect";
        }
        if (GetHexBits(GEMc, 15, 15) != 1) {
          errors += 1;
          edm::LogError("L1T|EMTF") << "Format identifier bits in GEMc are incorrect";
        }
        if (GetHexBits(GEMd, 15, 15) != 0) {
          errors += 1;
          edm::LogError("L1T|EMTF") << "Format identifier bits in GEMd are incorrect";
        }

        return errors;
      }

      /**
       * \brief Converts station, ring, sector, subsector, neighbor, and segment from the GEM output
       * \param station
       * \param ring
       * \param sector
       * \param subsector
       * \param neighbor
       * \param segment
       * \param evt_sector
       * \param frame
       * \param word
       * \param link is the "link index" field (0 - 6) in the EMTF DAQ document, not "link number" (1 - 7)
      */
      void convert_GEM_location(int& station,
                                int& ring,
                                int& sector,
                                int& subsector,
                                int& neighbor,
                                int& layer,  // is this correct for the GEM case?
                                const int evt_sector,
                                const int cluster_id,  // used to differentiate between GEM layer 1/2
                                const int link) {
        station =
            1;  // station is not encoded in the GEM frame for now. Set station = 1 since we only have GE1/1 for Run 3.
        ring = 1;  // GEMs are only in GE1/1 and GE2/1
        sector = -99;
        subsector = -99;
        neighbor = -99;
        layer = -99;  // the GEM layer is 1 or 2, depending on the cluster ID

        // Neighbor indicated by link == 6
        sector = (link != 6 ? evt_sector : (evt_sector == 1 ? 6 : evt_sector - 1));
        subsector = (link != 6 ? link : 0);  // TODO: verify subsector 0 in the neighbouring sector?
        neighbor = (link == 6 ? 1 : 0);  // TODO: verify that 6 is for the neighbour, not 0 (as written in EMTFBlockRPC)
        layer = (cluster_id % 8);        // + 1 if layer should be 1 or 2, otherwise layer is 0 or 1
      }

      /**
       * \brief Unpack the GEM payload in the EMTF DAQ payload
       *
       * The GEM payload consists of up to 14 clusters per link (for the two GEM layers)
       * 7 links (including neighbour).
       * The EMTF firmware packs each cluster word into one 64-bit EMTF record, and
       * that is what is unpacked here.
       */
      bool GEMBlockUnpacker::unpack(const Block& block, UnpackerCollections* coll) {
        // std::cout << "Inside EMTFBlockGEM.cc: unpack" << std::endl;

        // Get the payload for this block, made up of 16-bit words (0xffff)
        // Format defined in MTF7Payload::getBlock() in src/Block.cc
        // payload[0] = bits 0-15, payload[1] = 16-31, payload[3] = 32-47, etc.
        auto payload = block.payload();

        // Run 3 has a different EMTF DAQ output format since August 26th
        // Computed as (Year - 2000)*2^9 + Month*2^5 + Day (see Block.cc and EMTFBlockTrailers.cc)
        bool run3_DAQ_format =
            (getAlgoVersion() >=
             11546);  // Firmware from 26.08.22 which enabled new Run 3 DAQ format for GEMs - EY 13.09.22
        bool reducedDAQWindow =
            (getAlgoVersion() >=
             11656);  // Firmware from 08.12.22 which is used as a flag for new reduced readout window - EY 01.03.23

        int nTPs = run3_DAQ_format ? 2 : 1;

        // Check Format of Payload
        l1t::emtf::GEM GEM_;
        for (int err = 0; err < checkFormat(block); ++err) {
          GEM_.add_format_error();
        }

        // Assign payload to 16-bit words
        uint16_t GEMa = payload[0];
        uint16_t GEMb = payload[1];
        uint16_t GEMc = payload[2];
        uint16_t GEMd = payload[3];

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

          // TODO: Verify this is correct for GEM
          GEMPadDigiClusterCollection* res_GEM;
          res_GEM = static_cast<EMTFCollections*>(coll)->getEMTFGEMPadClusters();

          ////////////////////////////
          // Unpack the GEM Data Record
          ////////////////////////////
          if (run3_DAQ_format) {  // Run 3 DAQ format has 2 TPs per block
            if (i == 1) {
              GEM_.set_pad(GetHexBits(GEMa, 0, 8));
              GEM_.set_partition(GetHexBits(GEMa, 9, 11));
              GEM_.set_cluster_size(GetHexBits(GEMa, 12, 14));

              if (reducedDAQWindow)  // reduced DAQ window is used only after run3 DAQ format
                GEM_.set_tbin(GetHexBits(GEMb, 0, 2) + 1);
              else
                GEM_.set_tbin(GetHexBits(GEMb, 0, 2));
              GEM_.set_vp(GetHexBits(GEMb, 3, 3));
              GEM_.set_bc0(GetHexBits(GEMb, 7, 7));
              GEM_.set_cluster_id(GetHexBits(GEMb, 8, 11));
              GEM_.set_link(GetHexBits(GEMb, 12, 14));
            } else if (i == 2) {
              GEM_.set_pad(GetHexBits(GEMc, 0, 8));
              GEM_.set_partition(GetHexBits(GEMc, 9, 11));
              GEM_.set_cluster_size(GetHexBits(GEMc, 12, 14));

              if (reducedDAQWindow)  // reduced DAQ window is used only after run3 DAQ format
                GEM_.set_tbin(GetHexBits(GEMd, 0, 2) + 1);
              else
                GEM_.set_tbin(GetHexBits(GEMd, 0, 2));
              GEM_.set_vp(GetHexBits(GEMd, 3, 3));
              GEM_.set_bc0(GetHexBits(GEMd, 7, 7));
              GEM_.set_cluster_id(GetHexBits(GEMd, 8, 11));
              GEM_.set_link(GetHexBits(GEMd, 12, 14));
            }
          } else {
            GEM_.set_pad(GetHexBits(GEMa, 0, 8));
            GEM_.set_partition(GetHexBits(GEMa, 9, 11));
            GEM_.set_cluster_size(GetHexBits(GEMa, 12, 14));

            GEM_.set_cluster_id(GetHexBits(GEMb, 8, 11));
            GEM_.set_link(GetHexBits(GEMb, 12, 14));

            GEM_.set_gem_bxn(GetHexBits(GEMc, 0, 11));
            GEM_.set_bc0(GetHexBits(GEMc, 14, 14));

            GEM_.set_tbin(GetHexBits(GEMd, 0, 2));
            GEM_.set_vp(GetHexBits(GEMd, 3, 3));

            // GEM_.set_dataword(uint64_t dataword);
          }

          // Convert specially-encoded GEM quantities
          // TODO: is the RPC or CSC method for this function better... - JS 06.07.20
          int _station, _ring, _sector, _subsector, _neighbor, _layer;
          convert_GEM_location(_station,
                               _ring,
                               _sector,
                               _subsector,
                               _neighbor,
                               _layer,
                               (res->at(iOut)).PtrEventHeader()->Sector(),
                               GEM_.ClusterID(),
                               GEM_.Link());

          // Rotate by 20 deg to match GEM convention in CMSSW) // FIXME VERIFY
          // int _sector_gem = (_subsector < 5) ? _sector : (_sector % 6) + 1; //
          int _sector_gem = _sector;
          // Rotate by 2 to match GEM convention in CMSSW (GEMDetId.h) // FIXME VERIFY
          int _subsector_gem = ((_subsector + 1) % 6) + 1;
          // Define chamber number) // FIXME VERIFY
          int _chamber = (_sector_gem - 1) * 6 + _subsector_gem;
          // Define CSC-like subsector) // FIXME WHY?? VERIFY
          int _subsector_csc = (_station != 1) ? 0 : ((_chamber % 6 > 2) ? 1 : 2);

          Hit_.set_station(_station);
          Hit_.set_ring(_ring);
          Hit_.set_sector(_sector);
          Hit_.set_subsector(_subsector_csc);
          Hit_.set_chamber(_chamber);
          Hit_.set_neighbor(_neighbor);

          // Fill the EMTFHit
          ImportGEM(Hit_, GEM_, (res->at(iOut)).PtrEventHeader()->Endcap(), (res->at(iOut)).PtrEventHeader()->Sector());

          // Set the stub number for this hit
          // Each chamber can send up to 2 stubs per BX // FIXME is this true for GEM, are stubs relevant for GEMs?
          // Also count stubs in corresponding CSC chamber; GEM hit counting is on top of LCT counting
          Hit_.set_stub_num(0);
          // See if matching hit is already in event record
          bool exact_duplicate = false;
          for (auto const& iHit : *res_hit) {
            if (Hit_.BX() == iHit.BX() && Hit_.Endcap() == iHit.Endcap() && Hit_.Station() == iHit.Station() &&
                Hit_.Chamber() == iHit.Chamber()) {
              if ((iHit.Is_CSC() == 1 && iHit.Ring() == 2) ||
                  (iHit.Is_GEM() == 1)) {  // Copied from RPC, but GEM has no ring 2/3...
                if (Hit_.Neighbor() == iHit.Neighbor()) {
                  Hit_.set_stub_num(Hit_.Stub_num() + 1);
                  if (iHit.Is_GEM() == 1 && iHit.Ring() == Hit_.Ring() && iHit.Roll() == Hit_.Roll() &&
                      iHit.Pad() == Hit_.Pad()) {
                    exact_duplicate = true;
                  }
                }
              }
            }
          }  // End loop: for (auto const & iHit : *res_hit)

          // Reject TPs with out-of-range BX values. This needs to be adjusted if we increase l1a_window parameter in EMTF config - EY 03.08.2022
          if (Hit_.BX() > 3 or Hit_.BX() < -3) {
            edm::LogWarning("L1T|EMTF") << "EMTF unpacked  GEM digis with out-of-range BX! BX " << Hit_.BX()
                                        << ", endcap " << Hit_.Endcap() << ", station " << Hit_.Station()
                                        << ", neighbor " << Hit_.Neighbor() << ", ring " << Hit_.Ring() << ", chamber "
                                        << Hit_.Chamber() << ", roll " << Hit_.Roll() << ", pad " << Hit_.Pad()
                                        << std::endl;
            return true;
          }

          // TODO: Re-enable once GEM TP data format is fixed
          // if (exact_duplicate)
          //   edm::LogWarning("L1T|EMTF") << "EMTF unpacked duplicate GEM digis: BX " << Hit_.BX() << ", endcap "
          //                               << Hit_.Endcap() << ", station " << Hit_.Station() << ", neighbor "
          //                               << Hit_.Neighbor() << ", ring " << Hit_.Ring() << ", chamber " << Hit_.Chamber()
          //                               << ", roll " << Hit_.Roll() << ", pad " << Hit_.Pad() << std::endl;

          (res->at(iOut)).push_GEM(GEM_);
          if (!exact_duplicate and Hit_.Valid())
            res_hit->push_back(Hit_);

          if (!exact_duplicate and Hit_.Valid())
            res_GEM->insertDigi(Hit_.GEM_DetId(), Hit_.CreateGEMPadDigiCluster());

          // Finished with unpacking one GEM Data Record
        }
        return true;

      }  // End bool GEMBlockUnpacker::unpack

      // bool GEMBlockPacker::pack(const Block& block, UnpackerCollections *coll) {
      // 	std::cout << "Inside GEMBlockPacker::pack" << std::endl;
      // 	return true;
      // } // End bool GEMBlockPacker::pack

    }  // End namespace emtf
  }    // End namespace stage2
}  // End namespace l1t

DEFINE_L1T_UNPACKER(l1t::stage2::emtf::GEMBlockUnpacker);
// DEFINE_L1T_PACKER(l1t::stage2::emtf::GEMBlockPacker);
