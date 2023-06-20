#include "EMTFUnpackerTools.h"
#include "DataFormats/L1TMuon/interface/L1TMuonSubsystems.h"

namespace l1t {
  namespace stage2 {
    namespace emtf {

      void ImportME(EMTFHit& _hit, const l1t::emtf::ME _ME, const int _endcap, const int _evt_sector) {
        _hit.set_endcap(_endcap == 1 ? 1 : -1);
        _hit.set_sector_idx(_endcap == 1 ? _evt_sector - 1 : _evt_sector + 5);

        _hit.set_wire(_ME.Wire());
        _hit.set_strip(_ME.Strip());
        _hit.set_quality(_ME.Quality());
        _hit.set_pattern(_ME.CLCT_pattern());
        _hit.set_bend((_ME.LR() == 1) ? 1 : -1);
        _hit.set_valid(_ME.VP());
        _hit.set_sync_err(_ME.SE());
        _hit.set_bx(_ME.TBIN() - 3);
        _hit.set_bc0(_ME.BC0());
        _hit.set_subsystem(l1tmu::kCSC);
        // _hit.set_layer();

        // Run 3 OTMB
        _hit.set_strip_quart_bit(_ME.Quarter_strip());
        _hit.set_strip_eighth_bit(_ME.Eighth_strip());
        _hit.set_slope(_ME.Slope());
        _hit.set_pattern_run3(_ME.Run3_pattern());

        // Run 3 muon shower
        _hit.set_muon_shower_inTime(_ME.MUS_inTime());
        _hit.set_muon_shower_outOfTime(_ME.MUS_outOfTime());
        _hit.set_muon_shower_valid(_ME.MUSV());

        _hit.set_ring(L1TMuonEndCap::calc_ring(_hit.Station(), _hit.CSC_ID(), _hit.Strip()));
        _hit.set_chamber(
            L1TMuonEndCap::calc_chamber(_hit.Station(), _hit.Sector(), _hit.Subsector(), _hit.Ring(), _hit.CSC_ID()));

        _hit.SetCSCDetId(_hit.CreateCSCDetId());
        //_hit.SetCSCLCTDigi ( _hit.CreateCSCCorrelatedLCTDigi() );

        // Station, CSC_ID, Sector, Subsector, and Neighbor filled in
        // EventFilter/L1TRawToDigi/src/implementations_stage2/EMTFBlockME.cc
        // "set_layer()" is not invoked, so Layer is not yet filled - AWB 21.04.16

      }  // End ImportME

      void ImportRPC(EMTFHit& _hit, const l1t::emtf::RPC _RPC, const int _endcap, const int _evt_sector) {
        _hit.set_endcap(_endcap == 1 ? 1 : -1);
        _hit.set_sector_idx(_endcap == 1 ? _evt_sector - 1 : _evt_sector + 5);

        _hit.set_phi_fp(_RPC.Phi() * 4);      // 1/4th the precision of CSC LCTs
        _hit.set_theta_fp(_RPC.Theta() * 4);  // 1/4th the precision of CSC LCTs
        _hit.set_bx(_RPC.TBIN() - 3);
        _hit.set_valid(_RPC.VP());
        _hit.set_bc0(_RPC.BC0());
        _hit.set_subsystem(l1tmu::kRPC);

        _hit.SetRPCDetId(_hit.CreateRPCDetId());
        // // Not yet implemented - AWB 15.03.17
        // _hit.SetRPCDigi  ( _hit.CreateRPCDigi() );

        // Convert integer values to degrees
        _hit.set_phi_loc(L1TMuonEndCap::calc_phi_loc_deg(_hit.Phi_fp()));
        _hit.set_phi_glob(L1TMuonEndCap::calc_phi_glob_deg(_hit.Phi_loc(), _evt_sector));
        _hit.set_theta(L1TMuonEndCap::calc_theta_deg_from_int(_hit.Theta_fp()));
        _hit.set_eta(L1TMuonEndCap::calc_eta_from_theta_deg(_hit.Theta(), _hit.Endcap()));

        // Station, Ring, Sector, Subsector, Neighbor, and PC/FS/BT_segment filled in
        // EventFilter/L1TRawToDigi/src/implementations_stage2/EMTFBlockRPC.cc - AWB 02.05.17

      }  // End ImportRPC

      void ImportGEM(EMTFHit& _hit, const l1t::emtf::GEM& _GEM, const int _endcap, const int _evt_sector) {
        constexpr uint8_t GEM_MAX_CLUSTERS_PER_LAYER = 8;
        _hit.set_endcap(_endcap == 1 ? 1 : -1);
        _hit.set_sector_idx(_endcap == 1 ? _evt_sector - 1 : _evt_sector + 5);

        _hit.set_pad(_GEM.Pad());
        _hit.set_pad_hi(_GEM.Pad() + (_GEM.ClusterSize() - 1));
        _hit.set_pad_low(_GEM.Pad());
        _hit.set_partition(_GEM.Partition());
        // TODO: verify layer naming is 0/1 and not 1/2
        _hit.set_layer(_GEM.ClusterID() < GEM_MAX_CLUSTERS_PER_LAYER ? 0 : 1);
        _hit.set_cluster_size(_GEM.ClusterSize());
        _hit.set_cluster_id(_GEM.ClusterID());
        // TODO: FIXME is this value known for GEM? - JS 13.07.20
        _hit.set_bx(_GEM.TBIN() - 3);
        _hit.set_valid(_GEM.VP());
        _hit.set_bc0(_GEM.BC0());
        _hit.set_subsystem(l1tmu::kGEM);

        _hit.set_ring(1);  // GEM only on ring 1
        // TODO: FIXME correct for GEM, should match CSC chamber, but GEM have 2 chambers (layers in a superchamber) per CSC chamber - JS 13.07.20
        // _hit.set_chamber(L1TMuonEndCap::calc_chamber(_hit.Station(), _hit.Sector(), _hit.Subsector(), _hit.Ring(), _hit.GEM_ID()));
        _hit.SetGEMDetId(_hit.CreateGEMDetId());
        // _hit.SetGEMDigi(_hit.CreateGEMPadDigi());

        // Station, Ring, Sector, Subsector, and Neighbor filled in
        // EventFilter/L1TRawToDigi/src/implementations_stage2/EMTFBlockGEM.cc - JS 13.07.20

      }  // End ImportGEM

      void ImportSP(EMTFTrack& _track, const l1t::emtf::SP _SP, const int _endcap, const int _evt_sector) {
        _track.set_endcap((_endcap == 1) ? 1 : -1);
        _track.set_sector(_evt_sector);
        _track.set_sector_idx((_endcap == 1) ? _evt_sector - 1 : _evt_sector + 5);
        _track.set_mode(_SP.Mode());
        _track.set_mode_inv((((_SP.Mode() >> 0) & 1) << 3) | (((_SP.Mode() >> 1) & 1) << 2) |
                            (((_SP.Mode() >> 2) & 1) << 1) | (((_SP.Mode() >> 3) & 1) << 0));
        _track.set_charge((_SP.C() == 1) ? -1 : 1);  // uGMT uses opposite of physical charge (to match pdgID)
        _track.set_bx(_SP.TBIN() - 3);
        _track.set_phi_fp(_SP.Phi_full());
        _track.set_phi_loc(L1TMuonEndCap::calc_phi_loc_deg(_SP.Phi_full()));
        _track.set_phi_glob(L1TMuonEndCap::calc_phi_glob_deg(_track.Phi_loc(), _track.Sector()));
        _track.set_eta(L1TMuonEndCap::calc_eta(_SP.Eta_GMT()));
        _track.set_pt((_SP.Pt_GMT() - 1) * 0.5);

        _track.set_gmt_pt(_SP.Pt_GMT());
        _track.set_gmt_phi(_SP.Phi_GMT());
        _track.set_gmt_eta(_SP.Eta_GMT());
        _track.set_gmt_quality(_SP.Quality_GMT());
        _track.set_gmt_charge(_SP.C());
        _track.set_gmt_charge_valid(_SP.VC());

        EMTFPtLUT _lut = {};
        _lut.address = _SP.Pt_LUT_addr();
        _track.set_PtLUT(_lut);

        // First_bx, Second_bx, Track_num, Has_neighbor, All_neighbor, and Hits should be filled in
        // EventFilter/L1TRawToDigi/src/implementations_stage2/EMTFBlockSP.cc - AWB 07.03.17

      }  // End ImportSP

    }  // End namespace emtf
  }    // End namespace stage2
}  // End namespace l1t
