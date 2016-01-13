#include <string>
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "sp_wrap.h"
extern bool __glob_change__;
extern size_t __glob_alwaysn__;

void sp_wrap::run
(
	unsigned q__io[5][9][2],
	unsigned wg__io[5][9][2],
	unsigned hstr__io[5][9][2],
	unsigned cpat__io[5][9][2],

	unsigned bt_phi__io [3],
	unsigned bt_theta__io [3],
	unsigned bt_cpattern__io [3],
	// ph and th deltas from best stations
	// [best_track_num], last index: [0] - best pair of stations, [1] - second best pair
	unsigned bt_delta_ph__io [3][2],
	unsigned bt_delta_th__io [3][2], 
	unsigned bt_sign_ph__io[3][2],
	unsigned bt_sign_th__io[3][2],
	// ranks [best_track_num]
	unsigned bt_rank__io [3],
	// segment IDs
	// [best_track_num][station 0-3]
	unsigned bt_vi__io [3][5], // valid
	unsigned bt_hi__io [3][5], // bx index
	unsigned bt_ci__io [3][5], // chamber
	unsigned bt_si__io [3][5] // segment
)
{

  static bool first_time = true;

  if (!built)
    {
      	sim_lib_init();
		seg_ch = 2;
		bw_ph = 8;
		bw_th = 7;
		bw_fph = 12;
		bw_fth = 8;
		bw_wg = 7;
		bw_ds = 7;
		bw_hs = 8;
		pat_w_st3 = 3;
		pat_w_st1 = pat_w_st3 + 1;
		full_pat_w_st3 = (1 << (pat_w_st3+1)) - 1;
		full_pat_w_st1 = (1 << (pat_w_st1+1)) - 1;
		padding_w_st1 = full_pat_w_st1 / 2;
		padding_w_st3 = full_pat_w_st3 / 2;
		red_pat_w_st3 = pat_w_st3 * 2 + 1;
		red_pat_w_st1 = pat_w_st1 * 2 + 1;
		fold = 4;
		th_ch11 = seg_ch*seg_ch;
		bw_q = 4;
		bw_addr = 7;
		ph_raw_w = (1 << pat_w_st3) * 15;
		th_raw_w = (1 << bw_th);
		max_drift = 3;
		bw_phi = 12;
		bw_eta = 7;
		ph_hit_w = 40+4;
		ph_hit_w20 = ph_hit_w;
		ph_hit_w10 = 20+4;
		th_hit_w = 56 + 8;
		endcap = 1;
		n_strips = (station <= 1 && cscid <= 2) ? 64 :
						 (station <= 1 && cscid >= 6) ? 64 : 80;
		n_wg = (station <= 1 && cscid <= 3) ? 48  :
					 (station <= 1 && cscid >= 6) ? 32  :
					 (station == 2 && cscid <= 3) ? 112 :
					 (station >= 3 && cscid <= 3) ? 96  : 64;
		th_coverage = (station <= 1 && cscid <= 2) ? 45  :
						 (station <= 1 && cscid >= 6) ? 27  :
						 (station <= 1 && cscid >= 3) ? 39  :
						 (station == 2 && cscid <= 2) ? 43  :
						 (station == 2 && cscid >= 3) ? 56  :
						 (station == 3 && cscid <= 2) ? 34  :
						 (station == 3 && cscid >= 3) ? 52  :
						 (station == 4 && cscid <= 2) ? 28  :
						 (station == 4 && cscid >= 3) ? 50  : 0;
		ph_coverage = (station <= 1 && cscid >= 6) ? 15 : //30 :
						   (station >= 2 && cscid <= 2) ? 40 : 20;
		th_ch = (station <= 1 && cscid <= 2) ? (seg_ch*seg_ch) : seg_ch;
		ph_reverse = (endcap == 1 && station >= 3) ? 1 : 
			   			   (endcap == 2 && station <  3) ? 1 : 0;
		th_mem_sz = (1 << bw_addr);
		th_corr_mem_sz = (1 << bw_addr);
		mult_bw = bw_fph + 11;
		ph_zone_bnd1 = (station <= 1 && cscid <= 2) ? 41 :
							(station == 2 && cscid <= 2) ? 41 :
							(station == 2 && cscid >  2) ? 87 :
							(station == 3 && cscid >  2) ? 49 :
							(station == 4 && cscid >  2) ? 49 : 127;
		ph_zone_bnd2 = (station == 3 && cscid >  2) ? 87 : 127;
		zone_overlap = 2;
		bwr = 6;
		bpow = 6;
		cnr = (1 << bpow);
		cnrex = ph_raw_w;
		build();
    }

  	if (first_time)
    {
		first_time = false;

		iadr = 0;
		s = 0;
		i = 0;
		j = 0;
		good_ev_cnt = 0;
		found_tr = 0;
		found_cand = 0;

		uut
			(
				qi,
				wgi,
				hstri,
		        cpati,
				csi,
				pps_csi,
				seli,
				addri,
				r_ini,
				r_outo,
				wei,
				bt_phi,
				bt_theta,
                bt_cpattern,
				bt_delta_ph,
				bt_delta_th,
				bt_sign_ph,
				bt_sign_th,
				bt_rank,
				bt_vi,
				bt_hi,
				bt_ci,
				bt_si,
				clki,
				clki
				); 

			// fill th LUTs
			Sreadmemh(edm::FileInPath("L1Trigger/CSCTrackFinder/data/core_upgrade/vl_lut/vl_th_corr_lut_endcap_1_sec_1_sub_1_st_1_ch_1.lut").fullPath().c_str(), uut.pcs.genblk.station11[0].csc11[0].pc11.th_corr_mem);
			Sreadmemh(edm::FileInPath("L1Trigger/CSCTrackFinder/data/core_upgrade/vl_lut/vl_th_corr_lut_endcap_1_sec_1_sub_1_st_1_ch_2.lut").fullPath().c_str(), uut.pcs.genblk.station11[0].csc11[1].pc11.th_corr_mem);
			Sreadmemh(edm::FileInPath("L1Trigger/CSCTrackFinder/data/core_upgrade/vl_lut/vl_th_corr_lut_endcap_1_sec_1_sub_1_st_1_ch_3.lut").fullPath().c_str(), uut.pcs.genblk.station11[0].csc11[2].pc11.th_corr_mem);
			Sreadmemh(edm::FileInPath("L1Trigger/CSCTrackFinder/data/core_upgrade/vl_lut/vl_th_corr_lut_endcap_1_sec_1_sub_2_st_1_ch_1.lut").fullPath().c_str(), uut.pcs.genblk.station11[1].csc11[0].pc11.th_corr_mem);
			Sreadmemh(edm::FileInPath("L1Trigger/CSCTrackFinder/data/core_upgrade/vl_lut/vl_th_corr_lut_endcap_1_sec_1_sub_2_st_1_ch_2.lut").fullPath().c_str(), uut.pcs.genblk.station11[1].csc11[1].pc11.th_corr_mem);
			Sreadmemh(edm::FileInPath("L1Trigger/CSCTrackFinder/data/core_upgrade/vl_lut/vl_th_corr_lut_endcap_1_sec_1_sub_2_st_1_ch_3.lut").fullPath().c_str(), uut.pcs.genblk.station11[1].csc11[2].pc11.th_corr_mem);

			Sreadmemh(edm::FileInPath("L1Trigger/CSCTrackFinder/data/core_upgrade/vl_lut/vl_th_lut_endcap_1_sec_1_sub_1_st_1_ch_1.lut").fullPath().c_str(), uut.pcs.genblk.station11[0].csc11[0].pc11.th_mem);
			Sreadmemh(edm::FileInPath("L1Trigger/CSCTrackFinder/data/core_upgrade/vl_lut/vl_th_lut_endcap_1_sec_1_sub_1_st_1_ch_2.lut").fullPath().c_str(), uut.pcs.genblk.station11[0].csc11[1].pc11.th_mem);
			Sreadmemh(edm::FileInPath("L1Trigger/CSCTrackFinder/data/core_upgrade/vl_lut/vl_th_lut_endcap_1_sec_1_sub_1_st_1_ch_3.lut").fullPath().c_str(), uut.pcs.genblk.station11[0].csc11[2].pc11.th_mem);
			Sreadmemh(edm::FileInPath("L1Trigger/CSCTrackFinder/data/core_upgrade/vl_lut/vl_th_lut_endcap_1_sec_1_sub_1_st_1_ch_4.lut").fullPath().c_str(), uut.pcs.genblk.station12[0].csc12[3].pc12.th_mem);
			Sreadmemh(edm::FileInPath("L1Trigger/CSCTrackFinder/data/core_upgrade/vl_lut/vl_th_lut_endcap_1_sec_1_sub_1_st_1_ch_5.lut").fullPath().c_str(), uut.pcs.genblk.station12[0].csc12[4].pc12.th_mem);
			Sreadmemh(edm::FileInPath("L1Trigger/CSCTrackFinder/data/core_upgrade/vl_lut/vl_th_lut_endcap_1_sec_1_sub_1_st_1_ch_6.lut").fullPath().c_str(), uut.pcs.genblk.station12[0].csc12[5].pc12.th_mem);
			Sreadmemh(edm::FileInPath("L1Trigger/CSCTrackFinder/data/core_upgrade/vl_lut/vl_th_lut_endcap_1_sec_1_sub_1_st_1_ch_7.lut").fullPath().c_str(), uut.pcs.genblk.station12[0].csc12[6].pc12.th_mem);
			Sreadmemh(edm::FileInPath("L1Trigger/CSCTrackFinder/data/core_upgrade/vl_lut/vl_th_lut_endcap_1_sec_1_sub_1_st_1_ch_8.lut").fullPath().c_str(), uut.pcs.genblk.station12[0].csc12[7].pc12.th_mem);
			Sreadmemh(edm::FileInPath("L1Trigger/CSCTrackFinder/data/core_upgrade/vl_lut/vl_th_lut_endcap_1_sec_1_sub_1_st_1_ch_9.lut").fullPath().c_str(), uut.pcs.genblk.station12[0].csc12[8].pc12.th_mem);
			
			Sreadmemh(edm::FileInPath("L1Trigger/CSCTrackFinder/data/core_upgrade/vl_lut/vl_th_lut_endcap_1_sec_1_sub_2_st_1_ch_1.lut").fullPath().c_str(), uut.pcs.genblk.station11[1].csc11[0].pc11.th_mem);
			Sreadmemh(edm::FileInPath("L1Trigger/CSCTrackFinder/data/core_upgrade/vl_lut/vl_th_lut_endcap_1_sec_1_sub_2_st_1_ch_2.lut").fullPath().c_str(), uut.pcs.genblk.station11[1].csc11[1].pc11.th_mem);
			Sreadmemh(edm::FileInPath("L1Trigger/CSCTrackFinder/data/core_upgrade/vl_lut/vl_th_lut_endcap_1_sec_1_sub_2_st_1_ch_3.lut").fullPath().c_str(), uut.pcs.genblk.station11[1].csc11[2].pc11.th_mem);
			Sreadmemh(edm::FileInPath("L1Trigger/CSCTrackFinder/data/core_upgrade/vl_lut/vl_th_lut_endcap_1_sec_1_sub_2_st_1_ch_4.lut").fullPath().c_str(), uut.pcs.genblk.station12[1].csc12[3].pc12.th_mem);
			Sreadmemh(edm::FileInPath("L1Trigger/CSCTrackFinder/data/core_upgrade/vl_lut/vl_th_lut_endcap_1_sec_1_sub_2_st_1_ch_5.lut").fullPath().c_str(), uut.pcs.genblk.station12[1].csc12[4].pc12.th_mem);
			Sreadmemh(edm::FileInPath("L1Trigger/CSCTrackFinder/data/core_upgrade/vl_lut/vl_th_lut_endcap_1_sec_1_sub_2_st_1_ch_6.lut").fullPath().c_str(), uut.pcs.genblk.station12[1].csc12[5].pc12.th_mem);
			Sreadmemh(edm::FileInPath("L1Trigger/CSCTrackFinder/data/core_upgrade/vl_lut/vl_th_lut_endcap_1_sec_1_sub_2_st_1_ch_7.lut").fullPath().c_str(), uut.pcs.genblk.station12[1].csc12[6].pc12.th_mem);
			Sreadmemh(edm::FileInPath("L1Trigger/CSCTrackFinder/data/core_upgrade/vl_lut/vl_th_lut_endcap_1_sec_1_sub_2_st_1_ch_8.lut").fullPath().c_str(), uut.pcs.genblk.station12[1].csc12[7].pc12.th_mem);
			Sreadmemh(edm::FileInPath("L1Trigger/CSCTrackFinder/data/core_upgrade/vl_lut/vl_th_lut_endcap_1_sec_1_sub_2_st_1_ch_9.lut").fullPath().c_str(), uut.pcs.genblk.station12[1].csc12[8].pc12.th_mem);

			Sreadmemh(edm::FileInPath("L1Trigger/CSCTrackFinder/data/core_upgrade/vl_lut/vl_th_lut_endcap_1_sec_1_st_2_ch_1.lut").fullPath().c_str(), uut.pcs.genblk.station[2].csc[0].pc.th_mem);
			Sreadmemh(edm::FileInPath("L1Trigger/CSCTrackFinder/data/core_upgrade/vl_lut/vl_th_lut_endcap_1_sec_1_st_2_ch_2.lut").fullPath().c_str(), uut.pcs.genblk.station[2].csc[1].pc.th_mem);
			Sreadmemh(edm::FileInPath("L1Trigger/CSCTrackFinder/data/core_upgrade/vl_lut/vl_th_lut_endcap_1_sec_1_st_2_ch_3.lut").fullPath().c_str(), uut.pcs.genblk.station[2].csc[2].pc.th_mem);
			Sreadmemh(edm::FileInPath("L1Trigger/CSCTrackFinder/data/core_upgrade/vl_lut/vl_th_lut_endcap_1_sec_1_st_2_ch_4.lut").fullPath().c_str(), uut.pcs.genblk.station[2].csc[3].pc.th_mem);
			Sreadmemh(edm::FileInPath("L1Trigger/CSCTrackFinder/data/core_upgrade/vl_lut/vl_th_lut_endcap_1_sec_1_st_2_ch_5.lut").fullPath().c_str(), uut.pcs.genblk.station[2].csc[4].pc.th_mem);
			Sreadmemh(edm::FileInPath("L1Trigger/CSCTrackFinder/data/core_upgrade/vl_lut/vl_th_lut_endcap_1_sec_1_st_2_ch_6.lut").fullPath().c_str(), uut.pcs.genblk.station[2].csc[5].pc.th_mem);
			Sreadmemh(edm::FileInPath("L1Trigger/CSCTrackFinder/data/core_upgrade/vl_lut/vl_th_lut_endcap_1_sec_1_st_2_ch_7.lut").fullPath().c_str(), uut.pcs.genblk.station[2].csc[6].pc.th_mem);
			Sreadmemh(edm::FileInPath("L1Trigger/CSCTrackFinder/data/core_upgrade/vl_lut/vl_th_lut_endcap_1_sec_1_st_2_ch_8.lut").fullPath().c_str(), uut.pcs.genblk.station[2].csc[7].pc.th_mem);
			Sreadmemh(edm::FileInPath("L1Trigger/CSCTrackFinder/data/core_upgrade/vl_lut/vl_th_lut_endcap_1_sec_1_st_2_ch_9.lut").fullPath().c_str(), uut.pcs.genblk.station[2].csc[8].pc.th_mem);
			
			Sreadmemh(edm::FileInPath("L1Trigger/CSCTrackFinder/data/core_upgrade/vl_lut/vl_th_lut_endcap_1_sec_1_st_3_ch_1.lut").fullPath().c_str(), uut.pcs.genblk.station[3].csc[0].pc.th_mem);
			Sreadmemh(edm::FileInPath("L1Trigger/CSCTrackFinder/data/core_upgrade/vl_lut/vl_th_lut_endcap_1_sec_1_st_3_ch_2.lut").fullPath().c_str(), uut.pcs.genblk.station[3].csc[1].pc.th_mem);
			Sreadmemh(edm::FileInPath("L1Trigger/CSCTrackFinder/data/core_upgrade/vl_lut/vl_th_lut_endcap_1_sec_1_st_3_ch_3.lut").fullPath().c_str(), uut.pcs.genblk.station[3].csc[2].pc.th_mem);
			Sreadmemh(edm::FileInPath("L1Trigger/CSCTrackFinder/data/core_upgrade/vl_lut/vl_th_lut_endcap_1_sec_1_st_3_ch_4.lut").fullPath().c_str(), uut.pcs.genblk.station[3].csc[3].pc.th_mem);
			Sreadmemh(edm::FileInPath("L1Trigger/CSCTrackFinder/data/core_upgrade/vl_lut/vl_th_lut_endcap_1_sec_1_st_3_ch_5.lut").fullPath().c_str(), uut.pcs.genblk.station[3].csc[4].pc.th_mem);
			Sreadmemh(edm::FileInPath("L1Trigger/CSCTrackFinder/data/core_upgrade/vl_lut/vl_th_lut_endcap_1_sec_1_st_3_ch_6.lut").fullPath().c_str(), uut.pcs.genblk.station[3].csc[5].pc.th_mem);
			Sreadmemh(edm::FileInPath("L1Trigger/CSCTrackFinder/data/core_upgrade/vl_lut/vl_th_lut_endcap_1_sec_1_st_3_ch_7.lut").fullPath().c_str(), uut.pcs.genblk.station[3].csc[6].pc.th_mem);
			Sreadmemh(edm::FileInPath("L1Trigger/CSCTrackFinder/data/core_upgrade/vl_lut/vl_th_lut_endcap_1_sec_1_st_3_ch_8.lut").fullPath().c_str(), uut.pcs.genblk.station[3].csc[7].pc.th_mem);
			Sreadmemh(edm::FileInPath("L1Trigger/CSCTrackFinder/data/core_upgrade/vl_lut/vl_th_lut_endcap_1_sec_1_st_3_ch_9.lut").fullPath().c_str(), uut.pcs.genblk.station[3].csc[8].pc.th_mem);
			
			Sreadmemh(edm::FileInPath("L1Trigger/CSCTrackFinder/data/core_upgrade/vl_lut/vl_th_lut_endcap_1_sec_1_st_4_ch_1.lut").fullPath().c_str(), uut.pcs.genblk.station[4].csc[0].pc.th_mem);
			Sreadmemh(edm::FileInPath("L1Trigger/CSCTrackFinder/data/core_upgrade/vl_lut/vl_th_lut_endcap_1_sec_1_st_4_ch_2.lut").fullPath().c_str(), uut.pcs.genblk.station[4].csc[1].pc.th_mem);
			Sreadmemh(edm::FileInPath("L1Trigger/CSCTrackFinder/data/core_upgrade/vl_lut/vl_th_lut_endcap_1_sec_1_st_4_ch_3.lut").fullPath().c_str(), uut.pcs.genblk.station[4].csc[2].pc.th_mem);
			Sreadmemh(edm::FileInPath("L1Trigger/CSCTrackFinder/data/core_upgrade/vl_lut/vl_th_lut_endcap_1_sec_1_st_4_ch_4.lut").fullPath().c_str(), uut.pcs.genblk.station[4].csc[3].pc.th_mem);
			Sreadmemh(edm::FileInPath("L1Trigger/CSCTrackFinder/data/core_upgrade/vl_lut/vl_th_lut_endcap_1_sec_1_st_4_ch_5.lut").fullPath().c_str(), uut.pcs.genblk.station[4].csc[4].pc.th_mem);
			Sreadmemh(edm::FileInPath("L1Trigger/CSCTrackFinder/data/core_upgrade/vl_lut/vl_th_lut_endcap_1_sec_1_st_4_ch_6.lut").fullPath().c_str(), uut.pcs.genblk.station[4].csc[5].pc.th_mem);
			Sreadmemh(edm::FileInPath("L1Trigger/CSCTrackFinder/data/core_upgrade/vl_lut/vl_th_lut_endcap_1_sec_1_st_4_ch_7.lut").fullPath().c_str(), uut.pcs.genblk.station[4].csc[6].pc.th_mem);
			Sreadmemh(edm::FileInPath("L1Trigger/CSCTrackFinder/data/core_upgrade/vl_lut/vl_th_lut_endcap_1_sec_1_st_4_ch_8.lut").fullPath().c_str(), uut.pcs.genblk.station[4].csc[7].pc.th_mem);
			Sreadmemh(edm::FileInPath("L1Trigger/CSCTrackFinder/data/core_upgrade/vl_lut/vl_th_lut_endcap_1_sec_1_st_4_ch_9.lut").fullPath().c_str(), uut.pcs.genblk.station[4].csc[8].pc.th_mem);

			//Sreadmemh(edm::FileInPath("L1Trigger/CSCTrackFinder/data/core_upgrade/vl_lut/ph_init_endcap_1_sect_1.lut").fullPath().c_str(), ph_init);
			Sreadmemh(edm::FileInPath("L1Trigger/CSCTrackFinder/data/core_upgrade/vl_lut/th_init_endcap_1_sect_1.lut").fullPath().c_str(), th_init);
			Sreadmemh(edm::FileInPath("L1Trigger/CSCTrackFinder/data/core_upgrade/vl_lut/ph_disp_endcap_1_sect_1.lut").fullPath().c_str(), ph_disp);
			Sreadmemh(edm::FileInPath("L1Trigger/CSCTrackFinder/data/core_upgrade/vl_lut/th_disp_endcap_1_sect_1.lut").fullPath().c_str(), th_disp);

			Sreadmemh(edm::FileInPath("L1Trigger/CSCTrackFinder/data/core_upgrade/vl_lut/ph_init_full_endcap_1_sect_1_st_0.lut").fullPath().c_str(), ph_init[0]);
			Sreadmemh(edm::FileInPath("L1Trigger/CSCTrackFinder/data/core_upgrade/vl_lut/ph_init_full_endcap_1_sect_1_st_1.lut").fullPath().c_str(), ph_init[1]);
			Sreadmemh(edm::FileInPath("L1Trigger/CSCTrackFinder/data/core_upgrade/vl_lut/ph_init_full_endcap_1_sect_1_st_2.lut").fullPath().c_str(), ph_init[2]);
			Sreadmemh(edm::FileInPath("L1Trigger/CSCTrackFinder/data/core_upgrade/vl_lut/ph_init_full_endcap_1_sect_1_st_3.lut").fullPath().c_str(), ph_init[3]);
			Sreadmemh(edm::FileInPath("L1Trigger/CSCTrackFinder/data/core_upgrade/vl_lut/ph_init_full_endcap_1_sect_1_st_4.lut").fullPath().c_str(), ph_init[4]);

		for (i = 0; i < 180; i = i+1)
		{

			wei = 0;
			for (k = 0; k < 5; k = k+1)
				csi[k] = 0;

				// write ph_init and th_init parameters into ME1/1 only
				if (i < 36)
				{
					csi[i/18][(i/6)%3] = 1;//[station][chamber]
					seli = 0;
					wei = 1;
					addri = i%6;
					if (( (addri) == 0)) { r_ini = ph_init[i/18][(i/6)%3]; } else 
					if (( (addri) == // ph_init_b
						1)) { r_ini = ph_disp[(i/18)*12 + (i/6)%3]; } else 
					if (( (addri) == // ph_disp_b
						2)) { r_ini = ph_init[i/18][(i/6)%3 + 9]; } else 
					if (( (addri) == // ph_init_a
						3)) { r_ini = ph_disp[(i/18)*12 + (i/6)%3 + 9]; } else 
					if (( (addri) == // ph_disp_a
						4)) { r_ini = th_init[(i/18)*12 + (i/6)%3]; } else 
					if (( (addri) == // th_init
						5)) { r_ini = th_disp[(i/18)*12 + (i/6)%3]; } 
				}

				// all other ME1 chambers
				if (i >= 36 && i < 36+48)
				{
					ii = i - 36;
					csi[ii/24][(ii/4)%6+3] = 1;//[station][chamber]
					seli = 0;
					wei = 1;
					addri = ii % 4;
					if (( (addri) == 0)) { r_ini = ph_init[ii/24][(ii/4)%6+3]; } else 
					if (( (addri) == // ph_init
						1)) { r_ini = th_init[(ii/24)*12 + (ii/4)%6+3]; } else 
					if (( (addri) == // th_init
						2)) { r_ini = ph_disp[(ii/24)*12 + (ii/4)%6+3]; } else 
					if (( (addri) == // ph_disp
						3)) { r_ini = th_disp[(ii/24)*12 + (ii/4)%6+3]; } 
				}

				// ME2,3,4 chambers
				if (i >= 36+48 && i < 36+48+108)
				{
					ii = i - (36+48);
					csi[ii/36+2][(ii/4)%9] = 1; //[station][chamber]
					seli = 0;
					wei = 1;
					addri = ii % 4;
					if (( (addri) == 0)) { r_ini = ph_init[ii/36+2][(ii/4)%9]; } else 
					if (( (addri) == // ph_init
						1)) { r_ini = th_init[(ii/36)*9 + (ii/4)%9 + 24]; } else 
					if (( (addri) == // th_init
						2)) { r_ini = ph_disp[(ii/36)*9 + (ii/4)%9 + 24]; } else 
					if (( (addri) == // ph_disp
						3)) { r_ini = th_disp[(ii/36)*9 + (ii/4)%9 + 24]; } 
				}


			for (j = 0; j < 2; j = j+1)
			{
				clk_drive(clki, j);

				while(true)
				{
					__glob_change__ = false;
					init();
					if (!__glob_change__) break;
					uut
						(
							qi,
							wgi,
							hstri,
							cpati,
							csi,
							pps_csi,
							seli,
							addri,
							r_ini,
							r_outo,
							wei,
							bt_phi,
							bt_theta,
							bt_cpattern,
							bt_delta_ph,
							bt_delta_th,
							bt_sign_ph,
							bt_sign_th,
							bt_rank,
							bt_vi,
							bt_hi,
							bt_ci,
							bt_si,
							clki,
							clki
							);

				}
			}
		}
    } // first time processing done

  // copy inputs to signals
  for (unsigned mi0 = 0; mi0 <= 4; mi0++)
    for (unsigned mi1 = 0; mi1 <= 8; mi1++)
      for (unsigned mi2 = 0; mi2 <= seg_ch-1; mi2++)
	qi[mi0][mi1][mi2] = q__io[mi0][mi1][mi2];
  for (unsigned mi0 = 0; mi0 <= 4; mi0++)
    for (unsigned mi1 = 0; mi1 <= 8; mi1++)
      for (unsigned mi2 = 0; mi2 <= seg_ch-1; mi2++)
	wgi[mi0][mi1][mi2] = wg__io[mi0][mi1][mi2];
  for (unsigned mi0 = 0; mi0 <= 4; mi0++)
    for (unsigned mi1 = 0; mi1 <= 8; mi1++)
      for (unsigned mi2 = 0; mi2 <= seg_ch-1; mi2++)
	hstri[mi0][mi1][mi2] = hstr__io[mi0][mi1][mi2];
  for (unsigned mi0 = 0; mi0 <= 4; mi0++)
    for (unsigned mi1 = 0; mi1 <= 8; mi1++)
      for (unsigned mi2 = 0; mi2 <= seg_ch-1; mi2++)
	cpati[mi0][mi1][mi2] = cpat__io[mi0][mi1][mi2];

  wei = 0;
	
  for (j = 0; j < 2; j = j+1)
  {
      clk_drive(clki, j);

      while(true)
	  {
		  __glob_change__ = false;
		  init();
		  if (!__glob_change__) break;
		  uut
			  (
				  qi,
				  wgi,
				  hstri,
				  cpati,
				  csi,
				  pps_csi,
				  seli,
				  addri,
				  r_ini,
				  r_outo,
				  wei,
				  bt_phi,
				  bt_theta,
				  bt_cpattern,
				  bt_delta_ph,
				  bt_delta_th,
				  bt_sign_ph,
				  bt_sign_th,
				  bt_rank,
				  bt_vi,
				  bt_hi,
				  bt_ci,
				  bt_si,
				  clki,
				  clki
				  );
	  }
  }
  // copy output signals to io
  for (unsigned mi0 = 0; mi0 <= 2; mi0++)
  {
	  bt_phi__io [mi0] =         bt_phi [mi0];
	  bt_theta__io [mi0] =		 bt_theta [mi0];
	  bt_cpattern__io [mi0] =	 bt_cpattern [mi0];
	  bt_rank__io [mi0] =		 bt_rank [mi0];

	  for (unsigned mi1 = 0; mi1 <= 1; mi1++)
	  {
		  bt_delta_ph__io [mi0][mi1] = bt_delta_ph [mi0][mi1];
		  bt_delta_th__io [mi0][mi1] = bt_delta_th [mi0][mi1]; 
		  bt_sign_ph__io[mi0][mi1] =	 bt_sign_ph[mi0][mi1];
		  bt_sign_th__io[mi0][mi1] =	 bt_sign_th[mi0][mi1];
	  }
	  for (unsigned mi1 = 0; mi1 <= 4; mi1++)
	  {
		  bt_vi__io [mi0][mi1] =		 bt_vi [mi0][mi1];
		  bt_hi__io [mi0][mi1] =		 bt_hi [mi0][mi1];
		  bt_ci__io [mi0][mi1] =		 bt_ci [mi0][mi1];
		  bt_si__io [mi0][mi1] =		 bt_si [mi0][mi1];
	  }
  }


}

void sp_wrap::defparam()
{
	station = 1;
	cscid = 1;
	max_ev = 505;
}

void sp_wrap::build()
{
	built = true;
	qi__storage.add_dim(4, 0);
	qi__storage.add_dim(8, 0);
	qi__storage.add_dim(seg_ch-1, 0);
	qi__storage.bw(3, 0);
	qi__storage.build();
	qi.add_dim(4, 0);
	qi.add_dim(8, 0);
	qi.add_dim(seg_ch-1, 0);
	qi.bw(3, 0);
	qi.build();
	qi.set_storage (&qi__storage);
	wgi__storage.add_dim(4, 0);
	wgi__storage.add_dim(8, 0);
	wgi__storage.add_dim(seg_ch-1, 0);
	wgi__storage.bw(bw_wg-1, 0);
	wgi__storage.build();
	wgi.add_dim(4, 0);
	wgi.add_dim(8, 0);
	wgi.add_dim(seg_ch-1, 0);
	wgi.bw(bw_wg-1, 0);
	wgi.build();
	wgi.set_storage (&wgi__storage);
	hstri__storage.add_dim(4, 0);
	hstri__storage.add_dim(8, 0);
	hstri__storage.add_dim(seg_ch-1, 0);
	hstri__storage.bw(bw_hs-1, 0);
	hstri__storage.build();
	hstri.add_dim(4, 0);
	hstri.add_dim(8, 0);
	hstri.add_dim(seg_ch-1, 0);
	hstri.bw(bw_hs-1, 0);
	hstri.build();
	hstri.set_storage (&hstri__storage);
	cpati__storage.add_dim(4, 0);
	cpati__storage.add_dim(8, 0);
	cpati__storage.add_dim(seg_ch-1, 0);
	cpati__storage.bw(3, 0);
	cpati__storage.build();
	cpati.add_dim(4, 0);
	cpati.add_dim(8, 0);
	cpati.add_dim(seg_ch-1, 0);
	cpati.bw(3, 0);
	cpati.build();
	cpati.set_storage (&cpati__storage);
	csi__storage.add_dim(4, 0);
	csi__storage.bw(8, 0);
	csi__storage.build();
	csi.add_dim(4, 0);
	csi.bw(8, 0);
	csi.build();
	csi.set_storage (&csi__storage);
	pps_csi__storage.add_dim(2, 0);
	pps_csi__storage.bw(4, 0);
	pps_csi__storage.build();
	pps_csi.add_dim(2, 0);
	pps_csi.bw(4, 0);
	pps_csi.build();
	pps_csi.set_storage (&pps_csi__storage);
	seli__storage.bw(1, 0);
	seli.bw(1, 0);
	seli.set_storage (&seli__storage);
	addri__storage.bw(bw_addr-1, 0);
	addri.bw(bw_addr-1, 0);
	addri.set_storage (&addri__storage);
	r_ini__storage.bw(11, 0);
	r_ini.bw(11, 0);
	r_ini.set_storage (&r_ini__storage);
	wei__storage.bw(0, 0);
	wei.bw(0, 0);
	wei.set_storage (&wei__storage);
	clki__storage.bw(0, 0);
	clki.bw(0, 0);
	clki.set_storage (&clki__storage);

	ph_init__storage.add_dim(4, 0);
	ph_init__storage.add_dim(11, 0);
	ph_init__storage.bw(bw_fph, 0);
	ph_init__storage.build();
	ph_init.add_dim(4, 0);
	ph_init.add_dim(11, 0);
	ph_init.bw(bw_fph, 0);
	ph_init.build();
	ph_init.set_storage (&ph_init__storage);

	th_init__storage.add_dim(50, 0);
	th_init__storage.bw(bw_th-1, 0);
	th_init__storage.build();
	th_init.add_dim(50, 0);
	th_init.bw(bw_th-1, 0);
	th_init.build();
	th_init.set_storage (&th_init__storage);
	ph_disp__storage.add_dim(50, 0);
	ph_disp__storage.bw(bw_ph, 0);
	ph_disp__storage.build();
	ph_disp.add_dim(50, 0);
	ph_disp.bw(bw_ph, 0);
	ph_disp.build();
	ph_disp.set_storage (&ph_disp__storage);
	th_disp__storage.add_dim(50, 0);
	th_disp__storage.bw(bw_th-1, 0);
	th_disp__storage.build();
	th_disp.add_dim(50, 0);
	th_disp.bw(bw_th-1, 0);
	th_disp.build();
	th_disp.set_storage (&th_disp__storage);
	quality__storage.add_dim(max_ev-1, 0);
	quality__storage.add_dim(4, 0);
	quality__storage.add_dim(8, 0);
	quality__storage.add_dim(seg_ch-1, 0);
	quality__storage.bw(3, 0);
	quality__storage.build();
	quality.add_dim(max_ev-1, 0);
	quality.add_dim(4, 0);
	quality.add_dim(8, 0);
	quality.add_dim(seg_ch-1, 0);
	quality.bw(3, 0);
	quality.build();
	quality.set_storage (&quality__storage);
	wiregroup__storage.add_dim(max_ev-1, 0);
	wiregroup__storage.add_dim(4, 0);
	wiregroup__storage.add_dim(8, 0);
	wiregroup__storage.add_dim(seg_ch-1, 0);
	wiregroup__storage.bw(6, 0);
	wiregroup__storage.build();
	wiregroup.add_dim(max_ev-1, 0);
	wiregroup.add_dim(4, 0);
	wiregroup.add_dim(8, 0);
	wiregroup.add_dim(seg_ch-1, 0);
	wiregroup.bw(6, 0);
	wiregroup.build();
	wiregroup.set_storage (&wiregroup__storage);
	hstrip__storage.add_dim(max_ev-1, 0);
	hstrip__storage.add_dim(4, 0);
	hstrip__storage.add_dim(8, 0);
	hstrip__storage.add_dim(seg_ch-1, 0);
	hstrip__storage.bw(bw_hs-1, 0);
	hstrip__storage.build();
	hstrip.add_dim(max_ev-1, 0);
	hstrip.add_dim(4, 0);
	hstrip.add_dim(8, 0);
	hstrip.add_dim(seg_ch-1, 0);
	hstrip.bw(bw_hs-1, 0);
	hstrip.build();
	hstrip.set_storage (&hstrip__storage);
	clctpat__storage.add_dim(max_ev-1, 0);
	clctpat__storage.add_dim(4, 0);
	clctpat__storage.add_dim(8, 0);
	clctpat__storage.add_dim(seg_ch-1, 0);
	clctpat__storage.bw(3, 0);
	clctpat__storage.build();
	clctpat.add_dim(max_ev-1, 0);
	clctpat.add_dim(4, 0);
	clctpat.add_dim(8, 0);
	clctpat.add_dim(seg_ch-1, 0);
	clctpat.bw(3, 0);
	clctpat.build();
	clctpat.set_storage (&clctpat__storage);
	v0__storage.bw(15, 0);
	v0.bw(15, 0);
	v0.set_storage (&v0__storage);
	v1__storage.bw(15, 0);
	v1.bw(15, 0);
	v1.set_storage (&v1__storage);
	v2__storage.bw(15, 0);
	v2.bw(15, 0);
	v2.set_storage (&v2__storage);
	v3__storage.bw(15, 0);
	v3.bw(15, 0);
	v3.set_storage (&v3__storage);
	v4__storage.bw(15, 0);
	v4.bw(15, 0);
	v4.set_storage (&v4__storage);
	v5__storage.bw(15, 0);
	v5.bw(15, 0);
	v5.set_storage (&v5__storage);
	pr_cnt__storage.add_dim(5, 0);
	pr_cnt__storage.add_dim(8, 0);
	pr_cnt__storage.bw(2, 0);
	pr_cnt__storage.build();
	pr_cnt.add_dim(5, 0);
	pr_cnt.add_dim(8, 0);
	pr_cnt.bw(2, 0);
	pr_cnt.build();
	pr_cnt.set_storage (&pr_cnt__storage);
	_event__storage.bw(9, 0);
	_event.bw(9, 0);
	_event.set_storage (&_event__storage);
	_bx_jitter__storage.bw(9, 0);
	_bx_jitter.bw(9, 0);
	_bx_jitter.set_storage (&_bx_jitter__storage);
	_endcap__storage.bw(1, 0);
	_endcap.bw(1, 0);
	_endcap.set_storage (&_endcap__storage);
	_sector__storage.bw(2, 0);
	_sector.bw(2, 0);
	_sector.set_storage (&_sector__storage);
	_subsector__storage.bw(1, 0);
	_subsector.bw(1, 0);
	_subsector.set_storage (&_subsector__storage);
	_station__storage.bw(2, 0);
	_station.bw(2, 0);
	_station.set_storage (&_station__storage);
	_cscid__storage.bw(3, 0);
	_cscid.bw(3, 0);
	_cscid.set_storage (&_cscid__storage);
	_bend__storage.bw(3, 0);
	_bend.bw(3, 0);
	_bend.set_storage (&_bend__storage);
	_halfstrip__storage.bw(7, 0);
	_halfstrip.bw(7, 0);
	_halfstrip.set_storage (&_halfstrip__storage);
	_valid__storage.bw(0, 0);
	_valid.bw(0, 0);
	_valid.set_storage (&_valid__storage);
	_quality__storage.bw(3, 0);
	_quality.bw(3, 0);
	_quality.set_storage (&_quality__storage);
	_pattern__storage.bw(3, 0);
	_pattern.bw(3, 0);
	_pattern.set_storage (&_pattern__storage);
	_wiregroup__storage.bw(6, 0);
	_wiregroup.bw(6, 0);
	_wiregroup.set_storage (&_wiregroup__storage);
	line__storage.bw(800, 1);
	line.bw(800, 1);
	line.set_storage (&line__storage);
	ev__storage.bw(9, 0);
	ev.bw(9, 0);
	ev.set_storage (&ev__storage);
	good_ev__storage.bw(4, 0);
	good_ev.bw(4, 0);
	good_ev.set_storage (&good_ev__storage);
	tphi__storage.bw(11, 0);
	tphi.bw(11, 0);
	tphi.set_storage (&tphi__storage);
	a__storage.bw(11, 0);
	a.bw(11, 0);
	a.set_storage (&a__storage);
	b__storage.bw(11, 0);
	b.bw(11, 0);
	b.set_storage (&b__storage);
	d__storage.bw(11, 0);
	d.bw(11, 0);
	d.set_storage (&d__storage);
	pts__storage.bw(0, 0);
	pts.bw(0, 0);
	pts.set_storage (&pts__storage);
	r_outo__storage.bw(11, 0);
	r_outo.bw(11, 0);
	r_outo.set_storage (&r_outo__storage);
	ph_ranko__storage.add_dim(3, 0);
	ph_ranko__storage.add_dim(ph_raw_w-1, 0);
	ph_ranko__storage.bw(5, 0);
	ph_ranko__storage.build();
	ph_ranko.add_dim(3, 0);
	ph_ranko.add_dim(ph_raw_w-1, 0);
	ph_ranko.bw(5, 0);
	ph_ranko.build();
	ph_ranko.set_storage (&ph_ranko__storage);
	ph__storage.add_dim(4, 0);
	ph__storage.add_dim(8, 0);
	ph__storage.add_dim(seg_ch-1, 0);
	ph__storage.bw(bw_fph-1, 0);
	ph__storage.build();
	ph.add_dim(4, 0);
	ph.add_dim(8, 0);
	ph.add_dim(seg_ch-1, 0);
	ph.bw(bw_fph-1, 0);
	ph.build();
	ph.set_storage (&ph__storage);
	th11__storage.add_dim(1, 0);
	th11__storage.add_dim(2, 0);
	th11__storage.add_dim(th_ch11-1, 0);
	th11__storage.bw(bw_th-1, 0);
	th11__storage.build();
	th11.add_dim(1, 0);
	th11.add_dim(2, 0);
	th11.add_dim(th_ch11-1, 0);
	th11.bw(bw_th-1, 0);
	th11.build();
	th11.set_storage (&th11__storage);
	th__storage.add_dim(4, 0);
	th__storage.add_dim(8, 0);
	th__storage.add_dim(seg_ch-1, 0);
	th__storage.bw(bw_th-1, 0);
	th__storage.build();
	th.add_dim(4, 0);
	th.add_dim(8, 0);
	th.add_dim(seg_ch-1, 0);
	th.bw(bw_th-1, 0);
	th.build();
	th.set_storage (&th__storage);
	vl__storage.add_dim(4, 0);
	vl__storage.add_dim(8, 0);
	vl__storage.bw(seg_ch-1, 0);
	vl__storage.build();
	vl.add_dim(4, 0);
	vl.add_dim(8, 0);
	vl.bw(seg_ch-1, 0);
	vl.build();
	vl.set_storage (&vl__storage);
	phzvl__storage.add_dim(4, 0);
	phzvl__storage.add_dim(8, 0);
	phzvl__storage.bw(2, 0);
	phzvl__storage.build();
	phzvl.add_dim(4, 0);
	phzvl.add_dim(8, 0);
	phzvl.bw(2, 0);
	phzvl.build();
	phzvl.set_storage (&phzvl__storage);
	me11a__storage.add_dim(1, 0);
	me11a__storage.add_dim(2, 0);
	me11a__storage.bw(seg_ch-1, 0);
	me11a__storage.build();
	me11a.add_dim(1, 0);
	me11a.add_dim(2, 0);
	me11a.bw(seg_ch-1, 0);
	me11a.build();
	me11a.set_storage (&me11a__storage);
	ph_zone__storage.add_dim(3, 0);
	ph_zone__storage.add_dim(4, 1);
	ph_zone__storage.bw(ph_raw_w-1, 0);
	ph_zone__storage.build();
	ph_zone.add_dim(3, 0);
	ph_zone.add_dim(4, 1);
	ph_zone.bw(ph_raw_w-1, 0);
	ph_zone.build();
	ph_zone.set_storage (&ph_zone__storage);
	patt_vi__storage.add_dim(3, 0);
	patt_vi__storage.add_dim(2, 0);
	patt_vi__storage.add_dim(3, 0);
	patt_vi__storage.bw(seg_ch-1, 0);
	patt_vi__storage.build();
	patt_vi.add_dim(3, 0);
	patt_vi.add_dim(2, 0);
	patt_vi.add_dim(3, 0);
	patt_vi.bw(seg_ch-1, 0);
	patt_vi.build();
	patt_vi.set_storage (&patt_vi__storage);
	patt_hi__storage.add_dim(3, 0);
	patt_hi__storage.add_dim(2, 0);
	patt_hi__storage.add_dim(3, 0);
	patt_hi__storage.bw(1, 0);
	patt_hi__storage.build();
	patt_hi.add_dim(3, 0);
	patt_hi.add_dim(2, 0);
	patt_hi.add_dim(3, 0);
	patt_hi.bw(1, 0);
	patt_hi.build();
	patt_hi.set_storage (&patt_hi__storage);
	patt_ci__storage.add_dim(3, 0);
	patt_ci__storage.add_dim(2, 0);
	patt_ci__storage.add_dim(3, 0);
	patt_ci__storage.bw(2, 0);
	patt_ci__storage.build();
	patt_ci.add_dim(3, 0);
	patt_ci.add_dim(2, 0);
	patt_ci.add_dim(3, 0);
	patt_ci.bw(2, 0);
	patt_ci.build();
	patt_ci.set_storage (&patt_ci__storage);
	patt_si__storage.add_dim(3, 0);
	patt_si__storage.add_dim(2, 0);
	patt_si__storage.bw(3, 0);
	patt_si__storage.build();
	patt_si.add_dim(3, 0);
	patt_si.add_dim(2, 0);
	patt_si.bw(3, 0);
	patt_si.build();
	patt_si.set_storage (&patt_si__storage);
	ph_num__storage.add_dim(3, 0);
	ph_num__storage.add_dim(2, 0);
	ph_num__storage.bw(bpow, 0);
	ph_num__storage.build();
	ph_num.add_dim(3, 0);
	ph_num.add_dim(2, 0);
	ph_num.bw(bpow, 0);
	ph_num.build();
	ph_num.set_storage (&ph_num__storage);
	ph_q__storage.add_dim(3, 0);
	ph_q__storage.add_dim(2, 0);
	ph_q__storage.bw(bwr-1, 0);
	ph_q__storage.build();
	ph_q.add_dim(3, 0);
	ph_q.add_dim(2, 0);
	ph_q.bw(bwr-1, 0);
	ph_q.build();
	ph_q.set_storage (&ph_q__storage);
	ph_match__storage.add_dim(3, 0);
	ph_match__storage.add_dim(2, 0);
	ph_match__storage.add_dim(3, 0);
	ph_match__storage.bw(bw_fph-1, 0);
	ph_match__storage.build();
	ph_match.add_dim(3, 0);
	ph_match.add_dim(2, 0);
	ph_match.add_dim(3, 0);
	ph_match.bw(bw_fph-1, 0);
	ph_match.build();
	ph_match.set_storage (&ph_match__storage);
	th_match__storage.add_dim(3, 0);
	th_match__storage.add_dim(2, 0);
	th_match__storage.add_dim(3, 0);
	th_match__storage.add_dim(seg_ch-1, 0);
	th_match__storage.bw(bw_th-1, 0);
	th_match__storage.build();
	th_match.add_dim(3, 0);
	th_match.add_dim(2, 0);
	th_match.add_dim(3, 0);
	th_match.add_dim(seg_ch-1, 0);
	th_match.bw(bw_th-1, 0);
	th_match.build();
	th_match.set_storage (&th_match__storage);
	th_match11__storage.add_dim(1, 0);
	th_match11__storage.add_dim(2, 0);
	th_match11__storage.add_dim(th_ch11-1, 0);
	th_match11__storage.bw(bw_th-1, 0);
	th_match11__storage.build();
	th_match11.add_dim(1, 0);
	th_match11.add_dim(2, 0);
	th_match11.add_dim(th_ch11-1, 0);
	th_match11.bw(bw_th-1, 0);
	th_match11.build();
	th_match11.set_storage (&th_match11__storage);
	bt_phi__storage.add_dim(2, 0);
	bt_phi__storage.bw(bw_fph-1, 0);
	bt_phi__storage.build();
	bt_phi.add_dim(2, 0);
	bt_phi.bw(bw_fph-1, 0);
	bt_phi.build();
	bt_phi.set_storage (&bt_phi__storage);
	bt_theta__storage.add_dim(2, 0);
	bt_theta__storage.bw(bw_th-1, 0);
	bt_theta__storage.build();
	bt_theta.add_dim(2, 0);
	bt_theta.bw(bw_th-1, 0);
	bt_theta.build();
	bt_theta.set_storage (&bt_theta__storage);
	bt_cpattern__storage.add_dim(2, 0);
	bt_cpattern__storage.bw(3, 0);
	bt_cpattern__storage.build();
	bt_cpattern.add_dim(2, 0);
	bt_cpattern.bw(3, 0);
	bt_cpattern.build();
	bt_cpattern.set_storage (&bt_cpattern__storage);
	bt_delta_ph__storage.add_dim(2, 0);
	bt_delta_ph__storage.add_dim(1, 0);
	bt_delta_ph__storage.bw(bw_fph-1, 0);
	bt_delta_ph__storage.build();
	bt_delta_ph.add_dim(2, 0);
	bt_delta_ph.add_dim(1, 0);
	bt_delta_ph.bw(bw_fph-1, 0);
	bt_delta_ph.build();
	bt_delta_ph.set_storage (&bt_delta_ph__storage);
	bt_delta_th__storage.add_dim(2, 0);
	bt_delta_th__storage.add_dim(1, 0);
	bt_delta_th__storage.bw(bw_th-1, 0);
	bt_delta_th__storage.build();
	bt_delta_th.add_dim(2, 0);
	bt_delta_th.add_dim(1, 0);
	bt_delta_th.bw(bw_th-1, 0);
	bt_delta_th.build();
	bt_delta_th.set_storage (&bt_delta_th__storage);
	bt_sign_ph__storage.add_dim(2, 0);
	bt_sign_ph__storage.bw(1, 0);
	bt_sign_ph__storage.build();
	bt_sign_ph.add_dim(2, 0);
	bt_sign_ph.bw(1, 0);
	bt_sign_ph.build();
	bt_sign_ph.set_storage (&bt_sign_ph__storage);
	bt_sign_th__storage.add_dim(2, 0);
	bt_sign_th__storage.bw(1, 0);
	bt_sign_th__storage.build();
	bt_sign_th.add_dim(2, 0);
	bt_sign_th.bw(1, 0);
	bt_sign_th.build();
	bt_sign_th.set_storage (&bt_sign_th__storage);
	bt_rank__storage.add_dim(2, 0);
	bt_rank__storage.bw(bwr, 0);
	bt_rank__storage.build();
	bt_rank.add_dim(2, 0);
	bt_rank.bw(bwr, 0);
	bt_rank.build();
	bt_rank.set_storage (&bt_rank__storage);
	bt_vi__storage.add_dim(2, 0);
	bt_vi__storage.add_dim(4, 0);
	bt_vi__storage.bw(seg_ch-1, 0);
	bt_vi__storage.build();
	bt_vi.add_dim(2, 0);
	bt_vi.add_dim(4, 0);
	bt_vi.bw(seg_ch-1, 0);
	bt_vi.build();
	bt_vi.set_storage (&bt_vi__storage);
	bt_hi__storage.add_dim(2, 0);
	bt_hi__storage.add_dim(4, 0);
	bt_hi__storage.bw(1, 0);
	bt_hi__storage.build();
	bt_hi.add_dim(2, 0);
	bt_hi.add_dim(4, 0);
	bt_hi.bw(1, 0);
	bt_hi.build();
	bt_hi.set_storage (&bt_hi__storage);
	bt_ci__storage.add_dim(2, 0);
	bt_ci__storage.add_dim(4, 0);
	bt_ci__storage.bw(2, 0);
	bt_ci__storage.build();
	bt_ci.add_dim(2, 0);
	bt_ci.add_dim(4, 0);
	bt_ci.bw(2, 0);
	bt_ci.build();
	bt_ci.set_storage (&bt_ci__storage);
	bt_si__storage.add_dim(2, 0);
	bt_si__storage.bw(4, 0);
	bt_si__storage.build();
	bt_si.add_dim(2, 0);
	bt_si.bw(4, 0);
	bt_si.build();
	bt_si.set_storage (&bt_si__storage);
	iadr__storage.bw(31, 0);
	iadr.bw(31, 0);
	iadr.set_storage (&iadr__storage);
	s__storage.bw(31, 0);
	s.bw(31, 0);
	s.set_storage (&s__storage);
	i__storage.bw(31, 0);
	i.bw(31, 0);
	i.set_storage (&i__storage);
	ii__storage.bw(31, 0);
	ii.bw(31, 0);
	ii.set_storage (&ii__storage);
	pi__storage.bw(31, 0);
	pi.bw(31, 0);
	pi.set_storage (&pi__storage);
	j__storage.bw(31, 0);
	j.bw(31, 0);
	j.set_storage (&j__storage);
	sn__storage.bw(31, 0);
	sn.bw(31, 0);
	sn.set_storage (&sn__storage);
	ist__storage.bw(31, 0);
	ist.bw(31, 0);
	ist.set_storage (&ist__storage);
	icid__storage.bw(31, 0);
	icid.bw(31, 0);
	icid.set_storage (&icid__storage);
	ipr__storage.bw(31, 0);
	ipr.bw(31, 0);
	ipr.set_storage (&ipr__storage);
	code__storage.bw(31, 0);
	code.bw(31, 0);
	code.set_storage (&code__storage);
	iev__storage.bw(31, 0);
	iev.bw(31, 0);
	iev.set_storage (&iev__storage);
	im__storage.bw(31, 0);
	im.bw(31, 0);
	im.set_storage (&im__storage);
	iz__storage.bw(31, 0);
	iz.bw(31, 0);
	iz.set_storage (&iz__storage);
	ir__storage.bw(31, 0);
	ir.bw(31, 0);
	ir.set_storage (&ir__storage);
	in__storage.bw(31, 0);
	in.bw(31, 0);
	in.set_storage (&in__storage);
	best_tracks__storage.bw(31, 0);
	best_tracks.bw(31, 0);
	best_tracks.set_storage (&best_tracks__storage);
	stat__storage.bw(31, 0);
	stat.bw(31, 0);
	stat.set_storage (&stat__storage);
	good_ev_cnt__storage.bw(31, 0);
	good_ev_cnt.bw(31, 0);
	good_ev_cnt.set_storage (&good_ev_cnt__storage);
	found_tr__storage.bw(31, 0);
	found_tr.bw(31, 0);
	found_tr.set_storage (&found_tr__storage);
	found_cand__storage.bw(31, 0);
	found_cand.bw(31, 0);
	found_cand.set_storage (&found_cand__storage);
	st__storage.bw(31, 0);
	st.bw(31, 0);
	st.set_storage (&st__storage);
	st_cnt__storage.bw(31, 0);
	st_cnt.bw(31, 0);
	st_cnt.set_storage (&st_cnt__storage);
	iseg__storage.bw(31, 0);
	iseg.bw(31, 0);
	iseg.set_storage (&iseg__storage);
	zi__storage.bw(31, 0);
	zi.bw(31, 0);
	zi.set_storage (&zi__storage);
	si__storage.bw(31, 0);
	si.bw(31, 0);
	si.set_storage (&si__storage);
	ip__storage.bw(31, 0);
	ip.bw(31, 0);
	ip.set_storage (&ip__storage);
	ibx__storage.bw(31, 0);
	ibx.bw(31, 0);
	ibx.set_storage (&ibx__storage);
	ich__storage.bw(31, 0);
	ich.bw(31, 0);
	ich.set_storage (&ich__storage);
	isg__storage.bw(31, 0);
	isg.bw(31, 0);
	isg.set_storage (&isg__storage);
	begin_time__storage.bw(31, 0);
	begin_time.bw(31, 0);
	begin_time.set_storage (&begin_time__storage);
	end_time__storage.bw(31, 0);
	end_time.bw(31, 0);
	end_time.set_storage (&end_time__storage);
	elapsed_time__storage.bw(31, 0);
	elapsed_time.bw(31, 0);
	elapsed_time.set_storage (&elapsed_time__storage);
	ev = 0;
	good_ev = 0;
}

void sp_wrap::init ()
{
	if (!built)
	{
								}
	else
	{
		qi__storage.init();
		wgi__storage.init();
		hstri__storage.init();
		cpati__storage.init();
		csi__storage.init();
		pps_csi__storage.init();
		seli__storage.init();
		addri__storage.init();
		r_ini__storage.init();
		wei__storage.init();
		clki__storage.init();
		ph_init__storage.init();
		th_init__storage.init();
		ph_disp__storage.init();
		th_disp__storage.init();
		quality__storage.init();
		wiregroup__storage.init();
		hstrip__storage.init();
		clctpat__storage.init();
		v0__storage.init();
		v1__storage.init();
		v2__storage.init();
		v3__storage.init();
		v4__storage.init();
		v5__storage.init();
		pr_cnt__storage.init();
		_event__storage.init();
		_bx_jitter__storage.init();
		_endcap__storage.init();
		_sector__storage.init();
		_subsector__storage.init();
		_station__storage.init();
		_cscid__storage.init();
		_bend__storage.init();
		_halfstrip__storage.init();
		_valid__storage.init();
		_quality__storage.init();
		_pattern__storage.init();
		_wiregroup__storage.init();
		line__storage.init();
		ev__storage.init();
		good_ev__storage.init();
		tphi__storage.init();
		a__storage.init();
		b__storage.init();
		d__storage.init();
		pts__storage.init();
		r_outo__storage.init();
		ph_ranko__storage.init();
		ph__storage.init();
		th11__storage.init();
		th__storage.init();
		vl__storage.init();
		phzvl__storage.init();
		me11a__storage.init();
		ph_zone__storage.init();
		patt_vi__storage.init();
		patt_hi__storage.init();
		patt_ci__storage.init();
		patt_si__storage.init();
		ph_num__storage.init();
		ph_q__storage.init();
		ph_match__storage.init();
		th_match__storage.init();
		th_match11__storage.init();
		bt_phi__storage.init();
		bt_theta__storage.init();
		bt_cpattern__storage.init();
		bt_delta_ph__storage.init();
		bt_delta_th__storage.init();
		bt_sign_ph__storage.init();
		bt_sign_th__storage.init();
		bt_rank__storage.init();
		bt_vi__storage.init();
		bt_hi__storage.init();
		bt_ci__storage.init();
		bt_si__storage.init();
		uut.init();

	}


}
