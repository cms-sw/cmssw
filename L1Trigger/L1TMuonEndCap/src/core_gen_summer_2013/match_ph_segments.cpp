// This C++ source file was automatically generated
// by VPPC from a Verilog HDL project.
// VPPC web-page: http://www.phys.ufl.edu/~madorsky/vppc/

// Author    : madorsky
// Timestamp : Thu Mar 12 14:54:00 2015

#include "match_ph_segments.h"

extern size_t __glob_alwaysn__;

void match_ph_segments::operator()
(
	signal_& ph_num__io,
	signal_& ph_q__io,
	signal_& ph__io,
	signal_& vl__io,
	signal_& th11__io,
	signal_& th__io,
	signal_& cpat__io,
	signal_& vi__io,
	signal_& hi__io,
	signal_& ci__io,
	signal_& si__io,
	signal_& ph_match__io,
	signal_& th_match__io,
	signal_& th_match11__io,
	signal_& cpat_match__io,
	signal_& ph_qr__io,
	signal_& clk__io
)
{
	if (!built)
	{
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
		ph_raw_w = (1 << pat_w_st3) * 15 + 2;
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
		// numbers of best ranks [zone][rank number]
		ph_num.attach(ph_num__io);
		// best ranks [zone][rank number]
		ph_q.attach(ph_q__io);
		// ph outputs delayed and extended [bx_history][station][chamber][segment]
// most recent in bx = 0
		ph.attach(ph__io);
		// valid flags
		vl.attach(vl__io);
		// thetas
		th11.attach(th11__io);
		th.attach(th__io);
		cpat.attach(cpat__io);
		clk.attach(clk__io);
		// find_segment outputs, in terms of segments match in zones [zone][pattern_num][station 0-3]
		vi.attach(vi__io);
		hi.attach(hi__io);
		ci.attach(ci__io);
		si.attach(si__io);
		ph_match.attach(ph_match__io);
		// suspected matching th coordinates, all taken from chambers where ph comes from
// [zone][pattern_num][station 0-3][segment]
		th_match.attach(th_match__io);
		th_match11.attach(th_match11__io);
		cpat_match.attach(cpat_match__io);
		// best ranks [zone][rank number]
		ph_qr.attach(ph_qr__io);
	}

	if (true)
	{
		// segments rerouted for find_segment inputs
		// see find_segment_reroute.xlsx for details
		// indexes are: [bx_history][chamber][segment]
		// excel file line shown at the } of each code line
		for (i = 0; i < max_drift; i = i+1) // bx_history loop
		{
			for (j = 0; j < 6; j = j+1) // chamber loop for 6-chamber finders
			{
				for (k = 0; k < th_ch11; k = k+1) // segment loop for ME11 only
				{
					th_seg___z0_s0[i][j][k] = (j < 3) ?	 th11[i][0][j][k] : th11[i][1][j-3][k]; // 2
					th_seg___z1_s0[i][j][k] = (j < 3) ?	 th11[i][0][j][k] : th11[i][1][j-3][k]; // 7
				}
				
				for (k = 0; k < seg_ch; k = k+1) // segment loop all other chambers
				{
					ph_seg___z0_s0[i][j][k] = (j < 3) ?	 ph[i][0][j][k] : ph[i][1][j-3][k]; // 2
					ph_seg_v_z0_s0[i][j][k] = (j < 3) ?	 vl[i][0][j][k] : vl[i][1][j-3][k]; 

					ph_seg___z1_s0[i][j][k] = (j < 3) ?	 ph[i][0][j][k] : ph[i][1][j-3][k]; // 7
					ph_seg_v_z1_s0[i][j][k] = (j < 3) ?	 vl[i][0][j][k] : vl[i][1][j-3][k];
					ph_seg___z1_s2[i][j][k] = ph[i][3][j+3][k]; // 9
					ph_seg_v_z1_s2[i][j][k] = vl[i][3][j+3][k];
					ph_seg___z1_s3[i][j][k] = ph[i][4][j+3][k]; // 10
					ph_seg_v_z1_s3[i][j][k] = vl[i][4][j+3][k];

					ph_seg___z2_s0[i][j][k] = (j < 3) ?	 ph[i][0][j+3][k] : ph[i][1][j][k]; // 12
					ph_seg_v_z2_s0[i][j][k] = (j < 3) ?	 vl[i][0][j+3][k] : vl[i][1][j][k];
					ph_seg___z2_s1[i][j][k] = ph[i][2][j+3][k]; // 13
					ph_seg_v_z2_s1[i][j][k] = vl[i][2][j+3][k]; 
					ph_seg___z2_s2[i][j][k] = ph[i][3][j+3][k]; // 14
					ph_seg_v_z2_s2[i][j][k] = vl[i][3][j+3][k];
					ph_seg___z2_s3[i][j][k] = ph[i][4][j+3][k]; // 15 
					ph_seg_v_z2_s3[i][j][k] = vl[i][4][j+3][k];

					ph_seg___z3_s0[i][j][k] = (j < 3) ?	 ph[i][0][j+6][k] : ph[i][1][j+3][k]; // 18
					ph_seg_v_z3_s0[i][j][k] = (j < 3) ?	 vl[i][0][j+6][k] : vl[i][1][j+3][k];	   
					ph_seg___z3_s1[i][j][k] = ph[i][2][j+3][k]; // 19							   
					ph_seg_v_z3_s1[i][j][k] = vl[i][2][j+3][k];									   
					ph_seg___z3_s2[i][j][k] = ph[i][3][j+3][k]; // 20							   
					ph_seg_v_z3_s2[i][j][k] = vl[i][3][j+3][k];                                    
				    ph_seg___z3_s3[i][j][k] = 0; // no station 4 in zone 3
				    ph_seg_v_z3_s3[i][j][k] = 0; // no station 4 in zone 3

					cpat_seg___z0_s0[i][j][k] = (j < 3) ?	 cpat[i][0][j][k] : cpat[i][1][j-3][k]; // 2

					cpat_seg___z1_s0[i][j][k] = (j < 3) ?	 cpat[i][0][j][k] : cpat[i][1][j-3][k]; // 7
					cpat_seg___z1_s2[i][j][k] = cpat[i][3][j+3][k]; // 9
					cpat_seg___z1_s3[i][j][k] = cpat[i][4][j+3][k]; // 10

					cpat_seg___z2_s0[i][j][k] = (j < 3) ?	 cpat[i][0][j+3][k] : cpat[i][1][j][k]; // 12
					cpat_seg___z2_s1[i][j][k] = cpat[i][2][j+3][k]; // 13
					cpat_seg___z2_s2[i][j][k] = cpat[i][3][j+3][k]; // 14
					cpat_seg___z2_s3[i][j][k] = cpat[i][4][j+3][k]; // 15 

					cpat_seg___z3_s0[i][j][k] = (j < 3) ?	 cpat[i][0][j+6][k] : cpat[i][1][j+3][k]; // 18
					cpat_seg___z3_s1[i][j][k] = cpat[i][2][j+3][k]; // 19
					cpat_seg___z3_s2[i][j][k] = cpat[i][3][j+3][k]; // 20
				    cpat_seg___z3_s3[i][j][k] = 0; // no station 4 in zone 3
					
					th_seg___z1_s2[i][j][k] = th[i][3][j+3][k]; // 9
					th_seg___z1_s3[i][j][k] = th[i][4][j+3][k]; // 10

					th_seg___z2_s0[i][j][k] = (j < 3) ?	 th[i][0][j+3][k] : th[i][1][j][k]; // 12
					th_seg___z2_s1[i][j][k] = th[i][2][j+3][k]; // 13
					th_seg___z2_s2[i][j][k] = th[i][3][j+3][k]; // 14
					th_seg___z2_s3[i][j][k] = th[i][4][j+3][k]; // 15 

					th_seg___z3_s0[i][j][k] = (j < 3) ?	 th[i][0][j+6][k] : th[i][1][j+3][k]; // 18
					th_seg___z3_s1[i][j][k] = th[i][2][j+3][k]; // 19
					th_seg___z3_s2[i][j][k] = th[i][3][j+3][k]; // 20
				    th_seg___z3_s3[i][j][k] = 0; // no station 4 in zone 3
				    

				}
			}
			for (j = 0; j < 3; j = j+1) // chamber loop for 3-chamber finders
			{
				for (k = 0; k < seg_ch; k = k+1) // segment loop
				{
					ph_seg___z0_s1[i][j][k] = ph[i][2][j][k]; // 3
					ph_seg_v_z0_s1[i][j][k] = vl[i][2][j][k]; 
					ph_seg___z0_s2[i][j][k] = ph[i][3][j][k]; // 4
					ph_seg_v_z0_s2[i][j][k] = vl[i][3][j][k];
					ph_seg___z0_s3[i][j][k] = ph[i][4][j][k]; // 5
					ph_seg_v_z0_s3[i][j][k] = vl[i][4][j][k];

					ph_seg___z1_s1[i][j][k] = ph[i][2][j][k]; // 8
					ph_seg_v_z1_s1[i][j][k] = vl[i][2][j][k];

					cpat_seg___z0_s1[i][j][k] = cpat[i][2][j][k]; // 3
					cpat_seg___z0_s2[i][j][k] = cpat[i][3][j][k]; // 4
					cpat_seg___z0_s3[i][j][k] = cpat[i][4][j][k]; // 5

					cpat_seg___z1_s1[i][j][k] = cpat[i][2][j][k]; // 8


					th_seg___z0_s1[i][j][k] = th[i][2][j][k]; // 3
					th_seg___z0_s2[i][j][k] = th[i][3][j][k]; // 4
					th_seg___z0_s3[i][j][k] = th[i][4][j][k]; // 5

					th_seg___z1_s1[i][j][k] = th[i][2][j][k]; // 8
				}
			}
		} // for (i = 0; i < max_drift; i = i+1)
	}

	beginalways();

	if (posedge (clk))
	{
		ph_qr = ph_q; // just propagate ranks to outputs
	}
	endalways();
	
    {
		for (ki = 0; ki < 3; ki = ki+1) // pattern loop
		  {
			 // name = fs_zone_station
			 gb.fs_loop[ki].fs_00.zone_cham = 6;  gb.fs_loop[ki].fs_00.zone_seg = th_ch11;  gb.fs_loop[ki].fs_00.station = 1;
			 gb.fs_loop[ki].fs_01.zone_cham = 3;  gb.fs_loop[ki].fs_01.zone_seg = seg_ch;   gb.fs_loop[ki].fs_01.station = 2;
			 gb.fs_loop[ki].fs_02.zone_cham = 3;  gb.fs_loop[ki].fs_02.zone_seg = seg_ch;   gb.fs_loop[ki].fs_02.station = 3;
			 gb.fs_loop[ki].fs_03.zone_cham = 3;  gb.fs_loop[ki].fs_03.zone_seg = seg_ch;   gb.fs_loop[ki].fs_03.station = 4;
			 gb.fs_loop[ki].fs_10.zone_cham = 6;  gb.fs_loop[ki].fs_10.zone_seg = th_ch11;  gb.fs_loop[ki].fs_10.station = 1;
			 gb.fs_loop[ki].fs_11.zone_cham = 3;  gb.fs_loop[ki].fs_11.zone_seg = seg_ch;   gb.fs_loop[ki].fs_11.station = 2;
			 gb.fs_loop[ki].fs_12.zone_cham = 6;  gb.fs_loop[ki].fs_12.zone_seg = seg_ch;   gb.fs_loop[ki].fs_12.station = 3;
			 gb.fs_loop[ki].fs_13.zone_cham = 6;  gb.fs_loop[ki].fs_13.zone_seg = seg_ch;   gb.fs_loop[ki].fs_13.station = 4;
			 gb.fs_loop[ki].fs_20.zone_cham = 6;  gb.fs_loop[ki].fs_20.zone_seg = seg_ch;   gb.fs_loop[ki].fs_20.station = 1;
			 gb.fs_loop[ki].fs_21.zone_cham = 6;  gb.fs_loop[ki].fs_21.zone_seg = seg_ch;   gb.fs_loop[ki].fs_21.station = 2;
			 gb.fs_loop[ki].fs_22.zone_cham = 6;  gb.fs_loop[ki].fs_22.zone_seg = seg_ch;   gb.fs_loop[ki].fs_22.station = 3;
			 gb.fs_loop[ki].fs_23.zone_cham = 6;  gb.fs_loop[ki].fs_23.zone_seg = seg_ch;   gb.fs_loop[ki].fs_23.station = 4;
			 gb.fs_loop[ki].fs_30.zone_cham = 6;  gb.fs_loop[ki].fs_30.zone_seg = seg_ch;   gb.fs_loop[ki].fs_30.station = 1;
			 gb.fs_loop[ki].fs_31.zone_cham = 6;  gb.fs_loop[ki].fs_31.zone_seg = seg_ch;   gb.fs_loop[ki].fs_31.station = 2;
			 gb.fs_loop[ki].fs_32.zone_cham = 6;  gb.fs_loop[ki].fs_32.zone_seg = seg_ch;   gb.fs_loop[ki].fs_32.station = 3;
			 gb.fs_loop[ki].fs_33.zone_cham = 6;  gb.fs_loop[ki].fs_33.zone_seg = seg_ch;   gb.fs_loop[ki].fs_33.station = 4;
 
				gb.fs_loop[ki].fs_00
	(
		ph_num[0][ki],
		ph_q[0][ki],
		ph_seg___z0_s0,
		ph_seg_v_z0_s0,
		th_seg___z0_s0,
		cpat_seg___z0_s0,
		vi[0][ki][0],
		hi[0][ki][0],
		ci[0][ki][0],
		si[0][ki][0],
		ph_match[0][ki][0],
		th_match11[0][ki],
		cpat_match[0][ki][0],
		clk
	);
				gb.fs_loop[ki].fs_01
	(
		ph_num[0][ki],
		ph_q[0][ki],
		ph_seg___z0_s1,
		ph_seg_v_z0_s1,
		th_seg___z0_s1,
		cpat_seg___z0_s1,
		vi[0][ki][1],
		hi[0][ki][1],
		ci[0][ki][1],
		si[0][ki][1],
		ph_match[0][ki][1],
		th_match[0][ki][1],
		cpat_match[0][ki][1],
		clk
	);
				gb.fs_loop[ki].fs_02
	(
		ph_num[0][ki],
		ph_q[0][ki],
		ph_seg___z0_s2,
		ph_seg_v_z0_s2,
		th_seg___z0_s2,
		cpat_seg___z0_s2,
		vi[0][ki][2],
		hi[0][ki][2],
		ci[0][ki][2],
		si[0][ki][2],
		ph_match[0][ki][2],
		th_match[0][ki][2],
		cpat_match[0][ki][2],
		clk
	);
				gb.fs_loop[ki].fs_03
	(
		ph_num[0][ki],
		ph_q[0][ki],
		ph_seg___z0_s3,
		ph_seg_v_z0_s3,
		th_seg___z0_s3,
		cpat_seg___z0_s3,
		vi[0][ki][3],
		hi[0][ki][3],
		ci[0][ki][3],
		si[0][ki][3],
		ph_match[0][ki][3],
		th_match[0][ki][3],
		cpat_match[0][ki][3],
		clk
	);
																																																																																														
				gb.fs_loop[ki].fs_10
	(
		ph_num[1][ki],
		ph_q[1][ki],
		ph_seg___z1_s0,
		ph_seg_v_z1_s0,
		th_seg___z1_s0,
		cpat_seg___z1_s0,
		vi[1][ki][0],
		hi[1][ki][0],
		ci[1][ki][0],
		si[1][ki][0],
		ph_match[1][ki][0],
		th_match11[1][ki],
		cpat_match[1][ki][0],
		clk
	);
				gb.fs_loop[ki].fs_11
	(
		ph_num[1][ki],
		ph_q[1][ki],
		ph_seg___z1_s1,
		ph_seg_v_z1_s1,
		th_seg___z1_s1,
		cpat_seg___z1_s1,
		vi[1][ki][1],
		hi[1][ki][1],
		ci[1][ki][1],
		si[1][ki][1],
		ph_match[1][ki][1],
		th_match[1][ki][1],
		cpat_match[1][ki][1],
		clk
	);
				gb.fs_loop[ki].fs_12
	(
		ph_num[1][ki],
		ph_q[1][ki],
		ph_seg___z1_s2,
		ph_seg_v_z1_s2,
		th_seg___z1_s2,
		cpat_seg___z1_s2,
		vi[1][ki][2],
		hi[1][ki][2],
		ci[1][ki][2],
		si[1][ki][2],
		ph_match[1][ki][2],
		th_match[1][ki][2],
		cpat_match[1][ki][2],
		clk
	);
				gb.fs_loop[ki].fs_13
	(
		ph_num[1][ki],
		ph_q[1][ki],
		ph_seg___z1_s3,
		ph_seg_v_z1_s3,
		th_seg___z1_s3,
		cpat_seg___z1_s3,
		vi[1][ki][3],
		hi[1][ki][3],
		ci[1][ki][3],
		si[1][ki][3],
		ph_match[1][ki][3],
		th_match[1][ki][3],
		cpat_match[1][ki][3],
		clk
	);
																																																																																														
				gb.fs_loop[ki].fs_20
	(
		ph_num[2][ki],
		ph_q[2][ki],
		ph_seg___z2_s0,
		ph_seg_v_z2_s0,
		th_seg___z2_s0,
		cpat_seg___z2_s0,
		vi[2][ki][0],
		hi[2][ki][0],
		ci[2][ki][0],
		si[2][ki][0],
		ph_match[2][ki][0],
		th_match[2][ki][0],
		cpat_match[2][ki][0],
		clk
	);
				gb.fs_loop[ki].fs_21
	(
		ph_num[2][ki],
		ph_q[2][ki],
		ph_seg___z2_s1,
		ph_seg_v_z2_s1,
		th_seg___z2_s1,
		cpat_seg___z2_s1,
		vi[2][ki][1],
		hi[2][ki][1],
		ci[2][ki][1],
		si[2][ki][1],
		ph_match[2][ki][1],
		th_match[2][ki][1],
		cpat_match[2][ki][1],
		clk
	);
				gb.fs_loop[ki].fs_22
	(
		ph_num[2][ki],
		ph_q[2][ki],
		ph_seg___z2_s2,
		ph_seg_v_z2_s2,
		th_seg___z2_s2,
		cpat_seg___z2_s2,
		vi[2][ki][2],
		hi[2][ki][2],
		ci[2][ki][2],
		si[2][ki][2],
		ph_match[2][ki][2],
		th_match[2][ki][2],
		cpat_match[2][ki][2],
		clk
	);
				gb.fs_loop[ki].fs_23
	(
		ph_num[2][ki],
		ph_q[2][ki],
		ph_seg___z2_s3,
		ph_seg_v_z2_s3,
		th_seg___z2_s3,
		cpat_seg___z2_s3,
		vi[2][ki][3],
		hi[2][ki][3],
		ci[2][ki][3],
		si[2][ki][3],
		ph_match[2][ki][3],
		th_match[2][ki][3],
		cpat_match[2][ki][3],
		clk
	);
																																																																																														
				gb.fs_loop[ki].fs_30
	(
		ph_num[3][ki],
		ph_q[3][ki],
		ph_seg___z3_s0,
		ph_seg_v_z3_s0,
		th_seg___z3_s0,
		cpat_seg___z3_s0,
		vi[3][ki][0],
		hi[3][ki][0],
		ci[3][ki][0],
		si[3][ki][0],
		ph_match[3][ki][0],
		th_match[3][ki][0],
		cpat_match[3][ki][0],
		clk
	);
				gb.fs_loop[ki].fs_31
	(
		ph_num[3][ki],
		ph_q[3][ki],
		ph_seg___z3_s1,
		ph_seg_v_z3_s1,
		th_seg___z3_s1,
		cpat_seg___z3_s1,
		vi[3][ki][1],
		hi[3][ki][1],
		ci[3][ki][1],
		si[3][ki][1],
		ph_match[3][ki][1],
		th_match[3][ki][1],
		cpat_match[3][ki][1],
		clk
	);
				gb.fs_loop[ki].fs_32
	(
		ph_num[3][ki],
		ph_q[3][ki],
		ph_seg___z3_s2,
		ph_seg_v_z3_s2,
		th_seg___z3_s2,
		cpat_seg___z3_s2,
		vi[3][ki][2],
		hi[3][ki][2],
		ci[3][ki][2],
		si[3][ki][2],
		ph_match[3][ki][2],
		th_match[3][ki][2],
		cpat_match[3][ki][2],
		clk
	);
				gb.fs_loop[ki].fs_33
	(
		ph_num[3][ki],
		ph_q[3][ki],
		ph_seg___z3_s3,
		ph_seg_v_z3_s3,
		th_seg___z3_s3,
		cpat_seg___z3_s3,
		vi[3][ki][3],
		hi[3][ki][3],
		ci[3][ki][3],
		si[3][ki][3],
		ph_match[3][ki][3],
		th_match[3][ki][3],
		cpat_match[3][ki][3],
		clk
	);
		} // block: fs_loop
	}
	
}

// vppc: code below this point are service functions, should not require your attention

// vppc: this function assigns initial values to parameters and localparams
void match_ph_segments::defparam()
{
	station = 1;
	cscid = 1;
}

// vppc: this function allocates memory for internal signals
void match_ph_segments::build()
{
	built = true;
	ph_num.add_dim(3, 0);
	ph_num.add_dim(2, 0);
	ph_num.bw(bpow, 0);
	ph_num.build();
	ph_q.add_dim(3, 0);
	ph_q.add_dim(2, 0);
	ph_q.bw(5, 0);
	ph_q.build();
	ph.add_dim(max_drift-1, 0);
	ph.add_dim(4, 0);
	ph.add_dim(8, 0);
	ph.add_dim(seg_ch-1, 0);
	ph.bw(bw_fph-1, 0);
	ph.build();
	vl.add_dim(max_drift-1, 0);
	vl.add_dim(4, 0);
	vl.add_dim(8, 0);
	vl.bw(seg_ch-1, 0);
	vl.build();
	th11.add_dim(max_drift-1, 0);
	th11.add_dim(1, 0);
	th11.add_dim(2, 0);
	th11.add_dim(th_ch11-1, 0);
	th11.bw(bw_th-1, 0);
	th11.build();
	th.add_dim(max_drift-1, 0);
	th.add_dim(4, 0);
	th.add_dim(8, 0);
	th.add_dim(seg_ch-1, 0);
	th.bw(bw_th-1, 0);
	th.build();
	cpat.add_dim(max_drift-1, 0);
	cpat.add_dim(4, 0);
	cpat.add_dim(8, 0);
	cpat.add_dim(seg_ch-1, 0);
	cpat.bw(3, 0);
	cpat.build();
	clk.bw(0, 0);
	vi.add_dim(3, 0);
	vi.add_dim(2, 0);
	vi.add_dim(3, 0);
	vi.bw(seg_ch-1, 0);
	vi.build();
	hi.add_dim(3, 0);
	hi.add_dim(2, 0);
	hi.add_dim(3, 0);
	hi.bw(1, 0);
	hi.build();
	ci.add_dim(3, 0);
	ci.add_dim(2, 0);
	ci.add_dim(3, 0);
	ci.bw(2, 0);
	ci.build();
	si.add_dim(3, 0);
	si.add_dim(2, 0);
	si.bw(3, 0);
	si.build();
	ph_match.add_dim(3, 0);
	ph_match.add_dim(2, 0);
	ph_match.add_dim(3, 0);
	ph_match.bw(bw_fph-1, 0);
	ph_match.build();
	th_match.add_dim(3, 0);
	th_match.add_dim(2, 0);
	th_match.add_dim(3, 0);
	th_match.add_dim(seg_ch-1, 0);
	th_match.bw(bw_th-1, 0);
	th_match.build();
	th_match11.add_dim(1, 0);
	th_match11.add_dim(2, 0);
	th_match11.add_dim(th_ch11-1, 0);
	th_match11.bw(bw_th-1, 0);
	th_match11.build();
	cpat_match.add_dim(3, 0);
	cpat_match.add_dim(2, 0);
	cpat_match.add_dim(3, 0);
	cpat_match.bw(3, 0);
	cpat_match.build();
	ph_qr.add_dim(3, 0);
	ph_qr.add_dim(2, 0);
	ph_qr.bw(5, 0);
	ph_qr.build();
	ph_seg___z0_s0__storage.add_dim(max_drift-1, 0);
	ph_seg___z0_s0__storage.add_dim(5, 0);
	ph_seg___z0_s0__storage.add_dim(seg_ch-1, 0);
	ph_seg___z0_s0__storage.bw(bw_fph-1, 0);
	ph_seg___z0_s0__storage.build();
	ph_seg___z0_s0.add_dim(max_drift-1, 0);
	ph_seg___z0_s0.add_dim(5, 0);
	ph_seg___z0_s0.add_dim(seg_ch-1, 0);
	ph_seg___z0_s0.bw(bw_fph-1, 0);
	ph_seg___z0_s0.build();
	ph_seg___z0_s0.set_storage (&ph_seg___z0_s0__storage);
	ph_seg_v_z0_s0__storage.add_dim(max_drift-1, 0);
	ph_seg_v_z0_s0__storage.add_dim(5, 0);
	ph_seg_v_z0_s0__storage.bw(seg_ch-1, 0);
	ph_seg_v_z0_s0__storage.build();
	ph_seg_v_z0_s0.add_dim(max_drift-1, 0);
	ph_seg_v_z0_s0.add_dim(5, 0);
	ph_seg_v_z0_s0.bw(seg_ch-1, 0);
	ph_seg_v_z0_s0.build();
	ph_seg_v_z0_s0.set_storage (&ph_seg_v_z0_s0__storage);
	ph_seg___z0_s1__storage.add_dim(max_drift-1, 0);
	ph_seg___z0_s1__storage.add_dim(2, 0);
	ph_seg___z0_s1__storage.add_dim(seg_ch-1, 0);
	ph_seg___z0_s1__storage.bw(bw_fph-1, 0);
	ph_seg___z0_s1__storage.build();
	ph_seg___z0_s1.add_dim(max_drift-1, 0);
	ph_seg___z0_s1.add_dim(2, 0);
	ph_seg___z0_s1.add_dim(seg_ch-1, 0);
	ph_seg___z0_s1.bw(bw_fph-1, 0);
	ph_seg___z0_s1.build();
	ph_seg___z0_s1.set_storage (&ph_seg___z0_s1__storage);
	ph_seg_v_z0_s1__storage.add_dim(max_drift-1, 0);
	ph_seg_v_z0_s1__storage.add_dim(2, 0);
	ph_seg_v_z0_s1__storage.bw(seg_ch-1, 0);
	ph_seg_v_z0_s1__storage.build();
	ph_seg_v_z0_s1.add_dim(max_drift-1, 0);
	ph_seg_v_z0_s1.add_dim(2, 0);
	ph_seg_v_z0_s1.bw(seg_ch-1, 0);
	ph_seg_v_z0_s1.build();
	ph_seg_v_z0_s1.set_storage (&ph_seg_v_z0_s1__storage);
	ph_seg___z0_s2__storage.add_dim(max_drift-1, 0);
	ph_seg___z0_s2__storage.add_dim(2, 0);
	ph_seg___z0_s2__storage.add_dim(seg_ch-1, 0);
	ph_seg___z0_s2__storage.bw(bw_fph-1, 0);
	ph_seg___z0_s2__storage.build();
	ph_seg___z0_s2.add_dim(max_drift-1, 0);
	ph_seg___z0_s2.add_dim(2, 0);
	ph_seg___z0_s2.add_dim(seg_ch-1, 0);
	ph_seg___z0_s2.bw(bw_fph-1, 0);
	ph_seg___z0_s2.build();
	ph_seg___z0_s2.set_storage (&ph_seg___z0_s2__storage);
	ph_seg_v_z0_s2__storage.add_dim(max_drift-1, 0);
	ph_seg_v_z0_s2__storage.add_dim(2, 0);
	ph_seg_v_z0_s2__storage.bw(seg_ch-1, 0);
	ph_seg_v_z0_s2__storage.build();
	ph_seg_v_z0_s2.add_dim(max_drift-1, 0);
	ph_seg_v_z0_s2.add_dim(2, 0);
	ph_seg_v_z0_s2.bw(seg_ch-1, 0);
	ph_seg_v_z0_s2.build();
	ph_seg_v_z0_s2.set_storage (&ph_seg_v_z0_s2__storage);
	ph_seg___z0_s3__storage.add_dim(max_drift-1, 0);
	ph_seg___z0_s3__storage.add_dim(2, 0);
	ph_seg___z0_s3__storage.add_dim(seg_ch-1, 0);
	ph_seg___z0_s3__storage.bw(bw_fph-1, 0);
	ph_seg___z0_s3__storage.build();
	ph_seg___z0_s3.add_dim(max_drift-1, 0);
	ph_seg___z0_s3.add_dim(2, 0);
	ph_seg___z0_s3.add_dim(seg_ch-1, 0);
	ph_seg___z0_s3.bw(bw_fph-1, 0);
	ph_seg___z0_s3.build();
	ph_seg___z0_s3.set_storage (&ph_seg___z0_s3__storage);
	ph_seg_v_z0_s3__storage.add_dim(max_drift-1, 0);
	ph_seg_v_z0_s3__storage.add_dim(2, 0);
	ph_seg_v_z0_s3__storage.bw(seg_ch-1, 0);
	ph_seg_v_z0_s3__storage.build();
	ph_seg_v_z0_s3.add_dim(max_drift-1, 0);
	ph_seg_v_z0_s3.add_dim(2, 0);
	ph_seg_v_z0_s3.bw(seg_ch-1, 0);
	ph_seg_v_z0_s3.build();
	ph_seg_v_z0_s3.set_storage (&ph_seg_v_z0_s3__storage);
	ph_seg___z1_s0__storage.add_dim(max_drift-1, 0);
	ph_seg___z1_s0__storage.add_dim(5, 0);
	ph_seg___z1_s0__storage.add_dim(seg_ch-1, 0);
	ph_seg___z1_s0__storage.bw(bw_fph-1, 0);
	ph_seg___z1_s0__storage.build();
	ph_seg___z1_s0.add_dim(max_drift-1, 0);
	ph_seg___z1_s0.add_dim(5, 0);
	ph_seg___z1_s0.add_dim(seg_ch-1, 0);
	ph_seg___z1_s0.bw(bw_fph-1, 0);
	ph_seg___z1_s0.build();
	ph_seg___z1_s0.set_storage (&ph_seg___z1_s0__storage);
	ph_seg_v_z1_s0__storage.add_dim(max_drift-1, 0);
	ph_seg_v_z1_s0__storage.add_dim(5, 0);
	ph_seg_v_z1_s0__storage.bw(seg_ch-1, 0);
	ph_seg_v_z1_s0__storage.build();
	ph_seg_v_z1_s0.add_dim(max_drift-1, 0);
	ph_seg_v_z1_s0.add_dim(5, 0);
	ph_seg_v_z1_s0.bw(seg_ch-1, 0);
	ph_seg_v_z1_s0.build();
	ph_seg_v_z1_s0.set_storage (&ph_seg_v_z1_s0__storage);
	ph_seg___z1_s1__storage.add_dim(max_drift-1, 0);
	ph_seg___z1_s1__storage.add_dim(2, 0);
	ph_seg___z1_s1__storage.add_dim(seg_ch-1, 0);
	ph_seg___z1_s1__storage.bw(bw_fph-1, 0);
	ph_seg___z1_s1__storage.build();
	ph_seg___z1_s1.add_dim(max_drift-1, 0);
	ph_seg___z1_s1.add_dim(2, 0);
	ph_seg___z1_s1.add_dim(seg_ch-1, 0);
	ph_seg___z1_s1.bw(bw_fph-1, 0);
	ph_seg___z1_s1.build();
	ph_seg___z1_s1.set_storage (&ph_seg___z1_s1__storage);
	ph_seg_v_z1_s1__storage.add_dim(max_drift-1, 0);
	ph_seg_v_z1_s1__storage.add_dim(2, 0);
	ph_seg_v_z1_s1__storage.bw(seg_ch-1, 0);
	ph_seg_v_z1_s1__storage.build();
	ph_seg_v_z1_s1.add_dim(max_drift-1, 0);
	ph_seg_v_z1_s1.add_dim(2, 0);
	ph_seg_v_z1_s1.bw(seg_ch-1, 0);
	ph_seg_v_z1_s1.build();
	ph_seg_v_z1_s1.set_storage (&ph_seg_v_z1_s1__storage);
	ph_seg___z1_s2__storage.add_dim(max_drift-1, 0);
	ph_seg___z1_s2__storage.add_dim(5, 0);
	ph_seg___z1_s2__storage.add_dim(seg_ch-1, 0);
	ph_seg___z1_s2__storage.bw(bw_fph-1, 0);
	ph_seg___z1_s2__storage.build();
	ph_seg___z1_s2.add_dim(max_drift-1, 0);
	ph_seg___z1_s2.add_dim(5, 0);
	ph_seg___z1_s2.add_dim(seg_ch-1, 0);
	ph_seg___z1_s2.bw(bw_fph-1, 0);
	ph_seg___z1_s2.build();
	ph_seg___z1_s2.set_storage (&ph_seg___z1_s2__storage);
	ph_seg_v_z1_s2__storage.add_dim(max_drift-1, 0);
	ph_seg_v_z1_s2__storage.add_dim(5, 0);
	ph_seg_v_z1_s2__storage.bw(seg_ch-1, 0);
	ph_seg_v_z1_s2__storage.build();
	ph_seg_v_z1_s2.add_dim(max_drift-1, 0);
	ph_seg_v_z1_s2.add_dim(5, 0);
	ph_seg_v_z1_s2.bw(seg_ch-1, 0);
	ph_seg_v_z1_s2.build();
	ph_seg_v_z1_s2.set_storage (&ph_seg_v_z1_s2__storage);
	ph_seg___z1_s3__storage.add_dim(max_drift-1, 0);
	ph_seg___z1_s3__storage.add_dim(5, 0);
	ph_seg___z1_s3__storage.add_dim(seg_ch-1, 0);
	ph_seg___z1_s3__storage.bw(bw_fph-1, 0);
	ph_seg___z1_s3__storage.build();
	ph_seg___z1_s3.add_dim(max_drift-1, 0);
	ph_seg___z1_s3.add_dim(5, 0);
	ph_seg___z1_s3.add_dim(seg_ch-1, 0);
	ph_seg___z1_s3.bw(bw_fph-1, 0);
	ph_seg___z1_s3.build();
	ph_seg___z1_s3.set_storage (&ph_seg___z1_s3__storage);
	ph_seg_v_z1_s3__storage.add_dim(max_drift-1, 0);
	ph_seg_v_z1_s3__storage.add_dim(5, 0);
	ph_seg_v_z1_s3__storage.bw(seg_ch-1, 0);
	ph_seg_v_z1_s3__storage.build();
	ph_seg_v_z1_s3.add_dim(max_drift-1, 0);
	ph_seg_v_z1_s3.add_dim(5, 0);
	ph_seg_v_z1_s3.bw(seg_ch-1, 0);
	ph_seg_v_z1_s3.build();
	ph_seg_v_z1_s3.set_storage (&ph_seg_v_z1_s3__storage);
	ph_seg___z2_s0__storage.add_dim(max_drift-1, 0);
	ph_seg___z2_s0__storage.add_dim(5, 0);
	ph_seg___z2_s0__storage.add_dim(seg_ch-1, 0);
	ph_seg___z2_s0__storage.bw(bw_fph-1, 0);
	ph_seg___z2_s0__storage.build();
	ph_seg___z2_s0.add_dim(max_drift-1, 0);
	ph_seg___z2_s0.add_dim(5, 0);
	ph_seg___z2_s0.add_dim(seg_ch-1, 0);
	ph_seg___z2_s0.bw(bw_fph-1, 0);
	ph_seg___z2_s0.build();
	ph_seg___z2_s0.set_storage (&ph_seg___z2_s0__storage);
	ph_seg_v_z2_s0__storage.add_dim(max_drift-1, 0);
	ph_seg_v_z2_s0__storage.add_dim(5, 0);
	ph_seg_v_z2_s0__storage.bw(seg_ch-1, 0);
	ph_seg_v_z2_s0__storage.build();
	ph_seg_v_z2_s0.add_dim(max_drift-1, 0);
	ph_seg_v_z2_s0.add_dim(5, 0);
	ph_seg_v_z2_s0.bw(seg_ch-1, 0);
	ph_seg_v_z2_s0.build();
	ph_seg_v_z2_s0.set_storage (&ph_seg_v_z2_s0__storage);
	ph_seg___z2_s1__storage.add_dim(max_drift-1, 0);
	ph_seg___z2_s1__storage.add_dim(5, 0);
	ph_seg___z2_s1__storage.add_dim(seg_ch-1, 0);
	ph_seg___z2_s1__storage.bw(bw_fph-1, 0);
	ph_seg___z2_s1__storage.build();
	ph_seg___z2_s1.add_dim(max_drift-1, 0);
	ph_seg___z2_s1.add_dim(5, 0);
	ph_seg___z2_s1.add_dim(seg_ch-1, 0);
	ph_seg___z2_s1.bw(bw_fph-1, 0);
	ph_seg___z2_s1.build();
	ph_seg___z2_s1.set_storage (&ph_seg___z2_s1__storage);
	ph_seg_v_z2_s1__storage.add_dim(max_drift-1, 0);
	ph_seg_v_z2_s1__storage.add_dim(5, 0);
	ph_seg_v_z2_s1__storage.bw(seg_ch-1, 0);
	ph_seg_v_z2_s1__storage.build();
	ph_seg_v_z2_s1.add_dim(max_drift-1, 0);
	ph_seg_v_z2_s1.add_dim(5, 0);
	ph_seg_v_z2_s1.bw(seg_ch-1, 0);
	ph_seg_v_z2_s1.build();
	ph_seg_v_z2_s1.set_storage (&ph_seg_v_z2_s1__storage);
	ph_seg___z2_s2__storage.add_dim(max_drift-1, 0);
	ph_seg___z2_s2__storage.add_dim(5, 0);
	ph_seg___z2_s2__storage.add_dim(seg_ch-1, 0);
	ph_seg___z2_s2__storage.bw(bw_fph-1, 0);
	ph_seg___z2_s2__storage.build();
	ph_seg___z2_s2.add_dim(max_drift-1, 0);
	ph_seg___z2_s2.add_dim(5, 0);
	ph_seg___z2_s2.add_dim(seg_ch-1, 0);
	ph_seg___z2_s2.bw(bw_fph-1, 0);
	ph_seg___z2_s2.build();
	ph_seg___z2_s2.set_storage (&ph_seg___z2_s2__storage);
	ph_seg_v_z2_s2__storage.add_dim(max_drift-1, 0);
	ph_seg_v_z2_s2__storage.add_dim(5, 0);
	ph_seg_v_z2_s2__storage.bw(seg_ch-1, 0);
	ph_seg_v_z2_s2__storage.build();
	ph_seg_v_z2_s2.add_dim(max_drift-1, 0);
	ph_seg_v_z2_s2.add_dim(5, 0);
	ph_seg_v_z2_s2.bw(seg_ch-1, 0);
	ph_seg_v_z2_s2.build();
	ph_seg_v_z2_s2.set_storage (&ph_seg_v_z2_s2__storage);
	ph_seg___z2_s3__storage.add_dim(max_drift-1, 0);
	ph_seg___z2_s3__storage.add_dim(5, 0);
	ph_seg___z2_s3__storage.add_dim(seg_ch-1, 0);
	ph_seg___z2_s3__storage.bw(bw_fph-1, 0);
	ph_seg___z2_s3__storage.build();
	ph_seg___z2_s3.add_dim(max_drift-1, 0);
	ph_seg___z2_s3.add_dim(5, 0);
	ph_seg___z2_s3.add_dim(seg_ch-1, 0);
	ph_seg___z2_s3.bw(bw_fph-1, 0);
	ph_seg___z2_s3.build();
	ph_seg___z2_s3.set_storage (&ph_seg___z2_s3__storage);
	ph_seg_v_z2_s3__storage.add_dim(max_drift-1, 0);
	ph_seg_v_z2_s3__storage.add_dim(5, 0);
	ph_seg_v_z2_s3__storage.bw(seg_ch-1, 0);
	ph_seg_v_z2_s3__storage.build();
	ph_seg_v_z2_s3.add_dim(max_drift-1, 0);
	ph_seg_v_z2_s3.add_dim(5, 0);
	ph_seg_v_z2_s3.bw(seg_ch-1, 0);
	ph_seg_v_z2_s3.build();
	ph_seg_v_z2_s3.set_storage (&ph_seg_v_z2_s3__storage);
	ph_seg___z3_s0__storage.add_dim(max_drift-1, 0);
	ph_seg___z3_s0__storage.add_dim(5, 0);
	ph_seg___z3_s0__storage.add_dim(seg_ch-1, 0);
	ph_seg___z3_s0__storage.bw(bw_fph-1, 0);
	ph_seg___z3_s0__storage.build();
	ph_seg___z3_s0.add_dim(max_drift-1, 0);
	ph_seg___z3_s0.add_dim(5, 0);
	ph_seg___z3_s0.add_dim(seg_ch-1, 0);
	ph_seg___z3_s0.bw(bw_fph-1, 0);
	ph_seg___z3_s0.build();
	ph_seg___z3_s0.set_storage (&ph_seg___z3_s0__storage);
	ph_seg_v_z3_s0__storage.add_dim(max_drift-1, 0);
	ph_seg_v_z3_s0__storage.add_dim(5, 0);
	ph_seg_v_z3_s0__storage.bw(seg_ch-1, 0);
	ph_seg_v_z3_s0__storage.build();
	ph_seg_v_z3_s0.add_dim(max_drift-1, 0);
	ph_seg_v_z3_s0.add_dim(5, 0);
	ph_seg_v_z3_s0.bw(seg_ch-1, 0);
	ph_seg_v_z3_s0.build();
	ph_seg_v_z3_s0.set_storage (&ph_seg_v_z3_s0__storage);
	ph_seg___z3_s1__storage.add_dim(max_drift-1, 0);
	ph_seg___z3_s1__storage.add_dim(5, 0);
	ph_seg___z3_s1__storage.add_dim(seg_ch-1, 0);
	ph_seg___z3_s1__storage.bw(bw_fph-1, 0);
	ph_seg___z3_s1__storage.build();
	ph_seg___z3_s1.add_dim(max_drift-1, 0);
	ph_seg___z3_s1.add_dim(5, 0);
	ph_seg___z3_s1.add_dim(seg_ch-1, 0);
	ph_seg___z3_s1.bw(bw_fph-1, 0);
	ph_seg___z3_s1.build();
	ph_seg___z3_s1.set_storage (&ph_seg___z3_s1__storage);
	ph_seg_v_z3_s1__storage.add_dim(max_drift-1, 0);
	ph_seg_v_z3_s1__storage.add_dim(5, 0);
	ph_seg_v_z3_s1__storage.bw(seg_ch-1, 0);
	ph_seg_v_z3_s1__storage.build();
	ph_seg_v_z3_s1.add_dim(max_drift-1, 0);
	ph_seg_v_z3_s1.add_dim(5, 0);
	ph_seg_v_z3_s1.bw(seg_ch-1, 0);
	ph_seg_v_z3_s1.build();
	ph_seg_v_z3_s1.set_storage (&ph_seg_v_z3_s1__storage);
	ph_seg___z3_s2__storage.add_dim(max_drift-1, 0);
	ph_seg___z3_s2__storage.add_dim(5, 0);
	ph_seg___z3_s2__storage.add_dim(seg_ch-1, 0);
	ph_seg___z3_s2__storage.bw(bw_fph-1, 0);
	ph_seg___z3_s2__storage.build();
	ph_seg___z3_s2.add_dim(max_drift-1, 0);
	ph_seg___z3_s2.add_dim(5, 0);
	ph_seg___z3_s2.add_dim(seg_ch-1, 0);
	ph_seg___z3_s2.bw(bw_fph-1, 0);
	ph_seg___z3_s2.build();
	ph_seg___z3_s2.set_storage (&ph_seg___z3_s2__storage);
	ph_seg_v_z3_s2__storage.add_dim(max_drift-1, 0);
	ph_seg_v_z3_s2__storage.add_dim(5, 0);
	ph_seg_v_z3_s2__storage.bw(seg_ch-1, 0);
	ph_seg_v_z3_s2__storage.build();
	ph_seg_v_z3_s2.add_dim(max_drift-1, 0);
	ph_seg_v_z3_s2.add_dim(5, 0);
	ph_seg_v_z3_s2.bw(seg_ch-1, 0);
	ph_seg_v_z3_s2.build();
	ph_seg_v_z3_s2.set_storage (&ph_seg_v_z3_s2__storage);
	ph_seg___z3_s3__storage.add_dim(max_drift-1, 0);
	ph_seg___z3_s3__storage.add_dim(5, 0);
	ph_seg___z3_s3__storage.add_dim(seg_ch-1, 0);
	ph_seg___z3_s3__storage.bw(bw_fph-1, 0);
	ph_seg___z3_s3__storage.build();
	ph_seg___z3_s3.add_dim(max_drift-1, 0);
	ph_seg___z3_s3.add_dim(5, 0);
	ph_seg___z3_s3.add_dim(seg_ch-1, 0);
	ph_seg___z3_s3.bw(bw_fph-1, 0);
	ph_seg___z3_s3.build();
	ph_seg___z3_s3.set_storage (&ph_seg___z3_s3__storage);
	ph_seg_v_z3_s3__storage.add_dim(max_drift-1, 0);
	ph_seg_v_z3_s3__storage.add_dim(5, 0);
	ph_seg_v_z3_s3__storage.bw(seg_ch-1, 0);
	ph_seg_v_z3_s3__storage.build();
	ph_seg_v_z3_s3.add_dim(max_drift-1, 0);
	ph_seg_v_z3_s3.add_dim(5, 0);
	ph_seg_v_z3_s3.bw(seg_ch-1, 0);
	ph_seg_v_z3_s3.build();
	ph_seg_v_z3_s3.set_storage (&ph_seg_v_z3_s3__storage);
	th_seg___z0_s0__storage.add_dim(max_drift-1, 0);
	th_seg___z0_s0__storage.add_dim(5, 0);
	th_seg___z0_s0__storage.add_dim(th_ch11-1, 0);
	th_seg___z0_s0__storage.bw(bw_th-1, 0);
	th_seg___z0_s0__storage.build();
	th_seg___z0_s0.add_dim(max_drift-1, 0);
	th_seg___z0_s0.add_dim(5, 0);
	th_seg___z0_s0.add_dim(th_ch11-1, 0);
	th_seg___z0_s0.bw(bw_th-1, 0);
	th_seg___z0_s0.build();
	th_seg___z0_s0.set_storage (&th_seg___z0_s0__storage);
	th_seg___z0_s1__storage.add_dim(max_drift-1, 0);
	th_seg___z0_s1__storage.add_dim(2, 0);
	th_seg___z0_s1__storage.add_dim(seg_ch-1, 0);
	th_seg___z0_s1__storage.bw(bw_th-1, 0);
	th_seg___z0_s1__storage.build();
	th_seg___z0_s1.add_dim(max_drift-1, 0);
	th_seg___z0_s1.add_dim(2, 0);
	th_seg___z0_s1.add_dim(seg_ch-1, 0);
	th_seg___z0_s1.bw(bw_th-1, 0);
	th_seg___z0_s1.build();
	th_seg___z0_s1.set_storage (&th_seg___z0_s1__storage);
	th_seg___z0_s2__storage.add_dim(max_drift-1, 0);
	th_seg___z0_s2__storage.add_dim(2, 0);
	th_seg___z0_s2__storage.add_dim(seg_ch-1, 0);
	th_seg___z0_s2__storage.bw(bw_th-1, 0);
	th_seg___z0_s2__storage.build();
	th_seg___z0_s2.add_dim(max_drift-1, 0);
	th_seg___z0_s2.add_dim(2, 0);
	th_seg___z0_s2.add_dim(seg_ch-1, 0);
	th_seg___z0_s2.bw(bw_th-1, 0);
	th_seg___z0_s2.build();
	th_seg___z0_s2.set_storage (&th_seg___z0_s2__storage);
	th_seg___z0_s3__storage.add_dim(max_drift-1, 0);
	th_seg___z0_s3__storage.add_dim(2, 0);
	th_seg___z0_s3__storage.add_dim(seg_ch-1, 0);
	th_seg___z0_s3__storage.bw(bw_th-1, 0);
	th_seg___z0_s3__storage.build();
	th_seg___z0_s3.add_dim(max_drift-1, 0);
	th_seg___z0_s3.add_dim(2, 0);
	th_seg___z0_s3.add_dim(seg_ch-1, 0);
	th_seg___z0_s3.bw(bw_th-1, 0);
	th_seg___z0_s3.build();
	th_seg___z0_s3.set_storage (&th_seg___z0_s3__storage);
	th_seg___z1_s0__storage.add_dim(max_drift-1, 0);
	th_seg___z1_s0__storage.add_dim(5, 0);
	th_seg___z1_s0__storage.add_dim(th_ch11-1, 0);
	th_seg___z1_s0__storage.bw(bw_th-1, 0);
	th_seg___z1_s0__storage.build();
	th_seg___z1_s0.add_dim(max_drift-1, 0);
	th_seg___z1_s0.add_dim(5, 0);
	th_seg___z1_s0.add_dim(th_ch11-1, 0);
	th_seg___z1_s0.bw(bw_th-1, 0);
	th_seg___z1_s0.build();
	th_seg___z1_s0.set_storage (&th_seg___z1_s0__storage);
	th_seg___z1_s1__storage.add_dim(max_drift-1, 0);
	th_seg___z1_s1__storage.add_dim(2, 0);
	th_seg___z1_s1__storage.add_dim(seg_ch-1, 0);
	th_seg___z1_s1__storage.bw(bw_th-1, 0);
	th_seg___z1_s1__storage.build();
	th_seg___z1_s1.add_dim(max_drift-1, 0);
	th_seg___z1_s1.add_dim(2, 0);
	th_seg___z1_s1.add_dim(seg_ch-1, 0);
	th_seg___z1_s1.bw(bw_th-1, 0);
	th_seg___z1_s1.build();
	th_seg___z1_s1.set_storage (&th_seg___z1_s1__storage);
	th_seg___z1_s2__storage.add_dim(max_drift-1, 0);
	th_seg___z1_s2__storage.add_dim(5, 0);
	th_seg___z1_s2__storage.add_dim(seg_ch-1, 0);
	th_seg___z1_s2__storage.bw(bw_th-1, 0);
	th_seg___z1_s2__storage.build();
	th_seg___z1_s2.add_dim(max_drift-1, 0);
	th_seg___z1_s2.add_dim(5, 0);
	th_seg___z1_s2.add_dim(seg_ch-1, 0);
	th_seg___z1_s2.bw(bw_th-1, 0);
	th_seg___z1_s2.build();
	th_seg___z1_s2.set_storage (&th_seg___z1_s2__storage);
	th_seg___z1_s3__storage.add_dim(max_drift-1, 0);
	th_seg___z1_s3__storage.add_dim(5, 0);
	th_seg___z1_s3__storage.add_dim(seg_ch-1, 0);
	th_seg___z1_s3__storage.bw(bw_th-1, 0);
	th_seg___z1_s3__storage.build();
	th_seg___z1_s3.add_dim(max_drift-1, 0);
	th_seg___z1_s3.add_dim(5, 0);
	th_seg___z1_s3.add_dim(seg_ch-1, 0);
	th_seg___z1_s3.bw(bw_th-1, 0);
	th_seg___z1_s3.build();
	th_seg___z1_s3.set_storage (&th_seg___z1_s3__storage);
	th_seg___z2_s0__storage.add_dim(max_drift-1, 0);
	th_seg___z2_s0__storage.add_dim(5, 0);
	th_seg___z2_s0__storage.add_dim(seg_ch-1, 0);
	th_seg___z2_s0__storage.bw(bw_th-1, 0);
	th_seg___z2_s0__storage.build();
	th_seg___z2_s0.add_dim(max_drift-1, 0);
	th_seg___z2_s0.add_dim(5, 0);
	th_seg___z2_s0.add_dim(seg_ch-1, 0);
	th_seg___z2_s0.bw(bw_th-1, 0);
	th_seg___z2_s0.build();
	th_seg___z2_s0.set_storage (&th_seg___z2_s0__storage);
	th_seg___z2_s1__storage.add_dim(max_drift-1, 0);
	th_seg___z2_s1__storage.add_dim(5, 0);
	th_seg___z2_s1__storage.add_dim(seg_ch-1, 0);
	th_seg___z2_s1__storage.bw(bw_th-1, 0);
	th_seg___z2_s1__storage.build();
	th_seg___z2_s1.add_dim(max_drift-1, 0);
	th_seg___z2_s1.add_dim(5, 0);
	th_seg___z2_s1.add_dim(seg_ch-1, 0);
	th_seg___z2_s1.bw(bw_th-1, 0);
	th_seg___z2_s1.build();
	th_seg___z2_s1.set_storage (&th_seg___z2_s1__storage);
	th_seg___z2_s2__storage.add_dim(max_drift-1, 0);
	th_seg___z2_s2__storage.add_dim(5, 0);
	th_seg___z2_s2__storage.add_dim(seg_ch-1, 0);
	th_seg___z2_s2__storage.bw(bw_th-1, 0);
	th_seg___z2_s2__storage.build();
	th_seg___z2_s2.add_dim(max_drift-1, 0);
	th_seg___z2_s2.add_dim(5, 0);
	th_seg___z2_s2.add_dim(seg_ch-1, 0);
	th_seg___z2_s2.bw(bw_th-1, 0);
	th_seg___z2_s2.build();
	th_seg___z2_s2.set_storage (&th_seg___z2_s2__storage);
	th_seg___z2_s3__storage.add_dim(max_drift-1, 0);
	th_seg___z2_s3__storage.add_dim(5, 0);
	th_seg___z2_s3__storage.add_dim(seg_ch-1, 0);
	th_seg___z2_s3__storage.bw(bw_th-1, 0);
	th_seg___z2_s3__storage.build();
	th_seg___z2_s3.add_dim(max_drift-1, 0);
	th_seg___z2_s3.add_dim(5, 0);
	th_seg___z2_s3.add_dim(seg_ch-1, 0);
	th_seg___z2_s3.bw(bw_th-1, 0);
	th_seg___z2_s3.build();
	th_seg___z2_s3.set_storage (&th_seg___z2_s3__storage);
	th_seg___z3_s0__storage.add_dim(max_drift-1, 0);
	th_seg___z3_s0__storage.add_dim(5, 0);
	th_seg___z3_s0__storage.add_dim(seg_ch-1, 0);
	th_seg___z3_s0__storage.bw(bw_th-1, 0);
	th_seg___z3_s0__storage.build();
	th_seg___z3_s0.add_dim(max_drift-1, 0);
	th_seg___z3_s0.add_dim(5, 0);
	th_seg___z3_s0.add_dim(seg_ch-1, 0);
	th_seg___z3_s0.bw(bw_th-1, 0);
	th_seg___z3_s0.build();
	th_seg___z3_s0.set_storage (&th_seg___z3_s0__storage);
	th_seg___z3_s1__storage.add_dim(max_drift-1, 0);
	th_seg___z3_s1__storage.add_dim(5, 0);
	th_seg___z3_s1__storage.add_dim(seg_ch-1, 0);
	th_seg___z3_s1__storage.bw(bw_th-1, 0);
	th_seg___z3_s1__storage.build();
	th_seg___z3_s1.add_dim(max_drift-1, 0);
	th_seg___z3_s1.add_dim(5, 0);
	th_seg___z3_s1.add_dim(seg_ch-1, 0);
	th_seg___z3_s1.bw(bw_th-1, 0);
	th_seg___z3_s1.build();
	th_seg___z3_s1.set_storage (&th_seg___z3_s1__storage);
	th_seg___z3_s2__storage.add_dim(max_drift-1, 0);
	th_seg___z3_s2__storage.add_dim(5, 0);
	th_seg___z3_s2__storage.add_dim(seg_ch-1, 0);
	th_seg___z3_s2__storage.bw(bw_th-1, 0);
	th_seg___z3_s2__storage.build();
	th_seg___z3_s2.add_dim(max_drift-1, 0);
	th_seg___z3_s2.add_dim(5, 0);
	th_seg___z3_s2.add_dim(seg_ch-1, 0);
	th_seg___z3_s2.bw(bw_th-1, 0);
	th_seg___z3_s2.build();
	th_seg___z3_s2.set_storage (&th_seg___z3_s2__storage);
	th_seg___z3_s3__storage.add_dim(max_drift-1, 0);
	th_seg___z3_s3__storage.add_dim(5, 0);
	th_seg___z3_s3__storage.add_dim(seg_ch-1, 0);
	th_seg___z3_s3__storage.bw(bw_th-1, 0);
	th_seg___z3_s3__storage.build();
	th_seg___z3_s3.add_dim(max_drift-1, 0);
	th_seg___z3_s3.add_dim(5, 0);
	th_seg___z3_s3.add_dim(seg_ch-1, 0);
	th_seg___z3_s3.bw(bw_th-1, 0);
	th_seg___z3_s3.build();
	th_seg___z3_s3.set_storage (&th_seg___z3_s3__storage);
	cpat_seg___z0_s0__storage.add_dim(max_drift-1, 0);
	cpat_seg___z0_s0__storage.add_dim(5, 0);
	cpat_seg___z0_s0__storage.add_dim(seg_ch-1, 0);
	cpat_seg___z0_s0__storage.bw(3, 0);
	cpat_seg___z0_s0__storage.build();
	cpat_seg___z0_s0.add_dim(max_drift-1, 0);
	cpat_seg___z0_s0.add_dim(5, 0);
	cpat_seg___z0_s0.add_dim(seg_ch-1, 0);
	cpat_seg___z0_s0.bw(3, 0);
	cpat_seg___z0_s0.build();
	cpat_seg___z0_s0.set_storage (&cpat_seg___z0_s0__storage);
	cpat_seg___z0_s1__storage.add_dim(max_drift-1, 0);
	cpat_seg___z0_s1__storage.add_dim(2, 0);
	cpat_seg___z0_s1__storage.add_dim(seg_ch-1, 0);
	cpat_seg___z0_s1__storage.bw(3, 0);
	cpat_seg___z0_s1__storage.build();
	cpat_seg___z0_s1.add_dim(max_drift-1, 0);
	cpat_seg___z0_s1.add_dim(2, 0);
	cpat_seg___z0_s1.add_dim(seg_ch-1, 0);
	cpat_seg___z0_s1.bw(3, 0);
	cpat_seg___z0_s1.build();
	cpat_seg___z0_s1.set_storage (&cpat_seg___z0_s1__storage);
	cpat_seg___z0_s2__storage.add_dim(max_drift-1, 0);
	cpat_seg___z0_s2__storage.add_dim(2, 0);
	cpat_seg___z0_s2__storage.add_dim(seg_ch-1, 0);
	cpat_seg___z0_s2__storage.bw(3, 0);
	cpat_seg___z0_s2__storage.build();
	cpat_seg___z0_s2.add_dim(max_drift-1, 0);
	cpat_seg___z0_s2.add_dim(2, 0);
	cpat_seg___z0_s2.add_dim(seg_ch-1, 0);
	cpat_seg___z0_s2.bw(3, 0);
	cpat_seg___z0_s2.build();
	cpat_seg___z0_s2.set_storage (&cpat_seg___z0_s2__storage);
	cpat_seg___z0_s3__storage.add_dim(max_drift-1, 0);
	cpat_seg___z0_s3__storage.add_dim(2, 0);
	cpat_seg___z0_s3__storage.add_dim(seg_ch-1, 0);
	cpat_seg___z0_s3__storage.bw(3, 0);
	cpat_seg___z0_s3__storage.build();
	cpat_seg___z0_s3.add_dim(max_drift-1, 0);
	cpat_seg___z0_s3.add_dim(2, 0);
	cpat_seg___z0_s3.add_dim(seg_ch-1, 0);
	cpat_seg___z0_s3.bw(3, 0);
	cpat_seg___z0_s3.build();
	cpat_seg___z0_s3.set_storage (&cpat_seg___z0_s3__storage);
	cpat_seg___z1_s0__storage.add_dim(max_drift-1, 0);
	cpat_seg___z1_s0__storage.add_dim(5, 0);
	cpat_seg___z1_s0__storage.add_dim(seg_ch-1, 0);
	cpat_seg___z1_s0__storage.bw(3, 0);
	cpat_seg___z1_s0__storage.build();
	cpat_seg___z1_s0.add_dim(max_drift-1, 0);
	cpat_seg___z1_s0.add_dim(5, 0);
	cpat_seg___z1_s0.add_dim(seg_ch-1, 0);
	cpat_seg___z1_s0.bw(3, 0);
	cpat_seg___z1_s0.build();
	cpat_seg___z1_s0.set_storage (&cpat_seg___z1_s0__storage);
	cpat_seg___z1_s1__storage.add_dim(max_drift-1, 0);
	cpat_seg___z1_s1__storage.add_dim(2, 0);
	cpat_seg___z1_s1__storage.add_dim(seg_ch-1, 0);
	cpat_seg___z1_s1__storage.bw(3, 0);
	cpat_seg___z1_s1__storage.build();
	cpat_seg___z1_s1.add_dim(max_drift-1, 0);
	cpat_seg___z1_s1.add_dim(2, 0);
	cpat_seg___z1_s1.add_dim(seg_ch-1, 0);
	cpat_seg___z1_s1.bw(3, 0);
	cpat_seg___z1_s1.build();
	cpat_seg___z1_s1.set_storage (&cpat_seg___z1_s1__storage);
	cpat_seg___z1_s2__storage.add_dim(max_drift-1, 0);
	cpat_seg___z1_s2__storage.add_dim(5, 0);
	cpat_seg___z1_s2__storage.add_dim(seg_ch-1, 0);
	cpat_seg___z1_s2__storage.bw(3, 0);
	cpat_seg___z1_s2__storage.build();
	cpat_seg___z1_s2.add_dim(max_drift-1, 0);
	cpat_seg___z1_s2.add_dim(5, 0);
	cpat_seg___z1_s2.add_dim(seg_ch-1, 0);
	cpat_seg___z1_s2.bw(3, 0);
	cpat_seg___z1_s2.build();
	cpat_seg___z1_s2.set_storage (&cpat_seg___z1_s2__storage);
	cpat_seg___z1_s3__storage.add_dim(max_drift-1, 0);
	cpat_seg___z1_s3__storage.add_dim(5, 0);
	cpat_seg___z1_s3__storage.add_dim(seg_ch-1, 0);
	cpat_seg___z1_s3__storage.bw(3, 0);
	cpat_seg___z1_s3__storage.build();
	cpat_seg___z1_s3.add_dim(max_drift-1, 0);
	cpat_seg___z1_s3.add_dim(5, 0);
	cpat_seg___z1_s3.add_dim(seg_ch-1, 0);
	cpat_seg___z1_s3.bw(3, 0);
	cpat_seg___z1_s3.build();
	cpat_seg___z1_s3.set_storage (&cpat_seg___z1_s3__storage);
	cpat_seg___z2_s0__storage.add_dim(max_drift-1, 0);
	cpat_seg___z2_s0__storage.add_dim(5, 0);
	cpat_seg___z2_s0__storage.add_dim(seg_ch-1, 0);
	cpat_seg___z2_s0__storage.bw(3, 0);
	cpat_seg___z2_s0__storage.build();
	cpat_seg___z2_s0.add_dim(max_drift-1, 0);
	cpat_seg___z2_s0.add_dim(5, 0);
	cpat_seg___z2_s0.add_dim(seg_ch-1, 0);
	cpat_seg___z2_s0.bw(3, 0);
	cpat_seg___z2_s0.build();
	cpat_seg___z2_s0.set_storage (&cpat_seg___z2_s0__storage);
	cpat_seg___z2_s1__storage.add_dim(max_drift-1, 0);
	cpat_seg___z2_s1__storage.add_dim(5, 0);
	cpat_seg___z2_s1__storage.add_dim(seg_ch-1, 0);
	cpat_seg___z2_s1__storage.bw(3, 0);
	cpat_seg___z2_s1__storage.build();
	cpat_seg___z2_s1.add_dim(max_drift-1, 0);
	cpat_seg___z2_s1.add_dim(5, 0);
	cpat_seg___z2_s1.add_dim(seg_ch-1, 0);
	cpat_seg___z2_s1.bw(3, 0);
	cpat_seg___z2_s1.build();
	cpat_seg___z2_s1.set_storage (&cpat_seg___z2_s1__storage);
	cpat_seg___z2_s2__storage.add_dim(max_drift-1, 0);
	cpat_seg___z2_s2__storage.add_dim(5, 0);
	cpat_seg___z2_s2__storage.add_dim(seg_ch-1, 0);
	cpat_seg___z2_s2__storage.bw(3, 0);
	cpat_seg___z2_s2__storage.build();
	cpat_seg___z2_s2.add_dim(max_drift-1, 0);
	cpat_seg___z2_s2.add_dim(5, 0);
	cpat_seg___z2_s2.add_dim(seg_ch-1, 0);
	cpat_seg___z2_s2.bw(3, 0);
	cpat_seg___z2_s2.build();
	cpat_seg___z2_s2.set_storage (&cpat_seg___z2_s2__storage);
	cpat_seg___z2_s3__storage.add_dim(max_drift-1, 0);
	cpat_seg___z2_s3__storage.add_dim(5, 0);
	cpat_seg___z2_s3__storage.add_dim(seg_ch-1, 0);
	cpat_seg___z2_s3__storage.bw(3, 0);
	cpat_seg___z2_s3__storage.build();
	cpat_seg___z2_s3.add_dim(max_drift-1, 0);
	cpat_seg___z2_s3.add_dim(5, 0);
	cpat_seg___z2_s3.add_dim(seg_ch-1, 0);
	cpat_seg___z2_s3.bw(3, 0);
	cpat_seg___z2_s3.build();
	cpat_seg___z2_s3.set_storage (&cpat_seg___z2_s3__storage);
	cpat_seg___z3_s0__storage.add_dim(max_drift-1, 0);
	cpat_seg___z3_s0__storage.add_dim(5, 0);
	cpat_seg___z3_s0__storage.add_dim(seg_ch-1, 0);
	cpat_seg___z3_s0__storage.bw(3, 0);
	cpat_seg___z3_s0__storage.build();
	cpat_seg___z3_s0.add_dim(max_drift-1, 0);
	cpat_seg___z3_s0.add_dim(5, 0);
	cpat_seg___z3_s0.add_dim(seg_ch-1, 0);
	cpat_seg___z3_s0.bw(3, 0);
	cpat_seg___z3_s0.build();
	cpat_seg___z3_s0.set_storage (&cpat_seg___z3_s0__storage);
	cpat_seg___z3_s1__storage.add_dim(max_drift-1, 0);
	cpat_seg___z3_s1__storage.add_dim(5, 0);
	cpat_seg___z3_s1__storage.add_dim(seg_ch-1, 0);
	cpat_seg___z3_s1__storage.bw(3, 0);
	cpat_seg___z3_s1__storage.build();
	cpat_seg___z3_s1.add_dim(max_drift-1, 0);
	cpat_seg___z3_s1.add_dim(5, 0);
	cpat_seg___z3_s1.add_dim(seg_ch-1, 0);
	cpat_seg___z3_s1.bw(3, 0);
	cpat_seg___z3_s1.build();
	cpat_seg___z3_s1.set_storage (&cpat_seg___z3_s1__storage);
	cpat_seg___z3_s2__storage.add_dim(max_drift-1, 0);
	cpat_seg___z3_s2__storage.add_dim(5, 0);
	cpat_seg___z3_s2__storage.add_dim(seg_ch-1, 0);
	cpat_seg___z3_s2__storage.bw(3, 0);
	cpat_seg___z3_s2__storage.build();
	cpat_seg___z3_s2.add_dim(max_drift-1, 0);
	cpat_seg___z3_s2.add_dim(5, 0);
	cpat_seg___z3_s2.add_dim(seg_ch-1, 0);
	cpat_seg___z3_s2.bw(3, 0);
	cpat_seg___z3_s2.build();
	cpat_seg___z3_s2.set_storage (&cpat_seg___z3_s2__storage);
	cpat_seg___z3_s3__storage.add_dim(max_drift-1, 0);
	cpat_seg___z3_s3__storage.add_dim(5, 0);
	cpat_seg___z3_s3__storage.add_dim(seg_ch-1, 0);
	cpat_seg___z3_s3__storage.bw(3, 0);
	cpat_seg___z3_s3__storage.build();
	cpat_seg___z3_s3.add_dim(max_drift-1, 0);
	cpat_seg___z3_s3.add_dim(5, 0);
	cpat_seg___z3_s3.add_dim(seg_ch-1, 0);
	cpat_seg___z3_s3.bw(3, 0);
	cpat_seg___z3_s3.build();
	cpat_seg___z3_s3.set_storage (&cpat_seg___z3_s3__storage);

}

// vppc: this function checks for changes in any signal on each simulation iteration
void match_ph_segments::init ()
{
	if (!built)
	{
			}
	else
	{
		ph_seg___z0_s0__storage.init();
		ph_seg_v_z0_s0__storage.init();
		ph_seg___z0_s1__storage.init();
		ph_seg_v_z0_s1__storage.init();
		ph_seg___z0_s2__storage.init();
		ph_seg_v_z0_s2__storage.init();
		ph_seg___z0_s3__storage.init();
		ph_seg_v_z0_s3__storage.init();
		ph_seg___z1_s0__storage.init();
		ph_seg_v_z1_s0__storage.init();
		ph_seg___z1_s1__storage.init();
		ph_seg_v_z1_s1__storage.init();
		ph_seg___z1_s2__storage.init();
		ph_seg_v_z1_s2__storage.init();
		ph_seg___z1_s3__storage.init();
		ph_seg_v_z1_s3__storage.init();
		ph_seg___z2_s0__storage.init();
		ph_seg_v_z2_s0__storage.init();
		ph_seg___z2_s1__storage.init();
		ph_seg_v_z2_s1__storage.init();
		ph_seg___z2_s2__storage.init();
		ph_seg_v_z2_s2__storage.init();
		ph_seg___z2_s3__storage.init();
		ph_seg_v_z2_s3__storage.init();
		ph_seg___z3_s0__storage.init();
		ph_seg_v_z3_s0__storage.init();
		ph_seg___z3_s1__storage.init();
		ph_seg_v_z3_s1__storage.init();
		ph_seg___z3_s2__storage.init();
		ph_seg_v_z3_s2__storage.init();
		ph_seg___z3_s3__storage.init();
		ph_seg_v_z3_s3__storage.init();
		th_seg___z0_s0__storage.init();
		th_seg___z0_s1__storage.init();
		th_seg___z0_s2__storage.init();
		th_seg___z0_s3__storage.init();
		th_seg___z1_s0__storage.init();
		th_seg___z1_s1__storage.init();
		th_seg___z1_s2__storage.init();
		th_seg___z1_s3__storage.init();
		th_seg___z2_s0__storage.init();
		th_seg___z2_s1__storage.init();
		th_seg___z2_s2__storage.init();
		th_seg___z2_s3__storage.init();
		th_seg___z3_s0__storage.init();
		th_seg___z3_s1__storage.init();
		th_seg___z3_s2__storage.init();
		th_seg___z3_s3__storage.init();
		cpat_seg___z0_s0__storage.init();
		cpat_seg___z0_s1__storage.init();
		cpat_seg___z0_s2__storage.init();
		cpat_seg___z0_s3__storage.init();
		cpat_seg___z1_s0__storage.init();
		cpat_seg___z1_s1__storage.init();
		cpat_seg___z1_s2__storage.init();
		cpat_seg___z1_s3__storage.init();
		cpat_seg___z2_s0__storage.init();
		cpat_seg___z2_s1__storage.init();
		cpat_seg___z2_s2__storage.init();
		cpat_seg___z2_s3__storage.init();
		cpat_seg___z3_s0__storage.init();
		cpat_seg___z3_s1__storage.init();
		cpat_seg___z3_s2__storage.init();
		cpat_seg___z3_s3__storage.init();
																																															gb.init();
	}
}
void match_ph_segments::gb__class::init()
{
	for (map <ull, fs_loop__class>::iterator mit = fs_loop.begin(); mit != fs_loop.end(); mit++)
		mit->second.init();
}
void match_ph_segments::gb__class::fs_loop__class::init()
{
	fs_00.init();
	fs_01.init();
	fs_02.init();
	fs_03.init();
	fs_10.init();
	fs_11.init();
	fs_12.init();
	fs_13.init();
	fs_20.init();
	fs_21.init();
	fs_22.init();
	fs_23.init();
	fs_30.init();
	fs_31.init();
	fs_32.init();
	fs_33.init();
}
