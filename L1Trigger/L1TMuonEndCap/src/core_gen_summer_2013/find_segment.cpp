// This C++ source file was automatically generated
// by VPPC from a Verilog HDL project.
// VPPC web-page: http://www.phys.ufl.edu/~madorsky/vppc/

// Author    : madorsky
// Timestamp : Thu Mar 12 14:54:00 2015

#include "find_segment.h"

extern size_t __glob_alwaysn__;

signal_ & find_segment::comp3_class::operator()
(
	signal_& a__io,
	signal_& b__io,
	signal_& c__io
)
{
	if (!built)
	{
		build();
	}
		a.attach(a__io);
		b.attach(b__io);
		c.attach(c__io);

		{
			r[2] = a <= b;
			r[1] = b <= c;
			r[0] = c <= a;
			if (( (r) == 0)) { comp3_retval = const_(2, 0x3UL); } else 
			if (( (r) == // invalid
				1)) { comp3_retval = const_(2, 0x2UL); } else 
			if (( (r) == // c
				2)) { comp3_retval = const_(2, 0x1UL); } else 
			if (( (r) == // b
				3)) { comp3_retval = const_(2, 0x1UL); } else 
			if (( (r) == // b
				4)) { comp3_retval = const_(2, 0x0UL); } else 
			if (( (r) == // a
				5)) { comp3_retval = const_(2, 0x2UL); } else 
			if (( (r) == // c
				6)) { comp3_retval = const_(2, 0x0UL); } else 
			if (( (r) == // a
				7)) { comp3_retval = const_(2, 0x0UL); } 
		}
	return comp3_retval;

}

// vppc: code below this point are service functions, should not require your attention

// vppc: this function assigns initial values to parameters and localparams
void find_segment::comp3_class::defparam()
{
}

// vppc: this function allocates memory for internal signals
void find_segment::comp3_class::build()
{
	built = true;
	a.bw(4, 0);
	b.bw(4, 0);
	c.bw(4, 0);
	r__storage.bw(2, 0);
	r.bw(2, 0);
	r.set_storage (&r__storage);
	comp3_retval__storage.bw(1, 0);
	comp3_retval.bw(1, 0);
	comp3_retval.set_storage (&comp3_retval__storage);

}

// vppc: this function checks for changes in any signal on each simulation iteration
void find_segment::comp3_class::init ()
{
	if (!built)
	{
	}
	else
	{
		r__storage.init();
		comp3_retval__storage.init();
	}
}
void find_segment::operator()
(
	signal_& ph_pat_p__io,
	signal_& ph_pat_q_p__io,
	signal_& ph_seg_p__io,
	signal_& ph_seg_v_p__io,
	signal_& th_seg_p__io,
	signal_& cpat_seg_p__io,
	signal_& vid__io,
	signal_& hid__io,
	signal_& cid__io,
	signal_& sid__io,
	signal_& ph_match__io,
	signal_& th_match__io,
	signal_& cpat_match__io,
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
		max_ph_diff = station == 1 ? const_(7, 15UL) : const_(7, 7UL);
		bw_phdiff = station == 1 ? 5 : 4;
		tot_diff = max_drift*zone_cham*seg_ch;
		nodiff = station == 1 ? const_(5, 31UL) : const_(4, 15UL);
		build();
		ph_pat_p.attach(ph_pat_p__io);
		ph_pat_q_p.attach(ph_pat_q_p__io);
		// ph from segments [bx_history][chamber][segment]
// segments are coming from chambers in the interesting zone only
// for example, in zone 0 ME1 segments should come from chambers
// subsector1: 1,2,3, subsector2: 1,2,3
		ph_seg_p.attach(ph_seg_p__io);
		// valid flags for segments
		ph_seg_v_p.attach(ph_seg_v_p__io);
		th_seg_p.attach(th_seg_p__io);
		cpat_seg_p.attach(cpat_seg_p__io);
		clk.attach(clk__io);
		// indexes of best match
		vid.attach(vid__io);
		hid.attach(hid__io);
		cid.attach(cid__io);
		sid.attach(sid__io);
		ph_match.attach(ph_match__io);
		// all th's from matching chamber, we don't know which one will fit best
		th_match.attach(th_match__io);
		cpat_match.attach(cpat_match__io);
	}


	beginalways();
	
	if (posedge (clk))
	{
		ph_pat = ph_pat_p;
		ph_pat_v = ph_pat_q_p != 0; // non-zero quality means valid pattern
		ph_seg = ph_seg_p;
		ph_seg_v = ph_seg_v_p;
		th_seg = th_seg_p;
		cpat_seg = cpat_seg_p;

		// calculate abs differences
		di = 0;
		for (i = 0; i < max_drift; i = i+1) // history loop
		{
			for (j = 0; j < zone_cham; j = j+1) // chamber loop
			{
				for (k = 0; k < seg_ch; k = k+1) // segment loop
				{
					// remove unused low bits from segment ph
					ph_segr = ph_seg[i][j][k](bw_fph-1 , bw_fph-bpow-1);

					// get abs difference
					if (ph_seg_v[i][j][k])
 					    ph_diff_tmp = (ph_pat > ph_segr) ? ph_pat - ph_segr : ph_segr - ph_pat;
					else
						ph_diff_tmp = nodiff; // if segment invalid put max value into diff

				    if (ph_diff_tmp > max_ph_diff)
 					    ph_diff[i*zone_cham*seg_ch + j*seg_ch + k] = nodiff; // difference is too high, cannot be the same pattern
				    else
				 	    ph_diff[i*zone_cham*seg_ch + j*seg_ch + k] = ph_diff_tmp(bw_phdiff-1,0);
				   

					ri = i;
					rj = j;
					rk = k;
					// diffi variables carry track indexes
					diffi0[i*zone_cham*seg_ch + j*seg_ch + k] = (ri, rj, rk);
				}
			}
		} // for (i = 0; i < max_drift; i = i+1)

		// sort differences
		// first stage
		for (i = 0; i < tot_diff/3; i = i+1)
		{
			// compare 3 values
			rcomp = comp3(ph_diff[i*3], ph_diff[i*3+1], ph_diff[i*3+2]);
			if (( (rcomp) == 0)) {  { cmp1[i] = ph_diff[i*3+0]; diffi1[i] = diffi0[i*3+0]; } } else 
			if (( (rcomp) == 1)) {  { cmp1[i] = ph_diff[i*3+1]; diffi1[i] = diffi0[i*3+1]; } } else 
			if (( (rcomp) == 2)) {  { cmp1[i] = ph_diff[i*3+2]; diffi1[i] = diffi0[i*3+2]; } } 
		}

		// second stage
		for (i = 0; i < tot_diff/9; i = i+1)
		{
			// compare 3 values
			rcomp = comp3(cmp1[i*3], cmp1[i*3+1], cmp1[i*3+2]);
			if (( (rcomp) == 0)) {  { cmp2[i] = cmp1[i*3+0]; diffi2[i] = diffi1[i*3+0]; } } else 
			if (( (rcomp) == 1)) {  { cmp2[i] = cmp1[i*3+1]; diffi2[i] = diffi1[i*3+1]; } } else 
			if (( (rcomp) == 2)) {  { cmp2[i] = cmp1[i*3+2]; diffi2[i] = diffi1[i*3+2]; } } 
		}

		// third stage
		for (i = 0; i < tot_diff/18; i = i+1)
		{
			// compare 2 values
			rcomp[0] = cmp2[i*2] >= cmp2[i*2+1];
			if (( (rcomp[0]) == 0)) {  { cmp3[i] = cmp2[i*2+0]; diffi3[i] = diffi2[i*2+0]; } } else 
			if (( (rcomp[0]) == 1)) {  { cmp3[i] = cmp2[i*2+1]; diffi3[i] = diffi2[i*2+1]; } } 
		}

		// last stage depends on number of input segments
		if (tot_diff == 36)
		{
			// compare 2 values
			rcomp[0] = cmp3[0] >= cmp3[1];
			if (( (rcomp[0]) == 0)) {  { cmp4 = cmp3[0]; diffi4 = diffi3[0]; } } else 
			if (( (rcomp[0]) == 1)) {  { cmp4 = cmp3[1]; diffi4 = diffi3[1]; } } 
		}
		else
		{
			cmp4 = cmp3[0]; 
			diffi4 = diffi3[0];
		}

		(hid, cid, sid) = diffi4;
		vid = ph_seg_v[hid][cid][sid];
		// if pattern invalid | all differences invalid remove valid flags 
		if (!ph_pat_v || cmp4 == nodiff) vid = 0;
		
		ph_match = ph_seg[hid][cid][sid]; // route best matching phi to output
		th_match = th_seg[hid][cid]; // route all th coords from matching chamber to output
		cpat_match = cpat_seg[hid][cid][sid]; // route pattern to output
	}
	endalways();
}

// vppc: code below this point are service functions, should not require your attention

// vppc: this function assigns initial values to parameters and localparams
void find_segment::defparam()
{
	station = 1;
	cscid = 1;
	zone_cham = 6;
	zone_seg = 2;
}

// vppc: this function allocates memory for internal signals
void find_segment::build()
{
	built = true;
	ph_pat_p.bw(bpow, 0);
	ph_pat_q_p.bw(5, 0);
	ph_seg_p.add_dim(max_drift-1, 0);
	ph_seg_p.add_dim(zone_cham-1, 0);
	ph_seg_p.add_dim(seg_ch-1, 0);
	ph_seg_p.bw(bw_fph-1, 0);
	ph_seg_p.build();
	ph_seg_v_p.add_dim(max_drift-1, 0);
	ph_seg_v_p.add_dim(zone_cham-1, 0);
	ph_seg_v_p.bw(seg_ch-1, 0);
	ph_seg_v_p.build();
	th_seg_p.add_dim(max_drift-1, 0);
	th_seg_p.add_dim(zone_cham-1, 0);
	th_seg_p.add_dim(zone_seg-1, 0);
	th_seg_p.bw(bw_th-1, 0);
	th_seg_p.build();
	cpat_seg_p.add_dim(max_drift-1, 0);
	cpat_seg_p.add_dim(zone_cham-1, 0);
	cpat_seg_p.add_dim(seg_ch-1, 0);
	cpat_seg_p.bw(3, 0);
	cpat_seg_p.build();
	clk.bw(0, 0);
	vid.bw(seg_ch-1, 0);
	hid.bw(1, 0);
	cid.bw(2, 0);
	sid.bw(0, 0);
	ph_match.bw(bw_fph-1, 0);
	th_match.add_dim(zone_seg-1, 0);
	th_match.bw(bw_th-1, 0);
	th_match.build();
	cpat_match.bw(3, 0);
	ph_pat__storage.bw(bpow, 0);
	ph_pat.bw(bpow, 0);
	ph_pat.set_storage (&ph_pat__storage);
	ph_pat_v__storage.bw(0, 0);
	ph_pat_v.bw(0, 0);
	ph_pat_v.set_storage (&ph_pat_v__storage);
	ph_seg__storage.add_dim(max_drift-1, 0);
	ph_seg__storage.add_dim(zone_cham-1, 0);
	ph_seg__storage.add_dim(seg_ch-1, 0);
	ph_seg__storage.bw(bw_fph-1, 0);
	ph_seg__storage.build();
	ph_seg.add_dim(max_drift-1, 0);
	ph_seg.add_dim(zone_cham-1, 0);
	ph_seg.add_dim(seg_ch-1, 0);
	ph_seg.bw(bw_fph-1, 0);
	ph_seg.build();
	ph_seg.set_storage (&ph_seg__storage);
	ph_seg_v__storage.add_dim(max_drift-1, 0);
	ph_seg_v__storage.add_dim(zone_cham-1, 0);
	ph_seg_v__storage.bw(seg_ch-1, 0);
	ph_seg_v__storage.build();
	ph_seg_v.add_dim(max_drift-1, 0);
	ph_seg_v.add_dim(zone_cham-1, 0);
	ph_seg_v.bw(seg_ch-1, 0);
	ph_seg_v.build();
	ph_seg_v.set_storage (&ph_seg_v__storage);
	th_seg__storage.add_dim(max_drift-1, 0);
	th_seg__storage.add_dim(zone_cham-1, 0);
	th_seg__storage.add_dim(zone_seg-1, 0);
	th_seg__storage.bw(bw_th-1, 0);
	th_seg__storage.build();
	th_seg.add_dim(max_drift-1, 0);
	th_seg.add_dim(zone_cham-1, 0);
	th_seg.add_dim(zone_seg-1, 0);
	th_seg.bw(bw_th-1, 0);
	th_seg.build();
	th_seg.set_storage (&th_seg__storage);
	cpat_seg__storage.add_dim(max_drift-1, 0);
	cpat_seg__storage.add_dim(zone_cham-1, 0);
	cpat_seg__storage.add_dim(seg_ch-1, 0);
	cpat_seg__storage.bw(3, 0);
	cpat_seg__storage.build();
	cpat_seg.add_dim(max_drift-1, 0);
	cpat_seg.add_dim(zone_cham-1, 0);
	cpat_seg.add_dim(seg_ch-1, 0);
	cpat_seg.bw(3, 0);
	cpat_seg.build();
	cpat_seg.set_storage (&cpat_seg__storage);
	ph_segr__storage.bw(bpow, 0);
	ph_segr.bw(bpow, 0);
	ph_segr.set_storage (&ph_segr__storage);
	ph_diff_tmp__storage.bw(bpow, 0);
	ph_diff_tmp.bw(bpow, 0);
	ph_diff_tmp.set_storage (&ph_diff_tmp__storage);
	ph_diff__storage.add_dim(tot_diff-1, 0);
	ph_diff__storage.bw(bw_phdiff-1, 0);
	ph_diff__storage.build();
	ph_diff.add_dim(tot_diff-1, 0);
	ph_diff.bw(bw_phdiff-1, 0);
	ph_diff.build();
	ph_diff.set_storage (&ph_diff__storage);
	rcomp__storage.bw(1, 0);
	rcomp.bw(1, 0);
	rcomp.set_storage (&rcomp__storage);
	diffi0__storage.add_dim(tot_diff-1, 0);
	diffi0__storage.bw(5, 0);
	diffi0__storage.build();
	diffi0.add_dim(tot_diff-1, 0);
	diffi0.bw(5, 0);
	diffi0.build();
	diffi0.set_storage (&diffi0__storage);
	cmp1__storage.add_dim(tot_diff/3-1, 0);
	cmp1__storage.bw(bw_phdiff-1, 0);
	cmp1__storage.build();
	cmp1.add_dim(tot_diff/3-1, 0);
	cmp1.bw(bw_phdiff-1, 0);
	cmp1.build();
	cmp1.set_storage (&cmp1__storage);
	diffi1__storage.add_dim(tot_diff/3-1, 0);
	diffi1__storage.bw(5, 0);
	diffi1__storage.build();
	diffi1.add_dim(tot_diff/3-1, 0);
	diffi1.bw(5, 0);
	diffi1.build();
	diffi1.set_storage (&diffi1__storage);
	cmp2__storage.add_dim(tot_diff/9-1, 0);
	cmp2__storage.bw(bw_phdiff-1, 0);
	cmp2__storage.build();
	cmp2.add_dim(tot_diff/9-1, 0);
	cmp2.bw(bw_phdiff-1, 0);
	cmp2.build();
	cmp2.set_storage (&cmp2__storage);
	diffi2__storage.add_dim(tot_diff/9-1, 0);
	diffi2__storage.bw(5, 0);
	diffi2__storage.build();
	diffi2.add_dim(tot_diff/9-1, 0);
	diffi2.bw(5, 0);
	diffi2.build();
	diffi2.set_storage (&diffi2__storage);
	cmp3__storage.add_dim(tot_diff/18-1, 0);
	cmp3__storage.bw(bw_phdiff-1, 0);
	cmp3__storage.build();
	cmp3.add_dim(tot_diff/18-1, 0);
	cmp3.bw(bw_phdiff-1, 0);
	cmp3.build();
	cmp3.set_storage (&cmp3__storage);
	diffi3__storage.add_dim(tot_diff/18-1, 0);
	diffi3__storage.bw(5, 0);
	diffi3__storage.build();
	diffi3.add_dim(tot_diff/18-1, 0);
	diffi3.bw(5, 0);
	diffi3.build();
	diffi3.set_storage (&diffi3__storage);
	cmp4__storage.bw(bw_phdiff-1, 0);
	cmp4.bw(bw_phdiff-1, 0);
	cmp4.set_storage (&cmp4__storage);
	diffi4__storage.bw(5, 0);
	diffi4.bw(5, 0);
	diffi4.set_storage (&diffi4__storage);
	ri__storage.bw(1, 0);
	ri.bw(1, 0);
	ri.set_storage (&ri__storage);
	rj__storage.bw(2, 0);
	rj.bw(2, 0);
	rj.set_storage (&rj__storage);
	rk__storage.bw(0, 0);
	rk.bw(0, 0);
	rk.set_storage (&rk__storage);

}

// vppc: this function checks for changes in any signal on each simulation iteration
void find_segment::init ()
{
	if (!built)
	{
					}
	else
	{
		ph_pat__storage.init();
		ph_pat_v__storage.init();
		ph_seg__storage.init();
		ph_seg_v__storage.init();
		th_seg__storage.init();
		cpat_seg__storage.init();
		ph_segr__storage.init();
		ph_diff_tmp__storage.init();
		ph_diff__storage.init();
		rcomp__storage.init();
		diffi0__storage.init();
		cmp1__storage.init();
		diffi1__storage.init();
		cmp2__storage.init();
		diffi2__storage.init();
		cmp3__storage.init();
		diffi3__storage.init();
		cmp4__storage.init();
		diffi4__storage.init();
		ri__storage.init();
		rj__storage.init();
		rk__storage.init();
																																																			}
}
