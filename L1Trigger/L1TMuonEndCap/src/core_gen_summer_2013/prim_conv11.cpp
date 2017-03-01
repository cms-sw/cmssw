// This C++ source file was automatically generated
// by VPPC from a Verilog HDL project.
// VPPC web-page: http://www.phys.ufl.edu/~madorsky/vppc/

// Author    : madorsky
// Timestamp : Thu Mar 12 14:54:01 2015

#include "prim_conv11.h"

extern size_t __glob_alwaysn__;

void prim_conv11::operator()
(
	signal_& quality__io,
	signal_& wiregroup__io,
	signal_& hstrip__io,
	signal_& clctpat__io,
	signal_& ph__io,
	signal_& th__io,
	signal_& vl__io,
	signal_& phzvl__io,
	signal_& me11a__io,
	signal_& clctpat_r__io,
	signal_& ph_hit__io,
	signal_& th_hit__io,
	signal_& sel__io,
	signal_& addr__io,
	signal_& r_in__io,
	signal_& r_out__io,
	signal_& we__io,
	signal_& clk__io,
	signal_& control_clk__io
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
		// input parameters from MPC
		quality.attach(quality__io);
		wiregroup.attach(wiregroup__io);
		hstrip.attach(hstrip__io);
		clctpat.attach(clctpat__io);
		sel.attach(sel__io);
		addr.attach(addr__io);
		r_in.attach(r_in__io);
		we.attach(we__io);
		clk.attach(clk__io);
		control_clk.attach(control_clk__io);
		// outputs
// low-precision ph, only for detection
// high-precision ph with displacement correction will be calculated when 
// 3 best tracks are found.
		ph.attach(ph__io);
		// full precision th, but without displacement correction, takes th duplication into account
		th.attach(th__io);
		// one-bit valid flags
		vl.attach(vl__io);
		phzvl.attach(phzvl__io);
		me11a.attach(me11a__io);
		clctpat_r.attach(clctpat_r__io);
		// ph and th raw hits
		ph_hit.attach(ph_hit__io);
		th_hit.attach(th_hit__io);
		r_out.attach(r_out__io);
	}

	
	 pc_id(3,0) = cscid;
	 pc_id(7,4) = station;
	
	 r_out = (sel == const_(2, 0x0UL)) ? params[addr] : 
				   (sel == const_(2, 0x1UL)) ? th_mem[addr] : 
				   (sel == const_(2, 0x2UL)) ? th_corr_mem[addr] : pc_id;

	beginalways();	

	if (posedge (control_clk))
	{
		if (( (sel) == 0)) {  { if (we) params     [addr] = r_in; } } else 
		if (( (sel) == 1)) {  { if (we) th_mem     [addr] = r_in; } } else 
		if (( (sel) == 2)) {  { if (we) th_corr_mem[addr] = r_in(3,0); } } 
	}
	endalways();

	beginalways();
	
	if (posedge (clk))
	{

		// zero outputs
		vl = 0;
		phzvl = 0;
		for (i = 0; i < seg_ch; i = i+1) { fph[i] = 0; clctpat_r[i] = 0; }
		for (i = 0; i < th_ch;  i = i+1) th[i] = 0;
		ph_hit = 0;
		th_hit = 0;

		for (i = 0; i < seg_ch; i = i+1)
		{

			factor[i] = (station <= 1 && cscid <= 2 && hstrip[i] > 127) ? 1707 : // ME1/1a
				 1301; // ME1/1b

			me11a_w[i] = (station <= 1 && cscid <= 2 && hstrip[i] > 127);
			if (( (clctpat[i]) == 0)) {  { clct_pat_corr = const_(3, 0x0UL); clct_pat_sign = 0; } } else 
			if (( (clctpat[i]) == 1)) {  { clct_pat_corr = const_(3, 0x0UL); clct_pat_sign = 0; } } else 
			if (( (clctpat[i]) == 2)) {  { clct_pat_corr = const_(3, 0x5UL); clct_pat_sign = 1; } } else 
			if (( (clctpat[i]) == 3)) {  { clct_pat_corr = const_(3, 0x5UL); clct_pat_sign = 0; } } else 
			if (( (clctpat[i]) == 4)) {  { clct_pat_corr = const_(3, 0x5UL); clct_pat_sign = 1; } } else 
			if (( (clctpat[i]) == 5)) {  { clct_pat_corr = const_(3, 0x5UL); clct_pat_sign = 0; } } else 
			if (( (clctpat[i]) == 6)) {  { clct_pat_corr = const_(3, 0x2UL); clct_pat_sign = 1; } } else 
			if (( (clctpat[i]) == 7)) {  { clct_pat_corr = const_(3, 0x2UL); clct_pat_sign = 0; } } else 
			if (( (clctpat[i]) == 8)) {  { clct_pat_corr = const_(3, 0x2UL); clct_pat_sign = 1; } } else 
			if (( (clctpat[i]) == 9)) {  { clct_pat_corr = const_(3, 0x2UL); clct_pat_sign = 0; } } else 
			if (( (clctpat[i]) == 10)) {  { clct_pat_corr = const_(3, 0x0UL); clct_pat_sign = 0; } } else  {  { clct_pat_corr = const_(3, 0x0UL); clct_pat_sign = 0; } } 

			// reverse clct pattern correction if chamber is reversed
//			if (ph_reverse) clct_pat_sign = ~clct_pat_sign;

			// convert into 1/8 strips and remove ME1/1a offset (512=128*4)
			eight_str[i]  = (const_s(2, 0x0UL), hstrip [i], const_s(2, 0x0UL)) - (me11a_w[i] ? 512 : 0);
			
			// clct pattern correction
			if (clct_pat_sign == 0) eight_str[i] = eight_str[i] + clct_pat_corr(2,1);
			else eight_str[i] = eight_str[i] - clct_pat_corr(2,1);

			if (quality[i])
			{
				vl[i] = 1;
				// ph conversion
				// for factors 1024 and 2048 the multiplier should be replaced with shifts by synthesizer
				mult = eight_str[i] * factor[i];
				ph_tmp = mult(mult_bw-1 , 10);
				ph_init_ix =  me11a_w[i] ? const_(3, 2UL) : const_(3, 0UL); // index of ph_init parameter to apply (different for ME11a and b)
				
				if (ph_reverse)
				{
					fph[i] = params[ph_init_ix] - ph_tmp;
					// set ph raw hits
					ph_hit[ph_coverage - ph_tmp(bw_fph-1,5) + params[ph_init_ix + const_(3, 1UL)](7,1)] = 1;
				}
				else
				{            
					fph[i] = params[ph_init_ix] + ph_tmp;
					// set ph raw hits
					ph_hit[ph_tmp(bw_fph-1,5) + params[ph_init_ix + const_(3, 1UL)](7,1)] = 1;
/* -----\/----- EXCLUDED -----\/-----
					// add hits to take ME11a strip ganging into account
					// offsets of 14 and 28 is what I observe from MC. Calculations show 11.6 and 23.2 (???)
					if (me11a_w[i])
					{
						ph_hit[ph_tmp(bw_fph-1,5) + params[2] + 14] = 1;
						ph_hit[ph_tmp(bw_fph-1,5) + params[2] + 28] = 1;
					}
 -----/\----- EXCLUDED -----/\----- */
				}
				
				wg = wiregroup[i];
				// th conversion
				// call appropriate LUT, it returns th[i] relative to wg0 of that chamber
				th_orig = th_mem[wg];


				// need th duplication here
				for (j = 0; j < seg_ch; j = j+1)
				{
					if (quality[j])
					{
						// calculate correction for each strip number
						// index is: (wiregroup(2 MS bits), dblstrip(5-bit for these chambers))
						index = (wg(5,4), eight_str[j](8,4));
						th_corr = th_corr_mem[index];

						// apply correction to the corresponding output
						if (ph_reverse) th_tmp = (th_orig - th_corr) & const_(6, 0x3fUL);
						else            th_tmp = (th_orig + th_corr) & const_(6, 0x3fUL);

						// check that correction did not make invalid value outside chamber coverage
						// this will actually take care of both positive and negative illegal values
						if (th_tmp < th_coverage) 
						{
							// apply initial th value for that chamber
							th[i*seg_ch+j] = th_tmp + params[4];
							// th hits
							th_hit[th_tmp + params[5]] = 1;
							
							// check which zones ph hits should be applied to
							if (th[i*seg_ch+j] <= (ph_zone_bnd1 + zone_overlap)) phzvl[0] = 1;
							if (th[i*seg_ch+j] >  (ph_zone_bnd2 - zone_overlap)) phzvl[2] = 1;
							if (
								(th[i*seg_ch+j] >  (ph_zone_bnd1 - zone_overlap)) &&
								(th[i*seg_ch+j] <= (ph_zone_bnd2 + zone_overlap))
								) phzvl[1] = 1;
							
						}
					}
				}
				clctpat_r[i] = clctpat[i]; // just propagate pattern downstream
			} // if (quality[i])
			
			ph[i] = fph[i];
			
		} // for (i = 0; i < seg_ch; i = i+1)
		me11a = me11a_w;
	}
	endalways();
}

// vppc: code below this point are service functions, should not require your attention

// vppc: this function assigns initial values to parameters and localparams
void prim_conv11::defparam()
{
	station = 1;
	cscid = 1;
}

// vppc: this function allocates memory for internal signals
void prim_conv11::build()
{
	built = true;
	quality.add_dim(seg_ch-1, 0);
	quality.bw(3, 0);
	quality.build();
	wiregroup.add_dim(seg_ch-1, 0);
	wiregroup.bw(bw_wg-1, 0);
	wiregroup.build();
	hstrip.add_dim(seg_ch-1, 0);
	hstrip.bw(bw_hs-1, 0);
	hstrip.build();
	clctpat.add_dim(seg_ch-1, 0);
	clctpat.bw(3, 0);
	clctpat.build();
	sel.bw(1, 0);
	addr.bw(bw_addr-1, 0);
	r_in.bw(11, 0);
	we.bw(0, 0);
	clk.bw(0, 0);
	control_clk.bw(0, 0);
	ph.add_dim(seg_ch-1, 0);
	ph.bw(bw_fph-1, 0);
	ph.build();
	th.add_dim(th_ch-1, 0);
	th.bw(bw_th-1, 0);
	th.build();
	vl.bw(seg_ch-1, 0);
	phzvl.bw(2, 0);
	me11a.bw(seg_ch-1, 0);
	clctpat_r.add_dim(seg_ch-1, 0);
	clctpat_r.bw(3, 0);
	clctpat_r.build();
	ph_hit.bw(ph_hit_w-1, 0);
	th_hit.bw(th_hit_w-1, 0);
	r_out.bw(11, 0);
	eight_str__storage.add_dim(seg_ch-1, 0);
	eight_str__storage.bw(bw_fph-1, 0);
	eight_str__storage.build();
	eight_str.add_dim(seg_ch-1, 0);
	eight_str.bw(bw_fph-1, 0);
	eight_str.build();
	eight_str.set_storage (&eight_str__storage);
	mult__storage.bw(mult_bw-1, 0);
	mult.bw(mult_bw-1, 0);
	mult.set_storage (&mult__storage);
	ph_tmp__storage.bw(bw_fph-1, 0);
	ph_tmp.bw(bw_fph-1, 0);
	ph_tmp.set_storage (&ph_tmp__storage);
	wg__storage.bw(bw_wg-1, 0);
	wg.bw(bw_wg-1, 0);
	wg.set_storage (&wg__storage);
	th_tmp__storage.bw(bw_th-1, 0);
	th_tmp.bw(bw_th-1, 0);
	th_tmp.set_storage (&th_tmp__storage);
	th_mem__storage.add_dim(th_mem_sz-1, 0);
	th_mem__storage.bw(5, 0);
	th_mem__storage.build();
	th_mem.add_dim(th_mem_sz-1, 0);
	th_mem.bw(5, 0);
	th_mem.build();
	th_mem.set_storage (&th_mem__storage);
	params__storage.add_dim(5, 0);
	params__storage.bw(11, 0);
	params__storage.build();
	params.add_dim(5, 0);
	params.bw(11, 0);
	params.build();
	params.set_storage (&params__storage);
	fph__storage.add_dim(seg_ch-1, 0);
	fph__storage.bw(bw_fph-1, 0);
	fph__storage.build();
	fph.add_dim(seg_ch-1, 0);
	fph.bw(bw_fph-1, 0);
	fph.build();
	fph.set_storage (&fph__storage);
	th_corr__storage.bw(3, 0);
	th_corr.bw(3, 0);
	th_corr.set_storage (&th_corr__storage);
	index__storage.bw(bw_ds+1, 0);
	index.bw(bw_ds+1, 0);
	index.set_storage (&index__storage);
	th_orig__storage.bw(5, 0);
	th_orig.bw(5, 0);
	th_orig.set_storage (&th_orig__storage);
	th_corr_mem__storage.add_dim(th_corr_mem_sz-1, 0);
	th_corr_mem__storage.bw(3, 0);
	th_corr_mem__storage.build();
	th_corr_mem.add_dim(th_corr_mem_sz-1, 0);
	th_corr_mem.bw(3, 0);
	th_corr_mem.build();
	th_corr_mem.set_storage (&th_corr_mem__storage);
	factor__storage.add_dim(seg_ch-1, 0);
	factor__storage.bw(10, 0);
	factor__storage.build();
	factor.add_dim(seg_ch-1, 0);
	factor.bw(10, 0);
	factor.build();
	factor.set_storage (&factor__storage);
	me11a_w__storage.bw(seg_ch-1, 0);
	me11a_w.bw(seg_ch-1, 0);
	me11a_w.set_storage (&me11a_w__storage);
	clct_pat_corr__storage.bw(2, 0);
	clct_pat_corr.bw(2, 0);
	clct_pat_corr.set_storage (&clct_pat_corr__storage);
	clct_pat_sign__storage.bw(0, 0);
	clct_pat_sign.bw(0, 0);
	clct_pat_sign.set_storage (&clct_pat_sign__storage);
	ph_init_ix__storage.bw(2, 0);
	ph_init_ix.bw(2, 0);
	ph_init_ix.set_storage (&ph_init_ix__storage);
	pc_id__storage.bw(7, 0);
	pc_id.bw(7, 0);
	pc_id.set_storage (&pc_id__storage);

}

// vppc: this function checks for changes in any signal on each simulation iteration
void prim_conv11::init ()
{
	if (!built)
	{
			}
	else
	{
		eight_str__storage.init();
		mult__storage.init();
		ph_tmp__storage.init();
		wg__storage.init();
		th_tmp__storage.init();
		th_mem__storage.init();
		params__storage.init();
		fph__storage.init();
		th_corr__storage.init();
		index__storage.init();
		th_orig__storage.init();
		th_corr_mem__storage.init();
		factor__storage.init();
		me11a_w__storage.init();
		clct_pat_corr__storage.init();
		clct_pat_sign__storage.init();
		ph_init_ix__storage.init();
		pc_id__storage.init();
																																															}
}
