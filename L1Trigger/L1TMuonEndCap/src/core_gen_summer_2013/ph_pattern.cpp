// This C++ source file was automatically generated
// by VPPC from a Verilog HDL project.
// VPPC web-page: http://www.phys.ufl.edu/~madorsky/vppc/

// Author    : madorsky
// Timestamp : Thu Mar 12 14:54:01 2015

#include "ph_pattern.h"

extern size_t __glob_alwaysn__;

void ph_pattern::operator()
(
	signal_& st1__io,
	signal_& st2__io,
	signal_& st3__io,
	signal_& st4__io,
	signal_& drifttime__io,
	signal_& foldn__io,
	signal_& qcode__io,
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
		// input raw hit bits
		st1.attach(st1__io);
		st2.attach(st2__io);
		st3.attach(st3__io);
		st4.attach(st4__io);
		drifttime.attach(drifttime__io);
		// number of current fold 
		foldn.attach(foldn__io);
		clk.attach(clk__io);
		// quality code output
		qcode.attach(qcode__io);
	}


	beginalways();
	
	
    if (posedge (clk)) 
    {


        for (mi = 0; mi < red_pat_w_st1; mi = mi + 1) 
        {
			if (( (mi) == 0)) {  {
					lyhits[2] = uor(st1(7,0));
					lyhits[1] = st2;
					lyhits[0] = (uor(st3(14,7))) | (uor(st4(14,7)));
					straightness = const_(3, 0x0UL);
				} } else 
			if (( (mi) == 1)) {  {
					lyhits[2] = uor(st1(30,23));
					lyhits[1] = st2;
					lyhits[0] = (uor(st3(7,0))) | (uor(st4(7,0)));
					straightness = const_(3, 0x0UL);
				} } else 
			if (( (mi) == 2)) {  {
					lyhits[2] = uor(st1(11,8));
					lyhits[1] = st2;
					lyhits[0] = (uor(st3(14,7))) | (uor(st4(14,7)));
					straightness = const_(3, 0x1UL);
				} } else 
			if (( (mi) == 3)) {  {
					lyhits[2] = uor(st1(22,19));
					lyhits[1] = st2;
					lyhits[0] = (uor(st3(7,0))) | (uor(st4(7,0)));
					straightness = const_(3, 0x1UL);
				} } else 
			if (( (mi) == 4)) {  {
					lyhits[2] = uor(st1(13,12));
					lyhits[1] = st2;
					lyhits[0] = (uor(st3(10,7))) | (uor(st4(10,7)));
					straightness = const_(3, 0x2UL);
				} } else 
			if (( (mi) == 5)) {  {
					lyhits[2] = uor(st1(18,17));
					lyhits[1] = st2;
					lyhits[0] = (uor(st3(7,5))) | (uor(st4(7,5)));
					straightness = const_(3, 0x2UL);
				} } else 
			if (( (mi) == 6)) {  {
					lyhits[2] = st1[14];
					lyhits[1] = st2;
					lyhits[0] = (uor(st3(8,7))) | (uor(st4(8,7)));
					straightness = const_(3, 0x3UL);
				} } else 
			if (( (mi) == 7)) {  {
					lyhits[2] = st1[16];
					lyhits[1] = st2;
					lyhits[0] = (uor(st3(7,6))) | (uor(st4(7,6)));
					straightness = const_(3, 0x3UL);
				} } else 
			if (( (mi) == 8)) {  {
					lyhits[2] = st1[15];
					lyhits[1] = st2;
					lyhits[0] = st3[7] | st4[7];
					straightness = const_(3, 0x4UL);
				} } 

			qcode_p[mi] = 0;
			
			if 
			(
			 bx[mi][foldn] == drifttime && // if drift time is up, find quality of this pattern
				// remove single-layer and ME3-4 hit patterns
			 lyhits != const_(3, 0x1UL) && 
			 lyhits != const_(3, 0x2UL) && 
			 lyhits != const_(3, 0x4UL) && 
			 lyhits != const_(3, 0x0UL)
			)
				// this quality code scheme is giving almost-equal priority to more stations and better straightness.
				// station 1 has higher weight, station 2 lower, st. 3 and 4 lowest
				qcode_p[mi] = (straightness[2], lyhits[2], straightness[1], lyhits[1], straightness[0], lyhits[0]);
			
			
            bx[mi][foldn] = (lyhits == const_(3, 0x0UL)) ? const_(3, 0UL) : 
							  (bx[mi][foldn] == const_(3, 7UL)) ? const_(3, 7UL) : 
							  bx[mi][foldn] + const_(3, 1UL); // bx starts counting at any hit in the pattern, even single
        }

        qcode = 0;
		// find max quality on each clock
		comp1[0] = qcode_p[0] > qcode_p[1] ? qcode_p[0] : qcode_p[1];
		comp1[1] = qcode_p[2] > qcode_p[3] ? qcode_p[2] : qcode_p[3];
		comp1[2] = qcode_p[4] > qcode_p[5] ? qcode_p[4] : qcode_p[5];
		comp1[3] = qcode_p[6] > qcode_p[7] ? qcode_p[6] : qcode_p[7];

		comp2[0] = comp1[0] > comp1[1] ? comp1[0] : comp1[1];
		comp2[1] = comp1[2] > comp1[3] ? comp1[2] : comp1[3];

		comp3 = comp2[0] > comp2[1] ? comp2[0] : comp2[1];

		qcode = comp3 > qcode_p[8] ? comp3 : qcode_p[8];

    }
	endalways();
}

// vppc: code below this point are service functions, should not require your attention

// vppc: this function assigns initial values to parameters and localparams
void ph_pattern::defparam()
{
	station = 1;
	cscid = 1;
}

// vppc: this function allocates memory for internal signals
void ph_pattern::build()
{
	built = true;
	st1.bw(full_pat_w_st1-1, 0);
	st2.bw(0, 0);
	st3.bw(full_pat_w_st3-1, 0);
	st4.bw(full_pat_w_st3-1, 0);
	drifttime.bw(2, 0);
	foldn.bw(2, 0);
	clk.bw(0, 0);
	qcode.bw(5, 0);
	bx__storage.add_dim(red_pat_w_st1-1, 0);
	bx__storage.add_dim(fold-1, 0);
	bx__storage.bw(2, 0);
	bx__storage.build();
	bx.add_dim(red_pat_w_st1-1, 0);
	bx.add_dim(fold-1, 0);
	bx.bw(2, 0);
	bx.build();
	bx.set_storage (&bx__storage);
	lyhits__storage.bw(2, 0);
	lyhits.bw(2, 0);
	lyhits.set_storage (&lyhits__storage);
	qcode_p__storage.add_dim(8, 0);
	qcode_p__storage.bw(5, 0);
	qcode_p__storage.build();
	qcode_p.add_dim(8, 0);
	qcode_p.bw(5, 0);
	qcode_p.build();
	qcode_p.set_storage (&qcode_p__storage);
	straightness__storage.bw(2, 0);
	straightness.bw(2, 0);
	straightness.set_storage (&straightness__storage);
	comp1__storage.add_dim(3, 0);
	comp1__storage.bw(5, 0);
	comp1__storage.build();
	comp1.add_dim(3, 0);
	comp1.bw(5, 0);
	comp1.build();
	comp1.set_storage (&comp1__storage);
	comp2__storage.add_dim(1, 0);
	comp2__storage.bw(5, 0);
	comp2__storage.build();
	comp2.add_dim(1, 0);
	comp2.bw(5, 0);
	comp2.build();
	comp2.set_storage (&comp2__storage);
	comp3__storage.bw(5, 0);
	comp3.bw(5, 0);
	comp3.set_storage (&comp3__storage);

}

// vppc: this function checks for changes in any signal on each simulation iteration
void ph_pattern::init ()
{
	if (!built)
	{
			}
	else
	{
		bx__storage.init();
		lyhits__storage.init();
		qcode_p__storage.init();
		straightness__storage.init();
		comp1__storage.init();
		comp2__storage.init();
		comp3__storage.init();
																																															}
}
