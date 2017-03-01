// This C++ source file was automatically generated
// by VPPC from a Verilog HDL project.
// VPPC web-page: http://www.phys.ufl.edu/~madorsky/vppc/

// Author    : madorsky
// Timestamp : Thu Mar 12 14:54:00 2015

#include "best_delta.h"

extern size_t __glob_alwaysn__;

void best_delta::operator()
(
	signal_& dth__io,
	signal_& sth__io,
	signal_& dvl__io,
	signal_& bth__io,
	signal_& bsg__io,
	signal_& bvl__io,
	signal_& bnm__io
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
		bw_num = (nseg == seg_ch * seg_ch) ? 2 : 3;
		nodiff = ((1 << (bw_th)) - 1);
		build();
		dth.attach(dth__io);
		sth.attach(sth__io);
		dvl.attach(dvl__io);
		bth.attach(bth__io);
		bsg.attach(bsg__io);
		bvl.attach(bvl__io);
		bnm.attach(bnm__io);
	}

	
	if (true)
	{
		// first comparator stage
		for (i = 0; i < nseg/2; i = i+1)
		{
			if (((dth[i*2] < dth[i*2+1]) && (dvl.bp(i*2 , 2) == const_(2, 0x3UL))) || (dvl.bp(i*2 , 2) == const_(2, 0x1UL)))
			{
				cmp1[i] = dth[i*2];
				sig1[i] = sth[i*2];
				num1[i] = i*2;
			}
			else
			{ 
				cmp1[i] = dth[i*2+1];
				sig1[i] = sth[i*2+1];
				num1[i] = i*2+1;
			}
			
			if (dvl.bp(i*2 , 2) == const_(2, 0x0UL)) cmp1[i] = nodiff; // if one of the inputs invalid, output = max
		}

		// second comparator stage
		for (i = 0; i < nseg/4; i = i+1)
		{
			if (cmp1[i*2] < cmp1[i*2+1])
			{
				cmp2[i] = cmp1[i*2];
				sig2[i] = sig1[i*2];
				num2[i] = num1[i*2];
			}
			else
			{ 
				cmp2[i] = cmp1[i*2+1];
				sig2[i] = sig1[i*2+1];
				num2[i] = num1[i*2+1];
			}
		}

		// third comparator stage if needed
		if (nseg/4 > 1)
		{
			if (cmp2[0] < cmp2[1])
			{
				bth = cmp2[0];
				bsg = sig2[0];
				bnm = num2[0];
			}
			else
			{ 
				bth = cmp2[1];
				bsg = sig2[1];
				bnm = num2[1];
			}
		}
		else
		{
			bth = cmp2[0];
			bsg = sig2[0];
			bnm = num2[0];
		}

		// output valid if one | more inputs are valid
		bvl = uor((dvl));
	}
}

// vppc: code below this point are service functions, should not require your attention

// vppc: this function assigns initial values to parameters and localparams
void best_delta::defparam()
{
	station = 1;
	cscid = 1;
	nseg = seg_ch * seg_ch;
}

// vppc: this function allocates memory for internal signals
void best_delta::build()
{
	built = true;
	dth.add_dim(nseg-1, 0);
	dth.bw(bw_th-1, 0);
	dth.build();
	sth.bw(nseg-1, 0);
	dvl.bw(nseg-1, 0);
	bth.bw(bw_th-1, 0);
	bsg.bw(0, 0);
	bvl.bw(0, 0);
	bnm.bw(bw_num-1, 0);
	one_val__storage.bw(0, 0);
	one_val.bw(0, 0);
	one_val.set_storage (&one_val__storage);
	cmp1__storage.add_dim(nseg/2-1, 0);
	cmp1__storage.bw(bw_th-1, 0);
	cmp1__storage.build();
	cmp1.add_dim(nseg/2-1, 0);
	cmp1.bw(bw_th-1, 0);
	cmp1.build();
	cmp1.set_storage (&cmp1__storage);
	cmp2__storage.add_dim(nseg/4-1, 0);
	cmp2__storage.bw(bw_th-1, 0);
	cmp2__storage.build();
	cmp2.add_dim(nseg/4-1, 0);
	cmp2.bw(bw_th-1, 0);
	cmp2.build();
	cmp2.set_storage (&cmp2__storage);
	sig1__storage.bw(nseg/2-1, 0);
	sig1.bw(nseg/2-1, 0);
	sig1.set_storage (&sig1__storage);
	sig2__storage.bw(nseg/4-1, 0);
	sig2.bw(nseg/4-1, 0);
	sig2.set_storage (&sig2__storage);
	num1__storage.add_dim(nseg/2-1, 0);
	num1__storage.bw(bw_num-1, 0);
	num1__storage.build();
	num1.add_dim(nseg/2-1, 0);
	num1.bw(bw_num-1, 0);
	num1.build();
	num1.set_storage (&num1__storage);
	num2__storage.add_dim(nseg/4-1, 0);
	num2__storage.bw(bw_num-1, 0);
	num2__storage.build();
	num2.add_dim(nseg/4-1, 0);
	num2.bw(bw_num-1, 0);
	num2.build();
	num2.set_storage (&num2__storage);

}

// vppc: this function checks for changes in any signal on each simulation iteration
void best_delta::init ()
{
	if (!built)
	{
				}
	else
	{
		one_val__storage.init();
		cmp1__storage.init();
		cmp2__storage.init();
		sig1__storage.init();
		sig2__storage.init();
		num1__storage.init();
		num2__storage.init();
																																																	}
}
