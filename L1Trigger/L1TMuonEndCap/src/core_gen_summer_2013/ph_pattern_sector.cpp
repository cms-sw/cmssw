// This C++ source file was automatically generated
// by VPPC from a Verilog HDL project.
// VPPC web-page: http://www.phys.ufl.edu/~madorsky/vppc/

// Author    : madorsky
// Timestamp : Thu Mar 12 14:54:01 2015

#include "ph_pattern_sector.h"

extern size_t __glob_alwaysn__;

void ph_pattern_sector::operator()
(
	signal_& st__io,
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
		// ph zones [zone][station]
		st.attach(st__io);
		drifttime.attach(drifttime__io);
		// number of current fold 
		foldn.attach(foldn__io);
		// quality code pattern numbers [zone][key_strip]
		clk.attach(clk__io);
		// quality codes output [zone][key_strip]
		qcode.attach(qcode__io);
	}

	
	{ 
		for (z = 0; z < 4; z = z+1)
		{
			// put station inputs into padded copy
			 stp [z][0](ph_raw_w + padding_w_st1-1 , padding_w_st1) = st[z][1];
			 stp [z][1](ph_raw_w + padding_w_st3-1 , padding_w_st3) = st[z][3];
			 stp [z][2](ph_raw_w + padding_w_st3-1 , padding_w_st3) = st[z][4];
		}

		for (z = 0; z < 4; z = z+1)
		{
			for (i = 0; i < ph_raw_w; i = i+1)
			{
					gb.ph_pat_zone[z].ph_pat_hit[i].php
	(
		stp [z][0](i+full_pat_w_st1-1 , i),
		st  [z][2][i],
		stp [z][1](i+full_pat_w_st3-1 , i),
		stp [z][2](i+full_pat_w_st3-1 , i),
		drifttime,
		foldn,
		qcode_p[z][i],
		clk
	);
			} // block: ph_pat_hit
		}
	}
	

	
	if (true)
	{
		// ghost cancellation logic
		for (zi = 0; zi < 4; zi = zi+1) // zone loop
		{
			for (ri = 0; ri < ph_raw_w; ri = ri+1) // pattern detector loop
			{
				qc = qcode_p[zi][ri]; // center quality is the current one

				if (ri == 0) // right } - special case
				{
					ql = qcode_p[zi][ri+1];
					qr = 0; // nothing to the right
				}
				else
				if (ri == ph_raw_w-1) // left } - special case
				{
					ql = 0; // nothing to the left
					qr = qcode_p[zi][ri-1];
				}
				else // all other patterns
				{
					ql = qcode_p[zi][ri+1];
					qr = qcode_p[zi][ri-1];
				}

				// cancellation conditions
				if (qc <=  ql || qc <  qr) // this pattern is lower quality than neighbors
				{
					qc = 0; // cancel
				}

				// put the results into outputs
				qcode[zi][ri] = qc;
			}
		}
	}
}

// vppc: code below this point are service functions, should not require your attention

// vppc: this function assigns initial values to parameters and localparams
void ph_pattern_sector::defparam()
{
	station = 1;
	cscid = 1;
}

// vppc: this function allocates memory for internal signals
void ph_pattern_sector::build()
{
	built = true;
	st.add_dim(3, 0);
	st.add_dim(4, 1);
	st.bw(ph_raw_w-1, 0);
	st.build();
	drifttime.bw(2, 0);
	foldn.bw(2, 0);
	clk.bw(0, 0);
	qcode.add_dim(3, 0);
	qcode.add_dim(ph_raw_w-1, 0);
	qcode.bw(5, 0);
	qcode.build();
	ql__storage.bw(5, 0);
	ql.bw(5, 0);
	ql.set_storage (&ql__storage);
	qr__storage.bw(5, 0);
	qr.bw(5, 0);
	qr.set_storage (&qr__storage);
	qc__storage.bw(5, 0);
	qc.bw(5, 0);
	qc.set_storage (&qc__storage);
	stp__storage.add_dim(3, 0);
	stp__storage.add_dim(2, 0);
	stp__storage.bw(ph_raw_w + padding_w_st1*2-1, 0);
	stp__storage.build();
	stp.add_dim(3, 0);
	stp.add_dim(2, 0);
	stp.bw(ph_raw_w + padding_w_st1*2-1, 0);
	stp.build();
	stp.set_storage (&stp__storage);
	qcode_p__storage.add_dim(3, 0);
	qcode_p__storage.add_dim(ph_raw_w-1, 0);
	qcode_p__storage.bw(5, 0);
	qcode_p__storage.build();
	qcode_p.add_dim(3, 0);
	qcode_p.add_dim(ph_raw_w-1, 0);
	qcode_p.bw(5, 0);
	qcode_p.build();
	qcode_p.set_storage (&qcode_p__storage);
		
	{
		for (z = 0; z < 4; z = z+1)
		{
			// fill padding with zeroes	
			 stp [z][0](padding_w_st1-1 , 0) = 0;
			 stp [z][0](ph_raw_w + padding_w_st1*2-1 , ph_raw_w + padding_w_st1) = 0;
			
			 stp [z][1](padding_w_st3-1 , 0) = 0;
			 stp [z][1](ph_raw_w + padding_w_st3*2-1 , ph_raw_w + padding_w_st3) = 0;
			
			 stp [z][2](padding_w_st3-1 , 0) = 0;
			 stp [z][2](ph_raw_w + padding_w_st3*2-1 , ph_raw_w + padding_w_st3) = 0;
		}
	}
	
}

// vppc: this function checks for changes in any signal on each simulation iteration
void ph_pattern_sector::init ()
{
	if (!built)
	{
			}
	else
	{
		ql__storage.init();
		qr__storage.init();
		qc__storage.init();
		stp__storage.init();
		qcode_p__storage.init();
																																															gb.init();
	}
}
void ph_pattern_sector::gb__class::init()
{
	for (map <ull, ph_pat_zone__class>::iterator mit = ph_pat_zone.begin(); mit != ph_pat_zone.end(); mit++)
		mit->second.init();
}
void ph_pattern_sector::gb__class::ph_pat_zone__class::init()
{
	for (map <ull, ph_pat_hit__class>::iterator mit = ph_pat_hit.begin(); mit != ph_pat_hit.end(); mit++)
		mit->second.init();
}
void ph_pattern_sector::gb__class::ph_pat_zone__class::ph_pat_hit__class::init()
{
	php.init();
}
