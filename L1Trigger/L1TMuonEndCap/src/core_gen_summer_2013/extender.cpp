// This C++ source file was automatically generated
// by VPPC from a Verilog HDL project.
// VPPC web-page: http://www.phys.ufl.edu/~madorsky/vppc/

// Author    : madorsky
// Timestamp : Thu Mar 12 14:54:00 2015

#include "extender.h"

extern size_t __glob_alwaysn__;

void extender::operator()
(
	signal_& inp__io,
	signal_& outp__io,
	signal_& drifttime__io,
	signal_& clk__io
)
{
	if (!built)
	{
		build();
		inp.attach(inp__io);
		drifttime.attach(drifttime__io);
		clk.attach(clk__io);
		outp.attach(outp__io);
	}
	if (glbl_gsr)
	{
		outp = 0;
		glbl_gsr = false;
	}


	beginalways();
	
	if (posedge (clk))
	{
	
		// build flops for each bit
		for (i = 0; i < bit_w; i = i+1)
		{
			if (inp[i]) outp[i] = 1;
			if (rd[i]) outp[i] = 0;
		}
		// read the outputs
		rd = mem[ra];
		
		// write all input bits into memory on each clock
		mem[wa] = inp;
		wa = (ra + drifttime + 1);
		ra = (ra + 1);
		
	}
	endalways();
}

// vppc: code below this point are service functions, should not require your attention

// vppc: this function assigns initial values to parameters and localparams
void extender::defparam()
{
	bit_w = 240;
}

// vppc: this function allocates memory for internal signals
void extender::build()
{
	built = true;
	inp.bw(bit_w-1, 0);
	drifttime.bw(2, 0);
	clk.bw(0, 0);
	outp.bw(bit_w-1, 0);
	mem__storage.add_dim(7, 0);
	mem__storage.bw(bit_w-1, 0);
	mem__storage.build();
	mem.add_dim(7, 0);
	mem.bw(bit_w-1, 0);
	mem.build();
	mem.set_storage (&mem__storage);
	ra__storage.bw(2, 0);
	ra.bw(2, 0);
	ra.set_storage (&ra__storage);
	wa__storage.bw(2, 0);
	wa.bw(2, 0);
	wa.set_storage (&wa__storage);
	rd__storage.bw(bit_w-1, 0);
	rd.bw(bit_w-1, 0);
	rd.set_storage (&rd__storage);
	ra = 0;

}

// vppc: this function checks for changes in any signal on each simulation iteration
void extender::init ()
{
	if (!built)
	{
		}
	else
	{
		mem__storage.init();
		ra__storage.init();
		wa__storage.init();
		rd__storage.init();
	}
}
