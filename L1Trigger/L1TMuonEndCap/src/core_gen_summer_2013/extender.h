// This C++ header file was automatically generated
// by VPPC from a Verilog HDL project.
// VPPC web-page: http://www.phys.ufl.edu/~madorsky/vppc/

// Author    : madorsky
// Timestamp : Thu Mar 12 14:54:00 2015

#ifndef __extender_h_file__
#define __extender_h_file__
#include "vppc_sim_lib.h"

class extender
{
 public:
	extender(){built = false; glbl_gsr = true; defparam();}
	void defparam();
	void build();
	bool built;
	bool glbl_gsr;
		// io bit width 
	unsigned bit_w;
	
		signal_ inp;
		signal_ drifttime;
		signal_ clk;
		signal_ outp;
			// block memory
signal_storage mem__storage;  signal_ mem;
			// read address
signal_storage ra__storage;  signal_ ra;
			// write address
signal_storage wa__storage;  signal_ wa;
			// read data, should be absorbed by BRAM
signal_storage rd__storage;  signal_ rd;
	
	
		unsigned i;
	
	void init ();
	void operator()
	(
	signal_& inp__io,
	signal_& outp__io,
	signal_& drifttime__io,
	signal_& clk__io
	);
};
#endif
