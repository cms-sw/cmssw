#ifndef _VPPC_SIM_LIB_H_FILE_
#define _VPPC_SIM_LIB_H_FILE_
#include <iostream>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <cstdarg>
#include <map>
#include <vector>
#include <exception>
#include <stdexcept> // out_of_range exception
#include <time.h>

using namespace std;

#define ull unsigned int 
#define dull unsigned long long
#define sull 32 // size of unsigned __int32 
#define sdull 64 // size of double variable 
#define mull (0xffffffffUL) // max value
#define mdull (0xffffffffffffffffULL)
#define max_depth 32
#define max_temp_sig 64 // has to be power of 2
#define temp_i_mask (max_temp_sig - 1) // temp signal index mask
#define max_bits 1024
#define const__p const_s_p

#define Stime clock()
#define repeat(a) for (int __rep_counter__ = 0; __rep_counter__ < a; __rep_counter__++)
#define Swrite(cl, ...) Sfwrite(stdout_sig, cl, ##__VA_ARGS__)
#define Sdisplay(cl, ...) Sfwrite(stdout_sig, (string)cl + "\n", ##__VA_ARGS__)
#define Sfdisplay(fd,cl, ...) Sfwrite(fd, (string)cl + "\n", ##__VA_ARGS__)

void sim_lib_init();

class signal_storage
{
public:
	signal_storage(){cell = NULL; dim_h = dim_l = 0;};
	ull *r, *rc; // storage for current and registered values, needs to be reserved
	// initialization before each sim iteration
	void init();
	void bw(size_t ih, size_t il);
	void add_dim(size_t h, size_t l);
	void build();
	signal_storage& operator[](ull i);

	bool change; // change in value flag
	bool assigned; // flag showing that it was assigned
	bool edge; // posedge/negedge flag, set only for clock by test fixture. change & edge = posedge, change & !edge = negedge
	size_t btw, wn, byten; // bit width, word count, byte count
	size_t alwaysn; 
	signal_storage* cell; // array of signals
	size_t dim_h, dim_l; // array dimensions
};

class signal_
{
public:
	signal_(){st = NULL; cell = NULL; dim_h = dim_l = 0;};
	// signal_ bitwidth declaration
	void bw(size_t h, size_t l);
	void set_storage(signal_storage* st);
	ull get_ull(size_t bnum);
	void set_ull(size_t bnum, size_t bcnt, ull val);
	ull* getval();
	void attach(signal_& src);
	void add_dim(size_t h, size_t l);
	void build();

	signal_& operator[](ull i); // single bit selection
	signal_& operator()(ull hn, ull ln); // bit portion selection
	signal_& bp(ull sn, ull lng); // +: bit portion selection
	signal_& bm(ull sn, ull lng); // -: bit portion selection
	signal_& operator=(signal_& oth); // assignment
	signal_& operator=(ull n); // assignment
	ull operator!(); 

	ull operator+ (signal_& arg);
	ull operator- (signal_& arg);
	ull operator++  (int){return *this = *this + 1;};
	ull operator--  (int){return *this = *this - 1;};

	ull operator/ (signal_& arg);
	ull operator% (signal_& arg);
	ull operator<  (signal_& arg);
	ull operator>  (signal_& arg);
	ull operator<= (signal_& arg);
	ull operator>= (signal_& arg);

	signal_& operator+= (signal_& arg){return *this = *this + arg;};
	signal_& operator-= (signal_& arg){return *this = *this - arg;};
	signal_& operator/= (signal_& arg){return *this = *this / arg;};
	signal_& operator%= (signal_& arg){return *this = *this % arg;};

	ull operator== (signal_& arg);
	ull operator!= (signal_& arg);

	signal_& operator| (signal_& arg);
	signal_& operator& (signal_& arg);
	signal_& operator^ (signal_& arg);

	signal_& operator|= (signal_& arg){return *this = *this | arg;};
	signal_& operator&= (signal_& arg){return *this = *this & arg;};
	signal_& operator^= (signal_& arg){return *this = *this ^ arg;};

	signal_& operator~ ();

	signal_& operator, (signal_& arg);

	operator ull(); // conversion operator

	friend ull uor  (signal_&);
	friend ull uxor (signal_&);
	friend ull uand (signal_&);
	friend ull uor  (ull);
	friend ull uxor (ull);
	friend ull uand (ull);
	friend ull const_(size_t sz, ull val); // for constants fitting into one ull
	friend signal_& const_s(size_t sz, dull val); // for constants fitting into one ull
	friend signal_& const_l(size_t sz, size_t count, ... ); // for longer constants
	friend signal_& const_s_p(size_t sz, dull val); // for permanent constants fitting into one ull
	friend signal_& const_l_p(size_t sz, size_t count, ... ); // for longer permanent constants
	friend bool posedge (signal_&);
	friend bool negedge (signal_&);
	friend void clk_drive(signal_& clk, ull v);

	signal_storage* st; // pointer to signal value storage
	ull *r, *rc; // copies of pointers from st. For temp signals, rc == rt;
	ull *rt; // storage for temp signal only
	size_t dh, dl; // high and low bit numbers as declared
	size_t sh, sl; // high and low bits that should be applied to storage
	// number of always block where it was assigned
	// this number is individual for each reference because portions of the original 
	// storage may be assigned in different ABs
	signal_ *ca1, *ca2; // concatenated signals
	size_t alwaysn; 
	signal_* cell; // array of signals if this is lowest level
	size_t dim_h, dim_l; // array dimensions
};


ull const_(size_t sz, ull val); // for constants fitting into one ull
signal_& const_s(size_t sz, dull val); // for constants fitting into one ull
signal_& const_l(size_t sz, size_t count, ... ); // for longer constants
signal_& const_s_p(size_t sz, dull val); // for permanent constants fitting into one ull
signal_& const_l_p(size_t sz, size_t count, ... ); // for longer permanent constants
signal_& const_l(size_t sz, size_t count, ... ); // for longer constants
void Sfwrite(signal_& fd, string format, ... );
int Ssscanf (signal_& line, string format, ... );
void Sreadmemh(string fname, signal_& dest, size_t adr = 0);
signal_ Sfopen(string fname, string mode = "w");
void Sfclose (signal_& fd);
int Sfgets (signal_& line, signal_& fd);
int Sfeof(signal_& fd);
void beginalways();
void endalways();


#endif


