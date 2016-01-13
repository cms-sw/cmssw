#include "vppc_sim_lib.h"
#include "string.h"
#include "stdio.h"

bool   __glob_change__ = false; // global change flag
size_t __glob_alwaysn__ = 0; // current always block number
size_t gan = 1; // glob_always next value
signal_ stemp[max_temp_sig]; // ring buffer of temporary signals
vector<signal_ *> perms; // vector or permanent signals
size_t stemp_i = 0; // temp signal index
size_t perm_i = 0; // permanent signal index
signal_ stdout_sig; // signal for printing into stdout, should contain 0s in first two data words

void sim_lib_init()
{
	// reserve storage for temp signals
	for (int i = 0; i < max_temp_sig; i++)
	{
		stemp[i].r = stemp[i].rc = stemp[i].rt = new ull[max_bits/sull];
	}
	// initialize stdout_sig
	stdout_sig.r = stdout_sig.rc = stdout_sig.rt = new ull[3];
	stdout_sig.r[0] = stdout_sig.r[1] = 0;
}

// get next temp signal from ring buffer
signal_& get_stemp()
{
	signal_& res = stemp[stemp_i = (stemp_i + 1) & temp_i_mask];
	res.rc = res.rt; // set storage pointer to temp storage, because it may have been repointed
	res.st = NULL; // no permanent storage
	res.ca1 = res.ca2 = NULL; // reset concatenation pointers
	res.alwaysn = __glob_alwaysn__; // assigned in current AB  
	return res;
}

void signal_::attach(signal_& src)
{
	if (cell == NULL) // single signal
	{
		// copy all parameters but storage indexes
		st      = src.st;       
		r		= src.r; 
		rc		= src.rc;
		rt		= src.rt;
		ca1		= src.ca1; 
		ca2		= src.ca2; 
		alwaysn	= src.alwaysn;

		// io may have different bit indexes and width
		// calculate new indexes
		sl		= src.sl; 
		size_t nsh = dh - dl + sl; // new high index
		sh = (src.sh >= nsh) ? nsh : src.sh; // new index cannot go above storage size
	}
	else // memory
	{
		// attach cell by cell
		for (size_t i = 0; i <= dim_h - dim_l; i++)
			cell[i].attach(src.cell[i]);
	}
}

void signal_storage::bw(size_t ih, size_t il)
{

	if (cell == NULL)
	{
		btw = ih - il + 1; // bit width
		wn = (btw-1) / sull + 2; // number of words for storage, one extra word for more efficient calculations
		byten = wn * sizeof (ull); // number of bytes

		// reserve storage
		r    = new ull[wn];
		rc   = new ull[wn];
		memset(r, 0, byten);
		memset(rc, 0, byten);
		assigned = change = edge = false;
		alwaysn = 0;
	}
	else // memory
	{
		// process cell by cell
		for (size_t i = 0; i <= dim_h - dim_l; i++)
			cell[i].bw(ih, il);
	}
}

void signal_storage::init()
{
	if (cell == NULL)
	{
		// check for changes
		if (assigned)
		{
			assigned = false;
			if (wn == 2) // less or eq to 32 bits
			{
				if ((change = r[0] - rc[0]))
				{
					// assign current value to registered value
					r[0] = rc[0];
					// tell simulator that one more iteration is needed
					__glob_change__ = true;
				}
			}
			else
			{
				change = false;
				for (size_t i = 0; i < wn-1; i++) 
					if (r[i] != rc[i])
					{
						// assign current value to registered value
						memcpy(r, rc, byten);
						// tell simulator that one more iteration is needed
						change = __glob_change__ = true;
						break;
					}
			}
		}
		else change = false;
	}
	else // memory
	{
		// process cell by cell
		for (size_t i = 0; i <= dim_h - dim_l; i++)
			cell[i].init();
	}
}

void signal_::bw(size_t ih, size_t il)
{
	if (cell == NULL) // single signal
	{
		// store bit indexes 
		dl = il;
		dh = ih;
		sl = 0;
		sh = ih - il;
		// reset concatenation pointers
		ca1 = ca2 = NULL;
		alwaysn = 0;
	}
	else // memory
	{
		// process cell by cell
		for (size_t i = 0; i <= dim_h - dim_l; i++)
			cell[i].bw(ih, il);
	}
}

void signal_::set_storage(signal_storage* st)
{
	if (cell == NULL) // single signal
	{
		this->st = st;
		r = st->r;
		rc = st->rc;
	}
	else // memory
	{
		// process cell by cell
		for (size_t i = 0; i <= dim_h - dim_l; i++)
			cell[i].set_storage(&(st->operator[](i)));
	}
}


signal_& signal_::operator=(signal_& oth)
{
	if (cell == NULL) // single signal
	{
		if (ca1 == NULL)
		{
			// not concatenation
			if (dh == 0) // single bit assignment simplified for performance
			{
				if (((oth.getval()[oth.sl >> 5]) >> (oth.sl & 0x1f)) & 1) 
					rc[sl >> 5] |= (1UL << (sl & 0x1f));
				else     
					rc[sl >> 5] &= ~(1UL << (sl & 0x1f));

			}
			else
			{
				size_t dbcnt = sh - sl + 1; // dest bit count

				for (size_t i = 0; i < dbcnt; i += sull) // copy ull by ull
					set_ull(i, (i+sull > dbcnt) ? (dbcnt & 0x1f) : sull, oth.get_ull(i));

			}
		}
		else
		{
			// concatenation
			*ca2 = oth; // assign to right part as is, if oth is longer than ca2 it will be trimmed
			size_t ca2l = ca2->dh - ca2->dl + 1; // ca2 length
			size_t othl = oth.dh - oth.dl + 1; // oth length
			if (othl > ca2l) // if oth longer than right part of concat
			{
				// assign remaining bits of oth to left part of concat
				*ca1 = oth(oth.dh, oth.dl + ca2l);
			}
		}

		alwaysn = __glob_alwaysn__; // set AB number
		if (st)
		{ 
			st->alwaysn = __glob_alwaysn__;
			st->assigned = true;
		}
	}
	else // memory
	{
		// process cell by cell
		for (size_t i = 0; i <= dim_h - dim_l; i++)
			cell[i] = oth[i];
	}
	return *this;
}

signal_& signal_::operator=(ull n)
{
	if (dh == 0) // single bit assignment simplified for performance
	{
		if (n & 1) 
			rc[sl >> 5] |= (1UL << (sl & 0x1f));
		else     
			rc[sl >> 5] &= ~(1UL << (sl & 0x1f));

	}
	else
	{
		size_t dbcnt = sh - sl + 1; // dest bit count

		// word 0
		set_ull(0, (sull > dbcnt) ? dbcnt : sull, n);

		for (size_t i = sull; i < dbcnt; i += sull) // the rest = 0
			set_ull(i, (i+sull > dbcnt) ? (dbcnt & 0x1f) : sull, 0UL);
	}
	alwaysn = __glob_alwaysn__; // set AB number
	if (st)
	{ 
		st->alwaysn = __glob_alwaysn__;
		st->assigned = true;
	}
	return *this;
}


signal_& signal_::operator[](ull i) // signle bit or memory cell selection
{
	if (cell == NULL)
	{
		signal_& t = get_stemp();
		t.set_storage(st); // same storage as in this
		t.sl = t.sh = (i-dl) + sl; // bit number in storage
		t.dl = t.dh = 0; // one bit signal
		return t;
	}
	else // memory
	{
		return cell[i-dim_l];
	}
}

signal_& signal_::bp(ull sn, ull lng) // +: bit portion selection
{
	return this->operator()(sn+lng-1, sn);
}

signal_& signal_::bm(ull sn, ull lng) // -: bit portion selection
{
	return this->operator()(sn, sn-lng+1);
}
 
signal_& signal_::operator()(ull hn, ull ln) // bit portion selection
{
	signal_& t = get_stemp();
	t.set_storage(st); // same storage as in this
	t.sl = (ln-dl) + sl; // low bit number in storage
	t.sh = (hn-dl) + sl; // high bit number in storage
	t.dl = 0; // low bit always 0
	t.dh = t.sh - t.sl; // high bit
	return t;
}

ull* signal_::getval()
{
	// if assigned in clocked AB, and now we're not in it, take registered value, otherwise current
	if (st) alwaysn = st->alwaysn; // check if storage was updated
	return (alwaysn  == __glob_alwaysn__) ? rc : r; 
}

// get a ull word from storage, bnum is lower bit number
ull signal_::get_ull(size_t bnum)
{
	dull res;
	if (ca1 == NULL) // regular signal
	{
 
        size_t lowb = sl + bnum;
		ull have_bits = sh - lowb + 1; // how many bits above lowb

		// construct dull pointer that covers two adjacent storage words
		// shift bits into position
		res = ((*((dull*)(getval() + (lowb >> 5)))) >> (lowb & 0x1f));

		if (have_bits < sull) res &= ~(mdull << have_bits); // mask extra bits
 	}
	else
	{
		// concatenation
		ull r1;
		if (bnum <= ca2->dh)
		{
			// ull starts in ca2
			res = ca2->get_ull(bnum);

			if (ca2->dh < bnum + sull)
			{
				// continues in ca1
				r1 = ca1->get_ull(0);
				// stitch ull together
				res |= (r1 << (ca2->dh - bnum + 1));
			}
		}
		else
		{
			// ull starts in ca1, no need to peek into ca2
			res = ca1->get_ull(bnum - ca2->dh - 1);
		}
	}
	return (ull) res;
}

void signal_::set_ull(size_t bnum, size_t bcnt, ull val)
{
	// regular signal
	size_t lowb = sl + bnum;
	size_t bn = lowb & 0x1f; // lower bit number

	// construct mask
	dull mask = (mdull << bn) ^ (mdull << (bn+bcnt));

	// construct dull pointer that covers two adjacent storage words
	dull* dest = (dull*)(rc + (lowb >> 5));

	// apply
	*dest = ((*dest) & (~mask)) | ((((dull)val) << bn) & mask);
}

ull uor(signal_& arg)
{
	ull res = 0UL;
	for (size_t i = 0; i <= arg.sh - arg.sl; i += sull) // check ull by ull
		if (arg.get_ull(i) != 0UL)
		{
			res = 1UL; // get out as soon as non-zero bit is found.
			break;
		}
	return res;
}

ull uand(signal_& arg)
{
	ull res = 0UL;
	//  IMPORTANT: add code here
	return res;
}

ull uxor(signal_& arg)
{
	ull res = 0UL;
	//  IMPORTANT: add code here
	return res;
}

ull uor  (ull arg)
{
	return arg != 0;
}

ull uxor (ull arg)
{
	//  IMPORTANT: add code here
	return 0UL; 
}

ull uand (ull arg)
{
	return arg == mull;
}



signal_& const_l_p(size_t sz, size_t count, ... )
{
	signal_* t;
	signal_storage* s;
	try
	{
		// get permanent signal if it was made already
		t = perms.at(perm_i);
	}
	catch (std::out_of_range& e)
	{
		// no signal there, need to make one
		t = new signal_();
		s = new signal_storage();
		t->bw(sz-1, 0);
		s->bw(sz-1, 0);
		t->set_storage(s);
		perms.push_back(t);
	}
	perm_i++; // move index

	t->dl = t->sl = 0; 
	t->dh = t->sh = sz - 1;

    va_list ap;
    va_start(ap, count); //Requires the last fixed parameter (to get the address)

	for(size_t j = 0, i = count-1; j < count; j++, i--)
	{
		t->rc[count - j - 1] =  
		t->r [count - j - 1] = va_arg(ap, ull); 
	}

	t->rc[count] = 0;
	t->r [count] = 0;

	va_end(ap);
    return *t;
}


signal_& const_s_p(size_t sz, dull val)
{
	signal_* t;
	signal_storage* s;
	try
	{
		// get permanent signal if it was made already
		t = perms.at(perm_i);
	}
	catch (std::out_of_range& e)
	{
		// no signal there, need to make one
		t = new signal_();
		s = new signal_storage();
		t->bw(sz-1, 0);
		s->bw(sz-1, 0);
		t->set_storage(s);
		perms.push_back(t);
	}
	perm_i++; // move index

	t->dl = t->sl = 0; 
	t->dh = t->sh = sz - 1;

	t->rc[0] = (ull)val;
	t->rc[1] = 0;

	t->r[0] = (ull)val;
	t->r[1] = 0;
    return *t;
}

signal_& const_l(size_t sz, size_t count, ... )
{
	signal_& t = get_stemp();
	t.dl = t.sl = 0; 
	t.dh = t.sh = sz - 1;
//	memset (t.rc, 0, (sz>>3)+1);

    va_list ap;
    va_start(ap, count); //Requires the last fixed parameter (to get the address)

	for(size_t j = 0, i = count-1; j < count; j++, i--)
	{
		t.rc[count - j - 1] = va_arg(ap, ull); 
	}

	t.rc[count] = 0;

	va_end(ap);
    return t;
}


signal_& const_s(size_t sz, dull val)
{
	signal_& t = get_stemp();
	t.dl = t.sl = 0; 
	t.dh = t.sh = sz - 1;
//	memset (t.rc, 0, (sz>>3)+1);

	t.rc[0] = (ull)val;
	t.rc[1] = 0;

    return t;
}


ull const_(size_t sz,ull val)
{
//	return (ull)(val & (~(mdull << sz)));
	return val;
}


ull signal_::operator!() // logical negation
{
	ull res = 1UL;
	for (size_t i = 0; i <= sh - sl; i += sull) // check ull by ull
		if (get_ull(i) != 0UL)
		{
			res = 0UL; // get out as soon as non-zero bit is found.
			break;
		}
	return res;
}

#define arithmop(op,cop) \
ull signal_::op (signal_& arg) \
{ \
	signal_& t = get_stemp(); \
	size_t ln  = sh - sl; /* length of this */ \
	size_t aln = arg.sh - arg.sl;  /* length of argument */ \
	t.dl = t.sl = 0; \
	t.dh = t.sh = (ln > aln) ? ln : aln; /* take larger length as result's length */ \
	t.rc[0] = get_ull(0) cop arg.get_ull(0); /* works only up to ull bits so far */ \
	return t; \
}

arithmop (operator+,+)
arithmop (operator-,-)
arithmop (operator/,/)
arithmop (operator%,%)

signal_& signal_::operator, (signal_& arg)
{
	signal_& t = get_stemp();
	t.ca1 = this;
	t.ca2 = &arg; // store pointers to concatenated signals
	// calculate declared bit numbers
	t.dl = 0;
	t.dh = (dh-dl) + (arg.dh-arg.dl) + 1;
	return t;
}

#define compop(op,cop) \
ull signal_::op (signal_& arg) \
{ \
	return get_ull(0) cop arg.get_ull(0); /* works only for ull bits currently */ \
}

compop (operator>,>)
compop (operator<,<)
compop (operator<=,<=)
compop (operator>=,>=)

#define bitwiseop(op,cop) \
signal_& signal_::op (signal_& arg) \
{ \
	signal_& t = get_stemp(); \
	size_t ln  = sh - sl; /* length of this */ \
	size_t aln = arg.sh - arg.sl;  /* length of argument */ \
	t.dl = t.sl = 0; \
	t.dh = t.sh = (ln > aln) ? ln : aln; /* take larger length as result's length */ \
	for (size_t i = 0; i < t.dh+1; i+=sull) \
		t.rc[i/sull] = get_ull(i) cop arg.get_ull(i); \
	return t; \
}

bitwiseop(operator|,|)
bitwiseop(operator&,&)
bitwiseop(operator^,^)

signal_& signal_::operator~ ()
{
	signal_& t = get_stemp(); 
	t.dl = t.sl = 0;
	t.dh = t.sh = sh - sl;
	for (size_t i = 0; i < t.dh+1; i+=sull) 
		t.rc[i/sull] = ~get_ull(i);
	return t;
}


ull signal_::operator== (signal_& arg)
{
	ull res = 1;
	size_t ln  = sh - sl; /* length of this */ 
	size_t aln = arg.sh - arg.sl;  /* length of argument */ 
	size_t mln = (ln > aln) ? ln : aln; /* take larger length */ 
	for (size_t i = 0; i < mln+1; i+=sull) 
		res &= get_ull(i) == arg.get_ull(i); 
	return res; 
}

ull signal_::operator!= (signal_& arg)
{
	ull res = 0;
	size_t ln  = sh - sl; /* length of this */ 
	size_t aln = arg.sh - arg.sl;  /* length of argument */ 
	size_t mln = (ln > aln) ? ln : aln; /* take larger length */ 
	for (size_t i = 0; i < mln+1; i+=sull) 
		res |= get_ull(i) != arg.get_ull(i); 
	return res; 
}


bool posedge (signal_& s)
{
	return s.st->change & s.st->edge;
}

bool negedge (signal_& s)
{
	return s.st->change & !s.st->edge;
}

signal_::operator ull() // conversion operator
{
	return get_ull(0);
}

// does not work very well for long decimals or octals so far
void Sfwrite(signal_& fd, string format, ... )
{
	size_t bitw, symbw, radbits = 1;
	ull wull;
	signal_* t;
	FILE* ff;
	ostringstream ostr;

	// convert file descriptor into regular FILE*
	dull ifd;
	ifd = fd.rc[0];
	if (sizeof (ifd) > 4) ifd |= ((dull)fd.rc[1]) << 32;  
	ff = (FILE*)ifd;
	if (ff == NULL) ff = stdout;

	va_list ap;
    va_start(ap, format); //Requires the last fixed parameter (to get the address)

	for (size_t p = 0; p < format.length(); p++)
	{
		if (format[p] != '%') ostr << format[p]; // regular symbol
		else
		{
			p++;
			if (format[p] == '%') ostr << '%'; // double % means print %
			else
			{
				// format specifier
				t = (signal_*)va_arg(ap, void*);
				size_t wn = (t->dh - t->dl) / sull; // number of words minus 1 (index of the last word)
				switch (format[p])
				{
				case 'b': radbits = 1; break;
				case 't':
				case 'd': ostr << dec; radbits = 3; break;
				case 'o': ostr << oct; radbits = 3; break;
				case 'h': ostr << hex; radbits = 4; break;
				}
				for (int i = (int)wn; i >= 0; i--)
				{
					// calculate bit width of remaining portion
					if (i == (int)wn) bitw = (t->dh - t->dl + 1) % sull;
					else bitw = sull;
					if (bitw == 0) bitw = sull;

					symbw = bitw/radbits; // number of symbols
					if (bitw % radbits != 0) symbw++;
					if (symbw > 0)
					{
						if (i == (int)wn && radbits != 1 && radbits != 4) ostr << setfill(' ');
						else ostr << setfill('0');
						if (radbits > 1) ostr << setw(symbw);

						switch (format[p])
						{
						case 'b':
							wull = t->get_ull(i * sull);
							for (int j = bitw-1; j >= 0; j--)
								ostr << ((wull >> j) & 1UL);
							break;
						case 't':
						case 'd': 
						case 'o': 
						case 'h': 
							ostr << t->get_ull(i * sull);
							break;
						}
					}
				}
			}
		}
	}

	va_end(ap);
	fprintf (ff, ostr.str().c_str());
}

void Sreadmemh(string fname, signal_& dest, size_t adr)
{
	ull val;
	ifstream ifs(fname.c_str());
	if (ifs.is_open())
	{
		while (!ifs.eof())
		{
			if (ifs >> hex >> val) dest[adr] = val;
			else break; // stream op returns false if the value is not decoded
			adr++;
		}
		ifs.close();
	}
	else
	{
		cout << "ERROR: file " << fname.c_str() << " cannot be opened for reading\n";
	}
}

signal_ Sfopen(string fname, string mode)
{
	FILE* fp = fopen(fname.c_str(), mode.c_str());
	if (fp == NULL)
	{
		cout << "ERROR: file " << fname.c_str() << " cannot be opened\n";
	}
	signal_* t = new signal_();
	t->rc = t->rt = new ull[3];
	t->dl = t->sl = 0; 
	t->dh = t->sh = 63;
	t->st = NULL; // no permanent storage
	t->ca1 = t->ca2 = NULL; // reset concatenation pointers
	t->alwaysn = __glob_alwaysn__; // assigned in current AB  

	t->rc[0] = (dull)fp & mull;
	if (sizeof (fp) > 4)
		t->rc[1] = (((dull)fp) >> sull) & mull;
	else
		t->rc[1] = 0;

	return *t;
}

void Sfclose (signal_& fd)
{
	if (fd != 0)
	{
		dull ifd;
		ifd = fd.rc[0];
		if (sizeof (ifd) > 4) ifd |= ((dull)fd.rc[1]) << 32;  
		fclose((FILE*)ifd);
	}
}

int Sfgets (signal_& line, signal_& fd)
{
	if (fd != 0)
	{
		char lbuf[1000];
		dull ifd;
		ifd = fd.rc[0];
		if (sizeof (ifd) > 4) ifd |= ((dull)fd.rc[1]) << 32;  
		fgets(lbuf, sizeof (lbuf), (FILE*)ifd);
		strcpy((char*)line.rc, lbuf);
		return !feof((FILE*)ifd);
	}
	else
	{
		return 0;
	}
}

int Sfeof(signal_& fd)
{
	dull ifd;
	ifd = fd.rc[0];
	if (sizeof (ifd) > 4) ifd |= ((dull)fd.rc[1]) << 32;  
	return feof((FILE*)ifd);
}

// so far only decodes up to 32-bit values, no binary
int Ssscanf (signal_& line, string format, ... )
{
	signal_* t;
	ull val;
	int count = 0;
	va_list ap;
    va_start(ap, format); //Requires the last fixed parameter (to get the address)
	istringstream iss((char*)line.rc);

	for (size_t p = 0; p < format.length(); p++)
	{
		if (format[p] == '%') // specifier
		{
			p++;
			if (format[p] != '%') // double % means print %
			{
				// format specifier
				t = (signal_*)va_arg(ap, void*);
				if (!iss.eof())
				{
					val = mull;
					switch (format[p])
					{
					case 'd': iss >> dec >> val; break;
					case 'o': iss >> oct >> val; break;
					case 'h': iss >> hex >> val; break;
					}
					if (val != mull)
					{
						*t = val;
						count ++;
					}
					else break;
				}
			}
		}
	}

	va_end(ap);
	return count; // +1 to mimic Xilinx ISim sscanf bug 
}


void clk_drive(signal_& clk, ull v)
{
	clk = v;

	if (v > 0UL) clk.st->edge = true;
	else clk.st->edge = false;
	clk.st->change = true;
	clk.st->assigned = true;
}

void beginalways()
{
	__glob_alwaysn__ = gan++;
	if (gan == 0) gan++;
}

void endalways()
{
	__glob_alwaysn__ = 0;
}

void signal_::add_dim(size_t h, size_t l)
{
	if (cell == NULL)
	{ // these are my dimensions, reserve cells
		cell = new signal_[h-l+1];
		// store my dimensions
		dim_h = h;
		dim_l = l;
	}
	else
	{
		// sub-array's dimensions, tell them
		for (size_t i = 0; i <= dim_h - dim_l; i++)
			cell[i].add_dim(h,l);
	}
}

void signal_::build()
{
}

void signal_storage::add_dim(size_t h, size_t l)
{
	if (cell == NULL)
	{ // these are my dimensions, reserve cells
		cell = new signal_storage[h-l+1];
		// store my dimensions
		dim_h = h;
		dim_l = l;
	}
	else
	{
		// sub-array's dimensions, tell them
		for (size_t i = 0; i <= dim_h - dim_l; i++)
			cell[i].add_dim(h,l);
	}
}

void signal_storage::build()
{
}

signal_storage& signal_storage::operator[](ull i) 
{
	return cell[i]; // not subtracting dim_l here because i is already 0-starting
}
