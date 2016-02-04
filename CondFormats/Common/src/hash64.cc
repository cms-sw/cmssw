#include "CondFormats/Common/interface/hash64.h"

/*
--------------------------------------------------------------------
mix -- mix 3 64-bit values reversibly.
mix() takes 48 machine instructions, but only 24 cycles on a superscalar
  machine (like Intel's new MMX architecture).  It requires 4 64-bit
  registers for 4::2 parallelism.
All 1-bit deltas, all 2-bit deltas, all deltas composed of top bits of
  (a,b,c), and all deltas of bottom bits were tested.  All deltas were
  tested both on random keys and on keys that were nearly all zero.
  These deltas all cause every bit of c to change between 1/3 and 2/3
  of the time (well, only 113/400 to 287/400 of the time for some
  2-bit delta).  These deltas all cause at least 80 bits to change
  among (a,b,c) when the mix is run either forward or backward (yes it
  is reversible).
This implies that a hash using mix64 has no funnels.  There may be
  characteristics with 3-bit deltas or bigger, I didn't test for
  those.
--------------------------------------------------------------------
*/
#define mix64(a,b,c) \
{ \
  a -= b; a -= c; a ^= (c>>43); \
  b -= c; b -= a; b ^= (a<<9); \
  c -= a; c -= b; c ^= (b>>8); \
  a -= b; a -= c; a ^= (c>>38); \
  b -= c; b -= a; b ^= (a<<23); \
  c -= a; c -= b; c ^= (b>>5); \
  a -= b; a -= c; a ^= (c>>35); \
  b -= c; b -= a; b ^= (a<<49); \
  c -= a; c -= b; c ^= (b>>11); \
  a -= b; a -= c; a ^= (c>>12); \
  b -= c; b -= a; b ^= (a<<18); \
  c -= a; c -= b; c ^= (b>>22); \
}

typedef  unsigned long  long ub8;   /* unsigned 8-byte quantities */
typedef  unsigned long  int  ub4;   /* unsigned 4-byte quantities */
typedef  unsigned       char ub1;


namespace cond {

  ub8 hash64( ub1 * k, ub8 length, ub8 level)
  // register ub1 *k;        /* the key */
  // register ub8  length;   /* the length of the key */
  // register ub8  level;    /* the previous hash, or an arbitrary value */
  {
    register ub8 a,b,c,len;
    
    /* Set up the internal state */
    len = length;
    a = b = level;                         /* the previous hash value */
    c = 0x9e3779b97f4a7c13LL; /* the golden ratio; an arbitrary value */
    
    /*---------------------------------------- handle most of the key */
    if (((unsigned long)k)&7)
      {
	while (len >= 24)
	  {
	    a += (k[0]        +((ub8)k[ 1]<< 8)+((ub8)k[ 2]<<16)+((ub8)k[ 3]<<24)
		  +((ub8)k[4 ]<<32)+((ub8)k[ 5]<<40)+((ub8)k[ 6]<<48)+((ub8)k[ 7]<<56));
	    b += (k[8]        +((ub8)k[ 9]<< 8)+((ub8)k[10]<<16)+((ub8)k[11]<<24)
		  +((ub8)k[12]<<32)+((ub8)k[13]<<40)+((ub8)k[14]<<48)+((ub8)k[15]<<56));
	    c += (k[16]       +((ub8)k[17]<< 8)+((ub8)k[18]<<16)+((ub8)k[19]<<24)
		  +((ub8)k[20]<<32)+((ub8)k[21]<<40)+((ub8)k[22]<<48)+((ub8)k[23]<<56));
	    mix64(a,b,c);
	    k += 24; len -= 24;
	  }
      }
    else
      {
	while (len >= 24)    /* aligned */
	  {
	    a += *(ub8 *)(k+0);
	    b += *(ub8 *)(k+8);
	    c += *(ub8 *)(k+16);
	    mix64(a,b,c);
	    k += 24; len -= 24;
	  }
      }
    
    /*------------------------------------- handle the last 23 bytes */
    c += length;
    switch(len)              /* all the case statements fall through */
      {
      case 23: c+=((ub8)k[22]<<56);
      case 22: c+=((ub8)k[21]<<48);
      case 21: c+=((ub8)k[20]<<40);
      case 20: c+=((ub8)k[19]<<32);
      case 19: c+=((ub8)k[18]<<24);
      case 18: c+=((ub8)k[17]<<16);
      case 17: c+=((ub8)k[16]<<8);
	/* the first byte of c is reserved for the length */
      case 16: b+=((ub8)k[15]<<56);
      case 15: b+=((ub8)k[14]<<48);
      case 14: b+=((ub8)k[13]<<40);
      case 13: b+=((ub8)k[12]<<32);
      case 12: b+=((ub8)k[11]<<24);
      case 11: b+=((ub8)k[10]<<16);
      case 10: b+=((ub8)k[ 9]<<8);
      case  9: b+=((ub8)k[ 8]);
      case  8: a+=((ub8)k[ 7]<<56);
      case  7: a+=((ub8)k[ 6]<<48);
      case  6: a+=((ub8)k[ 5]<<40);
      case  5: a+=((ub8)k[ 4]<<32);
      case  4: a+=((ub8)k[ 3]<<24);
      case  3: a+=((ub8)k[ 2]<<16);
      case  2: a+=((ub8)k[ 1]<<8);
      case  1: a+=((ub8)k[ 0]);
	/* case 0: nothing left to add */
      }
    mix64(a,b,c);
    /*-------------------------------------------- report the result */
    return c;
  }
  
  
}
