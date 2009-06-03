#ifndef CondCommon_Hash64_h
#define CondCommon_Hash64_h 

namespace cond {

  /*
    --------------------------------------------------------------------
    lookup8.c, by Bob Jenkins, January 4 1997, Public Domain.
    hash(), hash2(), hash3, and mix() are externally useful functions.
    Routines to test the hash are included if SELF_TEST is defined.
    You can use this free for any purpose.  It has no warranty.
    --------------------------------------------------------------------
  */
  
  
  /*
    --------------------------------------------------------------------
    hash() -- hash a variable-length key into a 64-bit value
    k     : the key (the unaligned variable-length array of bytes)
    len   : the length of the key, counting by bytes
    level : can be any 8-byte value
    Returns a 64-bit value.  Every bit of the key affects every bit of
    the return value.  No funnels.  Every 1-bit and 2-bit delta achieves
    avalanche.  About 41+5len instructions.
    
    The best hash table sizes are powers of 2.  There is no need to do
    mod a prime (mod is sooo slow!).  If you need less than 64 bits,
    use a bitmask.  For example, if you need only 10 bits, do
    h = (h & hashmask(10));
    In which case, the hash table should have hashsize(10) elements.
    
    If you are hashing n strings (ub1 **)k, do it like this:
    for (i=0, h=0; i<n; ++i) h = hash( k[i], len[i], h);
    
    By Bob Jenkins, Jan 4 1997.  bob_jenkins@burtleburtle.net.  You may
    use this code any way you wish, private, educational, or commercial,
    but I would appreciate if you give me credit.
    
    See http://burtleburtle.net/bob/hash/evahash.html
    Use for hash table lookup, or anything where one collision in 2^^64
    is acceptable.  Do NOT use for cryptographic purposes.
    --------------------------------------------------------------------
  */
  /*
    --------------------------------------------------------------------
    This is identical to hash() on little-endian machines, and it is much
    faster than hash(), but a little slower than hash2(), and it requires
    -- that all your machines be little-endian, for example all Intel x86
    chips or all VAXen.  It gives wrong results on big-endian machines.
    --------------------------------------------------------------------
  */
  
  unsigned long  long hash64( unsigned char * k, unsigned long  long length, unsigned long  long level);
  
  
}
#endif
