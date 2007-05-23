// $Id: SprRanluxEngine.cc,v 1.2 2006/10/19 21:27:52 narsky Exp $
// -*- C++ -*-
//
// -----------------------------------------------------------------------
//                             HEP Random
//                        --- RanluxEngine ---
//                      class implementation file
// -----------------------------------------------------------------------
// This file is part of Geant4 (simulation toolkit for HEP).
//
// Ranlux random number generator originally implemented in FORTRAN77
// by Fred James as part of the MATHLIB HEP library.
// 'RanluxEngine' is designed to fit into the CLHEP random number
// class structure.

// ===============================================================
// Adeyemi Adesanya - Created: 6th November 1995
// Gabriele Cosmo - Adapted & Revised: 22nd November 1995
// Adeyemi Adesanya - Added setSeeds() method: 2nd February 1996
// Gabriele Cosmo - Added flatArray() method: 8th February 1996
//                - Minor corrections: 31st October 1996
//                - Added methods for engine status: 19th November 1996
//                - Fixed bug in setSeeds(): 15th September 1997
// J.Marraffino   - Added stream operators and related constructor.
//                  Added automatic seed selection from seed table and
//                  engine counter: 14th Feb 1998
//                - Fixed bug: setSeeds() requires a zero terminated
//                  array of seeds: 19th Feb 1998
// Ken Smith      - Added conversion operators:  6th Aug 1998
// J. Marraffino  - Remove dependence on hepString class  13 May 1999
// M. Fischler    - In restore, checkFile for file not found    03 Dec 2004
// M. Fischler    - Methods put, getfor instance save/restore       12/8/04    
// M. Fischler    - split get() into tag validation and 
//                  getState() for anonymous restores           12/27/04    
// M. Fischler    - put/get for vectors of ulongs		3/14/05
// M. Fischler    - State-saving using only ints, for portability 4/12/05
//
// ===============================================================

#include "SprRanluxEngine.hh"
#include "SprSeedTable.hh"
#include <cmath>	// for pow()
#include <cstdlib>	// for abs() and div()
//#include <iostream>

using namespace std;


const int SprRanluxEngine::int_modulus = 0x1000000;
const double SprRanluxEngine::mantissa_bit_24 = pow(0.5,24.);
const double SprRanluxEngine::mantissa_bit_12 = pow(0.5,12.);

// Number of instances with automatic seed selection
int SprRanluxEngine::numEngines = 0;

// Maximum index into the seed table
const int SprRanluxEngine::maxIndex = 215;

SprRanluxEngine::SprRanluxEngine(long seed, int lux) {
   luxury = lux;
   setSeed(seed, luxury);
}

SprRanluxEngine::~SprRanluxEngine() {}

void SprRanluxEngine::setSeed(long seed, int lux) {

  // default seed from seed table
  if (seed == 0) {
    div_t temp = div(numEngines, maxIndex);
    numEngines++;
    seed = seedTable[abs(temp.rem)][0] ^ ((abs(temp.quot) & 0x007fffff) << 8);
  }

// The initialisation is carried out using a Multiplicative
// Congruential generator using formula constants of L'Ecuyer 
// as described in "A review of pseudorandom number generators"
// (Fred James) published in Computer Physics Communications 60 (1990)
// pages 329-344

  const int ecuyer_a = 53668;
  const int ecuyer_b = 40014;
  const int ecuyer_c = 12211;
  const int ecuyer_d = 2147483563;

  const int lux_levels[5] = {0,24,73,199,365};  

// number of additional random numbers that need to be 'thrown away'
// every 24 numbers is set using luxury level variable.

  theSeed = seed;
  if( (lux > 4)||(lux < 0) ){
     if(lux >= 24){
        nskip = lux - 24;
     }else{
        nskip = lux_levels[3]; // corresponds to default luxury level
     }
  }else{
     luxury = lux;
     nskip = lux_levels[luxury];
  }

   
  long next_seed = seed;
  for(int i = 0;i < 24;i++){
     long k_multiple = next_seed / ecuyer_a;
     next_seed = ecuyer_b * (next_seed - k_multiple * ecuyer_a) 
       - k_multiple * ecuyer_c ;
     if(next_seed < 0)next_seed += ecuyer_d;
     float_seed_table[i] = (next_seed % int_modulus) * mantissa_bit_24;
  }     

  i_lag = 23;
  j_lag = 9;
  carry = 0. ;

  if( float_seed_table[23] == 0. ) carry = mantissa_bit_24;
  
  count24 = 0;
}

double SprRanluxEngine::flat() {

  float next_random;
  float uni;
  int i;

  uni = float_seed_table[j_lag] - float_seed_table[i_lag] - carry;
  if(uni < 0. ){
     uni += 1.0;
     carry = mantissa_bit_24;
  }else{
     carry = 0.;
  }

  float_seed_table[i_lag] = uni;
  i_lag --;
  j_lag --;
  if(i_lag < 0) i_lag = 23;
  if(j_lag < 0) j_lag = 23;

  if( uni < mantissa_bit_12 ){
     uni += mantissa_bit_24 * float_seed_table[j_lag];
     if( uni == 0) uni = mantissa_bit_24 * mantissa_bit_24;
  }
  next_random = uni;
  count24 ++;

// every 24th number generation, several random numbers are generated
// and wasted depending upon the luxury level.

  if(count24 == 24 ){
     count24 = 0;
     for( i = 0; i != nskip ; i++){
         uni = float_seed_table[j_lag] - float_seed_table[i_lag] - carry;
         if(uni < 0. ){
            uni += 1.0;
            carry = mantissa_bit_24;
         }else{
            carry = 0.;
         }
         float_seed_table[i_lag] = uni;
	 i_lag --;
         j_lag --;
         if(i_lag < 0)i_lag = 23;
         if(j_lag < 0) j_lag = 23;
      }
  } 
  return (double) next_random;
}

void SprRanluxEngine::flatArray(int size, double* vect)
{
  float next_random;
  float uni;
  int i;
  int index;

  for (index=0; index<size; ++index) {
    uni = float_seed_table[j_lag] - float_seed_table[i_lag] - carry;
    if(uni < 0. ){
       uni += 1.0;
       carry = mantissa_bit_24;
    }else{
       carry = 0.;
    }

    float_seed_table[i_lag] = uni;
    i_lag --;
    j_lag --;
    if(i_lag < 0) i_lag = 23;
    if(j_lag < 0) j_lag = 23;

    if( uni < mantissa_bit_12 ){
       uni += mantissa_bit_24 * float_seed_table[j_lag];
       if( uni == 0) uni = mantissa_bit_24 * mantissa_bit_24;
    }
    next_random = uni;
    vect[index] = (double)next_random;
    count24 ++;

// every 24th number generation, several random numbers are generated
// and wasted depending upon the luxury level.

    if(count24 == 24 ){
       count24 = 0;
       for( i = 0; i != nskip ; i++){
           uni = float_seed_table[j_lag] - float_seed_table[i_lag] - carry;
           if(uni < 0. ){
              uni += 1.0;
              carry = mantissa_bit_24;
           }else{
              carry = 0.;
           }
           float_seed_table[i_lag] = uni;
           i_lag --;
           j_lag --;
           if(i_lag < 0)i_lag = 23;
           if(j_lag < 0) j_lag = 23;
        }
    }
  }
} 
