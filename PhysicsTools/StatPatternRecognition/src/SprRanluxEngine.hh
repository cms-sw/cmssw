// $Id: SprRanluxEngine.hh,v 1.2 2006/10/19 21:27:52 narsky Exp $
// -*- C++ -*-
//
// -----------------------------------------------------------------------
//                             HEP Random
//                        --- RanluxEngine ---
//                          class header file
// -----------------------------------------------------------------------
// This file is part of Geant4 (simulation toolkit for HEP).
//
// The algorithm for this random engine has been taken from the original
// implementation in FORTRAN by Fred James as part of the MATHLIB HEP
// library.
// The initialisation is carried out using a Multiplicative Congruential
// generator using formula constants of L'Ecuyer as described in "F.James,
// Comp. Phys. Comm. 60 (1990) 329-344".

// =======================================================================
// Adeyemi Adesanya - Created: 6th November 1995
// Gabriele Cosmo - Adapted & Revised: 22nd November 1995
// Adeyemi Adesanya - Added setSeeds() method: 2nd February 1996
// Gabriele Cosmo - Added flatArray() method: 8th February 1996
//                - Added methods for engine status: 19th November 1996
//                - Added default luxury value for setSeed()
//                  and setSeeds(): 21st July 1997
// J.Marraffino   - Added stream operators and related constructor.
//                  Added automatic seed selection from seed table and
//                  engine counter: 14th Feb 1998
// Ken Smith      - Added conversion operators:  6th Aug 1998
// Mark Fischler    Methods put, get for instance save/restore 12/8/04    
// Mark Fischler    methods for anonymous save/restore 12/27/04    
// =======================================================================

#ifndef _SprRanluxEngine_HH
#define _SprRanluxEngine_HH

/**
 * @author
 * @ingroup random
 */
class SprRanluxEngine {

public:

  SprRanluxEngine( long seed = 0, int lux = 3 );
  virtual ~SprRanluxEngine();
  // Constructors and destructor

// Luxury level is set in the same way as the original FORTRAN routine.
//  level 0  (p=24): equivalent to the original RCARRY of Marsaglia
//           and Zaman, very long period, but fails many tests.
//  level 1  (p=48): considerable improvement in quality over level 0,
//           now passes the gap test, but still fails spectral test.
//  level 2  (p=97): passes all known tests, but theoretically still
//           defective.
//  level 3  (p=223): DEFAULT VALUE.  Any theoretically possible
//           correlations have very small chance of being observed.
//  level 4  (p=389): highest possible luxury, all 24 bits chaotic.

  double flat();
  // It returns a pseudo random number between 0 and 1,
  // excluding the end points.

  void flatArray (int size, double* vect);
  // Fills the array "vect" of specified size with flat random values.

  void setSeed(long seed, int lux=3);
  // Sets the state of the algorithm according to seed.

private:
  long theSeed;

  int nskip, luxury;
  float float_seed_table[24];
  int i_lag,j_lag;  
  float carry;
  int count24;
  static const int int_modulus;
  static const double mantissa_bit_24;
  static const double mantissa_bit_12;
  static int numEngines;
  static const int maxIndex;
  static const long seedTable[215][2];
};

#endif
