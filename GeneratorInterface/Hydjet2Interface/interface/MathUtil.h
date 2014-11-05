//
//##############################################################################
//
//            Nikolai Amelin (C) JINR/Dubna 1999
//
//##############################################################################
//

#ifndef NAMathUtil_h
#define NAMathUtil_h 1

#include <iostream>
#include <math.h>
#include <Rtypes.h>

const double GeV   = 1.;
const double fermi = 1.;
const double hbarc = 0.197*GeV*fermi;
const double N_PI  = 3.14159265359;

const double N_INFINITY = 9.0E99;
const double N_SMALL = 1.E-10;


template <class T> inline void SwapObj(T* a, T* b)
{
  T tmp= *a;
    *a = *b;
      *b = tmp;
        }
        
        template <class T> inline void Swap(T& a, T& b)
{
  T tmp = a;
    a = b;
      b = tmp;
        }
        
        template <class T> inline T Min(T a, T b)
{
  return (a < b) ? a : b;
    }
    
    template <class T> inline T Max(T a, T b)
{
  return (a > b) ? a : b;
    }
    
    template <class T> inline T Abs(T a)
{
  return (a > 0) ? a : -a;
    }
    
    template <class T> inline T Sign(T A, T B)
{
  return (B > 0) ? Abs(A) : -Abs(A);
    }
    template <class T> inline T min(T a, T b) 
{
  return (a < b)?a:b;
    }
    
    template <class T> inline T max(T a, T b) 
{
  return (a > b)?a:b;
    }

#endif
