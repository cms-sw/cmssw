#include "DataFormats/Math/interface/VDTMath.h"
#include <stdlib.h>
#include <iostream>
#include <cassert>
#include <iomanip>

/*******************************************************************************
 * 
 * EXP IMPLEMENTATIONS
 * 
 ******************************************************************************/ 

//------------------------------------------------------------------------------

void vdt::__future_fast_exp_vect(double* input, double* output, const unsigned int arr_size) {

    for (unsigned int iarg = 0; iarg < arr_size; ++iarg) 
        output[iarg] = vdt::fast_exp(input[iarg]);
  }

//------------------------------------------------------------------------------

void vdt::fast_exp_vect(double* input, double* output, const unsigned int arr_size) {
  /**
    Vectorise the calculation in order to calculate exp of an array
    **/
  // input & output must not point to the same memory location
  // 	assert( input != output );

  int n;
  int* nv = new int[arr_size];
  double xx, px, qx;
  ieee754 u;

  // for vectorisation
  double x;
  /*gcc 46x
  Vector inside of loop cost: 52
  Vector outside of loop cost: 75
  Scalar iteration cost: 26
  Scalar outside cost: 1
  prologue iterations: 0
  epilogue iterations: 2
  Calculated minimum iters for profitability: 4*/
  for (unsigned int i = 0; i < arr_size; ++i) {
    x = input[i];

    nv[i] = n = int(LOG2E * x + 0.5);//std::floor( LOG2E * x + 0.5 );

    px = n;

    x -= px * 6.93145751953125E-1;
    x -= px * 1.42860682030941723212E-6;

    xx = x * x;

    // px = x * P(x**2).
    px = PX1exp;
    px *= xx;
    px += PX2exp;
    px *= xx;
    px += PX3exp;
    px *= x;

    // Evaluate Q(x**2).
    qx = QX1exp;
    qx *= xx;
    qx += QX2exp;
    qx *= xx;
    qx += QX3exp;
    qx *= xx;
    qx += QX4exp;

    // e**x = 1 + 2x P(x**2)/( Q(x**2) - P(x**2) )
    x = px / (qx - px);
    x = 1.0 + 2.0 * x;

    // partial
    output[i]=x;
                  
    } // end loop on input vals
   

  for (unsigned int i = 0; i < arr_size; ++i) {
      //     Build 2^n in double.
      n=nv[i];
      u.d = 0;
      n += 1023;    
      u.ll = (long long) (n) << 52;
      output[i] = output[i] * u.d; 

      if (input[i] > EXP_LIMIT)
              output[i] = std::numeric_limits<double>::infinity();
      if (input[i] < -EXP_LIMIT)
              output[i] = 0.;     
      }
      
   delete [] nv;
}

//------------------------------------------------------------------------------

void vdt::std_exp_vect(double* input, double* output, const unsigned int arr_size){
  
    for (unsigned int iarg = 0; iarg < arr_size; ++iarg) 
        output[iarg] = std::exp(input[iarg]);
  }   


//------------------------------------------------------------------------------

/*******************************************************************************
 * 
 * LOG IMPLEMENTATIONS
 * 
 ******************************************************************************/ 
 
//------------------------------------------------------------------------------  
// The implementation is such in order to vectorise also with gcc461
void vdt::fast_log_vect(double* original_input, 
                        double* output, 
                        const unsigned int arr_size){

  double* input = new double [arr_size];
  double* x_arr = new double [arr_size];
  int* fe_arr = new int [arr_size];
  
  double y, z;
  double px,qx;
  double x;
  int fe;

  for (unsigned int i = 0; i < arr_size; ++i) {
    input[i] = original_input[i];
    double x= input[i];


    /* separate mantissa from exponent
    */
    
    unsigned long long n = d2ll(x);

    unsigned long long le = ((n >> 52) & 0x7ffL);
    fe = (le-1023);
    n &=0xfffffffffffffLL;
    const unsigned long long p05 = d2ll(0.5);
    n |= p05;
    x = ll2d(n);

    /* logarithm using log(x) = z + z**3 P(z)/Q(z),
    * where z = 2(x-1)/x+1)
    */
    /* logarithm using log(1+x) = x - .5x**2 + x**3 P(x)/Q(x) */

    // blending   
    if( x > SQRTH ) fe+=1.;
    if( x < SQRTH )   x += x;
    x =   x - 1.0;
    
    x_arr[i]=x;
    fe_arr[i]=fe;
    
    }
    
  for (unsigned int i = 0; i < arr_size; ++i) {
    
    x = x_arr[i];
    fe = fe_arr[i];
    
    /* rational form */

    z = x*x;
    px =  PX1log;
    px *= x;    
    px += PX2log;
    px *= x;    
    px += PX3log;
    px *= x; 
    px += PX4log;
    px *= x; 
    px += PX5log;
    px *= x;
    px += PX6log;
    //
    //for the final formula
    px *= x; 
    px *= z;


    qx = x;
    qx += QX1log;
    qx *=x;
    qx += QX2log;
    qx *=x;    
    qx += QX3log;
    qx *=x;    
    qx += QX4log;
    qx *=x;    
    qx += QX5log;

    
    y = px / qx ;
   
    y -= fe * 2.121944400546905827679e-4; 
    y -= 0.5 * z  ;
    
    z = x + y;
    z += fe * 0.693359375;
    output[i]= z;
  }
    
  for (unsigned int i = 0; i < arr_size; ++i) {    
    if (original_input[i] > LOG_UPPER_LIMIT)
      output[i] = std::numeric_limits<double>::infinity();
    if (original_input[i] < LOG_LOWER_LIMIT)
      output[i] =  - std::numeric_limits<double>::infinity();                             
    }
  
  delete [] input;
  delete [] x_arr;
  delete [] fe_arr;
  
}

//------------------------------------------------------------------------------

void vdt::std_log_vect(double* input, double* output,const unsigned int arr_size){
  for (unsigned int i=0;i<arr_size;++i)
    output[i] = std::log(input[i]);
}

//------------------------------------------------------------------------------

void vdt::__future_fast_log_vect(double* input, double* output, const unsigned int arr_size) {
  
  for (unsigned int i=0;i<arr_size;++i)
    output[i] = vdt::fast_log( input[i] );
  
}


//------------------------------------------------------------------------------

/*******************************************************************************
 *
 * SIN COS IMPLEMENTATIONS
 *
 ******************************************************************************/
    
//------------------------------------------------------------------------------
    
    
void vdt::__future_fast_sin_vect(double* input, double* outupt, const unsigned int arr_size) {
    
  for (unsigned int i=0;i<arr_size;++i)
    outupt[0] = vdt::fast_sin( input[i] );
    
}     
    
//------------------------------------------------------------------------------
      
void vdt::__future_fast_cos_vect(double* input, double* outupt, const unsigned int arr_size) {
    
  for (unsigned int i=0;i<arr_size;++i)
    outupt[0] = vdt::fast_cos( input[i] );
    
}


                            
void vdt::print_instructions_info(){
/**
 * Check and print which instructions sets are enabled.
 **/  

  std::cout << "\nList of enabled instructions' sets:\n";
  
#ifdef __SSE2__
        std::cout << " - SSE2 instructions enabled" << std::endl;
#else
        std::cout << " - SSE2 instructions *not* enabled" << std::endl;
#endif
#ifdef __SSE3__
        std::cout << " - SSE3 instructions enabled" << std::endl;
#else
        std::cout << " - SSE3 instructions *not* enabled" << std::endl;
#endif

#ifdef __SSE4_1__
        std::cout << " - SSE4.1 instructions enabled" << std::endl;
#else
        std::cout << " - SSE4.1 instructions *not* enabled" << std::endl;
#endif
#ifdef __AVX__
        std::cout << " - AVX instructions enabled" << std::endl;
#else
        std::cout << " - AVX instructions *not* enabled" << std::endl;
#endif
       std::cout << "\n\n"; 
    } 



