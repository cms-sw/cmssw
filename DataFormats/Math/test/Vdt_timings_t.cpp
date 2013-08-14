/*******************************************************************************
 * 
 * Test program to keep under control the speed of the mathematical functions
 * Not as optimised as the math functions, I admit.
 * 
 * ****************************************************************************/

#include <iomanip>
#include <iostream>
#include <stdlib.h>
#include <omp.h>
#include <cmath>
#include <memory>
#include <string>

#include "TRandom3.h"

//#include "VDTMath.h"
#include "../interface/VDTMath.h"

void test_input(int argc, char** argv);
void generate_rndm_numbers(double* rnd_numbers,const unsigned int n_rnd_numbers, double min, double max);
double time_simple_function(double* rnd_numbers,
                                        unsigned int n_rnd_numbers, 
                                        const char*, 
                                        double(*func) (double), double scale =-1);
double time_vector_function(double* rnd_numbers,
                                        unsigned int n_rnd_numbers, 
                                        const char*, 
                                        void (*func) (double const *,double*,const unsigned int), double  scale =-1);

//------------------------------------------------------------------------------

const unsigned int REPETITIONS = 1000;
const double MAX_RND_EXP = vdt::EXP_LIMIT;
const double MIN_RND_LOG = vdt::LOG_LOWER_LIMIT;
const double MAX_RND_LOG = vdt::LOG_UPPER_LIMIT;
const double MAX_RND_SIN = vdt::SIN_UPPER_LIMIT;
const double MIN_RND_SIN = vdt::SIN_LOWER_LIMIT;
const double MAX_RND_ASIN = -1;
const double MIN_RND_ASIN = 1;
const double MAX_RND_TAN = vdt::TAN_LIMIT;
const double MAX_RND_ATAN = vdt::ATAN_LIMIT;
const double MAX_RND_SQRT = vdt::SQRT_LIMIT;

int main(int argc, char** argv){

  // Usage
  test_input(argc,argv);

  // print the instruction sets that are used
  vdt::print_instructions_info();

  // Fill a vector of random numbers
  int n_rnd_numbers = 100000;
  if (argc>=2)
    n_rnd_numbers = atoi(argv[1]);

  // Presentation of the test_input
  std::cout << "--------------------------------------------------------\n";
  std::cout << "Testing " << n_rnd_numbers << " random numbers " << REPETITIONS << " times.\n";  
   
  double* rnd_numbers = new double[n_rnd_numbers];
 
  // Generate the random numbers
  generate_rndm_numbers(rnd_numbers,n_rnd_numbers,-MAX_RND_EXP, MAX_RND_EXP);

  // Test the timings
  double scale=1;
  scale = time_simple_function(rnd_numbers,n_rnd_numbers, "std::exp", std::exp, scale );
  time_simple_function(rnd_numbers,n_rnd_numbers, "vdt::fast_exp", vdt::fast_exp, scale);  
  time_vector_function(rnd_numbers,n_rnd_numbers, "vdt::std_exp_vect", vdt::std_exp_vect, scale );
  time_vector_function(rnd_numbers,n_rnd_numbers, "vdt::fast_exp_vect_46", vdt::fast_exp_vect_46, scale );
  time_vector_function(rnd_numbers,n_rnd_numbers, "vdt::fast_exp_vect", vdt::fast_exp_vect, scale );  
  
  generate_rndm_numbers(rnd_numbers,n_rnd_numbers,MIN_RND_LOG, MAX_RND_LOG);  
  scale = time_simple_function(rnd_numbers,n_rnd_numbers, "std::log", std::log );
  time_simple_function(rnd_numbers,n_rnd_numbers, "vdt::fast_log", vdt::fast_log, scale);  
  time_vector_function(rnd_numbers,n_rnd_numbers, "vdt::std_log_vect", vdt::std_log_vect, scale );  
  time_vector_function(rnd_numbers,n_rnd_numbers, "vdt::fast_log_vect_46", vdt::fast_log_vect_46, scale );  
  time_vector_function(rnd_numbers,n_rnd_numbers, "vdt::fast_log_vect", vdt::fast_log_vect, scale );

  generate_rndm_numbers(rnd_numbers,n_rnd_numbers,-MAX_RND_SIN, MAX_RND_SIN);
  scale = time_simple_function(rnd_numbers,n_rnd_numbers, "std::sin", std::sin);
  time_simple_function(rnd_numbers,n_rnd_numbers, "vdt::fast_sin", vdt::fast_sin, scale );
  time_vector_function(rnd_numbers,n_rnd_numbers, "vdt::std_sin_vect", vdt::std_sin_vect, scale );
  time_vector_function(rnd_numbers,n_rnd_numbers, "vdt::fast_sin_vect", vdt::fast_sin_vect, scale );

  time_simple_function(rnd_numbers,n_rnd_numbers, "std::cos", std::cos );
  scale = time_simple_function(rnd_numbers,n_rnd_numbers, "vdt::fast_cos", vdt::fast_cos, scale );
  time_vector_function(rnd_numbers,n_rnd_numbers, "vdt::std_cos_vect", vdt::std_cos_vect, scale );
  time_vector_function(rnd_numbers,n_rnd_numbers, "vdt::fast_cos_vect", vdt::fast_cos_vect, scale );

  generate_rndm_numbers(rnd_numbers,n_rnd_numbers, MIN_RND_ASIN, MAX_RND_ASIN);
  scale = time_simple_function(rnd_numbers,n_rnd_numbers, "std::asin", std::asin );
  time_simple_function(rnd_numbers,n_rnd_numbers, "vdt::fast_asin", vdt::fast_asin, scale );    
  time_vector_function(rnd_numbers,n_rnd_numbers, "vdt::std_asin_vect", vdt::std_asin_vect, scale );
  time_vector_function(rnd_numbers,n_rnd_numbers, "vdt::fast_asin_vect", vdt::fast_asin_vect, scale );

  scale = time_simple_function(rnd_numbers,n_rnd_numbers, "std::acos", std::acos );
  time_simple_function(rnd_numbers,n_rnd_numbers, "vdt::fast_acos", vdt::fast_acos, scale );    
  time_vector_function(rnd_numbers,n_rnd_numbers, "vdt::std_acos_vect", vdt::std_acos_vect, scale );
  time_vector_function(rnd_numbers,n_rnd_numbers, "vdt::fast_acos_vect", vdt::fast_acos_vect, scale );  
  
  generate_rndm_numbers(rnd_numbers,n_rnd_numbers,-MAX_RND_TAN, MAX_RND_TAN);
  scale = time_simple_function(rnd_numbers,n_rnd_numbers, "std::tan", std::tan );
  time_simple_function(rnd_numbers,n_rnd_numbers, "vdt::fast_tan", vdt::fast_tan , scale);
  time_vector_function(rnd_numbers,n_rnd_numbers, "vdt::std_tan_vect", vdt::std_tan_vect, scale );
  time_vector_function(rnd_numbers,n_rnd_numbers, "vdt::fast_tan_vect", vdt::fast_tan_vect, scale );

  generate_rndm_numbers(rnd_numbers,n_rnd_numbers,-MAX_RND_ATAN, MAX_RND_ATAN);
  scale = time_simple_function(rnd_numbers,n_rnd_numbers, "std::atan", std::atan );
  time_simple_function(rnd_numbers,n_rnd_numbers, "vdt::fast_atan", vdt::fast_atan, scale );
  time_vector_function(rnd_numbers,n_rnd_numbers, "vdt::std_atan_vect", vdt::std_atan_vect, scale );
  time_vector_function(rnd_numbers,n_rnd_numbers, "vdt::fast_atan_vect", vdt::fast_atan_vect, scale );

  generate_rndm_numbers(rnd_numbers,n_rnd_numbers, 0, MAX_RND_SQRT);
  scale = time_simple_function(rnd_numbers,n_rnd_numbers, "std::isqrt", vdt::std_isqrt );
  time_simple_function(rnd_numbers,n_rnd_numbers, "vdt::fast_isqrt", vdt::fast_isqrt, scale );
  time_simple_function(rnd_numbers,n_rnd_numbers, "vdt::fast_approx_isqrt", vdt::fast_approx_isqrt, scale );
  time_vector_function(rnd_numbers,n_rnd_numbers, "vdt::std_isqrt_vect", vdt::std_isqrt_vect, scale );  
  time_vector_function(rnd_numbers,n_rnd_numbers, "vdt::fast_isqrt_vect", vdt::fast_isqrt_vect, scale );  
  time_vector_function(rnd_numbers,n_rnd_numbers, "vdt::fast_approx_isqrt_vect", vdt::fast_approx_isqrt_vect, scale );  

  scale = time_simple_function(rnd_numbers,n_rnd_numbers, "vdt::std_inv", vdt::std_inv ); 
  time_simple_function(rnd_numbers,n_rnd_numbers, "vdt::fast_inv", vdt::fast_inv, scale ); 
  time_simple_function(rnd_numbers,n_rnd_numbers, "vdt::fast_approx_inv", vdt::fast_approx_inv, scale ); 
  time_vector_function(rnd_numbers,n_rnd_numbers, "vdt::std_inv_vect", vdt::std_inv_vect, scale );  
  time_vector_function(rnd_numbers,n_rnd_numbers, "vdt::fast_inv_vect", vdt::fast_inv_vect, scale );    
  time_vector_function(rnd_numbers,n_rnd_numbers, "vdt::fast_approx_inv_vect", vdt::fast_approx_inv_vect, scale );    

   delete[] rnd_numbers;

}

//------------------------------------------------------------------------------

double time_vector_function(double* rnd_numbers,
                            unsigned int n_rnd_numbers, 
                            const char* name, 
                            void (*func) (double const *,double*,const unsigned int),
                            double scale){

  double* cache = new double [n_rnd_numbers];

  double init = 0;
  double delta_t=0;
  double tot_time=0;
  double tot_time_2=0; 
  for (unsigned int j=0;j<REPETITIONS;++j){
    init = omp_get_wtime();
    func(rnd_numbers,cache,n_rnd_numbers);
    delta_t = omp_get_wtime() - init;
    delta_t*=1000.;    
    tot_time+=delta_t;
    tot_time_2+=delta_t*delta_t; 
    // to fool the extreme compiler optimisations like -Ofast
    for (unsigned int i=0;i<n_rnd_numbers;++i)
      if (rnd_numbers[i] == 0x7FF8000000000000){
        std::cout << "Found vector!\n";    
        std::cout << rnd_numbers[i] << " " << cache[i]<<"\n";
        }
  }
  double mean_time = tot_time/REPETITIONS;
  double mean_time_e = sqrt((tot_time_2 - tot_time*tot_time/REPETITIONS)/(REPETITIONS-1));
  std::cout << "o Timing of "<< name << " is " << mean_time << "+-" << mean_time_e  << " ms" ;
  if (scale >0)
    printf(" ---> %2.2f X", scale/ mean_time);
  
  std::cout << std::endl;
  
  delete[] cache;
  
  return mean_time;

}

//------------------------------------------------------------------------------

double time_simple_function(double* rnd_numbers,
			  unsigned int n_rnd_numbers,
			  const char* name, 
			  double(*func) (double), double scale){

  double* cache = new double [n_rnd_numbers];

  double init = 0;
  double delta_t=0;
  double tot_time=0;  
  double tot_time_2=0;  
  for (unsigned int j=0;j<REPETITIONS;++j){
    init = omp_get_wtime();
    for (unsigned int i=0;i<n_rnd_numbers;++i)
      cache[i]=func(rnd_numbers[i]);
    delta_t = omp_get_wtime() - init;
    delta_t*=1000.;    
    tot_time+=delta_t;   
    tot_time_2+=delta_t*delta_t;  
    // to fool the extreme compiler optimisations like -Ofast
    for (unsigned int i=0;i<n_rnd_numbers;++i)
      if (rnd_numbers[i] == 0x7FF8000000000000){
        std::cout << "Found!\n";
        }
    }

  double mean_time = tot_time/REPETITIONS;
  double mean_time_e = sqrt((tot_time_2 - tot_time*tot_time/REPETITIONS)/(REPETITIONS-1));
  std::cout << "o Timing of "<< name << " is " << mean_time << "+-" << mean_time_e << " ms";
  
  if (scale >0)
    printf(" ---> %2.2f X", scale/mean_time);
  
  std::cout << std::endl;
  
  delete [] cache;
  return mean_time;
}

//------------------------------------------------------------------------------
void generate_rndm_numbers(double* rnd_numbers,const unsigned int n_rnd_numbers,double min, double max){
/**
 * Generate between -MAX_RND and MAX_RND double numbers
 **/
  TRandom3 rndgen;
  double init = omp_get_wtime();    
  for (unsigned int i=0;i<n_rnd_numbers;++i){      
    rnd_numbers[i] = rndgen.Uniform(min,max);
//     std::cout << "o " << min << " " << max << " " << rnd_numbers[i] << std::endl;
  }
  double delta_t = omp_get_wtime() - init;
  std::cout << "\n*** " << n_rnd_numbers 
            << " numbers (" << min << "," << max<< ") generated in " 
            << delta_t << " s \n";

  }

//------------------------------------------------------------------------------

void test_input(int argc, char** argv){

  if (argc!=1 and argc!=2){
    std::cout << "Usage: " << argv[0] << " [number of random numbers]\n";
    exit(1);
    }
}

//------------------------------------------------------------------------------
