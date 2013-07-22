/*******************************************************************************
 * 
 * Test program to keep under control the numerical output of the functions
 * Not as optimised as the math functions, I admit.
 * 
 * ****************************************************************************/

#include <cmath>
#include <bitset>
#include <iomanip>
#include <string>
#include <algorithm>
#include <assert.h>
#include <stdlib.h>

#include "TRandom3.h"
#include "TCanvas.h"
#include "TH1F.h"
#include "TCanvas.h"
#include "TROOT.h"

//#include "VDTMath.h"
#include "../interface/VDTMath.h"

//------------------------------------------------------------------------------
void test_input(int argc, char** argv);
void test_simple_function(double* rnd_numbers,
                          unsigned int n_rnd_numbers, 
                          const char*, 
                          double(*func) (double),
                          const char*, 
                          double(*ref_func) (double) = NULL);

void test_vector_function(double* rnd_numbers,
                          unsigned int n_rnd_numbers, 
                          const char*, 
                          void (*func) (double const *,double*,const unsigned int),
                          const char*, 
                          void (*ref_func) (double const *,double*,const unsigned int));

void generate_rndm_numbers(double* rnd_numbers,
                           const unsigned int n_rnd_numbers, 
                           double min, 
                           double max,
                           const char * name);

bool check_doubles(double x, double y, short nbit);
int doubles_bit_diff(double x, double y);
typedef  std::bitset<64> doublebits;
void print_double(doublebits x);
doublebits get_bits_array(double x);
bool get_bit (double d, unsigned short int i);
unsigned long long ll_doubles_diff(double x, double y);
void format_hist(TH1*, int);

//------------------------------------------------------------------------------
const double TH1AXIS_MAX=1e202;
const double MAX_RND_EXP = vdt::EXP_LIMIT;
const double MIN_RND_LOG = vdt::LOG_LOWER_LIMIT;
const double MAX_RND_LOG = vdt::LOG_UPPER_LIMIT;
const double MAX_RND_SIN = vdt::SIN_UPPER_LIMIT;
const double MIN_RND_SIN = vdt::SIN_LOWER_LIMIT;
const double MAX_RND_ASIN = 1;
const double MIN_RND_ASIN = -1;
const double MAX_RND_TAN = vdt::TAN_LIMIT;
const double MAX_RND_ATAN = vdt::ATAN_LIMIT;
const double MAX_RND_SQRT = vdt::SQRT_LIMIT;

// const double EPSILON = 1e-10;
const short int BIT_TOLERANCE = 5;

int main(int argc, char** argv){

  // Usage
  test_input(argc,argv);

  // print the instruction sets that are used
  vdt::print_instructions_info();
  
  // Fill a vector of random numbers
  int n_rnd_numbers = 10000;
  if (argc>=2)
    n_rnd_numbers = atoi(argv[1]);

  // Presentation of the test_input
  std::cout << "--------------------------------------------------------\n";
  std::cout << "Testing " << n_rnd_numbers << " random numbers\n";  
   
  double* rnd_numbers = new double[n_rnd_numbers];
 
  // Generate the random numbers
  generate_rndm_numbers(rnd_numbers,n_rnd_numbers,-MAX_RND_EXP, MAX_RND_EXP,"exp");

  // Test the sanity
  test_simple_function(rnd_numbers,n_rnd_numbers, 
                       "vdt::fast_exp", vdt::fast_exp, "std::exp", std::exp);  
  test_vector_function(rnd_numbers,n_rnd_numbers, 
                       "vdt::fast_exp_vect_46", vdt::fast_exp_vect_46, "vdt::std_exp_vect", vdt::std_exp_vect );
  test_vector_function(rnd_numbers,n_rnd_numbers, 
                       "vdt::fast_exp_vect", vdt::fast_exp_vect, "vdt::std_exp_vect", vdt::std_exp_vect  );  
  
  generate_rndm_numbers(rnd_numbers,n_rnd_numbers,MIN_RND_LOG, MAX_RND_LOG,"log");  
  test_simple_function(rnd_numbers,n_rnd_numbers, 
                       "vdt::fast_log", vdt::fast_log, "std::log", std::log);  
  test_vector_function(rnd_numbers,n_rnd_numbers, 
                       "vdt::fast_log_vect_46", vdt::fast_log_vect_46, "vdt::std_log_vect", vdt::std_log_vect);  
  test_vector_function(rnd_numbers,n_rnd_numbers, 
                       "vdt::fast_log_vect", vdt::fast_log_vect, "vdt::std_log_vect", vdt::std_log_vect );

  generate_rndm_numbers(rnd_numbers,n_rnd_numbers,MIN_RND_SIN, MAX_RND_SIN,"sin");
  test_simple_function(rnd_numbers,n_rnd_numbers,
                       "vdt::fast_sin", vdt::fast_sin, "std::sin", std::sin  );                           
  test_simple_function(rnd_numbers,n_rnd_numbers,
                       "vdt::fast_cos", vdt::fast_cos, "std::cos", std::cos  );
  test_vector_function(rnd_numbers,n_rnd_numbers, 
                        "vdt::fast_sin_vect", vdt::fast_sin_vect, "vdt::std_sin_vect", vdt::std_sin_vect );
  test_vector_function(rnd_numbers,n_rnd_numbers, 
                        "vdt::fast_cos_vect", vdt::fast_cos_vect, "vdt::std_cos_vect", vdt::std_cos_vect );

  generate_rndm_numbers(rnd_numbers,n_rnd_numbers,-1, 1,"asin");
  test_simple_function(rnd_numbers,n_rnd_numbers,
                       "vdt::fast_asin", vdt::fast_asin, "std::asin", std::asin  );                           
  test_simple_function(rnd_numbers,n_rnd_numbers,
                       "vdt::fast_acos", vdt::fast_acos, "std::acos", std::acos  );
  test_vector_function(rnd_numbers,n_rnd_numbers, 
                        "vdt::fast_asin_vect", vdt::fast_asin_vect, "vdt::std_asin_vect", vdt::std_asin_vect );
  test_vector_function(rnd_numbers,n_rnd_numbers, 
                        "vdt::fast_acos_vect", vdt::fast_acos_vect, "vdt::std_acos_vect", vdt::std_acos_vect );
  
  
  generate_rndm_numbers(rnd_numbers,n_rnd_numbers,-MAX_RND_TAN, MAX_RND_TAN,"tan");
  test_simple_function(rnd_numbers,n_rnd_numbers,
                       "vdt::fast_tan", vdt::fast_tan, "std::tan", std::tan  );
  test_vector_function(rnd_numbers,n_rnd_numbers, 
                        "vdt::fast_tan_vect", vdt::fast_tan_vect, "vdt::std_tan_vect", vdt::std_tan_vect );

  generate_rndm_numbers(rnd_numbers,n_rnd_numbers,-1*MAX_RND_ATAN, MAX_RND_ATAN,"atan");
  test_simple_function(rnd_numbers,n_rnd_numbers,
                       "vdt::fast_atan", vdt::fast_atan, "std::atan", std::atan  );
  test_vector_function(rnd_numbers,n_rnd_numbers, 
                        "vdt::fast_atan_vect", vdt::fast_atan_vect, "vdt::std_atan_vect", vdt::std_atan_vect );


  generate_rndm_numbers(rnd_numbers,n_rnd_numbers,0, MAX_RND_SQRT,"sqrt");
  test_simple_function(rnd_numbers,n_rnd_numbers, 
                       "vdt::fast_isqrt", vdt::fast_isqrt, "vdt::std_isqrt", vdt::std_isqrt);  
  test_simple_function(rnd_numbers,n_rnd_numbers, 
                       "vdt::fast_approx_isqrt", vdt::fast_approx_isqrt, "vdt::std_isqrt", vdt::std_isqrt);
  test_vector_function(rnd_numbers,n_rnd_numbers, 
                       "vdt::fast_isqrt_vect", vdt::fast_isqrt_vect, "vdt::std_isqrt_vect", vdt::std_isqrt_vect );
  test_vector_function(rnd_numbers,n_rnd_numbers, 
                       "vdt::fast_approx_isqrt_vect", vdt::fast_approx_isqrt_vect, "vdt::std_isqrt_vect", vdt::std_isqrt_vect );

  generate_rndm_numbers(rnd_numbers,n_rnd_numbers,-MAX_RND_SQRT, MAX_RND_SQRT,"inv");
  test_simple_function(rnd_numbers,n_rnd_numbers, 
                       "vdt::fast_inv", vdt::fast_inv, "vdt::std_inv", vdt::std_inv);
  test_simple_function(rnd_numbers,n_rnd_numbers, 
                       "vdt::fast_approx_inv", vdt::fast_approx_inv, "vdt::std_inv", vdt::std_inv);
  test_vector_function(rnd_numbers,n_rnd_numbers, 
                       "vdt::fast_inv_vect", vdt::fast_inv_vect, "vdt::std_inv_vect", vdt::std_inv_vect );
  test_vector_function(rnd_numbers,n_rnd_numbers, 
                       "vdt::fast_approx_inv_vect", vdt::fast_approx_inv_vect, "vdt::std_inv_vect", vdt::std_inv_vect );

  delete[] rnd_numbers;

}
  
//------------------------------------------------------------------------------
  
void generate_rndm_numbers(double* rnd_numbers,const unsigned int n_rnd_numbers,double min, double max, const char* name){
/**
 * Generate between -MAX_RND and MAX_RND double numbers
 **/


  std::string hist_name(name);
  hist_name+="_random_numbers";
  std::string hist_title(name);
  hist_title+=" Random Numbers;Number;Freq";
  //std::cout << "Random limits " << min << " " << max << std::endl;
  double hmin=min;
  double hmax=max;
  if (hmin<-TH1AXIS_MAX)
    hmin=-TH1AXIS_MAX;
  if (hmax>TH1AXIS_MAX)
    hmax=TH1AXIS_MAX;
  
  TH1F rndm_h(hist_name.c_str(), hist_title.c_str(),10000 , hmin,hmax);    
  format_hist(&rndm_h, kRed);

  gROOT->SetStyle("Plain");


  
  TRandom3 rndgen(1);
  for (unsigned int i=0;i<n_rnd_numbers;++i){      
    rnd_numbers[i] = rndgen.Uniform(hmin,hmax);
    //std::cout << "o " << rnd_numbers[i] << std::endl;
    rndm_h.Fill(rnd_numbers[i]);
  }


  TCanvas c;c.cd();
  if (rndm_h.GetSumOfWeights()>0)
    rndm_h.DrawNormalized("HIST");
  else
    rndm_h.Draw("HIST");
  hist_name+=".png";
  c.Print(hist_name.c_str());

  }

//------------------------------------------------------------------------------


void get_min_max (unsigned long long* arr, const unsigned int size, unsigned long long& min, unsigned long long& max ){
  
  max=std::numeric_limits<unsigned long long>::min();
  min=std::numeric_limits<unsigned long long>::max();
  
  for (unsigned int i=0;i<size;++i){
    if (arr[i]<min)
      min=arr[i];
    if (arr[i]>max)
      max=arr[i];          
  }
  
}

//------------------------------------------------------------------------------
// c++11!!
std::string make_hist_title(const char* name, const char* ref_name  ){
  
  std::string title (name);
  title+=" VS ";
  title+=ref_name;
  title+=";Bit Diff as int; Frequency";
  return title;
} 

std::string make_plot_filename (const char* name, const char* ref_name ){
  
  std::string filename (name);
  filename+="_VS_";
  filename+=ref_name;
  filename+=".png";
  std::replace(filename.begin(), filename.end(), ':', '_');

  return filename;
} 


void format_hist (TH1*h, int color=kBlue){  
  
  h->SetLineColor(color);
  h->SetLineWidth(3);
  h->SetFillStyle(3002);
  h->SetFillColor(color);
}
  
//------------------------------------------------------------------------------

void test_simple_function(double* rnd_numbers,
                          unsigned int n_rnd_numbers, 
                          const char* name, 
                          double(*func) (double),
                          const char* ref_name, 
                          double(*ref_func) (double)){
    
  std:: cout << "Testing " << name << " VS " << ref_name << std::endl;
  
  double fx1=0;
  double fx2=0;
  unsigned long long* diffs = new unsigned long long[n_rnd_numbers];  
  for (unsigned int i=0;i<n_rnd_numbers;++i){  
    fx1=func(rnd_numbers[i]);
    fx2=ref_func(rnd_numbers[i]);
//     check_dp(rnd_numbers[i],fx1,fx2);
//    doubles_bit_diff(fx1,fx2);
    diffs[i] = doubles_bit_diff(fx1,fx2);
    if (not check_doubles(fx1,fx2,BIT_TOLERANCE)){
         std::cout << std::setprecision(15);
         std::cout << "Difference between "<<name <<" and "<<ref_name <<" : " << "x=" << rnd_numbers[i]
                 << "    --> f1(x)="<< fx1 << " --- f2(x)=" << fx2 << std::endl;
        }
    }

  // Histogram for the differences
  unsigned long long min,max;
  get_min_max(diffs,n_rnd_numbers,min,max );
  std::string hist_title(make_hist_title(name,ref_name));
  TH1F diff_h(name, hist_title.c_str(),max-min +10 , min-.5,max-.5 +10 );    
  format_hist(&diff_h);
  for (unsigned int i=0;i<n_rnd_numbers;++i){
//     std::cout << "Filling " << name << " with " << diffs[i] << "\n";
    diff_h.Fill(diffs[i]);
  }
  TCanvas c;c.cd();
  diff_h.DrawNormalized("HIST");
  c.Print(make_plot_filename(name,ref_name).c_str());

  delete [] diffs;
  
  std:: cout << "----------------" << std::endl;                              
  
  }


//------------------------------------------------------------------------------

void test_vector_function(double* rnd_numbers,
                          unsigned int n_rnd_numbers, 
                          const char* name, 
                          void (*func) (double const *,double*,const unsigned int),
                          const char* ref_name, 
                          void (*ref_func) (double const *,double*,const unsigned int)){
  
  std:: cout << "Testing " << name << " VS " << ref_name << std::endl;  
  
  double* fx1  = new double[n_rnd_numbers];  
  double* fx2 = new double[n_rnd_numbers];     
  unsigned long long* diffs = new unsigned long long[n_rnd_numbers];     
  
  func (rnd_numbers,fx1, n_rnd_numbers);
  ref_func (rnd_numbers,fx2, n_rnd_numbers);
  
  for (unsigned int i=0;i<n_rnd_numbers;++i){
    diffs[i] = doubles_bit_diff(fx1[i],fx2[i]);
    if (not check_doubles(fx1[i],fx2[i],BIT_TOLERANCE)){
         std::cout << std::setprecision(15);
         std::cout << "Difference between "<<name <<" and "<<ref_name <<" : " << "x=" << rnd_numbers[i]
                 << "    --> f1(x)="<< fx1[i] << " --- f2(x)=" << fx2[i] << std::endl;
      }
    }
    
  // Histogram for the differences
  unsigned long long min,max;
  get_min_max(diffs,n_rnd_numbers,min,max );
  std::string hist_title(make_hist_title(name,ref_name));
  TH1F diff_h(name, hist_title.c_str(),max-min +10 , min-.5,max-.5 +10 );    
  format_hist(&diff_h);
  for (unsigned int i=0;i<n_rnd_numbers;++i){
//     std::cout << "Filling " << name << " with " << diffs[i] << "\n";
    diff_h.Fill(diffs[i]);
  }
  TCanvas c;c.cd();
  diff_h.DrawNormalized("HIST");
  c.Print(make_plot_filename(name,ref_name).c_str());
  
    
  delete [] fx1;
  delete [] fx2;
  delete [] diffs;
  
  std:: cout << "----------------" << std::endl;
  }


//------------------------------------------------------------------------------

const long long mask = 0x0000000000000001;

//------------------------------------------------------------------------------

bool get_bit (double d, unsigned short int i){

  union { double d; unsigned long long n; } tmp;
  tmp.d=d;
  unsigned long long n = tmp.n ;

  for (;i!=0;i--) n >>= 1;

  return  (n & mask);
  }

//------------------------------------------------------------------------------

// returns a bitarray with the bits of the double
doublebits get_bits_array(double x){
    doublebits barray;
    for (int i=0;i<64;++i)
     barray [i] = get_bit(x, i);
    return barray;
 }

//------------------------------------------------------------------------------

// prints a double separating sign exp and mantissa
void print_double(double xd){
    doublebits x = get_bits_array(xd);
    int i =63;
    std::cout << x[i] << " ";
    for (i=62;i>51;i--)
        std::cout << x[i];;
    std::cout << " ";
    for (;i>0;i--)
        std::cout << x[i];
    std::cout << x[i] << "\n";
    }
//------------------------------------------------------------------------------
int log2p1(unsigned long long i){
//   std::cout << "log2p1 of " << i ;
  int log2p1_index=0;
  for (;log2p1_index<=64;log2p1_index++){    
    if (i==0){
//       std::cout << " is " << j << std::endl;
      break;
    }
    i >>= 1;
  }
  
  return log2p1_index;
  
}

//------------------------------------------------------------------------------
// the bit are counted from less significative
bool check_doubles(double x, double y, short nbit){

  short diff_bit = doubles_bit_diff(x,y);
  if (diff_bit>=nbit){ // they are different
    std::cout << "\n\nThe numbers " << x << " and " << y << " are different at the bit " << diff_bit << "\n";
    print_double(x);
    print_double(y);    

    // now the arrow
   short compl_diff_bit = 63-diff_bit;
    for (short i=0;i<=compl_diff_bit;++i) std::cout << " ";
    if (compl_diff_bit>0)std::cout << " ";
    if (compl_diff_bit>12)std::cout << " ";

    std::cout << "^\n";
    return false;
    }
  return true;

}

//------------------------------------------------------------------------------

unsigned long long ll_doubles_diff(double x, double y){
  
  // New nice implementation
  unsigned long long xll = vdt::d2ll(x);
//   print_double(x);
  unsigned long long yll = vdt::d2ll(y);
//   print_double(y);
  unsigned long long diffll = std::max(xll,yll) - std::min(xll,yll);  
  return diffll;
}

//------------------------------------------------------------------------------

int doubles_bit_diff(double x, double y){
  return log2p1(ll_doubles_diff(x,y));  
}

//------------------------------------------------------------------------------

void test_input(int argc, char** argv){

  if (argc!=1 and argc!=2){
    std::cout << "Usage: " << argv[0] << " [number of random numbers]\n";
    exit(1);
    }
}

//------------------------------------------------------------------------------



