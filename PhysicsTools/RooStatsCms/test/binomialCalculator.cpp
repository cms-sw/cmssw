#include <cmath>
#include <iostream>
#include <sstream>
#include "PhysicsTools/RooStatsCms/interface/FeldmanCousinsBinomialInterval.h"
#include "PhysicsTools/RooStatsCms/interface/ClopperPearsonBinomialInterval.h"

using namespace std;

const char * prog = "binomialCalculator";

void usage(const char * prog) {
  cerr << prog << ": computes binomial confidence interval for a given extraction." <<endl
       << "usage:" << endl
       << prog << " n N" << endl
       << " where N is the total number of entries, n the number of efficient entries" <<endl ;
  exit(1);
}

template<typename T>
void compute(int n, int N, T& c) {
  const double cl[] = { 0.682, 0.90, 0.95 };   
  for(unsigned int i = 0; i < sizeof(cl)/sizeof(double); ++i) {
    double alpha = 1 - cl[i];
    c.init(alpha);
    double eff = double(n)/double(N);
    c.calculate(n, N);
    double eeffl = eff - c.lower();
    double eeffh = c.upper() - eff;
    cout << "CL = " << 1 - alpha << " (alpha = 1 - CL =" << alpha << ")" << endl; 
    cout << "eff = n/N = " << eff << " +/-(+" << eeffl << " -" << eeffh << ")" << endl;
    cout << cl[i]*100 << "% CL interval: ["  << c.lower() << ", " << c.upper() <<"]" << endl;
  }
}

int main(int argc, char * arg[]) {
  if(argc != 3) {
    cerr << prog << ": error: number of arguments should be 2, passed " << argc - 1<< endl; 
    usage(prog); 
  }
  stringstream sn(arg[1]);
  stringstream sN(arg[2]);
  int n, N;
  sn >> n;
  sN >> N;
  if(n > N) {
    cerr << prog << ": error: n can't be greater than N. Passed " << n << " > " << n << endl;
    usage(arg[0]);
  }
  cout << "Clopper-Pearson intervals:" << endl;
  ClopperPearsonBinomialInterval cp;
  compute(n, N, cp);
  cout << "Feldman-Cousins intervals:" << endl;
  FeldmanCousinsBinomialInterval fc;
  compute(n, N, fc);
  return 0;
}

