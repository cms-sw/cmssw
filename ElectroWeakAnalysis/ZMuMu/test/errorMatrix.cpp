#include <iostream>
#include <fstream>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>

using namespace std;

struct cov {
  cov(unsigned int _ix, unsigned int _iy, double _cxy, double _rho) :
    ix(_ix), iy(_iy), cxy(_cxy), rho(_rho) { }
  unsigned int ix, iy;
  double cxy, rho;
};

double operator<(const cov & c1, const cov & c2) {
  return fabs(c1.rho) > fabs(c2.rho);
}

int main() {
  const unsigned int n = 24;
  fstream file("error.txt");
  string name[n];
  double err[n][n];
  for(unsigned int i = 0; i < n; ++i)
    file >> name[i];
  for(unsigned int i = 0; i < n; ++i) {
    for(unsigned int j = 0; j < n; ++j) {
      file >> err[i][j];
    }
  }

  for(unsigned int i = 0; i < n; ++i) 
    for(unsigned int j = i; j < n; ++j) 
      if(fabs(err[i][j] - err[j][i])> 1.e-4) {
	cerr << "error: asymmetric matrix";
	exit(1);
      }

  for(unsigned int i = 0; i < n; ++i) {
    cout << "err(" << name[i] << ") = " << sqrt(err[i][i]) << endl;
  }
  
  vector<cov> covs;
  for(unsigned int i = 0; i < n; ++i) 
    for(unsigned int j = i+1; j < n; ++j) {
      double cxy = err[i][j];
      double ex = sqrt(err[i][i]), ey = sqrt(err[j][j]);
      if(ex > 0 && ey > 0) {
	double rho = cxy / (ex * ey);
	covs.push_back(cov(i, j, cxy, rho));
      }
    }
  
  sort(covs.begin(), covs.end());
  for(vector<cov>::const_iterator i = covs.begin(); i != covs.end(); ++i) {
    cout << "cov(" << name[i->ix] << ", " << name[i->iy] << ") = " << i->cxy << ", "
	 << " correlation = " << i->rho << endl;
  }
  return 0;
}
