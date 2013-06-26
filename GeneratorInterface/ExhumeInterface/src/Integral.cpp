//Simpson integrator, written to mirror CERNLIBS one

#include <cmath>

double dsimps_(double* F,double* A,double* B,int* N){
  
  double a = *A;
  double b = *B;
  int n = *N;
  double deltaX = (b-a)/n;
  double I=F[0] + F[n];
  
  for(int i=1;i<=n/2;i++){
    int j=2*i;
    int k =2*i-1;
    if(j<n) I += 2*F[j];
    if(k<n) I += 4*F[k];
    
  }
  return(I*deltaX/3);
}
