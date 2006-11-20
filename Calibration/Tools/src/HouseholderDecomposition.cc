/** \file HouseholderDecomposition.cc
 *
 * $Date: 2006/08/25 08:53:13 $
 * $Revision: 1.2 $
 *
 * \author Lorenzo Agostino, R.Ofierzynski, CERN
 */

#include "Calibration/Tools/interface/HouseholderDecomposition.h"
#include <cfloat>
#include <cmath>


HouseholderDecomposition::HouseholderDecomposition(bool normalise)
  :normaliseFlag(normalise)
{
}


HouseholderDecomposition::~HouseholderDecomposition()
{
}


vector<float> HouseholderDecomposition::iterate(const vector<vector<float> >& eventMatrix, const vector<float>& energyVector, const int nIter)
{
  vector<float> solution;
  vector<float> theCalibVector(energyVector.size(),1.);
  vector<vector<float> > myEventMatrix(eventMatrix);
  int Nevents = eventMatrix.size(); // Number of events to calibrate with
  int Nchannels = eventMatrix[0].size(); // Number of channel coefficients

  // Iterate the correction
  for (int iter=1;iter<=nIter;iter++) 
    {
      // make one iteration
      solution = iterate(myEventMatrix, energyVector);

      if (solution.empty()) return solution;
      // R.O.: or throw an exception, what's the standard CMS way ?

      // re-calibrate eventMatrix with solution
      for (int i=0; i<Nchannels; i++) 
	{
	  for (int ievent = 0; ievent<Nevents; ievent++)
	    {
	      myEventMatrix[ievent][i] *= solution[i];
	    }
	  // save solution into theCalibVector
	  theCalibVector[i] *= solution[i];
	}

    } // end iterate the correction

  return theCalibVector;
}


vector<float> HouseholderDecomposition::iterate(const vector<vector<float> >& eventMatrix, const vector<float>& energyVector)
{
  // An implementation of the Householder in-situ calibration algorithm 
  // (Least squares minimisation of residual R=b-Ax, with QR decomposition of A)
  // A: matrix of channel response for all calib events
  // x: vector of channel calibration coefficients
  // b: vector of energies
  // adapted from the original code by Matt Probert 9/08/01.

  vector<float> solution; 

  int m=eventMatrix.size();      // Number of events to calibrate with
  int n=eventMatrix[0].size();           // Number of channel coefficients to optimize

  cout << "Householder::runIter(): starting calibration optimization:" << endl;
  cout << "  Events:" << m << ", channels: " << n << endl;

  // Sanity check
  if (m != energyVector.size())
    {
      cout << "Householder::runIter(): matrix dimensions non-conformant. " << endl;
      cout << "  energyVector.size()=" << energyVector.size() << endl;
      cout << "  eventMatrix[0].size()=" << eventMatrix[0].size() << endl;
      cout << " ******************    ERROR   *********************" << endl;
      return solution; // empty vector
    }

  // Reserve workspace
  float e25p;
  int i,j;
  vector<vector<float> > A(eventMatrix);
  vector<float> energies(energyVector);

  float normalisation = 0.;
  
  // Normalise if normaliseFlag is set
  if (normaliseFlag) 
    {
      cout << "Householder::iterate(): Normalising event data" << endl;
      cout << "  WARNING: assuming 5x5 filtering has already been done" << endl;

      for (i=0; i<m; i++) 
	{
	  e25p = 0.;
	  for (j=0;j<n;j++){
	    e25p += eventMatrix[i][j]; // lorenzo -> trying to use ESetup which already performs calibs on rechits
	  }	
	  e25p /= energyVector[i];
	  normalisation += e25p;        // SUM e25p for all events
	}
      normalisation/=m;
      cout << "  Normalisation = " << normalisation << endl;
      
      for (i=0;i<energies.size();++i)
	energies[i]*=normalisation;
    }
  

  // This is where the work goes on...
  // matrix decomposition
  vector<vector<float> > Acopy(A);
  vector<float> alpha(n);
  vector<int> pivot(n);
  if( !decompose(m, n, A, alpha, pivot)) {
    cout << "Householder::runIter(): Failed: Singular condition in decomposition." 
	 << endl;
    cout << "***************** PROBLEM in DECOMPOSITION *************************"<<endl;
    return solution; // empty vector
  }

  /* DBL_EPSILON: Difference between 1.0 and the minimum float greater than 1.0 */
  float etasqr = DBL_EPSILON*DBL_EPSILON; 
  cout<<"LOOK at DBL_EPSILON :"<<DBL_EPSILON<<endl;

  vector<float> r(energies); // copy energies vector
  vector<float> e(n);
 
  // apply transformations to rhs - find solution vector
  solution.assign(n,0.);
  solve(m,n,A,alpha,pivot,r,solution);

  // compute residual vector r
  for (i=0;i<m;i++) {
    r[i]=energies[i];
    for (j=0;j<n;j++)
      r[i]-=Acopy[i][j]*solution[j];
  }
  // compute first correction vector e
  solve(m,n,A,alpha,pivot,r,e);

  float normy0=0.;
  float norme1=0.;
  float norme0;
    
  for (i=0;i<n;i++) {
    normy0+=solution[i]*solution[i];
    norme1+=e[i]*e[i];
  }
  
  cout << "Householder::runIter(): applying first correction" << endl;
  cout << " normy0 = " << normy0 << endl;
  cout << " norme1 = " << norme1 << endl;

  // not attempt at obtaining the solution is made unless the norm of the first
  // correction  is significantly smaller than the norm of the initial solution
  if (norme1>(0.0625*normy0)) {
    cout << "Householder::runIter(): first correction is too large. Failed." 
	 << endl;
  }

  // improve the solution
  for (i=0;i<n;i++)
    solution[i]+=e[i];

  cout << "Householder::runIter(): improving solution...." << endl;

  // only continue iteration if the correction was significant
  while (norme1>(etasqr*normy0)) {
    cout << "Householder::runIter(): norme1 = " << norme1 << endl;
    
    for (i=0;i<m;i++) {
      r[i] = energies[i];
      for (j=0;j<n;j++)
	r[i]-=Acopy[i][j]*solution[j];
    }

    // compute next correction vector
    solve(m,n,A,alpha,pivot,r,e);

    norme0=norme1;
    norme1=0.;
    for (i=0;i<n;i++)
      norme1+=e[i]*e[i];

    // terminate iteration if the norm of the new correction failed to decrease
    // significantly compared to the norm of the previous correction
    if (norme1>(0.0625*norme0))
      break;

    // apply correction vector
    for (i=0;i<n;i++)
      solution[i]+=e[i];
  }

return solution;
}


bool HouseholderDecomposition::decompose(const int m, const int n, vector<vector<float> >& qr,  vector<float>& alpha, vector<int>& pivot)
{
  int i,j,jbar,k;
  float beta,sigma,alphak,qrkk;
  vector<float> y(n);
  vector<float> sum(n);

  cout << "Householder::decompose() started" << endl;
  
  for (j=0;j<n;j++) {
    // jth column sum
    
    sum[j]=0.;
    for (i=0;i<m;i++)
//      cout << "0: qr[i][j]" << qr[i][j] << " i = " << i << " j = " << j << endl;
      sum[j]+=qr[i][j]*qr[i][j];

    pivot[j] = j;
  }
  
  for (k=0;k<n;k++) {
    // kth Householder transformation
    
    sigma = sum[k];
    jbar = k;
    
    for (j=k+1;j<n;j++) {
      if (sigma < sum[j]) {
	sigma = sum[j];
	jbar=j;
      }
    }

    if (jbar != k) {
      // column interchange
      i = pivot[k];
      pivot[k]=pivot[jbar];
      pivot[jbar]=i;
      sum[jbar]=sum[k];
      sum[k]=sigma;

      for (i=0;i<m;i++) {
	sigma=qr[i][k];
	qr[i][k]=qr[i][jbar];
//      cout << "A: qr[i][k]" << qr[i][k] << " i = " << i << " k = " << k << endl;
	qr[i][jbar]=sigma;
//      cout << "B: qr[i][jbar]" << qr[i][k] << " i = " << i << " jbar = " << jbar << endl;
      }
    } // end column interchange

    sigma=0.;
    for (i=k;i<m;i++){
      sigma+=qr[i][k]*qr[i][k];
//      cout << "C: qr[i][k]" << qr[i][k] << " i = " << i << " k = " << k << endl;
}

    if (sigma == 0.) {
      cout << "Householder::decompose() failed" << endl;
      return false;
    }

    qrkk = qr[k][k];

    if (qrkk < 0.) 
      alpha[k] = sqrt(sigma);
    else
      alpha[k] = sqrt(sigma) * (-1.);
    alphak = alpha[k];

    beta = 1/(sigma-qrkk*alphak);
    qr[k][k]=qrkk-alphak;

    for (j=k+1;j<n;j++) {
      y[j]=0.;
      for (i=k;i<m;i++)
	y[j]+=qr[i][k]*qr[i][j];
      y[j]*=beta;
    }

    for (j=k+1;j<n;j++) {
      
      for (i=k;i<m;i++) {
	qr[i][j]-=qr[i][k]*y[j];
	sum[j]-=qr[k][j]*qr[k][j];
      }
    }
  } // end of kth householder transformation

  cout << "Householder::decompose() finished" << endl;
  
  return true;
}


void HouseholderDecomposition::solve(int m, int n, const vector<vector<float> > &qr, const vector<float> &alpha, const vector<int> &pivot, 
				     vector<float> &r, vector<float> &y)
{
  vector<float> z(n,0.);

  float gamma;
  int i,j;

  cout << "Householder::solve() begin" << endl;

  for (j=0;j<n;j++) {
    // apply jth transformation to the right hand side
    gamma=0.;
    for (i=j;i<m;i++)
      gamma+=qr[i][j]*r[i];
    gamma/=(alpha[j]*qr[j][j]);

    for (i=j;i<m;i++)
      r[i]+=gamma*qr[i][j];
  }

  //  cout<<"OK1:"<<endl;
  z[n-1]=r[n-1]/alpha[n-1];
  //  cout<<"OK2:"<<endl;  

  for (i=n-2;i>=0;i--) {
    z[i]= r[i];
    for (j=i+1;j<n;j++)
      z[i]-=qr[i][j]*z[j];
    z[i]/=alpha[i];
  }
  //  cout<<"OK3:"<<endl;

  for (i=0;i<n;i++)
    y[pivot[i]]=z[i];

  cout << "Householder::solve() finished." << endl;

}
