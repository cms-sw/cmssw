//$Id: exampleMatrixTest.cc,v 1.1 2007/11/07 00:56:14 narsky Exp $

#include "PhysicsTools/StatPatternRecognition/interface/SprExperiment.hh"
#include "PhysicsTools/StatPatternRecognition/src/SprMatrix.hh"
#include "PhysicsTools/StatPatternRecognition/src/SprSymMatrix.hh"

#include <iostream>

using namespace std;


int main(int argc, char* argv[])
{
  const int dim = 3;

  SprSymMatrix s(dim);

  s[0][0] = 1; s[0][1] = 2; s[0][2] = 3;
               s[1][1] = 5; s[1][2] = 6;
                            s[2][2] = 2;

  cout << "Supplied matrix S:" << endl;
  for( int i=0;i<dim;i++ ) {
    for( int j=0;j<dim;j++ ) cout << s[i][j] << " ";
    cout << endl;
  }
  cout << endl;

  SprMatrix u = s.diagonalize();
  cout << "Diagonalized matrix S:" << endl;
  for( int i=0;i<dim;i++ ) {
    for( int j=0;j<dim;j++ ) cout << s[i][j] << " ";
    cout << endl;
  }
  cout << endl;

  cout << "Matrix U:" << endl;
  for( int i=0;i<dim;i++ ) {
    for( int j=0;j<dim;j++ ) cout << u[i][j] << " ";
    cout << endl;
  }
  cout << endl;

  cout << "Cross-check: U*S*T(U) must be equal to the original matrix" << endl;
  SprMatrix sold = u*s*u.T();
  for( int i=0;i<dim;i++ ) {
    for( int j=0;j<dim;j++ ) cout << sold[i][j] << " ";
    cout << endl;
  }
  cout << endl;

  return 0;
}
