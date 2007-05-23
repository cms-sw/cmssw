//$Id: exampleCorrTest.cc,v 1.2 2006/10/19 21:27:54 narsky Exp $
/*
  Tests zero correlation for two bivariate Gaussian densities:

  1) with covariance matrix (1,1)=(2,2)=1  (1,2)=(2,1)=0.5
  2) with covariance matrix (1,1)=(2,2)=1  (1,2)=(2,1)=0

  Obviously, in the first case correlation is inconsistent and the 2nd case
  consistent with zero.
*/

#include "PhysicsTools/StatPatternRecognition/interface/SprExperiment.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprData.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprEmptyFilter.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprSimpleReader.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprDataMoments.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprClass.hh"

#include <iostream>
#include <string>
#include <vector>

using namespace std;


int main(int argc, char ** argv)
{
  // init
  string trFile1 = "gausscorr_uniform_2d_train.pat";
  string trFile2 = "gauss_uniform_2d_train.pat";

  // read training data from file
  SprSimpleReader reader(1);
  auto_ptr<SprAbsFilter> filter1(reader.read(trFile1.c_str()));
  if( filter1.get() == 0 ) {
    cerr << "Unable to read data from file " << trFile1.c_str() << endl;
    return 1;
  }
  cout << "Total number of points read: " << filter1->size() << endl;
  cout << "Points in class 0: " << filter1->ptsInClass(0)
       << " 1: " << filter1->ptsInClass(1) << endl;
  auto_ptr<SprAbsFilter> filter2(reader.read(trFile2.c_str()));
  if( filter2.get() == 0 ) {
    cerr << "Unable to read data from file " << trFile2.c_str() << endl;
    return 1;
  }
  cout << "Total number of points read: " << filter2->size() << endl;
  cout << "Points in class 0: " << filter2->ptsInClass(0)
       << " 1: " << filter2->ptsInClass(1) << endl;

  // make filter
  filter1->chooseClasses(vector<SprClass>(1,SprClass(1)));// choose signal=1
  if( !filter1->filter() ) {
    cerr << "Cannot filter." << endl;
    return 2;
  }

  // make filter
  filter2->chooseClasses(vector<SprClass>(1,SprClass(1)));// choose signal=1
  if( !filter2->filter() ) {
    cerr << "Cannot filter." << endl;
    return 2;
  }

  // test correlation
  SprDataMoments moms1(filter1.get());
  double mean1(0), mean2(0), var1(0), var2(0);
  double corr01 = moms1.correl(0,1,mean1,mean2,var1,var2);
  double cl0 = moms1.zeroCorrCL(0,1);
  cout << endl;
  cout << "Results from correlated data:" << endl;
  cout << "Correlation = " << corr01 << endl;
  cout << "Consistent with 0 at " << cl0 << " level." << endl;
  //
  SprDataMoments moms2(filter2.get());
  corr01 = moms2.correl(0,1,mean1,mean2,var1,var2);
  cl0 = moms2.zeroCorrCL(0,1);
  cout << endl;
  cout << "Results from uncorrelated data:" << endl;
  cout << "Correlation = " << corr01 << endl;
  cout << "Consistent with 0 at " << cl0 << " level." << endl;

  // exit
  return 0;
}
