//$Id: exampleBootstrapVarianceEst.cc,v 1.4 2008/11/26 22:59:20 elmer Exp $
/*
  This example estimates RMS of the correlation coefficient estimator obtained 
  for 20 points drawn from a bivariate Gaussian density with covariance matrix 
  (1,1)=(2,2)=1; (1,2)=(2,1)=0.5. Histogram "corr" shows the true
  distribution of the correlation estimator. It is obtained by drawing 
  500 samples with 20 points per sample and computing the correlation estimate
  for each sample. Histogram "bCorrRMS" shows a bootstrap estimate of the RMS
  of the correlation estimator. The bootstrap estimate is obtained for
  each 20-point sample by resampling it 100 times. You can see that the 
  RMS of the true distribution of the correlation estimator is about 0.18,
  while the mean of the bootstrap distribution is about 0.16. This demonstrates
  that bootstrap gives a good estimate of the RMS of the correlation estimator.

  The whole point of this exercise is to show that bootstrap can be used as
  a reliable estimator of variance on a small sample when the true 
  underlying distribution of the estimator is unknown. Imagine that you
  have 20 points that come from an unknown distribution. You computed an 
  estimator of the correlation coefficient for these 20 points - how do you
  estimate the RMS of this estimator now without bootstrap?
*/

#include "PhysicsTools/StatPatternRecognition/interface/SprExperiment.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprBootstrap.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprDataMoments.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprEmptyFilter.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprSimpleReader.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAsciiWriter.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprClass.hh"

#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <memory>

using namespace std;


int main(int argc, char ** argv)
{
  // init
  string hbkFile = "bootstrap.out";

  // read training data from file
  string trFile = "gausscorr_uniform_2d_train.pat";
  SprSimpleReader reader(1);
  auto_ptr<SprAbsFilter> filter(reader.read(trFile.c_str()));
  if( filter.get() == 0 ) {
    cerr << "Unable to read data from file " << trFile.c_str() << endl;
    return 1;
  }
  vector<string> vars;
  filter->vars(vars);
  cout << "Read data from file " << trFile.c_str() 
       << " for variables";
  for( unsigned int i=0;i<vars.size();i++ ) 
    cout << " \"" << vars[i].c_str() << "\"";
  cout << endl;
  cout << "Total number of points read: " << filter->size() << endl;
  cout << "Points in class 0: " << filter->ptsInClass(0)
       << " 1: " << filter->ptsInClass(1) << endl;

  // set up filter
  SprEmptyFilter f1(filter.get());
  vector<SprClass> classes(1,SprClass(1));// choose class 1
  f1.chooseClasses(classes);
  if( !f1.filter() ) {
    cerr << "Unable to filter." << endl;
    return 1;
  }
  SprEmptyFilter f2(f1);// prepare consecutive filter
  cout << "Data filtered." << endl;
  cout << f2.size() << " events survived the filter." << endl;
  cout << "Points in class 0: " << f2.ptsInClass(0)
       << " 1: " << f2.ptsInClass(1) << endl;

  // set up an hbook writer
  auto_ptr<SprAbsWriter> hbk(new SprAsciiWriter("bootstrap"));
  if( !hbk->init(hbkFile.c_str()) ) {
    cerr << "Cannot open hbook file " << hbkFile.c_str() << endl;
    return 2;
  }
  vector<string> axes;
  axes.push_back("corr");
  axes.push_back("bCorrRMS");
  hbk->setAxes(axes);
  cout << "Axes set." << endl;

  // prepare boostrap, moments, replica data etc.
  SprBootstrap b(&f2);
  SprDataMoments moms(&f2);

  // split data into chunks of n points
  int n = 20;
  int nrep = 100;// number of Bootstrap replicas for each sample
  unsigned int imin(0), imax(0);
  vector<double> v(2), emp;
  while( imin < f1.size() ) {
    imax = imin + n;
    f2.setIndexRange(imin,imax);
    if( !f2.filter() ) {
      cerr << "Cannot filter at " << imin << endl;
      return 3;
    }
    double corr0 = moms.correl(0,1);
    // make boostrap replicas
    double meanb = 0;
    vector<double> bv(nrep);
    for( int j=0;j<nrep;j++ ) {
      const SprEmptyFilter* drep = b.weightedReplica();
      if( drep == 0 ) {
	cerr << "Unable to generate Bootstrap replica." << endl;
	return 4;
      }
      SprEmptyFilter frep(drep);
      SprDataMoments mrep(&frep);
      bv[j] = mrep.correl(0,1);
      meanb += bv[j];
      delete drep;
    }
    meanb /= nrep;
    double varb = 0;
    for( int j=0;j<nrep;j++ )
      varb += pow(bv[j]-meanb,2);
    varb /= (nrep-1);
    v[0] = corr0;
    v[1] = sqrt(varb);
    if( !hbk->write(1.,v,emp) ) {
      cerr << "Unable to write into hbook." << endl;
      return 5;
    }
    // step to next subsample
    imin += n;
  }

  // close writer
  if( !hbk->close() ) {
    cerr << "Unable to close hbook." << endl;
    return 6;
  }
  cout << "output successfully closed." << endl;

  // exit
  return 0;
}
