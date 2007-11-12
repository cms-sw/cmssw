// $Id: spr_transform.C,v 1.2 2007/11/12 04:41:16 narsky Exp $
//
// Load spr_plot.C before running spr_transform():
//
// .L spr_plot.C
// .L spr_transform.C
// spr_transform_1()
// spr_transform_2()
//
// Sometimes it is necessary to load libTree:
// .L $ROOTSYS/lib/libTree.so
/*
  The root training file for this example can be downloaded from
  http://root.cern.ch/files/tmva_example.root
  Signal and background distributions in this file are 4D Gaussians.
  That's why the optimal separation is linear and given by the Fisher
  discriminant (or logistic regression).
  Given that, I treated this as an exercise for illustration 
  and did not try to select classifier parameters carefully.
*/

#include <iostream>

using namespace std;


int spr_transform_1()
{
  // load lib
  gSystem->Load("/afs/cern.ch/user/n/narsky/w0/CMSSW_1_8_X_2007-11-09-0200/lib/slc4_ia32_gcc345/libPhysicsToolsStatPatternRecognition.so");
  
  // create main SPR object
  SprRootAdapter spr;

  // load training data
  spr.loadDataFromRoot("tmva_root.pat","train");

  // split data into train/test
  spr.split(0.5,true);
  
  // choose classes
  spr.chooseClasses("0:1");
  
  // show how much data we have
  const int nClasses = spr.nClasses();
  char classes[nClasses][200];
  int events [nClasses];
  double weights [nClasses];
  spr.showDataInClasses(classes,events,weights,"train");
  plotClasses("SPR_Class_1","train",nClasses,classes,events,weights);
  spr.showDataInClasses(classes,events,weights,"test");
  plotClasses("SPR_Class_2","test",nClasses,classes,events,weights);

  // compute correlations between variables
  const unsigned dim = spr.dim();
  double corr [dim*dim];
  char vars[dim][200];
  spr.vars(vars);
  spr.correlation(0,corr,"train");// background
  plotCorrelation("SPR_1B","background",dim,vars,corr);
  spr.correlation(1,corr,"train");// signal
  plotCorrelation("SPR_1S","signal",dim,vars,corr);

  // save original test data
  spr.saveTestData("test_orig.root");

  // perform PCA
  int verbose = 0;
  spr.trainVarTransformer("PCA",verbose);
  spr.transform();

  // Plot correlations after PCA transform.
  // Note that PCA transform does not change dimensionality.
  // But it changes variable names!
  spr.vars(vars);
  spr.correlation(0,corr,"train");// background
  plotCorrelation("SPR_2B","decorrelated background",dim,vars,corr);
  spr.correlation(1,corr,"train");// signal
  plotCorrelation("SPR_2S","decorrelated signal",dim,vars,corr);

  // save transformer to a file for future reference
  spr.saveVarTransformer("pca.spr");

  // select classifiers
  spr.addBoostedBinarySplits("splits",100,20);

  // train
  spr.train(verbose);
  
  // test
  spr.test();
  
  // get signal-vs-background curve for classifiers
  const int npts = 9;
  double signalEff [npts] = { 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 };
  const int ntrained = spr.nTrained();
  double bgrndEff [npts*ntrained];
  double bgrndErr [npts*ntrained];
  double fom [npts*ntrained];
  char classifiers[ntrained][200];
  spr.allEffCurves(npts,signalEff,classifiers,bgrndEff,bgrndErr,fom);
  plotEffCurveMulti("SPR_3",ntrained,npts,signalEff,
		    classifiers,bgrndEff,bgrndErr,0);
  
  // save transformed test data with classifier responses
  spr.saveTestData("test_transformed.root");

  // exit
  return 0;
}


/*
  The following shows how to apply a PCA transformation to a subset 
  of input variables.

  You need to prepare tmva_subset_root.pat. Copy tmva_root.pat and
  remove var2 on the Leaves: line. This will force the reader to read
  in only 3 variables: var1, var3, and var4.

  After you plot the results, you can see that only variables 
  var1, var3 and var4 have been decorrelated. Variable var2 shows
  substantial correlation with others.
*/
int spr_transform_2()
{
  // load lib
  gSystem->Load("/afs/cern.ch/user/n/narsky/w0/CMSSW_1_8_X_2007-11-09-0200/lib/slc4_ia32_gcc345/libPhysicsToolsStatPatternRecognition.so");
  
  // create main SPR object
  SprRootAdapter spr;

  // load training data
  spr.loadDataFromRoot("tmva_subset_root.pat","train");

  // split data into train/test
  spr.split(0.5,true);
  
  // choose classes
  spr.chooseClasses("0:1");
  
  // show how much data we have
  const int nClasses = spr.nClasses();
  char classes[nClasses][200];
  int events [nClasses];
  double weights [nClasses];
  spr.showDataInClasses(classes,events,weights,"train");
  plotClasses("SPR_Class_1","train",nClasses,classes,events,weights);
  spr.showDataInClasses(classes,events,weights,"test");
  plotClasses("SPR_Class_2","test",nClasses,classes,events,weights);

  // perform PCA
  int verbose = 0;
  spr.trainVarTransformer("PCA",verbose);
  spr.saveVarTransformer("pca.spr");

  // reload training and test data, now with all variables included
  spr.loadDataFromRoot("tmva_root.pat","train");
  spr.split(0.5,true);
  spr.chooseClasses("0:1");

  // compute correlations between variables before PCA transform
  const unsigned dim = spr.dim();
  double corr [dim*dim];
  char vars[dim][200];
  spr.vars(vars);
  spr.correlation(0,corr,"train");// background
  plotCorrelation("SPR_1B","background",dim,vars,corr);
  spr.correlation(1,corr,"train");// signal
  plotCorrelation("SPR_1S","signal",dim,vars,corr);

  // apply PCA transform
  spr.transform();

  // Plot correlations after PCA transform.
  spr.vars(vars);
  spr.correlation(0,corr,"train");// background
  plotCorrelation("SPR_2B","decorrelated background",dim,vars,corr);
  spr.correlation(1,corr,"train");// signal
  plotCorrelation("SPR_2S","decorrelated signal",dim,vars,corr);

  // exit
  return 0;
}
