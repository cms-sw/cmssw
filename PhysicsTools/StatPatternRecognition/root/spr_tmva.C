// $Id: spr_tmva.C,v 1.2 2007/11/12 06:19:16 narsky Exp $
//
// Load spr_plot.C before running spr_tmva():
//
// .L spr_plot.C
// .L spr_tmva.C
// spr_tmva()
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


int spr_tmva()
{
  // load lib
  gSystem->Load("/afs/cern.ch/user/n/narsky/w0/CMSSW_1_8_X_2007-11-29-1600/lib/slc4_ia32_gcc345/libPhysicsToolsStatPatternRecognition.so");
  
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

  // select classifiers
  spr.addFisher("fisher",1);
  spr.addLogitR("logitR",0.001,1.);
  spr.addTopdownTree("tree","Gini",100,0,false);// continuous tree
  spr.addBoostedBinarySplits("splits",100,20);
  spr.addBoostedDecisionTree("bdt",1000,300,5);
  spr.addRandomForest("rf",5,100,2,5);
  spr.addRandomForest("arcx4",5,100,2,5,true);
  spr.addStdBackprop("abackprop","4:4:1",100,0.1,0.1,1,5);

  //
  // Add a boosting sequence: decision tree + neural net
  //
  // Do not display validation loss for the NN - will be displayed by AdaBoost.
  SprAbsClassifier* nn 
    = spr.addStdBackprop("nn_for_booster","4:4:1",10,0.1,0.5,100,0);
  SprAbsClassifier* tree 
    = spr.addTopdownTree("tree_for_booster","Gini",2000,0,false);
  // array of classifiers for AdaBoost
  SprAbsClassifier* for_booster [2] = { tree, nn };
  // do not use pre-defined cuts on the classifier outputs -> AdaBoost will
  //   find the optimal cuts for each sub-classifier
  bool useCut [2] = { false, false };
  double cut [2];
  // make Discrete AdaBoost for boosting NN and DT in turns
  spr.addAdaBoost("boosted_tree_and_nn",
		  2,for_booster,useCut,cut,300,1,false,0.1,5);

  // train
  int verbose = 0;// use >0 to increase verbosity level
  spr.train(verbose);
  
  // remove classifiers that we don't want to have displayed
  spr.removeClassifier("nn_for_booster");
  spr.removeClassifier("tree_for_booster");

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
  
  // zoom in
  double signalEff2 [npts]  = { 0.01, 0.02, 0.03, 0.04, 0.05, 
				0.06, 0.07, 0.08, 0.09 };
  spr.allEffCurves(npts,signalEff2,classifiers,bgrndEff,bgrndErr,fom);
  plotEffCurveMulti("SPR_4",ntrained,npts,signalEff2,
  		    classifiers,bgrndEff,bgrndErr,0);

  // save test data with classifier response into a root file
  spr.saveTestData("mytest.root");

  // exit
  return 0;
}
