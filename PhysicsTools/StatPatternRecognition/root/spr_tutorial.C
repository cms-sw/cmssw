// $Id: spr_tutorial.C,v 1.6 2007/11/30 20:13:30 narsky Exp $
//
// Load spr_plot.C before running spr_tutorial():
//
// .L spr_plot.C
// .L spr_tutorial.C
// spr_tutorial()
//
// Sometimes it is necessary to load libTree:
// .L $ROOTSYS/lib/libTree.so

#include <iostream>

using namespace std;


int spr_tutorial()
{
  // load lib
  gSystem->Load("/afs/cern.ch/user/n/narsky/w0/CMSSW_1_8_X_2007-11-29-1600/lib/slc4_ia32_gcc345/libPhysicsToolsStatPatternRecognition.so");
  
  // create main SPR object
  SprRootAdapter spr;

  // exclude variable x1
  char excludeVars[1][200];
  strcpy(excludeVars[0],"x1");
  spr.chooseAllBut(1,excludeVars);

  // load training data
  spr.loadDataFromAscii(2,"lambda-train.pat","train");

  // split data into train/test as 0.7/0.3
  spr.split(0.7,false);

  // No, wait - I have test data in a separate file. Let us reload.
  // Also, I want to use variable x1 too now.
  spr.chooseAllVars();
  spr.loadDataFromAscii(2,"lambda-train.pat","train");
  spr.loadDataFromAscii(2,"lambda-test.pat","test");

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
  spr.addDecisionTree("tree1","Gini",100);// discrete tree
  spr.addTopdownTree("tree2","Gini",100,0,false);// continuous tree
  spr.addBumpHunter("bump","Gini",2000,0.1);
  spr.addBoostedBinarySplits("splits",20,5);
  spr.addBoostedDecisionTree("bdt_1",500,50,5);
  spr.addBoostedDecisionTree("bdt_2",1000,50,5);
  spr.addRandomForest("rf",5,50,6,5);
  spr.addRandomForest("arcx4",5,50,6,5,true);
  spr.addStdBackprop("backprop","6:6:4:1",100,0.05,0.1,100,5);

  // train
  int verbose = 0;// use >0 to increase verbosity level
  spr.train(verbose);

  // test
  spr.test();
  
  // get signal-vs-background curve for the continuous tree
  const int npts = 9;
  double signalEff [npts] = { 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 };
  double bgrndEffTree [npts];
  double bgrndErrTree [npts];
  double fomTree [npts];
  spr.effCurve("tree2",npts,signalEff,bgrndEffTree,bgrndErrTree,fomTree);
  plotEffCurve("SPR_2",npts,signalEff,"tree2",bgrndEffTree,bgrndErrTree,3);

  // ah, what the heck, let's get curves for all trained classifiers
  const int ntrained = spr.nTrained();
  double bgrndEff [npts*ntrained];
  double bgrndErr [npts*ntrained];
  double fom [npts*ntrained];
  char classifiers[ntrained][200];
  spr.allEffCurves(npts,signalEff,classifiers,bgrndEff,bgrndErr,fom);
  plotEffCurveMulti("SPR_3",ntrained,npts,signalEff,
		    classifiers,bgrndEff,bgrndErr,0);
  
  // Compare just random forest, neural net and fisher
  spr.removeClassifier("bdt_1");
  spr.removeClassifier("bdt_2");
  spr.removeClassifier("logitR");
  spr.removeClassifier("tree1");
  spr.removeClassifier("tree2");
  spr.removeClassifier("bump");
  spr.removeClassifier("splits");
  const int ntrained2 = spr.nTrained();
  double bgrndEff2 [npts*ntrained2];
  double bgrndErr2 [npts*ntrained2];
  double fom2 [npts*ntrained2];
  char classifiers2[ntrained2][200];
  spr.allEffCurves(npts,signalEff,classifiers2,bgrndEff2,bgrndErr2,fom2);
  plotEffCurveMulti("SPR_4",ntrained2,npts,signalEff,
		    classifiers2,bgrndEff2,bgrndErr2,0);

  // save the random forest into a file
  spr.saveClassifier("rf","lambda_rf.spr");

  // clean up classifier list
  spr.clearClassifiers();

  // read back the saved classifier from file
  spr.loadClassifier("rf","lambda_rf.spr");

  // recompute classifier responses on test data
  spr.test();

  // plot Gini index
  // Must use absolute weights to plot FOM.
  double signalEffRF [npts] = { 300, 600, 900, 1200, 1500, 1800, 2100, 2400, 2700};
  double bgrndEffRF [npts];
  double bgrndErrRF [npts];
  double fomRF [npts];
  spr.setCrit("Gini");
  spr.setEffCurveMode("absolute");
  spr.effCurve("rf",npts,signalEffRF,bgrndEffRF,bgrndErrRF,fomRF);
  plotFOMCurve("SPR_5","Gini",npts,signalEffRF,"rf",fomRF,5);

  // histogram output of random forest for signal and background
  const int nbin = 20;
  double xlo = 0.;
  double xhi = 1.;
  double sig[nbin], sigerr[nbin], bgr[nbin], bgrerr[nbin];
  spr.histogram("rf",xlo,xhi,nbin,sig,sigerr,bgr,bgrerr);
  plotHistogram("SPR_6","log","RandomForest output",
		xlo,xhi,nbin,sig,sigerr,bgr,bgrerr);

  // estimate importance of all variables
  // use 3 permutations per variable
  const unsigned nVars = spr.nClassifierVars("rf");
  char vars[nVars][200];
  double importance [nVars];
  double impError [nVars];
  spr.variableImportance("rf",3,vars,importance,impError);
  plotImportance("SPR_7","Variable Importance for RandomForest",
		 nVars,vars,importance,impError,true);

  // save test data with classifier response into a root file
  spr.saveTestData("mytest.root");

  // exit
  return 0;
}
