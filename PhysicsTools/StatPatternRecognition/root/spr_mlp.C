// $Id: spr_mlp.C,v 1.1 2007/10/30 18:56:12 narsky Exp $
//
// Load spr_plot.C before running spr_tutorial():
//
// .L spr_plot.C
// .L spr_mlp.C
// spr_mlp()
//
// Sometimes it is necessary to load libTree:
// .L $ROOTSYS/lib/libTree.so
/*
  The file for this exercise can be downloaded from
  http://root.cern.ch/files/mlpHiggs.root
*/

#include <iostream>

using namespace std;


int spr_mlp()
{
  // load lib
  gSystem->Load("/afs/cern.ch/user/n/narsky/w0/CMSSW_1_8_X_2007-11-29-1600/lib/slc4_ia32_gcc345/libPhysicsToolsStatPatternRecognition.so");
  
  // create main SPR object
  SprRootAdapter spr;
  
  // load training data
  spr.loadDataFromRoot("mlp_root.pat","train");
  
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
  spr.addStdBackprop("mlp","2:5:3:1",100,0.1,0.5,100,5);
  spr.addRandomForest("rf",100,400,1,20);
  spr.addRandomForest("arcx4",100,400,1,20,true);

  // train
  int verbose = 0;// use >0 to increase verbosity level
  spr.train(verbose);
  
  // test
  spr.test();
  
  // get the signal-vs-bgrnd curve
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

  // histogram output of ArcX4 for signal and background
  const int nbin = 40;
  double xlo = 0.;
  double xhi = 0.7;
  double sig[nbin], sigerr[nbin], bgr[nbin], bgrerr[nbin];
  spr.histogram("arcx4",xlo,xhi,nbin,sig,sigerr,bgr,bgrerr);
  plotHistogram("SPR_6","log","ArcX4 output",
    		xlo,xhi,nbin,sig,sigerr,bgr,bgrerr);

  // save NN into a file
  spr.saveClassifier("mlp","mlp.spr");

  // save test data with classifier response into a root file
  spr.saveTestData("mytest.root");

  // estimate importance of all variables
  // use 3 permutations per variable
  const unsigned nVars = spr.nClassifierVars("arcx4");
  char vars[nVars][200];
  double importance [nVars];
  double impError [nVars];
  spr.variableImportance("arcx4",3,vars,importance,impError);
  plotImportance("SPR_7","Variable Importance for ArcX4",
		 nVars,vars,importance,impError,true);

  // exit
  return 0;
}
