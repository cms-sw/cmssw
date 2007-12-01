// $Id: spr_var_selection.C,v 1.1 2007/11/30 20:13:35 narsky Exp $
//
// .L spr_plot.C
// .L spr_var_selection.C
// spr_var_selection()
//
// Sometimes it is necessary to load libTree:
// .L $ROOTSYS/lib/libTree.so
/*
  This example uses Cleveland heart-disease data available from UCI repository:
  http://mlearn.ics.uci.edu/databases/heart-disease/heart-disease.names
  Class is coded as an integer ranging from 0 (no disease) to 4 (various
  types of disease). The idea is merely to separate healthy people from
  non-healthy.
*/

#include <iostream>
#include <string>

using namespace std;


int spr_var_selection()
{
  // load lib
  gSystem->Load("/afs/cern.ch/user/n/narsky/w0/CMSSW_1_8_X_2007-11-29-1600/lib/slc4_ia32_gcc345/libPhysicsToolsStatPatternRecognition.so");
  
  // create main SPR object
  SprRootAdapter spr;
  
  // load training data
  spr.loadDataFromAscii(1,"cleveland.data","train");
  
  // split data into train/test
  spr.split(0.7,true,2627277);
  
  // choose classes
  spr.chooseClasses("0:.");
  
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

  // compute correlation with the class label
  double corr [dim];
  spr.correlationClassLabel("normal",vars,corr,"test");
  plotImportance("SPR_0","Correlation with class label",
		 dim,vars,corr,0,true);

  // select classifiers
  spr.addRandomForest("rf",2,200,0,10);

  // train
  int verbose = 0;// use >0 to increase verbosity level
  spr.train(verbose);
  
  // save classifier
  spr.saveClassifier("rf","rf.spr");

  // test
  spr.test();
  
  // estimate importance of all variables
  const unsigned nVars = spr.nClassifierVars("rf");
  double importance [nVars];
  double impError [nVars];
  spr.variableImportance("rf",10,vars,importance,impError);
  plotImportance("SPR_1","Variable Importance for RF",
		 nVars,vars,importance,impError,true);

  // estimate interactions
  // Note: the errors are zero because all points are used for
  // integration!
  const char* subset = "";
  unsigned nPoints = 0;// use all points
  double interaction [nVars];
  double intError [nVars];
  spr.variableInteraction("rf",subset,nPoints,
			  vars,interaction,intError); 
  plotImportance("SPR_3","Variable Interaction for RF",
		 nVars,vars,interaction,intError,true);

  // choose a subset of variables
  char useVars[5][200];
  strcpy(useVars[0],"sex");
  strcpy(useVars[1],"cp");
  strcpy(useVars[2],"oldpeak");
  strcpy(useVars[3],"ca");
  strcpy(useVars[4],"thal");
  spr.chooseVars(5,useVars);
  spr.loadDataFromAscii(1,"cleveland.data","train");

  // use identical splitting
  spr.split(0.7,true,2627277);
  spr.chooseClasses("0:.");

  // train RF on the reduced subset
  spr.addRandomForest("rf_reduced",2,200,0,10);
  spr.train(verbose);

  // save
  spr.saveClassifier("rf_reduced","rf_reduced.spr");

  // compute classifier responses
  spr.test();
  
  // recompute variable importance
  const unsigned nVarsReduced = spr.nClassifierVars("rf_reduced");
  char varsReduced [nVarsReduced][200];
  double importanceReduced [nVarsReduced];
  double impErrorReduced [nVarsReduced];
  spr.variableImportance("rf_reduced",10,varsReduced,
			 importanceReduced,impErrorReduced);
  plotImportance("SPR_2","Variable Importance for RF Reduced",
		 nVarsReduced,varsReduced,
		 importanceReduced,impErrorReduced,true);

  // reload data
  spr.chooseAllVars();
  spr.loadDataFromAscii(1,"cleveland.data","train");
  spr.split(0.7,true,2627277);
  spr.chooseClasses("0:.");

  // reload save classifiers
  spr.loadClassifier("rf","rf.spr");
  spr.loadClassifier("rf_reduced","rf_reduced.spr");

  // rerun test
  spr.test();

  // get the signal-vs-bgrnd curve
  const int ntrained = spr.nTrained();
  const int npts = 9;
  double signalEff [npts] = { 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 };
  double bgrndEff [ntrained*npts];
  double bgrndErr [ntrained*npts];
  double fom [ntrained*npts];
  char classifiers[ntrained][200];
  spr.allEffCurves(npts,signalEff,classifiers,bgrndEff,bgrndErr,fom);
  plotEffCurveMulti("SPR_RF",ntrained,npts,signalEff,
		    classifiers,bgrndEff,bgrndErr,0);

  // exit
  return 0;
}
