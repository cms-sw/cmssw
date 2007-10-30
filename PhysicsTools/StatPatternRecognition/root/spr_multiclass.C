// $Id: spr_multiclass.C,v 1.5 2007/10/30 00:15:35 narsky Exp $
//
// Load spr_plot.C before running spr_tutorial():
//
// .L spr_plot.C
// .L spr_multiclass.C
// spr_multiclass()
//
// Sometimes it is necessary to load libTree:
// .L $ROOTSYS/lib/libTree.so

#include <iostream>
#include <cassert>

using namespace std;


int spr_multiclass()
{
  // load lib
  gSystem->Load("/afs/cern.ch/user/n/narsky/w0/CMSSW_1_8_X_2007-10-28-1600/lib/slc4_ia32_gcc345/libPhysicsToolsStatPatternRecognition.so");
  
  // create main SPR object
  SprRootAdapter spr;

  // load training data
  spr.loadDataFromAscii(1,"cmc.data","train");

  // split data into train/test as 0.7/0.3
  spr.split(0.7);

  // choose classes
  spr.chooseClasses("1,2,3");
  
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
  spr.correlation(0,corr,"train");
  plotCorrelation("SPR_C0","class 1",dim,vars,corr);
  spr.correlation(1,corr,"train");
  plotCorrelation("SPR_C1","class 2",dim,vars,corr);
  spr.correlation(2,corr,"train");
  plotCorrelation("SPR_C2","class 3",dim,vars,corr);

  // Book a multi-class learner using boosted decision trees.
  SprAbsClassifier* bdt
    = spr.addBoostedDecisionTree("BDT_for_MC",100,100,0);
  assert( bdt != 0 );
  const int nMC = 3;
  int mcClasses [nMC] = { 1, 2, 3 };
  spr.setMultiClassLearner(bdt,nMC,mcClasses,"One-vs-All");

  // train
  spr.train();

  // test
  spr.test();

  // get classification table
  double classificationTable [nMC*nMC];
  spr.multiClassTable(nMC,mcClasses,classificationTable);
  plotMultiClassTable("MCTable1","One-vs-All BDT Test Data",
		      nClasses,classes,classificationTable);

  // retrain multi-class learner in one-vs-one mode
  spr.removeClassifier("MultiClassLearner");
  spr.setMultiClassLearner(bdt,nMC,mcClasses,"One-vs-One");
  spr.train();
  spr.test();
  spr.multiClassTable(nMC,mcClasses,classificationTable);
  plotMultiClassTable("MCTable2","One-vs-One BDT Test Data",
		      nClasses,classes,classificationTable);

  // save the multi-class learner based on BDT
  // and train another one using random forest
  spr.saveClassifier("MultiClassLearner","mcl_bdt.spr");
  spr.removeClassifier("MultiClassLearner");
  SprAbsClassifier* rf = spr.addRandomForest("RF_for_MC",5,50,0,0);
  spr.setMultiClassLearner(rf,nMC,mcClasses,"One-vs-All");
  spr.train();
  spr.test();
  spr.multiClassTable(nMC,mcClasses,classificationTable);
  plotMultiClassTable("MCTable3","One-vs-All RandomForest Test Data",
		      nClasses,classes,classificationTable);

  // get rid of random forest and load back BDT-based learner
  spr.removeClassifier("MultiClassLearner");
  spr.loadClassifier("MultiClassLearner","mcl_bdt.spr");
  spr.test();
  spr.multiClassTable(nMC,mcClasses,classificationTable);
  plotMultiClassTable("MCTable4","One-vs-One BDT Reload Test Data",
		      nClasses,classes,classificationTable);
  
  // estimate importance of all variables
  // use 3 permutations per variable
  const unsigned nVars = spr.nClassifierVars("MultiClassLearner");
  char vars[nVars][200];
  double importance [nVars];
  double impError [nVars];
  spr.variableImportance("MultiClassLearner",3,vars,importance,impError);
  plotImportance("MC_Var_Imp","Variable Importance for MultiClassLearner",
		 nVars,vars,importance,impError,true);

  // save test data with classifier response into a root file
  spr.saveTestData("mytest.root");

  // exit
  return 0;
}
