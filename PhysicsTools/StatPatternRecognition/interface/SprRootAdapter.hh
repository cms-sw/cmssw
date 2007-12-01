// File and Version Information:
//      $Id: SprRootAdapter.hh,v 1.8 2007/11/30 20:13:29 narsky Exp $
//
// Description:
//      Class SprRootAdapter :
//          Provides a wrapper compiled with rootcint
//          for access to SPR objects from an interactive Root session.
//
// Author List:
//      Ilya Narsky                     Original author
//
// Copyright Information:
//      Copyright (C) 2007              California Institute of Technology
//------------------------------------------------------------------------

#ifndef _SprRootAdapter_HH
#define _SprRootAdapter_HH

#include <map>
#include <vector>
#include <string>
#include <set>

class SprAbsFilter;
class SprAbsClassifier;
class SprAbsTrainedClassifier;
class SprPlotter;
class SprMultiClassLearner;
class SprTrainedMultiClassLearner;
class SprMultiClassPlotter;
class SprAbsTwoClassCriterion;
class SprIntegerBootstrap;
class SprAverageLoss;
class SprCoordinateMapper;
class SprAbsVarTransformer;


class SprRootAdapter
{
public:
  ~SprRootAdapter();

  SprRootAdapter();

  // Choose input variables.
  // For the moment, this method only works for input Ascii data.
  // Must be called before datasets are loaded.
  // Choosing input vars for Root data is handled by input configurations 
  // files. See SprRootReader.hh for more info.
  void chooseVars(int nVars, const char vars[][200]);
  void chooseAllBut(int nVars, const char vars[][200]);
  void chooseAllVars();

  // Load dataset.
  // Datatype must be either "train" or "test".
  // If you load a training dataset, all classifiers are cleared
  //   and you will need to train() again.
  // If you load a test dataset, you will have to re-run test() method.
  // Both training and test datasets must be loaded
  //   before training begins.
  // Instead of loading a test dataset, the user
  //   might opt to split the training dataset using split().
  bool loadDataFromAscii(int mode, 
			 const char* filename, 
			 const char* datatype="train");
  bool loadDataFromRoot(const char* filename,
			const char* datatype="train");

  // Split dataset into training and test subsets.
  // All classifiers are cleared.
  bool split(double fractionForTraining, bool randomize, int seed=0);

  // Return dimensionality of training data.
  unsigned dim() const;

  // Return variables from training data.
  // The user must book array of size this->dim().
  bool vars(char vars[][200]) const;

  // Return number of variables used for classifier
  unsigned nClassifierVars(const char* classifierName) const;

  // Return variables used for this classifier
  bool classifierVars(const char* classifierName, char vars[][200]) const;

  // This method must be called after training 
  // and test data have been loaded.
  bool chooseClasses(const char* inputClassString);

  // return number of classes
  int nClasses() const;

  // scale weights of events in class by this number
  // classtype is either "signal" or "background"
  bool scaleWeights(double w, const char* classtype="signal");

  // Show events and weights for the chosen classes.
  // The user must allocate arrays of size this->nClasses().
  bool showDataInClasses(char classes[][200],
			 int* events,
			 double* weights,
			 const char* datatype="train") const;

  //
  // Add classifiers.
  //

  //
  // nValidate, where appropriate, indicates the frequency 
  // of print-outs of the figure of merit evaluated on test data.
  //

  // Fisher
  SprAbsClassifier* addFisher(const char* classifierName, 
			      int mode=1);

  // Logistic Regression
  SprAbsClassifier* addLogitR(const char* classifierName,
                              double eps,// accuracy
                              double updateFactor=1.);

  // bump hunter
  SprAbsClassifier* addBumpHunter(const char* classifierName,
				  const char* criterion,
				  unsigned minEventsPerBump,
				  double peel);

  // Standalone decision tree with rectangular cuts.
  // Do not plug it in AdaBoost or Bagger!! Too slow.
  SprAbsClassifier* addDecisionTree(const char* classifierName,
				    const char* criterion,
				    unsigned leafSize);

  // Decision tree. Use discrete=true for AdaBoost!!!
  SprAbsClassifier* addTopdownTree(const char* classifierName,
				   const char* criterion,
				   unsigned leafSize,
				   unsigned nFeaturesToSample,
				   bool discrete=false);

  // Backprop neural net
  SprAbsClassifier* addStdBackprop(const char* classifierName,
                                   const char* structure,
                                   unsigned ncycles,
                                   double eta,
                                   double initEta,
                                   unsigned nInitPoints,
                                   unsigned nValidate);

  // AdaBoost: 
  // An array of weak classifiers of length nClassifier 
  // is supplied by the user.
  // mode represents AdaBoost mode: 1 = Discrete; 2 = Real; 3 = Epsilon
  SprAbsClassifier* addAdaBoost(const char* classifierName,
				int nClassifier,
				SprAbsClassifier** classifier,
				bool* useCut,
				double* cut,
				unsigned ncycles,
				int mode,
				bool bagInput,
				double epsilon,
				unsigned nValidate);

  // short-cut for boosted decision trees
  SprAbsClassifier* addBoostedDecisionTree(const char* classifierName,
					   int leafSize,
					   unsigned nTrees,
					   unsigned nValidate);

  // short-cut for boosted decision splits
  SprAbsClassifier* addBoostedBinarySplits(const char* classifierName,
					   unsigned nSplitsPerDim,
					   unsigned nValidate);


  // Bagger: 
  // An array of weak classifiers of length nClassifier
  // is supplied by the user.
  SprAbsClassifier* addBagger(const char* classifierName,
			      int nClassifier,
			      SprAbsClassifier** classifier,
			      unsigned ncycles,
			      bool discrete,
			      unsigned nValidate);

  // Short-cut for random forest.
  SprAbsClassifier* addRandomForest(const char* classifierName,
				    int leafSize,
				    unsigned nTrees,
				    unsigned nFeaturesToSample,
				    unsigned nValidate,
				    bool useArcE4=false);

  // Set multi class learner.
  // An array of nClass classes to be included in the multi-class
  // algorithm needs to be supplied by the user.
  SprMultiClassLearner* setMultiClassLearner(SprAbsClassifier* classifier,
					     int nClass,
					     const int* classes,
					     const char* mode="One-vs-All");

  // remove a specific classifier
  void removeClassifier(const char* classifierName);

  // clear classifier selection
  void clearClassifiers();

  // Switch the range of trained classifier 
  // from default [0,1] to (-inft,+infty)
  // and back. Only applies to Fisher, LogitR and AdaBoost.
  void useInftyRange() const;
  void use01Range() const;

  // return number of trained classifiers
  int nTrained() const {
    return trained_.size();
  }

  // save trained classifier into a file
  bool saveClassifier(const char* classifierName,
		      const char* filename) const;

  // load trained classifier from a file
  bool loadClassifier(const char* classifierName,
		      const char* filename);

  // train
  bool train(int verbose=0);

  // compute trained classifier responses for test data
  bool test();

  // Choose variable transformer. Possible choices are:
  /*
    PCA
  */
  bool trainVarTransformer(const char* name, int verbose=0);

  // save trained var transformer into a file
  bool saveVarTransformer(const char* filename) const;

  // load var transformer from a file
  bool loadVarTransformer(const char* filename);

  // transform training and test data using the supplied var transformer
  bool transform();

  // save test data with computed classifier responses into a Root file
  bool saveTestData(const char* filename) const;

  // Choose criterion to be computed in background-vs-signal plots.
  // Possible values are 
  /*
    correct_id
    S/sqrt(S+B)
    S/(S+B)
    TaggerEff
    Gini
    CrossEntropy
    InverseUL90
    BKDiscovery
    Punzi
  */
  // Must run setEffCurveMode("absolute") for this setting to take effect.
  bool setCrit(const char* criterion);

  // Switch between using absolute weights and relative efficiency 
  // for efficiency curves.
  // mode = relative or absolute
  bool setEffCurveMode(const char* mode="relative");

  // Compute background efficiency for given values of signal 
  // efficiency. The user must supply the array signalEff and allocate
  // arrays bgrndEff, bgrndErr and fom of the same size as signalEff.
  // See example in spr_tutorial.C.
  bool effCurve(const char* classifierName,
		int npts, const double* signalEff,
		double* bgrndEff, double* bgrndErr, double* fom) const;

  // The user must allocate arrys char classifiers[npts][200],
  // bgrndEff[ntrained*npts], bgrndErr[ntrained*npts], and 
  // fom[ntrained*npts]
  bool allEffCurves(int npts, const double* signalEff,
		    char classifiers[][200],
		    double* bgrndEff, double* bgrndErr, double* fom) const;

  // Histogram signal and background response values for 
  // a certain classifier. 
  // The user must book arrays of size nbin.
  bool histogram(const char* classifierName,
		 double xlo, double xhi, int nbin,
		 double* sig, double* sigerr,
		 double* bgr, double* bgrerr) const;

  // Compute correlations between all vars in data.
  // for class cls (0 for background, 1 for signal, ect).
  // The user must book an array of size this->dim()*this->dim().
  bool correlation(int cls, double* corr, const char* datatype="train") const;

  // Compute correlation with the class label.
  // mode can be set to "normal" or "abs".
  // In the abs mode, correlations between |X-M(X)| and class label
  // are computed for each variable, where M(X) is the mean of this
  // variable in the sample.
  // The user must book arrays vars and corr of size this->dim().
  bool correlationClassLabel(const char* mode, char vars[][200],
			     double* corr, const char* datatype="train") const;

  // Estimate variable importance for a given classifier.
  // The user must book 3 arrays of at least size nVars given 
  // by the chosen classifier, for the list of variables, 
  // importance and error on importance.
  // nPerm defines number of permutations per each variable.
  // The greater nPerm, the better is the accuracy of the estimate,
  // and the longer it takes to compute.
  bool variableImportance(const char* classifierName,
                          unsigned nPerm,
                          char vars[][200],
                          double* importance,
			  double* error) const;

  // Estimate interaction between the specified subset of variables
  // and all other variables used by the classifier.
  // The user must book 3 arrays of size at least D, where
  // D is the number of variables used by this classifier.
  // The subset can be an empty string '',
  // in which case interactions between each variable and all
  // others will be computed. For details, see SprClassifierEvaluator.hh.
  //
  // nPoints specifies the number of points used for integration of classifier
  // response over each variable. The user should weigh accuracy vs CPU time.
  bool variableInteraction(const char* classifierName,
			   const char* subset,
			   unsigned nPoints,
			   char vars[][200],
			   double* interaction,
			   double* error,
			   int verbose=0) const;

  // Returns classification table for the multi class learner
  // for the selected subset of classes. The user must provide
  // an array of nClass integers representing desired classes.
  // The user must book classificationTable, 
  // an array of doubles of size nClass*nClass.
  bool multiClassTable(int nClass,
		       const int* classes,
		       double* classificationTable) const;

private:
  bool checkData() const;
  void clearPlotters();
  bool addTrainable(const char* classifierName, SprAbsClassifier* c);
  bool mapVars(SprAbsTrainedClassifier* t);
  bool mapMCVars(const SprTrainedMultiClassLearner* t);
  static SprAbsTwoClassCriterion* makeCrit(const char* criterion);

  // variables
  std::set<std::string> includeVars_;
  std::set<std::string> excludeVars_;

  // data
  SprAbsFilter* trainData_;
  SprAbsFilter* testData_;
  bool needToTest_;

  // garbage collection
  SprAbsFilter* trainGarbage_;
  SprAbsFilter* testGarbage_;

  // classifiers
  std::map<std::string,SprAbsClassifier*> trainable_;
  std::map<std::string,SprAbsTrainedClassifier*> trained_;
  SprMultiClassLearner* mcTrainable_;
  SprTrainedMultiClassLearner* mcTrained_;
  std::map<SprAbsTrainedClassifier*,SprCoordinateMapper*> mapper_;
  SprCoordinateMapper* mcMapper_;

  // transformer
  SprAbsVarTransformer* trans_;

  // plotter
  SprAbsTwoClassCriterion* showCrit_;
  SprPlotter* plotter_;
  SprMultiClassPlotter* mcPlotter_;

  // various stuff to be cleaned
  std::vector<const SprAbsTwoClassCriterion*> crit_;
  std::vector<SprIntegerBootstrap*> bootstrap_;
  std::set<SprAbsClassifier*> aux_;
  std::vector<SprAverageLoss*> loss_;
};

#endif
