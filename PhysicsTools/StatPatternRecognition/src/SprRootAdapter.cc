//$Id: SprRootAdapter.cc,v 1.4 2007/12/01 01:29:46 narsky Exp $

#include "PhysicsTools/StatPatternRecognition/interface/SprExperiment.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprRootAdapter.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprData.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAbsFilter.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprEmptyFilter.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprClass.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAbsClassifier.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAbsTrainedClassifier.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprMultiClassLearner.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTrainedMultiClassLearner.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprSimpleReader.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprRootReader.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprPlotter.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprMultiClassPlotter.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprStringParser.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprUtils.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprClassifierReader.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprMultiClassReader.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprCoordinateMapper.hh"

#include "PhysicsTools/StatPatternRecognition/interface/SprFisher.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTrainedFisher.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprLogitR.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTrainedLogitR.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTopdownTree.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTrainedTopdownTree.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprDecisionTree.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTrainedDecisionTree.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAdaBoost.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTrainedAdaBoost.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprStdBackprop.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTrainedStdBackprop.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprBagger.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprArcE4.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTrainedBagger.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprBinarySplit.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTrainedBinarySplit.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprBumpHunter.hh"

#include "PhysicsTools/StatPatternRecognition/interface/SprAbsTwoClassCriterion.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTwoClassSignalSignif.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTwoClassIDFraction.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTwoClassTaggerEff.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTwoClassPurity.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTwoClassGiniIndex.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTwoClassCrossEntropy.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTwoClassUniformPriorUL90.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTwoClassBKDiscovery.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTwoClassPunzi.hh"

#include "PhysicsTools/StatPatternRecognition/interface/SprIntegerBootstrap.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAverageLoss.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprLoss.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprDataMoments.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprClassifierEvaluator.hh"

#include "PhysicsTools/StatPatternRecognition/interface/SprRootWriter.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprDataFeeder.hh"

#include "PhysicsTools/StatPatternRecognition/interface/SprTransformerFilter.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAbsVarTransformer.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprVarTransformerReader.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprPCATransformer.hh"

#include "PhysicsTools/StatPatternRecognition/src/SprSymMatrix.hh"
#include "PhysicsTools/StatPatternRecognition/src/SprVector.hh"

#include <iostream>
#include <cassert>
#include <cmath>
#include <utility>
#include <algorithm>
#include <memory>

using namespace std;


SprRootAdapter::~SprRootAdapter()
{
  delete trainData_;
  delete testData_;
  delete trainGarbage_;
  delete testGarbage_;
  delete trans_;
  delete showCrit_;
  this->clearClassifiers();
}


SprRootAdapter::SprRootAdapter()
  :
  includeVars_(),
  excludeVars_(),
  trainData_(0),
  testData_(0),
  needToTest_(true),
  trainGarbage_(),
  testGarbage_(),
  trainable_(),
  trained_(),
  mcTrainable_(0),
  mcTrained_(0),
  mapper_(),
  mcMapper_(),
  trans_(0),
  showCrit_(0),
  plotter_(0),
  mcPlotter_(0),
  crit_(),
  bootstrap_(),
  aux_(),
  loss_()
{}


void SprRootAdapter::chooseVars(int nVars, const char vars[][200])
{
  includeVars_.clear();
  for( int i=0;i<nVars;i++ )
    includeVars_.insert(vars[i]);
}


void SprRootAdapter::chooseAllBut(int nVars, const char vars[][200])
{
  excludeVars_.clear();
  for( int i=0;i<nVars;i++ )
    excludeVars_.insert(vars[i]);
}


void SprRootAdapter::chooseAllVars()
{
  includeVars_.clear();
  excludeVars_.clear();
}


bool SprRootAdapter::loadDataFromAscii(int mode, 
				       const char* filename,
				       const char* datatype)
{
  // init reader
  SprSimpleReader reader(mode);
  if( !reader.chooseVars(includeVars_) 
      || !reader.chooseAllBut(excludeVars_) ) {
    cerr << "Unable to choose variables." << endl;
    return false;
  }

  // get data type
  string sdatatype = datatype;
  if(      sdatatype == "train" ) {
    cout << "Warning: training data will be reloaded." << endl;
    this->clearClassifiers();
    delete trainData_;
    delete trainGarbage_;
    trainGarbage_ = 0;
    trainData_ = reader.read(filename);
    if( trainData_ == 0 ) {
      cerr << "Failed to read training data from file " 
	   << filename << endl;
      return false;
    }
    return true;
  }
  else if( sdatatype == "test" ) {
    cout << "Warning: test data will be reloaded." << endl;
    needToTest_ = true;
    delete testData_;
    delete testGarbage_;
    testGarbage_ = 0;
    testData_ = reader.read(filename);
    if( testData_ == 0 ) {
      cerr << "Failed to read test data from file " 
	   << filename << endl;
      return false;
    }
    return true;
  }
  cerr << "Unknown data type. Must be train or test." << endl;

  // exit
  return false;
}


bool SprRootAdapter::loadDataFromRoot(const char* filename,
				      const char* datatype)
{
  SprRootReader reader;
  string sdatatype = datatype;
  if(      sdatatype == "train" ) {
    this->clearClassifiers();
    delete trainData_;
    delete trainGarbage_;
    trainGarbage_ = 0;
    trainData_ = reader.read(filename);
    if( trainData_ == 0 ) {
      cerr << "Failed to read training data from file " 
	   << filename << endl;
      return false;
    }
    return true;
  }
  else if( sdatatype == "test" ) {
    needToTest_ = true;
    delete testData_;
    delete testGarbage_;
    testGarbage_ = 0;
    testData_ = reader.read(filename);
    if( testData_ == 0 ) {
      cerr << "Failed to read test data from file " 
	   << filename << endl;
      return false;
    }
    return true;
  }
  cerr << "Unknown data type. Must be train or test." << endl;
  return false;
}


unsigned SprRootAdapter::dim() const
{
  if( trainData_ == 0 ) {
    cerr << "Training data has not been loaded." << endl;
    return 0;
  }
  return trainData_->dim();
}


bool SprRootAdapter::vars(char vars[][200]) const
{
  if( trainData_ == 0 ) {
    cerr << "Training data has not been loaded." << endl;
    return false;
  }
  vector<string> svars;
  trainData_->vars(svars);
  assert( svars.size() == trainData_->dim() );
  for( unsigned int i=0;i<svars.size();i++ )
    strcpy(vars[i],svars[i].c_str());
  return true;
}


unsigned SprRootAdapter::nClassifierVars(const char* classifierName) const
{
  string sclassifier = classifierName;
  if( sclassifier == "MultiClassLearner" ) {
    if( mcTrained_ == 0 ) {
      cerr << "Classifier MultiClassLearner not found." << endl;
      return 0;
    }
    return mcTrained_->dim();
  }
  else {
    map<string,SprAbsTrainedClassifier*>::const_iterator found
      = trained_.find(sclassifier);
    if( found == trained_.end() ) {
      cerr << "Classifier " << sclassifier.c_str() << " not found." << endl;
      return 0;
    }
    return found->second->dim();
  }
  return 0;
}


bool SprRootAdapter::classifierVars(const char* classifierName, 
				    char vars[][200]) const
{
  string sclassifier = classifierName;
  vector<string> cVars;
  if( sclassifier == "MultiClassLearner" ) {
    if( mcTrained_ == 0 ) {
      cerr << "Classifier MultiClassLearner not found." << endl;
      return false;
    }
    mcTrained_->vars(cVars);
  }
  else {
    map<string,SprAbsTrainedClassifier*>::const_iterator found
      = trained_.find(sclassifier);
    if( found == trained_.end() ) {
      cerr << "Classifier " << sclassifier.c_str() << " not found." << endl;
      return false;
    }
    found->second->vars(cVars);
  }
  for( unsigned int i=0;i<cVars.size();i++ )
    strcpy(vars[i],cVars[i].c_str());
  return true;
}


int SprRootAdapter::nClasses() const
{
  if( trainData_ == 0 ) return 0;
  vector<SprClass> classes;
  trainData_->classes(classes);
  return classes.size();
}


bool SprRootAdapter::chooseClasses(const char* inputClassString)
{
  // sanity check
  if( trainData_ == 0 ) {
    cerr << "Training data has not been loaded." << endl;
    return false;
  }
  if( testData_ == 0 ) {
    cerr << "Test data has not been loaded." << endl;
    return false;
  }

  // set classes in data
  if( !trainData_->filterByClass(inputClassString) ) {
    cerr << "Unable to filter training data by class." << endl;
    return false;
  }
  if( !testData_->filterByClass(inputClassString) ) {
    cerr << "Unable to filter test data by class." << endl;
    return false;
  }

  // clean up
  this->clearClassifiers();

  // exit
  return true;
}


bool SprRootAdapter::scaleWeights(double w, const char* classtype)
{
  if( !this->checkData() ) return false;
  vector<SprClass> classes;
  trainData_->classes(classes);
  if(      classtype == "signal" ) {
    trainData_->scaleWeights(classes[1],w);
    testData_->scaleWeights(classes[1],w);
  }
  else if( classtype == "background" ) {
    trainData_->scaleWeights(classes[0],w);
    testData_->scaleWeights(classes[0],w);
  }
  return true;
}


bool SprRootAdapter::split(double fractionForTraining, 
			   bool randomize, int seed)
{
  // sanity check
  if( trainData_ == 0 ) {
    cerr << "Training data has not been loaded." << endl;
    return false;
  }

  // if test data was specified, issue a warning
  if( testData_ != 0 ) {
    cout << "Test data will be deleted." << endl;
    delete testData_;
    delete testGarbage_;
    testData_ = 0;
    testGarbage_ = 0;
  }

  // split training data
  vector<double> weights;
  SprData* splitted 
    = trainData_->split(fractionForTraining,weights,randomize,seed);
  if( splitted == 0 ) {
    cerr << "Unable to split training data." << endl;
    return false;
  }

  // make test data
  bool ownData = true;
  testData_ = new SprEmptyFilter(splitted,weights,ownData);

  // clear classifiers
  this->clearClassifiers();
  needToTest_ = true;

  // exit
  return true;
}


void SprRootAdapter::removeClassifier(const char* classifierName)
{
  bool removed = false;
  string sclassifier = classifierName;

  // remove multi-class learner
  if( sclassifier == "MultiClassLearner" ) {
    if( mcTrainable_!=0 || mcTrained_!=0 )
      cout << "Removing multi-class learner." << endl;
    else
      cout << "Multi-class learner not found." << endl;
    delete mcTrainable_; mcTrainable_ = 0;
    delete mcTrained_; mcTrained_ = 0;
    delete mcMapper_; mcMapper_ = 0;
    return;
  }

  // remove trainable
  map<string,SprAbsClassifier*>::iterator i1 = trainable_.find(sclassifier);
  if( i1 != trainable_.end() ) {
    delete i1->second;
    trainable_.erase(i1);
    cout << "Removed trainable classifier " << sclassifier.c_str() << endl;
    removed = true;
  }

  // remove trained
  map<string,SprAbsTrainedClassifier*>::iterator i2 
    = trained_.find(sclassifier);
  if( i2 != trained_.end() ) {
    map<SprAbsTrainedClassifier*,SprCoordinateMapper*>::iterator im
      = mapper_.find(i2->second);
    if( im != mapper_.end() ) {
      delete im->second;
      mapper_.erase(im);
    }
    delete i2->second;
    trained_.erase(i2);
    cout << "Removed trained classifier " << sclassifier.c_str() << endl;
    removed = true;
  }

  // exit
  if( !removed ) {
    cout << "Unable to remove. Classifier " << sclassifier.c_str()
	 << " not found." << endl;
  }
}


bool SprRootAdapter::saveClassifier(const char* classifierName,
				    const char* filename) const
{
  string sclassifier = classifierName;

  if( sclassifier == "MultiClassLearner" ) {
    if( mcTrained_ == 0 ) {
      cerr << "MultiClassLearner not found. Unable to save." << endl;
      return false;
    }
    if( !mcTrained_->store(filename) ) {    
      cerr << "Unable to store MultiClassLearner "
	   << " into file " << filename << endl;
      return false;
    }
    return true;
  }

  map<string,SprAbsTrainedClassifier*>::const_iterator found 
    = trained_.find(sclassifier);
  if( found == trained_.end() ) {
    cerr << "Classifier " << sclassifier.c_str() << " not found." << endl;
    return false;
  }
  if( !found->second->store(filename) ) {
    cerr << "Unable to store classifier " << sclassifier.c_str()
	 << " into file " << filename << endl;
    return false;
  }
  return true;
}


bool SprRootAdapter::loadClassifier(const char* classifierName,
				    const char* filename)

{
  // sanity check
  if( testData_ == 0 ) {
    cerr << "Test data has not been loaded." << endl;
    return false;
  }

  // string
  string sclassifier = classifierName;

  // load multi-class learner
  if( sclassifier == "MultiClassLearner" ) {
    if( mcTrained_ != 0 ) {
      cerr << "MultiClassLearner already exists. " 
	   << "Unable to load." << endl;
      return false;
    }
    SprMultiClassReader reader;
    if( !reader.read(filename) ) {
      cerr << "Unable to read classifier from file " << filename << endl;
      return false;
    }
    mcTrained_ = reader.makeTrained();
    assert( mcTrained_ != 0 );
    return true;
  }

  // check if exists
  map<string,SprAbsTrainedClassifier*>::iterator found 
    = trained_.find(sclassifier);
  if( found != trained_.end() ) {
    cerr << "Classifier " << sclassifier.c_str() << " already exists. " 
	 << "Unable to load." << endl;
    return false;
  }

  // read trained classifier
  SprAbsTrainedClassifier* t = SprClassifierReader::readTrained(filename);
  if( t == 0 ) {
    cerr << "Unable to read classifier from file " << filename << endl;
    return false;
  }
  if( !trained_.insert(pair<const string,
		       SprAbsTrainedClassifier*>(sclassifier,t)).second ) {
    cerr << "Unable to add classifier " << sclassifier.c_str() 
	 << " to list." << endl;
    return false;
  }

  // exit
  return true;
}


bool SprRootAdapter::mapVars(SprAbsTrainedClassifier* t)
{
  // sanity check
  assert( t != 0 );
  if( testData_ == 0 ) {
    cerr << "Test data has not been loaded." << endl;
    return false;
  }

  // get var lists
  vector<string> trainVars;
  vector<string> testVars;
  t->vars(trainVars);
  testData_->vars(testVars);

  // make mapper and insert if it does not exist yet
  SprCoordinateMapper* mapper 
    = SprCoordinateMapper::createMapper(trainVars,testVars);
  map<SprAbsTrainedClassifier*,SprCoordinateMapper*>::iterator
    found = mapper_.find(t);
  if( found == mapper_.end() ) {
    if( !mapper_.insert(pair<SprAbsTrainedClassifier* const,
			SprCoordinateMapper*>(t,mapper)).second ) {
      cerr << "Unable to insert mapper." << endl;
      delete mapper;
      return false;
    }
  }
  else {
    delete found->second;
    found->second = mapper;
  }

  // exit
  return true;
}


bool SprRootAdapter::mapMCVars(const SprTrainedMultiClassLearner* t)
{
  assert( t != 0 );
  if( testData_ == 0 ) {
    cerr << "Test data has not been loaded." << endl;
    return false;
  }
  vector<string> trainVars;
  vector<string> testVars;
  t->vars(trainVars);
  testData_->vars(testVars);
  delete mcMapper_;
  mcMapper_ = SprCoordinateMapper::createMapper(trainVars,testVars);
  return true;
}


void SprRootAdapter::clearPlotters()
{
  delete plotter_; plotter_ = 0;
  delete mcPlotter_; mcPlotter_ = 0;
}


void SprRootAdapter::clearClassifiers()
{
  // multiclass
  delete mcTrainable_;
  delete mcTrained_;
  delete mcMapper_;
  mcTrainable_ = 0;
  mcTrained_ = 0;
  mcMapper_ = 0;

  // others
  for( map<string,SprAbsClassifier*>::const_iterator 
	 i=trainable_.begin();i!=trainable_.end();i++ )
    delete i->second;
  for( map<string,SprAbsTrainedClassifier*>::const_iterator 
	 i=trained_.begin();i!=trained_.end();i++ )
    delete i->second;
  trainable_.clear();
  trained_.clear();
  for( map<SprAbsTrainedClassifier*,SprCoordinateMapper*>::const_iterator
	 i=mapper_.begin();i!=mapper_.end();i++ )
    delete i->second;
  mapper_.clear();
  for( unsigned int i=0;i<crit_.size();i++ )
    delete crit_[i];
  crit_.clear();
  for( unsigned int i=0;i<bootstrap_.size();i++ )
    delete bootstrap_[i];
  bootstrap_.clear();  
  for( set<SprAbsClassifier*>::const_iterator 
	 i=aux_.begin();i!=aux_.end();i++ ) delete *i;
  aux_.clear();
  for( unsigned int i=0;i<loss_.size();i++ )
    delete loss_[i];
  loss_.clear();

  // plotters
  this->clearPlotters();
}


bool SprRootAdapter::showDataInClasses(char classes[][200],
				       int* events,
				       double* weights,
				       const char* datatype) const
{
  // check if classes have been set
  if( trainData_ == 0 ) {
    cerr << "Training data has not been loaded." << endl;
    return false;
  }
  vector<SprClass> found;
  trainData_->classes(found);
  if( found.size() < 2 ) {
    cerr << "Classes have not been chosen." << endl;
    return false;
  }

  // check data type
  string sdatatype = datatype;
  SprAbsFilter* data = 0;
  if(      sdatatype == "train" )
    data = trainData_;
  else if( sdatatype == "test" )
    data = testData_;
  if( data == 0 ) {
    cerr << "Data of type " << sdatatype.c_str() 
	 << " has not been loaded." << endl;
    return false;
  }

  // get events and weights
  for( unsigned int i=0;i<found.size();i++ ) {
    strcpy(classes[i],found[i].toString().c_str());
    events[i] = data->ptsInClass(found[i]);
    weights[i] = data->weightInClass(found[i]);
  }

  // exit
  return true;
}


SprAbsClassifier* SprRootAdapter::addFisher(const char* classifierName, 
					    int mode)
{
  if( !this->checkData() ) return 0;
  SprFisher* c = new SprFisher(trainData_,mode);
  if( !this->addTrainable(classifierName,c) ) return 0;
  return c;
}


SprAbsClassifier* SprRootAdapter::addLogitR(const char* classifierName,
                                            double eps,
                                            double updateFactor)
{
  if( !this->checkData() ) return 0;
  SprLogitR* c = new SprLogitR(trainData_,eps,updateFactor);
  if( !this->addTrainable(classifierName,c) ) return 0;
  return c;
}


SprAbsClassifier* SprRootAdapter::addBumpHunter(const char* classifierName,
						const char* criterion,
						unsigned minEventsPerBump,
						double peel)
{
  // sanity check
  if( !this->checkData() ) return 0;

  // make criterion
  const SprAbsTwoClassCriterion* crit = SprRootAdapter::makeCrit(criterion);
  crit_.push_back(crit);

  // make bump hunter
  SprBumpHunter* c = new SprBumpHunter(trainData_,crit,1,
				       minEventsPerBump,peel);

  // exit
  if( !this->addTrainable(classifierName,c) ) return 0;
  return c;
}


SprAbsClassifier* SprRootAdapter::addDecisionTree(const char* classifierName,
						  const char* criterion,
						  unsigned leafSize)
{
  // sanity check
  if( !this->checkData() ) return 0;

  // make criterion
  const SprAbsTwoClassCriterion* crit = SprRootAdapter::makeCrit(criterion);
  crit_.push_back(crit);

  // params
  bool doMerge = !crit->symmetric();
  bool discrete = true;

  // make a tree
  SprDecisionTree* c = new SprDecisionTree(trainData_,crit,leafSize,
					   doMerge,discrete,0);

  // exit
  if( !this->addTrainable(classifierName,c) ) return 0;
  return c;
}


SprAbsClassifier* SprRootAdapter::addTopdownTree(const char* classifierName,
						 const char* criterion,
						 unsigned leafSize,
						 unsigned nFeaturesToSample,
						 bool discrete)
{
  // sanity check
  if( !this->checkData() ) return 0;

  // make criterion
  const SprAbsTwoClassCriterion* crit = SprRootAdapter::makeCrit(criterion);
  crit_.push_back(crit);

  // check
  bool doMerge = !crit->symmetric();
  if( doMerge ) {
    cout << "Warning: Merging has no effect for Topdown trees. "
	 << "Use addDecisionTree() for asymmetric optimization criteria."
	 << endl;
  }

  // params
  SprIntegerBootstrap* bs = 0;
  if( nFeaturesToSample > 0 ) {
    bs = new SprIntegerBootstrap(trainData_->dim(),nFeaturesToSample);
    bootstrap_.push_back(bs);
  }
  
  // make a tree
  SprTopdownTree* c = new SprTopdownTree(trainData_,crit,leafSize,
					 discrete,bs);

  // exit
  if( !this->addTrainable(classifierName,c) ) return 0;
  return c;
}


SprAbsClassifier* SprRootAdapter::addStdBackprop(const char* classifierName,
                                                 const char* structure,
                                                 unsigned ncycles,
                                                 double eta,
                                                 double initEta,
                                                 unsigned nInitPoints,
                                                 unsigned nValidate)
{
  // sanity check
  if( !this->checkData() ) return 0;
  
  // make neural net
  SprStdBackprop* c = new SprStdBackprop(trainData_,
                                         structure,
                                         ncycles,
                                         eta);
  if( !c->init(initEta,nInitPoints) ) {
    cerr << "Unable to initialize neural net." << endl;
    return 0;
  } 
  if( nValidate > 0 ) {
    if( testData_==0 || !c->setValidation(testData_,nValidate) ) {
        cout << "Unable to set validation data for classifier "
             << classifierName << endl;
    }
  }

  // exit
  if( !this->addTrainable(classifierName,c) ) return 0;
  return c;
}


SprAbsClassifier* SprRootAdapter::addAdaBoost(const char* classifierName,
					      int nClassifier,
					      SprAbsClassifier** classifier,
					      bool* useCut,
					      double* cut,
					      unsigned ncycles,
					      int mode,
					      bool bagInput,
					      double epsilon,
					      unsigned nValidate)
{
  // sanity check
  if( !this->checkData() ) return 0;

  // make AdaBoost mode
  SprTrainedAdaBoost::AdaBoostMode abMode = SprTrainedAdaBoost::Discrete;
  switch( mode )
    {
    case 1 :
      abMode = SprTrainedAdaBoost::Discrete;
      cout << "Will train Discrete AdaBoost." << endl;
      break;
    case 2 :
      abMode = SprTrainedAdaBoost::Real;
      cout << "Will train Real AdaBoost." << endl;
      break;
    case 3 :
      abMode = SprTrainedAdaBoost::Epsilon;
      cout << "Will train Epsilon AdaBoost." << endl;
      break;
   default :
      cout << "Will train Discrete AdaBoost." << endl;
      break;
    }
  
  // make AdaBoost
  bool useStandard = false;
  SprAdaBoost* c = new SprAdaBoost(trainData_,ncycles,
				   useStandard,abMode,bagInput);
  c->setEpsilon(epsilon);
  if( nValidate > 0 ) {
    SprAverageLoss* loss = new SprAverageLoss(&SprLoss::exponential);
    loss_.push_back(loss);
    if( testData_==0 || !c->setValidation(testData_,nValidate,loss) ) {
	cout << "Unable to set validation data for classifier "
	     << classifierName << endl;
    }
  }

  // add weak classifiers
  for( int i=0;i<nClassifier;i++ ) {
    bool status = false;
    if( useCut[i] )
      status = c->addTrainable(classifier[i],SprUtils::lowerBound(cut[i]));
    else
      status = c->addTrainable(classifier[i]);
    if( !status ) {
      cerr << "Unable to add classifier " << i << " to AdaBoost." << endl;
      return 0;
    }
  }

  // exit
  if( !this->addTrainable(classifierName,c) ) return 0;
  return c;
}



SprAbsClassifier* SprRootAdapter::addBagger(const char* classifierName,
					    int nClassifier,
					    SprAbsClassifier** classifier,
					    unsigned ncycles,
					    bool discrete,
					    unsigned nValidate)
{
  // sanity check
  if( !this->checkData() ) return 0;

  // make bagger
  SprBagger* c = new SprBagger(trainData_,ncycles,discrete);
  if( nValidate > 0 ) {
    SprAverageLoss* loss = new SprAverageLoss(&SprLoss::quadratic);
    loss_.push_back(loss);
    if( testData_==0 || !c->setValidation(testData_,nValidate,0,loss) ) {
	cout << "Unable to set validation data for classifier "
	     << classifierName << endl;
    }
  }

  // add weak classifiers
  for( int i=0;i<nClassifier;i++ ) {
    if( !c->addTrainable(classifier[i]) ) {
      cerr << "Unable to add classifier " << i << " to Bagger." << endl;
      return 0;
    }
  }

  // exit
  if( !this->addTrainable(classifierName,c) ) return 0;
  return c;
}


SprAbsClassifier* SprRootAdapter::addBoostedDecisionTree(
					    const char* classifierName,
					    int leafSize,
					    unsigned nTrees,
					    unsigned nValidate)
{
  // sanity check
  if( !this->checkData() ) return 0;

  // make a decision tree
  const SprAbsTwoClassCriterion* crit = new SprTwoClassGiniIndex;
  crit_.push_back(crit);
  // bool doMerge = false;
  bool discrete = true;
  SprTopdownTree* tree = new SprTopdownTree(trainData_,crit,leafSize,
					    discrete,0);
  aux_.insert(tree);
  
  // make AdaBoost
  bool useStandard = false;
  bool bagInput = false;
  SprAdaBoost* c = new SprAdaBoost(trainData_,nTrees,useStandard,
				   SprTrainedAdaBoost::Discrete,bagInput);
  if( nValidate > 0 ) {
    SprAverageLoss* loss = new SprAverageLoss(&SprLoss::exponential);
    loss_.push_back(loss);
    if( testData_==0 || !c->setValidation(testData_,nValidate,loss) ) {
	cout << "Unable to set validation data for classifier "
	     << classifierName << endl;
    }
  }

  // add classifier
  if( !c->addTrainable(tree,SprUtils::lowerBound(0.5)) ) {
    cerr << "Cannot add decision tree to AdaBoost." << endl;
    return 0;
  }

  // exit
  if( !this->addTrainable(classifierName,c) ) return 0;
  return c;
}


SprAbsClassifier* SprRootAdapter::addBoostedBinarySplits(
				          const char* classifierName,
					  unsigned nSplitsPerDim,
					  unsigned nValidate)
{
  // sanity check
  if( !this->checkData() ) return 0;

  // make AdaBoost
  bool useStandard = false;
  bool bagInput = false;
  SprAdaBoost* c = new SprAdaBoost(trainData_,
				   nSplitsPerDim*trainData_->dim(),
				   useStandard,
				   SprTrainedAdaBoost::Discrete,bagInput);
  if( nValidate > 0 ) {
    SprAverageLoss* loss = new SprAverageLoss(&SprLoss::exponential);
    loss_.push_back(loss);
    if( testData_==0 || !c->setValidation(testData_,nValidate,loss) ) {
	cout << "Unable to set validation data for classifier "
	     << classifierName << endl;
    }
  }

  // add splits to AdaBoost
  const SprAbsTwoClassCriterion* crit = new SprTwoClassIDFraction;
  crit_.push_back(crit);
  for( unsigned int i=0;i<trainData_->dim();i++ ) {
    SprBinarySplit* split = new SprBinarySplit(trainData_,crit,i);
    aux_.insert(split);
    if( !c->addTrainable(split,SprUtils::lowerBound(0.5)) ) {
      cerr << "Cannot add binary split to AdaBoost." << endl;
      delete c;
      return 0;
    }
  }

  // exit
  if( !this->addTrainable(classifierName,c) ) return 0;
  return c;
}


SprAbsClassifier* SprRootAdapter::addRandomForest(const char* classifierName,
						  int leafSize,
						  unsigned nTrees,
						  unsigned nFeaturesToSample,
						  unsigned nValidate,
						  bool useArcE4)
{
  // sanity check
  if( !this->checkData() ) return 0;

  // make a decision tree
  const SprAbsTwoClassCriterion* crit = new SprTwoClassGiniIndex;
  crit_.push_back(crit);
  SprIntegerBootstrap* bs = 0;
  if( nFeaturesToSample > 0 ) {
    bs = new SprIntegerBootstrap(trainData_->dim(),nFeaturesToSample);
    bootstrap_.push_back(bs);
  }
  //bool doMerge = false;
  bool discrete = false;
  SprTopdownTree* tree = new SprTopdownTree(trainData_,crit,leafSize,
					    discrete,bs);
  aux_.insert(tree);
  
  // make Bagger
  SprBagger* c = 0;
  if( useArcE4 )
    c = new SprArcE4(trainData_,nTrees,discrete);
  else
    c = new SprBagger(trainData_,nTrees,discrete);
  if( nValidate > 0 ) {
    SprAverageLoss* loss = new SprAverageLoss(&SprLoss::quadratic);
    loss_.push_back(loss);
    if( testData_==0 || !c->setValidation(testData_,nValidate,0,loss) ) {
	cout << "Unable to set validation data for classifier "
	     << classifierName << endl;
    }
  }

  // add classifier
  if( !c->addTrainable(tree) ) {
    cerr << "Cannot add decision tree to RandomForest." << endl;
    return 0;
  }

  // exit
  if( !this->addTrainable(classifierName,c) ) return 0;
  return c;
}


SprMultiClassLearner* SprRootAdapter::setMultiClassLearner(
					     SprAbsClassifier* classifier,
					     int nClass,
					     const int* classes,
					     const char* mode)
{
  // sanity check
  if( !this->checkData() ) return 0;

  // check if there is a multi-class learner already
  if( mcTrainable_ != 0 ) {
    cerr << "MultiClassLearner already exists. "
	 << "Must delete before making a new one." << endl;
    return 0;
  }

  // prepare vector of classes
  assert( nClass > 0 );
  vector<int> vclasses(&classes[0],&classes[nClass]);

  // decode mode
  string smode = mode;
  SprMultiClassLearner::MultiClassMode mcMode = SprMultiClassLearner::User;
  if(      smode == "One-vs-All" )
    mcMode = SprMultiClassLearner::OneVsAll;
  else if( smode == "One-vs-One" )
    mcMode = SprMultiClassLearner::OneVsOne;
  else {
    cerr << "Unknown mode for MultiClassLearner." << endl;
    return 0;
  }

  // make the learner
  SprMatrix indicator;
  mcTrainable_ = new SprMultiClassLearner(trainData_,classifier,vclasses,
					  indicator,mcMode);

  // move the classifier from trainable list to aux
  for( map<std::string,SprAbsClassifier*>::iterator i=trainable_.begin();
       i!=trainable_.end();i++ ) {
    if( i->second == classifier ) 
      trainable_.erase(i);
  }
  aux_.insert(classifier);

  // exit
  return mcTrainable_;
}


bool SprRootAdapter::checkData() const
{
  if( trainData_ == 0 ) {
    cerr << "Training data has not been loaded." << endl;
    return false;
  }
  if( testData_ == 0 ) {
    cerr << "Test data has not been loaded." << endl;
    return false;
  }
  vector<SprClass> classes;
  trainData_->classes(classes);
  if( classes.size() < 2 ) {
    cerr << "Classes have not been chosen." << endl;
    return false;
  }
  return true;
}


bool SprRootAdapter::addTrainable(const char* classifierName, 
				  SprAbsClassifier* c)
{
  assert( c != 0 );
  string sclassifier = classifierName;

  // check that classifier does not exist
  map<string,SprAbsClassifier*>::const_iterator found =
    trainable_.find(sclassifier);
  if( found != trainable_.end() ) {
    cerr << "Classifier " << sclassifier.c_str() 
         << " already exists." << endl;
    delete c;
    return false;
  }

  // add
  if( !trainable_.insert(pair<const string,
                              SprAbsClassifier*>(sclassifier,c)).second ) {
    cerr << "Unable to add classifier " << sclassifier.c_str() << endl;
    delete c;
    return false;
  }

  // exit
  return true;
}


void SprRootAdapter::useInftyRange() const
{
  for( map<string,SprAbsTrainedClassifier*>::const_iterator
         i=trained_.begin();i!=trained_.end();i++ ) {
    SprAbsTrainedClassifier* trained = i->second;
    if(      trained->name() == "AdaBoost" ) {
      SprTrainedAdaBoost* specific 
        = static_cast<SprTrainedAdaBoost*>(trained);
      specific->useStandard();
    }
    else if( trained->name() == "Fisher" ) {
      SprTrainedFisher* specific 
        = static_cast<SprTrainedFisher*>(trained);
      specific->useStandard();
    }
    else if( trained->name() == "LogitR" ) {
      SprTrainedLogitR* specific 
        = static_cast<SprTrainedLogitR*>(trained);
      specific->useStandard();
    }
  }
}


void SprRootAdapter::use01Range() const
{
  for( map<string,SprAbsTrainedClassifier*>::const_iterator
         i=trained_.begin();i!=trained_.end();i++ ) { 
    SprAbsTrainedClassifier* trained = i->second; 
    if(      trained->name() == "AdaBoost" ) { 
      SprTrainedAdaBoost* specific  
        = static_cast<SprTrainedAdaBoost*>(trained);
      specific->useNormalized(); 
    } 
    else if( trained->name() == "Fisher" ) { 
      SprTrainedFisher* specific  
        = static_cast<SprTrainedFisher*>(trained);
      specific->useNormalized(); 
    } 
    else if( trained->name() == "LogitR" ) { 
      SprTrainedLogitR* specific  
        = static_cast<SprTrainedLogitR*>(trained);
      specific->useNormalized(); 
    } 
  }
}


bool SprRootAdapter::train(int verbose)
{
  // sanity check
  if( !this->checkData() ) return false;
  if( trainable_.empty() && mcTrainable_==0 ) {
    cerr << "No classifiers selected for training." << endl;
    return false;
  }

  // clean up responses
  this->clearPlotters();

  // train
  bool oneSuccess = false;
  for( map<string,SprAbsClassifier*>::const_iterator
        i=trainable_.begin();i!=trainable_.end();i++ ) {
    if( trained_.find(i->first) != trained_.end() ) {
      cout << "Trained classifier " << i->first.c_str()
           << " already exists. Skipping..." << endl;
      continue;
    }
    cout << "Training classifier " << i->first.c_str() << endl;
    if( !i->second->train(verbose) ) {
      cerr << "Unable to train classifier " << i->first.c_str() << endl;
      continue;
    }
    SprAbsTrainedClassifier* t = i->second->makeTrained();
    if( t == 0 ) {
      cerr << "Failed to make trained classifier " << i->first.c_str() << endl;
      continue;
    }
    if( !trained_.insert(pair<const string,
                         SprAbsTrainedClassifier*>(i->first,t)).second ) {
      cerr << "Failed to insert trained classifier." << endl;
      return false;
    }
    oneSuccess = true;
  }

  // multi class learner
  if( mcTrainable_ != 0 ) {
    if( mcTrained_ == 0 ) {
      cout << "Training MultiClassLearner." << endl;
      if( mcTrainable_->train(verbose) ) {
	mcTrained_ = mcTrainable_->makeTrained();
	if( mcTrained_ == 0 )
	  cerr << "Failed to make trained MultiClassLearner." << endl;
	else
	  oneSuccess = true;
      }
      else
	cerr << "Failed to train MultiClassLearner." << endl;
    }
    else
      cout << "Trained MultiClassLearner already exists. Skipping..." << endl;
  }

  // check if any classifiers succeeded
  if( !oneSuccess ) {
    cerr << "No classifiers have been trained successfully." << endl;
    return false;
  }

  // exit
  return true;
}


bool SprRootAdapter::test()
{
  // sanity check
  if( trained_.empty() && mcTrained_==0 ) {
    cerr << "No classifiers have been trained." << endl;
    return false;
  }
  if( testData_==0 || testData_->empty() ) {
    cerr << "No test data available." << endl;
    return false;
  }

  // check classes
  vector<SprClass> classes;
  testData_->classes(classes);
  if( classes.size() < 2 ) {
    cerr << "Less than 2 classes found in test data." << endl;
    return false;
  }

  // cleaned up responses
  this->clearPlotters();

  // get data size
  int N = testData_->size();

  // all two-class classifiers
  if( !trained_.empty() ) {
    // map variables
    for( map<string,SprAbsTrainedClassifier*>::const_iterator 
	 i=trained_.begin();i!=trained_.end();i++ ) {
      if( !this->mapVars(i->second) ) {
	cerr << "Unable to map variables for classifier " 
	     << i->first.c_str() << endl;
	return false;
      }
    }

    // compute responses
    vector<SprPlotter::Response> responses;
    for( int n=0;n<N;n++ ) {
      const SprPoint* p = (*testData_)[n];
      int cls = -1;
      if(      classes[0] == p->class_ ) 
	cls = 0;
      else if( classes[1] == p->class_ )
	cls = 1;
      else
	continue;
      double w = testData_->w(n);
      SprPlotter::Response resp(cls,w);
      for( map<string,SprAbsTrainedClassifier*>::const_iterator
	     i=trained_.begin();i!=trained_.end();i++ ) {
	vector<double> mapped;
	map<SprAbsTrainedClassifier*,SprCoordinateMapper*>::const_iterator 
	  found = mapper_.find(i->second);
	assert( found != mapper_.end() );
	found->second->map(p->x_,mapped);
	resp.set(i->first.c_str(),i->second->response(mapped));
      }
      responses.push_back(resp);
    }
    
    // make plotter
    plotter_ = new SprPlotter(responses);
    plotter_->setCrit(showCrit_);
  }

  // multi class
  if( mcTrained_ != 0 ) {
    if( !this->mapMCVars(mcTrained_) ) {
      cerr << "Unable to map variables for classifier MultiClassLearner." 
	   << endl;
      return false;
    }
    vector<int> mcClasses;
    mcTrained_->classes(mcClasses);
    vector<SprMultiClassPlotter::Response> responses;
    for( int n=0;n<N;n++ ) {
      const SprPoint* p = (*testData_)[n];
      int cls = p->class_;
      if( find(mcClasses.begin(),mcClasses.end(),cls) == mcClasses.end() )
	continue;
      double w = testData_->w(n);
      vector<double> mapped;
      assert( mcMapper_ != 0 );
      mcMapper_->map(p->x_,mapped);
      map<int,double> output;
      int assigned = mcTrained_->response(mapped,output);
      responses.push_back(SprMultiClassPlotter::Response(cls,w,
							 assigned,output));
    }
    mcPlotter_ = new SprMultiClassPlotter(responses);
  }

  // exit
  needToTest_ = false;
  return true;
}


bool SprRootAdapter::setCrit(const char* criterion)
{
  showCrit_ = SprRootAdapter::makeCrit(criterion);
  if( showCrit_ == 0 ) return false;
  if( plotter_ != 0 ) 
    plotter_->setCrit(showCrit_);
  return true;
}


bool SprRootAdapter::setEffCurveMode(const char* mode)
{
  string smode = mode;
  if( plotter_ == 0 ) {
    cerr << "Unable to set the efficiency plotting mode. "
	 << "Run test() first to fill out the plotter." << endl;
    return false;
  }
  if(      smode == "relative" )
    plotter_->useRelative();
  else if( smode == "absolute" )
    plotter_->useAbsolute();
  else {
    cerr << "Unknown mode for efficiency curve." << endl;
    return false;
  }
  return true;
}


bool SprRootAdapter::effCurve(const char* classifierName,
			      int npts, const double* signalEff,
			      double* bgrndEff, double* bgrndErr, double* fom) 
  const
{
  string sclassifier = classifierName;

  // sanity check
  if( npts == 0 ) return true;
  if( plotter_ == 0 ) {
    cerr << "No responses for test data have been computed. " 
	 << "Run test() first." << endl;
    return false;
  }

  // make vector of signal efficiencies
  vector<double> vSignalEff(npts);
  for( int i=0;i<npts;i++ )
    vSignalEff[i] = signalEff[i];

  // compute the curve
  vector<SprPlotter::FigureOfMerit> vBgrndEff;
  if( !plotter_->backgroundCurve(vSignalEff,sclassifier.c_str(),vBgrndEff) ) {
    cerr << "Unable to compute the background curve for classifier "
	 << sclassifier.c_str() << endl;
    return false;
  }
  assert(static_cast<int>(vBgrndEff.size()) == npts );

  // convert the vector into arrays
  // double bgrW = plotter_->bgrndWeight();
  for( int i=0;i<npts;i++ ) {
    bgrndEff[i] = vBgrndEff[i].bgrWeight;
    bgrndErr[i] = ( vBgrndEff[i].bgrNevts==0 ? 0 
		    : bgrndEff[i]/sqrt(double(vBgrndEff[i].bgrNevts)) );
    fom[i] = vBgrndEff[i].fom;
  }

  // exit
  return true;
}


bool SprRootAdapter::allEffCurves(int npts, const double* signalEff,
				  char classifiers[][200],
				  double* bgrndEff, double* bgrndErr,
				  double* fom) const
{
  if( trained_.empty() || plotter_==0 ) {
    cerr << "Efficiency curves cannot be computed." << endl;
    return false;
  }
  double* eff = bgrndEff;
  double* err = bgrndErr;
  double* myfom = fom;
  int curr = 0;
  for( map<string,SprAbsTrainedClassifier*>::const_iterator
	 i=trained_.begin();i!=trained_.end();i++ ) {
    if( !this->effCurve(i->first.c_str(),npts,signalEff,eff,err,myfom) ) {
      cerr << "Unable to compute efficiency for classifier "
	   << i->first.c_str() << endl;
      return false;
    }
    strcpy(classifiers[curr++],i->first.c_str());
    eff += npts;
    err += npts;
    myfom += npts;
  }
  return true;
}


bool SprRootAdapter::correlation(int cls, double* corr, const char* datatype) 
  const
{
  // sanity check
  string sdatatype = datatype;
  SprAbsFilter* data = 0;
  if(      sdatatype == "train" )
    data = trainData_;
  else if( sdatatype == "test" )
    data = testData_;
  if( data == 0 ) {
    cerr << "Data of type " << sdatatype.c_str()
         << " has not been loaded." << endl;
    return false;
  }

  // make a temp copy of data
  SprEmptyFilter tempData(data);

  // check classes
  vector<SprClass> classes;
  tempData.classes(classes);
  if( (cls+1) > (int)classes.size() ) {
    cerr << "Class " << cls << " is not found in data." << endl;
    return false;
  }
  SprClass chosenClass = classes[cls];

  // filter data by class
  vector<SprClass> chosen(1,chosenClass);
  tempData.chooseClasses(chosen);
  if( !tempData.filter() ) {
    cerr << "Unable to filter data on class " << cls << endl;
    return false;
  }

  // compute
  SprDataMoments moms(&tempData);
  SprSymMatrix cov;
  SprVector mean;
  if( !moms.covariance(cov,mean) ) {
    cerr << "Unable to compute covariance matrix." << endl;
    return false;
  }

  // compute variances
  int N = cov.num_row();
  vector<double> rms(N);
  vector<int> positive(N,0);
  for( int i=0;i<N;i++ ) {
    if( cov[i][i] < SprUtils::eps() ) {
      cout << "Variance for variable " << i << " is zero." << endl;
      rms[i] = 0;
    }
    else {
      rms[i] = sqrt(cov[i][i]);
      positive[i] = 1;
    }
  }

  // fill out array
  for( int i=0;i<N-1;i++ ) {
    for( int j=i+1;j<N;j++ ) {
      int ind = i*N+j;
      if( positive[i]*positive[j] > 0 ) 
	corr[ind] = cov[i][j]/rms[i]/rms[j];
      else
	corr[ind] = 0;
    }
  }
  for( int i=0;i<N;i++ ) corr[i*(N+1)] = 1.;
  for( int i=1;i<N;i++ ) {
    for( int j=0;j<i;j++ ) {
      corr[i*N+j] = corr[i+j*N];
    }
  }

  // exit
  return true;
}


bool SprRootAdapter::correlationClassLabel(const char* mode,
					   char vars[][200],
					   double* corr, 
					   const char* datatype) const
{
  // sanity check
  string sdatatype = datatype;
  SprAbsFilter* data = 0;
  if(      sdatatype == "train" )
    data = trainData_;
  else if( sdatatype == "test" )
    data = testData_;
  if( data == 0 ) {
    cerr << "Data of type " << sdatatype.c_str()
         << " has not been loaded." << endl;
    return false;
  }

  // fill out vars
  unsigned dim = data->dim();
  vector<string> dataVars;
  data->vars(dataVars);
  assert( dataVars.size() == dim );
  for( unsigned int d=0;d<dim;d++ )
    strcpy(vars[d],dataVars[d].c_str());

  // compute correlation
  SprDataMoments moms(data);
  string smode = mode;
  double mean(0), var(0);
  if(      smode == "normal" ) {
    for( unsigned int d=0;d<dim;d++ )
      corr[d] = moms.correlClassLabel(d,mean,var);
  }
  else if( smode == "abs" ) {
    for( unsigned int d=0;d<dim;d++ )
      corr[d] = moms.absCorrelClassLabel(d,mean,var);
  }
  else {
    cerr << "Unknown mode in correlationClassLabel." << endl;
    return false;
  }

  // exit
  return true;
}


bool SprRootAdapter::variableImportance(const char* classifierName,
                                        unsigned nPerm,
                                        char vars[][200], 
                                        double* importance,
					double* error) const
{
  // sanity check
  if( testData_ == 0 ) {
    cerr << "Test data has not been loaded." << endl;
    return false;
  }
  if( needToTest_ ) {
    cerr << "Test data has changed. Need to run test() again." << endl;
    return false;
  }

  // find classifier and mapper
  string sclassifier = classifierName;
  SprCoordinateMapper* mapper = 0;
  SprAbsTrainedClassifier* trained = 0;
  SprTrainedMultiClassLearner* mcTrained = 0;
  if( sclassifier == "MultiClassLearner" ) {
    mapper = mcMapper_;
    if( mcTrained_ == 0 ) {
      cerr << "Classifier MultiClassLearner not found." << endl;
      return false;
    }
    mcTrained = mcTrained_;
  }
  else {
    map<string,SprAbsTrainedClassifier*>::const_iterator ic
      = trained_.find(sclassifier);
    if( ic == trained_.end() ) {
      cerr << "Classifier " << sclassifier.c_str() << " not found." << endl;
      return false;
    }
    trained = ic->second;
    assert( trained != 0 );
    map<SprAbsTrainedClassifier*,SprCoordinateMapper*>::const_iterator im
      = mapper_.find(trained);
    if( im != mapper_.end() )
      mapper = im->second;
  }

  // compute importance
  vector<SprClassifierEvaluator::NameAndValue> lossIncrease;
  if( !SprClassifierEvaluator::variableImportance(testData_,
						  trained,
						  mcTrained,
						  mapper,
						  nPerm,
						  lossIncrease) ) {
    cerr << "Unable to estimate variable importance." << endl;
    return false;
  }

  // convert result into arrays
  for( unsigned int d=0;d<lossIncrease.size();d++ ) {
    strcpy(vars[d],lossIncrease[d].first.c_str());
    importance[d] = lossIncrease[d].second.first;
    error[d] = lossIncrease[d].second.second;
  }

  // exit
  return true;
}


bool SprRootAdapter::variableInteraction(const char* classifierName,
					 const char* subset,
					 unsigned nPoints,
					 char vars[][200],
					 double* interaction,
					 double* error,
					 int verbose) const
{
  // sanity check
  if( testData_ == 0 ) {
    cerr << "Test data has not been loaded." << endl;
    return false;
  }
  if( needToTest_ ) {
    cerr << "Test data has changed. Need to run test() again." << endl;
    return false;
  }

  // find classifier and mapper
  string sclassifier = classifierName;
  SprCoordinateMapper* mapper = 0;
  SprAbsTrainedClassifier* trained = 0;
  SprTrainedMultiClassLearner* mcTrained = 0;
  if( sclassifier == "MultiClassLearner" ) {
    mapper = mcMapper_;
    if( mcTrained_ == 0 ) {
      cerr << "Classifier MultiClassLearner not found." << endl;
      return false;
    }
    mcTrained = mcTrained_;
  }
  else {
    map<string,SprAbsTrainedClassifier*>::const_iterator ic
      = trained_.find(sclassifier);
    if( ic == trained_.end() ) {
      cerr << "Classifier " << sclassifier.c_str() << " not found." << endl;
      return false;
    }
    trained = ic->second;
    assert( trained != 0 );
    map<SprAbsTrainedClassifier*,SprCoordinateMapper*>::const_iterator im
      = mapper_.find(trained);
    if( im != mapper_.end() )
      mapper = im->second;
  }

  // compute interaction
  vector<SprClassifierEvaluator::NameAndValue> varInteraction;
  if( !SprClassifierEvaluator::variableInteraction(testData_,
						   trained,
						   mcTrained,
						   mapper,
						   subset,
						   nPoints,
						   varInteraction,
						   verbose) ) {
    cerr << "Unable to estimate variable interactions." << endl;
    return false;
  }

  // convert result into arrays
  for( unsigned int d=0;d<varInteraction.size();d++ ) {
    strcpy(vars[d],varInteraction[d].first.c_str());
    interaction[d] = varInteraction[d].second.first;
    error[d] = varInteraction[d].second.second;
  }

  // exit
  return true;
}


bool SprRootAdapter::histogram(const char* classifierName,
			       double xlo, double xhi, int nbin,
			       double* sig, double* sigerr,
			       double* bgr, double* bgrerr) const
{
  // sanity check
  if( plotter_ == 0 ) {
    cerr << "No response vectors found. Nothing to histogram." << endl;
    return false;
  }
  if( xhi < xlo ) {
    cerr << "requested lower X limit greater than upper X limit." << endl;
    return false;
  }

  // call through
  double dx = (xhi-xlo) / nbin;
  vector<pair<double,double> > sigHist;
  vector<pair<double,double> > bgrHist;
  int nFilledBins = plotter_->histogram(classifierName,
					xlo,xhi,dx,sigHist,bgrHist);
  if( nFilledBins < nbin ) {
    cerr << "Requested " << nbin << " bins but filled only "
	 << nFilledBins << ". Unable to plot histogram." << endl;
    return false;
  }

  // copy histogram content
  for( int i=0;i<nbin;i++ ) {
    sig[i]    = sigHist[i].first;
    sigerr[i] = sigHist[i].second;
    bgr[i]    = bgrHist[i].first;
    bgrerr[i] = bgrHist[i].second;
  }

  // exit
  return true;
}


SprAbsTwoClassCriterion* SprRootAdapter::makeCrit(const char* criterion)
{
  SprAbsTwoClassCriterion* crit = 0;
  string scrit = criterion;
  if(      scrit == "correct_id" ) {
      crit = new SprTwoClassIDFraction;
      cout << "Optimization criterion set to "
           << "Fraction of correctly classified events " << endl;
  }
  else if( scrit == "S/sqrt(S+B)" ) {
    crit = new SprTwoClassSignalSignif;
    cout << "Optimization criterion set to "
	 << "Signal significance S/sqrt(S+B) " << endl;
  }
  else if( scrit == "S/(S+B)" ) {
    crit = new SprTwoClassPurity;
    cout << "Optimization criterion set to "
	 << "Purity S/(S+B) " << endl;
  }
  else if( scrit == "TaggerEff" ) {
    crit = new SprTwoClassTaggerEff;
    cout << "Optimization criterion set to "
	 << "Tagging efficiency Q = e*(1-2w)^2 " << endl;
  }
  else if( scrit == "Gini" ) {
    crit = new SprTwoClassGiniIndex;
    cout << "Optimization criterion set to "
	 << "Gini index  -1+p^2+q^2 " << endl;
  }
  else if( scrit == "CrossEntropy" ) {
    crit = new SprTwoClassCrossEntropy;
    cout << "Optimization criterion set to "
	 << "Cross-entropy p*log(p)+q*log(q) " << endl;
  }
  else if( scrit == "CrossEntropy" ) {
    crit = new SprTwoClassUniformPriorUL90;
    cout << "Optimization criterion set to "
	 << "Inverse of 90% Bayesian upper limit with uniform prior" << endl;
  }
  else if( scrit == "BKDiscovery" ) {
    crit = new SprTwoClassBKDiscovery;
    cout << "Optimization criterion set to "
	 << "Discovery potential 2*(sqrt(S+B)-sqrt(B))" << endl;
  }
  else if( scrit == "Punzi" ) {
    crit = new SprTwoClassPunzi(1.);
    cout << "Optimization criterion set to "
         << "Punzi's sensitivity S/(0.5*nSigma+sqrt(B))" << endl;
  }
  else {
    cerr << "Unknown criterion specified." << endl;
    return 0;
  }
  return crit;
}


bool SprRootAdapter::multiClassTable(int nClass,
				     const int* classes,
				     double* classificationTable) const
{
  // sanity check
  if( mcPlotter_ == 0 ) {
    cerr << "No response vectors found. "
	 << "Cannot compute classification table." << endl;
    return false;
  }

  // make a list of classes to be included
  vector<int> vclasses(&classes[0],&classes[nClass]);

  // call mutliclass plotter
  map<int,vector<double> > mcClassificationTable;
  map<int,double> weightInClass;
  //double loss = mcPlotter_->multiClassTable(vclasses,
  //					    mcClassificationTable,
  //					    weightInClass);

  // convert the map into an array
  for( int ic=0;ic<nClass;ic++ ) {
    map<int,vector<double> >::const_iterator found 
      = mcClassificationTable.find(classes[ic]);
    if( found == mcClassificationTable.end() ) {
      for( int j=0;j<nClass;j++ )
	classificationTable[j+ic*nClass] = 0;
    }
    else {
      assert( static_cast<int>(found->second.size()) == nClass );
      for( int j=0;j<nClass;j++ )
	classificationTable[j+ic*nClass] = (found->second)[j];
    }
  }

  // exit
  return true;
}


bool SprRootAdapter::saveTestData(const char* filename) const
{
  // sanity check
  if( testData_ == 0 ) {
    cerr << "Test data has not been loaded." << endl;
    return false;
  }
  if( (!trained_.empty() || mcTrained_!=0) && needToTest_ ) {
    cerr << "Test data has changed. Need to run test() again." << endl;
    return false;
  }
  if( trained_.empty() && mcTrained_==0 ) {
    cout << "No trained classifiers found. " 
	 << "Data will be saved without any classifiers." << endl;
  }

  // create writer and feeder
  SprRootWriter writer("TestData");
  if( !writer.init(filename) ) {
    cerr << "Unable to open output file " << filename << endl;
    return false;
  }
  SprDataFeeder feeder(testData_,&writer);

  // add classifiers
  for( map<string,SprAbsTrainedClassifier*>::const_iterator 
       i=trained_.begin();i!=trained_.end();i++ ) {
    SprCoordinateMapper* mapper = 0;
    map<SprAbsTrainedClassifier*,SprCoordinateMapper*>::const_iterator
      found = mapper_.find(i->second);
    if( found != mapper_.end() )
      mapper = found->second->clone();
    if( !feeder.addClassifier(i->second,i->first.c_str(),mapper) ) {
      cerr << "Unable to add classifier " << i->first.c_str() 
	   << " to feeder." << endl;
      return false;
    }
  }
  if( mcTrained_ != 0 ) {
    SprCoordinateMapper* mapper = ( mcMapper_==0 ? 0 : mcMapper_->clone() );
    if( !feeder.addMultiClassLearner(mcTrained_,"MultiClassLearner",mapper) ) {
      cerr << "Unable to add MultiClassLearner to feeder." << endl;
      return false;
    }
  }

  // feed
  if( !feeder.feed(1000) ) {
    cerr << "Unable to feed data into writer." << endl;
    return false;
  }

  // exit
  return true;
}


bool SprRootAdapter::trainVarTransformer(const char* name, int verbose)
{
  // sanity check
  if( trainData_ == 0 ) {
    cerr << "Training data has not been loaded." << endl;
    return false;
  }

  // make a transformer
  if( trans_ != 0 ) delete trans_;
  string sname = name;
  if(      sname == "PCA" )
    trans_ = new SprPCATransformer();
  else {
    cerr << "Unknown VarTransformer type requested: " << sname.c_str() << endl;
    return false;
  }

  // train
  if( !trans_->train(trainData_,verbose) ) {
    cerr << "Unable to train VarTransformer." << endl;
    return false;
  }

  // exit
  return true;
}


bool SprRootAdapter::saveVarTransformer(const char* filename) const
{
  // sanity check
  if( trans_ == 0 ) {
    cerr << "No VarTransformer found. Unable to save." << endl;
    return false;
  }

  // save
  if( !trans_->store(filename) ) {
    cerr << "Unable to save VarTransformer to file " << filename << endl;
    return false;
  }

  // exit
  return true;
}


bool SprRootAdapter::loadVarTransformer(const char* filename)
{
  if( trans_ != 0 ) delete trans_;
  trans_ = SprVarTransformerReader::read(filename);
  if( trans_ == 0 ) {
    cerr << "Unable to load VarTransformer from file " << filename << endl;
    return false;
  }
  return true;
}


bool SprRootAdapter::transform()
{
  // sanity check
  if( trainData_ == 0 ) {
    cerr << "Training data has not been loaded. Unable to transform." << endl;
    return false;
  }
  if( testData_ == 0 ) {
    cerr << "Test data has not been loaded. Unable to transform." << endl;
    return false;
  }
  if( trans_ == 0 ) {
    cerr << "No VarTransformer found. Unable to transform." << endl;
    return false;
  }

  // make new data filters
  SprTransformerFilter* trainData = new SprTransformerFilter(trainData_);
  SprTransformerFilter* testData = new SprTransformerFilter(testData_);

  // transform
  bool replaceOriginalData = true;
  if( !trainData->transform(trans_,replaceOriginalData) ) {
    cerr << "Unable to transform training data." << endl;
    return false;
  }
  if( !testData->transform(trans_,replaceOriginalData) ) {
    cerr << "Unable to transform test data." << endl;
    return false;
  }

  // get rid of old non-transformed data
  if( trainGarbage_ == 0 )
    trainGarbage_ = trainData_;
  else
    delete trainData_;
  if( testGarbage_ == 0 )
    testGarbage_ = testData_;
  else
    delete testData_;
  trainData_ = trainData;
  testData_ = testData;

  // exit
  return true;
}
