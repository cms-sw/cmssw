//$Id: SprRootAdapter.cc,v 1.2 2007/08/30 17:54:42 narsky Exp $

#include "PhysicsTools/StatPatternRecognition/interface/SprExperiment.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprRootAdapter.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprData.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAbsFilter.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprEmptyFilter.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprClass.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAbsClassifier.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAbsTrainedClassifier.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprSimpleReader.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprRootReader.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprPlotter.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprStringParser.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprUtils.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprClassifierReader.hh"
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
#include "PhysicsTools/StatPatternRecognition/interface/SprIntegerPermutator.hh"

#include "PhysicsTools/StatPatternRecognition/src/SprSymMatrix.hh"
#include "PhysicsTools/StatPatternRecognition/src/SprVector.hh"

#include <iostream>
#include <cassert>
#include <cmath>
#include <utility>

using namespace std;


SprRootAdapter::~SprRootAdapter()
{
  delete trainData_;
  delete testData_;
  delete showCrit_;
  this->clearClassifiers();
}


SprRootAdapter::SprRootAdapter()
  :
  includeVars_(),
  excludeVars_(),
  trainData_(0),
  testData_(0),
  trainable_(),
  trained_(),
  mapper_(),
  plotter_(0),
  showCrit_(0),
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
  this->clearClassifiers();
  string sdatatype = datatype;
  SprSimpleReader reader(mode);
  if( !reader.chooseVars(includeVars_) 
      || !reader.chooseAllBut(excludeVars_) ) {
    cerr << "Unable to choose variables." << endl;
    return false;
  }
  if(      sdatatype == "train" ) {
    delete trainData_;
    trainData_ = reader.read(filename);
    if( trainData_ == 0 ) {
      cerr << "Failed to read training data from file " 
	   << filename << endl;
      return false;
    }
    return true;
  }
  else if( sdatatype == "test" ) {
    delete testData_;
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


bool SprRootAdapter::loadDataFromRoot(const char* filename,
				      const char* datatype)
{
  this->clearClassifiers();
  string sdatatype = datatype;
  SprRootReader reader;
  if(      sdatatype == "train" ) {
    delete trainData_;
    trainData_ = reader.read(filename);
    if( trainData_ == 0 ) {
      cerr << "Failed to read training data from file " 
	   << filename << endl;
      return false;
    }
    return true;
  }
  else if( sdatatype == "test" ) {
    delete testData_;
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
  for( int i=0;i<svars.size();i++ )
    strcpy(vars[i],svars[i].c_str());
  return true;
}


unsigned SprRootAdapter::nClassifierVars(const char* classifierName) const
{
  string sclassifier = classifierName;
  map<string,SprAbsTrainedClassifier*>::const_iterator found
    = trained_.find(sclassifier);
  if( found == trained_.end() ) {
    cerr << "Classifier " << sclassifier.c_str() << " not found." << endl;
    return 0;
  }
  return found->second->dim();
}


bool SprRootAdapter::classifierVars(const char* classifierName, 
				    char vars[][200]) const
{
  string sclassifier = classifierName;
  map<string,SprAbsTrainedClassifier*>::const_iterator found
    = trained_.find(sclassifier);
  if( found == trained_.end() ) {
    cerr << "Classifier " << sclassifier.c_str() << " not found." << endl;
    return false;
  }
  vector<string> cVars;
  found->second->vars(cVars);
  for( int i=0;i<cVars.size();i++ )
    strcpy(vars[i],cVars[i].c_str());
  return true;
}


bool SprRootAdapter::remapVars()
{
  for( map<SprAbsTrainedClassifier*,SprCoordinateMapper*>::iterator
	 i=mapper_.begin();i!=mapper_.end();i++ )
    delete i->second;
  mapper_.clear();
  for( map<string,SprAbsTrainedClassifier*>::iterator
	 i=trained_.begin();i!=trained_.end();i++ ) {
    if( !this->mapVars(i->second) ) {
      cerr << "Unable to map vars for classifier "
	   << i->first.c_str() << endl;
      return false;
    }
  }
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

  // get classes
  vector<SprClass> classes;
  if( !SprAbsFilter::decodeClassString(inputClassString,classes) ) {
    cerr << "Unable to decode class string " << inputClassString << endl;
    return false;
  }

  // set classes in data
  trainData_->chooseClasses(classes);
  if( !trainData_->filter() ) {
    cerr << "Unable to filter training data by class." << endl;
    return false;
  }
  testData_->chooseClasses(classes);
  if( !testData_->filter() ) {
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


bool SprRootAdapter::split(double fractionForTraining, bool randomize)
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
    testData_ = 0;
  }

  // split training data
  vector<double> weights;
  SprData* splitted = trainData_->split(fractionForTraining,weights,randomize);
  if( splitted == 0 ) {
    cerr << "Unable to split training data." << endl;
    return false;
  }

  // make test data
  bool ownData = true;
  testData_ = new SprEmptyFilter(splitted,weights,ownData);

  // clear classifiers
  this->clearClassifiers();

  // exit
  return true;
}


void SprRootAdapter::removeClassifier(const char* classifierName)
{
  bool removed = false;
  string sclassifier = classifierName;

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

  // map
  if( !this->mapVars(t) ) {
    cerr << "Unable to map variables for classifier " 
	 << sclassifier.c_str() << endl;
    return false;
  }

  // exit
  return true;
}


bool SprRootAdapter::mapVars(SprAbsTrainedClassifier* t)
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
  SprCoordinateMapper* mapper 
    = SprCoordinateMapper::createMapper(trainVars,testVars);
  if( !mapper_.insert(pair<SprAbsTrainedClassifier* const,
		      SprCoordinateMapper*>(t,mapper)).second )
    return false;
  return true;
}


void SprRootAdapter::clearClassifiers()
{
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
  delete plotter_; plotter_ = 0;
  for( int i=0;i<crit_.size();i++ )
    delete crit_[i];
  crit_.clear();
  for( int i=0;i<bootstrap_.size();i++ )
    delete bootstrap_[i];
  bootstrap_.clear();  
  for( int i=0;i<aux_.size();i++ )
    delete aux_[i];
  aux_.clear();
  for( int i=0;i<loss_.size();i++ )
    delete loss_[i];
  loss_.clear();
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
  for( int i=0;i<found.size();i++ ) {
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
      cerr << "Unable to add classifier " << i << " to AdaBoost." << endl;
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
  bool doMerge = false;
  bool discrete = true;
  SprTopdownTree* tree = new SprTopdownTree(trainData_,crit,leafSize,
					    discrete,0);
  aux_.push_back(tree);
  
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
  SprAdaBoost* c = new SprAdaBoost(trainData_,nSplitsPerDim,useStandard,
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
  for( int i=0;i<trainData_->dim();i++ ) {
    SprBinarySplit* split = new SprBinarySplit(trainData_,crit,i);
    aux_.push_back(split);
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
  bool doMerge = false;
  bool discrete = false;
  SprTopdownTree* tree = new SprTopdownTree(trainData_,crit,leafSize,
					    discrete,bs);
  aux_.push_back(tree);
  
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
  if( trainable_.empty() ) {
    cerr << "No classifiers selected for training." << endl;
    return false;
  }

  // clean up responses
  delete plotter_; plotter_ = 0;

  // map test variables onto train variables
  SprCoordinateMapper* mapper = 0;
  if( testData_ != 0 ) {
    vector<string> testVars;
    vector<string> trainVars;
    trainData_->vars(trainVars);
    testData_->vars(testVars);
    mapper = SprCoordinateMapper::createMapper(trainVars,testVars);
    if( mapper == 0 ) {
      cerr << "Unable to map training vars onto test vars." << endl;
      return false;
    }
  }    

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
    if( mapper != 0 ) {
      if( !mapper_.insert(pair<SprAbsTrainedClassifier* const,
			  SprCoordinateMapper*>(t,mapper->clone())).second ) {
	cerr << "Unable to insert mapper." << endl;
	return false;
      }
    }
    oneSuccess = true;
  }

  // clean up mapper
  delete mapper;

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
  if( trained_.empty() ) {
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
  delete plotter_; plotter_ = 0;

  // compute responses
  int N = testData_->size();
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
      vector<double> mapped(p->x_);
      map<SprAbsTrainedClassifier*,SprCoordinateMapper*>::const_iterator 
	found = mapper_.find(i->second);
      if( found != mapper_.end() )
	found->second->map(p->x_,mapped);
      resp.set(i->first.c_str(),i->second->response(mapped));
    }
    responses.push_back(resp);
  }

  // make plotter
  plotter_ = new SprPlotter(responses);
  plotter_->setCrit(showCrit_);

  // exit
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
  assert( vBgrndEff.size() == npts );

  // convert the vector into arrays
  double bgrW = plotter_->bgrndWeight();
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


bool SprRootAdapter::correlation(int cls, double* corr, const char* datatype) const
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

  // check classes
  vector<SprClass> classes;
  data->classes(classes);
  if( (cls+1) > classes.size() ) {
    cerr << "Class " << cls << " is not found in data." << endl;
    return false;
  }
  SprClass chosenClass = classes[cls];

  // filter data by class
  vector<SprClass> chosen(1,chosenClass);
  data->chooseClasses(chosen);
  if( !data->filter() ) {
    cerr << "Unable to filter data on class " << cls << endl;
    return false;
  }

  // compute
  SprDataMoments moms(data);
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

  // restore classes in data
  data->chooseClasses(classes);
  if( !data->filter() ) {
    cerr << "Unable to restore classes in data." << endl;
    return false;
  }

  // exit
  return true;
}


bool SprRootAdapter::variableImportance(const char* classifierName,
                                        unsigned nPerm,
                                        char vars[][200], 
                                        double* importance) const
{
  // sanity check
  if( testData_ == 0 ) {
    cerr << "Test data has not been loaded." << endl;
    return false;
  }
  if( nPerm == 0 ) {
    cerr << "No permutations requested. Will use one by default." << endl;
    nPerm = 1;
  }

  // find classifier and mapper
  string sclassifier = classifierName;
  map<string,SprAbsTrainedClassifier*>::const_iterator ic
    = trained_.find(sclassifier);
  if( ic == trained_.end() ) {
    cerr << "Classifier " << sclassifier.c_str() << " not found." << endl;
    return false;
  }
  SprAbsTrainedClassifier* trained = ic->second;
  SprCoordinateMapper* mapper = 0;
  map<SprAbsTrainedClassifier*,SprCoordinateMapper*>::const_iterator im
    = mapper_.find(trained);
  if( im != mapper_.end() )
    mapper = im->second;

  // check classes
  vector<SprClass> classes; 
  testData_->classes(classes); 
  if( classes.size() < 2 ) {
    cerr << "Classes have not been set." << endl;
    return false; 
  }

  // make loss
  SprAverageLoss* loss = new SprAverageLoss(&SprLoss::quadratic);
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

  //
  // pass through all variables
  //
  vector<string> testVars;
  trained->vars(testVars);
  int N = testData_->size();
  SprIntegerPermutator permu(N);
  int nVars = testVars.size();
  vector<pair<string,double> > losses;

  // make first pass without permutations
  for( int n=0;n<N;n++ ) {
    const SprPoint* p = (*testData_)[n];
    const SprPoint* mappedP = p;
    int icls = -1;
    if(      p->class_ == classes[0] )
      icls = 0;
    else if( p->class_ == classes[1] )
      icls = 1;
    else
      continue;
    if( mapper != 0 ) mappedP = mapper->output(p);
    loss->update(icls,trained->response(mappedP),testData_->w(n));
    if(  mapper != 0 ) mapper->clear();
  }
  double nominalLoss = loss->value();

  // loop over permutations
  cout << "Will perform " << nPerm << " permutations per variable." << endl;
  for( int d=0;d<nVars;d++ ) {
    cout << "Permuting variable " << testVars[d].c_str() << endl;

    // map this var
    int mappedD = d;
    if( mapper != 0 )
      mappedD = mapper->mappedIndex(d);
    assert( mappedD>=0 && mappedD<testData_->dim() );

    // pass through all points permuting them
    double aveLoss = 0;
    for( int i=0;i<=nPerm;i++ ) {

      // permute this variable
      vector<unsigned> seq;
      if( !permu.sequence(seq) ) {
        cerr << "Unable to permute points." << endl;
        return 5;
      }

      // pass through points
      loss->reset();
      for( int n=0;n<N;n++ ) {
        SprPoint p(*(*testData_)[n]);
        p.x_[mappedD] = (*testData_)[seq[n]]->x_[mappedD];
        const SprPoint* mappedP = &p;
        int icls = -1;
        if(      p.class_ == classes[0] )
          icls = 0;
        else if( p.class_ == classes[1] )
          icls = 1;
        else
          continue;
        if( mapper != 0 ) mappedP = mapper->output(&p);
        loss->update(icls,trained->response(mappedP),testData_->w(n));
        if( mapper != 0 ) mapper->clear();
      }

      // store loss
      aveLoss += loss->value();
    }// end loop over permutations

    // get and store average loss
    aveLoss = (aveLoss-nominalLoss)/nPerm;
    losses.push_back(pair<string,double>(testVars[d],aveLoss));
  }// end loop over variables
  assert( losses.size() == nVars );

  // copy computed values
  for( int d=0;d<nVars;d++ ) {
    strcpy(vars[d],losses[d].first.c_str());
    importance[d] = losses[d].second;
  }

  // exit
  return true;
}


bool SprRootAdapter::histogram(const char* classifierName,
			       double xlo, double dx, int nbin,
			       double* sig, double* sigerr,
			       double* bgr, double* bgrerr) const
{
  // sanity check
  if( plotter_ == 0 ) {
    cerr << "No response vectors found. Nothing to histogram." << endl;
    return false;
  }

  // call through
  double xhi = xlo + dx*nbin;
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
