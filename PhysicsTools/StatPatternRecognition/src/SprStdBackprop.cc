//$Id: SprStdBackprop.cc,v 1.3 2008/11/26 22:59:20 elmer Exp $

#include "PhysicsTools/StatPatternRecognition/interface/SprExperiment.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprStdBackprop.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAbsFilter.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAverageLoss.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprLoss.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprStringParser.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTransformation.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprUtils.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprDefs.hh"

#include <stdio.h>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <utility>
#include <cassert>
#include <cstring>

using namespace std;


SprStdBackprop::~SprStdBackprop()
{
  if( ownLoss_ ) {
    delete loss_;
    loss_ = 0;
    ownLoss_ = false;
  }  
}

SprStdBackprop::SprStdBackprop(SprAbsFilter* data)
  :
  SprAbsClassifier(data),
  structure_(),
  cls0_(0),
  cls1_(1),
  cycles_(0),
  eta_(0.1),
  configured_(false),
  initialized_(false),
  initEta_(0.1),
  initPoints_(data->size()),
  rndm_(),
  permu_(data->size()),
  allowPermu_(true),
  nNodes_(0),
  nLinks_(0),
  nodeType_(),
  nodeActFun_(),
  nodeAct_(),
  nodeOut_(),
  nodeNInputLinks_(),
  nodeFirstInputLink_(),
  linkSource_(),
  nodeBias_(),
  linkWeight_(),
  cut_(SprUtils::lowerBound(0.5)),
  valData_(0),
  valPrint_(0),
  loss_(0),
  ownLoss_(false),
  initialDataWeights_()
{
  this->setClasses();
}


SprStdBackprop::SprStdBackprop(SprAbsFilter* data, 
			       unsigned cycles,
			       double eta)
  :
  SprAbsClassifier(data),
  structure_(),
  cls0_(0),
  cls1_(1),
  cycles_(cycles),
  eta_(eta),
  configured_(false),
  initialized_(false),
  initEta_(0.1),
  initPoints_(data->size()),
  rndm_(),
  permu_(data->size()),
  allowPermu_(true),
  nNodes_(0),
  nLinks_(0),
  nodeType_(),
  nodeActFun_(),
  nodeAct_(),
  nodeOut_(),
  nodeNInputLinks_(),
  nodeFirstInputLink_(),
  linkSource_(),
  nodeBias_(),
  linkWeight_(),
  cut_(SprUtils::lowerBound(0.5)),
  valData_(0),
  valPrint_(0),
  loss_(0),
  ownLoss_(false),
  initialDataWeights_()
{
  this->setClasses();
  cout << "StdBackprop initialized with classes " << cls0_ << " " << cls1_
       << " nCycles=" << cycles_ << " LearningRate=" << eta_ << endl;
}


SprStdBackprop::SprStdBackprop(SprAbsFilter* data, 
			       const char* structure,
			       unsigned cycles,
			       double eta)
  :
  SprAbsClassifier(data),
  structure_(structure),
  cls0_(0),
  cls1_(1),
  cycles_(cycles),
  eta_(eta),
  configured_(false),
  initialized_(false),
  initEta_(0.1),
  initPoints_(data->size()),
  rndm_(),
  permu_(data->size()),
  allowPermu_(true),
  nNodes_(0),
  nLinks_(0),
  nodeType_(),
  nodeActFun_(),
  nodeAct_(),
  nodeOut_(),
  nodeNInputLinks_(),
  nodeFirstInputLink_(),
  linkSource_(),
  nodeBias_(),
  linkWeight_(),
  cut_(SprUtils::lowerBound(0.5)),
  valData_(0),
  valPrint_(0),
  loss_(0),
  ownLoss_(false),
  initialDataWeights_()
{
  this->setClasses();
  bool status = this->createNet();
  assert( status );
  cout << "StdBackprop initialized with classes " << cls0_ << " " << cls1_
       << " nCycles=" << cycles_ << " structure=" << structure_.c_str()
       << " LearningRate=" << eta_ << endl;
}


SprTrainedStdBackprop* SprStdBackprop::makeTrained() const 
{
  SprTrainedStdBackprop* t = new SprTrainedStdBackprop(structure_.c_str(),
						       nodeType_,nodeActFun_,
						       nodeNInputLinks_,
						       nodeFirstInputLink_,
						       linkSource_,nodeBias_,
						       linkWeight_);
  t->setCut(cut_);

  // vars
  vector<string> vars;
  data_->vars(vars);
  t->setVars(vars);

  // exit
  return t;
}


bool SprStdBackprop::createNet() 
{
  // init
  configured_ = false;

  // sanity check
  if( structure_.empty() ) {
    cerr << "No network structure specified. Exiting." << endl;
    return false;
  }

  // parse
  vector<vector<int> > layers;
  SprStringParser::parseToInts(structure_.c_str(),layers);

  // check output
  if( layers.size() < 3 ) {
    cerr << "Not enough layers in the neural net: " << layers.size() 
	 << " for structure " << structure_.c_str() << endl;
    return false;
  }
  if( layers[0].size()!=1 || layers[0][0]!=static_cast<int>(data_->dim()) ) {
    cerr << "Size of the input layer " << layers[0][0]
	 << " must be equal to the dimensionality of input data " 
	 << data_->dim() << endl;
    return false;
  }
  for( unsigned int i=1;i<layers.size()-1;i++ ) {
    if( layers[i].size()!=1 || layers[i][0]<=0 ) {
      cerr << "Error in specifying hidden layer " << i << endl;
      return false;
    }
  }
  if( layers[layers.size()-1].size()!=1 || layers[layers.size()-1][0]!=1 ) {
    cerr << "This NN implementation can only handle "
	 << "one node in the output layer." << endl;
    return false;
  }

  // create net
  nNodes_ = 0;
  for( unsigned int i=0;i<layers.size();i++ ) nNodes_ += layers[i][0];
  nodeType_.clear(); nodeType_.resize(nNodes_,SprNNDefs::INPUT);
  nodeActFun_.clear(); nodeActFun_.resize(nNodes_,SprNNDefs::ID);
  nodeAct_.clear(); nodeAct_.resize(nNodes_,0);
  nodeOut_.clear(); nodeOut_.resize(nNodes_,0);
  nodeNInputLinks_.clear(); nodeNInputLinks_.resize(nNodes_,0);
  nodeFirstInputLink_.clear(); nodeFirstInputLink_.resize(nNodes_,-1);
  nodeBias_.clear(); nodeBias_.resize(nNodes_,0);
  int index = 0;

  // input nodes
  // keep this commented out - this is just for clarity but in fact
  //   this code does nothing
  /*
  for( unsigned int i=0;i<layers[0][0];i++ ) {
    nodeType_[index]           = SprNNDefs::INPUT;
    nodeActFun_[index]         = SprNNDefs::ID;
    nodeNInputLinks_[index]    = 0; 
    nodeFirstInputLink_[index] = -1;
    index++;
  }
  */

  // hidden nodes
  index = layers[0][0];
  int firstLink = 0;
  linkSource_.clear();
  int nstart(0), nend(0);// flat node indices for the previous layer
  for( unsigned int i=1;i<layers.size()-1;i++ ) {
    nstart = nend;
    nend += layers[i-1][0];
    for( int j=0;j<layers[i][0];j++ ) {
      nodeType_[index]           = SprNNDefs::HIDDEN;
      nodeActFun_[index]         = SprNNDefs::LOGISTIC;
      nodeNInputLinks_[index]    = layers[i-1][0]; 
      nodeFirstInputLink_[index] = firstLink;
      firstLink += layers[i-1][0];
      index++;
      for( int n=nstart;n<nend;n++ ) linkSource_.push_back(n);
    }
  }

  // output nodes
  assert( index == (nNodes_-1) );
  nodeType_[index]           = SprNNDefs::OUTPUT;
  nodeActFun_[index]         = SprNNDefs::LOGISTIC;
  nodeNInputLinks_[index]    = layers[layers.size()-2][0]; 
  nodeFirstInputLink_[index] = firstLink;
  nstart = nend;
  nend += layers[layers.size()-2][0];
  for( int n=nstart;n<nend;n++ ) linkSource_.push_back(n);

  // links
  nLinks_ = linkSource_.size();
  linkWeight_.clear(); linkWeight_.resize(nLinks_,0);

  // exit
  configured_ = true;
  return true;
}


bool SprStdBackprop::init(double eta, unsigned nPoints)
{
  if( initialized_ ) return true;
  initEta_ = eta;
  initPoints_ = nPoints;
  unsigned valPrint = valPrint_;
  valPrint_ = 0;
  initialized_ = this->doTrain(initPoints_,1,initEta_,true,1);
  valPrint_ = valPrint;
  return initialized_;
}


bool SprStdBackprop::train(int verbose)
{
  // sanity check
  if( cycles_ == 0 ) {
    cout << "No training cycles for neural net requested. " 
	 << "Will exit without training." << endl;
    return true;
  }
  if( !configured_ ) {
    cerr << "Neural net configuration not specified." << endl;
    return false;
  }

  // initialize
  if( !initialized_ ) {
    if( verbose > 0 ) {
      cout << "Initializing network with learning rate " << initEta_ 
	   << " and number of points for initialization " << initPoints_ 
	   << endl;
    }
    if( !this->init(initEta_,initPoints_) ) {
      cerr << "Unable to initialize network." << endl;
      return false;
    }
    if( verbose > 0 )
      cout << "Neural net initialized." << endl;
  }

  // train
  return this->doTrain(data_->size(),cycles_,eta_,false,verbose);
}


bool SprStdBackprop::doTrain(unsigned nPoints, unsigned nCycles, 
			     double eta, bool randomizeEta, int verbose)
{
  // normalize data weights
  data_->weights(initialDataWeights_);
  vector<SprClass> classes(2);
  classes[0] = cls0_; classes[1] = cls1_;
  double wtot = data_->ptsInClass(cls0_) + data_->ptsInClass(cls1_);
  data_->normalizeWeights(classes,wtot);

  // permute input events
  unsigned size = data_->size();
  if( nPoints==0 || nPoints>size ) {
    if( verbose > 1 ) {
      cout << "Resetting the number of training points "
	   << "to the max number of points available." << endl;
    }
    nPoints = size;
  }
  vector<unsigned> indices;
  if( allowPermu_ ) {
    if( !permu_.sequence(indices) ) {
      cerr << "Unable to permute input indices for training." << endl;
      return this->prepareExit(false);
    }
  }
  else {
    for( unsigned i=0;i<nPoints;i++ ) indices.push_back(i);
  }

  // validate before training starts
  if( valPrint_!=0 ) {
    if( !this->printValidation(0) ) {
      cerr << "Unable to print out validation data." << endl;
      return this->prepareExit(false);
    }
  }

  // train
  for( unsigned int ncycle=1;ncycle<=nCycles;ncycle++ ) {
    // message
    if( verbose > 0 ) {
      if( ncycle%10 == 0 )
	cout << "Training neural net at cycle " << ncycle << endl;
    }

    // do two passes of propagation
    for( unsigned int i=0;i<nPoints;i++ ) {
      unsigned ipt = indices[i];
      const SprPoint* p = (*data_)[ipt];
      int cls = -1;
      if(      p->class_ == cls0_ )
	cls = 0;
      else if( p->class_ == cls1_ )
	cls = 1;
      else
	continue;

      // forward pass
      double output = this->forward(p->x_);

      // generate random learning factors for first cycle
      double w = data_->w(ipt);
      vector<double> etaV(nLinks_+1,w*eta);
      if( randomizeEta ) {
	double* r = new double [nLinks_+1];
	rndm_.sequence(r,nLinks_);
	for( int j=0;j<=nLinks_;j++ ) etaV[j] = eta*r[j];
	delete [] r;
      }

      // backward pass
      if( !this->backward(cls,output,etaV) ) {
	cerr << "Unable to backward-propagate at cycle " << ncycle << endl;
	return this->prepareExit(false);
      }
    }// end of do two passes of propagation

    // validate
    if( valPrint_!=0 && (ncycle%valPrint_)==0 ) {
      if( !this->printValidation(ncycle) ) {
	cerr << "Unable to print out validation data." << endl;
	return this->prepareExit(false);
      }
    }
  }

  // exit
  return this->prepareExit(true);
}


double SprStdBackprop::forward(const std::vector<double>& v)
{
  // Initialize and process input nodes
  nodeOut_.clear(); nodeOut_.resize(nNodes_,0);
  int d = 0;
  for( int i=0;i<nNodes_;i++ ) {
    if( nodeType_[i] == SprNNDefs::INPUT )
      nodeOut_[i] = v[d++];
    else
      break;
  }

  // Process hidden and output nodes
  for( int i=0;i<nNodes_;i++ ) {
    nodeAct_[i] = 0;
    if( nodeNInputLinks_[i] > 0 ) {
      for( int j=nodeFirstInputLink_[i];
	   j<nodeFirstInputLink_[i]+nodeNInputLinks_[i];j++ ) {
        nodeAct_[i] += nodeOut_[linkSource_[j]] * linkWeight_[j];
      }
      nodeOut_[i] = this->activate(nodeAct_[i]+nodeBias_[i],nodeActFun_[i]);
    }
  }

  // Find output node and return result
  return nodeOut_[nNodes_-1];
}


bool SprStdBackprop::backward(int cls, double output, 
			      const std::vector<double>& etaV)
{
  // make temp copies
  vector<double> tempLinkWeight(linkWeight_);
  vector<double> tempNodeBias(nodeBias_);

  // reset gradients
  vector<double> nodeGradient(nNodes_,0);

  // gradient in the output node
  nodeGradient[nNodes_-1] = (double(cls)-output) *
    this->act_deriv(nodeAct_[nNodes_-1]+nodeBias_[nNodes_-1],
		    nodeActFun_[nNodes_-1]);
  nodeBias_[nNodes_-1] += etaV[nLinks_] * nodeGradient[nNodes_-1];

  // propagate backwards thru hidden nodes
  for( unsigned int target=nNodes_-1;target>=0;target-- ) {
    if( nodeNInputLinks_[target] > 0 ) {
      for( int link=nodeFirstInputLink_[target];
	   link<nodeFirstInputLink_[target]+nodeNInputLinks_[target];
	   link++ ) {
	int source = linkSource_[link];
	linkWeight_[link] += etaV[link] 
	  * nodeGradient[target] * nodeOut_[source];
	if( nodeType_[source] == SprNNDefs::HIDDEN ) {
	  nodeGradient[source] += 
	    this->act_deriv(nodeAct_[source]+tempNodeBias[source],
			    nodeActFun_[source]) 
	    * tempLinkWeight[link] * nodeGradient[target];
	  nodeBias_[source] += etaV[link] * nodeGradient[source];
	}
      }
    }
  }

  // exit
  return true;
}


bool SprStdBackprop::reset()
{
  initialized_ = false;
  nodeBias_.clear(); nodeBias_.resize(nNodes_,0);
  nodeAct_.clear(); nodeAct_.resize(nNodes_,0);
  nodeOut_.clear(); nodeOut_.resize(nNodes_,0);
  linkWeight_.clear(); linkWeight_.resize(nLinks_,0);
  return true;
}


bool SprStdBackprop::setData(SprAbsFilter* data)
{
  assert( data != 0 );
  data_ = data;
  return this->reset();
}


void SprStdBackprop::print(std::ostream& os) const 
{
  os << "Trained StdBackprop with configuration " 
     << structure_.c_str() << " " << SprVersion << endl; 
  os << "Activation functions: Identity=1, Logistic=2" << endl;
  os << "Cut: " << cut_.size();
  for( unsigned int i=0;i<cut_.size();i++ )
    os << "      " << cut_[i].first << " " << cut_[i].second;
  os << endl;
  os << "Nodes: " << nNodes_ << endl;
  for( int i=0;i<nNodes_;i++ ) {
    char nodeType = 0;
    switch( nodeType_[i] )
      {
      case SprNNDefs::INPUT :
	nodeType = 'I';
	break;
      case SprNNDefs::HIDDEN :
	nodeType = 'H';
	break;
      case SprNNDefs::OUTPUT :
	nodeType = 'O';
	break;
      }
    int actFun = 0;
    switch( nodeActFun_[i] )
      {
      case SprNNDefs::ID :
	actFun = 1;
	break;
      case SprNNDefs::LOGISTIC :
	actFun = 2;
	break;
      }
    os << setw(6) << i
       << "    Type: "           << nodeType
       << "    ActFunction: "    << actFun
       << "    NInputLinks: "    << setw(6) << nodeNInputLinks_[i]
       << "    FirstInputLink: " << setw(6) << nodeFirstInputLink_[i]
       << "    Bias: "           << nodeBias_[i]
       << endl;
  }
  os << "Links: " << nLinks_ << endl;
  for( int i=0;i<nLinks_;i++ ) {
    os << setw(6) << i
       << "    Source: " << setw(6) << linkSource_[i]
       << "    Weight: " << linkWeight_[i]
       << endl;
  }
}


void SprStdBackprop::setClasses()
{
  vector<SprClass> classes;
  data_->classes(classes);
  int size = classes.size();
  if( size > 0 ) cls0_ = classes[0];
  if( size > 1 ) cls1_ = classes[1];
  cout << "Classes for StdBackprop are set to " 
       << cls0_ << " " << cls1_ << endl;
}


bool SprStdBackprop::setValidation(const SprAbsFilter* valData, 
				   unsigned valPrint,
				   SprAverageLoss* loss)
{
  // set
  valData_ = valData;
  valPrint_ = valPrint;

  // if no loss specified, use quadratic by default
  loss_ = loss;
  ownLoss_ = false;
  if( loss_ == 0 ) {
    loss_ = new SprAverageLoss(&SprLoss::quadratic);
    ownLoss_ = true;
  }

  // exit
  return true;
}


bool SprStdBackprop::printValidation(unsigned cycle)
{
  // reset loss
  assert( loss_ != 0 );
  loss_->reset();

  // make trained NN
  SprTrainedStdBackprop* t = this->makeTrained();

  // loop through validation data
  for( unsigned int i=0;i<valData_->size();i++ ) {
    const SprPoint* p = (*valData_)[i];
    double r = t->response(p->x_);
    double w = valData_->w(i);
    if( p->class_!=cls0_ && p->class_!=cls1_ ) w = 0;
    if(      p->class_ == cls0_ )
      loss_->update(0,r,w);
    else if( p->class_ == cls1_ )
      loss_->update(1,r,w);
  }

  // compute fom
  cout << "Validation Loss=" << loss_->value()
       << " at cycle " << cycle << endl;

  // exit
  return true;
}


double SprStdBackprop::activate(double x, SprNNDefs::ActFun f) const 
{
  switch (f) 
    {
    case SprNNDefs::ID :
      return x;
      break;
    case SprNNDefs::LOGISTIC :
      return SprTransformation::logit(x);
      break;
    default :
      cerr << "Unknown activation function " 
	   << f << " in SprTrainedStdBackprop::activate" << endl;
      return 0;
    }
  return 0;
}


double SprStdBackprop::act_deriv(double x, SprNNDefs::ActFun f) const 
{
  switch (f) 
    {
    case SprNNDefs::ID :
      return 1;
      break;
    case SprNNDefs::LOGISTIC :
      return SprTransformation::logit_deriv(x);
      break;
    default :
      cerr << "Unknown activation function " 
	   << f << " in SprTrainedStdBackprop::activate" << endl;
      return 0;
    }
  return 0;
}


bool SprStdBackprop::prepareExit(bool status)
{
  data_->setWeights(initialDataWeights_);
  return status;
}


bool SprStdBackprop::readSNNS(const char* netfile) 
{
  // sanity check and init
  if( 0 == netfile ) return false;
  structure_ = "Unknown";
  configured_ = false;
  initialized_ = false;
  string nfile = netfile;
  bool success = false;

  // open file
  ifstream file(nfile.c_str());
  if( !file ) {
    cerr << "Unable to open file " << nfile.c_str() << endl;
    return false;
  }

  // Read header of network definition file
  string line;
  unsigned nLine = 0;
  nLine++;
  nNodes_ = 0;
  while( getline(file,line) ) {
    const char* searchfor = "no. of units :";
    size_t pos = line.find(searchfor);
    if( pos != string::npos ) {
      line.erase(0,pos+strlen(searchfor)+1);
      istringstream istnodes(line);
      istnodes >> nNodes_;
      break;
    }
    nLine++;
  }
  if( nNodes_ <= 0 ) {
    cerr << "Can't find units line in file " << nfile.c_str() << endl;
    return false;
  }
  nLine++;
  if( !getline(file,line) ) {
    cerr << "Cannot read from " << nfile.c_str() << " line " << nLine << endl;
    return false;
  }
  nLinks_ = 0;
  const char* searchfor = "no. of connections :";
  size_t pos = line.find(searchfor);
  if( pos != string::npos ) {
    line.erase(0,pos+strlen(searchfor)+1);
    istringstream istconns(line);
    istconns >> nLinks_;
  }
  if( nLinks_ <= 0 ) {
    cerr << "Can't find connections line in file " << nfile.c_str() << endl;
    return false;
  }
  //  cout << "Nodes and links: " << nNodes_ << " " << nLinks_ << endl;

  // Allocate space for node and link data
  nodeType_.clear(); nodeType_.resize(nNodes_,SprNNDefs::INPUT);
  nodeActFun_.clear(); nodeActFun_.resize(nNodes_,SprNNDefs::ID);
  nodeAct_.clear(); nodeAct_.resize(nNodes_,0);
  nodeOut_.clear(); nodeOut_.resize(nNodes_,0);
  nodeNInputLinks_.clear(); nodeNInputLinks_.resize(nNodes_,0);
  nodeFirstInputLink_.clear(); nodeFirstInputLink_.resize(nNodes_,-1);
  nodeBias_.clear(); nodeBias_.resize(nNodes_,0);
  linkSource_.clear(); linkSource_.resize(nLinks_,0);
  linkWeight_.clear(); linkWeight_.resize(nLinks_,0);
    
  // Here we should check that we are reading the correct type of network,
  // i.e. one using the Act_Logistic activation function ...

  //
  // Read node information
  //
  nLine++;
  bool found = false;
  while( getline(file,line) ) {
    size_t pos = line.find("unit definition section :");
    if( pos != string::npos ) {
      found = true;
      break;
    }
    nLine++;
  }
  if( !found ) {
    cerr << "Can't find unit definition section in file " 
	 << nfile.c_str() << endl;
    return false;
  }
  // skip 3 lines
  for( unsigned int i=0;i<3;i++ ) {
    nLine++;
    if( !getline(file,line) ) {
      cerr << "Cannot read from " << nfile.c_str() 
	   << " line " << nLine << endl;
      return false;
    }
  }
  // read nodes one by one
  unsigned nOutput = 0;
  for( int node=0;node<nNodes_;node++ ) {
    nLine++;
    if( !getline(file,line) ) {
      cerr << "Cannot read from " << nfile.c_str() 
	   << " line " << nLine << endl;
      return false;
    }
    istringstream istnode(line);
    int id = 0;
    istnode >> id;
    if( id != (node+1) ) {
      cerr << "Node ID does not match on line " << nLine << endl;
      return false;
    }
    char c;
    double dummy;
    for( int i=0;i<3;i++ ) istnode >> c;
    istnode >> dummy >> c >> nodeBias_[node] >> c;
    istnode >> c;
    switch( c ) 
      {
      case 'i' :
	nodeType_[node] = SprNNDefs::INPUT;
	nodeActFun_[node] = SprNNDefs::ID;
	break;
      case 'h' :
	nodeType_[node] = SprNNDefs::HIDDEN;
	nodeActFun_[node] = SprNNDefs::LOGISTIC;
	break;
      case 'o' :
	nodeType_[node] = SprNNDefs::OUTPUT;
	nodeActFun_[node] = SprNNDefs::LOGISTIC;
	nOutput++;
	break;
      default :
	cerr << "Unknown node type on line " << nLine << endl;
	return false;
      }
  }
  if( nOutput > 1 ) {
    cerr << "More than one output node cannot be handled "
	 << "by this implementation" << endl;
    return false;
  }
  //  cout << "Unit definition section has been read " << nLine << endl;

  //
  // Read link information
  //
  nLine++;
  found = false;
  while( getline(file,line) ) {
    size_t pos = line.find("connection definition section :");
    if( pos != string::npos ) {
      found = true;
      break;
    }
    nLine++;
  }
  if( !found ) {
    cerr << "Can't find connection definition section in file " 
	 << nfile.c_str() << endl;
    return false;
  }
  // skip 3 lines
  for( unsigned int i=0;i<3;i++ ) {
    nLine++;
    if( !getline(file,line) ) {
      cerr << "Cannot read from " << nfile.c_str() 
	   << " line " << nLine << endl;
      return false;
    }
  }
  // read links one by one
  int link = 0;
  string prevLine;
  while( getline(file,line) ) {
    nLine++;
    // if the last symbol is comma, continue to next line
    if( line.at(line.find_last_not_of(' ')) == ',' ) {
      prevLine = line;
      continue;
    }
    line = prevLine+line;
    prevLine = "";
    // get target
    size_t separ_pos = line.find_first_of('|');
    if( separ_pos == string::npos ) {
      cerr << "Cannot read from " << nfile.c_str() 
	   << " line " << nLine << endl;
      return false;
    }
    string target_str = line.substr(0,separ_pos);
    line.erase(0,separ_pos+1);
    int target = atoi(target_str.c_str());
    if( target<=0 || target>nNodes_ ) {
      cerr << "Unable to read target node from "
	   << nfile.c_str() << " on line " << nLine 
	   << " : nNodes=" << nNodes_ << " target=" << target << endl;
      return false;
    }
    target--;// offset by 1 to start numbering from 0 instead of 1
    // assign first link for the target
    nodeFirstInputLink_[target] = link;
    // skip one field
    separ_pos = line.find_first_of('|');
    if( separ_pos == string::npos ) {
      cerr << "Cannot read from " << nfile.c_str() 
	   << " line " << nLine << endl;
      return false;
    }
    // get source
    string sources_str = line.substr(separ_pos+1);
    vector<string> sources;
    while( sources_str.find(',') != string::npos ) {
      size_t comma_pos = sources_str.find_first_of(',');
      sources.push_back(sources_str.substr(0,comma_pos));
      sources_str.erase(0,comma_pos+1);
    }
    sources.push_back(sources_str);// leftover to get the last source
    for( unsigned int i=0;i<sources.size();i++ ) {
      string current_source = sources[i];
      size_t doubledot_pos = current_source.find(':');
      if( doubledot_pos == string::npos ) {
	cerr << "Cannot read from " << nfile.c_str() 
	     << " line " << nLine << endl;
	return false;
      }
      string source_id = current_source.substr(0,doubledot_pos);
      string source_weight = current_source.substr(doubledot_pos+1);
      int source = atoi(source_id.c_str());
      double weight = atof(source_weight.c_str());
      if( source<=0 || source>nNodes_ ) {
	cerr << "Unable to read source node from "
	     << nfile.c_str() << " on line " << nLine << endl;
	return false;
      }
      source--;// offset by 1 to start numbering from 0 instead of 1
      // build link
      linkSource_[link] = source;
      linkWeight_[link] = weight;
      nodeNInputLinks_[target]++;
      // increment link
      link++;
    }
    if( link == nLinks_ ) {
      success = true;
      break;
    }
  }

  // exit
  if( success ) {
    configured_ = true;
    initialized_ = true;
  }
  return success;
}


bool SprStdBackprop::readSPR(const char* netfile)
{
  // sanity check and init
  if( 0 == netfile ) return false;
  string nfile = netfile;

  // open file
  ifstream file(nfile.c_str());
  if( !file ) {
    cerr << "Unable to open file " << nfile.c_str() << endl;
    return false;
  }

  // read the file
  unsigned skipLines = 0;
  return this->resumeReadSPR(nfile.c_str(),file,skipLines);
}

bool SprStdBackprop::resumeReadSPR(const char* netfile,
				   std::ifstream& file, 
				   unsigned& skipLines)
{
  // init
  unsigned& nLine = skipLines;
  structure_ = "Unknown";
  configured_ = false;
  initialized_ = false;
  string nfile = netfile;

  // read header
  string line;
  for( unsigned int i=0;i<2;i++ ) {
    nLine++;
    if( !getline(file,line) ) {
      cerr << "Unable to read line " << nLine 
	   << " from " << nfile.c_str() << endl;
      return false;
    }
  }

  // read the cut
  string dummy;
  nLine++;
  if( !getline(file,line) ) {
    cerr << "Unable to read line " << nLine 
	 << " from " << nfile.c_str() << endl;
    return false;
  }
  istringstream istcut(line);
  istcut >> dummy;
  int nCut = 0;
  istcut >> nCut;
  cut_.clear();
  double low(0), high(0);
  for( int i=0;i<nCut;i++ ) {
    istcut >> low >> high;
    cut_.push_back(SprInterval(low,high));
  }

  // read number of nodes
  nLine++;
  if( !getline(file,line) ) {
    cerr << "Unable to read line " << nLine 
	 << " from " << nfile.c_str() << endl;
    return false;
  }
  istringstream istNnodes(line);
  istNnodes >> dummy >> nNodes_;
  if( nNodes_ <= 0 ) {
    cerr << "Rean an invalid number of NN nodes: " << nNodes_ << endl;
    return false;
  }
  
  // init nodes
  nodeType_.clear(); nodeType_.resize(nNodes_,SprNNDefs::INPUT);
  nodeActFun_.clear(); nodeActFun_.resize(nNodes_,SprNNDefs::ID);
  nodeAct_.clear(); nodeAct_.resize(nNodes_,0);
  nodeOut_.clear(); nodeOut_.resize(nNodes_,0);
  nodeNInputLinks_.clear(); nodeNInputLinks_.resize(nNodes_,0);
  nodeFirstInputLink_.clear(); nodeFirstInputLink_.resize(nNodes_,-1);
  nodeBias_.clear(); nodeBias_.resize(nNodes_,0);

  // read nodes
  for( int node=0;node<nNodes_;node++ ) {
    nLine++;
    if( !getline(file,line) ) {
      cerr << "Unable to read line " << nLine 
	   << " from " << nfile.c_str() << endl;
      return false;
    }
    istringstream istnode(line);
    int index = -1;
    istnode >> index;
    if( index != node ) {
      cerr << "Incorrect node number on line " << nLine
	   << ": Expect " << node << " Actual " << index << endl;
      return false;
    }
    istnode >> dummy;
    char nodeType;
    istnode >> nodeType;
    switch( nodeType )
      {
      case 'I' :
	nodeType_[node] = SprNNDefs::INPUT;
	break;
      case 'H' :
	nodeType_[node] = SprNNDefs::HIDDEN;
	break;
      case 'O' :
	nodeType_[node] = SprNNDefs::OUTPUT;
	break;
      default :
	cerr << "Unknown node type on line " << nLine 
	     << " in " << nfile.c_str() << endl;
	return false;
      }
    istnode >> dummy;
    int actFun = 0;
    istnode >> actFun;
    switch( actFun )
      {
      case 1 :
	nodeActFun_[node] = SprNNDefs::ID;
	break;
      case 2 :
	nodeActFun_[node] = SprNNDefs::LOGISTIC;
	break;
      default :
	cerr << "Unknown activation function on line " << nLine
	     << " in " << nfile.c_str() << endl;
	return false;
      }
    istnode >> dummy;
    istnode >> nodeNInputLinks_[node];
    istnode >> dummy;
    istnode >> nodeFirstInputLink_[node];
    istnode >> dummy;
    istnode >> nodeBias_[node];
  }// nodes done

  // read number of nodes
  nLine++;
  if( !getline(file,line) ) {
    cerr << "Unable to read line " << nLine 
	 << " from " << nfile.c_str() << endl;
    return false;
  }
  istringstream istNlinks(line);
  istNlinks >> dummy >> nLinks_;
  if( nLinks_ <= 0 ) {
    cerr << "Rean an invalid number of NN links: " << nLinks_ << endl;
    return false;
  }
  
  // init links
  linkSource_.clear(); linkSource_.resize(nLinks_,0);
  linkWeight_.clear(); linkWeight_.resize(nLinks_,0);

  // read links
  for( int link=0;link<nLinks_;link++ ) {
    nLine++;
    if( !getline(file,line) ) {
      cerr << "Unable to read line " << nLine 
	   << " from " << nfile.c_str() << endl;
      return false;
    }
    istringstream istlink(line);
    int index = -1;
    istlink >> index;
    if( index != link ) {
      cerr << "Incorrect link number on line " << nLine
	   << ": Expect " << link << " Actual " << index << endl;
      return false;
    }
    istlink >> dummy;
    istlink >> linkSource_[link];
    istlink >> dummy;
    istlink >> linkWeight_[link];
  }// links done

  // exit
  configured_ = true;
  initialized_ = true;
  return true;
}
