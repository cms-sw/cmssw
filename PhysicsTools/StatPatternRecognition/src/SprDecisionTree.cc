//$Id: SprDecisionTree.cc,v 1.2 2007/09/21 22:32:10 narsky Exp $

#include "PhysicsTools/StatPatternRecognition/interface/SprExperiment.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprDecisionTree.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAbsTwoClassCriterion.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTreeNode.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprUtils.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprDefs.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprIntegerBootstrap.hh"

#include <stdio.h>
#include <functional>
#include <algorithm>
#include <cassert>

using namespace std;


struct SDTCmpPairFirst
  : public binary_function<pair<double,const SprTreeNode*>,
			   pair<double,const SprTreeNode*>,
			   bool> {
  bool operator()(const pair<double,const SprTreeNode*>& l, 
		  const pair<double,const SprTreeNode*>& r)
    const {
    return (l.first < r.first);
  }
};
 

SprDecisionTree::~SprDecisionTree()
{
  delete root_;
}


SprDecisionTree::SprDecisionTree(SprAbsFilter* data, 
				 const SprAbsTwoClassCriterion* crit,
				 int nmin, bool doMerge, bool discrete,
				 SprIntegerBootstrap* bootstrap)
  :
  SprAbsClassifier(data),
  cls0_(0),
  cls1_(1),
  crit_(crit),
  nmin_(nmin),
  doMerge_(doMerge),
  discrete_(discrete),
  canHavePureNodes_(true),
  fastSort_(false),
  showBackgroundNodes_(false),
  bootstrap_(bootstrap),
  root_(0),
  nodes1_(),
  nodes0_(),
  fullNodeList_(),
  fom_(0),
  w0_(0),
  w1_(0),
  n0_(0),
  n1_(0),
  splits_()
{
  // check nmin
  if( nmin_ <= 0 ) {
    cout << "Resetting minimal number of events per node to 1." << endl;
    nmin_ = 1;
  }
  cout << "Decision tree initialized mith minimal number of events per node "
       << nmin_ << endl;

  // check bootstrap
  if( bootstrap_ != 0 ) {
    cout << "Decision tree will resample at most " 
	 << bootstrap->nsample() << " features." << endl;
  }

  // check discrete
  if( doMerge_ && !discrete_ ) {
    discrete_ = true;
    cout << "Warning: continuous output is not allowed for trees with "
	 << "merged terminal nodes." << endl;
    cout << "Switching to discrete (0/1) tree output." << endl;
  }

  // make root
  root_ = new SprTreeNode(crit,data,doMerge,nmin_,discrete_,
			  canHavePureNodes_,fastSort_,bootstrap_);

  // set classes
  this->setClasses();
  bool status = root_->setClasses(cls0_,cls1_);
  assert ( status );
}


void SprDecisionTree::setClasses() 
{
  vector<SprClass> classes;
  data_->classes(classes);
  int size = classes.size();
  if( size > 0 ) cls0_ = classes[0];
  if( size > 1 ) cls1_ = classes[1];
  //  cout << "Classes for decision tree are set to " 
  //       << cls0_ << " " << cls1_ << endl;
}


SprTrainedDecisionTree* SprDecisionTree::makeTrained() const 
{
  // prepare vectors of accepted regions
  vector<SprBox> nodes1(nodes1_.size());

  // copy box limits
  for( unsigned int i=0;i<nodes1_.size();i++ )
    nodes1[i] = nodes1_[i]->limits_;

  // make tree
  SprTrainedDecisionTree* t =  new SprTrainedDecisionTree(nodes1);

  // vars
  vector<string> vars;
  data_->vars(vars);
  t->setVars(vars);

  // exit
  return t;
}


const SprTreeNode* SprDecisionTree::next(const SprTreeNode* node) const
{
  // travel up
  const SprTreeNode* temp = node;
  while( temp->parent_!=0 && temp->parent_->right_==temp )
    temp = temp->parent_;

  // if root, exit
  if( temp->parent_ == 0 ) return 0;

  // go over the hill
  temp = temp->parent_->right_;

  // travel down
  while( temp->left_ != 0 )
    temp = temp->left_;

  // exit
  return temp;
}


const SprTreeNode* SprDecisionTree::first() const
{
  const SprTreeNode* temp = root_;
  while( temp->left_ != 0 )
    temp = temp->left_;
  return temp;
}


bool SprDecisionTree::train(int verbose)
{
  // train the tree
  fullNodeList_.clear();
  fullNodeList_.push_back(root_);
  unsigned int splitIndex = 0;
  while( splitIndex < fullNodeList_.size() ) {
    SprTreeNode* node = fullNodeList_[splitIndex];
    if( !node->split(fullNodeList_,splits_,verbose) ) {
      cerr << "Unable to split node with index " << splitIndex << endl;
      return false;
    }
    splitIndex++;
  }

  // merge
  if( !this->merge(1,doMerge_,nodes1_,fom_,w0_,w1_,n0_,n1_,verbose) ) {
    cerr << "Unable to merge signal nodes." << endl;
    return false;
  }
  if( doMerge_ ) showBackgroundNodes_ = false;
  if( showBackgroundNodes_ ) {
    double fom(0), w0(0), w1(0);
    unsigned n0(0), n1(0);
    if( !this->merge(0,false,nodes0_,fom,w0,w1,n0,n1,verbose) ) {
      cerr << "Unable to merge background nodes." << endl;
      return false;
    }
    // show overall FOM
    double totFom = crit_->fom(w0,w0_,w1_,w1);
    if( verbose > 0 ) {
      cout << "Included " << nodes1_.size()+nodes0_.size() 
	   << " nodes with overall FOM=" << totFom << endl;
    }
  }

  // exit
  return true;
}


bool SprDecisionTree::reset()
{
  delete root_;
  root_ = new SprTreeNode(crit_,data_,doMerge_,nmin_,discrete_,
			  canHavePureNodes_,fastSort_,bootstrap_);
  if( !root_->setClasses(cls0_,cls1_) ) return false;
  nodes1_.clear();
  nodes0_.clear();
  fullNodeList_.clear();
  w0_ = 0; w1_ = 0;
  n0_ = 0; n1_ = 0;
  fom_ = SprUtils::min();
  return true;
}


bool SprDecisionTree::setData(SprAbsFilter* data)
{
  assert( data != 0 );
  data_ = data;
  return this->reset();
}


bool SprDecisionTree::merge(int category, bool doMerge,
			    std::vector<const SprTreeNode*>& nodes,
			    double& fomtot, double& w0tot, double& w1tot,
			    unsigned& n0tot, unsigned& n1tot, int verbose) 
  const
{
  // find leaf nodes
  vector<const SprTreeNode*> collect;
  const SprTreeNode* temp = this->first();
  while( temp != 0 ) {
    if( temp->nodeClass() == category )
      collect.push_back(temp);
    temp = this->next(temp);
  }
  if( collect.empty() ) {
    if( verbose > 0 )
      cerr << "No leaf nodes found for category " << category << endl;
    return true;
  }
  unsigned int size = collect.size();
  if( verbose > 1 ) {
    cout << "Found " << size << " leaf nodes in category " 
	 << category << ":     ";
    for( unsigned int i=0;i<size;i++ )
      cout << collect[i]->id() << " ";
    cout << endl;
  }

  // sort leaf nodes by purity
  vector<pair<double,const SprTreeNode*> > purity(size);
  for( unsigned int i=0;i<size;i++ ) {
    const SprTreeNode* node = collect[i];
    double w0 = node->w0();
    double w1 = node->w1();
    if( (w1+w0) < SprUtils::eps() ) {
      cerr << "Found a node without events: " << node->id() << endl;
      return false;
    }
    if(      category == 1 )
      purity[i] = pair<double,const SprTreeNode*>(w1/(w1+w0),node);
    else if( category == 0 )
      purity[i] = pair<double,const SprTreeNode*>(w0/(w1+w0),node);
  }
  stable_sort(purity.begin(),purity.end(),not2(SDTCmpPairFirst()));
  for( unsigned int i=0;i<size;i++ ) {
    collect[i] = purity[i].second;
  }
  if( verbose > 1 ) {
    cout << "Nodes sorted by purity: " << endl;
    for( unsigned int i=0;i<size;i++ )
      cout << collect[i]->id() << " ";
    cout << endl;
  }

  // add nodes in the order of decreasing purity
  vector<double> fomVec(size), w0Vec(size), w1Vec(size);
  vector<unsigned> n0Vec(size), n1Vec(size);
  double w0(0), w1(0);
  unsigned n0(0), n1(0);
  for( unsigned int j=0;j<size;j++ ) {
    const SprTreeNode* node = collect[j];
    double w0add = node->w0();
    double w1add = node->w1();
    w0 += w0add;
    w1 += w1add;
    n0 += node->n0();
    n1 += node->n1();
    double fom = 0;
    if(      category == 1 )
      fom = crit_->fom(0,w0,w1,0);
    else if( category == 0 ) 
      fom = crit_->fom(w0,0,0,w1);
    fomVec[j] = fom;
    w0Vec[j] = w0;
    w1Vec[j] = w1;
    n0Vec[j] = n0;
    n1Vec[j] = n1;
    if( verbose > 1 ) {
      cout << "Adding node " << node->id() 
	   << " with " << w0add << " background and "
	   << w1add << " signal weights at overall FOM=" << fom 
	   << endl;
    }
  }

  // find the combination with largest FOM
  int best = size-1;
  if( doMerge ) {
    // if nodes have equal FOM's, prefer those with more events
    vector<double>::reverse_iterator iter 
      = max_element(fomVec.rbegin(),fomVec.rend());
    best = iter - fomVec.rbegin();
    best = size-1 - best;
  }
  double fom0 = fomVec[best];
  w0 = w0Vec[best];
  w1 = w1Vec[best];
  n0 = n0Vec[best];
  n1 = n1Vec[best];
  nodes.clear();
  for( int i=0;i<=best;i++ ) {
    nodes.push_back(collect[i]);
  }

  // message
  if( verbose > 0 ) {
    cout << "Included " << nodes.size() 
	 << " nodes in category " << category
	 << " with overall FOM=" << fom0 
	 << "    W1=" << w1 << " W0=" << w0 
	 << "    N1=" << n1 << " N0=" << n0 << endl;
  }
  if( verbose > 1 ) {
    cout << "Node list: ";
    for( unsigned int i=0;i<nodes.size();i++ ) cout << nodes[i]->id() << " ";
    cout << endl;
  }

  // assign FOM and weights
  fomtot = fom0;
  w0tot = w0;
  w1tot = w1;
  n0tot = n0;
  n1tot = n1;

  // exit
  return true;
}


void SprDecisionTree::print(std::ostream& os) const
{
  // header
  char s [200];
  sprintf(s,"Trained DecisionTree %-6i signal nodes.    Overall FOM=%-10g W0=%-10g W1=%-10g N0=%-10i N1=%-10i    Version=%s",nodes1_.size(),fom_,w0_,w1_,n0_,n1_,SprVersion.c_str());
  os << s << endl;
  os << "-------------------------------------------------------" << endl;

  // get vars
  vector<string> vars;
  data_->vars(vars);

  // signal nodes
  os << "-------------------------------------------------------" << endl;
  os << "Signal nodes:" << endl;
  os << "-------------------------------------------------------" << endl;
  for( unsigned int i=0;i<nodes1_.size();i++ ) {
    const SprBox& limits = nodes1_[i]->limits_;
    int size = limits.size();
    char s [200];
    sprintf(s,"Node %6i    Size %-4i    FOM=%-10g W0=%-10g W1=%-10g N0=%-10i N1=%-10i",i,size,nodes1_[i]->fom(),nodes1_[i]->w0(),nodes1_[i]->w1(),nodes1_[i]->n0(),nodes1_[i]->n1());
    os << s << endl;
    for( SprBox::const_iterator iter = 
	   limits.begin();iter!=limits.end();iter++ ) {
      unsigned d = iter->first;
      assert( d < vars.size() );
      char s [200];
      sprintf(s,"Variable %30s    Limits  %15g %15g",
	      vars[d].c_str(),iter->second.first,iter->second.second);
      os << s << endl;
    }
    os << "-------------------------------------------------------" << endl;
  }

  // background nodes
  if( showBackgroundNodes_ ) {
    os << "-------------------------------------------------------" << endl;
    os << "Background nodes:" << endl;
    os << "-------------------------------------------------------" << endl;
    for( unsigned int i=0;i<nodes0_.size();i++ ) {
      const SprBox& limits = nodes0_[i]->limits_;
      int size = limits.size();
      char s [200];
      sprintf(s,"Node %6i    Size %-4i    FOM=%-10g W0=%-10g W1=%-10g N0=%-10i N1=%-10i",i,size,nodes0_[i]->fom(),nodes0_[i]->w0(),nodes0_[i]->w1(),nodes0_[i]->n0(),nodes0_[i]->n1());
      os << s << endl;
      for( SprBox::const_iterator iter = 
	     limits.begin();iter!=limits.end();iter++ ) {
	unsigned d = iter->first;
	assert( d < vars.size() );
	char s [200];
	sprintf(s,"Variable %30s    Limits  %15g %15g",
		vars[d].c_str(),iter->second.first,iter->second.second);
	os << s << endl;
      }
      os << "-------------------------------------------------------" << endl;
    }
  }
}


void SprDecisionTree::startSplitCounter()
{
  splits_.clear();
  splits_.resize(data_->dim(),pair<int,double>(0,0));
}


void SprDecisionTree::printSplitCounter(std::ostream& os) const
{
  unsigned dim = data_->dim();
  assert( splits_.size() == dim );
  vector<string> vars;
  data_->vars(vars);
  assert( vars.size() == dim );
  os << "Tree splits on variables:" << endl;
  for( unsigned int i=0;i<dim;i++ ) {
    char s [200];
    sprintf(s,"Variable %30s    Splits  %10i    Delta FOM  %10.5f",
	    vars[i].c_str(),splits_[i].first,splits_[i].second);
    os << s << endl;
  }
}


bool SprDecisionTree::setClasses(const SprClass& cls0, const SprClass& cls1) 
{
  cls0_ = cls0;
  cls1_ = cls1;
  if( root_ != 0 ) 
    return root_->setClasses(cls0,cls1);
  return true;
}
