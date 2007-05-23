//$Id: SprTopdownTree.cc,v 1.4 2007/02/05 21:49:46 narsky Exp $

#include "PhysicsTools/StatPatternRecognition/interface/SprExperiment.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTopdownTree.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTreeNode.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTrainedNode.hh"

#include <iostream>
#include <map>
#include <utility>
#include <cassert>

using namespace std;


SprTopdownTree::SprTopdownTree(SprAbsFilter* data, 
			       const SprAbsTwoClassCriterion* crit,
			       int nmin, bool doMerge, bool discrete,
			       SprIntegerBootstrap* bootstrap)
  :
  SprDecisionTree(data,crit,nmin,doMerge,discrete,bootstrap)
{
  cout << "Using a Topdown tree." << endl;
}


SprTrainedTopdownTree* SprTopdownTree::makeTrained() const
{
  vector<const SprTrainedNode*> nodes;
  if( !this->makeTrainedNodes(nodes) ) {
    cerr << "SprTrainedTopdownTree unable to make trained nodes." << endl;
    return 0;
  }
  return new SprTrainedTopdownTree(nodes,true);
}


bool SprTopdownTree::makeTrainedNodes(std::vector<const SprTrainedNode*>& 
				      nodes) const
{
  // sanity check
  if( fullNodeList_.empty() || root_->id_!=0 || fullNodeList_[0]!=root_ ) {
    cerr << "Tree is not properly configured. Unable to make trained nodes." 
	 << endl;
    return false;
  }

  // copy all nodes into the map
  map<int,SprTrainedNode*> copy;
  for( int i=0;i<fullNodeList_.size();i++ ) {
    SprTrainedNode* node = fullNodeList_[i]->makeTrained();
    copy.insert(pair<const int,SprTrainedNode*>(node->id_,node));
  }

  // make sure the first node has id 0
  if( copy.begin()->first != 0 ) {
    cerr << "First id in the replicated map is not zero." << endl;
    return false;
  }

  // resolve mother/daughter references
  for( int i=0;i<fullNodeList_.size();i++ ) {
    const SprTreeNode* old = fullNodeList_[i];
    map<int,SprTrainedNode*>::iterator iter = copy.find(old->id_);
    assert( iter != copy.end() );
    if( old->left_ != 0 ) {
      map<int,SprTrainedNode*>::iterator dau1 = copy.find(old->left_->id_);
      assert( dau1 != copy.end() );
      iter->second->toDau1_ = dau1->second;
      dau1->second->toParent_ = iter->second;
    }
    if( old->right_ != 0 ) {
      map<int,SprTrainedNode*>::iterator dau2 = copy.find(old->right_->id_);
      assert( dau2 != copy.end() );
      iter->second->toDau2_ = dau2->second;
      dau2->second->toParent_ = iter->second;
    }
  }

  // convert the map into a plain vector
  nodes.clear();
  for( map<int,SprTrainedNode*>::iterator iter = copy.begin();
       iter!=copy.end();iter++ ) {
    nodes.push_back(iter->second);
  }

  // exit
  return true;
}
