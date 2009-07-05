//$Id: SprTrainedTopdownTree.cc,v 1.2 2007/09/21 22:32:10 narsky Exp $

#include "PhysicsTools/StatPatternRecognition/interface/SprExperiment.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTrainedTopdownTree.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTrainedNode.hh"

#include <map>
#include <utility>

using namespace std;


SprTrainedTopdownTree::~SprTrainedTopdownTree()
{
  if( ownTree_ ) {
    for( unsigned int i=0;i<nodes_.size();i++ ) delete nodes_[i];
    ownTree_ = false;
  }
}


double SprTrainedTopdownTree::response(const std::vector<double>& v) const
{
  const SprTrainedNode* node = nodes_[0];
  while( node->d_ >= 0 ) {
    assert( node->d_ < static_cast<int>(v.size()) );
    if( v[node->d_] < node->cut_ )
      node = node->toDau1_;
    else
      node = node->toDau2_;
  }
  return node->score_;
}


void SprTrainedTopdownTree::print(std::ostream& os) const
{
  os << "Trained TopdownTree " << SprVersion << endl;
  os << "Nodes: " << nodes_.size() << " nodes." << endl;
  for( unsigned int i=0;i<nodes_.size();i++ ) {
    const SprTrainedNode* node = nodes_[i];
    os << "Id: "         << node->id_
       << " Score: "     << node->score_
       << " Dim: "       << node->d_
       << " Cut: "       << node->cut_
       << " Daughters: " << (node->toDau1_==0 ? -1 : node->toDau1_->id_)
       << " "            << (node->toDau2_==0 ? -1 : node->toDau2_->id_)
       << endl;
  }
}


bool SprTrainedTopdownTree::replicate(const std::vector<
				      const SprTrainedNode*>& nodes)
{
  // copy all nodes into the map
  map<int,SprTrainedNode*> copy;
  for( unsigned int i=0;i<nodes.size();i++ ) {
    SprTrainedNode* node = new SprTrainedNode(*nodes[i]);
    copy.insert(pair<const int,SprTrainedNode*>(node->id_,node));
  }

  // make sure the first node has id 0
  if( copy.begin()->first != 0 ) {
    cerr << "First id in the replicated map is not zero." << endl;
    return false;
  }

  // resolve mother/daughter references
  for( unsigned int i=0;i<nodes.size();i++ ) {
    const SprTrainedNode* old = nodes[i];
    map<int,SprTrainedNode*>::iterator iter = copy.find(old->id_);
    assert( iter != copy.end() );
    if( old->toDau1_ != 0 ) {
      map<int,SprTrainedNode*>::iterator dau1 = copy.find(old->toDau1_->id_);
      assert( dau1 != copy.end() );
      iter->second->toDau1_ = dau1->second;
      dau1->second->toParent_ = iter->second;
    }
    if( old->toDau2_ != 0 ) {
      map<int,SprTrainedNode*>::iterator dau2 = copy.find(old->toDau2_->id_);
      assert( dau2 != copy.end() );
      iter->second->toDau2_ = dau2->second;
      dau2->second->toParent_ = iter->second;
    }
  }

  // convert the map into a plain vector
  nodes_.clear();
  for( map<int,SprTrainedNode*>::iterator iter = copy.begin();
       iter!=copy.end();iter++ ) {
    nodes_.push_back(iter->second);
  }

  // exit
  return true;
}


void SprTrainedTopdownTree::printFunction(std::ostream& os,
					  const SprTrainedNode* currentNode,
					  int indentLevel) const
{
  // Use root if no node given.
  const SprTrainedNode* node; 
  if( currentNode == 0 ) 
    node = nodes_[0]; 
  else 
    node = currentNode; 
 
  // Print this node. 
  if( node->d_ >= 0 ) { 
    for( int I=0;I<indentLevel;I++ ) os << " ";
    os << "if( V[" << node->d_ << "] < " << node->cut_ << " ) {" << endl; 
    this->printFunction(os,node->toDau1_,indentLevel+2); 
    for( int I=0;I<indentLevel;I++ ) os << " ";
    os << "}" << endl; 
    for( int I=0;I<indentLevel;I++ ) os << " ";
    os << "else /*if( V[" << node->d_ << "] >= " 
       << node->cut_ << " )*/ {" << endl; 
    this->printFunction(os,node->toDau2_,indentLevel+2); 
    for( int I=0;I<indentLevel;I++ ) os << " ";
    os << "}" << endl; 
  } 
  else { 
    for( int I=0;I<indentLevel;I++ ) os << " ";
    os << "R += " << node->score_ << ";" << endl; 
  } 
}
