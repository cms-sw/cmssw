#ifndef x_graphwalker_h
#define x_graphwalker_h

#include "DetectorDescription/Core/interface/adjgraph.h"

#include <vector>
#include <queue>

//#include <iostream> // debug

/** a walker for an acyclic directed multigraph */
template <class N, class E>
class graphwalker
{
public:
  typedef typename graph<N,E>::index_type index_type;
  
  typedef typename graph<N,E>::index_result index_result;

  typedef typename graph<N,E>::edge_type edge_type;
  
  typedef typename graph<N,E>::edge_list edge_list;
  
  typedef typename graph<N,E>::edge_iterator edge_iterator;
  
  typedef typename graph<N,E>::const_edge_iterator const_edge_iterator;
  
  // only a const-edge_range!
  typedef typename std::pair<const_edge_iterator, const_edge_iterator> edge_range;
  
  //typedef std::pair<edge_range,edge_range> stacked_element;
  
  typedef std::vector<edge_range> stack_type;
  typedef std::queue<edge_type> bfs_type;

  typedef bool /*std::pair<const N &, bool>*/ result_type;
  
  typedef typename graph<N,E>::value_type value_type;
  
  
public:
  //! creates a walker rooted by the first candidate root found in the underlying graph
  graphwalker(const graph<N,E> &);

  //! creates a walker rooted by the node given
  graphwalker(const graph<N,E> &, const N & );
   
  // operations
  
  result_type firstChild();
  
  result_type nextSibling();
  
  result_type parent();
  
  result_type next();
  
  inline value_type current() const;
  
  result_type next_bfs();
  value_type current_bfs() const;
    
  void reset();
  
  const stack_type & stack() const { return stack_;}
  
protected:
  // stack_.back().first corresponds to index of the current node!
  stack_type stack_; // hierarchical stack used in navigation
  bfs_type queue_; // breath first search queue
  edge_list root_; // root of the walker
  //std::vector<N> rootCandidates_; 
  const graph<N,E> & graph_;
  //jklsdfjklsdfkljsdfakjl;
private:
  graphwalker();

};



template<class N, class E>
graphwalker<N,E>::graphwalker(const graph<N,E> & g)
 : graph_(g)
{  // complexity = (no nodes) * (no edges)
   graph_.findRoots(root_);
   stack_.push_back(edge_range(root_.begin(),root_.end())); 
   if (root_.size()) {
     queue_.push(root_[0]);
   }
}


template<class N, class E>
graphwalker<N,E>::graphwalker(const graph<N,E> & g, const N & root)
 : graph_(g)
{
   index_result rr = graph_.nodeIndex(root);
   if (!rr.second) // no such root node, no walker can be created!
     throw root;
     
   root_.push_back(edge_type(rr.first, 0));
   stack_.push_back(edge_range(root_.begin(),root_.end()));   
   queue_.push(root_[0]);
}


template<class N, class E>
typename graphwalker<N,E>::value_type graphwalker<N,E>::current() const
{
   const edge_range & er = stack_.back();
/*
   const N & n = graph_.nodeData(er.first->first);
   const E & e = er.first->first;
   return value_type(n,e);
*/   
   return value_type(graph_.nodeData(er.first->first), graph_.edgeData(er.first->second)); 
}


template<class N, class E>
typename graphwalker<N,E>::value_type graphwalker<N,E>::current_bfs() const
{
   const edge_type & e = queue_.front();
   return value_type(graph_.nodeData(e.first), graph_.edgeData(e.second)); 
}


template<class N, class E>
void graphwalker<N,E>::reset()
{
  //std::cout << "graphwalker::reset" << std::endl;
  stack_.clear();
  stack_.push_back(edge_range(root_.begin(),root_.end()));
  queue_.clear();
  if (root_.size()) {
    queue_.push(root_[0]);   
  }
}


template<class N, class E>
typename graphwalker<N,E>::result_type graphwalker<N,E>::firstChild()
{
   result_type result = false;
   const edge_range & adjEdges
     = graph_.edges(stack_.back().first->first);
   if (adjEdges.first != adjEdges.second) {
     stack_.push_back(adjEdges);
     result = true;
   }
   return result;
}


template<class N, class E>
typename graphwalker<N,E>::result_type graphwalker<N,E>::nextSibling()
{
   result_type result = false;
   //if (stack_.size() > 1) { only if single-root should be enforced ...
     edge_range & siblings = stack_.back();
     if (siblings.first != (siblings.second - 1) ) {
       ++siblings.first;
       result = true;
     }  
   //}
   return result;
}


template<class N, class E>
typename graphwalker<N,E>::result_type graphwalker<N,E>::parent()
{
   //std::cout << "graphwalker::parent()" << std::endl;
   result_type result = false;
   if (stack_.size()>1) {
     stack_.pop_back();
     result = true;
   }
   return result;
}


template<class N, class E>
typename graphwalker<N,E>::result_type graphwalker<N,E>::next()
{
   result_type result = false;
   if (firstChild()) {
     result = true;
   }  
   else if(stack_.size()>1 && nextSibling()) {
     result = true;
   }  
   else {  
     while(parent()) {
       if(stack_.size()>1 && nextSibling()) {
         result = true;
	 break;
       }
     }
   }   	   
   return result;
}   

template<class N, class E>
typename graphwalker<N,E>::result_type graphwalker<N,E>::next_bfs()
{
   result_type result(false);
   if (!queue_.empty()) {
     const edge_type & e = queue_.front();
     const edge_range & er = graph_.edges(e.first);
     const_edge_iterator it(er.first), ed(er.second);
     for (; it != ed; ++it) {
       queue_.push(*it);
     }
     queue_.pop();
     if (!queue_.empty()) {
       result=true;
     }
   }
   return result;
}
#endif
