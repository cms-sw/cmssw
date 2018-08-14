#ifndef DATA_FORMATS_MATH_GRAPH_H
#define DATA_FORMATS_MATH_GRAPH_H

#include <iostream>
#include <map>
#include <vector>

// Adjecencylist Graph
namespace math {
  
// N,E must be concepts of default constructable, assignable, copyable, operator<
template <class N, class E> 
class Graph
{
public:
  using index_type = std::vector<double>::size_type;
  // (node-index target, edge)
  using edge_type = std::pair<index_type, index_type>;
  // (std::vector of edge_types for the adj_list)
  using edge_list = std::vector<edge_type>;
  // (node-index -> edge_list) the adjacency-list
  using adj_list = std::vector<edge_list>;
  
  class const_iterator
  {
    friend class Graph<N, E>;
  public:
    using index_type = Graph::index_type;
    using adj_list = Graph::adj_list;
    using edge_list = Graph::edge_list;
    
    struct value_type
    {
      friend class Graph<N,E>::const_iterator;
      value_type( const Graph & g, index_type a, index_type e ) 
	: gr_( g ), a_( a ), e_( e ) 
	{ }
      
      const N & from( void ) const { return gr_.nodeData( a_ ); }
      const N & to( void )   const { return gr_.nodeData( gr_.adjl_[a_][e_].first ); }
      const E & edge( void ) const { return gr_.edgeData( gr_.adjl_[a_][e_].second ); }
      
    private:
      const Graph & gr_;
      index_type a_, e_;       
    };
    
    using reference = value_type&;
    using pointer = value_type*;
       
    bool operator==( const const_iterator & i ) const {  
      return (( vt_.a_ == i.vt_.a_ ) && ( vt_.e_ == i.vt_.e_ )) ? true : false;
    }
    
    bool operator!=( const const_iterator & i ) const {
      return (( vt_.a_ == i.vt_.a_ ) && ( vt_.e_ == i.vt_.e_ )) ? false : true;
    }

    void operator++() {
      while( vt_.gr_.size() > vt_.a_ )
      {
        index_type i = vt_.gr_.adjl_[vt_.a_].size();
        if( i > vt_.e_+1 )
	{
	  ++vt_.e_;
	  return;
	}
	vt_.e_=0;
	++vt_.a_;
	while( vt_.gr_.size() > vt_.a_ )
	{
	  if( vt_.gr_.adjl_[vt_.a_].size())
	  {
	    return;
	  }
	  ++vt_.a_;
	}
      } 
    }

    const value_type & operator*() const {
      return vt_;    
    }
    
    const value_type * operator->() const {
      return &vt_;
    }  
    
  private:
    explicit const_iterator( const Graph & g )
      : vt_( g, 0, 0 )
      {}
      
    const_iterator( const Graph & g, index_type ait, index_type eit ) 
      : vt_( g, ait, eit )
      {} 
       
    value_type vt_;
      
    bool operator<( const const_iterator & i ) const { 
      return ( vt_.a_ < i.vt_.a_ ) && ( vt_.e_ < i.vt_.e_ );
    }

    bool operator>(const const_iterator & i) const {
      return ( vt_.a_ > i.vt_.a_ ) && ( vt_.e_ > i.vt_.e_ );
    }
  };

  // Graphtypes
  
  struct value_type {
    value_type(const N & n, const E & e) : first(n), second(e) { }
    const N & first;
    const E & second;
    N firstToValue() const { return first; }
    E secondToValue() const { return second; }
  };

  // (node-index -> node)
  using node_list = std::vector<N>;
  using edge_store = std::vector<E>;
  
  // (node-index -> edge_list) the adjacency-list
  using adj_iterator = adj_list::iterator;
  using const_adj_iterator = adj_list::const_iterator;
  
  // assigns a node-index to the node
  using indexer_type = std::map<N, index_type>;
  using indexer_iterator = typename indexer_type::iterator;
  using const_indexer_iterator = typename indexer_type::const_iterator;
  
  // supported iterators and ranges
  using edge_iterator = edge_list::iterator;
  using const_edge_iterator = edge_list::const_iterator;
  using edge_range = std::pair<edge_iterator, edge_iterator>;
  using const_edge_range = std::pair<const_edge_iterator, const_edge_iterator>;
  using index_result = std::pair<index_type, bool>;  
  
public:
  // creation, deletion
  Graph() : edges_(1)  { }
  ~Graph() { }

  // operations
  
  // O(log(n)), n...number of nodes
  index_type addNode(const N &); 
  // O(log(n*e)), n,e...number of nodes,edges
  void addEdge(const N & from, const N & to, const E & edge);
  
  // O(1)
  //index_type addNode(const node_type &);
  // O(log(e))
  //index_type addEdge(const node_type & from, const node_type & to, const E & e);
 
  inline index_result nodeIndex(const N &) const;
  
  //index_type edgeIndex(const E &) const;
  
  // indexed edge_ranges, O(1) operation
  inline edge_range edges(index_type nodeIndex);
  inline const_edge_range edges(index_type nodeIndex) const;
  
  // indexed edge_ranges, O(log(n)) operation, n...number of nodes
  inline edge_range edges(const N &);
  inline const_edge_range edges(const N &) const;
  
  inline const N & nodeData(const edge_type &) const;
  inline const N & nodeData(index_type) const;
  inline const N & nodeData(const const_adj_iterator &) const;
 
  // replace oldNode by newNode O(log(n))
  bool replace(const N  & oldNode , const N & newNode );
   
  //replace oldEdge by newEdge
  bool replaceEdge(const E & ldEdge, const E &newEdge ); 
   
  const E & edgeData(index_type i) const { return edges_[i]; }
  // const N & nodeData(const adj_iterator &) const;
  // index of a node (O(log(n))
  
  //! it clear everything!
  void clear();
  // access to the linear-iterator
  const_iterator begin_iter() const { return const_iterator(*this); }    
  
  const_iterator end_iter() const { return const_iterator(*this, adjl_.size(),0); }
  
  size_t edge_size() const { return edges_.size(); }
  
  // access to the adjacency-list
  adj_iterator begin() { return adjl_.begin(); } 
  const_adj_iterator begin() const { return adjl_.begin(); }
  adj_iterator end() { return adjl_.end(); }
  const_adj_iterator end() const { return adjl_.end(); }
  auto size() const -> adj_list::size_type { return adjl_.size(); }
  
  // finds all roots of the Graph and puts them into the edge_list
  void findRoots(edge_list &) const;
  
  // inverts the directed Graph, i.e. edge(A,B) -> edge(B,A)
  void invert(Graph & g) const;

  void swap( Graph<N, E> & );
  
  // Data   
private:
  
  // adjacency list
  adj_list adjl_;
  
  // std::mapping of index to node
  node_list nodes_;
  
  // std::mapping of indes to edge
  edge_store edges_;
  
  // indexer for N and E
  indexer_type indexer_; // eIndexer_;
  
  // dummy
  edge_list emptyEdges_;
  
};

							

template<class N, class E>
typename Graph<N,E>::index_type Graph<N,E>::addNode(const N & node)
{
  index_type idx = indexer_.size() ; //  +1;
  std::pair<indexer_iterator,bool> result 
    = indexer_.insert(typename indexer_type::value_type(node,idx));
  
  if ( result.second ) { // new index!
    nodes_.emplace_back(node);
    adjl_.emplace_back(edge_list());
  }  
  else {
    idx = result.first->second;
  }
  return idx;
}


template<class N, class E>
typename Graph<N,E>::index_result Graph<N,E>::nodeIndex(const N & node) const
{
  typename indexer_type::const_iterator result = indexer_.find(node);
  index_type idx = 0;
  bool flag = false;
  if (result != indexer_.end()) {
    flag = true;
    idx = result->second;
  }
  return index_result(idx, flag);
}


template<class N, class E>
void Graph<N,E>::addEdge(const N & from, const N & to, const E & edge)
{
  index_type iFrom = addNode(from);
  index_type iTo   = addNode(to);
  
  adjl_[iFrom].emplace_back(edge_type(iTo,edges_.size()));
  edges_.emplace_back(edge);
}


template<class N, class E>
typename Graph<N,E>::edge_range Graph<N,E>::edges(index_type nodeIndex)
{
  edge_list & edges = adjl_[nodeIndex];
  return edge_range(edges.begin(), edges.end());
}


template<class N, class E>
typename Graph<N,E>::const_edge_range Graph<N,E>::edges(index_type nodeIndex) const
{
  const edge_list & edges = adjl_[nodeIndex];
  return const_edge_range(edges.begin(), edges.end());
}


template<class N, class E>
typename Graph<N,E>::edge_range Graph<N,E>::edges(const N & node)
{
  index_result idxResult = nodeIndex(node);
  edge_range result(emptyEdges_.begin(),emptyEdges_.end());
  if (idxResult.second) {
    result = edges(idxResult.first);
  }   
  return result;
}


template<class N, class E>
typename Graph<N,E>::const_edge_range Graph<N,E>::edges(const N & node) const
{
  index_result idxResult = nodeIndex(node);
  const_edge_range result(emptyEdges_.begin(),emptyEdges_.end());
  if (idxResult.second) {
    result = edges(idxResult.first);
  }   
  return result;
}


template<class N, class E>
const N & Graph<N,E>::nodeData(const edge_type & edge) const
{
  return nodes_[edge.first];
}


template<class N, class E>
const N & Graph<N,E>::nodeData(index_type i) const
{
  return nodes_[i];
}


template<class N, class E>
const N & Graph<N,E>::nodeData(const const_adj_iterator & it) const
{
  return nodes_[it-adjl_.begin()];
}

template<class N, class E>
void Graph<N,E>::findRoots(edge_list & result) const
{
  result.clear();
      
  const_adj_iterator it = begin();   
  const_adj_iterator ed = end();
  std::vector<bool> rootCandidate(size(), true);
  
  for (; it != ed; ++it) {
    const edge_list & el = *it;
    for (auto const & el_it : el) {
      rootCandidate[el_it.first]=false; 
    }
  }
  std::vector<bool>::size_type v_sz = 0;
  std::vector<bool>::size_type v_ed = rootCandidate.size();
  for (; v_sz < v_ed; ++v_sz) {
    if (rootCandidate[v_sz]) {
      result.emplace_back(edge_type(v_sz,0));    
    }
  }  
}

template<class N, class E>
bool Graph<N,E>::replace(const N & oldNode, const N & newNode)
{
  typename indexer_type::iterator it = indexer_.find(oldNode);
  if (it != indexer_.end()) {
    index_type oldIndex = it->second;
    nodes_[oldIndex]=newNode;
    indexer_[newNode]=oldIndex;
    indexer_.erase(it);
  }  
  else throw(oldNode);
  return true;   
}

template<class N, class E>
bool Graph<N,E>::replaceEdge(const E & oldEdge, const E & newEdge)
{
  typename edge_store::size_type it = 0;
  typename edge_store::size_type ed = edges_.size();
  bool result = false;
  for (; it < ed; ++it) {
    if ( edges_[it] == oldEdge ) {
      result = true;
      edges_[it] = newEdge;
      break;
    }
  }
  return result;
}

template<class N, class E>
void Graph<N,E>::clear()
{
  adjl_.clear();
  nodes_.clear();
  edges_.clear();
  indexer_.clear();
}

template<class N, class E>
void Graph<N,E>::invert(Graph<N,E> & g) const
{
  adj_list::size_type it = 0;
  adj_list::size_type ed = adjl_.size();
  // loop over adjacency-list of this Graph
  for (; it < ed; ++it) {
    const edge_list & el = adjl_[it];
    edge_list::size_type eit = 0;
    edge_list::size_type eed = el.size();
    // loop over edges of current node
    for (; eit < eed; ++eit) {
      const edge_type & e = el[eit];
      g.addEdge(nodeData(e.first), nodeData(it), edgeData(e.second));
    } 
  } 
}

template<class N, class E>
void Graph<N,E>::swap( Graph<N, E> & g) { 
  adjl_.swap(g.adjl_);
  nodes_.swap(g.nodes_);
  edges_.swap(g.edges_);
  indexer_.swap(g.indexer_);
  emptyEdges_.swap(g.emptyEdges_);
}

template<typename T> std::ostream & operator<<(std::ostream & o, const std::vector< std::vector<std::pair<T,T> > > v)
{
  typedef typename std::vector<std::vector<std::pair<T,T> > > v_t;
  typedef typename std::vector<std::pair<T,T> > i_t;
  
  typename v_t::const_iterator it(v.begin()), ed(v.end());
  for (; it != ed; ++it) {
    typename i_t::const_iterator iit(it->begin()), ied(it->end());
    for(; iit != ied; ++iit) {
      o << iit->first << ':' << iit->second << std::endl;
    }
  }
  return o;
}
 
} // namespace math

#endif
