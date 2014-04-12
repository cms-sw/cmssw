#ifndef DDD_graph_h
#define DDD_graph_h

#include <vector>
#include <map>
#include <iostream>

// Adjecencylist graph

// N,E must be concepts of default constructable, assignable, copyable, operator<
template <class N, class E> 
class graph
{
public:
  typedef std::vector<double>::size_type index_type;
  // (node-index target, edge)
  typedef std::pair<index_type, index_type> edge_type;
  // (std::vector of edge_types for the adj_list)
  typedef std::vector<edge_type> edge_list;
  // (node-index -> edge_list) the adjacency-list
  typedef std::vector<edge_list> adj_list;
  
  class const_iterator
  {
    friend class graph<N, E>;
  public:
    typedef typename graph::index_type index_type;
    typedef typename graph::adj_list adj_list;
    typedef typename graph::edge_list edge_list;
    
    struct value_type
    {
      friend class graph<N,E>::const_iterator;
      value_type( const graph & g, index_type a, index_type e ) 
	: gr_( g ), a_( a ), e_( e ) 
	{ }
      
      const N & from( void ) const { return gr_.nodeData( a_ ); }
      const N & to( void )   const { return gr_.nodeData( gr_.adjl_[a_][e_].first ); }
      const E & edge( void ) const { return gr_.edgeData( gr_.adjl_[a_][e_].second ); }
      
    private:
      const graph & gr_;
      index_type a_, e_;       
    };
    
    typedef value_type& reference;
    typedef value_type* pointer;
       
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
    explicit const_iterator( const graph & g )
      : vt_( g, 0, 0 )
      {}
      
    const_iterator( const graph & g, index_type ait, index_type eit ) 
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

  void dump_graph( void ) const;  
  // Graphtypes
  
  struct value_type {
    value_type(const N & n, const E & e) : first(n), second(e) { }
    const N & first;
    const E & second;
    N firstToValue() const { return first; }
    E secondToValue() const { return second; }
  };

  // (node-index -> node)
  typedef std::vector<N> node_list;
  typedef std::vector<E> edge_store;
  
  // (node-index -> edge_list) the adjacency-list
  typedef typename adj_list::iterator adj_iterator;
  typedef typename adj_list::const_iterator const_adj_iterator;
    
  
  // assigns a node-index to the node
  typedef std::map<N, index_type> indexer_type;
  typedef typename indexer_type::iterator indexer_iterator;
  typedef typename indexer_type::const_iterator const_indexer_iterator;
  
  // supported iterators and ranges
  typedef typename edge_list::iterator edge_iterator;
  
  typedef typename edge_list::const_iterator const_edge_iterator;
  
  typedef std::pair<edge_iterator,edge_iterator> edge_range;
  
  typedef std::pair<const_edge_iterator, const_edge_iterator> const_edge_range;

  typedef std::pair<index_type, bool> index_result;  
  
public:
  // creation, deletion
  graph() : edges_(1)  { }
  ~graph() { }

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
  typename adj_list::size_type size() const { return adjl_.size(); }
  
  // finds all roots of the graph and puts them into the edge_list
  void findRoots(edge_list &) const;
  
  // inverts the directed graph, i.e. edge(A,B) -> edge(B,A)
  void invert(graph & g) const;

  void swap( graph<N, E> & );
  
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
typename graph<N,E>::index_type graph<N,E>::addNode(const N & node)
{
  index_type idx = indexer_.size() ; //  +1;
  std::pair<indexer_iterator,bool> result 
    = indexer_.insert(typename indexer_type::value_type(node,idx));
  
  if ( result.second ) { // new index!
    nodes_.push_back(node);
    adjl_.push_back(edge_list());
  }  
  else {
    idx = result.first->second;
  }
  return idx;
}


template<class N, class E>
typename graph<N,E>::index_result graph<N,E>::nodeIndex(const N & node) const
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
void graph<N,E>::addEdge(const N & from, const N & to, const E & edge)
{
  index_type iFrom = addNode(from);
  index_type iTo   = addNode(to);
  
  adjl_[iFrom].push_back(edge_type(iTo,edges_.size()));
  edges_.push_back(edge);
}


template<class N, class E>
typename graph<N,E>::edge_range graph<N,E>::edges(index_type nodeIndex)
{
  edge_list & edges = adjl_[nodeIndex];
  return edge_range(edges.begin(), edges.end());
}


template<class N, class E>
typename graph<N,E>::const_edge_range graph<N,E>::edges(index_type nodeIndex) const
{
  const edge_list & edges = adjl_[nodeIndex];
  return const_edge_range(edges.begin(), edges.end());
}


template<class N, class E>
typename graph<N,E>::edge_range graph<N,E>::edges(const N & node)
{
  index_result idxResult = nodeIndex(node);
  edge_range result(emptyEdges_.begin(),emptyEdges_.end());
  if (idxResult.second) {
    result = edges(idxResult.first);
  }   
  return result;
}


template<class N, class E>
typename graph<N,E>::const_edge_range graph<N,E>::edges(const N & node) const
{
  index_result idxResult = nodeIndex(node);
  const_edge_range result(emptyEdges_.begin(),emptyEdges_.end());
  if (idxResult.second) {
    result = edges(idxResult.first);
  }   
  return result;
}


template<class N, class E>
const N & graph<N,E>::nodeData(const edge_type & edge) const
{
  return nodes_[edge.first];
}


template<class N, class E>
const N & graph<N,E>::nodeData(index_type i) const
{
  return nodes_[i];
}


template<class N, class E>
const N & graph<N,E>::nodeData(const const_adj_iterator & it) const
{
  return nodes_[it-adjl_.begin()];
}

template<class N, class E>
void graph<N,E>::findRoots(edge_list & result) const
{
  result.clear();
      
  const_adj_iterator it = begin();   
  const_adj_iterator ed = end();
  std::vector<bool> rootCandidate(size(), true);
  
  for (; it != ed; ++it) {
    const edge_list & el = *it;
    typename edge_list::const_iterator el_it = el.begin();
    typename edge_list::const_iterator el_ed = el.end();
    for (; el_it != el_ed; ++el_it) {
      rootCandidate[el_it->first]=false; 
      //el_rt = el_it; // stop at the first encountered candidate!
      //std::cout << "graphwalker: found a root candidate = " << g.nodeData(el_rt->first) << std::endl;
      //break; 
    }
  }
  std::vector<bool>::size_type v_sz = 0;
  std::vector<bool>::size_type v_ed = rootCandidate.size();
  for (; v_sz < v_ed; ++v_sz) {
    if (rootCandidate[v_sz]) {
      //std::cout << "found = " << g.nodeData(v_sz) << std::endl;
      result.push_back(edge_type(v_sz,0));    
    }
  }  
}

template<class N, class E>
bool graph<N,E>::replace(const N & oldNode, const N & newNode)
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
bool graph<N,E>::replaceEdge(const E & oldEdge, const E & newEdge)
{
  typename edge_store::size_type it = 0;
  typename edge_store::size_type ed = edges_.size();
  bool result = false;
  //std::cout << "newedge=" << newEdge << std::endl;
  for (; it < ed; ++it) {
    //std::cout << "edge=" << edges_[it] << " ";
    if ( edges_[it] == oldEdge ) {
      //std::cout << "FOUND!" << std::endl;
      result = true;
      edges_[it] = newEdge;
      break;
    }
  }
  //std::cout << std::endl;
  return result;
}

template<class N, class E>
void graph<N,E>::clear()
{
  adjl_.clear();
  nodes_.clear();
  edges_.clear();
  indexer_.clear();
}

template<class N, class E>
void graph<N,E>::dump_graph() const
{
  //  std::cout << adjl_ << std::endl;
  /*
    std::cout << "Nodes and their indices:" << std::endl;
    typename indexer_type::const_iterator it = indexer_.begin();
    for (; it != indexer_.end(); ++it) {
    std::cout << ' ' << it->first << ' ' << it->second << std::endl;
    }
  */   
}


template<class N, class E>
void graph<N,E>::invert(graph<N,E> & g) const
{
  adj_list::size_type it = 0;
  adj_list::size_type ed = adjl_.size();
  // loop over adjacency-list of this graph
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
void graph<N,E>::swap( graph<N, E> & g) { 
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
#endif
