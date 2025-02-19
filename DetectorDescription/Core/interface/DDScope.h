#ifndef DDCore_DDScope_h
#define DDCore_DDScope_h

#include <vector>
#include "DetectorDescription/Core/interface/DDExpandedNode.h"

enum dd_scope_class { different_branch, subtree, supertree, delete_action };

//! Classification of scope describe by A towards B
/**
  The leaf-node of A defines the root of a subtree in the DDExpandedView, so does the 
  leaf-node of B.
  - returns different_branch, if the leaf-node of A defines a different subtree than the leaf-node of B
  - returns subtree, if the leaf-node of A is in the subtree rooted by the leaf-node of B
  - returns supertree, if the leaf-node of B is in the subtree rooted by the leaf-node of A
*/
struct DDScopeClassification
{
  dd_scope_class operator()( const DDGeoHistory &, const DDGeoHistory & ) const;
};  

//! defines subtrees in the expanded-view
/**
  One scope is defined by a set of DDGeoHistory. 
*/
class DDScope
{
  friend std::ostream & operator<<( std::ostream &, const DDScope & );
  
public:
  typedef std::vector<DDGeoHistory> scope_type;
  
  //! empty scope
  DDScope( void );
  
  //! scope with a single subtree
  DDScope( const DDGeoHistory &, int depth = 0 );
  
  ~DDScope( void );
  
  //! Adds a scope. No new scope will be added if s is already contained in one of the subtrees
  /**
    returns true, if scope has changed, else false.
  */  
  bool addScope( const DDGeoHistory & s );
  
  //! subtrees of the scope are only transversed down to the given level
  void setDepth( int );  
  
  //! return the depth to wich the subtrees are restricted
  int depth( void ) const;
  
  //! returns the scope container
  const scope_type & scope( void ) const;
  
protected:
  scope_type subtrees_;
  DDScopeClassification classify_;
  int depth_;
};

std::ostream & operator<<( std::ostream &, const DDScope & );

#endif
