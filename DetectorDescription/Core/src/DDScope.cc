#include "DetectorDescription/Core/interface/DDScope.h"

dd_scope_class DDScopeClassification::operator()( const DDGeoHistory & left,
                                                  const DDGeoHistory & right ) const
{
  dd_scope_class result = subtree;
  DDGeoHistory::const_iterator lit = left.begin(); // left-iterator
  DDGeoHistory::const_iterator rit = right.begin(); // right-iterator

  while( lit != left.end() && rit != right.end())
  {
    if( lit->siblingno() != rit->siblingno())
    {
      result = different_branch;
      break;
    }  
    ++lit;
    ++rit;
  }
  
  if( result != different_branch )
  {
    if( lit == left.end()) { // left history leaf node marks the root of a subtree which contains
      result = supertree;    // the leaf node of the right history or both roots are the same ...
    }
    else
    {
      result = subtree;
    }    
  }
  return result;
}

DDScope::DDScope( void )
 : depth_( 0 )
{}

DDScope::DDScope( const DDGeoHistory & h, int depth )
 : depth_( depth )
{
  subtrees_.push_back( h );
}

DDScope::~DDScope( void ) 
{}

bool
DDScope::addScope( const DDGeoHistory & h )
{
  bool result = false;
  scope_type::iterator it = subtrees_.begin();
  scope_type buf;
  int supertreeCount = 0;
  bool diffBranch = false;
  bool subTree = false;
   
  for(; it != subtrees_.end(); ++it )
  {
    dd_scope_class classification = classify_( h, *it );
    switch( classification )
    { 
    case different_branch:
      buf.push_back( *it );
      diffBranch = true;
      break;
     
    case subtree:
      buf.push_back( *it );
      subTree = true;
      break;
     
    case supertree:
      ++supertreeCount;
      if( supertreeCount == 1 )
	buf.push_back(h);
      break;
       
    default:
      break;  
    }
  }
   
  if( diffBranch )
  {
    if( subTree == false )
    {
      buf.push_back(h);
    }  
  }
   
  if( !subtrees_.size()) 
    subtrees_.push_back( h );
  else  
    subtrees_ = buf;
   
   return result;
}

void
DDScope::setDepth( int d )
{
  depth_ = d;
}

int
DDScope::depth( void ) const
{
  return depth_;
}

const DDScope::scope_type &
DDScope::scope( void ) const
{
  return subtrees_;
}

std::ostream & operator<<( std::ostream & os, const DDScope & scope )
{
   DDScope::scope_type::const_iterator it = scope.subtrees_.begin();
   for(; it != scope.subtrees_.end(); ++ it )
   {
     os << *it << std::endl;
   }
   return os;
}
