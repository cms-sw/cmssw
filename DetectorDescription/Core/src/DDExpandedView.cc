#include "DetectorDescription/Core/interface/DDExpandedView.h"

#include <memory>
#include <ostream>

#include "DetectorDescription/Core/interface/DDComparator.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDPosData.h"
#include "DataFormats/Math/interface/Graph.h"
#include "DataFormats/Math/interface/GraphWalker.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Math/GenVector/Cartesian3D.h"
#include "Math/GenVector/DisplacementVector3D.h"
#include "Math/GenVector/Rotation3D.h"

class DDPartSelection;

/** 
   After construction the instance corresponds to the root of the geometrical tree.
*/
DDExpandedView::DDExpandedView( const DDCompactView & cpv )
  : walker_(nullptr),
    w2_( cpv.graph(), cpv.root()),
    trans_( DDTranslation()),
    rot_( DDRotationMatrix()),
    depth_( 0 ),
    worldpos_( cpv.worldPosition())
{
  walker_ = &w2_;

  const DDPosData * pd((*walker_).current().second);
  if (!pd)
    pd = worldpos_;
  DDExpandedNode expn((*walker_).current().first,
                      pd,
		      trans_,
		      rot_,
		      0);
  
  // starting point for position calculations, == root of expanded view
  history_.emplace_back(expn);		      		      
}

DDExpandedView::~DDExpandedView() { }  

const DDLogicalPart & DDExpandedView::logicalPart() const 
{ 
  return history_.back().logp_;
}


const DDTranslation & DDExpandedView::translation() const 
{ 
  return history_.back().trans_; 
}


const DDRotationMatrix & DDExpandedView::rotation() const 
{ 
  return history_.back().rot_; 
}


const DDGeoHistory & DDExpandedView::geoHistory() const 
{ 
  return history_; 
} 


int DDExpandedView::depth() const
{
  return depth_;
}


int DDExpandedView::copyno() const 
{ 
  return history_.back().copyno();
}

/** 
   returns true, if a next sibling exists and updates \c this
   otherwise returns false.
   If a scope was set, the nextSibling of the root of the scope is not selected.
*/    
bool DDExpandedView::nextSibling() 
{
  bool result(false);
  if (!scope_.empty() && history_.back() == scope_.back()) {
   ; // no-next-sibling, if current node is the root of the scope!
  } 
  else {
    if ((*walker_).nextSibling()) {
      DDExpandedNode & expn(history_.back()); // back of history_ is always current node
      DDCompactView::walker_type::value_type curr = (*walker_).current();
      DDPosData const * posdOld = expn.posd_;
      expn.logp_=curr.first;
      expn.posd_=curr.second;
      
      DDGeoHistory::size_type hsize = history_.size();
      
      
      if (hsize>1) {
	const DDExpandedNode & expnBefore(history_[hsize-2]);
	
	// T = T1 + INV[R1] * T2
	expn.trans_  = expnBefore.trans_ + (expnBefore.rot_ * expn.posd_->trans());
     
	// R = R1*INV[R2]	
	// VI in principle we can do this
	if ( !(expn.posd_->rot()==posdOld->rot()) ) {
	  expn.rot_ = expnBefore.rot_ * expn.posd_->rot();//.inverse();
	}
      }
      else {
	expn.trans_ = expn.posd_->trans();
	expn.rot_ = expn.posd_->rot();//.inverse();
      }   
      ++expn.siblingno_;
      result = true; 
    }
  }
  return result;
}
 
 
/** 
   returns true, if a child of the current node exists and updates \c this
   otherwise returns false
*/    
bool DDExpandedView::firstChild()
{
  bool result(false);
  bool depthNotReached(true);
  
  // Check for the depth within the scope ...
  if (depth_) {
    if ( (history_.size()-scope_.size())==depth_ ) {
      depthNotReached=false;
    }
  }
  if (depthNotReached) {
    if ((*walker_).firstChild()) {
      DDExpandedNode & expnBefore(history_.back());
      DDCompactView::walker_type::value_type curr = (*walker_).current();
 
      DDPosData * newPosd = curr.second;
    
          // T = ... (see nextSiblinig())
      DDTranslation newTrans = expnBefore.trans_ + expnBefore.rot_ * newPosd->trans();
    
      // R = ... (see nextSibling())
      DDRotationMatrix newRot =  expnBefore.rot_ *  newPosd->rot();//.inverse();
    
      // create a new Expanded node and push it to the history ...
      DDExpandedNode expn(curr.first, curr.second,
                          newTrans, newRot, 0);
    
      history_.emplace_back(expn);			
      result = true;                     
    } // if firstChild 
  } // if depthNotReached
  return result;
} 


/** 
   returns ture, if a parent exists and updates \c this otherwise returns
   false. When false is returned, the root node of the scope is reached.
*/    
bool DDExpandedView::parent()
{
  bool result(false);
  bool scopeRoot(false);
  
  // check for a scope
  if (!scope_.empty()) {
    if (scope_.back() == history_.back()) { 
      // the current node is the root of the scope
      scopeRoot = true;
    }  
  }
  
  if (!scopeRoot) {
    if ((*walker_).parent()) {
      history_.pop_back();
      result = true;
    }
  }   
  
  return result;  
}

// same implementation as in GraphWalker !
/** 
   Tree transversal:
    
   - try to go to the first child
      
   - else try to go to the next sibling
      
   - else try to go to the next sibling of the parent
      
   Currently the whole remaining subtree is transversed when next() is
   subsequently called.
      
   \todo stop, when subtree down a specified node has been transversed    
*/      
bool DDExpandedView::next()
{
  bool res(false);
  if(firstChild()) 
    res=true;
  else if(nextSibling())
    res=true;
  else {
   while(parent()) {
     if(nextSibling()) {
       res=true;
       break;
     }  
   }
  }
  return res;
}


/** broad first */
bool DDExpandedView::nextB()
{
   bool res(false);
   return res;  
}


void dump(const DDGeoHistory & history)
{
   edm::LogInfo("DDExpandedView")  << "--GeoHistory-Dump--[" << std::endl;
   int i=0;
   for( const auto& it : history ) {
     edm::LogInfo("DDExpandedView")  << " " << i << it.logicalPart() << std::endl;
     ++i;	  
   }
   edm::LogInfo("DDExpandedView")  << "]---------" << std::endl;
}

/** 
   User specific data can be attached to single nodes or a selection of
   nodes in the expanded view through the DDSpecifics interface.
      
   The resulting std::vector is of size 0 if no specific data was attached.
   
*/
std::vector< const DDsvalues_type *>  DDExpandedView::specifics() const
{
  // backward compatible
  std::vector<const DDsvalues_type * > result;
  specificsV(result);
  return result;
}

void
DDExpandedView::specificsV(std::vector<const DDsvalues_type * > & result) const
{
  const auto & specs = logicalPart().attachedSpecifics();
  if( !specs.empty())
  {
    result.reserve(specs.size());
    for( const auto& it : specs ) {
      // a part selection
      const DDPartSelection & psel = *(it.first);
      const DDGeoHistory & hist = geoHistory();
      
      if (DDCompareEqual(hist, psel)()) 
	result.emplace_back( it.second );
    }
  }  
}
							   
DDsvalues_type DDExpandedView::mergedSpecifics() const {
  DDsvalues_type merged;
  mergedSpecificsV(merged);
  return merged;
}

void DDExpandedView::mergedSpecificsV(DDsvalues_type & merged) const
{
  merged.clear();
  const auto& specs = logicalPart().attachedSpecifics();
  if (specs.empty()) return;
  const DDGeoHistory & hist = geoHistory();
  for( const auto& it : specs ) {
    if (DDCompareEqual(hist, *it.first)())
      merge(merged,*it.second);
  }
}

/**
   All navigational commands only operate in the subtree rooted by the
   node marked by the node of the DDGeoHistory returned by this method.
   If the size() of the scope equals 0, the full scope covering the 
   whole expanded-view is set (default).
*/
const DDGeoHistory & DDExpandedView::scope() const
{
   return scope_;
}   

void DDExpandedView::clearScope()
{
  scope_.clear();
  depth_=0;
}

void DDExpandedView::reset()
{
   clearScope();
   while(parent()) 
     ;
}


/**
   The scope of the expanded-view is set to the subtree rooted by the node 
   marked by the DDGeohistory hist.
   The current not of the expanded view is set to the root of the subtree.
   All navigational methods apply only on the subtree.
    
   In case of hist not marking a valid node in the expanded-view, the
   state of the expanded-view is unchanged and false is returned by setScope().
   Otherwise true is returned.
*/
bool DDExpandedView::setScope(const DDGeoHistory & sc, int depth)
{
  bool result(false);
  
  DDGeoHistory buf = scope_; // save current scope
  scope_.clear(); // sets scope to global (full) scope

  while (parent()) ; // move up to the root of the expanded-view
  
  if (descend(sc)) { // try to move down the given scope-history ...
    scope_ = sc;
    depth_ = depth;
    result = true;
  }  
  else {
    scope_ = buf;
  }
  
  return result;  
}


/**
  goTo will reset the ExpandedView if pos is not a valid position.
  Currently no checks are implemented to verify that pos is within the
  current scope of the ExpandedView.
  \todo check whether pos is in the current scope 
*/
bool DDExpandedView::goToHistory(const DDGeoHistory & pos)
{
  bool result = true;
  int tempD = depth_;
  DDGeoHistory tempScope = scope_;
  reset();
  DDGeoHistory::size_type s = pos.size();
  for( DDGeoHistory::size_type j=1; j<s; ++j) {
    if (! firstChild()) {
      result = false;
      break;
    }  
    int i=0;
    for (; i<pos[j].siblingno(); ++i) {
      if (! nextSibling()) {
	result = false;
      }	
    }
  }
  
  if (!result) {
    reset();
    setScope(tempScope, tempD);
  } 
  else {
    scope_ = tempScope;
    depth_ = tempD;
  }

  return result;
}

//! \todo implement it simpler using DDExpandedNode::siblingno() 
bool DDExpandedView::descend(const DDGeoHistory & sc) 
{
  DDGeoHistory::size_type mxx = sc.size();
  DDGeoHistory::size_type cur = 0;
  bool result(false);
  
  /* algo: compare currerent node in expanded-view with current-node in sc
           if matching:
	     (A)go to first child in expanded-view, go one level deeper in sc
	     iterate over all children in expanded-view until one of them
	     matches the current node in sc. 
	     if no one matches, return false
	     else continue at (A)
	   else return false
  */	   
  const DDExpandedNode & curNode = history_.back();
  
  if (!sc.empty()) {
    if (curNode==sc[cur]) {
      bool res(false);
      while(cur+1 < mxx && firstChild()) {
        ++cur;
        if (!(history_.back()==sc[cur])) {
	  while(nextSibling()) {
	    if (history_.back()==sc[cur]) {
	      res=true;
	      break;
	    }  
	  }
	}  
	else {
	  res=true;
	}
	if (res==false) 
	  break;  
      }
      result = res;
    } 	  
  }
  return result; 
}


bool DDExpandedView::goTo(const nav_type & newpos) {
  return goTo(&newpos.front(),newpos.size());

}

bool DDExpandedView::goTo(NavRange newpos) {
  return goTo(newpos.first,newpos.second);
}

bool DDExpandedView::goTo(int const * newpos, size_t sz)
{
  bool result(false);
  
  // save the current position
  DDGeoHistory savedPos = history_;
   
  // reset to root node 
  //FIXME: reset to root of scope!!
  reset();
  
  // try to navigate down to the newpos
  for (size_t i = 1; i < sz; ++i) {
    result = firstChild();
    if (result) {
      int pos = newpos[i];
      for(int k=0; k<pos; ++k) {
        result = nextSibling();
      }
    }
    else {
      break;
    }   
  }
  
  if (!result) {
    goToHistory(savedPos);
  }
  return result;
}

DDExpandedView::nav_type DDExpandedView::navPos() const
{
  DDGeoHistory::size_type i=0;
  DDGeoHistory::size_type j=history_.size();
  nav_type pos(j);  
  
  for (;i<j;++i)
    pos[i] = history_[i].siblingno();
    
  return pos;   
}

DDExpandedView::nav_type DDExpandedView::copyNumbers() const
{
  DDGeoHistory::size_type it = 0;
  DDGeoHistory::size_type sz = history_.size();
  nav_type result(sz);
  
  for (; it < sz; ++it) {
    result[it] = history_[it].copyno();
  }
  return result;
}

std::string printNavType(int const * n, size_t sz){
  std::ostringstream oss;
  oss << '(' ;
  for (int const * it=n; it != n+sz; ++it) {
    oss << *it << ',';
  }
  oss << ')';
  return oss.str();
}
