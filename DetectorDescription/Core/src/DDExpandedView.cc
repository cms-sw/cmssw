#include "DetectorDescription/Core/interface/DDExpandedView.h"
#include "DetectorDescription/Core/interface/DDComparator.h"
#include "DetectorDescription/Base/interface/DDdebug.h"

/** 
   After construction the instance corresponds to the root of the geometrical tree.
*/
DDExpandedView::DDExpandedView(const DDCompactView & cpv)
 : walker_(0),w2_(cpv.graph(),cpv.root()), trans_(DDTranslation()), rot_(DDRotationMatrix()),
   depth_(0), worldpos_(0)
{
  //  std::cout << "Building a DDExpandedView" << std::endl;
  // MEC:2010-02-08 - consider the ROOT as where you want to start LOOKING at
  // the DDD, and worldpos_ as the "real" root node of the graph.  MOVE all this 
  // logic to DDCompactView.  This should really be just the traverser...
  DDRotation::StoreT::instance().setReadOnly(false);
  worldpos_ = new DDPosData(DDTranslation(),DDRotation(),0);     
  DDRotation::StoreT::instance().setReadOnly(true);
  
  walker_ = &w2_;

  //  std::cout << "Walker: current.first=" << (*walker_).current().first << std::endl;
  //  std::cout << "Walker: current.second=" << (*walker_).current().second << std::endl;
  
  DDPosData * pd((*walker_).current().second);
  if (!pd)
    pd = worldpos_;  
  DDExpandedNode expn((*walker_).current().first,
                      pd,
		      trans_,
		      rot_,
		      0);
  
  // starting point for position calculations, == root of expanded view
  history_.push_back(expn);		      		      
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
  //return (*walker_).current().second->copyno_; 
}

  
namespace {

  struct Counter {
    int same;
    int diff;
    ~Counter() {
    }

  };

  inline Counter & counter() {
    static Counter local;
    return local;
  }


}


/** 
   returns true, if a next sibling exists and updates \c this
   otherwise returns false.
   If a scope was set, the nextSibling of the root of the scope is not selected.
*/    
bool DDExpandedView::nextSibling() 
{
  bool result(false);
  if (scope_.size() && history_.back() == scope_.back()) {
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
	expn.trans_  = expnBefore.trans_ + (expnBefore.rot_ * expn.posd_->trans_);
     
	// R = R1*INV[R2]	
	// VI in principle we can do this
	if ( !(expn.posd_->rot()==posdOld->rot()) ) {
	  expn.rot_ = expnBefore.rot_ * expn.posd_->rot();//.inverse();
	  ++counter().diff;
	}else ++counter().same;

      }
      else {
	expn.trans_ = expn.posd_->trans_;
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
      DDTranslation newTrans = expnBefore.trans_ + expnBefore.rot_ * newPosd->trans_;
    
      // R = ... (see nextSibling())
      DDRotationMatrix newRot =  expnBefore.rot_ *  newPosd->rot();//.inverse();
    
      // create a new Expanded node and push it to the history ...
      DDExpandedNode expn(curr.first, curr.second,
                          newTrans, newRot, 0);
    
      history_.push_back(expn);			
    
      /* debug output
      edm::LogInfo("DDExpandedView")  << "FIRSTCHILD: name=" << expn.logicalPart().ddname() 
           << " rot=";
	 
      if (expn.absRotation().isIdentity())
        edm::LogInfo("DDExpandedView")  << "[none]" << std::endl;
      else
        edm::LogInfo("DDExpandedView")  << expn.absRotation() << std::endl;
      */
    
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
  if (scope_.size()) {
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

/*
bool DDExpandedView::hasChildren() const
{
  bool result = false;
   
  return result;
}
*/

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
     //DCOUT('C', "pa=" << logicalPart() );
     if(nextSibling()) {
       //DCOUT('C', "ns=" << logicalPart() );
       res=true;
       break;
     }  
   }
   //DCOUT('C', current().first << " "<< current().second );
  }
  return res;
}


/** broad first */
bool DDExpandedView::nextB()
{
   bool res(false);
   return res;  
}


void dump(const DDGeoHistory & h)
{
   DDGeoHistory::const_iterator it = h.begin();
   edm::LogInfo("DDExpandedView")  << "--GeoHistory-Dump--[" << std::endl;
   int i=0;
   for (; it != h.end(); ++it) {
     edm::LogInfo("DDExpandedView")  << " " << i << it->logicalPart() << std::endl;
     /*
          << "     "  << it->logicalPart().material() << std::endl
	  << "     "  << it->logicalPart().solid() << std::endl;
     */
     ++i;	  
   }
   edm::LogInfo("DDExpandedView")  << "]---------" << std::endl;
}

/** 
   User specific data can be attac
hed to single nodes or a selection of
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

void  DDExpandedView::specificsV(std::vector<const DDsvalues_type * > & result) const
{
  unsigned int i(0);
  //edm::LogInfo("DDExpandedView")  << " in ::specifics " << std::endl;
  const std::vector<std::pair<DDPartSelection*, DDsvalues_type*> > & specs = logicalPart().attachedSpecifics();
  if (specs.size()) { // do only if SpecPar has data defined 
    //edm::LogInfo("DDExpandedView")  << " found: specifics size=" << specs.size() << std::endl;
    result.reserve(specs.size());
    for (; i<specs.size(); ++i) {
      const std::pair<DDPartSelection*,DDsvalues_type*>& sp = specs[i];
      // a part selection
      const DDPartSelection & psel = *(sp.first);
      //edm::LogInfo("DDExpandedView")  << " partsel.size = " << psel.size() << std::endl;
      //edm::LogInfo("DDExpandedView")  << " geohistory   = " << geoHistory() << std::endl;
      const DDGeoHistory & hist = geoHistory();
      
      //dump(hist);
      //dump(psel);
      
      if (DDCompareEqual(hist, psel)()) //edm::LogInfo("DDExpandedView")  << "MATCH!!!!" << std::endl;
	result.push_back( sp.second );
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
  const std::vector<std::pair<DDPartSelection*, DDsvalues_type*> > & specs = logicalPart().attachedSpecifics();
  if (specs.empty()) return;
  const DDGeoHistory & hist = geoHistory();
  for (size_t i=0; i<specs.size(); ++i) {
    const std::pair<DDPartSelection*,DDsvalues_type*>& sp = specs[i];
    const DDPartSelection & psel = *(sp.first);
    if (DDCompareEqual(hist, psel)())
      merge(merged,*sp.second);
  }
  // std::sort(merged.begin(),merged.end());
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
  //DCOUT('G', " goto- target= " << pos );
  DDGeoHistory tempScope = scope_;
  reset();
  DDGeoHistory::size_type s = pos.size();
  for( DDGeoHistory::size_type j=1; j<s; ++j) {
    if (! firstChild()) {
      result = false;
      //edm::LogError("DDExpandedView") << " ERROR!  , wrong usage of DDExpandedView::goTo! " << std::endl;
      //exit(1);
      break;
    }  
    int i=0;
    for (; i<pos[j].siblingno(); ++i) {
      if (! nextSibling()) {
        //edm::LogError("DDExpandedView") << " ERROR!  , wrong usage of DDExpandedView::goTo! " << std::endl;        
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
  
  //DCOUT('G', " goto-result = " << history_ );
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
  
  if (sc.size()) {
    //DCOUT('x', "curN=" << curNode.logicalPart() << " scope[0]=" << sc[cur].logicalPart() );
    if (curNode==sc[cur]) {
      bool res(false);
      while(cur+1 < mxx && firstChild()) {
        ++cur;
        //DCOUT('x', "fc-curN=" << history_.back().logicalPart() << " scope[x]=" << sc[cur].logicalPart() );
        if (!(history_.back()==sc[cur])) {
	  while(nextSibling()) {
	    //DCOUT('x', "ns-curN=" << history_.back().logicalPart() << " scope[x]=" << sc[cur].logicalPart() );
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
  //nav_type savedPos = navPos(); 
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

std::ostream & printNavType(std::ostream & os, int const * n, size_t sz){
  os << '(' ;
  for (int const * it=n; it != n+sz; ++it) {
    os << *it << ',';
  }
  os << ')';
  return os;
}



//THIS IS WRONG, THIS IS WRONG, THIS IS WRONG (not functional wrong but in any other case!)
//THIS IS WRONG, THIS IS STUPID, i bin a depp ...
/*
void doit(DDGeoHistory& h) {
  DDRotationMatrix m1, m2, m3;
  DDGeoHistory::size_type s(h.size());
  std::vector<DDRotationMatrix> rotVec(s);
  std::vector<DDTranslation> transVec(s);
  
  DDGeoHistory::size_type c(s);
  for (int i=0; i<s; ++i) {
    rotVec[i]   = h[i].posd_->rot_;
    transVec[i] = h[i].posd_->trans_;          
  }
    
  if (s>1) {
    for (int i=1; i<s; ++i) {
      rotVec[i] = rotVec[i-1]*rotVec[i];
      //h[i].rot_ = h[i-1].posd_->rot_ * h[i].posd_->rot_;
    }
    
    for (int i=1; i<s; ++i)
       transVec[i] = transVec[i-1] + rotVec[i-1]*transVec[i];
       //h[i].trans_ = h[i-1].trans_ + h[i-1].rot_ * h[i]  
  }
  h[s-1].trans_ = transVec[s-1];
  h[s-1].rot_ = rotVec[s-1];

}
*/

