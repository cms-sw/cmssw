#include "DetectorDescription/Core/interface/DDNumberingScheme.h"
#include "DetectorDescription/Core/interface/DDExpandedView.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"

// Message logger.
#include "FWCore/MessageLogger/interface/MessageLogger.h"

DDNumberingScheme::~DDNumberingScheme() {}


DDDefaultNumberingScheme::DDDefaultNumberingScheme(const DDExpandedView & ex)
{ 
  // extremely memory consuming & slow.
  
  /* - assign a node-number (from 0, ...) to every node in the view
     - save the node-number in a map, key is the stack of sibling-numbers of the node
       -> enables node to id calculation (slow O(log(max. node-number))).
     - save in a std::vector the stack of sibling-numbers; index in the std::vector is the
       assigned node number -> enables id to node calculation (fast)  
  */   
  typedef std::map<nav_type,int>::iterator m_it;
  
  DDExpandedView e = ex;
  e.reset();
  bool go = true;
  int count = 0;
  while (go) {
    std::pair<m_it,bool> res = path2id_.insert(std::make_pair(e.navPos(),count));
    id2path_.push_back(res.first);
    ++count;
    go = e.next();
  }
}


DDDefaultNumberingScheme::DDDefaultNumberingScheme(const DDFilteredView & fv)
{ 
  // very memory consuming & slow, depending on the amount of nodes
  // selected by the FilteredView; same algorithm then in ctor above
  typedef std::map<nav_type,int>::iterator m_it;
  
  DDFilteredView f = fv;
  f.reset();
  bool go = true;
  int count = 0;
  while (go) {
    std::pair<m_it,bool> res = path2id_.insert(std::make_pair(f.navPos(),count));
    id2path_.push_back(res.first);
    ++count;
    go = f.next();
  }
}


DDDefaultNumberingScheme::~DDDefaultNumberingScheme()
{ }


int DDDefaultNumberingScheme::id(const DDExpandedView & e) const
{
  return id(e.navPos());
}
  

int DDDefaultNumberingScheme::id(const DDFilteredView & f) const
{
 return id(f.navPos());
}
  

int DDDefaultNumberingScheme::id(const DDDefaultNumberingScheme::nav_type & n) const
{
  std::map<nav_type,int>::const_iterator it = path2id_.find(n);
  int result = -1;
  if (it != path2id_.end())
    result = it->second;
  return result;  
}


bool DDDefaultNumberingScheme::node(int id, DDExpandedView & view) const
{
 return view.goTo(idToNavType(id));
}


bool DDDefaultNumberingScheme::node(int id, DDFilteredView & view) const
{
 edm::LogError("DDNumberingScheme") << "DDDefaultNumberingScheme::node(int,DDFilteredView&) NOT IMPLEMENTED!" 
           << std::endl;
 return view.goTo(idToNavType(id));
}


DDNumberingScheme::nav_type DDDefaultNumberingScheme::idToNavType(int id) const
{ 
  std::vector<int>::size_type pos = id;
  nav_type result;
  if ( (id>=(int)id2path_.size()) || ( id < 0) ) 
    ;
     
  else {
    std::map<nav_type,int>::iterator it = id2path_[pos];
    result = it->first;
  }
  return result;
}

