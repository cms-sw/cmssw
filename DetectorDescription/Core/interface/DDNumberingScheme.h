#ifndef DDNumberingScheme_h
#define DDNumberingScheme_h

#include <vector>

class DDExpandedView;
//typename DDExpandedView::nav_type;
class DDFilteredView;
//class DDFilteredView::nav_type;
//! Base for user specfic numbering schemes
/**
  implements a default - numbering scheme
    throws an DDException when it fails ...
*/
class DDNumberingScheme //: throw DDException
{
public:
  typedef std::vector<int> nav_type;
  virtual ~DDNumberingScheme();
  
  //! calculate the id of a given node
  virtual int id(const DDExpandedView &) const = 0;

  //! calculate the id of a given node
  virtual int id(const DDFilteredView &) const = 0;
  
  //! calculate the id of a given node
  virtual int id(const nav_type &) const = 0 ;
  
  
  //! calculate the node given an id
  /** 
    returns true, if a node was found. view then corresponds to this node.
  */  
  virtual bool node(int id, DDExpandedView & view) const = 0;

  //! calculate the node given an id
  /** 
    returns true, if a node was found. view then corresponds to this node.
  */  
  virtual bool node(int id, DDFilteredView & view) const = 0;
  
};




#include <map>
//! Default numbering scheme
/**
  implements a default - numbering scheme
    throws an DDException when it fails ...
*/
class DDDefaultNumberingScheme : public DDNumberingScheme //: throw DDException
{
public:
  typedef DDNumberingScheme::nav_type nav_type;
  DDDefaultNumberingScheme(const DDExpandedView & e);
  DDDefaultNumberingScheme(const DDFilteredView & f);
  virtual ~DDDefaultNumberingScheme();
  
  //! calculate the id of a given node
  virtual int id(const DDExpandedView &) const;
  
  //! calculate the id of a given node
  virtual int id(const DDNumberingScheme::nav_type &) const;
  
  //! calculate the id of a given node
  virtual int id(const DDFilteredView &) const;

  //! calculate the node given an id
  /** 
    returns true, if a node was found. view then corresponds to this node.
  */  
  virtual bool node(int id, DDExpandedView & view) const;

  //! calculate the node given an id
  /** 
    returns true, if a node was found. view then corresponds to this node.
  */  
  virtual bool node(int id, DDFilteredView & view) const;

protected:
  DDNumberingScheme::nav_type idToNavType(int id) const;

protected:
  std::map<nav_type,int> path2id_;
  std::vector<std::map<nav_type,int>::iterator> id2path_;
};




#endif
