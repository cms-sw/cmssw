#ifndef DDExpandedView_h
#define DDExpandedView_h

#include <iostream>
#include <vector>
#include <string>
#include <map>

#include "DetectorDescription/DDCore/interface/DDTransform.h"
#include "DetectorDescription/DDCore/interface/DDCompactView.h"
#include "DetectorDescription/DDCore/interface/DDLogicalPart.h"
#include "DetectorDescription/DDCore/interface/DDPosData.h"
#include "DetectorDescription/DDCore/interface/DDExpandedNode.h"
#include "DetectorDescription/DDCore/interface/DDsvalues.h"

class DDFilteredView;

/**
  DDExpandedView provides a tree-walker (iterator) for the 
  expanded view of the detector description. 
  Further it provides a registration mechanism for call-backs
  whenever a node in the expanded view becomes current and
  fullfills the user-defined predicate.
  
  FIXME: DDExpandedView: in the Prototype just one class -
  FIXME:                 later seperation of interface & implementation!
*/
//! Provides an exploded view of the detector (tree-view)
/** Taking a DDCompactView the DDExpandedView expands the compact-view into
    a detector tree. One instance of DDExpandedView corresponds to one node
    in the tree. It is the 'current' node. By using tree navigation ( next(), nextSibling(),
    parent(), firstChild() ) the DDExpandedView represents the new corresponding node.
*/  
class DDExpandedView
{
  friend class DDFilteredView;
  
public:
  //! vector of sibling numbers
  typedef vector<int> nav_type;
  
  //! Constructs an expanded-view based on the compact-view
  DDExpandedView(const DDCompactView &);
  
  virtual ~DDExpandedView();
  
  //! The logical-part of the current node in the expanded-view
  const DDLogicalPart & logicalPart() const;
  
  //! The absolute translation of the current node
  const DDTranslation & translation() const;
  
  //! The absolute rotation of the current node
  const DDRotationMatrix & rotation() const;
  
  //! The list of ancestors up to the root-node of the current node
  const DDGeoHistory & geoHistory() const;
  
  //! transversed the DDExpandedView according to the given stack of sibling numbers
  bool goTo(const nav_type &);
  
  //! return the stack of sibling numbers which indicates the current position in the DDExpandedView
  nav_type navPos() const;
  
  //! return the stack of copy numbers
  nav_type copyNumbers() const;
  
  //! User specific data attached to the current node
  vector<const DDsvalues_type * > specifics() const;
  void specificsV(vector<const DDsvalues_type * > & vc ) const;

  DDsvalues_type mergedSpecifics() const;
  void mergedSpecificsV(DDsvalues_type & res) const;
  
  //! Copy number associated with the current node  
  int copyno() const;
  
  // navigation 

  //! The scope of the expanded-view.
  const DDGeoHistory & scope() const;
  
  //! sets the scope of the expanded view
  bool setScope(const DDGeoHistory & hist, int depth=0);
  
  //! clears the scope; the full tree is available, depth=0
  void clearScope();
  
  //! depth of the scope. 0 means unrestricted depth.
  int depth() const;
    
  //! set current node to the next node in the expanded tree
  bool next(); 
  
  //! broad search order of next()
  bool nextB();
  
  //! set the current node to the next sibling ...
  bool nextSibling();
  
  //! set the current node to the first child ...
  bool firstChild();
  
  //! set the current node to the parent node ...
  bool parent();
  
  //! true, if a call to firstChild() would succeed (current node has at least one child)
  //bool hasChildren() const;
  
  //! clears the scope and sets the ExpandedView to its root-node
  void reset();
  
/**    NOT IN THE PROTOTYPE
  \todo void addNodeAction(const DDNodeAction &);
  
  \todo bool removeNodeAction(const DDNodeAction &);
*/
  /* was protected, now public; was named goTo, now goToHistory*/
  bool goToHistory(const DDGeoHistory & sc);
  
protected:
  bool descend(const DDGeoHistory & sc);

protected:
  DDCompactView::walker_type * walker_; //!< the tricky walker
  DDCompactView::walker_type w2_;
  const DDTranslation trans_;
  const DDRotationMatrix rot_;
  DDGeoHistory history_; //!< vector of DDExpandedNode
  DDGeoHistory scope_; //!< scope of the expanded view
  unsigned int depth_; //!< depth of the scope, 0==unrestricted depth
  DDPosData * worldpos_ ; //!< ???
  vector<nav_type> nextBStack_;
  //map<string,string> dummySpecifics_;    
};

ostream & operator<<(ostream &, const DDExpandedView::nav_type &);
#endif
