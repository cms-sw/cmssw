#ifndef DDCompactViewImpl_h
#define DDCompactViewImpl_h

//  IMPORTANT: DO NOT USE THIS:  It is here temporarily only to obey the rule that 
//  any .h that includes a .h even if it is not meant to be a public interface must
//  remain in the interface directory.  If you fix/move/change this please delete
//  this comment!  -- Michael Case 2007-04-05

#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDPosData.h"
#include "DetectorDescription/Core/interface/graphwalker.h"

class DDPartSelector;
//typedef TreeNode<DDPhysicalPart,int> expnode_t;

class DDCompactViewImpl 
{
public:
  //typedef TreeNode<DDLogicalPart, DDPosPart > TreeNav;
  
  //FIXME: DDCompactViewImpl.h: Graph<DDLogPart, DDPosPart> :
  //FIXME:                      take more efficient EdgeData!
  typedef ::graph<DDLogicalPart, DDPosData* > GraphNav;
  //typedef GraphPath<DDLogicalPart, DDPosData*> GraphNavPaths;
  //typedef GraphWalker<DDLogicalPart,DDPosData*> walker_t;
  
  // internal use ! (see comments in DDCompactView(bool)!)
  explicit DDCompactViewImpl();
  
  DDCompactViewImpl(const DDLogicalPart & rootnodedata);
  
  ~DDCompactViewImpl();
  
  //FIXME: DDCompactView::setRoot(..) check if new root exists ...
  void setRoot(const DDLogicalPart & root) { root_=root; }

  const DDLogicalPart & root() const { return root_; }
  
  //deprecated!
  void print(std::ostream & os) { /*graph_.printHierarchy(os,root_);*/ }
  
  DDLogicalPart & current() const;
  
  //  std::pair<bool,DDPhysicalPart> goTo(const DDPartSelector &) const;
  
  //expnode_t * expand(const DDPartSelector & path) const;
 
  const GraphNav & graph() const { return graph_; }
  /**
   returns a walker beginning at the root of the expanded-view
   FIXME: CompactView::walker: it is assumed, that the root of walker is
   FIXME:                      world volume (just 1 copy, unrotated, unpositioned)
  */ 
  graphwalker<DDLogicalPart,DDPosData*> walker() const; 
  
  //double weight(DDLogicalPart &);
  double weight(const DDLogicalPart &) const;

  void swap( DDCompactViewImpl& );  
  /**
   will return a walker beginning at the node specified by the PartSelector   
  
  walker_t walker(const DDPartSelector&) const;
  
  */
protected:    
  
  // recursive tree building ...
  //void buildTree(const DDLogicalPart & nodata, TreeNav* parent);
  
  // recursive graph building (acyclic directed (parent->child))
  void buildGraph();
  void buildPaths();
  
  DDLogicalPart root_;
  //DDPhysicalPart current_;
  GraphNav graph_;
  //GraphNavPaths * paths_;
  //GraphWalker<DDLogicalPart,DDPosData*> * rootWalker_;
  //TreeNav *  tn_;
  //TreeNav * root_;
};
#endif
