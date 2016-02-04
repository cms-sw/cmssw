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

class DDCompactViewImpl 
{
public:
  
  typedef ::graph<DDLogicalPart, DDPosData* > GraphNav;
  explicit DDCompactViewImpl();
  DDCompactViewImpl(const DDLogicalPart & rootnodedata);
  ~DDCompactViewImpl();
  
  //reassigns root with no check!!!
  void setRoot(const DDLogicalPart & root) { root_=root; }

  const DDLogicalPart & root() const { return root_; }
  
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

  void position (const DDLogicalPart & self,
		 const DDLogicalPart & parent,
		 int copyno,
		 const DDTranslation & trans,
		 const DDRotation & rot,
		 const DDDivision * div);

  void swap( DDCompactViewImpl& );  
  /**
   will return a walker beginning at the node specified by the PartSelector   
  
  walker_t walker(const DDPartSelector&) const;
  
  */
protected:    
  // internal use ! (see comments in DDCompactView(bool)!)
  DDLogicalPart root_;
  GraphNav graph_;
};
#endif
