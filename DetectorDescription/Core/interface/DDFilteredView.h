#ifndef DDCore_DDFilteredView_h
#define DDCore_DDFilteredView_h

#include <vector>
#include <utility>
#include "DetectorDescription/Core/interface/DDExpandedView.h"
#include "DetectorDescription/Core/interface/DDFilter.h"

class DDScope;

class DDFilteredView
{
public:
  typedef DDExpandedView::nav_type nav_type;
  
  DDFilteredView(const DDCompactView &);
  
  ~DDFilteredView();
  
  void addFilter(const DDFilter &, DDLogOp op = DDLogOp::AND);
  
    //! The logical-part of the current node in the filtered-view
  const DDLogicalPart & logicalPart() const;
  
  //! The absolute translation of the current node
  const DDTranslation & translation() const;
  
  //! The absolute rotation of the current node
  const DDRotationMatrix & rotation() const;
  
  //! The list of ancestors up to the root-node of the current node
  const DDGeoHistory & geoHistory() const;
  
  //! return the stack of sibling numbers
  nav_type navPos() const;
  bool goTo(const nav_type &);
  
  //! return the stack of copy numbers
  nav_type copyNumbers() const;
   
  //! User specific data attached to the current node
  void specificsV(std::vector<const DDsvalues_type * > & result) const;
  std::vector<const DDsvalues_type * > specifics() const;
  void mergedSpecificsV(DDsvalues_type & merged) const;
  DDsvalues_type mergedSpecifics() const;
  
  //! Copy number associated with the current node  
  int copyno() const;
  
  // navigation 

  //! The scope of the expanded-view.
  const DDGeoHistory & scope() const;
  
  //! sets the scope of the expanded view
  bool setScope(const DDGeoHistory & hist);
  
  //! clears the scope; the full tree is available, depth=0
  void clearScope();
    
  //! set current node to the next node in the filtered tree
  bool next(); 
  
  //! set the current node to the next sibling ...
  bool nextSibling();
  
  //! set the current node to the first child ...
  bool firstChild();
  
  //! set the current node to the parent node ...
  bool parent();
  
  //! clears the scope and sets the filtered view to its root-node
  void reset();
  
  void print();
  
  const std::vector<DDGeoHistory> & history() const;
     
private:
  bool filter();

  DDExpandedView epv_;
  std::vector<DDFilter const *> criteria_;
  std::vector<DDLogOp> logOps_; // logical operation for merging the result of 2 filters
  std::vector<DDGeoHistory> parents_; // filtered-parents
};

#endif
