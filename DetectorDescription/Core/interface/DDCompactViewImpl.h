#ifndef DETECTOR_DESCRIPTION_CORE_DD_COMPACT_VIEW_IMPL_H
# define DETECTOR_DESCRIPTION_CORE_DD_COMPACT_VIEW_IMPL_H

#include "DetectorDescription/Core/interface/DDTranslation.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDPosData.h"
#include "DetectorDescription/Core/interface/DDTransform.h"
#include "DataFormats/Math/interface/Graph.h"
#include "DataFormats/Math/interface/GraphWalker.h"

class DDDivision;
struct DDPosData;

class DDCompactViewImpl 
{
public:
  
  using Graph = math::Graph<DDLogicalPart, DDPosData* >;
  using GraphWalker = math::GraphWalker<DDLogicalPart, DDPosData* >;

  explicit DDCompactViewImpl();
  DDCompactViewImpl(const DDLogicalPart & rootnodedata);
  ~DDCompactViewImpl();
  
  //reassigns root with no check!!!
  void setRoot(const DDLogicalPart & root) { root_=root; }

  const DDLogicalPart & root() const { return root_; }
  
  DDLogicalPart & current() const;
  
  const Graph& graph() const { return graph_; }
  GraphWalker walker() const;

  void position (const DDLogicalPart & self,
		 const DDLogicalPart & parent,
		 int copyno,
		 const DDTranslation & trans,
		 const DDRotation & rot,
		 const DDDivision * div);

  void swap( DDCompactViewImpl& );  

protected:    
  // internal use ! (see comments in DDCompactView(bool)!)
  DDLogicalPart root_;
  Graph graph_;
};

#endif
