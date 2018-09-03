#ifndef DETECTOR_DESCRIPTION_CORE_DD_COMPACT_VIEW_H
#define DETECTOR_DESCRIPTION_CORE_DD_COMPACT_VIEW_H

#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "DetectorDescription/Core/interface/DDRotationMatrix.h"
#include "DetectorDescription/Core/interface/DDTranslation.h"
#include "DetectorDescription/Core/interface/Store.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDPosData.h"
#include "DetectorDescription/Core/interface/DDTransform.h"
#include "DataFormats/Math/interface/Graph.h"
#include "DataFormats/Math/interface/GraphWalker.h"

class DDCompactViewImpl;
class DDDivision;
class DDName;
struct DDPosData;

namespace DDI {
  class LogicalPart;
  class Material;
  class Solid;
  class Specific;
}

/**
  Navigation through the compact view of the detector ...
*/
//MEC: these comments are kept from original... Will we ever do this? don't think so.
//FIXME: DDCompactView: check for proper acyclic directed graph structure!!
//FIXME:
//FIXME:         X          [A-X] ... LogicalPart
//FIXME:        / \             | ... PosPart (directed parten to child)
//FIXME:       A   A
//FIXME:       |   | 
//FIXME:       B   C      
//FIXME:
//FIXME:    THIS IS NOT ALLOWED, but currently can be specified using DDL ....
//FIXME:

//! Compact representation of the geometrical detector hierarchy
/** A DDCompactView represents the detector as an acyclic directed multigraph.
    The nodes are instances of DDLogicalPart while the edges are pointers to
    DDPosData. Edges are directed from parent-node to child-node. 
    Edges represented by DDPosData are the relative translation and rotation
    accompanied by a copy-number of the child-node towards the parent-node.
    
    One node is explicitly marked as the root node. It is the DDLogicalPart which
    defines the base coordinate system for the detector. All subdetectors are
    directly or inderectly positioned inside the root-node. 
    
    Example:
    
    The figureshows a compact-view graph consisting of 16 DDLogicalParts 
    interconnected by 20 edges represented by pointers to DDPosData.
    \image html compact-view.gif
    \image latex compact-view.eps
    
    The compact-view also serves as base for calculating nodes in the expanded
    view. Paths through the compact-view can be viewed as nodes in the expanded-view
    (expansion of an acyclic directed multigraph into a tree). In the figure there are
    5 different paths from CMS to Module2 (e.g. CMS-Pos1->Ecal-Pos4->EEndcap-Pos21->Module2)
    thus there will be 5 nodes of Module2 in the expanded view.

MEC:
    There has been a re-purposing of the DDCompactView to not only hold the 
    representation described above (in detail this is the DDCompactViewImpl)
    but also own the memory of the stores refered to by the graph.

    DDCompactView now owns the DDMaterial, DDSpecific, DDLogicalPart,
    DDRotation, DDSolid and etc.  Removal of the global one-and-only 
    stores, methods and details such as DDRoot will mean that all of
    these will be accessed via the DDCompactView.
*/
class DDCompactView
{ 
public:
  using Graph = math::Graph<DDLogicalPart, DDPosData* >;
  using GraphWalker = math::GraphWalker<DDLogicalPart, DDPosData* >;
  
  //! Creates a compact-view 
  explicit DDCompactView();

  //! Creates a compact-view using a different root of the geometry hierarchy
  explicit DDCompactView( const DDName& );
  
  ~DDCompactView();
  
  //! Creates a compact-view using a different root of the geometry hierarchy.
  // NOTE: It cannot be used to modify the stores if they are locked.
  explicit DDCompactView(const DDLogicalPart & rootnodedata);
  
  //! Provides read-only access to the data structure of the compact-view.
  const Graph & graph() const;
  GraphWalker walker() const;

  //! returns the DDLogicalPart representing the root of the geometrical hierarchy
  const DDLogicalPart & root() const;
  
  //! The absolute position of the world
  const DDPosData * worldPosition() const;

  void position (const DDLogicalPart & self,
		 const DDLogicalPart & parent,
		 const std::string& copyno,
		 const DDTranslation & trans,
		 const DDRotation & rot,
		 const DDDivision * div = nullptr);
  
  void position (const DDLogicalPart & self,
		 const DDLogicalPart & parent,
		 int copyno,
		 const DDTranslation & trans,
		 const DDRotation & rot,
		 const DDDivision * div = nullptr);
  
  void setRoot(const DDLogicalPart & root);

  void lockdown();
  
 private:
  void swap( DDCompactView& );

  std::unique_ptr<DDCompactViewImpl> rep_;
  std::unique_ptr<DDPosData> worldpos_ ;

  DDI::Store<DDName, std::unique_ptr<DDI::Material>> matStore_;
  DDI::Store<DDName, std::unique_ptr<DDI::Solid>> solidStore_;
  DDI::Store<DDName, std::unique_ptr<DDI::LogicalPart>> lpStore_;
  DDI::Store<DDName, std::unique_ptr<DDI::Specific>> specStore_;
  DDI::Store<DDName, std::unique_ptr<DDRotationMatrix>> rotStore_;
};

#endif
