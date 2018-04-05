#ifndef DDCompactView_h
#define DDCompactView_h

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
class DDPartSelector;
class DDPhysicalPart;
struct DDPosData;

namespace DDI {
  class LogicalPart;
  class Material;
  class Solid;
  class Specific;
}


/**
  Navigation through the compact view of the detector ...

Updated: Michael Case [ MEC ] 2010-02-11

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

//typedef TreeNode<DDPhysicalPart,int> expnode_t;
//! type of data representation of DDCompactView
//typedef graph<DDLogicalPart,DDPosData*> graph_type; //:typedef Graph<DDLogicalPart,DDPosData*> graph_type;

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
  //! container-type of children of a given node in the compact-view
  typedef std::vector<DDLogicalPart> logchild_type;
  
  //! container-type of pairs of children nodes and their relative position data of a given node in the compact-view
  typedef std::vector< std::pair<DDLogicalPart,DDPosData*> > poschildren_type;
  
  //! pair ...
  typedef std::pair<DDLogicalPart,DDPosData*> pos_type;
  
  typedef math::GraphWalker<DDLogicalPart,DDPosData*> walker_type;
  
  //! type of representation of the compact-view (acyclic directed multigraph)
  /** Nodes are instances of DDLogicalPart, edges are pointers to instances of DDPosData */
  typedef math::Graph<DDLogicalPart,DDPosData*> graph_type;
    
  //! Creates a compact-view 
  explicit DDCompactView();
  
  //! \b EXPERIMENTAL! Creates a compact-view using a different root of the geometrical hierarchy
  explicit DDCompactView(const DDLogicalPart & rootnodedata);
  
  ~DDCompactView();
  
  //! Provides read-only access to the data structure of the compact-view.
  const graph_type & graph() const;

  //! returns the DDLogicalPart representing the root of the geometrical hierarchy
  const DDLogicalPart & root() const;
  
  //! The absolute position of the world
  const DDPosData * worldPosition() const;

  //! Prototype version of calculating the weight of a detector component
  double weight(const DDLogicalPart & p) const;

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
  
  // ************************************************************************
  // UNSTABLE STUFF below! DON'T USE!
  // ************************************************************************
  
  //! \b don't \b use : interface not stable ....
  void setRoot(const DDLogicalPart & root);

  //! \b dont't \b use ! Proper implementation missing ...
  walker_type walker() const;

  // ---------------------------------------------------------------
  // +++ DDCore INTERNAL USE ONLY ++++++++++++++++++++++++++++++++++
    
  // to modify the structure! DDCore internal!
  graph_type & writeableGraph();

  void swap( DDCompactView& );

  void lockdown();
  
 private:
  std::unique_ptr<DDCompactViewImpl> rep_;
  std::unique_ptr<DDPosData> worldpos_ ;
  
    // 2010-01-27 memory patch
    // for copying and protecting DD Store's after parsing is complete.
    DDI::Store<DDName, DDI::Material*> matStore_;
    DDI::Store<DDName, DDI::Solid*> solidStore_;
    DDI::Store<DDName, DDI::LogicalPart*> lpStore_;
    DDI::Store<DDName, DDI::Specific*> specStore_;
    DDI::Store<DDName, DDRotationMatrix*> rotStore_;    

};

//! global type for a compact-view walker
typedef DDCompactView::walker_type walker_type;
#endif
