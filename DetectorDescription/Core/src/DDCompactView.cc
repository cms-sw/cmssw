#include "DetectorDescription/Core/interface/DDCompactView.h"

#include <cstdlib>

#include "DetectorDescription/Core/interface/DDRotationMatrix.h"
#include "DetectorDescription/Core/interface/DDBase.h"
#include "DetectorDescription/Core/interface/DDCompactViewImpl.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDMaterial.h"
#include "DetectorDescription/Core/interface/DDPosData.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDSpecifics.h"
#include "DetectorDescription/Core/src/LogicalPart.h"
#include "DetectorDescription/Core/src/Material.h"
#include "DetectorDescription/Core/src/Solid.h"
#include "DetectorDescription/Core/src/Specific.h"

class DDDivision;

/** 
   Compact-views can be created only after an appropriate geometrical hierarchy
   has been defined using DDpos(). 
      
   Further it is assumed that the DDLogicalPart defining the root of the
   geometrical hierarchy has been defined using DDRootDef - singleton.
   It will be extracted from this singleton using DDRootDef::instance().root()!
      
   The first time this constructor gets called, the internal graph representation
   is build up. Subsequent calls will return immidiately providing access to
   the already built up compact-view representation.
      
   Currently the only usefull methods are DDCompactView::graph(), DDCompactView::root() !
       
   \todo define a stable interface for navigation (don't expose the user to the graph!)
*/    
// 
DDCompactView::DDCompactView(const DDLogicalPart & rootnodedata)
  : rep_( std::make_shared<DDCompactViewImpl>( rootnodedata )),
    worldpos_( std::make_shared<DDPosData>( DDTranslation(), DDRotation(), 0 ))
{}

/** 
   The compact-view is kept in an acyclic directed multigraph represented
   by an instance of class Graph<DDLogicalPart, DDPosData*). 
   Graph provides methods for navigating its content.
*/      
const DDCompactView::GraphType &
DDCompactView::graph() const 
{ 
  return rep_->graph(); 
}

const DDLogicalPart &
DDCompactView::root() const
{
  return rep_->root(); 
} 
  
const DDPosData*
DDCompactView::worldPosition() const
{
  return worldpos_.get();
}

DDCompactView::WalkerType DDCompactView::walker() const
{
  return rep_->walker();
}
    
/** 
   Example:
  
      \code
      // Fetch a compact-view
      DDCompactView view;
      
      // Fetch the part you want to weigh
      DDLogicalPart tracker(DDName("Tracker","tracker.xml"));
      
      // Weigh it
      edm::LogInfo ("DDCompactView") << "Tracker weight = " 
           << view.weight(tracker) / kg 
	   << " kg" << std::endl;
      \endcode
      
      The weight of all children is calculated as well.
*/    
double DDCompactView::weight(const DDLogicalPart & p) const
{
  return rep_->weight(p);
}  

void DDCompactView::position (const DDLogicalPart & self, 
			      const DDLogicalPart & parent,
			      const std::string& copyno,
			      const DDTranslation & trans,
			      const DDRotation & rot,
			      const DDDivision * div)
{
  int cpno = atoi(copyno.c_str());
  position(self,parent,cpno,trans,rot, div);
}

void DDCompactView::position (const DDLogicalPart & self,
			      const DDLogicalPart & parent,
			      int copyno,
			      const DDTranslation & trans,
			      const DDRotation & rot,
			      const DDDivision * div)
{
  rep_->position( self, parent, copyno, trans, rot, div );
}

DDCompactView::DDCompactView()
  : rep_( std::make_shared<DDCompactViewImpl>()),
    worldpos_( std::make_shared<DDPosData>( DDTranslation(), DDRotation(), 0 ))
{ }


