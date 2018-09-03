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
DDCompactView::DDCompactView( const DDLogicalPart & rootnodedata )
  : rep_( new DDCompactViewImpl( rootnodedata )),
    worldpos_( new DDPosData( DDTranslation(), DDRotation(), 0 ))
{
  DDMaterial::StoreT::instance().setReadOnly(false);
  DDSolid::StoreT::instance().setReadOnly(false);
  DDLogicalPart::StoreT::instance().setReadOnly(false);
  DDSpecifics::StoreT::instance().setReadOnly(false);
  DDRotation::StoreT::instance().setReadOnly(false);
}

DDCompactView::~DDCompactView() = default;

/** 
   The compact-view is kept in an acyclic directed multigraph represented
   by an instance of class Graph<DDLogicalPart, DDPosData*). 
   Graph provides methods for navigating its content.
*/      
const DDCompactView::Graph & DDCompactView::graph() const 
{ 
  return rep_->graph(); 
}

DDCompactView::GraphWalker
DDCompactView::walker() const
{
  return rep_->walker();
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

void
DDCompactView::position( const DDLogicalPart & self, 
			 const DDLogicalPart & parent,
			 const std::string& copyno,
			 const DDTranslation & trans,
			 const DDRotation & rot,
			 const DDDivision * div )
{
  int cpno = atoi( copyno.c_str());
  position( self, parent, cpno, trans, rot, div );
}

void
DDCompactView::position( const DDLogicalPart & self,
			 const DDLogicalPart & parent,
			 int copyno,
			 const DDTranslation & trans,
			 const DDRotation & rot,
			 const DDDivision * div )
{
  rep_->position( self, parent, copyno, trans, rot, div );
}

// UNSTABLE STUFF below ...
void DDCompactView::setRoot(const DDLogicalPart & root)
{
  rep_->setRoot(root);
}  

void DDCompactView::swap( DDCompactView& repToSwap ) {
  rep_->swap ( *(repToSwap.rep_) );
}

DDCompactView::DDCompactView()
  : rep_( new DDCompactViewImpl ),
    worldpos_( new DDPosData( DDTranslation(), DDRotation(), 0 ))
{}

void
DDCompactView::lockdown() {
  // FIXME: The store is not read-only
  // at this point we should have a valid store of DDObjects and we will move these
  // to the local storage area using swaps with the existing Singleton<Store...>'s
  DDMaterial::StoreT::instance().swap( matStore_ );
  DDSolid::StoreT::instance().swap( solidStore_ );
  DDLogicalPart::StoreT::instance().swap( lpStore_ );
  DDSpecifics::StoreT::instance().swap( specStore_ );
  DDRotation::StoreT::instance().swap( rotStore_ );

  // // 2010-01-27 memory patch
  // // not sure this will stay, but for now we want to explicitely lock the global stores.
  // // lock the global stores.
  // DDMaterial::StoreT::instance().setReadOnly(false);
  // DDSolid::StoreT::instance().setReadOnly(false);
  // DDLogicalPart::StoreT::instance().setReadOnly(false);
  // DDSpecifics::StoreT::instance().setReadOnly(false);
  // DDRotation::StoreT::instance().setReadOnly(false);

}

