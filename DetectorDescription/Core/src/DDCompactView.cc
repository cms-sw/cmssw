#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDMaterial.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDRoot.h"
#include "DetectorDescription/Core/interface/DDName.h"
#include "DetectorDescription/Core/interface/DDAlgo.h"
#include "DetectorDescription/ExprAlgo/interface/AlgoPos.h"
#include "DetectorDescription/Base/interface/DDdebug.h"
#include "DetectorDescription/Base/interface/DDException.h"

#include <iostream>

//DDCompactViewmpl * DDCompactView::global_ = 0;

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
  : rep_(new DDCompactViewImpl(rootnodedata))
{ }

DDCompactView::~DDCompactView() 
{  
  if (rep_ == 0) {
    std::cout << "deleting DDCompactView and found it to be 0 already (i.e. swapped earlier, I believe" << std::endl;
  } else {
    delete rep_;
  }
  DCOUT_V('C',"DC: Deleting compact VIEW!!!!!!!!! <<<<<<<<<<<<<<<=================="); 
}


/** 
   The compact-view is kept in an acyclic directed multigraph represented
   by an instance of class Graph<DDLogicalPart, DDPosData*). 
   Graph provides methods for navigating its content.
*/      
const DDCompactView::graph_type & DDCompactView::graph() const 
{ 
  return rep_->graph(); 
}

DDCompactView::graph_type & DDCompactView::writeableGraph() 
{
  return const_cast<graph_type&>(rep_->graph());
}

const DDLogicalPart & DDCompactView::root() const
{ 
  return rep_->root(); 
} 


DDCompactView::walker_type DDCompactView::walker() const
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

void DDCompactView::algoPosPart(const DDLogicalPart & self,
				const DDLogicalPart & parent,
				DDAlgo & algo
				) {
  static int cnt_=0;
  ++cnt_;
  if (algo.rep().numRegistered() == 0) {
    std::string e;
    e = "DDalgoPosPart: algorithmic positioning\n";
    e += "\t[" + algo.name().ns() 
               + ":" 
	       + algo.name().name() 
	       + "] is not defined!\n";
    throw DDException(e);
  }
  
  LogDebug ("AlgoPos")  << "DDAlgoPositioner, algo=" << std::endl << algo << std::endl;
  int inner=0;
  do { 
    ++inner;
    DDRotationMatrix * rmp = new DDRotationMatrix(algo.rotation());
    DDRotation anonymRot = DDanonymousRot(rmp);
    DDTranslation tr(algo.translation());
    position(self, parent, algo.label(), tr, anonymRot); 
    algo.next();
  } 
  while(algo.go());

}

void DDCompactView::position (const DDLogicalPart & self, 
			      const DDLogicalPart & parent,
			      std::string copyno,
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


// >>---==========================<()>==========================---<<

// UNSTABLE STUFF below ...
void DDCompactView::setRoot(const DDLogicalPart & root)
{
  rep_->setRoot(root);
}  

void DDCompactView::swap( DDCompactView& repToSwap ) {
  rep_->swap ( *(repToSwap.rep_) );
}

DDCompactView::DDCompactView() : rep_(new DDCompactViewImpl) { }
