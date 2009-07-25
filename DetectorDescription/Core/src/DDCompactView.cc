#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDRoot.h"
#include "DetectorDescription/Base/interface/DDdebug.h"
#include "DetectorDescription/Base/interface/DDException.h"

#include <iostream>

//DDCompactViewImpl * DDCompactView::global_ = 0;

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
DDCompactView::DDCompactView()
  : rep_(0)
{
    global();
}  

// NOTE TO SELF:  Mike, DO NOT try to fix the memory leak here by going global again!!!
DDCompactView::DDCompactView(const DDLogicalPart & rootnodedata)
  : rep_(new DDCompactViewImpl(rootnodedata))
  {
    // global_ = rep_;
  }

// Prototypish, to be used from within DDpos(...)
// It will produce a new, empty compact-view with NO root-defined
// Subsequent calls will be used for DDCore internal processing 
//  (e.g. fetching the compact-view after parsing and restructuring
//   it to get rid of refletions ...)
DDCompactView::DDCompactView(bool add)
  : rep_(0)
{ /*
   if (!global_) {
     global_ = new DDCompactViewImpl();
   }
   rep_ = global_; */
   global();
}    


DDCompactView::~DDCompactView() 
  {  
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


void DDCompactView::optimize()
{
  // deprecated!
  //int i;
}

DDCompactView::graph_type & DDCompactView::writeableGraph() 
{
  return const_cast<graph_type&>(rep_->graph());
}

    
void DDCompactView::print(std::ostream & os) const
{ 
  rep_->print(os); 
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

// >>---==========================<()>==========================---<<

// UNSTABLE STUFF below ...

void DDCompactView::global() 
{
 // DDLogicalPart l;
 // DDName n = l.ddname();
  static DDCompactViewImpl* global_=0;
  if (!global_) {
    DDLogicalPart rt = DDRootDef::instance().root();
    if(rt.isDefined().first) { 
      const DDName & tmp = rt.ddname();
      DDLogicalPart aRoot(tmp);
      global_ = new DDCompactViewImpl(aRoot); // FIXME!!!!!!!
    }  
  }
  // DCOUT('C', "DC: CompactView root=" << DDRootDef::instance().root() );
  if (!global_)
    throw DDException("Global CompactView construction failed ..");  
  global_->setRoot(DDRootDef::instance().root());

  rep_ = global_;
}


void DDCompactView::setRoot(const DDLogicalPart & root)
{
  rep_->setRoot(root);
}  

void DDCompactView::swap( DDCompactView& repToSwap ) {
  rep_->swap ( *(repToSwap.rep_) );
}
#include "DetectorDescription/Core/interface/DDMaterial.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDTransform.h"
//#include "DetectorDescription/Core/interface/DDPosPart.h"
#include "DetectorDescription/Core/interface/DDSpecifics.h"
//#include "DetectorDescription/ExprAlgo/interface/ExprEvalSingleton.h"

void DDCompactView::clear()
{

 graph_type & g = writeableGraph();
 
 graph_type::adj_iterator ait = g.begin();
 for (; ait != g.end(); ++ait) {
   graph_type::edge_list::iterator eit = ait->begin();
   for (; eit != ait->end(); ++eit) {
     //DDTranslation* tp = const_cast<DDTranslation*>(&(eit->second->trans_));
     //delete tp;
     delete g.edgeData(eit->second); eit->second=0;
   }
 } 

 DDMaterial::clear();
 DDLogicalPart::clear();
 DDSolid::clear();
 DDRotation::clear();
 DDSpecifics::clear();
 DDValue::clear(); 

 //NOT GOOD Practice! either! -- Mike Case
 LPNAMES::instance().clear();
 DIVNAMES::instance().clear();
 //NOT GOOD Practice! -- Mike Case
 DDName::Registry& reg_ = DDI::Singleton<DDName::Registry>::instance();
 reg_.clear();
 
/* 
 walker_type w(g,root());
 bool goOn=true;
 std::vector<DDPosData*> v;
 while (goOn) {
   graph_type::value_type c = w.current();
   //delete &(c.second->trans_);// &(c.second->trans_)=0;
   v.push_back(c.second); //c.second=0;
   goOn = w.next();
 }  
 std::vector<DDPosData*>::iterator it = v.begin();
 for (; it != v.end(); ++it) 
   delete *it;
*/   
//(mec:2007-06-07) Do not understand, but setting this to 0 caused memory crashes, like Ptr was being
// deleted twice or something?  I only get the error message when exiting iguana :-(
//(mec:2007-06-08) Got it.  In the destructor of XMLIdealGeometrySource I do the illegal thing and "grab"
// DDCompactView cpv; then I cpv.clear();  Since clear() sets this rep_=0, when the boost autopointer
// goes out of scope, this 0 representation causes it to blow up...  SOOO leave this problem until DD is
// re-written...
// rep_=0;  
}
