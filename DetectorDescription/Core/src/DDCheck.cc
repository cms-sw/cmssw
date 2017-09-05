#include "DetectorDescription/Core/src/DDCheck.h"

#include <map>

#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/Core/interface/DDExpandedView.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDMaterial.h"
#include "DetectorDescription/Core/interface/DDName.h"
#include "DetectorDescription/Core/interface/DDPosData.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDSolidShapes.h"
#include "DetectorDescription/Core/interface/DDTransform.h"
#include "DataFormats/Math/interface/Graph.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

bool DDCheckLP( const DDLogicalPart & lp , std::ostream & os )
{
   bool result = false;
   // is it defined or just declared?
   if (!lp) {
     os << "LogicalPart: " << lp << " is not defined!" << std::endl;
   }
   else {
     // check solid
     if (!lp.solid()) {
       os << "LogicalPart: " << lp << "| no solid defined, solid=" 
          << lp.solid() << std::endl;
     }
     else if(lp.solid().shape()==dd_not_init) {
       os << "LogicalPart: " << lp << "| solid not init, solid=" 
          << lp.solid() << std::endl;
     }
     // check material
     if (!lp.material()) {
       os << "LogicalPart: " << lp << "| no material defined, material=" 
          << lp.material() << std::endl;
     }
     else { // check consituents recursively
     
     }      
   }
   return result;
}


// checks PosData* if it contains sensfull stuff ...
//:void DDCheckPD(const DDLogicalPart & lp, graph_type::neighbour_type & nb, std::ostream & os)
bool DDCheckPD(const DDLogicalPart & lp, DDCompactView::graph_type::edge_range nb, const DDCompactView::graph_type & g, std::ostream & os)
{
   bool result = false;
   if (nb.first != nb.second) {
     for (; nb.first != nb.second; ++(nb.first)) {
       if (! nb.first->second ) {
         edm::LogInfo ("DDCheck") << "PosData of LogicalPart " << lp.name() << " missing." << std::endl;
	 edm::LogInfo ("DDCheck") << "  the LogicalPart is meant to be a daughter volume, but its position is missing!" << std::endl;
       } 
       else { // check for the rotation matrix being present
         const DDRotation & r = g.edgeData(nb.first->second)->ddrot();
	 if (! r.isDefined().second ) {
	 //if (! nb.first->second->rot_.rotation() ) {
	   const DDRotation & r = g.edgeData(nb.first->second)->ddrot();
	   os << "Rotationmatrix is not defined: " << r << std::endl;
	 }
       }
     }  
   }
   return result;
}   


bool DDCheckConnect(const DDCompactView & cpv, std::ostream & os)
{
  bool result = false;
  os << std::endl << "Checking connectivity of CompactView:" << std::endl;
  
  // Algorithm:
  // Pass 1: walk the whole graph, mark every visited LogicalPart-node
  // Pass 2: iterate over all nodes, check with marked nodes from Pass 1
  
  // Pass 1:
  std::map<DDLogicalPart,bool> visited;
  //  walker_type wkr = DDCompactView().walker();
  walker_type wkr = cpv.walker();
  visited[wkr.current().first]=true;
  while(wkr.next()) {
    //    std::cout << "DDCheck" << "   " << wkr.current().first << std::endl;
    visited[wkr.current().first]=true;
  }
  os << " CompactView has " << visited.size() 
     << " (multiple-)connected LogicalParts with root=" << cpv.root().ddname() << std::endl;
  
  // Pass 2:
  DDCompactView::graph_type & g = const_cast<DDCompactView::graph_type&>(cpv.graph());

  int uc = 0;
  DDCompactView::graph_type::adj_list::size_type it = 0;
  
  for(; it < g.size(); ++it) { 
    if (! visited[g.nodeData(it)] ) {
      ++uc;
      os << " " << g.nodeData(it).ddname();
    }
  }
  os << std::endl;
  os << " There were " << uc << " unconnected nodes found." << std::endl << std::endl;  
  if (uc) {
    os << std::endl;
    os << " ****************************************************************************" << std::endl;
    os << " WARNING: The geometrical hierarchy may not be complete. " << std::endl
       << "          Expect unpredicted behaviour using DDCore/interface (i.e. SEGFAULT)"
       << std::endl;
    os << " ****************************************************************************" << std::endl;
    os << std::endl;    
       
  }  
  return result;
}

// iterates over the whole compactview and chechs whether the posdata
// (edges in the acyclic graph ..) are complete
bool DDCheckAll(const DDCompactView & cpv, std::ostream & os)
{
   bool result = false;
   // const_cast because graph_type does not provide const_iterators!
   DDCompactView::graph_type & g = const_cast<DDCompactView::graph_type&>(cpv.graph());
   
   // basic debuggger
   std::map< std::pair<std::string,std::string>, int > lp_names;
   
   DDCompactView::graph_type::adj_list::size_type it = 0;
   for(; it < g.size(); ++it) { 
     const DDLogicalPart & lp = g.nodeData(it);
     lp_names[std::make_pair(lp.name().ns(),lp.name().name())]++;
   }
   
   for( const auto& mit : lp_names ) {
     if (mit.second >1) {
       os << "interesting: " << mit.first.first << ":" << mit.first.second
          << " counted " << mit.second << " times!" << std::endl;
       os << " Names of LogicalParts seem not to be unique!" << std::endl << std::endl;	  
       result = true;
	    
     }
   }
   // iterate over all nodes in the graph (nodes are logicalparts,
   // edges are posdata*
   for(it=0; it < g.size(); ++it) { 
     const DDLogicalPart & lp = g.nodeData(it);
     result |= DDCheckLP(lp,os);
     result |= DDCheckPD(lp ,g.edges(it), g, os);
   }
   
   // Check the connectivity of the graph..., takes quite some time & memory
   result |= DDCheckConnect(cpv, os);
   return result;
}

// comprehensive check, very cpu intensive!
// - expands the compact-view
// - detects cyclic containment of parts (not yet)
// - checks for completeness of solid & material definition / logical-part
bool DDCheck(std::ostream&os)
{
   bool result = false;
   os << "DDCore: start comprehensive checking" << std::endl;
   DDCompactView cpv; // THE one and only (prototype restriction) CompactView
   DDExpandedView exv(cpv);
   result |=  DDCheckAll(cpv,os);
   
   // done
   os << "DDCore: end of comprehensive checking" << std::endl;
   
   if (result) { // at least one error found
     edm::LogError("DDCheck") << std::endl << "DDD:DDCore:DDCheck: found inconsistency problems!" << std::endl;
   }
     	  
   return result;
}

bool DDCheck(const DDCompactView& cpv, std::ostream&os)
{
   bool result = false;
   os << "DDCore: start comprehensive checking" << std::endl;
   //   DDCompactView cpv; // THE one and only (prototype restriction) CompactView
   DDExpandedView exv(cpv);
   result |=  DDCheckAll(cpv,os);
   
   // done
   os << "DDCore: end of comprehensive checking" << std::endl;
   
   if (result) { // at least one error found
     edm::LogError("DDCheck") << std::endl << "DDD:DDCore:DDCheck: found inconsistency problems!" << std::endl;
   }
     	  
   return result;
}
