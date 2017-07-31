#include "DetectorDescription/Core/interface/DDCompactViewImpl.h"

#include <math.h>
#include <ostream>
#include <utility>
#include <vector>

#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include "CLHEP/Units/SystemOfUnits.h"
#include "DetectorDescription/Core/interface/DDName.h"
#include "DetectorDescription/Core/interface/DDPosData.h"
#include "DataFormats/Math/interface/GraphWalker.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

DDCompactViewImpl::DDCompactViewImpl(const DDLogicalPart & rootnodedata)
  :  root_(rootnodedata)
{
  LogDebug("DDCompactViewImpl") << "Root node data = " << rootnodedata << std::endl;
}

DDCompactViewImpl::~DDCompactViewImpl() 
{  
   GraphNav::adj_list::size_type it = 0;
   if ( graph_.size() == 0 ) {
     LogDebug("DDCompactViewImpl") << "In destructor, graph is empty.  Root:" << root_ << std::endl;
   } else {
     LogDebug("DDCompactViewImpl") << "In destructor, graph is NOT empty.  Root:" << root_ << " graph_.size() = " << graph_.size() << std::endl;
     for (; it < graph_.size() ; ++it) {
       GraphNav::edge_range erange = graph_.edges(it); //it->second.begin();
       for(; erange.first != erange.second; ++(erange.first)) {
	 DDPosData * pd = graph_.edgeData(erange.first->second);
	 delete pd;
	 pd=0;
       }  
     }
   }
   edm::LogInfo("DDCompactViewImpl") << std::endl << "DDD transient representation has been destructed." << std::endl << std::endl;   
}

math::GraphWalker<DDLogicalPart,DDPosData*> DDCompactViewImpl::walker() const
{
  return math::GraphWalker<DDLogicalPart,DDPosData*>(graph_,root_);
}

#include "DetectorDescription/Core/interface/DDMaterial.h"
// calculates the weight and caches it in LogicalPartImpl
//double DDCompactViewImpl::weight(DDLogicalPart & part)
#include "DetectorDescription/Core/interface/DDSolid.h"

class DDDivision;

double DDCompactViewImpl::weight(const DDLogicalPart & aPart) const
{
 // return 0;

   if (!aPart)
     return -1;
   DDLogicalPart part = aPart;
   //double result;  
   if (part.weight())
     return part.weight();
   
   // weigth = (parent.vol - children.vol)*parent.density + weight(children)
   double childrenVols=0;
   double childrenWeights=0;
   WalkerType walker(graph_,part);
   if(walker.firstChild()) {
     bool doIt=true;
     while(doIt) {
       double a_vol;
       DDLogicalPart child(walker.current().first);
       a_vol=child.solid().volume();
       if (a_vol <= 0.) {
         edm::LogError("DDCompactViewImpl")  << "DD-WARNING: volume of solid=" << aPart.solid() 
	       << "is zero or negative, vol=" << a_vol/m3 << "m3" << std::endl;
       }
       childrenVols += a_vol;
       childrenWeights += weight(child); // recursive
       doIt=walker.nextSibling();
     }
   }
   
   double dens = part.material().density();
   if (dens <=0) {
     edm::LogError("DDCompactViewImpl")  << "DD-WARNING: density of material=" << part.material().ddname() 
	   << " is negative or zero, rho=" << dens/g*cm3 << "g/cm3" << std::endl;
   }
   double p_vol  = part.solid().volume();
   double w =   (p_vol - childrenVols)*dens + childrenWeights; 
   if (  (fabs(p_vol) - fabs(childrenVols))/fabs(p_vol) > 1.01 ) {
     edm::LogError("DDCompactViewImpl")  << "DD-WARNING: parent-volume smaller than children, parent=" 
          << part.ddname() << " difference-vol=" 
	   << (p_vol - childrenVols)/m3 << "m3, this is " 
	   << (childrenVols - p_vol)/p_vol << "% of the parent-vol." << std::endl;
   }
  
   //part.rep().weight_=w;
   part.weight() = w;
   return w;
   
}

void DDCompactViewImpl::position (const DDLogicalPart & self,
			      const DDLogicalPart & parent,
			      int copyno,
			      const DDTranslation & trans,
			      const DDRotation & rot,
			      const DDDivision * div)
{
  DDPosData * pd = new DDPosData(trans,rot,copyno,div);
  graph_.addEdge(parent,self,pd);
}


void DDCompactViewImpl::swap( DDCompactViewImpl& implToSwap ) {
  graph_.swap(implToSwap.graph_);
}

DDCompactViewImpl::DDCompactViewImpl() { }
