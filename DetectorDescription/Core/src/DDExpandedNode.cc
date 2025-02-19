#include "DetectorDescription/Core/interface/DDExpandedNode.h"
#include "DetectorDescription/Core/interface/DDPosData.h"

DDExpandedNode::DDExpandedNode(const DDLogicalPart & lp, 
                               DDPosData * pd, 
	                       const DDTranslation & t, 
	                       const DDRotationMatrix & r,
			       int siblingno)
 : logp_(lp), posd_(pd), trans_(t), rot_(r), siblingno_(siblingno)
{ }


DDExpandedNode::~DDExpandedNode()
{ }   
   

bool DDExpandedNode::operator==(const DDExpandedNode & n) const {
  return ( (logp_==n.logp_) && 
	   (posd_->copyno_ == n.posd_->copyno_) ); 
  
}	 		  		 

     
int DDExpandedNode::copyno() const 
{ 
  return posd_->copyno_; 
}

#include <ostream>

std::ostream & operator<<(std::ostream & os, const DDExpandedNode & n)
{
  os << n.logicalPart().name() 
     << '[' << n.copyno() << ']';
     //<< ',' << n.siblingno()  << ']';
  return os;
}


std::ostream & operator<<(std::ostream & os, const DDGeoHistory & h)
{
   DDGeoHistory::const_iterator it = h.begin();
   for (; it != h.end(); ++it) {
     os << '/' << *it;
   }
   return os;
}

