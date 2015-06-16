#include "DetectorDescription/Core/interface/DDExpandedNode.h"
#include "DetectorDescription/Core/interface/DDPosData.h"
#include <ostream>

DDExpandedNode::DDExpandedNode(const DDLogicalPart & lp, 
                               const DDPosData * pd, 
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
  assert( posd_ );
  return posd_->copyno_; 
}

std::ostream & operator<<(std::ostream & os, const DDExpandedNode & n)
{
  os << n.logicalPart().name() 
     << '[' << n.copyno() << ']';
  return os;
}

std::ostream & operator<<(std::ostream & os, const DDGeoHistory & h)
{
   for( const auto& it : h ) {
     os << '/' << it;
   }
   return os;
}
