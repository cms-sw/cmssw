#include "DetectorDescription/Core/interface/DDExpandedNode.h"

#include <cassert>
#include <ostream>

#include "DetectorDescription/Core/interface/DDBase.h"
#include "DetectorDescription/Core/interface/DDName.h"
#include "DetectorDescription/Core/interface/DDPosData.h"

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
	   (posd_->copyno() == n.posd_->copyno()) );  
}	 		  		 
     
int DDExpandedNode::copyno() const 
{
  assert( posd_ );
  return posd_->copyno(); 
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
