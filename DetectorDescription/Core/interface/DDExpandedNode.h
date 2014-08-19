#ifndef DDExpandedNode_h
#define DDExpandedNode_h

#include "DetectorDescription/Core/interface/DDTransform.h"
#include "DetectorDescription/Base/interface/DDTranslation.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include <vector>
#include <iosfwd>

class DDExpandedView;
struct DDPosData;

//! represents one node in the DDExpandedView
class DDExpandedNode
{
   friend class DDExpandedView;

public:
  DDExpandedNode(const DDLogicalPart & lp, 
                 const DDPosData * pd, 
	         const DDTranslation & t, 
	         const DDRotationMatrix & r,
		 int siblingno);
 
  ~DDExpandedNode();
  

  bool operator==(const DDExpandedNode & n) const;
  
  //! the LogicalPart describing this node    
  const DDLogicalPart & logicalPart() const { return logp_; }
  
  
  //! absolute translation of this node   
  const DDTranslation & absTranslation() const { return trans_; }
  
  
  //! absolute rotation of this node  
  const DDRotationMatrix & absRotation() const { return rot_; }
    
  //! copy number of this node  
  int copyno() const;
  
  //! sibling number of this node
  int siblingno() const { return siblingno_; }  
  
  
  const DDPosData * posdata() const { return posd_; }
   
private:
  DDLogicalPart logp_; // logicalpart to provide access to solid & material information
  const DDPosData * posd_;
  DDTranslation trans_;  // absolute translation
  DDRotationMatrix rot_; // absolute rotation
  int siblingno_; // internal sibling-numbering from 0 to max-sibling
};



//! function object to compare to ExpandedNodes 
/**
  compares (for STL usage) two DDExpandedNodes for 
*/
struct DDExpandedNodeLess
{

  bool operator()(const DDExpandedNode & n1, const DDExpandedNode & n2)
  {
     const DDTranslation & t1 = n1.absTranslation();
     const DDTranslation & t2 = n2.absTranslation();
     
     bool result = false;
     
     // 'alphabetical ordering' according to absolute position
     
     if (t1.z() < t2.z()) 
     {
       result=true;
     } 
     else if ( (t1.z()==t2.z()) && (t1.y() < t2.y()))
     {
       result=true;
     }  
     else if ( (t1.z()==t2.z()) && (t1.y()==t2.y()) && (t1.x()<t2.x()))
     {
       result=true;
     } 
     else if (n1.siblingno() < n2.siblingno())
     {
       result=true;
     }
     else if (n1.logicalPart().ddname() < n2.logicalPart().ddname())
     {
       result=true;
     }  
     
     return result;
  }
  
};


//! Geometrical 'path' of the current node up to the root-node
typedef std::vector<DDExpandedNode> DDGeoHistory; 

std::ostream & operator<<(std::ostream &, const DDExpandedNode &);
std::ostream & operator<<(std::ostream &, const DDGeoHistory &);
#endif 

