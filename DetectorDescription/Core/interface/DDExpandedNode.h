#ifndef DDExpandedNode_h
#define DDExpandedNode_h

#include <vector>
#include <iostream>
#include "DetectorDescription/DDCore/interface/DDTransform.h"
#include "DetectorDescription/DDCore/interface/DDLogicalPart.h"
class DDExpandedView;
class DDPosData;
struct DDExpandedNodeLess;

//! represents one node in the DDExpandedView
class DDExpandedNode
{
   friend class DDExpandedView;
   friend class DDExpandedNodeLess;

public:
  DDExpandedNode(const DDLogicalPart & lp, 
                 DDPosData * pd, 
	         const DDTranslation & t, 
	         const DDRotationMatrix & r,
		 int siblingno);
 
  ~DDExpandedNode();
  
  bool operator==(const DDExpandedNode & n) const;				 		  		 
  
  //! the LogicalPart describing this node    
  const DDLogicalPart & logicalPart() const;
  
  //! absolute translation of this node   
  const DDTranslation & absTranslation() const;
  
  //! absolute rotation of this node  
  const DDRotationMatrix & absRotation() const;
  
  //! copy number of this node  
  int copyno() const;
  
  //! sibling number of this node
  int siblingno() const;
  
  const DDPosData * posdata() const { return posd_; }
   
private:
  DDLogicalPart logp_; // logicalpart to provide access to solid & material information
  DDPosData * posd_;
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
     const DDTranslation & t1 = n1.trans_;
     const DDTranslation & t2 = n2.trans_;
     
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
     else if (n1.logp_.ddname() < n2.logp_.ddname())
     {
       result=true;
     }  
     
     return result;
  }
  
};


//! Geometrical 'path' of the current node up to the root-node
typedef vector<DDExpandedNode> DDGeoHistory; 

ostream & operator<<(ostream &, const DDExpandedNode &);
ostream & operator<<(ostream &, const DDGeoHistory &);
#endif 

