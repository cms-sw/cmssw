#ifndef DDPosPart_h
#define DDPosPart_h
/** \file Interface for construction geometrical hierarchies
*/


#include <string>
#include <iostream>
#include "DetectorDescription/Core/interface/DDName.h"
#include "DetectorDescription/Core/interface/DDTransform.h"

class DDLogicalPart;
class DDRotation;
class DDDivision;


/**
   root of the detector is never positioned using DDpos,
   it's implicitly declared by the first layer of daughters!
*/
//! Function for building geometrical hierarchies 
/** The internal model of the geometrical hierarchy is an acylic directed
    multigraph. Users have access to it by consulting DDCompactView. 
    
    DDPos() is used to define 2 nodes (\a self and \a mother) and a connecting
    edge (the relative translations \a trans and rotation \a rot of \a self towards
    \a mother).

    \arg \c self DDName of DDLogicalPart corresponding to the volume to be positions
    \arg \c mother DDName of DDLogicalPart corresponding to the volume \c self is positioned into
    \arg \c trans relative translation of \c self towards \c mother
    \arg \c rot relative rotation of \c self toward \c mother
    \arg \c div is the division algorithm that made this pos part.
    
    The root of the geometrical hierarchy is implicitly defined. It never shows
    up as a \a self parameter in DDPos().

    \todo Check: After calling DDPos the user must not free allocated memory of \a trans .
*/ 
void DDpos(const DDLogicalPart & self,
           const DDLogicalPart & parent,
	   std::string copyno,
	   const DDTranslation & trans,
	   const DDRotation & rot,
	   const DDDivision * div = NULL);

void DDpos(const DDLogicalPart & self,
           const DDLogicalPart & parent,
	   int copyno,
	   const DDTranslation & trans,
	   const DDRotation & rot,
	   const DDDivision * div = NULL);
	   
#endif
