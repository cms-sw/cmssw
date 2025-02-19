#ifndef DDPath_h
#define DDPath_h

#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDPartSelection.h"

#include <vector>
#include <utility>
#include <string>

//enum typ_t { ddunknown, ddanynode, ddanychild, ddanylogp, ddanyposp, ddchildlogp,  ddchildposp };
typedef ddselection_type typ_t;

struct   DDPathLevel {
    DDPathLevel(DDLogicalPart lp, int cp, typ_t t)
     : lp_(lp), copyno_(cp), typ_(t) { }
    
    DDLogicalPart lp_;
    
    int copyno_;
    
    typ_t typ_;
};


typedef std::vector< DDPathLevel > DDPath;

#endif
