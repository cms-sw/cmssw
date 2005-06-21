#ifndef DDPath_h
#define DDPath_h

namespace std{} using namespace std;

#include <vector>
#include <utility>
#include <string>
#include "DetectorDescription/DDCore/interface/DDLogicalPart.h"
#include "DetectorDescription/DDCore/interface/DDPartSelection.h"

//enum typ_t { ddunknown, ddanynode, ddanychild, ddanylogp, ddanyposp, ddchildlogp,  ddchildposp };
typedef ddselection_type typ_t;

struct   DDPathLevel {
    DDPathLevel(DDLogicalPart lp, int cp, typ_t t)
     : lp_(lp), copyno_(cp), typ_(t) { }
    
    DDLogicalPart lp_;
    
    int copyno_;
    
    typ_t typ_;
};


typedef vector< DDPathLevel > DDPath;

#endif
