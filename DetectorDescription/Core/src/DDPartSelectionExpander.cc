#include "DetectorDescription/Core/interface/DDPartSelectionExpander.h"

class DDPartSelection;

DDPartSelectionExpander::DDPartSelectionExpander(const DDCompactView & cpv)
 : graph_(cpv.graph())
 { }
 
size_t
DDPartSelectionExpander::expand(const DDPartSelection & , PartSelectionTree & ) const
{
   size_t count(0);
   return count;
}
