

#include "DetectorDescription/Core/interface/DDPartSelectionExpander.h"

#include <iostream>
#include <cassert>


DDPartSelectionExpander::DDPartSelectionExpander(const DDCompactView & cpv)
 : graph_(cpv.graph())
 { }
 
 
size_t DDPartSelectionExpander::expand(const DDPartSelection & /*input*/, PartSelectionTree & /*result*/) const
{
   //assert(input.size()>0);
   typedef DDCompactView::walker_type walker_type;
   size_t count(0);
   return count;
}
