#include "DetectorDescription/Core/interface/DDPartSelectionExpander.h"

DDPartSelectionExpander::DDPartSelectionExpander(const DDCompactView & cpv)
 : graph_(cpv.graph())
 { }
 
size_t DDPartSelectionExpander::expand(const DDPartSelection & /*input*/, PartSelectionTree & /*result*/) const
{
   size_t count(0);
   return count;
}
