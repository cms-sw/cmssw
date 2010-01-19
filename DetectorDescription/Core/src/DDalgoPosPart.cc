//#include <cstdio>
#include "DetectorDescription/Core/interface/DDalgoPosPart.h"
#include "DetectorDescription/Core/interface/DDPosPart.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDAlgo.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/Base/interface/DDException.h"
#include "DetectorDescription/ExprAlgo/interface/AlgoPos.h"

// Message logger.
DDAlgoPositioner::DDAlgoPositioner ( DDCompactView * cpv ) : cpv_(cpv) 
{ }

DDAlgoPositioner::~DDAlgoPositioner () { }
void DDAlgoPositioner::operator()(const DDLogicalPart & self,
				  const DDLogicalPart & parent,
				  DDAlgo & algo
				  )
{
  cpv_->algoPosPart(self, parent, algo);
}
