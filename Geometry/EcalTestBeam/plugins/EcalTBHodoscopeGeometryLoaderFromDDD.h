#ifndef GEOMETRY_ECALTBHODOSCOPEGEOMETRYLOADERFROMDDD_H
#define GEOMETRY_ECALTBHODOSCOPEGEOMETRYLOADERFROMDDD_H 1

#include "Geometry/CaloGeometry/interface/CaloVGeometryLoader.h"

#include "Geometry/EcalTestBeam/interface/EcalHodoscopeNumberingScheme.h"


/** \class EcalTBHodoscopeGeometryLoaderFromDDD
 *
 *
 *   
 * $Id: EcalTBHodoscopeGeometryLoaderFromDDD.h,v 1.1 2007/04/15 23:16:29 wmtan Exp $
 * \author P. Meridiani - INFN Roma 1
*/

class DDCompactView;
class DDFilteredView;
class DDFilter;
class CaloSubdetectorGeometry;
#include <memory>
#include <string>

class EcalTBHodoscopeGeometryLoaderFromDDD
{
 public:
  EcalTBHodoscopeGeometryLoaderFromDDD();

  virtual ~EcalTBHodoscopeGeometryLoaderFromDDD() 
    {
      if (_scheme)
	delete _scheme;
    };

  std::auto_ptr<CaloSubdetectorGeometry> load(const DDCompactView* cpv);  

 private:
  void makeGeometry(const DDCompactView* cpv,CaloSubdetectorGeometry* ebg);
  unsigned int getDetIdForDDDNode(const DDFilteredView &fv);
  std::string getDDDString(std::string s,DDFilteredView* fv);
  DDFilter* getDDFilter();
  EcalHodoscopeNumberingScheme* _scheme;
};

#endif
