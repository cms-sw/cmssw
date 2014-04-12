#include "Geometry/ForwardGeometry/interface/ZdcGeometry.h"
#include "Geometry/ForwardGeometry/interface/CastorGeometry.h"

#include "Geometry/CaloEventSetup/interface/CaloGeometryDBEP.h"
#include "Geometry/CaloEventSetup/interface/CaloGeometryDBReader.h"

template class CaloGeometryDBEP< ZdcGeometry    , CaloGeometryDBReader > ;
template class CaloGeometryDBEP< CastorGeometry , CaloGeometryDBReader > ;

typedef CaloGeometryDBEP< ZdcGeometry , CaloGeometryDBReader > ZdcGeometryFromDBEP ;

DEFINE_FWK_EVENTSETUP_MODULE(ZdcGeometryFromDBEP);

typedef CaloGeometryDBEP< CastorGeometry , CaloGeometryDBReader > CastorGeometryFromDBEP ;

DEFINE_FWK_EVENTSETUP_MODULE(CastorGeometryFromDBEP);
