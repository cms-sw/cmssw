/****************************************************************************
*
* Authors:
*	Jan Kaspar (jan.kaspar@gmail.com) 
*
****************************************************************************/

#ifndef Geometry_VeryForwardGeometryBuilder_DDDTotemRPConstruction
#define Geometry_VeryForwardGeometryBuilder_DDDTotemRPConstruction

#include "Geometry/VeryForwardGeometryBuilder/interface/DetGeomDesc.h"
#include "Geometry/VeryForwardGeometryBuilder/interface/DDDTotemRPCommon.h"

class DDCompactView;
class DDFilteredView;


/**
 * \brief Builds structure of DetGeomDesc objects out of DDCompactView (resp. DDFilteredView).
 *
 * It adds detector IDs (via class TotemRPDetId).
 * intended to be called from: modul TotemRPDetGeomDescESModule.
 **/

class DDDTotemRPContruction {
	public:
		DDDTotemRPContruction();
		const DetGeomDesc* construct(const DDCompactView* cpv);

	protected:
		void buildDetGeomDesc(DDFilteredView *fv, DetGeomDesc *gd);
};

#endif // Geometry_VeryForwardGeometryBuilder_DDDTotemRPConstruction
