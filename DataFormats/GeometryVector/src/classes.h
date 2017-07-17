#include "DataFormats/GeometryVector/interface/Basic3DVector.h"
//

#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometryVector/interface/Point2DBase.h"
#include "DataFormats/GeometryVector/interface/Point3DBase.h"
#include "DataFormats/GeometryVector/interface/LocalTag.h"
#include "DataFormats/GeometryVector/interface/GlobalTag.h"
#include "DataFormats/GeometryVector/interface/LocalTag.h"
#include "DataFormats/GeometryVector/interface/MeasurementTag.h"
#include "DataFormats/GeometryVector/interface/Point2DBase.h"
#include "DataFormats/GeometryVector/interface/Point3DBase.h"
#include "DataFormats/GeometryVector/interface/Vector2DBase.h"
#include "DataFormats/GeometryVector/interface/Vector3DBase.h"

namespace DataFormats_GeometryVector {
  struct dictionary {
    Vector2DBase<float,MeasurementTag> dummy7;
    Vector3DBase<float,MeasurementTag> dummy6;
    Vector2DBase<float,GlobalTag> dummy8;
    Vector3DBase<float,GlobalTag> dummy9;
    Vector2DBase<float,LocalTag> dummy10;
    Vector3DBase<float,LocalTag> dummy11;
    Point2DBase<float,MeasurementTag> dummy12;
    Point3DBase<float,MeasurementTag> dummy13;
    Point2DBase<float,GlobalTag> dummy14;
    Point3DBase<float,GlobalTag> dummy15;
    Point2DBase<float,LocalTag> dummy16;
    Point3DBase<float,LocalTag> dummy17;
    Geom::Phi<float> dummy18;
    Geom::Theta<float> dummy19;
    Geom::Phi<double> dummy20;
    Geom::Theta<double> dummy21;
    Basic2DVector<double> dummy22;
    Basic3DVector<double> dummy23;
  };
}
