#include "Geometry/HGCalCommonData/plugins/dd4hep/HGCalDD4HepHelper.h"
#include "DD4hep/DD4hepUnits.h"

const double HGCalDD4HepHelper::convert2mm(double length) { return (length / dd4hep::mm); }
