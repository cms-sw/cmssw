#include "CrystalCalo2DPositionCalcWithDepth.h"

typedef CrystalCalo2DPositionCalcWithDepth<DetId::Ecal,EcalShashlik> Shashlik2DPositionCalcWithDepth;

DEFINE_EDM_PLUGIN(PFCPositionCalculatorFactory,
		  Shashlik2DPositionCalcWithDepth,
		  "Shashlik2DPositionCalcWithDepth");



