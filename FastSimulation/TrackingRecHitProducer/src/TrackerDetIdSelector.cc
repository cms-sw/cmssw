#include "FastSimulation/TrackingRecHitProducer/interface/TrackerDetIdSelector.h"

#define DETIDFCT(NAME) NAME, [](const TrackerTopology& trackerTopology, const DetId& detId) -> int

#define TOPOFCT(NAME) \
#NAME, [](const TrackerTopology& trackerTopology, const DetId& detId) -> int { return trackerTopology.NAME(detId); }

const TrackerDetIdSelector::StringFunctionMap TrackerDetIdSelector::functionTable = {

    {DETIDFCT("subdetId"){return (uint32_t)detId.subdetId();
}
}
,

    {DETIDFCT("BPX"){return PixelSubdetector::PixelBarrel;
}
}
, {DETIDFCT("FPX"){return PixelSubdetector::PixelEndcap;
}
}
, {DETIDFCT("TIB"){return StripSubdetector::TIB;
}
}
, {DETIDFCT("TID"){return StripSubdetector::TID;
}
}
, {DETIDFCT("TOB"){return StripSubdetector::TOB;
}
}
, {DETIDFCT("TEC"){return StripSubdetector::TEC;
}
}
,

    {DETIDFCT("BARREL"){return 0;
}
}
, {DETIDFCT("ZMINUS"){return 1;
}
}
, {DETIDFCT("ZPLUS"){return 2;
}
}
,

    {TOPOFCT(layer)}, {TOPOFCT(module)}, {TOPOFCT(side)},

    {TOPOFCT(pxbLadder)}, {TOPOFCT(pxbLayer)}, {TOPOFCT(pxbModule)},

    {TOPOFCT(pxfBlade)}, {TOPOFCT(pxfDisk)}, {TOPOFCT(pxfModule)}, {TOPOFCT(pxfPanel)}, {TOPOFCT(pxfSide)},

    {TOPOFCT(tibGlued)}, {TOPOFCT(tibIsDoubleSide)}, {TOPOFCT(tibIsExternalString)}, {TOPOFCT(tibIsInternalString)},
    {TOPOFCT(tibIsRPhi)}, {TOPOFCT(tibIsStereo)}, {TOPOFCT(tibIsZMinusSide)}, {TOPOFCT(tibIsZPlusSide)},
    {TOPOFCT(tibLayer)}, {TOPOFCT(tibModule)}, {TOPOFCT(tibOrder)}, {TOPOFCT(tibSide)}, {TOPOFCT(tibStereo)},
    {TOPOFCT(tibString)},

    {TOPOFCT(tidGlued)}, {TOPOFCT(tidIsBackRing)}, {TOPOFCT(tidIsDoubleSide)}, {TOPOFCT(tidIsFrontRing)},
    {TOPOFCT(tidIsRPhi)}, {TOPOFCT(tidIsStereo)}, {TOPOFCT(tidIsZMinusSide)}, {TOPOFCT(tidIsZPlusSide)},
    {TOPOFCT(tidModule)}, {TOPOFCT(tidOrder)}, {TOPOFCT(tidRing)}, {TOPOFCT(tidSide)}, {TOPOFCT(tidStereo)},
    {TOPOFCT(tidWheel)},

    {TOPOFCT(tobGlued)}, {TOPOFCT(tobIsDoubleSide)}, {TOPOFCT(tobIsRPhi)}, {TOPOFCT(tobIsStereo)},
    {TOPOFCT(tobIsZMinusSide)}, {TOPOFCT(tobIsZPlusSide)}, {TOPOFCT(tobLayer)}, {TOPOFCT(tobModule)}, {TOPOFCT(tobRod)},
    {TOPOFCT(tobSide)}, {TOPOFCT(tobStereo)},

    {TOPOFCT(tecGlued)}, {TOPOFCT(tecIsBackPetal)}, {TOPOFCT(tecIsDoubleSide)}, {TOPOFCT(tecIsFrontPetal)},
    {TOPOFCT(tecIsRPhi)}, {TOPOFCT(tecIsStereo)}, {TOPOFCT(tecIsZMinusSide)}, {TOPOFCT(tecIsZPlusSide)},
    {TOPOFCT(tecModule)}, {TOPOFCT(tecOrder)}, {TOPOFCT(tecPetalNumber)}, {TOPOFCT(tecRing)}, {TOPOFCT(tecSide)},
    {TOPOFCT(tecStereo)}, {
  TOPOFCT(tecWheel)
}
}
;

ExpressionAST operator>(ExpressionAST const& lhs, ExpressionAST const& rhs) {
  ExpressionAST ast = BinaryOP(BinaryOP::OP::GREATER, lhs, rhs);
  return ast;
}

ExpressionAST operator>=(ExpressionAST const& lhs, ExpressionAST const& rhs) {
  ExpressionAST ast = BinaryOP(BinaryOP::OP::GREATER_EQUAL, lhs, rhs);
  return ast;
}

ExpressionAST operator==(ExpressionAST const& lhs, ExpressionAST const& rhs) {
  ExpressionAST ast = BinaryOP(BinaryOP::OP::EQUAL, lhs, rhs);
  return ast;
}

ExpressionAST operator<=(ExpressionAST const& lhs, ExpressionAST const& rhs) {
  ExpressionAST ast = BinaryOP(BinaryOP::OP::LESS_EQUAL, lhs, rhs);
  return ast;
}

ExpressionAST operator<(ExpressionAST const& lhs, ExpressionAST const& rhs) {
  ExpressionAST ast = BinaryOP(::BinaryOP::OP::LESS, lhs, rhs);
  return ast;
}

ExpressionAST operator!=(ExpressionAST const& lhs, ExpressionAST const& rhs) {
  ExpressionAST ast = BinaryOP(BinaryOP::OP::NOT_EQUAL, lhs, rhs);
  return ast;
}

ExpressionAST operator&&(ExpressionAST const& lhs, ExpressionAST const& rhs) {
  ExpressionAST ast = BinaryOP(BinaryOP::OP::AND, lhs, rhs);
  return ast;
}

ExpressionAST operator||(ExpressionAST const& lhs, ExpressionAST const& rhs) {
  ExpressionAST ast = BinaryOP(BinaryOP::OP::OR, lhs, rhs);
  return ast;
}

ExpressionAST& ExpressionAST::operator!() {
  expr = UnaryOP(UnaryOP::OP::NEG, expr);
  return *this;
}
