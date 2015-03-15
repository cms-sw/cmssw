#include "FastSimulation/TrackingRecHitProducer/interface/TrackerDetIdSelector.h"

const TrackerDetIdSelector::StringFunctionMap TrackerDetIdSelector::_functions = {

    {"subdetId",[](const TrackerTopology& trackerTopology, const DetId& detId) -> int {return detId.subdetId();}},
    {"BPX",[](const TrackerTopology& trackerTopology, const DetId& detId) -> int {return PixelSubdetector::PixelBarrel;}},
    {"FPX",[](const TrackerTopology& trackerTopology, const DetId& detId) -> int {return PixelSubdetector::PixelEndcap;}},
    {"TIB",[](const TrackerTopology& trackerTopology, const DetId& detId) -> int {return StripSubdetector::TIB;}},
    {"TID",[](const TrackerTopology& trackerTopology, const DetId& detId) -> int {return StripSubdetector::TID;}},
    {"TOB",[](const TrackerTopology& trackerTopology, const DetId& detId) -> int {return StripSubdetector::TOB;}},
    {"TEC",[](const TrackerTopology& trackerTopology, const DetId& detId) -> int {return StripSubdetector::TEC;}},
    {"pxbLayer",[](const TrackerTopology& trackerTopology, const DetId& detId) -> int {return trackerTopology.pxbLayer(detId);}},
    {"pxbLadder",[](const TrackerTopology& trackerTopology, const DetId& detId) -> int {return trackerTopology.pxbLadder(detId);}},
    {"pxbModule",[](const TrackerTopology& trackerTopology, const DetId& detId) -> int {return trackerTopology.pxbModule(detId);}}

};


ExpressionAST operator>(ExpressionAST const& lhs, ExpressionAST const& rhs)
{
    ExpressionAST ast = BinaryOP(BinaryOP::OP::GREATER, lhs, rhs);
    return ast;
}

ExpressionAST operator>=(ExpressionAST const& lhs, ExpressionAST const& rhs)
{
    ExpressionAST ast = BinaryOP(BinaryOP::OP::GREATER_EQUAL, lhs, rhs);
    return ast;
}

ExpressionAST operator==(ExpressionAST const& lhs, ExpressionAST const& rhs)
{
    ExpressionAST ast =  BinaryOP(BinaryOP::OP::EQUAL, lhs, rhs);
    return ast;
}

ExpressionAST operator<=(ExpressionAST const& lhs, ExpressionAST const& rhs)
{
    ExpressionAST ast = BinaryOP(BinaryOP::OP::LESS_EQUAL, lhs, rhs);
    return ast;
}

ExpressionAST operator<(ExpressionAST const& lhs, ExpressionAST const& rhs)
{
    ExpressionAST ast = BinaryOP(::BinaryOP::OP::LESS, lhs, rhs);
    return ast;
}

ExpressionAST operator!=(ExpressionAST const& lhs, ExpressionAST const& rhs)
{
    ExpressionAST ast = BinaryOP(BinaryOP::OP::NOT_EQUAL, lhs, rhs);
    return ast;
}

ExpressionAST operator&(ExpressionAST const& lhs, ExpressionAST const& rhs)
{
    ExpressionAST ast = BinaryOP(::BinaryOP::OP::AND, lhs, rhs);
    return ast;
}

ExpressionAST operator|(ExpressionAST const& lhs, ExpressionAST const& rhs)
{
    ExpressionAST ast = BinaryOP(BinaryOP::OP::OR, lhs, rhs);
    return ast;
}


ExpressionAST& ExpressionAST::operator!()
{
    expr = UnaryOP(UnaryOP::OP::NEG, expr);
    return *this;
}
