#include "FastSimulation/TrackingRecHitProducer/interface/TrackerDetIdSelector.h"

#define DETIDFCT(NAME) \
    NAME ,[](const TrackerTopology& trackerTopology, const DetId& detId) -> int

const TrackerDetIdSelector::StringFunctionMap TrackerDetIdSelector::functionTable = {

    {DETIDFCT("subdetId"){return detId.subdetId();}},

    {DETIDFCT("BPX"){return PixelSubdetector::PixelBarrel;}},
    {DETIDFCT("FPX"){return PixelSubdetector::PixelEndcap;}},
    {DETIDFCT("TIB"){return StripSubdetector::TIB;}},
    {DETIDFCT("TID"){return StripSubdetector::TID;}},
    {DETIDFCT("TOB"){return StripSubdetector::TOB;}},
    {DETIDFCT("TEC"){return StripSubdetector::TEC;}},

    {DETIDFCT("layer"){return trackerTopology.layer(detId);}},
    {DETIDFCT("module"){return trackerTopology.module(detId);}},
    {DETIDFCT("side"){return trackerTopology.side(detId);}},

    {DETIDFCT("pxbLayer"){return trackerTopology.pxbLayer(detId);}},
    {DETIDFCT("pxbLadder"){return trackerTopology.pxbLadder(detId);}},
    {DETIDFCT("pxbModule"){return trackerTopology.pxbModule(detId);}}
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
    ExpressionAST ast = BinaryOP(BinaryOP::OP::AND, lhs, rhs);
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
