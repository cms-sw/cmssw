#include "FastSimulation/TrackingRecHitProducer/interface/TrackerDetIdSelector.h"

#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"

#include <boost/config/warning_disable.hpp>
#include <boost/spirit/include/qi.hpp>
#include <boost/spirit/include/qi_rule.hpp>
#include <boost/spirit/include/qi_grammar.hpp>
#include <boost/phoenix.hpp>

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

namespace detail {
  ExpressionAST opGreater(ExpressionAST const& lhs, ExpressionAST const& rhs) {
    return BinaryOP(BinaryOP::OP::GREATER, lhs, rhs);
  }

  ExpressionAST opGreaterEq(ExpressionAST const& lhs, ExpressionAST const& rhs) {
    return BinaryOP(BinaryOP::OP::GREATER_EQUAL, lhs, rhs);
  }

  ExpressionAST opEq(ExpressionAST const& lhs, ExpressionAST const& rhs) {
    return BinaryOP(BinaryOP::OP::EQUAL, lhs, rhs);
  }

  ExpressionAST opLesserEq(ExpressionAST const& lhs, ExpressionAST const& rhs) {
    return BinaryOP(BinaryOP::OP::LESS_EQUAL, lhs, rhs);
  }

  ExpressionAST opLesser(ExpressionAST const& lhs, ExpressionAST const& rhs) {
    return BinaryOP(::BinaryOP::OP::LESS, lhs, rhs);
  }

  ExpressionAST opNotEq(ExpressionAST const& lhs, ExpressionAST const& rhs) {
    return BinaryOP(BinaryOP::OP::NOT_EQUAL, lhs, rhs);
  }

  ExpressionAST opAnd(ExpressionAST const& lhs, ExpressionAST const& rhs) {
    return BinaryOP(BinaryOP::OP::AND, lhs, rhs);
  }

  ExpressionAST opOr(ExpressionAST const& lhs, ExpressionAST const& rhs) {
    return BinaryOP(BinaryOP::OP::OR, lhs, rhs);
  }

}  // namespace detail
ExpressionAST& ExpressionAST::operator!() {
  expr = UnaryOP(UnaryOP::OP::NEG, expr);
  return *this;
}

template <typename ITERATOR>
struct TrackerDetIdSelectorGrammar : boost::spirit::qi::grammar<ITERATOR,
                                                                ExpressionAST(),
                                                                boost::spirit::ascii::space_type,
                                                                boost::spirit::qi::locals<ExpressionAST> > {
  boost::spirit::qi::rule<ITERATOR, std::string(), boost::spirit::ascii::space_type> identifierFctRule;

  boost::spirit::qi::rule<ITERATOR, ExpressionAST(), boost::spirit::ascii::space_type> identifierRule, expressionRule;

  boost::spirit::qi::
      rule<ITERATOR, ExpressionAST(), boost::spirit::ascii::space_type, boost::spirit::qi::locals<ExpressionAST> >
          comboRule;

  TrackerDetIdSelectorGrammar() : TrackerDetIdSelectorGrammar::base_type(comboRule) {
    namespace qi = boost::spirit::qi;
    namespace ascii = boost::spirit::ascii;
    namespace phoenix = boost::phoenix;

    identifierFctRule = qi::lexeme[+qi::alpha[qi::_val += qi::_1]];

    identifierRule = (qi::true_[qi::_val = 1] | qi::false_[qi::_val = 0]) | (qi::int_[qi::_val = qi::_1]) |
                     identifierFctRule[qi::_val = qi::_1];

    comboRule = (expressionRule[qi::_a = qi::_1] >>
                 *((qi::lit("&&") >> expressionRule[qi::_a = qi::_a && qi::_1]) |
                   (qi::lit("||") >> expressionRule[qi::_a = qi::_a || qi::_1])))[qi::_val = qi::_a];

    expressionRule = qi::lit("(") >> comboRule[qi::_val = qi::_1] >> qi::lit(")") |
                     (identifierRule >> qi::lit(">") >> identifierRule)[qi::_val = qi::_1 > qi::_2] |
                     (identifierRule >> qi::lit(">=") >> identifierRule)[qi::_val = qi::_1 >= qi::_2] |
                     (identifierRule >> qi::lit("<") >> identifierRule)[qi::_val = qi::_1 < qi::_2] |
                     (identifierRule >> qi::lit("<=") >> identifierRule)[qi::_val = qi::_1 <= qi::_2] |
                     (identifierRule >> qi::lit("==") >> identifierRule)[qi::_val = qi::_1 == qi::_2] |
                     (identifierRule >> qi::lit("!=") >> identifierRule)[qi::_val = qi::_1 != qi::_2] |
                     identifierRule[qi::_val = qi::_1];
  }
};

bool TrackerDetIdSelector::passSelection(const std::string& selectionStr) const {
  std::string::const_iterator begin = selectionStr.cbegin();
  std::string::const_iterator end = selectionStr.cend();

  TrackerDetIdSelectorGrammar<std::string::const_iterator> grammar;
  ExpressionAST exprAST;

  bool success = boost::spirit::qi::phrase_parse(begin, end, grammar, boost::spirit::ascii::space, exprAST);
  if (begin != end) {
    throw cms::Exception("FastSimulation/TrackingRecHitProducer/TrackerDetIdSelector",
                         "parsing selection '" + selectionStr + "' failed at " +
                             std::string(selectionStr.cbegin(), begin) + "^^^" + std::string(begin, end));
  }
  if (!success) {
    throw cms::Exception("FastSimulation/TrackingRecHitProducer/TrackerDetIdSelector",
                         "parsing selection '" + selectionStr + "' failed.");
  }
  /* Comment out for debugging
    WalkAST walker(_detId,_trackerTopology);
    walker(exprAST);
    std::cout<<std::endl;
    */
  return exprAST.evaluate(_detId, _trackerTopology);
}
