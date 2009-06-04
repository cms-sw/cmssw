#include "LogicalBinaryOperator.h"
using namespace reco::parser;
template <>
bool LogicalBinaryOperator<std::logical_and<bool> >::operator()(const ROOT::Reflex::Object &o) const {
   return (*lhs_)(o) && (*rhs_)(o);
}
template <>
bool LogicalBinaryOperator<std::logical_or<bool> >::operator()(const ROOT::Reflex::Object &o) const {
   return (*lhs_)(o) || (*rhs_)(o);
}

