#ifndef CommonTools_Utils_ExpressionNumber_h
#define CommonTools_Utils_ExpressionNumber_h
/* \class reco::parser::ExpressionNumber
 *
 * Numberical expression
 *
 * \author original version: Chris Jones, Cornell, 
 *         adapted by Luca Lista, INFN
 *
 * \version $Revision: 1.2 $
 *
 */
#include "CommonTools/Utils/interface/parser/ExpressionBase.h"

namespace reco {
  namespace parser {
    struct ExpressionNumber : public ExpressionBase {
      double value(const edm::ObjectWithDict&) const override { return value_; }
      ExpressionNumber(double value) : value_(value) {}

    private:
      double value_;
    };
  }  // namespace parser
}  // namespace reco

#endif
