#ifndef Parser_ComparisonBase_h
#define Parser_ComparisonBase_h
/* \class reco::parser::ComparisonBase
 *
 * Base class for comparison operator
 *
 * \author original version: Chris Jones, Cornell, 
 *         adapted by Luca Lista, INFN
 *
 * \version $Revision: 1.2 $
 *
 */

namespace reco {
  namespace parser {
    struct ComparisonBase {
      virtual ~ComparisonBase() {}
      virtual bool compare(double, double) const = 0;
    };
  }  // namespace parser
}  // namespace reco

#endif
