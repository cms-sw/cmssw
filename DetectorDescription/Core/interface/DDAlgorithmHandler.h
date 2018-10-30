#ifndef DD_DDALGORITHMHANDLER_H
#define DD_DDALGORITHMHANDLER_H

#include <string>

#include "DetectorDescription/Core/interface/DDAlgorithm.h"
#include "DetectorDescription/Core/interface/DDTypes.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"

class DDAlgorithm;
class DDCompactView;

//! wrapper around a DDAlgorithm
/** used from DDParser for setting up, initializing, and executing an DDAlgorithm */
class DDAlgorithmHandler
{
 public:
  //! creates an DDAlgorithm wrapper
  DDAlgorithmHandler() {}
  
  //! initializes the wrapped algorithm algo_ and does some pre- and post-processing
  /** pre- and postprocessing mainly covers exception handling,
      the algorithm object algo_ is fetched from the plugin-manager */
  void initialize( const DDName & algoName,
		   const DDLogicalPart & parent,
		   const DDNumericArguments & nArgs,
		   const DDVectorArguments & vArgs,
		   const DDMapArguments & mArgs,
		   const DDStringArguments & sArgs,
		   const DDStringVectorArguments & svArgs);
  
  //! executes the wrapped algorithm algo_; some pre- and post-processing (exception handling)
  void execute( DDCompactView& );

 private:
  std::unique_ptr<DDAlgorithm> algo_;   //!< the wrapped algorithm object
};

#endif //  DD_DDALGORITHMHANDLER_H
