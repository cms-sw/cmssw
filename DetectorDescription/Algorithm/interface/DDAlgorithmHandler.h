#ifndef DD_DDALGORITHMHANDLER_H
#define DD_DDALGORITHMHANDLER_H

#include <string>
#include "DetectorDescription/Algorithm/interface/DDAlgorithm.h"
#include "DetectorDescription/Base/interface/DDException.h"

//! wrapper around a DDAlgorithm
/** used from DDParser for setting up, initializing, and executing an DDAlgorithm */
class DDAlgorithmHandler
{
 public:
  //! creates an DDAlgorithm wrapper
  /** @param a is a pointer to an DDAlgorithm object; 
      its memory is NOT managed by DDAlgorithmHandler */
  DDAlgorithmHandler();
  
  virtual  ~DDAlgorithmHandler();
  
  //! initializes the wrapped algorithm algo_ and does some pre- and post-processing
  /** pre- and postprocessing mainly covers exception handling,
      the algorithm object algo_ is fetched from the plugin-manager */
  void initialize(const std::string & algoName,
		  const DDLogicalPart & parent,
		  const DDNumericArguments & nArgs,
                  const DDVectorArguments & vArgs,
                  const DDMapArguments & mArgs,
                  const DDStringArguments & sArgs,
		  const DDStringVectorArguments & svArgs) throw (DDException);

  //! executes the wrapped algorithm algo_; some pre- and post-processing (exception handling)
  void execute( DDCompactView& ) throw (DDException);

 
 private:
  DDAlgorithm * algo_;   //!< the wrapped algorithm object
  std::string algoname_; //!< name of the algorithm object
  DDLogicalPart parent_; //!< parent logical part 
};

#endif //  DD_DDALGORITHMHANDLER_H
