#ifndef DD_ALGO_PLUGIN_DD_ALGORITHM_H
#define DD_ALGO_PLUGIN_DD_ALGORITHM_H

#include "DetectorDescription/Core/interface/DDTypes.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include <vector>

class DDAlgorithmHandler;
class DDCompactView;

/** Abstract Base of all DDD algorithms. */
class DDAlgorithm
{
  friend class  DDAlgorithmHandler;
  
 public:
  virtual ~DDAlgorithm() {}

  //! fetch the algorithm parameters 
  /** an implementation of the initialize() method should initialize the algorithm
      by processing the provided parameters. Typically, data members of the
      derived algorithm are given meaningfull values. 
      Examples:\code double offset = nArgs["z-offset"];
      \code std::vector<double> vec = vArgs["X-Positions"];*/
  virtual void initialize( const DDNumericArguments & nArgs,
			   const DDVectorArguments & vArgs,
			   const DDMapArguments & mArgs,
			   const DDStringArguments & sArgs,
			   const DDStringVectorArguments & vsArgs ) = 0;

  //! execute the algorithm
  /** an implementation of the execute() method creates detector description
      objects such as DDLogicalPart, DDSolid, ... */
  virtual void execute( DDCompactView& ) = 0;

 protected:
   
  //! returns the parent logical-part under which the algorithm creates sub-structures
  const DDLogicalPart & parent() const { return parent_; }

  //! the algorithm may only create sub-detector elements within on parent
  void setParent( const DDLogicalPart & parent ) { parent_ = parent; }

 private:  
  //! parent part in which the algorithm creates sub-structures
  DDLogicalPart parent_;
};

#endif // DD_ALGO_PLUGIN_DD_ALGORITHM_H
