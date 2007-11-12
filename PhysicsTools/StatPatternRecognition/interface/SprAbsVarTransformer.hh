// File and Version Information:
//      $Id: SprAbsVarTransformer.hh,v 1.1 2007/11/07 00:56:14 narsky Exp $
//
// Description:
//      Class SprAbsVarTransformer :
//          Interface for classes defining variable transformations.
//
// Environment:
//      Software developed for the BaBar Detector at the SLAC B-Factory.
//
// Author List:
//      Ilya Narsky                     Original author
//
// Copyright Information:
//      Copyright (C) 2007              California Institute of Technology
//------------------------------------------------------------------------
 
#ifndef _SprAbsVarTransformer_HH
#define _SprAbsVarTransformer_HH

#include <vector>
#include <iostream>

class SprAbsFilter;


class SprAbsVarTransformer
{
public:
  virtual ~SprAbsVarTransformer() {}

  SprAbsVarTransformer() : oldVars_(), newVars_() {}

  /*
    VarTransformer name.
  */
  virtual std::string name() const = 0;

  /*
    Computes transformation using input data. 
    Returns true on success, false otherwise.
  */
  virtual bool train(const SprAbsFilter* data, int verbose=0) = 0;

  // Applies transformation.
  virtual void transform(const std::vector<double>& in,
			 std::vector<double>& out) const = 0;

  // Applies inverse transformation.
  virtual void inverse(const std::vector<double>& in,
		       std::vector<double>& out) const = 0;

  // Status of the transformer - if returns true, ready to transform.
  virtual bool ready() const = 0;

  // Variable access.
  void oldVars(std::vector<std::string>& vars) const { vars = oldVars_; }
  unsigned oldDim() const { return oldVars_.size(); }
  void newVars(std::vector<std::string>& vars) const { vars = newVars_; }
  unsigned newDim() const { return newVars_.size(); }
  void setOldVars(const std::vector<std::string>& vars) { oldVars_ = vars; }
  void setNewVars(const std::vector<std::string>& vars) { newVars_ = vars; }

  // Output
  virtual void print(std::ostream& os) const = 0;
  bool store(const char* filename) const;

protected:
  std::vector<std::string> oldVars_;
  std::vector<std::string> newVars_;
};

#endif
