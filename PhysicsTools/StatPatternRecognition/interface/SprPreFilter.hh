//------------------------------------------------------------------------
// File and Version Information:
//      $Id: SprPreFilter.hh,v 1.4 2007/10/22 21:23:40 narsky Exp $
//
// Description:
//      Class SprPreFilter :
//         User-defined filter applied.
//
// Environment:
//      Software developed for the BaBar Detector at the SLAC B-Factory.
//
// Author List:
//      Ilya Narsky                     Original author
//
// Copyright Information:
//      Copyright (C) 2007              California Institute of Technology
//
//------------------------------------------------------------------------

#ifndef _SprPreFilter_HH
#define _SprPreFilter_HH

#include <vector>
#include <string>
#include <utility>


class SprPreFilter
{
public:
  typedef std::vector<std::string> (*SprPreVars)();
  typedef bool (*SprPreSelection)(const std::vector<double>&);
  typedef void (*SprPreTransform)(const std::vector<double>&,
				  std::vector<double>&);
  typedef std::vector<int> (*SprPreClasses)();
  typedef int (*SprPreClassDefinition)(const std::vector<double>&);

  virtual ~SprPreFilter() {}

  SprPreFilter()
    : userSelectionVars_(), userSelection_(0), selectionVarToIndex_(),
      userTransformInputVars_(), userTransformOutputVars_(), 
      userTransform_(0), transformVarToIndex_(),
      userClasses_(),
      userClassVars_(), userClassDefinition_(0), classVarToIndex_()
  {}

  SprPreFilter(const SprPreFilter& other)
    :
    userSelectionVars_(other.userSelectionVars_),
    userSelection_(other.userSelection_),
    selectionVarToIndex_(other.selectionVarToIndex_),
    userTransformInputVars_(other.userTransformInputVars_),
    userTransformOutputVars_(other.userTransformOutputVars_),
    userTransform_(other.userTransform_),
    transformVarToIndex_(other.transformVarToIndex_),
    userClasses_(other.userClasses_),
    userClassVars_(other.userClassVars_),
    userClassDefinition_(other.userClassDefinition_),
    classVarToIndex_(other.classVarToIndex_)
  {}

  // Accept or reject a point given its class and coordinates.
  bool pass(int icls, const std::vector<double>& input) const;


  // Transform variable names.
  bool transformVars(const std::vector<std::string>& input, 
		     std::vector<std::string>& output) const;

  // Transform point coordinates.
  bool transformCoords(const std::vector<double>& input, 
		       std::vector<double>& output) const;

  // Compute the user-defined class for this event.
  std::pair<int,bool> computeClass(const std::vector<double>& input) const;

  // Supply variables from input data.
  // This method needs to be called after the set methods.
  bool setVars(const std::vector<std::string>& vars);

  /*
    Note: selection is always applied before transformation!!!
    Selection requirements are imposed on untransformed variables!
  */

  // define user selection
  bool setSelection(SprPreVars userVars, 
		    SprPreSelection selection,
		    SprPreClasses classes);

  // define user transformation
  bool setTransform(SprPreVars inputVars,
		    SprPreVars outputVars,
		    SprPreTransform transform);

  // define user-defined class
  bool setClass(SprPreVars userVars,
		SprPreClassDefinition classDefinition);

protected:
  bool resetSelection();
  bool resetTransform();
  bool resetClass();
  bool setVarIndex(const std::vector<std::string>& dataVars,
		   const std::vector<std::string>& userVars,
		   std::vector<int>& indexMap);

  std::vector<std::string> userSelectionVars_;
  SprPreSelection userSelection_;
  std::vector<int> selectionVarToIndex_;

  std::vector<std::string> userTransformInputVars_;
  std::vector<std::string> userTransformOutputVars_;
  SprPreTransform userTransform_;
  std::vector<int> transformVarToIndex_;

  std::vector<int> userClasses_;

  std::vector<std::string> userClassVars_;
  SprPreClassDefinition userClassDefinition_;
  std::vector<int> classVarToIndex_;
};

#endif
