//------------------------------------------------------------------------
// File and Version Information:
//      $Id: SprRootWriter.hh,v 1.3 2006/11/13 19:09:40 narsky Exp $
//
// Description:
//      Class SprRootWriter :
//         Writes data and classifier output into a root tree.
//
// Environment:
//      Software developed for the BaBar Detector at the SLAC B-Factory.
//
// Author List:
//      Joshua Boehm                    Original Author
//      Ilya Narsky                     Created Design Spec. and Modified 
//                                      to obey internal coding style
// Copyright Information:
//      Copyright (C) 2006              California Institute of Technology
//      Copyright (C) 2006              Harvard University
//
//------------------------------------------------------------------------
 
#ifndef _SprRootWriter_HH
#define _SprRootWriter_HH

#include "PhysicsTools/StatPatternRecognition/interface/SprAbsWriter.hh"
#include <TTree.h>

#include <vector>
#include <string>


class SprRootWriter : public SprAbsWriter
{
public:
  virtual ~SprRootWriter();

  SprRootWriter(const char* label) 
    : SprAbsWriter(label), 
      fname_(), 
      axes_(), 
      tuple_(0),
      setBranches_(false),
      data_(0)
  {}

  // init output file
  bool init(const char* filename);

  // close output
  bool close();

  // set variables
  bool setAxes(const std::vector<std::string>& axes) {
    axes_ = axes;
    return true;
  }
  bool addAxis(const char* axis) {
    axes_.push_back(axis);
    return true;
  }

  /*
    The produced ntuple obeys the following conventions:
    The 1st stored variable is the point index and named "index".
    The 2nd stored variable is named "classification" and represents
    the true category of a data point if it is known; 
    if unknown, "classification" will be set to 0.
    The 3rd stored variable is the weight of this event, labeled as "weight".

    This method writes a tree of type ClassRecord containing 
    a single branch Vars.  Vars containes data point coordinates 
    followed by 1D responses from various classifiers. 
    The order of classifier responses in f has to be the same as the order
    in which axes have been added to the writer.
  */
  bool write(int cls, unsigned index, double weight, 
	     const std::vector<double>& v, 
	     const std::vector<double>& f);

private:
  int SetBranches();

  std::string fname_;
  std::vector<std::string> axes_;
  TTree* tuple_;
  bool setBranches_;
  Float_t* data_;  
};

#endif
