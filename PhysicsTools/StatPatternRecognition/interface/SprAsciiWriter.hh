//------------------------------------------------------------------------
// File and Version Information:
//      $Id: SprAsciiWriter.hh,v 1.3 2006/11/13 19:09:39 narsky Exp $
//
// Description:
//      Class SprAsciiWriter :
//         Writes data and classifier output into an ascii file.
//
// Environment:
//      Software developed for the BaBar Detector at the SLAC B-Factory.
//
// Author List:
//      Ilya Narsky                     Original author
//
// Copyright Information:
//      Copyright (C) 2005              California Institute of Technology
//
//------------------------------------------------------------------------
 
#ifndef _SprAsciiWriter_HH
#define _SprAsciiWriter_HH

#include "PhysicsTools/StatPatternRecognition/interface/SprAbsWriter.hh"

#include <vector>
#include <string>
#include <fstream>


class SprAsciiWriter : public SprAbsWriter
{
public:
  virtual ~SprAsciiWriter() {}

  SprAsciiWriter(const char* label)
    : SprAbsWriter(label), 
      firstCall_(true),
      outfile_(), 
      axes_()
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
    See SprTupleWriter.hh for conventions used in writing output data.
  */
  bool write(int cls, unsigned index, double weight, 
	     const std::vector<double>& v, 
	     const std::vector<double>& f);

private:
  bool firstCall_;
  std::ofstream outfile_;
  std::vector<std::string> axes_;
};

#endif
