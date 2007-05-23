//------------------------------------------------------------------------
// File and Version Information:
//      $Id: SprAbsWriter.hh,v 1.3 2006/11/13 19:09:38 narsky Exp $
//
// Description:
//      Class SprAbsWriter :
//         writes classified data to a file
//
// Environment:
//      Software developed for the BaBar Detector at the SLAC B-Factory.
//
// Author List:
//      Ilya Narsky                     Original author
//
// Copyright Information:
//      Copyright (C) 2004              California Institute of Technology
//
//------------------------------------------------------------------------
 
#ifndef _SprAbsWriter_HH
#define _SprAbsWriter_HH

#include "PhysicsTools/StatPatternRecognition/interface/SprPoint.hh"
#include <vector>
#include <string>


class SprAbsWriter
{
public:
  virtual ~SprAbsWriter() {}

  SprAbsWriter(const char* label) : label_(label) {}

  // Init output file.
  virtual bool init(const char* filename) = 0;

  // Close output. Writer must be closed at end of operation!!!
  virtual bool close() = 0;

  // Set up axes to hold output data.
  virtual bool setAxes(const std::vector<std::string>& axes) = 0;
  virtual bool addAxis(const char* axis) = 0;

  /*
    Write a vector of data point coordinates 
    followed by a vector of 1D responses from various classifiers. 
  */
  virtual bool write(int cls, unsigned index, double weight,
		     const std::vector<double>& v, 
		     const std::vector<double>& f) = 0;

  bool write(double weight, 
	     const std::vector<double>& v, 
	     const std::vector<double>& f) {
    return this->write(0,0,weight,v,f);
  }

  bool write(double weight, 
	     const SprPoint* p, 
	     const std::vector<double>& f) {
    return this->write(p->class_,p->index_,weight,p->x_,f);
  }

protected:
  std::string label_;
};

#endif
