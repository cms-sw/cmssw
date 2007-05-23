//------------------------------------------------------------------------
// File and Version Information:
//      $Id: SprTupleWriter.hh,v 1.3 2006/11/13 19:09:40 narsky Exp $
//
// Description:
//      Class SprTupleWriter :
//         Writes data and classifier output into a tuple.
//         If the tuple manager is not supplied by the user,
//         will book an HbkFile by default.
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
 
#ifndef _SprTupleWriter_HH
#define _SprTupleWriter_HH

#include "PhysicsTools/StatPatternRecognition/interface/SprAbsWriter.hh"

#include <vector>
#include <string>

class HepTupleManager;
class HepTuple;


class SprTupleWriter : public SprAbsWriter
{
public:
  virtual ~SprTupleWriter();

  SprTupleWriter(const char* label) 
    : SprAbsWriter(label), 
      fname_(), 
      axes_(), 
      tupleMgr_(0), 
      tuple_(0),
      ownMgr_(false)
  {}

  SprTupleWriter(const char* label, HepTupleManager* tupleMgr) 
    : SprAbsWriter(label), 
      fname_(), 
      axes_(), 
      tupleMgr_(tupleMgr), 
      tuple_(0),
      ownMgr_(false)
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
    The 2nd stored variable is named "i" and represents
    the true category of a data point if it is known; 
    if unknown, "i" will be set to 0.
    The 3rd stored variable is the weight of this event, labeled as "w".

    This method writes a vector of data point coordinates 
    followed by a vector of 1D responses from various classifiers. 
    The order of classifier responses in f has to be the same as the order
    in which axes have been added to the writer.
  */
  bool write(int cls, unsigned index, double weight, 
	     const std::vector<double>& v, 
	     const std::vector<double>& f);

private:
  std::string fname_;
  std::vector<std::string> axes_;
  HepTupleManager* tupleMgr_;
  HepTuple* tuple_;
  bool ownMgr_;
};

#endif
