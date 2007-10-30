//------------------------------------------------------------------------
// File and Version Information:
//      $Id: SprMultiClassPlotter.hh,v 1.1 2007/10/22 21:23:40 narsky Exp $
//
// Description:
//      Class SprMultiClassPlotter :
//         tools for plotting classification results
//
// Author List:
//      Ilya Narsky                     Original author
//
// Copyright Information:
//      Copyright (C) 2007              California Institute of Technology
//
//------------------------------------------------------------------------
 
#ifndef _SprMultiClassPlotter_HH
#define _SprMultiClassPlotter_HH

#include <map>
#include <vector>


class SprMultiClassPlotter
{
public:

  struct Response {
    int cls;// true class
    double weight;
    int assigned;// class assigned to this event
    std::map<int,double> response;// responses for classes
    
    ~Response() {}

    Response() : cls(0), weight(0), assigned(0), response() {}
    
    Response(int c, double w, 
	     int assignedClass, const std::map<int,double>& resp)
      : cls(c), weight(w), assigned(assignedClass), response(resp) {}
    
    Response(const Response& other)
      : cls(other.cls), 
	weight(other.weight), 
	assigned(other.assigned),
	response(other.response) {}
  };

  virtual ~SprMultiClassPlotter() {}

  SprMultiClassPlotter(const std::vector<Response>& responses)
    : responses_(responses) {}

  SprMultiClassPlotter(const SprMultiClassPlotter& other)
    : responses_(other.responses_) {}

  // Computes classification table for the multi-class learner
  // for the classes requested by the user. Returns an overall misid rate.
  double multiClassTable(const std::vector<int>& classes,
		 std::map<int,std::vector<double> >& classificationTable,
			 std::map<int,double>& weightInClass,
			 bool normalizePerClass=true) const;

private:
  std::vector<Response> responses_;
};

#endif
