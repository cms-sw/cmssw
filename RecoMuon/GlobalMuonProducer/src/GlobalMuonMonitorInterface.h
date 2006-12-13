#ifndef GlobalMuonMonitorInterface_H
#define GlobalMuonMonitorInterface_H

/** \class GlobalMuonMonitorInterface
 *
 * Interface to the Muon inter-module Monitoring Module.
 *  
 *  $Date: $
 *  $Revision: $
 *
 * \author A. Everett - Purdue University
 *
 */

#include <string>

using namespace std;
//using namespace edm;


class GlobalMuonMonitorInterface{
  
 public:
  
  GlobalMuonMonitorInterface(){}
  virtual ~GlobalMuonMonitorInterface(){}
  
  virtual void book1D(string name, string title, int nchX, 
		      double lowX, double highX) = 0; 
  virtual void book1D(string level, string name, string title, 
		      int nchX, double lowX, double highX) = 0;
  virtual void book2D(string name, string title, int nchX, 
		      double lowX, double highX, int nchY,
		      double lowY, double highY) = 0;
  virtual void book2D(string level, string name, string title,
		      int nchX, double lowX, double highX, 
		      int nchY, double lowY, double highY) = 0;
  virtual void save(string) = 0;
  virtual void fill1(string, double a, double b=1.) = 0;
  virtual void fill1(string, string, double a, double b=1.) = 0;
  virtual void fill2(string, double a, double b, double c=1.) = 0;
  virtual void fill2(string, string, double a, double b, double c=1.) = 0;
  
 private:
  
};

#endif
