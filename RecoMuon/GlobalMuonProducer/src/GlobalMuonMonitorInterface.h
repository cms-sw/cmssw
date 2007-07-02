#ifndef GlobalMuonMonitorInterface_H
#define GlobalMuonMonitorInterface_H

/** \class GlobalMuonMonitorInterface
 *
 * Interface to the Muon inter-module Monitoring Module.
 *  
 *  $Date: 2006/12/13 20:22:39 $
 *  $Revision: 1.1 $
 *
 * \author A. Everett - Purdue University
 *
 */

#include <string>

class GlobalMuonMonitorInterface{
  
 public:
  
  GlobalMuonMonitorInterface(){}
  virtual ~GlobalMuonMonitorInterface(){}
  
  virtual void book1D(std::string name, std::string title, int nchX, 
		      double lowX, double highX) = 0; 
  virtual void book1D(std::string level, std::string name, std::string title, 
		      int nchX, double lowX, double highX) = 0;
  virtual void book2D(std::string name, std::string title, int nchX, 
		      double lowX, double highX, int nchY,
		      double lowY, double highY) = 0;
  virtual void book2D(std::string level, std::string name, std::string title,
		      int nchX, double lowX, double highX, 
		      int nchY, double lowY, double highY) = 0;
  virtual void save(std::string) = 0;
  virtual void fill1(std::string, double a, double b=1.) = 0;
  virtual void fill1(std::string, std::string, double a, double b=1.) = 0;
  virtual void fill2(std::string, double a, double b, double c=1.) = 0;
  virtual void fill2(std::string, std::string, double a, double b, double c=1.) = 0;
  
 private:
  
};

#endif
