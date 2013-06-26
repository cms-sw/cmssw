#ifndef Histos_H
#define Histos_H

/**
 * This class provides an interface to root histograms
 *
 * \author Patrick Janot
 * $Date: 16 Jan 2004 19:30 */
// for debugging

//#include "TROOT.h"
#include "TObject.h"

#include <string>
#include <map>
#include <iostream>
//#include <sstream>

class Histos
{
 public:

  typedef std::map<std::string,TObject*>::const_iterator HistoItr;

  static Histos* instance();

  /// Destructor
  virtual ~Histos();
  
  /// Book an histogram (1D or 2D)
  void book(const std::string& name, 
	    int nx  , float xmin   , float xmax,
	    int ny=0, float ymin=0., float ymax=0.);

  // Same as before. Creates n2-n1 histos with name name+n1 ... name+n2
  void bookByNumber(const std::string& name, int n1,int n2,
	    int nx  , float xmin   , float xmax,
	    int ny=0, float ymin=0., float ymax=0.);


  /// Book a TProfile
  /// option="S" -> spread 
  ///        ""  -> error on mean (from Root documentation)
  void book(const std::string& name, int nx, float xmin, float xmax,
	    const std::string& option);


  /// Write one or all histogram(s) in a file
  void put(const std::string& file, std::string name="");

  /// Divide two histograms and put the result in the first
  void divide(const std::string& h1, const std::string& h2, const std::string& h3);

  /// Fill an histogram
  void fill(const std::string& name, float val1, float val2=1., float val3=1.);

  // Fill a series of histos. The name is name+number
  void fillByNumber(const std::string& name,int number,float val1,float val2=1.,float val3=1.);

  /// Add any object
  void addObject(const std::string& name, TObject * obj);

  /// 
    void debug(std::string p="") const {std::cout << " Histos myMap : "<< &theHistos << " " << p <<std::endl;}

 private:

  // The constructor is hidden as we do not want to construct
  // more than one instance.
  Histos();

  // The instance
  static Histos* myself;

  // The histos booked
  TObject* theHisto;
  std::map<std::string,TObject*> theHistos;
  std::map<std::string,unsigned> theTypes;
  std::map<std::string,TObject*> theObjects;
  
};
#endif
