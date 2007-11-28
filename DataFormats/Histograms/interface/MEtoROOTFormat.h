#ifndef MEtoROOTFormat_h
#define MEtoROOTFormat_h

/** \class MEtoROOT
 *  
 *  DataFormat class to hold the information from a ME tranformed into
 *  ROOT objects as appropriate
 *
 *  $Date: 2007/11/20 12:45:10 $
 *  $Revision: 1.3 $
 *  \author M. Strang SUNY-Buffalo
 */

#include <string>

class MEtoROOT
{

 public:
  
  MEtoROOT() {}
  virtual ~MEtoROOT() {}

  struct ROOTObject
  {
    std::string dirpath;
  };

  void putRootObject(std::string dirpath);

  ROOTObject getRootObject() const {return test;}

 private:

  ROOTObject test;

}; // end class declaration

#endif
