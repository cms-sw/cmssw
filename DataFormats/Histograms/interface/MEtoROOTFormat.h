#ifndef MEtoROOTFormat_h
#define MEtoROOTFormat_h

/** \class MEtoROOT
 *  
 *  DataFormat class to hold the information from a ME tranformed into
 *  ROOT objects as appropriate
 *
 *  $Date: 2007/12/05 05:37:14 $
 *  $Revision: 1.3 $
 *  \author M. Strang SUNY-Buffalo
 */

#include <TObject.h>
#include <TH1F.h>
#include <TH2F.h>
#include <TH3F.h>
#include <TProfile.h>
#include <TProfile2D.h>
#include <TObjString.h>

#include <string>
#include <vector>
#include <memory>
#include <map>

template <class T>
class MEtoROOT
{

 public:
  
  MEtoROOT() {}
  virtual ~MEtoROOT() {}

  typedef std::vector<uint32_t> TagList;

  struct MEROOTObject
  {
    std::string	name;
    TagList 	tags;
    T	        object;
  };

  typedef std::vector<MEROOTObject> MERootObjectVector;

  void putMERootObject(std::vector<std::string> name,
		       std::vector<TagList> tags,
		       std::vector<T> object);

  MERootObjectVector getMERootObject() const {return MERootObject;}

 private:

  MERootObjectVector MERootObject;

}; // end class declaration

#endif
