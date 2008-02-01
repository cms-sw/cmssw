#ifndef MEtoEDMFormat_h
#define MEtoEDMFormat_h

/** \class MEtoEDM
 *  
 *  DataFormat class to hold the information from a ME tranformed into
 *  ROOT objects as appropriate
 *
 *  $Date: 2008/01/25 23:14:04 $
 *  $Revision: 1.5 $
 *  \author M. Strang SUNY-Buffalo
 */

#include <TObject.h>
#include <TH1F.h>
#include <TH2F.h>
#include <TH3F.h>
#include <TProfile.h>
#include <TProfile2D.h>
#include <TObjString.h>
#include <TString.h>

#include <string>
#include <vector>
#include <memory>
#include <map>

template <class T>
class MEtoEDM
{

 public:
  
  MEtoEDM() {}
  virtual ~MEtoEDM() {}

  typedef std::vector<uint32_t> TagList;

  struct MEtoEDMObject
  {
    std::string	name;
    TagList 	tags;
    T	        object;
  };

  typedef std::vector<MEtoEDMObject> MEtoEdmObjectVector;

  void putMEtoEdmObject(std::vector<std::string> name,
			std::vector<TagList> tags,
			std::vector<T> object);

  MEtoEdmObjectVector getMEtoEdmObject() const {return MEtoEdmObject;}

 private:

  MEtoEdmObjectVector MEtoEdmObject;

}; // end class declaration

#endif
