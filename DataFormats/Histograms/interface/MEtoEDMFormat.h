#ifndef MEtoEDMFormat_h
#define MEtoEDMFormat_h

/** \class MEtoEDM
 *  
 *  DataFormat class to hold the information from a ME tranformed into
 *  ROOT objects as appropriate
 *
 *  $Date: 2008/02/05 23:45:49 $
 *  $Revision: 1.2 $
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

  void putMEtoEdmObject(const std::vector<std::string> &name,
			const std::vector<TagList> &tags,
			const std::vector<T> &object)
    {
      MEtoEdmObject.resize(name.size());
      for (unsigned int i = 0; i < name.size(); ++i) {
	MEtoEdmObject[i].name = name[i];
	MEtoEdmObject[i].tags = tags[i];
	MEtoEdmObject[i].object = object[i];
      }
    }

  const MEtoEdmObjectVector & getMEtoEdmObject() const
    { return MEtoEdmObject; }

  bool mergeProduct(const MEtoEDM<T> &newMEtoEDM)
    {
      const MEtoEdmObjectVector &newMEtoEDMObject = newMEtoEDM.getMEtoEdmObject();
      for (unsigned int i = 0; i < MEtoEdmObject.size(); ++i) {
        MEtoEdmObject[i].object.Add(&newMEtoEDMObject[i].object);
      }
      return true;
    }

 private:

  MEtoEdmObjectVector MEtoEdmObject;

}; // end class declaration

template <>
inline bool
MEtoEDM<double>::mergeProduct(const MEtoEDM<double> &newMEtoEDM)
{ return true; }

template <>
inline bool
MEtoEDM<int>::mergeProduct(const MEtoEDM<int> &newMEtoEDM)
{ return true; }

template <>
inline bool
MEtoEDM<TString>::mergeProduct(const MEtoEDM<TString> &newMEtoEDM)
{ return true; }

#endif
