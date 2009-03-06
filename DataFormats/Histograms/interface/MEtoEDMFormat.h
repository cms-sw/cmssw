#ifndef MEtoEDMFormat_h
#define MEtoEDMFormat_h

/** \class MEtoEDM
 *  
 *  DataFormat class to hold the information from a ME tranformed into
 *  ROOT objects as appropriate
 *
 *  $Date: 2008/08/09 16:09:33 $
 *  $Revision: 1.6 $
 *  \author M. Strang SUNY-Buffalo
 */

#include <TObject.h>
#include <TH1F.h>
#include <TH1S.h>
#include <TH2F.h>
#include <TH2S.h>
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
    std::string release;
    int run;
    std::string datatier;
  };

  typedef std::vector<MEtoEDMObject> MEtoEdmObjectVector;

  void putMEtoEdmObject(const std::vector<std::string> &name,
			const std::vector<TagList> &tags,
			const std::vector<T> &object,
			const std::vector<std::string> &release,
			const std::vector<int> &run,
			const std::vector<std::string> &datatier)
    {
      MEtoEdmObject.resize(name.size());
      for (unsigned int i = 0; i < name.size(); ++i) {
	MEtoEdmObject[i].name = name[i];
	MEtoEdmObject[i].tags = tags[i];
	MEtoEdmObject[i].object = object[i];
	MEtoEdmObject[i].release = release[i];
	MEtoEdmObject[i].run = run[i];
	MEtoEdmObject[i].datatier = datatier[i];
      }
    }

  const MEtoEdmObjectVector & getMEtoEdmObject() const
    { return MEtoEdmObject; }

  bool mergeProduct(const MEtoEDM<T> &newMEtoEDM) {
    const MEtoEdmObjectVector &newMEtoEDMObject = 
      newMEtoEDM.getMEtoEdmObject();
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
{
 const MEtoEdmObjectVector &newMEtoEDMObject =
   newMEtoEDM.getMEtoEdmObject();
 for (unsigned int i = 0; i < MEtoEdmObject.size(); ++i) {
   if ( MEtoEdmObject[i].name.find("processedEvents") != std::string::npos ) {
     MEtoEdmObject[i].object += (newMEtoEDMObject[i].object);
   }
 }
 return true;
}

template <>
inline bool
MEtoEDM<TString>::mergeProduct(const MEtoEDM<TString> &newMEtoEDM)
{ return true; }

#endif
