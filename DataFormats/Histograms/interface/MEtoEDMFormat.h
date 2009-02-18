#ifndef MEtoEDMFormat_h
#define MEtoEDMFormat_h

/** \class MEtoEDM
 *  
 *  DataFormat class to hold the information from a ME tranformed into
 *  ROOT objects as appropriate
 *
 *  $Date: 2008/10/15 13:44:42 $
 *  $Revision: 1.9 $
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
    bool warn = false;
    std::vector<bool> tmp(newMEtoEDMObject.size(), false);
    for (unsigned int i = 0; i < MEtoEdmObject.size(); ++i) {
      unsigned int j = 0;
      while (j < newMEtoEDMObject.size() &&
             (strcmp(MEtoEdmObject[i].name.c_str(),
                     newMEtoEDMObject[j].name.c_str()) != 0)) ++j;
      if (j < newMEtoEDMObject.size()) {
        if (MEtoEdmObject[i].object.GetNbinsX() == newMEtoEDMObject[j].object.GetNbinsX() &&
            MEtoEdmObject[i].object.GetXaxis()->GetXmin() == newMEtoEDMObject[j].object.GetXaxis()->GetXmin() &&
            MEtoEdmObject[i].object.GetXaxis()->GetXmax() == newMEtoEDMObject[j].object.GetXaxis()->GetXmax() &&
            MEtoEdmObject[i].object.GetNbinsY() == newMEtoEDMObject[j].object.GetNbinsY() &&
            MEtoEdmObject[i].object.GetYaxis()->GetXmin() == newMEtoEDMObject[j].object.GetYaxis()->GetXmin() &&
            MEtoEdmObject[i].object.GetYaxis()->GetXmax() == newMEtoEDMObject[j].object.GetYaxis()->GetXmax() &&
            MEtoEdmObject[i].object.GetNbinsZ() == newMEtoEDMObject[j].object.GetNbinsZ() &&
            MEtoEdmObject[i].object.GetZaxis()->GetXmin() == newMEtoEDMObject[j].object.GetZaxis()->GetXmin() &&
            MEtoEdmObject[i].object.GetZaxis()->GetXmax() == newMEtoEDMObject[j].object.GetZaxis()->GetXmax()) {
          MEtoEdmObject[i].object.Add(&newMEtoEDMObject[j].object);
        } else {
	  std::cout << "ERROR MEtoEDM::mergeProducts(): different axis limits - DQM ME '" << MEtoEdmObject[i].name << "' not merged" <<  std::endl;
          //std::cout << MEtoEdmObject[i].object.GetNbinsX() << " " << newMEtoEDMObject[j].object.GetNbinsX() << std::endl;
          //std::cout << MEtoEdmObject[i].object.GetXaxis()->GetXmin() << " " << newMEtoEDMObject[j].object.GetXaxis()->GetXmin() << std::endl;
          //std::cout << MEtoEdmObject[i].object.GetXaxis()->GetXmax() << " " << newMEtoEDMObject[j].object.GetXaxis()->GetXmax() << std::endl;
          //std::cout << MEtoEdmObject[i].object.GetNbinsY() << " " << newMEtoEDMObject[j].object.GetNbinsY() << std::endl;
          //std::cout << MEtoEdmObject[i].object.GetYaxis()->GetXmin() << " " << newMEtoEDMObject[j].object.GetYaxis()->GetXmin() << std::endl;
          //std::cout << MEtoEdmObject[i].object.GetYaxis()->GetXmax() << " " << newMEtoEDMObject[j].object.GetYaxis()->GetXmax() << std::endl;
          //std::cout << MEtoEdmObject[i].object.GetNbinsZ() << " " << newMEtoEDMObject[j].object.GetNbinsZ() << std::endl;
          //std::cout << MEtoEdmObject[i].object.GetZaxis()->GetXmin() << " " << newMEtoEDMObject[j].object.GetZaxis()->GetXmin() << std::endl;
          //std::cout << MEtoEdmObject[i].object.GetZaxis()->GetXmax() << " " << newMEtoEDMObject[j].object.GetZaxis()->GetXmax() << std::endl;
        }
        tmp[j] = true;
      } else {
        warn = true;
      }
    }
    for (unsigned int j = 0; j < newMEtoEDMObject.size(); ++j) {
      if (!tmp[j]) {
        warn = true;
        MEtoEdmObject.push_back(newMEtoEDMObject[j]);
      }
    }
    if (warn) {
      std::cout << "WARNING MEtoEDM::mergeProducts(): problem found" << std::endl;
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
   if ( MEtoEdmObject[i].name.find("EventInfo/processedEvents") != std::string::npos ) {
     MEtoEdmObject[i].object += (newMEtoEDMObject[i].object);
   }
   if ( MEtoEdmObject[i].name.find("EventInfo/iEvent") != std::string::npos ||
        MEtoEdmObject[i].name.find("EventInfo/iLumiSection") != std::string::npos) {
        if (MEtoEdmObject[i].object < newMEtoEDMObject[i].object) 
                           MEtoEdmObject[i].object = (newMEtoEDMObject[i].object);
   }
   
 }
 return true;
}

template <>
inline bool
MEtoEDM<TString>::mergeProduct(const MEtoEDM<TString> &newMEtoEDM)
{ return true; }

#endif
