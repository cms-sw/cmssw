#ifndef MEtoEDMFormat_h
#define MEtoEDMFormat_h

/** \class MEtoEDM
 *  
 *  DataFormat class to hold the information from a ME tranformed into
 *  ROOT objects as appropriate
 *
 *  $Date: 2009/07/21 17:52:46 $
 *  $Revision: 1.13 $
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
  explicit MEtoEDM(size_t reservedSize) {
    MEtoEdmObject.reserve(reservedSize);
  }
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

  void putMEtoEdmObject(const std::string &name,
			const TagList &tags,
			const T &object,
			const std::string &release,
			const int run,
			const std::string &datatier)
    {
      typename MEtoEdmObjectVector::value_type temp;
      temp.name = name;
      temp.tags = tags;
      temp.object = object;
      temp.release = release;
      temp.run = run;
      temp.datatier = datatier;
      MEtoEdmObject.push_back(temp);
    }

  const MEtoEdmObjectVector & getMEtoEdmObject() const
    { return MEtoEdmObject; }

  bool mergeProduct(const MEtoEDM<T> &newMEtoEDM) {
    const MEtoEdmObjectVector &newMEtoEDMObject = 
      newMEtoEDM.getMEtoEdmObject();
    const size_t nObjects = newMEtoEDMObject.size();
    //  NOTE: we remember the present size since we will only add content
    //        from newMEtoEDMObject after this point
    const size_t nOldObjects = MEtoEdmObject.size();

   // if the old and new are not the same size, we want to report a problem
   bool warn = (nObjects == nOldObjects);

   for (unsigned int i = 0; i < nObjects; ++i) {
     unsigned int j = 0;
     // see if the name is already in the old container up to the point where
     // we may have added new entries in the container
     const std::string& name =newMEtoEDMObject[i].name;
     while (j <  nOldObjects && (MEtoEdmObject[j].name != name) ) ++j;
     if (j >= nOldObjects) {
       // this value is only in the new container, not the old one
       MEtoEdmObject.push_back(newMEtoEDMObject[i]);
       warn = true;
     } else {
       // this value is also in the new container: add the two 
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
#if 0
          std::cout << MEtoEdmObject[i].object.GetNbinsX() << " " << newMEtoEDMObject[j].object.GetNbinsX() << std::endl;
          std::cout << MEtoEdmObject[i].object.GetXaxis()->GetXmin() << " " << newMEtoEDMObject[j].object.GetXaxis()->GetXmin() << std::endl;
          std::cout << MEtoEdmObject[i].object.GetXaxis()->GetXmax() << " " << newMEtoEDMObject[j].object.GetXaxis()->GetXmax() << std::endl;
          std::cout << MEtoEdmObject[i].object.GetNbinsY() << " " << newMEtoEDMObject[j].object.GetNbinsY() << std::endl;
          std::cout << MEtoEdmObject[i].object.GetYaxis()->GetXmin() << " " << newMEtoEDMObject[j].object.GetYaxis()->GetXmin() << std::endl;
          std::cout << MEtoEdmObject[i].object.GetYaxis()->GetXmax() << " " << newMEtoEDMObject[j].object.GetYaxis()->GetXmax() << std::endl;
          std::cout << MEtoEdmObject[i].object.GetNbinsZ() << " " << newMEtoEDMObject[j].object.GetNbinsZ() << std::endl;
          std::cout << MEtoEdmObject[i].object.GetZaxis()->GetXmin() << " " << newMEtoEDMObject[j].object.GetZaxis()->GetXmin() << std::endl;
          std::cout << MEtoEdmObject[i].object.GetZaxis()->GetXmax() << " " << newMEtoEDMObject[j].object.GetZaxis()->GetXmax() << std::endl;
#endif
       }
     }
   }
   if (warn) {
     std::cout << "WARNING MEtoEDM::mergeProducts(): problem found" << std::endl;
   }
   return true;
  }

  void swap(MEtoEDM<T>& iOther) {
    MEtoEdmObject.swap(iOther.MEtoEdmObject);
  }
 private:

  MEtoEdmObjectVector MEtoEdmObject;

}; // end class declaration

template <>
inline bool
MEtoEDM<double>::mergeProduct(const MEtoEDM<double> &newMEtoEDM)
{
  const MEtoEdmObjectVector &newMEtoEDMObject =
    newMEtoEDM.getMEtoEdmObject();
  const size_t nObjects = newMEtoEDMObject.size();
  //  NOTE: we remember the present size since we will only add content
  //        from newMEtoEDMObject after this point
  const size_t nOldObjects = MEtoEdmObject.size();

  // if the old and new are not the same size, we want to report a problem
  bool warn = (nObjects == nOldObjects);

  for (unsigned int i = 0; i < nObjects; ++i) {
    unsigned int j = 0;
    // see if the name is already in the old container up to the point where
    // we may have added new entries in the container
    const std::string& name =newMEtoEDMObject[i].name;
    while (j <  nOldObjects && (MEtoEdmObject[j].name != name) ) ++j;
    if (j >= nOldObjects) {
      // this value is only in the new container, not the old one
      MEtoEdmObject.push_back(newMEtoEDMObject[i]);
      warn = true;
    }
  }
  if (warn) {
    std::cout << "WARNING MEtoEDM::mergeProducts(): problem found" << std::endl;
  }
  return true;
}

template <>
inline bool
MEtoEDM<int>::mergeProduct(const MEtoEDM<int> &newMEtoEDM)
{
  const MEtoEdmObjectVector &newMEtoEDMObject =
    newMEtoEDM.getMEtoEdmObject();
  const size_t nObjects = newMEtoEDMObject.size();
  //  NOTE: we remember the present size since we will only add content
  //        from newMEtoEDMObject after this point
  const size_t nOldObjects = MEtoEdmObject.size();

  // if the old and new are not the same size, we want to report a problem
  bool warn = (nObjects == nOldObjects);

  for (unsigned int i = 0; i < nObjects; ++i) {
    unsigned int j = 0;
    // see if the name is already in the old container up to the point where
    // we may have added new entries in the container
    const std::string& name =newMEtoEDMObject[i].name;
    while (j <  nOldObjects && (MEtoEdmObject[j].name != name) ) ++j;
    if (j >= nOldObjects) {
      // this value is only in the new container, not the old one
      MEtoEdmObject.push_back(newMEtoEDMObject[i]);
      warn = true;
    } else {
      // this value is also in the new container: add the two
      if ( MEtoEdmObject[i].name.find("EventInfo/processedEvents") != std::string::npos ) {
        MEtoEdmObject[i].object += (newMEtoEDMObject[j].object);
      }
      if ( MEtoEdmObject[i].name.find("EventInfo/iEvent") != std::string::npos ||
           MEtoEdmObject[i].name.find("EventInfo/iLumiSection") != std::string::npos) {
        if (MEtoEdmObject[i].object < newMEtoEDMObject[j].object) {
          MEtoEdmObject[i].object = (newMEtoEDMObject[j].object);
        }
      }
    }
  }
  if (warn) {
    std::cout << "WARNING MEtoEDM::mergeProducts(): problem found" << std::endl;
  }
  return true;
}

template <>
inline bool
MEtoEDM<TString>::mergeProduct(const MEtoEDM<TString> &newMEtoEDM)
{
  const MEtoEdmObjectVector &newMEtoEDMObject =
    newMEtoEDM.getMEtoEdmObject();
  const size_t nObjects = newMEtoEDMObject.size();
  //  NOTE: we remember the present size since we will only add content
  //        from newMEtoEDMObject after this point
  const size_t nOldObjects = MEtoEdmObject.size();

  // if the old and new are not the same size, we want to report a problem
  bool warn = (nObjects == nOldObjects);

  for (unsigned int i = 0; i < nObjects; ++i) {
    unsigned int j = 0;
    // see if the name is already in the old container up to the point where
    // we may have added new entries in the container
    const std::string& name =newMEtoEDMObject[i].name;
    while (j <  nOldObjects && (MEtoEdmObject[j].name != name) ) ++j;
    if (j >= nOldObjects) {
      // this value is only in the new container, not the old one
      MEtoEdmObject.push_back(newMEtoEDMObject[i]);
      warn = true;
    }
  }
  if (warn) {
    std::cout << "WARNING MEtoEDM::mergeProducts(): problem found" << std::endl;
  }
  return true;
}

#endif
