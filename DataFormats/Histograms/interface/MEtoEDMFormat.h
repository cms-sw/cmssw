#ifndef MEtoEDMFormat_h
#define MEtoEDMFormat_h

/** \class MEtoEDM
 *  
 *  DataFormat class to hold the information from a ME tranformed into
 *  ROOT objects as appropriate
 *
 *  $Date: 2010/09/14 09:12:54 $
 *  $Revision: 1.28 $
 *  \author M. Strang SUNY-Buffalo
 */

#include <TObject.h>
#include <TH1F.h>
#include <TH1S.h>
#include <TH1D.h>
#include <TH2F.h>
#include <TH2S.h>
#include <TH2D.h>
#include <TH3F.h>
#include <TProfile.h>
#include <TProfile2D.h>
#include <TObjString.h>
#include <TString.h>
#include <TList.h>

#include <string>
#include <vector>
#include <memory>
#include <map>
#include <stdint.h>

#define debug 0

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
  };

  typedef std::vector<MEtoEDMObject> MEtoEdmObjectVector;

  void putMEtoEdmObject(const std::string &name,
			const TagList &tags,
			const T &object)
    {
      typename MEtoEdmObjectVector::value_type temp;
      MEtoEdmObject.push_back(temp);
      MEtoEdmObject.back().name = name;
      MEtoEdmObject.back().tags = tags;
      MEtoEdmObject.back().object = object;
    }

  const MEtoEdmObjectVector & getMEtoEdmObject() const
    { return MEtoEdmObject; }

  bool mergeProduct(const MEtoEDM<T> &newMEtoEDM) {
    const MEtoEdmObjectVector &newMEtoEDMObject = newMEtoEDM.getMEtoEdmObject();
    const size_t nObjects = newMEtoEDMObject.size();
    //  NOTE: we remember the present size since we will only add content
    //        from newMEtoEDMObject after this point
    const size_t nOldObjects = MEtoEdmObject.size();

   // if the old and new are not the same size, we want to report a problem
   if (nObjects != nOldObjects) {
     std::cout << "WARNING MEtoEDM::mergeProducts(): the lists of histograms to be merged have different sizes: new=" << nObjects << ", old=" << nOldObjects << std::endl;
   }

   for (unsigned int i = 0; i < nObjects; ++i) {
     unsigned int j = 0;
     // see if the name is already in the old container up to the point where
     // we may have added new entries in the container
     const std::string& name = newMEtoEDMObject[i].name;
     if (i < nOldObjects && (MEtoEdmObject[i].name == name)) {
       j = i;
     } else {
       j = 0;
       while (j <  nOldObjects && (MEtoEdmObject[j].name != name) ) ++j;
     }
     if (j >= nOldObjects) {
       // this value is only in the new container, not the old one
#if debug
       std::cout << "WARNING MEtoEDM::mergeProducts(): adding new histogram '" << name << "'" << std::endl;
#endif
       MEtoEdmObject.push_back(newMEtoEDMObject[i]);
     } else if (MEtoEdmObject[j].object.TestBit(TH1::kCanRebin) == true && newMEtoEDMObject[i].object.TestBit(TH1::kCanRebin) == true) {
       TList list;
       list.Add((TObject*)&newMEtoEDMObject[i].object);
       if (MEtoEdmObject[j].object.Merge(&list) == -1) {
	 std::cout << "ERROR MEtoEDM::mergeProducts(): merge failed for '" << name << "'" <<  std::endl;
       }
     } else {
       // this value is also in the new container: add the two 
       if (MEtoEdmObject[j].object.GetNbinsX()           == newMEtoEDMObject[i].object.GetNbinsX()           &&
           MEtoEdmObject[j].object.GetXaxis()->GetXmin() == newMEtoEDMObject[i].object.GetXaxis()->GetXmin() &&
           MEtoEdmObject[j].object.GetXaxis()->GetXmax() == newMEtoEDMObject[i].object.GetXaxis()->GetXmax() &&
           MEtoEdmObject[j].object.GetNbinsY()           == newMEtoEDMObject[i].object.GetNbinsY()           &&
           MEtoEdmObject[j].object.GetYaxis()->GetXmin() == newMEtoEDMObject[i].object.GetYaxis()->GetXmin() &&
           MEtoEdmObject[j].object.GetYaxis()->GetXmax() == newMEtoEDMObject[i].object.GetYaxis()->GetXmax() &&
           MEtoEdmObject[j].object.GetNbinsZ()           == newMEtoEDMObject[i].object.GetNbinsZ()           &&
           MEtoEdmObject[j].object.GetZaxis()->GetXmin() == newMEtoEDMObject[i].object.GetZaxis()->GetXmin() &&
           MEtoEdmObject[j].object.GetZaxis()->GetXmax() == newMEtoEDMObject[i].object.GetZaxis()->GetXmax()) {
         MEtoEdmObject[j].object.Add(&newMEtoEDMObject[i].object);
       } else {
          std::cout << "ERROR MEtoEDM::mergeProducts(): found histograms with different axis limits, '" << name << "' not merged" <<  std::endl;
#if debug
          std::cout << MEtoEdmObject[j].name                         << " " << newMEtoEDMObject[i].name                         << std::endl;
          std::cout << MEtoEdmObject[j].object.GetNbinsX()           << " " << newMEtoEDMObject[i].object.GetNbinsX()           << std::endl;
          std::cout << MEtoEdmObject[j].object.GetXaxis()->GetXmin() << " " << newMEtoEDMObject[i].object.GetXaxis()->GetXmin() << std::endl;
          std::cout << MEtoEdmObject[j].object.GetXaxis()->GetXmax() << " " << newMEtoEDMObject[i].object.GetXaxis()->GetXmax() << std::endl;
          std::cout << MEtoEdmObject[j].object.GetNbinsY()           << " " << newMEtoEDMObject[i].object.GetNbinsY()           << std::endl;
          std::cout << MEtoEdmObject[j].object.GetYaxis()->GetXmin() << " " << newMEtoEDMObject[i].object.GetYaxis()->GetXmin() << std::endl;
          std::cout << MEtoEdmObject[j].object.GetYaxis()->GetXmax() << " " << newMEtoEDMObject[i].object.GetYaxis()->GetXmax() << std::endl;
          std::cout << MEtoEdmObject[j].object.GetNbinsZ()           << " " << newMEtoEDMObject[i].object.GetNbinsZ()           << std::endl;
          std::cout << MEtoEdmObject[j].object.GetZaxis()->GetXmin() << " " << newMEtoEDMObject[i].object.GetZaxis()->GetXmin() << std::endl;
          std::cout << MEtoEdmObject[j].object.GetZaxis()->GetXmax() << " " << newMEtoEDMObject[i].object.GetZaxis()->GetXmax() << std::endl;
#endif
       }
     }
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
  const MEtoEdmObjectVector &newMEtoEDMObject = newMEtoEDM.getMEtoEdmObject();
  const size_t nObjects = newMEtoEDMObject.size();
  //  NOTE: we remember the present size since we will only add content
  //        from newMEtoEDMObject after this point
  const size_t nOldObjects = MEtoEdmObject.size();

  // if the old and new are not the same size, we want to report a problem
  if (nObjects != nOldObjects) {
    std::cout << "WARNING MEtoEDM::mergeProducts(): the lists of histograms to be merged have different sizes: new=" << nObjects << ", old=" << nOldObjects << std::endl;
  }

  for (unsigned int i = 0; i < nObjects; ++i) {
    unsigned int j = 0;
    // see if the name is already in the old container up to the point where
    // we may have added new entries in the container
    const std::string& name = newMEtoEDMObject[i].name;
    if (i < nOldObjects && (MEtoEdmObject[i].name == name)) {
      j = i;
    } else {
      j = 0;
      while (j <  nOldObjects && (MEtoEdmObject[j].name != name) ) ++j;
    }
    if (j >= nOldObjects) {
      // this value is only in the new container, not the old one
#if debug
      std::cout << "WARNING MEtoEDM::mergeProducts(): adding new histogram '" << name << "'" << std::endl;
#endif
      MEtoEdmObject.push_back(newMEtoEDMObject[i]);
    }
  }
  return true;
}

template <>
inline bool
MEtoEDM<int>::mergeProduct(const MEtoEDM<int> &newMEtoEDM)
{
  const MEtoEdmObjectVector &newMEtoEDMObject = newMEtoEDM.getMEtoEdmObject();
  const size_t nObjects = newMEtoEDMObject.size();
  //  NOTE: we remember the present size since we will only add content
  //        from newMEtoEDMObject after this point
  const size_t nOldObjects = MEtoEdmObject.size();

  // if the old and new are not the same size, we want to report a problem
  if (nObjects != nOldObjects) {
    std::cout << "WARNING MEtoEDM::mergeProducts(): the lists of histograms to be merged have different sizes: new=" << nObjects << ", old=" << nOldObjects << std::endl;
  }

  for (unsigned int i = 0; i < nObjects; ++i) {
    unsigned int j = 0;
    // see if the name is already in the old container up to the point where
    // we may have added new entries in the container
    const std::string& name = newMEtoEDMObject[i].name;
    if (i < nOldObjects && (MEtoEdmObject[i].name == name)) {
      j = i;
    } else {
      j = 0;
      while (j <  nOldObjects && (MEtoEdmObject[j].name != name) ) ++j;
    }
    if (j >= nOldObjects) {
      // this value is only in the new container, not the old one
#if debug
      std::cout << "WARNING MEtoEDM::mergeProducts(): adding new histogram '" << name << "'" << std::endl;
#endif
      MEtoEdmObject.push_back(newMEtoEDMObject[i]);
    } else {
      // this value is also in the new container: add the two
      if ( MEtoEdmObject[j].name.find("EventInfo/processedEvents") != std::string::npos ) {
        MEtoEdmObject[j].object += (newMEtoEDMObject[i].object);
      }
      if ( MEtoEdmObject[j].name.find("EventInfo/iEvent") != std::string::npos ||
           MEtoEdmObject[j].name.find("EventInfo/iLumiSection") != std::string::npos) {
        if (MEtoEdmObject[j].object < newMEtoEDMObject[i].object) {
          MEtoEdmObject[j].object = (newMEtoEDMObject[i].object);
        }
      }
    }
  }
  return true;
}

template <>
inline bool
MEtoEDM<long long>::mergeProduct(const MEtoEDM<long long> &newMEtoEDM)
{
  const MEtoEdmObjectVector &newMEtoEDMObject = newMEtoEDM.getMEtoEdmObject();
  const size_t nObjects = newMEtoEDMObject.size();
  //  NOTE: we remember the present size since we will only add content
  //        from newMEtoEDMObject after this point
  const size_t nOldObjects = MEtoEdmObject.size();

  // if the old and new are not the same size, we want to report a problem
  if (nObjects != nOldObjects) {
    std::cout << "WARNING MEtoEDM::mergeProducts(): the lists of histograms to be merged have different sizes: new=" << nObjects << ", old=" << nOldObjects << std::endl;
  }

  for (unsigned int i = 0; i < nObjects; ++i) {
    unsigned int j = 0;
    // see if the name is already in the old container up to the point where
    // we may have added new entries in the container
    const std::string& name = newMEtoEDMObject[i].name;
    if (i < nOldObjects && (MEtoEdmObject[i].name == name)) {
      j = i;
    } else {
      j = 0;
      while (j <  nOldObjects && (MEtoEdmObject[j].name != name) ) ++j;
    }
    if (j >= nOldObjects) {
      // this value is only in the new container, not the old one
#if debug
      std::cout << "WARNING MEtoEDM::mergeProducts(): adding new histogram '" << name << "'" << std::endl;
#endif
      MEtoEdmObject.push_back(newMEtoEDMObject[i]);
    } else {
      // this value is also in the new container: add the two
      if ( MEtoEdmObject[j].name.find("EventInfo/processedEvents") != std::string::npos ) {
        MEtoEdmObject[j].object += (newMEtoEDMObject[i].object);
      }
      if ( MEtoEdmObject[j].name.find("EventInfo/iEvent") != std::string::npos ||
           MEtoEdmObject[j].name.find("EventInfo/iLumiSection") != std::string::npos) {
        if (MEtoEdmObject[j].object < newMEtoEDMObject[i].object) {
          MEtoEdmObject[j].object = (newMEtoEDMObject[i].object);
        }
      }
    }
  }
  return true;
}

template <>
inline bool
MEtoEDM<TString>::mergeProduct(const MEtoEDM<TString> &newMEtoEDM)
{
  const MEtoEdmObjectVector &newMEtoEDMObject = newMEtoEDM.getMEtoEdmObject();
  const size_t nObjects = newMEtoEDMObject.size();
  //  NOTE: we remember the present size since we will only add content
  //        from newMEtoEDMObject after this point
  const size_t nOldObjects = MEtoEdmObject.size();

  // if the old and new are not the same size, we want to report a problem
  if (nObjects != nOldObjects) {
    std::cout << "WARNING MEtoEDM::mergeProducts(): the lists of histograms to be merged have different sizes: new=" << nObjects << ", old=" << nOldObjects << std::endl;
  }

  for (unsigned int i = 0; i < nObjects; ++i) {
    unsigned int j = 0;
    // see if the name is already in the old container up to the point where
    // we may have added new entries in the container
    const std::string& name = newMEtoEDMObject[i].name;
    if (i < nOldObjects && (MEtoEdmObject[i].name == name)) {
      j = i;
    } else {
      j = 0;
      while (j <  nOldObjects && (MEtoEdmObject[j].name != name) ) ++j;
    }
    if (j >= nOldObjects) {
      // this value is only in the new container, not the old one
#if debug
      std::cout << "WARNING MEtoEDM::mergeProducts(): adding new histogram '" << name << "'" << std::endl;
#endif
      MEtoEdmObject.push_back(newMEtoEDMObject[i]);
    }
  }
  return true;
}

#endif
