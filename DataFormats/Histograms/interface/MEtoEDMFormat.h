#ifndef MEtoEDMFormat_h
#define MEtoEDMFormat_h

/** \class MEtoEDM
 *  
 *  DataFormat class to hold the information from a ME tranformed into
 *  ROOT objects as appropriate
 *
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
#include <THashList.h>
#include <TList.h>

#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <map>
#include <stdint.h>

#define METOEDMFORMAT_DEBUG 0

namespace {
  //utility function to check the consistency of the axis labels
  //taken from TH1::CheckBinLabels
  bool CheckBinLabels(const TAxis* a1, const TAxis * a2)
  {
    // check that axis have same labels
    THashList *l1 = (const_cast<TAxis*>(a1))->GetLabels();
    THashList *l2 = (const_cast<TAxis*>(a2))->GetLabels();
    
    if (!l1 && !l2 )
      return true;
    if (!l1 ||  !l2 ) {
      return false;
    }
    // check now labels sizes  are the same
    if (l1->GetSize() != l2->GetSize() ) {
      return false;
    }
    for (int i = 1; i <= a1->GetNbins(); ++i) {
      TString label1 = a1->GetBinLabel(i);
      TString label2 = a2->GetBinLabel(i);
      if (label1 != label2) {
	return false;
      }
    }
    return true;
  }
}

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
#if METOEDMFORMAT_DEBUG
       std::cout << "WARNING MEtoEDM::mergeProducts(): adding new histogram '" << name << "'" << std::endl;
#endif
       MEtoEdmObject.push_back(newMEtoEDMObject[i]);
     } else if (MEtoEdmObject[j].object.CanExtendAllAxes() && newMEtoEDMObject[i].object.CanExtendAllAxes()) {
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
           MEtoEdmObject[j].object.GetZaxis()->GetXmax() == newMEtoEDMObject[i].object.GetZaxis()->GetXmax() &&
	   CheckBinLabels((TAxis*)MEtoEdmObject[j].object.GetXaxis(),(TAxis*)newMEtoEDMObject[i].object.GetXaxis()) &&
	   CheckBinLabels((TAxis*)MEtoEdmObject[j].object.GetYaxis(),(TAxis*)newMEtoEDMObject[i].object.GetYaxis()) &&
	   CheckBinLabels((TAxis*)MEtoEdmObject[j].object.GetZaxis(),(TAxis*)newMEtoEDMObject[i].object.GetZaxis()) ) {
         MEtoEdmObject[j].object.Add(&newMEtoEDMObject[i].object);
       } else {
          std::cout << "ERROR MEtoEDM::mergeProducts(): found histograms with different axis limits or different labels, '" << name << "' not merged" <<  std::endl;
#if METOEDMFORMAT_DEBUG
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
#if METOEDMFORMAT_DEBUG
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
#if METOEDMFORMAT_DEBUG
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
#if METOEDMFORMAT_DEBUG
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
#if METOEDMFORMAT_DEBUG
      std::cout << "WARNING MEtoEDM::mergeProducts(): adding new histogram '" << name << "'" << std::endl;
#endif
      MEtoEdmObject.push_back(newMEtoEDMObject[i]);
    }
  }
  return true;
}

#endif
