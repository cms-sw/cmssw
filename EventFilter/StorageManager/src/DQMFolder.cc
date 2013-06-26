// $Id: DQMFolder.cc,v 1.3 2013/04/22 16:19:36 wmtan Exp $
/// @file: DQMFolder.cc

#include "EventFilter/StorageManager/interface/DQMFolder.h"

#include "TH1.h"
#include "TObjString.h"
#include "TProfile.h"

#include "classlib/utils/Regexp.h"

// 15-Jul-2008, KAB: copied from DQMStore
static const lat::Regexp s_rxmeval ("^<(.*)>(i|f|s|t|qr)=(.*)</\\1>$");

using namespace stor;


DQMFolder::DQMFolder()
{}

DQMFolder::~DQMFolder() 
{
  for (DQMObjectsMap::const_iterator it = dqmObjects_.begin(),
         itEnd = dqmObjects_.end(); it != itEnd; ++it)
  {
    TObject* object = it->second;
    if ( object != NULL ) { delete(object); } 
  }
  dqmObjects_.clear();
}

void DQMFolder::addObjects(const std::vector<TObject*>& toList)
{
  for (std::vector<TObject*>::const_iterator it = toList.begin(), itEnd = toList.end();
       it != itEnd; ++it)
  {
    TObject* object = *it;
    if (object)
    {
      std::string objectName = getSafeMEName(object);
      
      DQMObjectsMap::iterator pos = dqmObjects_.lower_bound(objectName);
      if ( pos == dqmObjects_.end() || (dqmObjects_.key_comp()(objectName, pos->first)) )
      {
        pos = dqmObjects_.insert(pos,
          DQMFolder::DQMObjectsMap::value_type(objectName, object));
      }
      else
      {
        TObject* storedObject = pos->second;
        if ( object->InheritsFrom("TProfile") && 
          storedObject->InheritsFrom("TProfile") )
        {
          TProfile* newProfile    = static_cast<TProfile*>(object);
          TProfile* storedProfile = static_cast<TProfile*>(storedObject);
          if (newProfile->GetEntries() > 0)
          {
            storedProfile->Add(newProfile);
          }
        }
        else if ( object->InheritsFrom("TH1") && 
          storedObject->InheritsFrom("TH1") )
        {
          TH1* newHistogram    = static_cast<TH1*>(object);
          TH1* storedHistogram = static_cast<TH1*>(storedObject);
          if (newHistogram->GetEntries() > 0)
          {
            storedHistogram->Add(newHistogram);
          }
        }
        else
        {
          // 15-Jul-2008, KAB - switch to the first instance at the 
          // request of Andreas Meyer...
          
          //// Unrecognized objects just take the last instance
          //delete(storedObject);
          //folder->dqmObjects_[objectName] = object->Clone(object->GetName());
        }
        delete(object);
      }
    }
  }
}

void DQMFolder::fillObjectVector(std::vector<TObject*>& vector) const
{
  for ( DQMObjectsMap::const_iterator it = dqmObjects_.begin(), itEnd = dqmObjects_.end();
        it != itEnd; ++it )
  {
    vector.push_back(it->second);
  }
}

// 15-Jul-2008, KAB - this method should probably exist in DQMStore
// rather than here, but I'm going for expediency...
// The main purpose of the method is to pull out the ME name
// from scalar MEs (ints, floats, strings)
std::string DQMFolder::getSafeMEName(TObject *object)
{
  std::string rawName = object->GetName();
  std::string safeName = rawName;

  lat::RegexpMatch patternMatch;
  if (dynamic_cast<TObjString *>(object) &&
      s_rxmeval.match(rawName, 0, 0, &patternMatch)) {
    safeName = patternMatch.matchString(rawName, 1);
  }

  return safeName;
}


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
