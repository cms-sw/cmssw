// $Id: DQMInstance.cc,v 1.17 2010/03/03 15:23:21 mommsen Exp $
/// @file: DQMInstance.cc

#include <iostream>

#include "FWCore/Utilities/interface/DebugMacros.h"
#include "EventFilter/StorageManager/interface/DQMInstance.h"
#include "TH1.h"
#include "TProfile.h"
#include "TObjArray.h"
#include "TObjString.h"
#include "TString.h"
#include "TDirectory.h"
#include "sys/stat.h"
#include "classlib/utils/Regexp.h"
#include "boost/lexical_cast.hpp"
#include <cstdio>

// 15-Jul-2008, KAB: copied from DQMStore
static const lat::Regexp s_rxmeval ("^<(.*)>(i|f|s|t|qr)=(.*)</\\1>$");

using edm::debugit;
using namespace std;
using namespace stor;

DQMGroupDescriptor::DQMGroupDescriptor(DQMInstance *instance,
				       DQMGroup * group):
  instance_(instance),
  group_(group)
{}

DQMGroupDescriptor::~DQMGroupDescriptor() {}

DQMFolder::DQMFolder()
{}

DQMFolder::~DQMFolder() 
{
  for (std::map<std::string, TObject * >::iterator i0 = 
	 dqmObjects_.begin(); i0 != dqmObjects_.end() ; ++i0)
  {
    TObject * object = i0->second;
    if ( object != NULL ) { delete(object); } 
  }
  dqmObjects_.clear();
}

DQMGroup::DQMGroup(const int readyTime, const unsigned int expectedUpdates):
  nUpdates_(0),
  readyTime_(readyTime),
  expectedUpdates_(expectedUpdates),
  wasServedSinceUpdate_(false)
{
  lastUpdate_  = new TTimeStamp(0,0);
  lastServed_  = new TTimeStamp(0,0);
  firstUpdate_ = new TTimeStamp(0,0);
  firstUpdate_->Set();
}

void DQMGroup::setLastEvent(int lastEvent)
{
  if ( lastEvent_ != lastEvent )
  {
    lastEvent_ = lastEvent;
    ++nUpdates_;
  }
  wasServedSinceUpdate_ = false;
  lastUpdate_->Set();  
}

void DQMGroup::setServedSinceUpdate()
{
  wasServedSinceUpdate_=true;
}

bool DQMGroup::isReady(int currentTime) const
{
  return ( isComplete() || isStale(currentTime) );
}

bool DQMGroup::isComplete() const
{
  return ( expectedUpdates_ == nUpdates_ );
}

bool DQMGroup::isStale(int currentTime) const
{
  time_t lastUpdateSecs = lastUpdate_->GetSec();
  return ( lastUpdateSecs > 0 && (currentTime - lastUpdateSecs) > readyTime_);
}

DQMGroup::~DQMGroup() 
{
  if ( firstUpdate_ != NULL ) { delete(firstUpdate_);}
  if ( lastUpdate_  != NULL ) { delete(lastUpdate_);}
  if ( lastServed_  != NULL ) { delete(lastServed_);}
  for (std::map<std::string, DQMFolder * >::iterator i0 = 
	 dqmFolders_.begin(); i0 != dqmFolders_.end() ; ++i0)
  {
    DQMFolder * folder = i0->second;
    if ( folder != NULL ) { delete(folder); } 
  }
  dqmFolders_.clear();
}


DQMInstance::DQMInstance(const int runNumber, 
			 const int lumiSection, 
			 const int instance,
			 const int purgeTime,
                         const int readyTime,
                         const unsigned int expectedUpdates):
  runNumber_(runNumber),
  lumiSection_(lumiSection),
  instance_(instance),
  nUpdates_(0),
  purgeTime_(purgeTime),
  readyTime_(readyTime),
  expectedUpdates_(expectedUpdates)
{
  firstUpdate_ = new TTimeStamp();
  lastUpdate_  = new TTimeStamp();
}


DQMInstance::~DQMInstance()
{
  if ( firstUpdate_ != NULL ) { delete(firstUpdate_);}
  if ( lastUpdate_ != NULL )  { delete(lastUpdate_);}

  for (DQMGroupsMap::iterator i0 = 
	 dqmGroups_.begin(); i0 != dqmGroups_.end() ; ++i0)
  {
    DQMGroup * group = i0->second;
    if ( group != NULL ) { delete(group); } 
  }
  dqmGroups_.clear();
}


unsigned int DQMInstance::updateObject(const std::string groupName,
                                       const std::string objectDirectory, 
			               TObject          *object,
			               const int         eventNumber)
{
  lastEvent_ = eventNumber;
  std::string objectName = getSafeMEName(object);

  DQMGroupsMap::iterator groupPos = dqmGroups_.lower_bound(groupName);
  if ( groupPos == dqmGroups_.end() || (dqmGroups_.key_comp()(groupName, groupPos->first)) )
  {
    groupPos = dqmGroups_.insert(groupPos, DQMGroupsMap::value_type(groupName,
        new DQMGroup(readyTime_,expectedUpdates_)));
  }
  DQMGroup * group = groupPos->second;

  DQMGroup::DQMFoldersMap::iterator folderPos = group->dqmFolders_.lower_bound(objectDirectory);
  if ( folderPos == group->dqmFolders_.end() || (group->dqmFolders_.key_comp()(objectDirectory, folderPos->first)) )
  {
    folderPos = group->dqmFolders_.insert(folderPos, DQMGroup::DQMFoldersMap::value_type(objectDirectory,
        new DQMFolder()));
  }
  DQMFolder * folder = folderPos->second;

  DQMFolder::DQMObjectsMap::iterator objectPos = folder->dqmObjects_.lower_bound(objectName);
  if ( objectPos == folder->dqmObjects_.end() || (folder->dqmObjects_.key_comp()(objectName, objectPos->first)) )
  {
    objectPos = folder->dqmObjects_.insert(objectPos, DQMFolder::DQMObjectsMap::value_type(objectName,
        object->Clone(object->GetName())));
  }
  else
  {
    TObject * storedObject = objectPos->second;
    if ( object->InheritsFrom("TProfile") && 
	 storedObject->InheritsFrom("TProfile") )
    {
      TProfile * newProfile    = static_cast<TProfile*>(object);
      TProfile * storedProfile = static_cast<TProfile*>(storedObject);
      storedProfile->Add(newProfile);
    }
    else if ( object->InheritsFrom("TH1") && 
	 storedObject->InheritsFrom("TH1") )
    {
      TH1 * newHistogram    = static_cast<TH1*>(object);
      TH1 * storedHistogram = static_cast<TH1*>(storedObject);
      storedHistogram->Add(newHistogram);
    }
    else
    {
      // 15-Jul-2008, KAB - switch to the first instance at the 
      // request of Andreas Meyer...

      //// Unrecognized objects just take the last instance
      //delete(storedObject);
      //folder->dqmObjects_[objectName] = object->Clone(object->GetName());
    }
  }

  group->setLastEvent(eventNumber);
  nUpdates_++;
  lastUpdate_->Set();
  return(nUpdates_);
}

bool DQMInstance::isReady(int currentTime) const
{
  // 29-Oct-2008, KAB - if there are no groups, return false
  // so that newly constructed DQMInstance objects don't report
  // ready==true before any groups have even been created.
  if ( dqmGroups_.empty() ) return false;

  bool readyFlag = true;

  for (DQMGroupsMap::const_iterator
         it = dqmGroups_.begin(), itEnd = dqmGroups_.end();
       it != itEnd ; ++it)
  {
    DQMGroup * group = it->second;
    if ( group && ! (group->isReady(currentTime) && group->wasServedSinceUpdate()) ) {
      readyFlag = false;
    }
  }
  return readyFlag;
}

bool DQMInstance::isStale(int currentTime) const
{
  time_t lastUpdateSecs = lastUpdate_->GetSec();
  return( lastUpdateSecs > 0 && (currentTime - lastUpdateSecs) > purgeTime_);
}

double DQMInstance::writeFile(std::string filePrefix, bool endRunFlag) const
{
  double size = 0;
  char fileName[1024];
  TTimeStamp now;
  now.Set();
  std::string runString("Run ");
  runString.append(boost::lexical_cast<std::string>(runNumber_));

  for (DQMGroupsMap::const_iterator
	 it = dqmGroups_.begin(), itEnd = dqmGroups_.end();
       it != itEnd; ++it)
  {
    DQMGroup * group = it->second;
    if (endRunFlag) {
      sprintf(fileName,"%s/DQM_V0001_EvF_R%9.9d.root", 
              filePrefix.c_str(), runNumber_);
    }
    else {
      sprintf(fileName,"%s/DQM_V0001_EvF_R%9.9d_L%6.6d.root", 
              filePrefix.c_str(), runNumber_, lumiSection_);
    }

    TFile * file = new TFile(fileName,"UPDATE");
    if (( file != NULL ) && file->IsOpen())
    {
      int ctr=0;
      double originalFileSize = file->GetSize();

      // First create directories inside the root file
      TString token("/");
      for ( std::map<std::string, DQMFolder *>::iterator i1 = 
	      group->dqmFolders_.begin(); i1 != group->dqmFolders_.end(); ++i1)
      {
	std::string folderName = i1->first;
	TString path(folderName.c_str());
	DQMFolder * folder = i1->second;

	TObjArray * tokens = path.Tokenize(token);

        // 15-Oct-2008, KAB - add several extra levels to the
        // directory structure to match consumer-based histograms.
        // The TObjArray is the owner of the memory used by its elements,
        // so it takes care of deleting the extra entries that we add.
        TObjString * tmpEntry;
        int origSize = tokens->GetEntries();
        if (origSize >= 2) {
          tokens->Expand(origSize + 3);
        }
        else {
          tokens->Expand(origSize + 2);
        }
        for (int idx = origSize-1; idx >= 0; --idx) {
          tmpEntry = (TObjString *) tokens->RemoveAt(idx);
          if (origSize >= 2 && idx > 0) {
            tokens->AddAt(tmpEntry, idx+3);
          }
          else {
            tokens->AddAt(tmpEntry, idx+2);
          }
        }
        if (origSize >= 2) {
          tmpEntry = new TObjString("Run summary");
          tokens->AddAt(tmpEntry, 3);
        }
        tmpEntry = new TObjString(runString.c_str());
        tokens->AddAt(tmpEntry, 1);
        tmpEntry = new TObjString("DQMData");
        tokens->AddAt(tmpEntry, 0);

	int nTokens = tokens->GetEntries();
	TDirectory * newDir = NULL;
	TDirectory * oldDir = (TDirectory *)file;
	for ( int j=0; j<nTokens; j++)
	{
	  TString newDirName = ((TObjString *)tokens->At(j))->String();
	  oldDir->cd();
	  newDir = oldDir->GetDirectory(newDirName.Data(),kFALSE,"cd");
	  if ( newDir == NULL )
	  {
	    newDir = oldDir->mkdir(newDirName.Data());
	  }
	  oldDir = newDir;
	}
	delete(tokens);
        oldDir->cd();

	for ( std::map<std::string, TObject *>::iterator i2 = 
	      folder->dqmObjects_.begin(); i2 != folder->dqmObjects_.end(); 
	      ++i2)
	{
	  std::string objectName = i2->first;
	  TObject *object = i2->second;
	  if ( object != NULL ) 
	  {
	    object->Write();
	    ctr++;
	  }
	}
      }
      file->Close();
      size += file->GetSize() - originalFileSize;
      delete(file);
      chmod(fileName,S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | S_IROTH );
      FDEBUG(1) << "Wrote file " << fileName << " " << ctr << " objects"
		<< std::endl; 
    }
  }
  return(size);
}

DQMGroup * DQMInstance::getDQMGroup(std::string groupName) const
{
  DQMGroup* dqmGroup = 0;
  DQMGroupsMap::const_iterator it = dqmGroups_.find(groupName);
  if ( it != dqmGroups_.end() ) dqmGroup = it->second;
  return dqmGroup;
}

// 15-Jul-2008, KAB - this method should probably exist in DQMStore
// rather than here, but I'm going for expediency...
// The main purpose of the method is to pull out the ME name
// from scalar MEs (ints, floats, strings)
std::string DQMInstance::getSafeMEName(TObject *object)
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
