// $Id: DQMInstance.cc,v 1.19 2010/03/04 17:00:33 mommsen Exp $
/// @file: DQMInstance.cc

#include <cstdio>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <sys/types.h>
#include <sys/stat.h>

#include "FWCore/Utilities/interface/DebugMacros.h"
#include "EventFilter/StorageManager/interface/DQMInstance.h"

#include "TH1.h"
#include "TProfile.h"
#include "TObjArray.h"
#include "TObjString.h"
#include "TString.h"
#include "TDirectory.h"

#include "classlib/utils/Regexp.h"

// 15-Jul-2008, KAB: copied from DQMStore
static const lat::Regexp s_rxmeval ("^<(.*)>(i|f|s|t|qr)=(.*)</\\1>$");

using edm::debugit;
using namespace std;
using namespace stor;


////////////////////////
// DQMGroupDescriptor //
////////////////////////

DQMGroupDescriptor::DQMGroupDescriptor(DQMInstance *instance,
				       DQMGroup * group):
  instance_(instance),
  group_(group)
{}

DQMGroupDescriptor::~DQMGroupDescriptor() {}


///////////////
// DQMFolder //
///////////////

DQMFolder::DQMFolder()
{}

DQMFolder::~DQMFolder() 
{
  for (DQMObjectsMap::iterator i0 = 
	 dqmObjects_.begin(); i0 != dqmObjects_.end() ; ++i0)
  {
    TObject * object = i0->second;
    if ( object != NULL ) { delete(object); } 
  }
  dqmObjects_.clear();
}

void DQMFolder::addObjects(std::vector<TObject *> toList)
{
  for (std::vector<TObject *>::const_iterator it = toList.begin(), itEnd = toList.end();
       it != itEnd; ++it)
  {
    TObject *object = *it;
    if (object)
    {
      std::string objectName = getSafeMEName(object);
      
      DQMObjectsMap::iterator pos = dqmObjects_.lower_bound(objectName);
      if ( pos == dqmObjects_.end() || (dqmObjects_.key_comp()(objectName, pos->first)) )
      {
        pos = dqmObjects_.insert(pos, DQMFolder::DQMObjectsMap::value_type(objectName,
            object->Clone(object->GetName())));
      }
      else
      {
        TObject * storedObject = pos->second;
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
      delete(object);
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

unsigned int DQMFolder::writeObjects() const
{
  unsigned int ctr(0);
  for ( DQMObjectsMap::const_iterator it = dqmObjects_.begin(), itEnd = dqmObjects_.end();
        it != itEnd; ++it)
  {
    std::string objectName = it->first;
    TObject *object = it->second;
    if ( object != NULL ) 
    {
      object->Write();
      ctr++;
    }
  }
  return ctr;
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


//////////////
// DQMGroup //
//////////////

DQMGroup::DQMGroup(const time_t readyTime, const unsigned int expectedUpdates):
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

DQMGroup::~DQMGroup() 
{
  delete(firstUpdate_);
  delete(lastUpdate_);
  delete(lastServed_);

  for (std::map<std::string, DQMFolder * >::iterator i0 = 
	 dqmFolders_.begin(); i0 != dqmFolders_.end() ; ++i0)
  {
    DQMFolder * folder = i0->second;
    if ( folder != NULL ) { delete(folder); } 
  }
  dqmFolders_.clear();
}

void DQMGroup::addEvent(std::auto_ptr<DQMEvent::TObjectTable> toTablePtr)
{
  for (
    DQMEvent::TObjectTable::const_iterator it = toTablePtr->begin(),
      itEnd = toTablePtr->end();
    it != itEnd; 
    ++it
  ) 
  {
    std::string subFolderName = it->first;

    DQMFoldersMap::iterator pos = dqmFolders_.lower_bound(subFolderName);
    if ( pos == dqmFolders_.end() || (dqmFolders_.key_comp()(subFolderName, pos->first)) )
    {
      pos = dqmFolders_.insert(pos, DQMGroup::DQMFoldersMap::value_type(subFolderName,
          new DQMFolder()));
    }
    DQMFolder * folder = pos->second;
    folder->addObjects(it->second);
  }

  ++nUpdates_;
  wasServedSinceUpdate_ = false;
  lastUpdate_->Set();  
}

size_t DQMGroup::populateTable(DQMEvent::TObjectTable& table) const
{
  size_t subFolderSize = 0;

  for ( DQMFoldersMap::const_iterator it = dqmFolders_.begin(), itEnd = dqmFolders_.end();
        it != itEnd; ++it )
  {
    std::string folderName = it->first;
    const DQMFolder * folder = it->second;

    DQMEvent::TObjectTable::iterator pos = table.lower_bound(folderName);
    if ( pos == table.end() || (table.key_comp()(folderName, pos->first)) )
    {
      std::vector<TObject *> newObjectVector;
      pos = table.insert(pos, DQMEvent::TObjectTable::value_type(folderName, newObjectVector));
      subFolderSize += 2*sizeof(uint32) + folderName.length();
    }
    folder->fillObjectVector(pos->second);
  }
  return subFolderSize;
}

unsigned int DQMGroup::fillFile(TFile* file, const TString& runString) const
{
  unsigned int ctr(0);

  // First create directories inside the root file
  for ( DQMFoldersMap::const_iterator it = dqmFolders_.begin(), itEnd = dqmFolders_.end();
        it != itEnd; ++it)
  {
    TString path(it->first.c_str());
    const DQMFolder * folder = it->second;

    TObjArray * tokens = path.Tokenize("/");

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
    tmpEntry = new TObjString(runString);
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

    ctr += folder->writeObjects();
  }
  return ctr;
}

bool DQMGroup::isStale(time_t now) const
{
  time_t lastUpdateSecs = lastUpdate_->GetSec();
  return ( lastUpdateSecs > 0 && (now - lastUpdateSecs) > readyTime_);
}


/////////////////
// DQMInstance //
/////////////////

DQMInstance::DQMInstance(const int runNumber,
			 const int lumiSection,
			 const int updateNumber,
			 const time_t purgeTime,
                         const time_t readyTime,
                         const unsigned int expectedUpdates):
  runNumber_(runNumber),
  lumiSection_(lumiSection),
  updateNumber_(updateNumber),
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
  delete(firstUpdate_);
  delete(lastUpdate_);

  for (DQMGroupsMap::iterator i0 = 
	 dqmGroups_.begin(); i0 != dqmGroups_.end() ; ++i0)
  {
    DQMGroup * group = i0->second;
    if ( group != NULL ) { delete(group); } 
  }
  dqmGroups_.clear();
}

void DQMInstance::addEvent(const std::string topFolderName, std::auto_ptr<DQMEvent::TObjectTable> toTablePtr)
{
  DQMGroupsMap::iterator pos = dqmGroups_.lower_bound(topFolderName);
  if ( pos == dqmGroups_.end() || (dqmGroups_.key_comp()(topFolderName, pos->first)) )
  {
    pos = dqmGroups_.insert(pos, DQMGroupsMap::value_type(topFolderName,
        new DQMGroup(readyTime_,expectedUpdates_)));
  }
  DQMGroup * group = pos->second;
  group->addEvent(toTablePtr);

  ++nUpdates_;
  lastUpdate_->Set();
}

bool DQMInstance::isReady() const
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
    if ( group && ! (group->isComplete() && group->wasServedSinceUpdate()) ) {
      readyFlag = false;
    }
  }
  return readyFlag;
}

bool DQMInstance::isStale(time_t now) const
{
  time_t lastUpdateSecs = lastUpdate_->GetSec();
  return( lastUpdateSecs > 0 && (now - lastUpdateSecs) > purgeTime_);
}

double DQMInstance::writeFile(std::string filePrefix, bool endRunFlag) const
{
  double size(0);
  ostringstream runStr;
  runStr << "Run " << runNumber_;
  const TString runString( runStr.str() );

  ostringstream fileNameStr;
  fileNameStr << filePrefix << "/DQM_V0001_EvF_R"
    << std::setfill('0') << std::setw(9) << runNumber_;
  if ( !endRunFlag )
  {
    fileNameStr << "_L" << std::setw(6) << lumiSection_;
  }
  fileNameStr << ".root";
  const TString fileName( fileNameStr.str() );

  TFile * file = new TFile(fileName,"UPDATE");
  if ( file && file->IsOpen() )
  {
    const double originalFileSize = file->GetSize();
    unsigned int ctr(0);
  
    for (DQMGroupsMap::const_iterator it = dqmGroups_.begin(), itEnd = dqmGroups_.end();
         it != itEnd; ++it)
    {
      ctr += it->second->fillFile(file, runString);
    }
    
    file->Close();
    size = file->GetSize() - originalFileSize;
    delete(file);
    chmod(fileName,S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | S_IROTH );
    FDEBUG(1) << "Wrote file " << fileName << " " << ctr << " objects"
              << std::endl; 
  }
  return size;
}

DQMGroup * DQMInstance::getDQMGroup(std::string groupName) const
{
  DQMGroup* dqmGroup = 0;
  DQMGroupsMap::const_iterator it = dqmGroups_.find(groupName);
  if ( it != dqmGroups_.end() ) dqmGroup = it->second;
  return dqmGroup;
}



/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
