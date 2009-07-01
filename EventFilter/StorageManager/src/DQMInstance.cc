//
//  File: EventFilter/src/SMProxyServer/DQMInstance.cc
//
//  Author:  W.Badgett (FNAL)
//
//  Container class for a single instance of a set of DQM objects
//

#include <iostream>

#include "FWCore/Utilities/interface/DebugMacros.h"
#include "EventFilter/StorageManager/interface/DQMInstance.h"
#include "TH1.h"
#include "TObjArray.h"
#include "TObjString.h"
#include "TString.h"
#include "TDirectory.h"
#include "sys/stat.h"
#include "classlib/utils/Regexp.h"
#include "boost/lexical_cast.hpp"

// 15-Jul-2008, KAB: copied from DQMStore
static const lat::Regexp s_rxmeval ("^<(.*)>(i|f|s|qr)=(.*)</\\1>$");

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
}

DQMGroup::DQMGroup(int readyTime):
  nUpdates_(0),
  readyTime_(readyTime)
{
  lastUpdate_  = new TTimeStamp(0,0);
  lastServed_  = new TTimeStamp(0,0);
  firstUpdate_ = new TTimeStamp(0,0);
  firstUpdate_->Set();
}

void DQMGroup::incrementUpdates() 
{ 
  nUpdates_++;
  wasServedSinceUpdate_ = false;
  lastUpdate_->Set();  
}

bool DQMGroup::isReady(int currentTime)
{
  time_t lastUpdateSecs = lastUpdate_->GetSec();
  // 29-Oct-2008, KAB - added a test that lastUpdateSecs is greater
  // than zero so that we don't report that a brand-new group is
  // ready before any updates have been added.
  return( lastUpdateSecs > 0 &&
          ( currentTime - lastUpdateSecs ) > readyTime_);
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
}


DQMInstance::DQMInstance(int runNumber, 
			 int lumiSection, 
			 int instance,
			 int purgeTime,
			 int readyTime):
  runNumber_(runNumber),
  lumiSection_(lumiSection),
  instance_(instance),
  nUpdates_(0),
  purgeTime_(purgeTime),
  readyTime_(readyTime)
{
  firstUpdate_ = new TTimeStamp();
  lastUpdate_  = new TTimeStamp();
}


DQMInstance::~DQMInstance()
{
  if ( firstUpdate_ != NULL ) { delete(firstUpdate_);}
  if ( lastUpdate_ != NULL )  { delete(lastUpdate_);}

  for (std::map<std::string, DQMGroup * >::iterator i0 = 
	 dqmGroups_.begin(); i0 != dqmGroups_.end() ; ++i0)
  {
    DQMGroup  * group     = i0->second;
    if ( group != NULL ) { delete(group); } 
  }
}


int DQMInstance::updateObject(std::string groupName,
			      std::string objectDirectory, 
			      TObject    *object,
			      int eventNumber)
{
  lastEvent_ = eventNumber;
  std::string objectName = getSafeMEName(object);
  DQMGroup * group = dqmGroups_[groupName];
  if ( group == NULL )
  {
    group = new DQMGroup(readyTime_);
    dqmGroups_[groupName] = group;
  }

  DQMFolder * folder = group->dqmFolders_[objectDirectory];
  if ( folder == NULL )
  {
    folder = new DQMFolder();
    group->dqmFolders_[objectDirectory] = folder;
  }

  group->setLastEvent(eventNumber);
  TObject * storedObject = folder->dqmObjects_[objectName];
  if ( storedObject == NULL )
  {
    folder->dqmObjects_[objectName] = object->Clone(object->GetName());
  }
  else
  {
    if ( object->InheritsFrom("TH1") && 
	 storedObject->InheritsFrom("TH1") )
    {
      TH1 * newHistogram    = (TH1 *)object;
      TH1 * storedHistogram = (TH1 *)storedObject;
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

  group->incrementUpdates();
  nUpdates_++;
  lastUpdate_->Set();
  return(nUpdates_);
}

bool DQMInstance::isReady(int currentTime)
{
  bool readyFlag = true;

  // 29-Oct-2008, KAB - if there are no groups, switch
  // the default to false so that newly constructed DQMInstance
  // objects don't report ready==true before any groups have
  // even been created.
  if (dqmGroups_.size() == 0) readyFlag = false;

  for (std::map<std::string, DQMGroup * >::iterator i0 = 
	 dqmGroups_.begin(); i0 != dqmGroups_.end() ; ++i0)
  {
    DQMGroup  * group     = i0->second;
    if ( group != NULL && ! group->isReady(currentTime) ) {
      readyFlag = false;
    }
  }
  return readyFlag;
}

bool DQMInstance::isStale(int currentTime)
{
  return( ( currentTime - lastUpdate_->GetSec() ) > purgeTime_);
}

int DQMInstance::writeFile(std::string filePrefix, bool endRunFlag)
{
  int reply = 0;
  char fileName[1024];
  TTimeStamp now;
  now.Set();
  std::string runString("Run ");
  runString.append(boost::lexical_cast<std::string>(runNumber_));

  for (std::map<std::string, DQMGroup * >::iterator i0 = 
	 dqmGroups_.begin(); i0 != dqmGroups_.end() ; ++i0)
  {
    DQMGroup * group = i0->second;
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
	    reply++;
	    ctr++;
	  }
	}
      }
      file->Close();
      delete(file);
      chmod(fileName,S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | S_IROTH );
      FDEBUG(1) << "Wrote file " << fileName << " " << ctr << " objects"
		<< std::endl; 
    }
  }
  return(reply);
}

DQMGroup * DQMInstance::getDQMGroup(std::string groupName)
{ return(dqmGroups_[groupName]);}

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
