//
//  File: EventFilter/src/SMProxyServer/DQMInstance.cc
//
//  Author:  W.Badgett (FNAL)
//
//  Container class for a single instance of a set of DQM objects
//
//  $Id: DQMInstance.cc,v 1.1.2.2 2007/05/08 00:14:50 hcheung Exp $
//

#include <iostream>
#include <vector>

#include "FWCore/Utilities/interface/DebugMacros.h"
#include "EventFilter/StorageManager/interface/DQMInstance.h"
#include "TObject.h"
#include "TH1.h"
#include "TFile.h"

using edm::debugit;
using namespace std;
using namespace stor;

DQMGroupDescriptor::DQMGroupDescriptor(DQMInstance *instance,
				       DQMGroup * group):
  instance_(instance),
  group_(group)
{}

DQMGroupDescriptor::~DQMGroupDescriptor() {}

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
  return( ( currentTime - lastUpdate_->GetSec() ) > readyTime_);
}

DQMGroup::~DQMGroup() 
{
  if ( firstUpdate_ != NULL ) { delete(firstUpdate_);}
  if ( lastUpdate_  != NULL ) { delete(lastUpdate_);}
  if ( lastServed_  != NULL ) { delete(lastServed_);}
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
    std::string groupName = i0->first;
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
  std::string fullObjectName = objectDirectory+"/"+object->GetName();
  DQMGroup * group = dqmGroups_[groupName];
  if ( group == NULL )
  {
    group = new DQMGroup(readyTime_);
    dqmGroups_[groupName] = group;
  }

  group->setLastEvent(eventNumber);
  TObject * storedObject = group->dqmObjects_[fullObjectName];
  if ( storedObject == NULL )
  {
    group->dqmObjects_[fullObjectName] = object->Clone(object->GetName());
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
      delete(storedObject);
      group->dqmObjects_[fullObjectName] = object->Clone(object->GetName());
    }
  }

  group->incrementUpdates();
  nUpdates_++;
  lastUpdate_->Set();
  return(nUpdates_);
}

bool DQMInstance::isStale(int currentTime)
{
  return( ( currentTime - lastUpdate_->GetSec() ) > purgeTime_);
}

int DQMInstance::writeFile(std::string filePrefix)
{
  int reply = 0;
  char fileName[1024];
  TTimeStamp now;
  now.Set();

  for (std::map<std::string, DQMGroup * >::iterator i0 = 
	 dqmGroups_.begin(); i0 != dqmGroups_.end() ; ++i0)
  {
    std::string groupName = i0->first;
    DQMGroup * group = i0->second;
    sprintf(fileName,"%s_%s_%d_%d_%d.root", 
	    filePrefix.c_str(), 
	    groupName.c_str(), 
	    runNumber_, 
	    lumiSection_, 
	    instance_);
    TFile * file = new TFile(fileName,"RECREATE");
    if (( file != NULL ) && file->IsOpen())
    {
      int ctr=0;
      for ( std::map<std::string, TObject *>::iterator i1 = 
	      group->dqmObjects_.begin(); i1 != group->dqmObjects_.end(); ++i1)
      {
	std::string objectName = i1->first;
	TObject *object = i1->second;
	if ( object != NULL ) 
	{
	  file->cd();
	  object->Write();
	  reply++;
	  ctr++;
	}
      }
      file->Close();
      if ( file != NULL ) { delete(file);}
      FDEBUG(1) << "Wrote file " << fileName << " " << ctr << " objects"
		<< std::endl; 
    }
  }
  return(reply);
}

DQMGroup * DQMInstance::getDQMGroup(std::string groupName)
{ return(dqmGroups_[groupName]);}
