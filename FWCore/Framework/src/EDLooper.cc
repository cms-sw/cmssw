// -*- C++ -*-
//
// Package:     <package>
// Module:      EDLooper
// 
// Description: <one line class summary>
//
// Implementation:
//     <Notes on implementation>
//
// Author:      Valentin Kuznetsov
// Created:     Wed Jul  5 11:44:26 EDT 2006
// $Id: EDLooper.cc,v 1.8 2007/06/29 03:43:21 wmtan Exp $
//
// Revision history
//
// $Log: EDLooper.cc,v $
// Revision 1.8  2007/06/29 03:43:21  wmtan
// Remove unnecessary #includes
//
// Revision 1.7  2007/06/25 23:22:13  wmtan
// Remove unnecessary includes
//
// Revision 1.6  2007/06/22 23:26:33  wmtan
// Add Run and Lumi loops to the EventProcessor
//
// Revision 1.5  2007/01/19 05:25:11  wmtan
// Evaluate end() only at the beginning of an iteration
//
// Revision 1.4  2006/12/19 00:28:56  wmtan
// changed (u)long to (u)int so that data is the same size on 32 and 64 bit machines
//
// Revision 1.3  2006/10/13 01:47:35  wmtan
// Remove unnecessary argument from runOnce()
//
// Revision 1.2  2006/07/28 13:24:34  valya
// Modified endOfLoop, now it accepts counter as a second argument. Add EDLooper calls to beginOfJob/endOfJob in EventProcessor
//
// Revision 1.1  2006/07/23 01:24:34  valya
// Add looper support into framework. The base class is EDLooper. All the work done in EventProcessor and EventHelperLooper
//

// system include files
// You may have to uncomment some of these or other stl headers
// depending on what other header files you include (e.g. FrameAccess etc.)!
#include <iostream>
#include <sstream>
//#include <vector>
//#include <set>
//#include <map>
//#include <algorithm>
//#include <utility>

// user include files
#include "FWCore/Framework/interface/EDLooper.h"
#include "FWCore/Framework/interface/EDLooperHelper.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventHelperDescription.h"

namespace edm {
//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
EDLooper::EDLooper() : name_("EDLooper"), passID_("pass"), processID_(passID_)
{
}

// EDLooper::EDLooper( const EDLooper& rhs )
// {
//    // do actual copying here; if you implemented
//    // operator= correctly, you may be able to use just say      
//    *this = rhs;
// }

EDLooper::~EDLooper()
{
}

//
// assignment operators
//
// const EDLooper& EDLooper::operator=( const EDLooper& rhs )
// {
//   if( this != &rhs ) {
//      // do actual copying here, plus:
//      // "SuperClass"::operator=( rhs );
//   }
//
//   return *this;
// }

//
// member functions
//

//called once per job just before the first processing 
void 
EDLooper::beginOfJob(const edm::EventSetup&)
{
}
  
//called just before the job is going to end 
void EDLooper::endOfJob()
{
}

void
EDLooper::loop(EDLooperHelper& iHelper, 
              unsigned int numberToProcess) 
{
   unsigned int iCounter = 0;
   Status status=kContinue;
   /* // show up all available data
   const std::set<eventsetup::EventSetupRecordKey>& recs = modifyingRecords();
   std::set<eventsetup::EventSetupRecordKey>::const_iterator iter = recs.begin(), iterEnd = recs.end();
   for(iter; iter != iterEnd; ++iter) {
      std::cout<<iter->name()<<std::endl;
   }
   */
   const edm::EventSetup* eventSetup = 0;
   do {
       boost::shared_ptr<edm::LuminosityBlockPrincipal> lbp;
     boost::shared_ptr<edm::RunPrincipal> rp;
       startingNewLoop(iCounter);
       do {
           EventHelperDescription evtDesc = iHelper.runOnce(rp,lbp);
           if(evtDesc.eventPrincipal_.get()==0) {
              break;
           }
           std::auto_ptr<edm::EventPrincipal> pep = evtDesc.eventPrincipal_;
           eventSetup = evtDesc.eventSetup_;
           edm::ModuleDescription modDesc;
           modDesc.moduleName_="EDLooper";
           modDesc.moduleLabel_="";
           Event event(*pep.get(),modDesc);
           status = duringLoop(event,*eventSetup);
           if (status!=kContinue) {
               break;
           }
       } while(1);
       status = endOfLoop(*eventSetup,iCounter);
       if (status!=kContinue) {
           break;
       }
       ++iCounter;
       // modify passID of the looper to keep track how many times we process the same loop
       std::ostringstream pid;
       pid<<iCounter;
       passID_=processID_+"_"+pid.str();
       iHelper.rewind(modifyingRecords());
   } while(1);
   return;
}

//
// const member functions
//
std::set<eventsetup::EventSetupRecordKey> 
EDLooper::modifyingRecords() const
{
  return std::set<eventsetup::EventSetupRecordKey> ();
}
//
// static member functions
//

}
