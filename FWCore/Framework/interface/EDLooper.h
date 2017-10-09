#ifndef FWCore_Framework_EDLooper_h
#define FWCore_Framework_EDLooper_h
// -*- C++ -*-
//
// Package:     Framework
// Module:      EDLooper
// 
/**\class EDLooper EDLooper.h FWCore/Framework/interface/EDLooper.h

 Description: Standard base class for looping components
 
 This abstract class forms the basis of being able to loop through a list of events multiple times.
 
*/
//
// Author:      Valentin Kuznetsov
// Created:     Wed Jul  5 11:42:17 EDT 2006
//

#include "FWCore/Framework/interface/EDLooperBase.h"

#include <set>
#include <memory>

namespace edm {

  class EDLooper : public EDLooperBase
  {
    public:

      EDLooper();
      virtual ~EDLooper();

      EDLooper(EDLooper const&) = delete; // Disallow copying and moving
      EDLooper& operator=(EDLooper const&) = delete; // Disallow copying and moving

    private:

    /**Called after all event modules have had a chance to process the edm::Event.
     */
    virtual Status duringLoop(const edm::Event&, const edm::EventSetup&) = 0; 
    
    /**override base class interface and just call the above duringLoop
     */
    virtual Status duringLoop(const edm::Event&, const edm::EventSetup&, ProcessingController& );
    
    
  };
}

#endif
