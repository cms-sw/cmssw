#ifndef Modules_IterateNTimesLooper_h
#define Modules_IterateNTimesLooper_h
// -*- C++ -*-
//
// Package:     Modules
// Class  :     IterateNTimesLooper
// 
/**\class IterateNTimesLooper IterateNTimesLooper.h FWCore/Modules/interface/IterateNTimesLooper.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Tue Jul 11 11:16:12 EDT 2006
// $Id: IterateNTimesLooper.h,v 1.2 2006/07/28 13:24:35 valya Exp $
//

// system include files

// user include files
#include "FWCore/Framework/interface/EDLooper.h"

// forward declarations

class IterateNTimesLooper : public edm::EDLooper
{

   public:
      IterateNTimesLooper(const edm::ParameterSet& );
      virtual ~IterateNTimesLooper();

      // ---------- const member functions ---------------------

      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------
      virtual void startingNewLoop(unsigned int ) ; 
      virtual Status duringLoop(const edm::Event&, const edm::EventSetup&) ; 
      virtual Status endOfLoop(const edm::EventSetup&, unsigned int) ; 
      
   private:
      IterateNTimesLooper(const IterateNTimesLooper&); // stop default

      const IterateNTimesLooper& operator=(const IterateNTimesLooper&); // stop default

      // ---------- member data --------------------------------
      unsigned int max_;
      unsigned int times_;
      bool shouldStop_;
};


#endif
