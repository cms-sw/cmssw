#ifndef Fireworks_Core_CmsShowTaskTimer_h
#define Fireworks_Core_CmsShowTaskTimer_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     CmsShowTaskTimer
//
/**\class CmsShowTaskTimer CmsShowTaskTimer.h Fireworks/Core/src/CmsShowTaskTimer.h

   Description: <one line class summary>

   Usage:
    <usage>

 */
//
// Original Author:  Joshua Berger
//         Created:  Fri Jul 25 11:49:18 EDT 2008
//

// system include files
#include <TTimer.h>

// user include files

// forward declarations
class CmsShowTaskExecutorBase;

class CmsShowTaskTimer : public TTimer
{

public:
   CmsShowTaskTimer(CmsShowTaskExecutorBase* taskExec, Long_t milliSec = 0, Bool_t mode = kTRUE);
   virtual ~CmsShowTaskTimer();

   // ---------- const member functions ---------------------

   // ---------- static member functions --------------------

   // ---------- member functions ---------------------------
   virtual Bool_t Notify();

private:
   CmsShowTaskTimer(const CmsShowTaskTimer&);    // stop default

   const CmsShowTaskTimer& operator=(const CmsShowTaskTimer&);    // stop default

   // ---------- member data --------------------------------
   CmsShowTaskExecutorBase* m_taskExec;

};


#endif
