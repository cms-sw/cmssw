// $Id: Stone_SKEL.h 2089 2008-11-23 20:31:03Z matevz $

// Copyright (C) 1999-2008, Matevz Tadel. All rights reserved.
// This file is part of GLED, released under GNU General Public License version 2.
// For the licensing terms see $GLEDSYS/LICENSE or http://www.gnu.org/.

#ifndef Fireworks_Core_TRootXTReq_h
#define Fireworks_Core_TRootXTReq_h

class TMutex;
class TCondition;
class TThread;

class TTimer;
class TSignalHandler;

#include <TString.h>

#include <list>

class TRootXTReq
{
private:
   typedef std::list<TRootXTReq*> lpXTReq_t;

   TCondition               *m_return_condition;

   static lpXTReq_t          sQueue;
   static Long_t             sRootThread;
   static TMutex            *sQueueMutex;
   static TSignalHandler    *sSigHandler;
   static bool               sSheduled;

   virtual void Act() = 0;

protected:
   TString                   mName;

   void post_request();

public:
   TRootXTReq(const char* n="TRootXTReq");
   virtual ~TRootXTReq();

   void ShootRequest();
   void ShootRequestAndWait();

   // --- Static interface ---

   static void Bootstrap(Long_t root_thread);
   static void Shutdown();

   static void ProcessQueue();

};

#endif
