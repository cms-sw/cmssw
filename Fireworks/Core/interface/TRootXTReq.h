// $Id: TRootXTReq.h,v 1.2 2010/07/01 18:49:58 chrjones Exp $

#ifndef Fireworks_Core_TRootXTReq_h
#define Fireworks_Core_TRootXTReq_h

class TMutex;
class TCondition;
class TThread;

class TTimer;
class TSignalHandler;

#include <TString.h>

#include <list>
#include <pthread.h>

class TRootXTReq
{
private:
   typedef std::list<TRootXTReq*> lpXTReq_t;

   TCondition               *m_return_condition;

   static lpXTReq_t          sQueue;
   static pthread_t          sRootThread;
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

   static void Bootstrap(pthread_t root_thread);
   static void Shutdown();

   static void ProcessQueue();

};

#endif
