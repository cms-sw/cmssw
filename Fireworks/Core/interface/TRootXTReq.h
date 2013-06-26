// $Id: TRootXTReq.h,v 1.3 2011/01/24 17:44:21 amraktad Exp $

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
