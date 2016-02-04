#ifndef Fireworks_Core_FWFFService_h
#define Fireworks_Core_FWFFService_h


#include "Fireworks/Core/interface/CmsShowMainBase.h"
#include "Fireworks/FWInterface/interface/FWFFHelper.h"
#include <string>
#include <Rtypes.h>
#include <memory>

namespace edm
{
   class ParameterSet;
   class ActivityRegistry;
   class Run;
   class Event;
   class EventSetup;
}

class FWFFNavigator;
class FWFFMetadataManager;

namespace fireworks
{
   class Context;
}

class TEveManager;
class TEveElement;
class TEveMagField;
class TEveTrackPropagator;
class TRint;

class FWFFService : public CmsShowMainBase
{
public:
   FWFFService(const edm::ParameterSet&, edm::ActivityRegistry&);
   virtual ~FWFFService();

   // ---------- const member functions ---------------------

   // ---------- static member functions --------------------

   // ---------- member functions ---------------------------

   void postBeginJob();
   void postEndJob();

   void postBeginRun(const edm::Run&, const edm::EventSetup&);

   void postProcessEvent(const edm::Event&, const edm::EventSetup&);

   void display(const std::string& info="");

   TEveMagField* getMagField();
   void          setupFieldForPropagator(TEveTrackPropagator* prop);

   virtual void checkPosition();
   virtual void stopPlaying() {}
   virtual void autoLoadNewEvent() {}

   void quit();
private:
   FWFFService(const FWFFService&);                  // stop default
   const FWFFService& operator=(const FWFFService&); // stop default

   // ---------- member data --------------------------------
   
   std::auto_ptr<FWFFNavigator>        m_navigator;
   std::auto_ptr<FWFFMetadataManager>  m_metadataManager;
   std::auto_ptr<fireworks::Context>   m_context;

   FWFFHelper    m_appHelper;
   TEveManager  *m_EveManager;
   TRint        *m_Rint;

   TEveMagField *m_MagField;
   

   bool          m_AllowStep;
   bool          m_ShowEvent;
   bool          m_firstTime;
};

#endif
