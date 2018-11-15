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
   ~FWFFService() override;

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

   void checkPosition() override;
   void stopPlaying() override {}
   void autoLoadNewEvent() override {}

   void quit() override;
private:
   FWFFService(const FWFFService&) = delete;                  // stop default
   const FWFFService& operator=(const FWFFService&) = delete; // stop default

   // ---------- member data --------------------------------
   
   std::unique_ptr<FWFFNavigator>        m_navigator;
   std::unique_ptr<FWFFMetadataManager>  m_metadataManager;
   std::unique_ptr<fireworks::Context>   m_context;

   FWFFHelper    m_appHelper;
   TEveManager  *m_EveManager;
   TRint        *m_Rint;

   TEveMagField *m_MagField;
   

   bool          m_AllowStep;
   bool          m_ShowEvent;
   bool          m_firstTime;
};

#endif
