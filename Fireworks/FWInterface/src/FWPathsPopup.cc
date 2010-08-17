#include "Fireworks/FWInterface/src/FWPathsPopup.h"
#include "FWCore/Framework/interface/ScheduleInfo.h"
#include "Fireworks/Core/src/FWDialogBuilder.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "TGLabel.h"
#include "TSystem.h"

FWPathsPopup::FWPathsPopup(const edm::ScheduleInfo *info)
   : TGMainFrame(gClient->GetRoot(), 200, 200),
     m_info(info),
     m_moduleName(0),
     m_moduleLabel(0)
{
   FWDialogBuilder builder(this);
   builder.indent(4)
          .addLabel(" ", 15, 1, &m_moduleName)
          .addLabel(" ", 15, 1 ,&m_moduleLabel);
   MapSubwindows();
   Layout();
}

/** Gets called by CMSSW as we process events. **/
void
FWPathsPopup::postModule(edm::ModuleDescription const& description)
{
   m_moduleName->SetText(description.moduleName().c_str());
   m_moduleLabel->SetText(description.moduleLabel().c_str());
   gSystem->ProcessEvents();
}
