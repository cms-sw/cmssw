#include "Fireworks/FWInterface/src/FWPathsPopup.h"
#include "Fireworks/FWInterface/src/FWPSetTableManager.h"
#include "Fireworks/FWInterface/src/FWPSetCellEditor.h"

#include "Fireworks/FWInterface/interface/FWFFLooper.h"
#include "Fireworks/TableWidget/interface/FWTableWidget.h"


#include "Fireworks/Core/src/FWDialogBuilder.h"
#include "Fireworks/Core/interface/FWGUIManager.h"
#include "Fireworks/Core/interface/fwLog.h"

#include "FWCore/Framework/interface/ScheduleInfo.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Common/interface/TriggerNames.h"

#include "FWCore/PythonParameterSet/interface/PythonProcessDesc.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "DataFormats/Provenance/interface/ProcessHistory.h"
#include "DataFormats/Common/interface/TriggerResults.h"



#include "TGLabel.h"
#include "KeySymbols.h"

void
FWPathsPopup::windowIsClosing()
{
   UnmapWindow();
   DontCallClose();
}

FWPathsPopup::FWPathsPopup(FWFFLooper *looper, FWGUIManager *guiManager)
   : TGMainFrame(gClient->GetRoot(), 400, 600),
     m_info(0),
     m_looper(looper),
     m_hasChanges(false),
     m_moduleLabel(0),
     m_moduleName(0),
     m_apply(0),
     m_psTable(new FWPSetTableManager()),
     m_guiManager(guiManager)
{
   gVirtualX->SelectInput(GetId(), kKeyPressMask | kKeyReleaseMask | kExposureMask |
                          kPointerMotionMask | kStructureNotifyMask | kFocusChangeMask |
                          kEnterWindowMask | kLeaveWindowMask);
   this->Connect("CloseWindow()","FWPathsPopup",this,"windowIsClosing()");

   FWDialogBuilder builder(this);
   builder.indent(4)
      .spaceDown(10)
      .addLabel("Filter:").floatLeft(4).expand(false, false)
      .addTextEntry("", &m_search).expand(true, false)
      .spaceDown(10)
      .addTable(m_psTable, &m_tableWidget).expand(true, true)
      .addTextButton("Apply changes and reload", &m_apply);

   FWPSetCellEditor *editor = new  FWPSetCellEditor(m_tableWidget->body(), "");
   editor->SetBackgroundColor(gVirtualX->GetPixel(kYellow-7));
   editor->SetFrameDrawn(false);
   m_psTable->setCellValueEditor(editor);
   m_psTable->m_editor->Connect("ReturnPressed()", "FWPathsPopup", this, "applyEditor()");

   m_apply->Connect("Clicked()", "FWPathsPopup", this, "scheduleReloadEvent()");
   m_apply->SetEnabled(false);
   m_search->SetEnabled(true);
   m_search->Connect("TextChanged(const char *)", "FWPathsPopup",
                     this, "updateFilterString(const char *)");
   m_tableWidget->SetBackgroundColor(0xffffff);
   m_tableWidget->SetLineSeparatorColor(0x000000);
   m_tableWidget->SetHeaderBackgroundColor(0xececec);
   m_tableWidget->Connect("cellClicked(Int_t,Int_t,Int_t,Int_t,Int_t,Int_t)",
                          "FWPathsPopup",this,
                          "cellClicked(Int_t,Int_t,Int_t,Int_t,Int_t,Int_t)");
   m_tableWidget->disableGrowInWidth();

   SetWindowName("CMSSW Configuration Editor");
   MapSubwindows();
   editor->UnmapWindow();

   Layout();
}

/** Handle pressing of esc. 
 */
Bool_t
FWPathsPopup::HandleKey(Event_t*event)
{
   UInt_t keysym = event->fCode;

   if (keysym == (UInt_t) gVirtualX->KeysymToKeycode(kKey_Escape))
   {
      // called from FWPSetCellEditor
      m_psTable->cancelEditor();
      m_psTable->setSelection(-1, -1, 0);
   }
   return TGMainFrame::HandleKey(event);
}

/** Proxies the applyEditor() method in the model so that it can be connected to GUI, signals.
  */
void
FWPathsPopup::applyEditor()
{
   bool applied = m_psTable->applyEditor();
   if (applied)
      m_apply->SetEnabled(true);
}

/** Handles clicking on the table cells.
    
    * Clicking on a cell in the first column opens / closes a given node. 
    * Clicking on a cell in the second column moves the editor to that cell. 
 
  */
void 
FWPathsPopup::cellClicked(Int_t iRow, Int_t iColumn, Int_t iButton, Int_t iKeyMod, Int_t, Int_t)
{
   if (iButton != kButton1)
      return;
   

   if (iColumn == 0 || iColumn == 1)
   {
      // Save and close the previous editor, if required.
      if (m_psTable->selectedColumn() == 1
          && m_psTable->selectedRow() != -1)
      {
         int oldIndex = m_psTable->rowToIndex()[m_psTable->selectedRow()];
         FWPSetTableManager::PSetData& oldData = m_psTable->data()[oldIndex];

         if (oldData.editable)
            applyEditor();
      }

      m_psTable->setSelection(iRow, iColumn, iKeyMod);

      if (iColumn == 0)
         m_psTable->setExpanded(iRow);
   }
}

void
FWPathsPopup::updateFilterString(const char *str)
{
   m_psTable->applyEditor();
   m_psTable->setSelection(-1, -1, 0);
   m_psTable->updateFilter(str);
}

/** Finish the setup of the GUI */
void
FWPathsPopup::setup(const edm::ScheduleInfo *info)
{
   assert(info);
   m_info = info;
}

/** Gets called by CMSSW as modules are about to be processed. **/
void
FWPathsPopup::postModule(edm::ModuleDescription const& description)
{
   m_guiManager->updateStatus((description.moduleName() + " processed.").c_str());
   gSystem->ProcessEvents();
}

/** Gets called by CMSSW as we process modules. **/
void
FWPathsPopup::preModule(edm::ModuleDescription const& description)
{
   m_guiManager->updateStatus(("Processing " + description.moduleName() + "...").c_str());
   gSystem->ProcessEvents();
}


void
FWPathsPopup::postProcessEvent(edm::Event const& event, edm::EventSetup const& eventSetup)
{
   m_guiManager->updateStatus("Done processing.");
   gSystem->ProcessEvents();

   // Get the last process name from the process history:
   // this should be the one specified in the cfg file
 
   if (event.processHistory().empty()) {
      fwLog(fwlog::kInfo) << "Path GUI:: no process history available.\n";
      return;
   }
   edm::ProcessHistory::const_iterator pi = event.processHistory().end() - 1;
   std::string processName = pi->processName();
   
   // It's called TriggerResults but actually contains info on all paths
   edm::InputTag tag("TriggerResults", "", processName);
   edm::Handle<edm::TriggerResults> triggerResults;
   event.getByLabel(tag, triggerResults);

   std::vector<FWPSetTableManager::PathUpdate> pathUpdates;

   if (triggerResults.isValid())
   {
      edm::TriggerNames triggerNames = event.triggerNames(*triggerResults);
     
      for (size_t i = 0, e = triggerResults->size(); i != e; ++i)
      {
         FWPSetTableManager::PathUpdate update;
         update.pathName = triggerNames.triggerName(i);
         update.passed = triggerResults->accept(i);
         update.choiceMaker = triggerResults->index(i);
         pathUpdates.push_back(update);
      }
   }
   m_psTable->updateSchedule(m_info);
   m_psTable->update(pathUpdates);
   m_psTable->dataChanged();
   m_tableWidget->body()->DoRedraw();
}



/** Modifies the module and asks the looper to reload the event.
 
    1. Read the configuration snippet from the GUI,
    2. Use the python interpreter to parse it and get the new
      parameter set.
    3. Notify the looper about the changes.

    FIXME: implement 2 and 3.
  */
void
FWPathsPopup::scheduleReloadEvent()
{
   applyEditor();
   try
   {
      for (size_t mni = 0, mne = m_psTable->modules().size(); mni != mne; ++mni)
      {
         FWPSetTableManager::ModuleInfo &module = m_psTable->modules()[mni];
         if (module.dirty == false)
            continue;
         FWPSetTableManager::PSetData &data = m_psTable->entries()[module.entry];
         m_looper->requestChanges(data.label, * module.current_pset);
      }
      m_hasChanges = true;
      m_apply->SetEnabled(false);
      gSystem->ExitLoop();
   }
   catch (boost::python::error_already_set)
   {
      edm::pythonToCppException("Configuration");
      Py_Finalize();
   }
   catch (cms::Exception &exception)
   {
      std::cout << exception.what() << std::endl;
   }
   // Return control to the FWFFLooper so that it can decide what to do next.
}
