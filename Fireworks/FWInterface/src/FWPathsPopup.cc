#include "Fireworks/FWInterface/src/FWPathsPopup.h"
#include "Fireworks/FWInterface/interface/FWFFLooper.h"
#include "FWCore/Framework/interface/ScheduleInfo.h"
#include "FWCore/PythonParameterSet/interface/MakeParameterSets.h"
#include "FWCore/ParameterSet/interface/ProcessDesc.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "Fireworks/Core/src/FWDialogBuilder.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "TGLabel.h"
#include "TGTextEdit.h"
#include "TGText.h"
#include "TSystem.h"
#include "TGTextView.h"
#include "TGHtml.h"

#include <iostream>

FWPathsPopup::FWPathsPopup(FWFFLooper *looper)
   : TGMainFrame(gClient->GetRoot(), 200, 200),
     m_info(0),
     m_looper(looper),
     m_hasChanges(false),
     m_moduleLabel(0),
     m_moduleName(0),
     m_modulePathsText(0),
     m_modulePathsHtml(0),
     m_textEdit(0),
     m_apply(0)
{
   FWDialogBuilder builder(this);
   builder.indent(4)
          .addLabel("Available paths", 10)
          //.addLabel(" ", 15, 1, &m_moduleName)
          //.addLabel(" ", 15, 1 ,&m_moduleLabel)
          .addTextView("", &m_modulePathsText)
          .spaceDown(10)
          //.addHtml(&m_modulePathsHtml)
          .addTextEdit("", &m_textEdit)
          .addTextButton("Apply changes and reload", &m_apply);

   m_apply->Connect("Clicked()", "FWPathsPopup", this, "scheduleReloadEvent()");
   m_apply->SetEnabled(true);

   MapSubwindows();
   Layout();
}

/** Finish the setup of the GUI */
void
FWPathsPopup::setup(const edm::ScheduleInfo *info)
{
   assert(info);
   m_info = info;

   m_info->availableModuleLabels(m_availableModuleLabels);
   m_info->availablePaths(m_availablePaths);

   //makePathsTextView();
   //makePathsHtmlView();
}

void
FWPathsPopup::makePathsHtmlView()
{
  // ROOT, are you really asking me to do it this way?

  TString html;

  html = "<html><head><title>Module paths</title></head><body>";

  for ( std::vector<std::string>::iterator pi = m_availablePaths.begin(),
                                        piEnd = m_availablePaths.end();
        pi != piEnd; ++pi )
  {
    html += "<h1>";
    html += *pi;
    html += "</h1>";

    std::vector<std::string> modulesInPath;
    m_info->modulesInPath(*pi, modulesInPath);

    for ( std::vector<std::string>::iterator mi = modulesInPath.begin(),
                                          miEnd = modulesInPath.end();
          mi != miEnd; ++mi )
    {
      html += "<h2>";
      html += *mi;
      html += "</h2>";
    }
  } 

  html += "</font>";
  html += "</body></html>";
  
  m_modulePathsHtml->ParseText((char*)html.Data());
}

// Make a text view (indenting to create "poor-man's html")
// as right now TGHtml has a font problem which causes a crash

void 
FWPathsPopup::makePathsTextView()
{
  m_modulePathsText->Clear();

  for ( std::vector<std::string>::iterator pi = m_availablePaths.begin(),
                                        piEnd = m_availablePaths.end();
        pi != piEnd; ++pi )
  {
    std::vector<std::string> modulesInPath;
    m_info->modulesInPath(*pi, modulesInPath);

    m_modulePathsText->AddLineFast((*pi).c_str());

    for ( std::vector<std::string>::iterator mi = modulesInPath.begin(),
                                          miEnd = modulesInPath.end();
          mi != miEnd; ++mi )
    {
      std::string str = "  "+(*mi);
      m_modulePathsText->AddLineFast(str.c_str());
  
      const edm::ParameterSet* parameterSet = m_info->parametersForModule(*mi);

      // Need to clean up the formatting of this

      for ( edm::ParameterSet::table::const_iterator ti = 
              parameterSet->tbl().begin(), tiEnd = parameterSet->tbl().end();
            ti != tiEnd; ++ti )
      {
        std::string tstr = "    " + ti->first + ti->second.toString();
        m_modulePathsText->AddLineFast(tstr.c_str());
      }
    
      for ( edm::ParameterSet::psettable::const_iterator pi = 
              parameterSet->psetTable().begin(), piEnd = parameterSet->psetTable().end();
            pi != piEnd; ++pi )
      {
        std::string pstr = "    " + pi->first + pi->second.toString();
        m_modulePathsText->AddLineFast(pstr.c_str());
      }

      for ( edm::ParameterSet::vpsettable::const_iterator vpi = 
              parameterSet->vpsetTable().begin(), vpiEnd = parameterSet->vpsetTable().end();
            vpi != vpiEnd; ++vpi )
      {
        std::string vpstr = "   " + vpi->first + vpi->second.toString();
        m_modulePathsText->AddLineFast(vpstr.c_str());
      }
     
    }
  } 
  m_modulePathsText->Update();
}

/** Gets called by CMSSW as we process events. **/
void
FWPathsPopup::postModule(edm::ModuleDescription const& description)
{
  //m_moduleName->SetText(description.moduleName().c_str());
  //m_moduleLabel->SetText(description.moduleLabel().c_str());
   gSystem->ProcessEvents();
}

void
FWPathsPopup::postProcessEvent(edm::Event const& event, edm::EventSetup const& eventSetup)
{
  gSystem->ProcessEvents();
  makePathsTextView();
}



#include "FWCore/PythonParameterSet/interface/PythonProcessDesc.h"
#include "FWCore/Utilities/interface/Exception.h"

using namespace boost::python;


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
   PythonProcessDesc desc;
   std::string pythonSnippet("import FWCore.ParameterSet.Config as cms\n"
                             "process=cms.Process('Dummy')\n");
   TGText *text = m_textEdit->GetText();
   for (size_t li = 0, le = text->RowCount(); li != le; ++li)
   {
      char *buf = text->GetLine(TGLongPosition(0, li), text->GetLineLength(li));
      if (!buf)
         continue;
      pythonSnippet += buf;
      free(buf);
   }

   try
   {
      PythonProcessDesc pydesc(pythonSnippet);
      boost::shared_ptr<edm::ProcessDesc> desc = pydesc.processDesc();
      boost::shared_ptr<edm::ParameterSet> ps = desc->getProcessPSet();
      const edm::ParameterSet::table &pst = ps->tbl();
      const edm::ParameterSet::table::const_iterator &mi= pst.find("@all_modules");
      if (mi == pst.end())
         throw cms::Exception("cmsShow") << "@all_modules not found";
      // FIXME: we are actually interested in "@all_modules" entry.
      std::vector<std::string> modulesInConfig(mi->second.getVString());
      std::vector<std::string> parameterNames;

      for (size_t mni = 0, mne = modulesInConfig.size(); mni != mne; ++mni)
      {
         const std::string &moduleName = modulesInConfig[mni];
         std::cout << moduleName << std::endl;
         const edm::ParameterSet *modulePSet(ps->getPSetForUpdate(moduleName));
         parameterNames = modulePSet->getParameterNames();
         for (size_t pi = 0, pe = parameterNames.size(); pi != pe; ++pi)
            std::cout << "  " << parameterNames[pi] << std::endl;
         m_looper->requestChanges(moduleName, *modulePSet);
      }
      m_hasChanges = true;
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
