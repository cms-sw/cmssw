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
#include "TGHtml.h"

#include <iostream>
#include <sstream>

FWPathsPopup::FWPathsPopup(FWFFLooper *looper)
   : TGMainFrame(gClient->GetRoot(), 200, 200),
     m_info(0),
     m_looper(looper),
     m_hasChanges(false),
     m_moduleLabel(0),
     m_moduleName(0),
     m_modulePathsHtml(0),
     m_textEdit(0),
     m_apply(0)
{
   FWDialogBuilder builder(this);
   builder.indent(4)
          .addLabel("Available paths", 10)
          .spaceDown(10)
          .addHtml(&m_modulePathsHtml)
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

   makePathsView();
}

// It would be nice if we could use some of the 
// utilities from Entry. 
// Why couldn't the type just be the type?
const char*
FWPathsPopup::typeCodeToChar(char typeCode)
{
  switch(typeCode)
  {
  case 'b':  return "Bool";
  case 'B':  return "VBool";
  case 'i' : return "vint32";
  case 'I' : return "int32";
  case 'u' : return "vuint32";
  case 'U' : return "uint32";
  case 'l' : return "vint64";
  case 'L' : return "int64";
  case 'x' : return "vuint64";
  case 'X' : return "uint64";
  case 's' : return "vstring";
  case 'S' : return "string";
  case 'd' : return "vdouble";
  case 'D' : return "double";
  case 'p' : return "vPSet";
  case 'P' : return "PSet";
  case 'T' : return "path";
  case 'F' : return "FileInPath";
  case 't' : return "InputTag";
  case 'v' : return "VInputTag";
  case 'e' : return "VEventID";
  case 'E' : return "EventID";
  case 'm' : return "VLuminosityBlockID";
  case 'M' : return "LuminosityBlockID";
  case 'a' : return "VLuminosityBlockRange";    
  case 'A' : return "LuminosityBlockRange";
  case 'r' : return "VEventRange";
  case 'R' : return "EventRange";
  default:   return "Type not known";
  }
}

// Crikey! I'm ending up writing a ParameterSet parser here!
// I probably could use the results from the << operator
// in Entry but it's not quite what I want for format.
// Also, it's not clear to me how to break it up into html
// elements.
void 
FWPathsPopup::handleEntry(const edm::Entry& entry, 
                          const std::string& key, TString& html)
{
  html += "<li>" + key + "    " 
          + (entry.isTracked() ? "tracked    " : "untracked    ")
          + typeCodeToChar(entry.typeCode());

  switch(entry.typeCode())
  {
  case 'b':
    {
      std::stringstream ss;
      ss << entry.getBool();
      html += "    " + ss.str();
      break;
    }
  case 'B':
    {
      html += "   [Ack! no access from entry for VBool?]";
      break;
    }
  case 'i':
    {
      std::stringstream ss;
      html += "    ";
      std::vector<int> ints = entry.getVInt32();
      for ( std::vector<int>::const_iterator ii = ints.begin(), iiEnd = ints.end();
            ii != iiEnd; ++ii )
      {
        ss << *ii <<"  ";
        html += ss.str();
      }
      break;
    }
  case 'I':
    {
      std::stringstream ss;
      ss << entry.getInt32();
      html += "   " + ss.str();
      break;
    }
  case 'u':
    {
      std::stringstream ss;
      html += "    ";
      std::vector<unsigned> us = entry.getVUInt32();
      for ( std::vector<unsigned>::const_iterator ui = us.begin(), uiEnd = us.end();
            ui != uiEnd; ++ui )
      {
        ss << *ui <<" ";
        html += ss.str();
      } 
      break;
    }
  case 'U':
    {
      std::stringstream ss;
      ss << entry.getUInt32();
      html += "   " + ss.str();
      break;
    }
  case 'l':
    {
      std::stringstream ss;
      html += "    ";
      std::vector<long long> ints = entry.getVInt64();
      for ( std::vector<long long>::const_iterator ii = ints.begin(), iiEnd = ints.end();
            ii != iiEnd; ++ii )
      {
        ss << *ii << " ";
        html += ss.str();
      }
      break;
    }
  case 'L':
    {
      std::stringstream ss;
      ss << entry.getInt64();
      html += "   " + ss.str();
      break;
    }
  case 'x':
    {
      std::stringstream ss;
      html += "    ";
      // This the 1st time in my life I have written "unsigned long long"! Exciting.
      std::vector<unsigned long long> us = entry.getVUInt64();
      for ( std::vector<unsigned long long>::const_iterator ui = us.begin(), uiEnd = us.end();
            ui != uiEnd; ++ui )
      {
        ss << *ui <<" ";
        html += ss.str();
      }
      break;
    }
  case 'X':
    {
      std::stringstream ss;
      ss << entry.getUInt64();
      html += "    " + ss.str();
      break;
    }
  case 's':
    {
      std::vector<std::string> strs = entry.getVString();
      html += "    ";
      for ( std::vector<std::string>::const_iterator si = strs.begin(), siEnd = strs.end();
            si != siEnd; ++si )
      {
        html += *si + " ";
      }
      break;
    }
  case 'S':
    {
      html += "    " + entry.getString();
      break;
    }
  case 'd':
    {
      std::stringstream ss;
      html += "    ";
      std::vector<double> ds = entry.getVDouble();
      for ( std::vector<double>::const_iterator di = ds.begin(), diEnd = ds.end();
            di != diEnd; ++di )
      {
        ss << *di <<" ";
        html += ss.str();
      }
      break;
    }
  case 'D':
    {   
      std::stringstream ss;
      ss << entry.getDouble();
      html += "    " + ss.str();
      break;
    }
  case 'p':
    {
      std::vector<edm::ParameterSet> psets = entry.getVPSet();
      html += "    ";
      for ( std::vector<edm::ParameterSet>::const_iterator psi = psets.begin(), psiEnd = psets.end();
            psi != psiEnd; ++psi )
      {
        handlePSet(&(*psi), html);
      }
      break;
    }
  case 'P':
    {    
      handlePSet(&(entry.getPSet()), html);
      break;
    }
  case 'v':
    {
      std::stringstream ss;
      html += "    ";
      std::vector<edm::InputTag> tags = entry.getVInputTag();
      for ( std::vector<edm::InputTag>::const_iterator ti = tags.begin(), tiEnd = tags.end();
            ti != tiEnd; ++ti )
      {
        ss << ti->encode() <<" ";
        html += ss.str();
      }
      break;
    }
  default:
    {
      html += "   [Not supported yet. Are you sure you want this?]";
      break;
    }
  }
}

void 
FWPathsPopup::handlePSetEntry(const edm::ParameterSetEntry& entry, 
                              const std::string& key, TString& html)
{
  html += "<li>" + key + "    " 
          + (entry.isTracked() ? "tracked    " : "untracked    ")
          + "PSet";

  handlePSet(&(entry.pset()), html);
}

void 
FWPathsPopup::handleVPSetEntry(const edm::VParameterSetEntry& entry, 
                               const std::string& key, TString& html)
{
  html += "<li>" + key + "    " 
          + (entry.isTracked() ? "tracked    " : "untracked    ")
          + "vPSet";

  for ( std::vector<edm::ParameterSet>::const_iterator psi = entry.vpset().begin(),
                                                    psiEnd = entry.vpset().end();
        psi != psiEnd; ++psi )
  {
    handlePSet(&(*psi), html);
  }
}       

void 
FWPathsPopup::handlePSet(const edm::ParameterSet* ps, TString& html)
{
  html += "<ul>";

  for ( edm::ParameterSet::table::const_iterator ti = 
          ps->tbl().begin(), tiEnd = ps->tbl().end();
        ti != tiEnd; ++ti )
    handleEntry(ti->second, ti->first, html);
    
  for ( edm::ParameterSet::psettable::const_iterator pi = 
          ps->psetTable().begin(), piEnd = ps->psetTable().end();
        pi != piEnd; ++pi )
    handlePSetEntry(pi->second, pi->first, html);

  for ( edm::ParameterSet::vpsettable::const_iterator vpi = 
          ps->vpsetTable().begin(), vpiEnd = ps->vpsetTable().end();
        vpi != vpiEnd; ++vpi )
    handleVPSetEntry(vpi->second, vpi->first, html);

  html += "</ul>";
}

//#include <fstream>
//std:ofstream fout("path-view.html");

void
FWPathsPopup::makePathsView()
{
  m_modulePathsHtml->Clear();

  TString html;

  html = "<html><head><title>Available paths</title></head><body>";

  for ( std::vector<std::string>::iterator pi = m_availablePaths.begin(),
                                        piEnd = m_availablePaths.end();
        pi != piEnd; ++pi )
  {
    html += "<h1>"+ *pi + "</h1>";

    std::vector<std::string> modulesInPath;
    m_info->modulesInPath(*pi, modulesInPath);

    for ( std::vector<std::string>::iterator mi = modulesInPath.begin(),
                                          miEnd = modulesInPath.end();
          mi != miEnd; ++mi )
    {
      const edm::ParameterSet* ps = m_info->parametersForModule(*mi);

      // Need to get the module type from the parameter set before we handle the set itself
      const edm::ParameterSet::table& pst = ps->tbl();    
      const edm::ParameterSet::table::const_iterator ti = pst.find("@module_edm_type");

      html += "<h2>" + ti->second.getString() + "  " + *mi  + "</h2>";
      handlePSet(ps, html); 
    }
  } 

  html += "</body></html>";
  //fout<< html <<std::endl;

  m_modulePathsHtml->ParseText((char*)html.Data());
}


/** Gets called by CMSSW as we process events. **/
void
FWPathsPopup::postModule(edm::ModuleDescription const& description)
{
   gSystem->ProcessEvents();
}

void
FWPathsPopup::postProcessEvent(edm::Event const& event, edm::EventSetup const& eventSetup)
{
  gSystem->ProcessEvents();
  makePathsView();
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
