#include <stdexcept>
#include <cassert>
#include "TGClient.h"
#include "TGHtml.h"
#include "TGText.h"
#include "TSystem.h"
#include "Fireworks/Core/interface/CmsShowHelpPopup.h"

CmsShowHelpPopup::CmsShowHelpPopup (const std::string &filename,
                                    const std::string &windowname,
                                    const TGWindow* p, UInt_t w, UInt_t h)
   : TGTransientFrame(gClient->GetDefaultRoot(), p, w, h),
     m_helpHtml(new TGHtml(this, w, h))
{
   AddFrame(m_helpHtml, new TGLayoutHints(kLHintsTop | kLHintsLeft |
                                          kLHintsExpandX | kLHintsExpandY));
   SetWindowName(windowname.c_str());
   TGText text;
   text.Load(helpFileName(filename).c_str());
  
   static TString path = Form("%s/src/Fireworks/Core/data/",gSystem->Getenv("CMSSW_BASE"));
   if ( gSystem->AccessPathName(path.Data()) ){ // cannot find directory   
      assert(gSystem->Getenv("CMSSW_RELEASE_BASE"));
      path = Form("%s/src/Fireworks/Core/data/",gSystem->Getenv("CMSSW_RELEASE_BASE"));
   }
   m_helpHtml->SetBaseUri(path.Data());
   // printf("%s ... \n",  m_helpHtml->GetBaseUri());
   
   m_helpHtml->ParseText((char *)text.AsString().Data());

   MapSubwindows();
   m_helpHtml->Layout();
}

CmsShowHelpPopup::~CmsShowHelpPopup()
{
   delete m_helpHtml;
}

std::string CmsShowHelpPopup::helpFileName (const std::string &filename)
{
   const char* cmspath = gSystem->Getenv("CMSSW_BASE");
   if(0 == cmspath) {
      throw std::runtime_error("CMSSW_BASE environment variable not set");
   }
   return std::string(cmspath) + "/src/Fireworks/Core/data/" + filename;
}
