#include <stdexcept>
#include "TGClient.h"
#include "TGHtml.h"
#include "TGText.h"
#include "TSystem.h"
#include "Fireworks/Core/interface/CmsShowHelpPopup.h"

CmsShowHelpPopup::CmsShowHelpPopup (const TGWindow* p, UInt_t w, UInt_t h)
     : TGTransientFrame(gClient->GetDefaultRoot(), p, w, h),
       m_helpHtml(new TGHtml(this, w, h))
{
     AddFrame(m_helpHtml, new TGLayoutHints(kLHintsTop | kLHintsLeft |
					    kLHintsExpandX | kLHintsExpandY));
     SetWindowName("CmsShow help");
     TGText text;
     text.Load("random.html");
     m_helpHtml->ParseText((char *)text.AsString().Data());
     MapSubwindows();
     m_helpHtml->Layout();
}

CmsShowHelpPopup::~CmsShowHelpPopup()
{
     delete m_helpHtml;
}

const char *CmsShowHelpPopup::helpFileName ()
{
     static TString *help_file = 0;
     if (help_file == 0) {
	  const char* cmspath = gSystem->Getenv("CMSSW_BASE");
	  if(0 == cmspath) {
	       throw std::runtime_error("CMSSW_BASE environment variable not set");
	  }
	  help_file = new TString(Form("%s/src/Fireworks/Core/scripts/help.html",
				       gSystem->Getenv("CMSSW_BASE")));
     }
     return help_file->Data();
}
