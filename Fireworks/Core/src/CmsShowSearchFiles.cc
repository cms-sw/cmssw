#include <stdexcept>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <set>
#include "TGClient.h"
#include "TGHtml.h"
#include "TGButton.h"
#include "TGMenu.h"
#include "TGLabel.h"
#include "TGTextEntry.h"
#include "TPluginManager.h"
#include "TUrl.h"
#include "TSocket.h"
#include "TSystem.h"
#include "TVirtualX.h"
#include "TPRegexp.h"
#include "Fireworks/Core/interface/CmsShowSearchFiles.h"
#include "Fireworks/Core/interface/fwLog.h"
#include "Fireworks/Core/interface/fwPaths.h"

class FWHtml : public TGHtml {
public:
   FWHtml(const TGWindow* p, int w, int h, int id = -1):
   TGHtml(p,w,h,id) {}
   
   int	IsVisited(const char* iCheck) override {
      std::string check(GetBaseUri());
      check +=iCheck;
      int value = m_visited.find(check)==m_visited.end()?kFALSE:kTRUE;
      return value;
   }
   void addToVisited(const char* iToVisit) {
      m_visited.insert(iToVisit);
   }
private:
   std::set<std::string> m_visited;
};

const static unsigned int s_columns = 3;
const static char* const s_prefixes[][s_columns] ={ 
  {"http://", "Web site known by you",0},
  {"file:","Local file [you must type full path name]",0},
  {"dcap://","dCache [FNAL]",0},
  {"rfio://","Castor [CERN]",0}
  
};

static const std::string s_httpPrefix("http:");
static const std::string s_filePrefix("file:");
static const std::string s_rootPostfix(".root");


namespace {
float
getURLResponseTime(const char* url)
{
  TString com = "ping -q -c 1 -n " + TString(url) + "| tail -n 1";
  FILE* p = gSystem->OpenPipe(com, "r");
  TString l;
  l.Gets(p);
  gSystem->ClosePipe(p);

  TPMERegexp re("([\\d\\.]+)");
  if (re.Match(l))
    return  re[1].Atof();
  else
    return -1;
}
}

CmsShowSearchFiles::CmsShowSearchFiles (const char *filename,
                                        const char* windowname,
                                        const TGWindow* p, UInt_t w, UInt_t h)
   : TGTransientFrame(gClient->GetDefaultRoot(), p, w, h)
{
   TGVerticalFrame* vf = new TGVerticalFrame(this);
   this->AddFrame(vf,new TGLayoutHints(kLHintsExpandX|kLHintsExpandY,5,5,5,5));
   TGHorizontalFrame* urlFrame = new TGHorizontalFrame(this);
   vf->AddFrame(urlFrame,new TGLayoutHints(kLHintsExpandX,5,0,5,5));
   
   TGLabel* urlLabel = new TGLabel(urlFrame,"URL");
   urlFrame->AddFrame(urlLabel, new TGLayoutHints(kLHintsLeft|kLHintsCenterY,1,1,1,1));
   m_choosePrefix = new TGTextButton(urlFrame,"Choose Prefix");
   urlFrame->AddFrame(m_choosePrefix, new TGLayoutHints(kLHintsLeft,1,1,1,1));
   
   m_file= new TGTextEntry(urlFrame);
   urlFrame->AddFrame(m_file, new TGLayoutHints(kLHintsExpandX,1,0,1,1));
   m_file->Connect("TextChanged(const char*)", "CmsShowSearchFiles",this,"fileEntryChanged(const char*)");
   m_file->Connect("ReturnPressed()", "CmsShowSearchFiles",this,"updateBrowser()");
   
   m_webFile = new FWHtml(vf,1,1);
   m_webFile->Connect("MouseDown(const char*)","CmsShowSearchFiles",this,"hyperlinkClicked(const char*)");
   vf->AddFrame(m_webFile, new TGLayoutHints(kLHintsExpandX|kLHintsExpandY,1,1,1,1));
   
   TGHorizontalFrame* buttonFrame = new TGHorizontalFrame(vf);
   vf->AddFrame(buttonFrame, new TGLayoutHints(kLHintsExpandX,1,10,1,10));
   
   m_openButton = new TGTextButton(buttonFrame,"Open");
   buttonFrame->AddFrame(m_openButton, new TGLayoutHints(kLHintsRight,5,5,1,1));
   m_openButton->SetEnabled(kFALSE);
   m_openButton->Connect("Clicked()","CmsShowSearchFiles",this,"openClicked()");

   TGTextButton* cancel = new TGTextButton(buttonFrame,"Cancel");
   buttonFrame->AddFrame(cancel, new TGLayoutHints(kLHintsRight,5,5,1,1));
   cancel->Connect("Clicked()","CmsShowSearchFiles",this,"UnmapWindow()");

   SetWindowName(windowname);
   float x1 = getURLResponseTime("lxplus.cern.ch");
   float x2 = getURLResponseTime("uaf.t2.ucsd.edu");
   // printf("timtes %f %f \n", x1, x2); fflush(stdout);

   std::string path;
   if (x1 > 0 && x1 < x2)
      path = Form("http://fireworks.web.cern.ch/fireworks/%d/",fireworks::supportedDataFormatsVersion()[0] );
   else if (x2 > 0)
      path = Form("http://uaf.t2.ucsd.edu/fireworks/%d/", fireworks::supportedDataFormatsVersion()[0]);

   if (!path.empty())
      fwLog(fwlog::kInfo) << "Search files at " << path  << "." << std::endl;

   sendToWebBrowser(path);


   MapSubwindows();
   Layout();
   m_prefixMenu=0;
   m_choosePrefix->Connect("Clicked()","CmsShowSearchFiles",this,"showPrefixes()");
}

CmsShowSearchFiles::~CmsShowSearchFiles()
{
   delete m_prefixMenu;
}

//______________________________________________________________________________

void 
CmsShowSearchFiles::prefixChoosen(Int_t iIndex)
{
   //   printf ("pref chosen %d \n", iIndex);
   m_file->SetText(m_prefixes[iIndex].c_str(),kFALSE);
   m_openButton->SetEnabled(kFALSE);

   if(m_prefixComplete[iIndex]) {
      //gClient->NeedRedraw(this);
      gClient->NeedRedraw(m_choosePrefix);
      gClient->NeedRedraw(m_webFile);
      gClient->ProcessEventsFor(this);
      updateBrowser();
   } else {
      std::string p;
      sendToWebBrowser(p);
   }
}

//______________________________________________________________________________


void 
CmsShowSearchFiles::fileEntryChanged(const char* iFileName)
{
   std::string fileName =iFileName;
   size_t index = fileName.find_last_of(".");
   std::string postfix;
   if(index != std::string::npos) {
      postfix=fileName.substr(index,std::string::npos);
   }
   if(postfix ==s_rootPostfix) {
      m_openButton->SetEnabled(kTRUE);
   } else {
      m_openButton->SetEnabled(kFALSE);
   }
}

//______________________________________________________________________________

void 
CmsShowSearchFiles::updateBrowser()
{
   std::string n = m_file->GetText();
   sendToWebBrowser(n);
}


//______________________________________________________________________________

void 
CmsShowSearchFiles::hyperlinkClicked(const char* iLink)
{
   m_file->SetText(iLink,kTRUE);
   
   m_webFile->addToVisited(iLink);
   std::string fileName =iLink;
   size_t index = fileName.find_last_of(".");
   std::string postfix = fileName.substr(index,std::string::npos);
   
   if(postfix !=s_rootPostfix) {
      updateBrowser();
   } else {
      openClicked();
   }
}

//______________________________________________________________________________

void 
CmsShowSearchFiles::openClicked()
{
   m_openCalled=true;
   this->UnmapWindow();
}

//______________________________________________________________________________

void 
CmsShowSearchFiles::showPrefixes()
{
   if(0==m_prefixMenu) {
      m_prefixMenu = new TGPopupMenu(this);
      const char* const (*itEnd)[s_columns] = s_prefixes+sizeof(s_prefixes)/sizeof(const char*[3]);
      int index = 0;
      for(const char* const (*it)[s_columns] = s_prefixes;
          it != itEnd;
          ++it,++index) {
         //only add the protocols this version of the code actually can load
         std::string prefix = std::string((*it)[0]).substr(0,std::string((*it)[0]).find_first_of(":")+1);
         if(s_httpPrefix==prefix ||
            s_filePrefix==prefix ||
            (gPluginMgr->FindHandler("TSystem",prefix.c_str()) && 
             gPluginMgr->FindHandler("TSystem",prefix.c_str())->CheckPlugin() != -1)) {
	    m_prefixMenu->AddEntry((std::string((*it)[0])+" ("+((*it)[1])+")").c_str(),index);
	    m_prefixes.push_back((*it)[0]);
	    m_prefixComplete.push_back(0!=(*it)[2]);
         }
      }
      m_prefixMenu->Connect("Activated(Int_t)","CmsShowSearchFiles",this,"prefixChoosen(Int_t)");
   }
   m_prefixMenu->PlaceMenu(m_choosePrefix->GetX(),m_choosePrefix->GetY(),true,true);
}


//______________________________________________________________________________


    //Copied from TGHtmlBrowser
static std::string readRemote(const char *url)
{
   // Read (open) remote files.
   
   char *buf = 0;
   TUrl fUrl(url);

   TString msg = "GET ";
   msg += fUrl.GetProtocol();
   msg += "://";
   msg += fUrl.GetHost();
   msg += ":";
   msg += fUrl.GetPort();
   msg += "/";
   msg += fUrl.GetFile();
   msg += "\r\n";
   
   TString uri(url);
   if (!uri.BeginsWith("http://"))
      return std::string();
   TSocket s(fUrl.GetHost(), fUrl.GetPort());
   if (!s.IsValid())
      return std::string();
   if (s.SendRaw(msg.Data(), msg.Length()) == -1)
      return std::string();
   Int_t size = 1024*1024;
   buf = (char *)calloc(size, sizeof(char));
   if (s.RecvRaw(buf, size) == -1) {
      free(buf);
      return std::string();
   }
   std::string returnValue(buf);
   free(buf);
   return returnValue;
}

//______________________________________________________________________________

void 
CmsShowSearchFiles::sendToWebBrowser(std::string& fileName)
{
   //  std::cout << "CmsShowSearchFiles::sendToWebBrowser " <<  fileName << std::endl ;

   size_t index = fileName.find_first_of(":");
   if(index != std::string::npos) {
      ++index;
   } else {
      index = 0;
   }
   std::string prefix = fileName.substr(0,index);

   m_webFile->Clear();
   if(prefix == s_httpPrefix) {
      gVirtualX->SetCursor(GetId(),gVirtualX->CreateCursor(kWatch));
      //If you clicked a hyperlink then the cursor is still a hand but we now
      // want it to be a watch
      gVirtualX->SetCursor(m_webFile->GetId(),gVirtualX->CreateCursor(kWatch));
      //If we don't call ProcessEventsFor then the cursor will not be updated
      gClient->ProcessEventsFor(this);
      TUrl url(fileName.c_str());
      std::string buffer = readRemote(url.GetUrl());

      if (buffer.size()) {
         m_webFile->SetBaseUri(url.GetUrl());
         m_webFile->ParseText(const_cast<char*>(buffer.c_str()));
      }
      else {
         readError();
      }
      gVirtualX->SetCursor(GetId(),gVirtualX->CreateCursor(kPointer));
      gVirtualX->SetCursor(m_webFile->GetId(),gVirtualX->CreateCursor(kPointer));
   } else {
      readInfo();
   }
   m_webFile->Layout();
}

//______________________________________________________________________________
void 
CmsShowSearchFiles::readError()
{

  const static char *s_readError[] = {
      "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.01 Transitional//EN\" \"http://www.w3c.org/TR/1999/REC-html401-19991224/loose.dtd\"> ",
      "<HTML><HEAD><TITLE>HTTP Read Error</TITLE> ",
      "<META http-equiv=Content-Type content=\"text/html; charset=UTF-8\"></HEAD> ",
      "<BODY> ",
      "<P>Unknown error while trying to get file via http</P>",
      "</BODY></HTML> ",
      0
   };

   m_webFile->SetBaseUri("");
   for (int i=0; s_readError[i]; i++) {
      m_webFile->ParseText(const_cast<char *>(s_readError[i]));
   }
}

void 
CmsShowSearchFiles::readInfo()
{
   const static char *s_noBrowserMessage[] = {
      "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.01 Transitional//EN\" \"http://www.w3c.org/TR/1999/REC-html401-19991224/loose.dtd\"> ",
      "<HTML><HEAD><TITLE>No Browser Available</TITLE> ",
      "<META http-equiv=Content-Type content=\"text/html; charset=UTF-8\"></HEAD> ",
      "<BODY> ",
      //"No file browser is available for this prefix.  You can still type the full URL into the above text box to open the EDM ROOT file.<BR>",
      //"Only a prefix beginning in <STRONG>http:</STRONG> which contains a site name (e.g. http://www.site.org) is supported for browsing."
      "<b>Welcome....</b><BR>",
      "<BR>",
      "<b>You may look at examples:</b><BR>",  
      "If you are in Europe, open example data files at CERN :  ", " <a href=" , Form("http://fireworks.web.cern.ch/fireworks/%d/",fireworks::supportedDataFormatsVersion()[0] ),"> http://fireworks.web.cern.ch/fireworks/ </a><BR>",
      "If you are in US, open example data files at UCSD:  ", " <a href=" , Form("http://uaf-2.t2.ucsd.edu/fireworks/%d/",fireworks::supportedDataFormatsVersion()[0] ),">http://uaf-2.t2.ucsd.edu/fireworks/ </a><BR>",
      "<BR>"
      "<b>You also may load files with Choose Prefix </b><BR>"
      "</BODY></HTML> ",
      0
   };

   m_webFile->SetBaseUri("");
   for (int i=0; s_noBrowserMessage[i]; i++) {
      m_webFile->ParseText((char *)s_noBrowserMessage[i]);
   }

}
//______________________________________________________________________________

std::string 
CmsShowSearchFiles::chooseFileFromURL()
{
   DontCallClose();
   Connect("CloseWindow()","CmsShowSearchFiles",this,"UnmapWindow()");
   m_openCalled = false;
   MapWindow();
   gClient->WaitForUnmap(this);

   if(!m_openCalled) {
      return std::string();
   }
   return m_file->GetText();
}
