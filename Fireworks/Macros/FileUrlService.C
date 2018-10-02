// Author: Matevz Tadel, September 2010
//
// Server / client classes for serving a list of files.
// Push and pull modes are supported.
// Works OK, needs to be incorporated into Fireworks (when needed).

#if defined(__CINT__) && !defined(__MAKECINT__)
{
   Info("FileUrlService.C", "Has to be run in compiled mode ... doing this for you.");
   gSystem->CompileMacro("FileUrlService.C");
   FileUrlService();
}
#else

#include "TError.h"
#include "TList.h"

#include "TMessage.h"
#include "TMonitor.h"
#include "TSocket.h"
#include "TServerSocket.h"

#include "TThread.h"

#include <cassert>
#include <memory>
#include <sstream>
#include <stdexcept>

#include <list>
#include <map>


//==============================================================================
// Utilities
//==============================================================================

namespace
{
   TString RecvLine(TSocket* socket)
   {
      std::vector<char> buf;
      buf.reserve(256);

      char c = 255;
      while (c != 0)
      {
         Int_t ret = socket->RecvRaw(&c, 1);
         if (ret < 0)
         {
            std::ostringstream err;
            err << "Reading line from socket failed, error code " << ret << ".";
            throw std::runtime_error(err.str());
         }
         if (c == 13)
            continue;
         if (c == 10)
            c = 0;
         buf.push_back(c);
      }

      return TString(&buf[0]);
   }

   void SendLine(TSocket* socket, const TString& msg)
   {
      // Send msg ... new-line character is added here.

      static const char eol(10);

      socket->SendRaw(msg.Data(), msg.Length());
      socket->SendRaw(&eol,       1);
   }
}


//==============================================================================
// FileUrlRequest
//==============================================================================

class FileUrlRequest : public TObject
{
public:
   enum EMode { kGoodBye=-1, kPullNext, kPullLast, kServerPush };

   EMode   fMode;
   Int_t   fFileId;
   TString fFileUrl;
   Bool_t  fFileOK;

   FileUrlRequest(EMode mode=kPullNext) :
      TObject(),
      fMode(mode),
      fFileId(-1),
      fFileOK(kFALSE) {}

   void SetFile(Int_t id, const TString& f)
   {
      fFileId  = id;
      fFileUrl = f;
      fFileOK  = kTRUE;
   }

   void ResetFile()
   {
      fFileId  = -1;
      fFileUrl = "";
      fFileOK  = kFALSE;
   }

   ClassDef(FileUrlRequest, 1);
};


//==============================================================================
// FileUrlBase
//==============================================================================


class FileUrlBase
{
protected:
   void      SendMessage   (TSocket& socket, TMessage& msg, const TString& eh);
   TMessage* ReceiveMessage(TSocket& socket, UInt_t type, const TString& eh);

   void      SendString    (TSocket& socket, TString& str, const TString& eh);
   void      ReceiveString (TSocket& socket, TString& str, const TString& eh);

   void      SendRequest   (TSocket& socket, FileUrlRequest& request, const TString& eh);
   void      ReceiveRequest(TSocket& socket, FileUrlRequest& request, const TString& eh);

   void      SendGoodBye   (TSocket& socket, const TString& eh);

   static const UInt_t gkUrlRequestMT; // Message type
   static const UInt_t gkStringMT;     // Message type

public:
   FileUrlBase() {}
   virtual ~FileUrlBase() {}

   ClassDef(FileUrlBase, 0);
};

//==============================================================================

const UInt_t FileUrlBase::gkUrlRequestMT = 666;
const UInt_t FileUrlBase::gkStringMT     = 667;

//------------------------------------------------------------------------------

void FileUrlBase::SendMessage(TSocket& socket, TMessage& msg, const TString& eh)
{
   Int_t len = socket.Send(msg);
   if (len <= 0)
   {
      std::ostringstream err;
      err << eh  << " Failed sending message, ret = "
          << len << ".";
      throw std::runtime_error(err.str());
   }
}

TMessage* FileUrlBase::ReceiveMessage(TSocket& socket, UInt_t type, const TString& eh)
{
   TMessage *msg = 0;
   Int_t len = socket.Recv(msg);
   if (len <= 0)
   {
      std::ostringstream err;
      err << eh  << " Failed receiving message, ret = "
          << len << ".";
      throw std::runtime_error(err.str());
   }
   if (type != 0 && msg->What() != type)
   {
      std::ostringstream err;
      err << eh  << " Wrong message type received. Expected " << type
          << ", got " << msg->What() << ".";
      throw std::runtime_error(err.str());
   }
   return msg;
}

//------------------------------------------------------------------------------

void FileUrlBase::SendString(TSocket& socket, TString& str, const TString& eh)
{
   TMessage msg(gkStringMT);
   str.Streamer(msg);
   SendMessage(socket, msg, eh);
}

void FileUrlBase::ReceiveString(TSocket& socket, TString& str, const TString& eh)
{
   std::unique_ptr<TMessage> msg(ReceiveMessage(socket, gkStringMT, eh));
   str.Streamer(*msg);
}

//------------------------------------------------------------------------------

void FileUrlBase::SendRequest(TSocket& socket, FileUrlRequest& request, const TString& eh)
{
   TMessage msg(gkUrlRequestMT);
   request.Streamer(msg);
   SendMessage(socket, msg, eh);
}

void FileUrlBase::ReceiveRequest(TSocket& socket, FileUrlRequest& request, const TString& eh)
{
   std::unique_ptr<TMessage> msg(ReceiveMessage(socket, gkUrlRequestMT, eh));
   request.Streamer(*msg);
}

//------------------------------------------------------------------------------

void FileUrlBase::SendGoodBye(TSocket& socket, const TString& eh)
{
   static FileUrlRequest sByeRequest(FileUrlRequest::kGoodBye);

   SendRequest(socket, sByeRequest, eh);
}


//==============================================================================
// FileUrlServer
//==============================================================================

class FileUrlServer : public FileUrlBase
{
protected:
   typedef std::map<Int_t, TString>            mIdToFile_t;
   typedef mIdToFile_t::iterator               mIdToFile_i; 

   typedef std::map<TSocket*, FileUrlRequest>  mSockToReq_t;
   typedef mSockToReq_t::iterator              mSockToReq_i;

   // ----------------------------------------------------------------

   TString                   fName;
   TString                   fUrlPrefix;

   Int_t                     fFileNMax;
   Int_t                     fFileLastId;
   mIdToFile_t               fFileMap;
   mSockToReq_t              fPushClientMap;

   TMonitor                 *fMonitor;
   TServerSocket            *fServerSocket;
   TThread                  *fServerThread;
   TMutex                   *fMutex;

   TString                   fNewFileFile;
   TTimer                   *fNewFileFileCheckTimer;

   // ----------------------------------------------------------------

   void AcceptConnection();
   void CloseConnection(TSocket *cs);

   void MessageFrom(TSocket* cs);

   // ----------------------------------------------------------------

   static void* RunServer(FileUrlServer* srv);

public:
   FileUrlServer(const char* name, Int_t port);
   virtual ~FileUrlServer();

   // --------------------------------

   TString GetUrlPrefix() const            { return fUrlPrefix; }
   void    SetUrlPrefix(const TString& up) { fUrlPrefix = up;   }

   Int_t   GetFileNMax() const             { return fFileNMax; }
   void    SetFileNMax(Int_t n)            { fFileNMax = n; RemoveExtraFiles(); }

   Int_t   GetFileLastId() const           { return fFileLastId; }

   // --------------------------------

   void AddFile(const TString& file);
   void RemoveFile(const TString& file);
   void RemoveExtraFiles();

   // --------------------------------

   // Pain: how to determine file is complete and closed?
   // Use the LastFile thing from afs
   void BeginMonitoringNewFileFile(const TString& file, Int_t sec=1);
   void CheckNewFileFile();
   void EndMonitoringNewFileFile();

   ClassDef(FileUrlServer, 0);
};

//==============================================================================

FileUrlServer::FileUrlServer(const char* name, Int_t port) :
   fName(name),
   fFileNMax(0),
   fFileLastId(0),
   fMonitor(0), fServerSocket(0), fServerThread(0), fMutex(0),
   fNewFileFileCheckTimer(0)
{
   fServerSocket = new TServerSocket(port, kTRUE, 4, 4096);

   if (!fServerSocket->IsValid())
   {
      std::ostringstream err;
      err << "Creation of server socket failed ";
      switch (fServerSocket->GetErrorCode())
      {
         case -1:
            err << "in low level socket().";
            break;
         case -2:
            err << "in low level bind().";
            break;
         case -3:
            err << "in low level listen().";
            break;
         default:
            err << "with unknown error " << fServerSocket->GetErrorCode() << ".";
            break;
      }
      throw std::runtime_error(err.str());
   }

   fMonitor = new TMonitor(kFALSE);
   fMonitor->Add(fServerSocket);

   fMutex = new TMutex(kTRUE);
   fServerThread = new TThread((TThread::VoidRtnFunc_t) RunServer);
   fServerThread->Run(this);
}

FileUrlServer::~FileUrlServer()
{
   if (fNewFileFileCheckTimer)
   {
      EndMonitoringNewFileFile();
   }

   fServerThread->Kill();
   fServerThread->Join();
   delete fServerThread;

   {
      std::unique_ptr<TList> socks(fMonitor->GetListOfActives());
      while ( ! socks->IsEmpty())
      {
         TObject *obj = socks->First();
         socks->RemoveFirst();
         delete obj;
      }
   }
   delete fMonitor;

   delete fMutex;
}

//------------------------------------------------------------------------------

void FileUrlServer::AddFile(const TString& file)
{
   static const TString _eh("FileUrlServer::AddFile");

   TLockGuard _lck(fMutex);

   ++fFileLastId;
   fFileMap[fFileLastId] = file;

   for (mSockToReq_i i = fPushClientMap.begin(); i != fPushClientMap.end(); ++i)
   {
      i->second.SetFile(fFileLastId, fUrlPrefix + file);
      SendRequest(*i->first, i->second, _eh);
   }

   RemoveExtraFiles();
}

void FileUrlServer::RemoveFile(const TString& file)
{
   TLockGuard _lck(fMutex);

   for (mIdToFile_i i = fFileMap.begin(); i != fFileMap.end(); ++i)
   {
      if (i->second == file)
      {
         fFileMap.erase(i);
         break;
      }
   }
}

void FileUrlServer::RemoveExtraFiles()
{
   TLockGuard _lck(fMutex);

   if (fFileNMax > 0)
   {
      while ((Int_t) fFileMap.size() > fFileNMax)
      {
         fFileMap.erase(fFileMap.begin());
      }
   }
}

//------------------------------------------------------------------------------

void FileUrlServer::AcceptConnection()
{
   static const TString _eh("FileUrlServer::AcceptConnection");

   TLockGuard _lck(fMutex);

   TSocket *cs = fServerSocket->Accept();

   Info(_eh, "Connection from %s:%d.",
        cs->GetInetAddress().GetHostName(), fServerSocket->GetLocalPort());

   TString hello;
   hello.Form("Hello! I am FileUrlServer serving '%s'.", fName.Data());
   SendString(*cs, hello, _eh);

   fMonitor->Add(cs);
}

void FileUrlServer::CloseConnection(TSocket *cs)
{
   fMonitor->Remove(cs);
   fPushClientMap.erase(cs);
   delete cs;
}

//------------------------------------------------------------------------------

void FileUrlServer::MessageFrom(TSocket* cs)
{
   static const TString _eh("FileUrlServer::MessageFrom");

   TLockGuard _lck(fMutex);

   try
   {
      FileUrlRequest req;

      ReceiveRequest(*cs, req, _eh);

      switch (req.fMode)
      {
         case FileUrlRequest::kGoodBye:
         {
            CloseConnection(cs);
            break;
         }
         case FileUrlRequest::kPullNext:
         {
            req.fFileOK = kFALSE;
            mIdToFile_i i = fFileMap.upper_bound(req.fFileId);
            if (i != fFileMap.end())
            {
               req.SetFile(i->first, fUrlPrefix + i->second);
            }
            SendRequest(*cs, req, _eh);
            break;
         }
         case FileUrlRequest::kPullLast:
         {
            req.fFileOK = kFALSE;
            if ( ! fFileMap.empty())
            {
               mIdToFile_i i = --fFileMap.end();
               if (i->first > req.fFileId)
               {
                  req.SetFile(i->first, fUrlPrefix + i->second);
               }
            }
            SendRequest(*cs, req, _eh);
            break;
         }
         case FileUrlRequest::kServerPush:
         {
            fPushClientMap[cs] = req;
            break;
         }
      }
   }
   catch (std::runtime_error& err)
   {
      Error(_eh, err.what());
      CloseConnection(cs);
   }
}

//------------------------------------------------------------------------------

void* FileUrlServer::RunServer(FileUrlServer* srv)
{
   TThread::SetCancelDeferred();

   TList ready;

   while (kTRUE)
   {
      TThread::SetCancelOn();
      srv->fMonitor->Select(&ready, 0, 1000000);
      TThread::SetCancelOff();

      while ( ! ready.IsEmpty())
      {
         TSocket *sock = (TSocket*) ready.First();
         ready.RemoveFirst();

         if (sock == srv->fServerSocket)
         {
            srv->AcceptConnection();
         }
         else
         {
            srv->MessageFrom(sock);
         }
      }
   }

   return 0;
}

//------------------------------------------------------------------------------

void FileUrlServer::BeginMonitoringNewFileFile(const TString& file, Int_t sec)
{
   static const TString _eh("FileUrlServer::BeginMonitoringNewFileFile");

   if (fNewFileFileCheckTimer != 0)
   {
      Error(_eh, "File monitoring already in progress, end it first.");
      return;
   }

   fNewFileFile = file;

   fNewFileFileCheckTimer = new TTimer(1000l*sec);
   fNewFileFileCheckTimer->Connect("Timeout()", "FileUrlServer", this, "CheckNewFileFile()");

   gTQSender = fNewFileFileCheckTimer;
   CheckNewFileFile();
   gTQSender = 0;
}

void FileUrlServer::CheckNewFileFile()
{
   // Should be private, but is a slot.

   static const TString _eh("FileUrlServer::CheckNewFileFile");

   if (fNewFileFileCheckTimer == 0 || gTQSender != fNewFileFileCheckTimer)
   {
      Error(_eh, "Timer is not running or method called directly.");
      return;
   }

   fNewFileFileCheckTimer->TurnOff();

   FILE *fp = fopen(fNewFileFile, "r");
   if (fp == 0)
   {
      Warning(_eh, "Could not open new-file file for reading.");
      return;
   }
   TString line;
   line.Gets(fp, kTRUE);
   fclose(fp);
   if ( ! line.IsNull())
   {
      TLockGuard _lck(fMutex);

      if (!fFileMap.empty())
      {
         mIdToFile_i i = --fFileMap.end();
         if (i->second != line)
         {
            printf("Previous: '%s'\n", i->second.Data());
            printf("New:      '%s'\n", line.Data());
            AddFile(line);
         }
      }
   }
   fNewFileFileCheckTimer->Reset();
   fNewFileFileCheckTimer->TurnOn();
}

void FileUrlServer::EndMonitoringNewFileFile()
{
   static const TString _eh("FileUrlServer::EndMonitoringNewFileFile");

   if (fNewFileFileCheckTimer == 0)
   {
      Error(_eh, "No file monitoring in progress.");
      return;
   }

   fNewFileFileCheckTimer->TurnOff();
   delete fNewFileFileCheckTimer;
   fNewFileFileCheckTimer = 0;
}


//==============================================================================
// FileUrlClient
//==============================================================================

class FileUrlClient : public TQObject,
                      public FileUrlBase 
{
protected:
   TString        fHost;
   Int_t          fPort;

   FileUrlRequest fRequest;

   TSocket       *fSocket;     // Non-zero when in server-push mode.
   TMonitor      *fMonitor; //

   TSocket*  OpenSocket(const TString& eh);
   void      CleanupServerPushMode();

public:
   FileUrlClient(const char* host, Int_t port) :
      fHost(host),
      fPort(port),
      fSocket(0),
      fMonitor(0)
   {}

   virtual ~FileUrlClient()
   {
      if (fSocket) EndServerPushMode();
   }

   // --------------------------------

   Bool_t  HasCurrentFile() const { return fRequest.fFileOK;  }
   TString GetCurrentFile() const { return fRequest.fFileUrl; }

   // --------------------------------

   void GetNextFile(Bool_t loop=kFALSE);
   void GetLastFile();

   void BeginServerPushMode();
   void ServerPushAction();
   void NewFileArrived(); // *SIGNAL*
   void EndServerPushMode();

   ClassDef(FileUrlClient, 0);
};

//==============================================================================

TSocket* FileUrlClient::OpenSocket(const TString& eh)
{
   std::unique_ptr<TSocket> s(new TSocket(fHost, fPort));

   if (!s->IsValid())
   {
      std::ostringstream err;
      err << eh    << " Failed connecting to "
          << fHost << ":" << fPort << ".";
      throw std::runtime_error(err.str());
   }

   TString str;
   ReceiveString(*s, str, eh);
   Info(eh, TString("socket opened, greeting is: ") + str);

   return s.release();
}

//------------------------------------------------------------------------------

void FileUrlClient::GetNextFile(Bool_t loop)
{
   static const TString _eh("FileUrlClient::GetNextFile");

   if (fSocket != 0)
   {
      Error(_eh, "Currently in server-push mode.");
      return;
   }

   std::unique_ptr<TSocket> s(OpenSocket(_eh));

   fRequest.fMode = FileUrlRequest::kPullNext;

   SendRequest(*s, fRequest, _eh);
   ReceiveRequest(*s, fRequest, _eh);

   if (!fRequest.fFileOK && loop)
   {
      fRequest.ResetFile();
      SendRequest(*s, fRequest, _eh);
      ReceiveRequest(*s, fRequest, _eh);
   }

   Info(_eh, "loop=%s; ok=%d, id=%d, file=%s",
        loop ? "true" : "false",
        fRequest.fFileOK, fRequest.fFileId, fRequest.fFileUrl.Data());

   SendGoodBye(*s, _eh);
}

void FileUrlClient::GetLastFile()
{
   static const TString _eh("FileUrlClient::GetLastFile");

   if (fSocket != 0)
   {
      Error(_eh, "Currently in server-push mode.");
      return;
   }

   std::unique_ptr<TSocket> s(OpenSocket(_eh));

   fRequest.fMode = FileUrlRequest::kPullLast;

   SendRequest(*s, fRequest, _eh);
   ReceiveRequest(*s, fRequest, _eh);

   Info(_eh, "ok=%d, id=%d, file=%s",
        fRequest.fFileOK, fRequest.fFileId, fRequest.fFileUrl.Data());

   SendGoodBye(*s, _eh);
}

//------------------------------------------------------------------------------

void FileUrlClient::BeginServerPushMode()
{
   static const TString _eh("FileUrlClient::BeginServerPushMode");

   if (fSocket != 0)
   {
      Error(_eh, "Already in server-push mode.");
      return;
   }

   try
   {
      std::unique_ptr<TSocket> s(OpenSocket(_eh));
      fRequest.fMode = FileUrlRequest::kServerPush;
      SendRequest(*s, fRequest, _eh);
      fSocket = s.release();
   }
   catch (std::runtime_error& err)
   {
      Error(_eh, "Sending of request failed, ending server-push mode.");
      EndServerPushMode();
      throw;
   }

   fMonitor = new TMonitor();
   fMonitor->Add(fSocket);
   fMonitor->Connect("Ready(TSocket*)", "FileUrlClient", this, "ServerPushAction()");
}

void FileUrlClient::ServerPushAction()
{
   static const TString _eh("FileUrlClient::ServerPushAction");

   if (fSocket == 0)
   {
      Error(_eh, "Should only get called from TMonitor in server-push mode.");
      return;
   }

   try
   {
      ReceiveRequest(*fSocket, fRequest, _eh);
   }
   catch (std::runtime_error& err)
   {
      Error(_eh, "Receiving of request failed, ending server-push mode.");
      CleanupServerPushMode();
      return;
   }

   Info(_eh, "ok=%d, id=%d, file=%s",
        fRequest.fFileOK, fRequest.fFileId, fRequest.fFileUrl.Data());

   NewFileArrived();
}

void FileUrlClient::NewFileArrived()
{
   Emit("NewFileArrived()");
}

void FileUrlClient::EndServerPushMode()
{
   static const TString _eh("FileUrlClient::EndServerPushMode");

   if (fSocket == 0)
   {
      Error(_eh, "Currently not in server-push mode.");
      return;
   }

   SendGoodBye(*fSocket, _eh);

   CleanupServerPushMode();
}

void FileUrlClient::CleanupServerPushMode()
{
   delete fMonitor; fMonitor = 0;
   delete fSocket;  fSocket  = 0;
}


//==============================================================================
// main() substitute
//==============================================================================

FileUrlServer *g_fu_server = 0;
FileUrlClient *g_fu_client = 0;

void FileUrlService()
{
   printf("FileUrlService loaded ... starting server at port 4444!\n");
   g_fu_server = new FileUrlServer("Testos", 4444);
   g_fu_server->SetUrlPrefix("http://matevz.web.cern.ch/matevz/tmp/");
   g_fu_server->AddFile("EVDISPSM_1284666332001.root");
   g_fu_server->AddFile("EVDISPSM_1284666332002.root");
   g_fu_server->AddFile("EVDISPSM_1284666332003.root");

   g_fu_client = new FileUrlClient("pcalice14", 4444);
   // g_fu_client->GetNextFile();
}


//==============================================================================
// Directory watching
//==============================================================================
//
// This doen't work at all for afs.
// Is OK for local file-systems.

/*

#include <sys/inotify.h>

#include "TSystem.h"

class DirectoryMonitor
{
   Int_t fFd;
   Int_t fWd;

   TFileHandler *fFileHandler;

public:
   DirectoryMonitor(const TString& path="/afs/cern.ch/cms/CAF/CMSCOMM/COMM_GLOBAL/EventDisplay/RootFileTempStorageArea")
   // DirectoryMonitor(const TString& path="/tmp")
   {
      fFd = inotify_init();
      if (fFd < 0)
      {
         perror("inotfiy_init");
         return;
      }

      fWd = inotify_add_watch(fFd, path, IN_CLOSE | IN_MOVED_TO);

      fFileHandler = new TFileHandler(fFd, TFileHandler::kRead);
      fFileHandler->Connect("Notified()", "DirectoryMonitor", this, "StuffReady()");

      gSystem->AddFileHandler(fFileHandler);

      printf("fd %d, wd %d\n", fFd, fWd);
   }

   virtual ~DirectoryMonitor()
   {
      gSystem->RemoveFileHandler(fFileHandler);
      inotify_rm_watch(fFd, fWd);
      close(fFd);
   }

   void StuffReady()
   {
      printf("StuffReady ... do read it!\n");

      char buf[1024];
      ssize_t n = read(fFd, buf, 1024);
      if (n < 0)
      {
         perror("StuffReady -- read failed");
         return;
      }
      printf("  Read %zd bytes.\n", n);

      int i = 0;
      while (i < n)
      {
         struct inotify_event& e = *(struct inotify_event*)(&buf[i]);
         printf("  wd=%d mask=%x cookie=%u len=%u, name=%s\n",
                e.wd, e.mask, e.cookie, e.len, e.len ? e.name : "<none>");

         i += sizeof(struct inotify_event) + e.len;
      }
   }

   ClassDef(DirectoryMonitor, 0);
};

*/

#endif
