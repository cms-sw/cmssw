#include "FWCore/Services/src/InitRootHandlers.h"

#include "DataFormats/Common/interface/RefCoreStreamer.h"
#include "DataFormats/Streamer/interface/StreamedProductStreamer.h"
#include "FWCore/MessageLogger/interface/ELseverityLevel.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/PluginManager/interface/PluginCapabilities.h"
#include "FWCore/RootAutoLibraryLoader/interface/RootAutoLibraryLoader.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/DictionaryTools.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/TypeWithDict.h"
#include "FWCore/Utilities/interface/UnixSignalHandlers.h"

#include <sstream>
#include <string.h>

#include "Cintex/Cintex.h"
#include "G__ci.h"
#include "TROOT.h"
#include "TError.h"
#include "TFile.h"
#include "TH1.h"
#include "TSystem.h"
#include "TUnixSystem.h"
#include "TTree.h"
#include "TVirtualStreamerInfo.h"

#include "TThread.h"
#include "TClassTable.h"
#include "Reflex/Type.h"


namespace {
  enum class SeverityLevel {
    kInfo,
    kWarning,
    kError,
    kSysError,
    kFatal
  };
  
  static thread_local bool s_ignoreWarnings = false;

  static bool s_ignoreEverything = false;

  void RootErrorHandlerImpl(int level, char const* location, char const* message) {

  bool die = false;

  // Translate ROOT severity level to MessageLogger severity level

    SeverityLevel el_severity = SeverityLevel::kInfo;

    if (level >= kFatal) {
      el_severity = SeverityLevel::kFatal;
    } else if (level >= kSysError) {
      el_severity = SeverityLevel::kSysError;
    } else if (level >= kError) {
      el_severity = SeverityLevel::kError;
    } else if (level >= kWarning) {
      el_severity = s_ignoreWarnings ? SeverityLevel::kInfo : SeverityLevel::kWarning;
    }

    if(s_ignoreEverything) {
      el_severity = SeverityLevel::kInfo;
    }

  // Adapt C-strings to std::strings
  // Arrange to report the error location as furnished by Root

    std::string el_location = "@SUB=?";
    if (location != 0) el_location = std::string("@SUB=")+std::string(location);

    std::string el_message  = "?";
    if (message != 0) el_message  = message;

  // Try to create a meaningful id string using knowledge of ROOT error messages
  //
  // id ==     "ROOT-ClassName" where ClassName is the affected class
  //      else "ROOT/ClassName" where ClassName is the error-declaring class
  //      else "ROOT"

    std::string el_identifier = "ROOT";

    std::string precursor("class ");
    size_t index1 = el_message.find(precursor);
    if (index1 != std::string::npos) {
      size_t index2 = index1 + precursor.length();
      size_t index3 = el_message.find_first_of(" :", index2);
      if (index3 != std::string::npos) {
        size_t substrlen = index3-index2;
        el_identifier += "-";
        el_identifier += el_message.substr(index2,substrlen);
      }
    } else {
      index1 = el_location.find("::");
      if (index1 != std::string::npos) {
        el_identifier += "/";
        el_identifier += el_location.substr(0, index1);
      }
    }

  // Intercept some messages and upgrade the severity

      if ((el_location.find("TBranchElement::Fill") != std::string::npos)
       && (el_message.find("fill branch") != std::string::npos)
       && (el_message.find("address") != std::string::npos)
       && (el_message.find("not set") != std::string::npos)) {
        el_severity = SeverityLevel::kFatal;
      }

      if ((el_message.find("Tree branches") != std::string::npos)
       && (el_message.find("different numbers of entries") != std::string::npos)) {
        el_severity = SeverityLevel::kFatal;
      }


  // Intercept some messages and downgrade the severity

      if ((el_message.find("no dictionary for class") != std::string::npos) ||
          (el_message.find("already in TClassTable") != std::string::npos) ||
          (el_message.find("matrix not positive definite") != std::string::npos) ||
          (el_message.find("not a TStreamerInfo object") != std::string::npos) ||
          (el_location.find("Fit") != std::string::npos) ||
          (el_location.find("TDecompChol::Solve") != std::string::npos) ||
          (el_location.find("THistPainter::PaintInit") != std::string::npos) ||
          (el_location.find("TUnixSystem::SetDisplay") != std::string::npos) ||
          (el_location.find("TGClient::GetFontByName") != std::string::npos) ||
	  (el_message.find("nbins is <=0 - set to nbins = 1") != std::string::npos) ||
          (level < kError and
           (el_location.find("CINTTypedefBuilder::Setup")!= std::string::npos) and
           (el_message.find("possible entries are in use!") != std::string::npos))) {
        el_severity = SeverityLevel::kInfo;
      }

    if (el_severity == SeverityLevel::kInfo) {
      // Don't throw if the message is just informational.
      die = false;
    } else {
      die = true;
    }

  // Feed the message to the MessageLogger and let it choose to suppress or not.

  // Root has declared a fatal error.  Throw an EDMException unless the
  // message corresponds to a pending signal. In that case, do not throw
  // but let the OS deal with the signal in the usual way.
    if (die && (el_location != std::string("@SUB=TUnixSystem::DispatchSignals"))) {
       std::ostringstream sstr;
       sstr << "Fatal Root Error: " << el_location << "\n" << el_message << '\n';
       edm::Exception except(edm::errors::FatalRootError, sstr.str());
       except.addAdditionalInfo(except.message());
       except.clearMessage();
       throw except;
     
    }

    // Typically, we get here only for informational messages,
    // but we leave the other code in just in case we change
    // the criteria for throwing.
    if (el_severity == SeverityLevel::kFatal) {
      edm::LogError("Root_Fatal") << el_location << el_message;
    } else if (el_severity == SeverityLevel::kSysError) {
      edm::LogError("Root_Severe") << el_location << el_message;
    } else if (el_severity == SeverityLevel::kError) {
      edm::LogError("Root_Error") << el_location << el_message;
    } else if (el_severity == SeverityLevel::kWarning) {
      edm::LogWarning("Root_Warning") << el_location << el_message ;
    } else if (el_severity == SeverityLevel::kInfo) {
      edm::LogInfo("Root_Information") << el_location << el_message ;
    }
  }

  void RootErrorHandler(int level, bool, char const* location, char const* message) {
    RootErrorHandlerImpl(level, location, message);
  }

  extern "C" {
    void sig_dostack_then_abort(int sig,siginfo_t*,void*) {
      if (gSystem) {
        const char* signalname = "unknown";
        switch (sig) {
          case SIGBUS:
            signalname = "bus error";
            break;
          case SIGSEGV:
            signalname = "segmentation violation";
            break;
          case SIGILL:
            signalname = "illegal instruction"; 
          default:
            break;
        }
        edm::LogError("FatalSystemSignal")<<"A fatal system signal has occurred: "<<signalname;
        std::cerr<< "\n\nA fatal system signal has occurred: "<<signalname<<"\n"
                 <<"The following is the call stack containing the origin of the signal.\n"
                 <<"NOTE:The first few functions on the stack are artifacts of processing the signal and can be ignored\n\n";
        
        gSystem->StackTrace();
        std::cerr<<"\nA fatal system signal has occurred: "<<signalname<<"\n";
      }
      ::abort();
    }
    
    void sig_abort(int sig, siginfo_t*, void*) {
      ::abort();
    }
  }
}  // end of unnamed namespace

namespace edm {
  namespace service {
    InitRootHandlers::InitRootHandlers (ParameterSet const& pset)
      : RootHandlers(),
        unloadSigHandler_(pset.getUntrackedParameter<bool> ("UnloadRootSigHandler")),
        resetErrHandler_(pset.getUntrackedParameter<bool> ("ResetRootErrHandler")),
        loadAllDictionaries_(pset.getUntrackedParameter<bool>("LoadAllDictionaries")),
        autoLibraryLoader_(loadAllDictionaries_ or pset.getUntrackedParameter<bool> ("AutoLibraryLoader"))
    {
      
      if(unloadSigHandler_) {
      // Deactivate all the Root signal handlers and restore the system defaults
        gSystem->ResetSignal(kSigChild);
        gSystem->ResetSignal(kSigBus);
        gSystem->ResetSignal(kSigSegmentationViolation);
        gSystem->ResetSignal(kSigIllegalInstruction);
        gSystem->ResetSignal(kSigSystem);
        gSystem->ResetSignal(kSigPipe);
        gSystem->ResetSignal(kSigAlarm);
        gSystem->ResetSignal(kSigUrgent);
        gSystem->ResetSignal(kSigFloatingException);
        gSystem->ResetSignal(kSigWindowChanged);
      } else if(pset.getUntrackedParameter<bool>("AbortOnSignal")){
        //NOTE: ROOT can also be told to abort on these kinds of problems BUT
        // it requires an TApplication to be instantiated which causes problems
        gSystem->ResetSignal(kSigBus);
        gSystem->ResetSignal(kSigSegmentationViolation);
        gSystem->ResetSignal(kSigIllegalInstruction);
        installCustomHandler(SIGBUS,sig_dostack_then_abort);
        sigBusHandler_ = std::shared_ptr<const void>(nullptr,[](void*) {
          installCustomHandler(SIGBUS,sig_abort);
        });
        installCustomHandler(SIGSEGV,sig_dostack_then_abort);
        sigSegvHandler_ = std::shared_ptr<const void>(nullptr,[](void*) {
          installCustomHandler(SIGSEGV,sig_abort);
        });
        installCustomHandler(SIGILL,sig_dostack_then_abort);
        sigIllHandler_ = std::shared_ptr<const void>(nullptr,[](void*) {
          installCustomHandler(SIGILL,sig_abort);
        });
      }

      if(resetErrHandler_) {

      // Replace the Root error handler with one that uses the MessageLogger
        SetErrorHandler(RootErrorHandler);
      }

      // Enable automatic Root library loading.
      if(autoLibraryLoader_) {
        RootAutoLibraryLoader::enable();
        if(loadAllDictionaries_) {
          RootAutoLibraryLoader::loadAll();
        }
      }

      // Enable Cintex.
      ROOT::Cintex::Cintex::Enable();

      // Set ROOT parameters.
      TTree::SetMaxTreeSize(kMaxLong64);
      TH1::AddDirectory(kFALSE);
      G__SetCatchException(0);

      // Set custom streamers
      setRefCoreStreamer();
      setStreamedProductStreamer();

      // Load the library containing dictionaries for std:: classes, if not already loaded.
      if (!TypeWithDict(typeid(std::vector<std::vector<unsigned int> >)).hasDictionary()) {
         edmplugin::PluginCapabilities::get()->load(dictionaryPlugInPrefix() + "std::vector<std::vector<unsigned int> >");
      }

      int debugLevel = pset.getUntrackedParameter<int>("DebugLevel");
      if(debugLevel >0) {
	gDebug = debugLevel;
      }
    }

    InitRootHandlers::~InitRootHandlers () {
      // close all open ROOT files
      // We get a new iterator each time,
      // because closing a file can invalidate the iterator
      while(gROOT->GetListOfFiles()->GetSize()) {
        TIter iter(gROOT->GetListOfFiles());
        TFile* f = dynamic_cast<TFile*>(iter.Next());
        if(f) f->Close();
      }
    }
    
    void InitRootHandlers::willBeUsingThreads() {
      //Tell Root we want to be multi-threaded
      TThread::Initialize();
      //When threading, also have to keep ROOT from logging all TObjects into a list
      TObject::SetObjectStat(false);
      
      //Have to avoid having Streamers modify themselves after they have been used
      TVirtualStreamerInfo::Optimize(false);

      if(not this->autoLibraryLoader_) {
        RootAutoLibraryLoader::enable();
      }
      if(not this->loadAllDictionaries_) {
        RootAutoLibraryLoader::loadAll();
      }

      //Must force all streamers into existence
      std::vector<const char*> knownClassNames;
      int const kNClasses = gClassTable->Classes();
      knownClassNames.reserve(kNClasses);
      for(int i = 0; i<kNClasses; ++i) {
        auto const name = gClassTable->At(i);
        knownClassNames.push_back(name);
      }

      const std::string kFalse("false");
      //Wrappers are required to work
      for(auto name : knownClassNames) {
        if(strncmp(name,"edm::Wrapper<",13)==0) {
          TClass* c=TClass::GetClass(name);
          //can't use the name to lookup ReflexType becuase ROOT removes 'std::'
          assert(c->GetTypeInfo());
          auto t = Reflex::Type::ByTypeInfo(*(c->GetTypeInfo()));
          Reflex::PropertyList wp = t.Properties();
          if(wp.HasProperty("persistent") and wp.PropertyAsString("persistent") == kFalse) continue;
          //std::cout <<"start GetStreamerInfo for "<<name<<std::endl;
          c->GetStreamerInfo();
        }
      }

      const std::vector<const char*> kKnownNames = {"TH2F","TArrayF"};
      for(auto name: kKnownNames) {
        TClass* c=TClass::GetClass(name);
        assert(c);
        c->GetStreamerInfo();
      }

      s_ignoreEverything=true;
      std::shared_ptr<void*> guard(nullptr,[](void*) { s_ignoreEverything=false;});

      const Reflex::Type kDefaultType;
      for(auto name : knownClassNames) {
        if(strncmp(name,"edm::Wrapper<",13)==0) continue;
        //std::cout <<"look at class "<<name<<std::endl;
        TClass* c=TClass::GetClass(name);
        if(c == nullptr) continue;
        auto id = c->GetTypeInfo();
        if( nullptr == id) continue;
        auto t = Reflex::Type::ByTypeInfo(*id);
        if(t == kDefaultType) continue;
        Reflex::PropertyList wp = t.Properties();
        if(wp.HasProperty("persistent") and wp.PropertyAsString("persistent") == kFalse) continue;
        
        if( (not (c->Property() & kIsAbstract) ) and
           c->HasDefaultConstructor() and
           (0 != c->GetClassVersion())  and
           ( 0 == (gClassTable->GetPragmaBits(name) & TClassTable::kNoStreamer) )
           ) {
          //std::cout <<"start GetStreamerInfo for "<<name<<" "<<c->GetClassVersion()<<std::endl;
          c->GetStreamerInfo();
        }
      }
    }
    
    void InitRootHandlers::initializeThisThreadForUse() {
      static thread_local TThread guard;
    }

    void InitRootHandlers::fillDescriptions(ConfigurationDescriptions& descriptions) {
      ParameterSetDescription desc;
      desc.setComment("Centralized interface to ROOT.");
      desc.addUntracked<bool>("UnloadRootSigHandler", false)
          ->setComment("If True, signals are handled by this service, rather than by ROOT.");
      desc.addUntracked<bool>("ResetRootErrHandler", true)
          ->setComment("If True, ROOT messages (e.g. errors, warnings) are handled by this service, rather than by ROOT.");
      desc.addUntracked<bool>("AutoLibraryLoader", true)
          ->setComment("If True, enables automatic loading of data dictionaries.");
      desc.addUntracked<bool>("LoadAllDictionaries",false)
          ->setComment("If True, loads all ROOT dictionaries.");
      desc.addUntracked<bool>("AbortOnSignal",true)
      ->setComment("If True, do an abort when a signal occurs that causes a crash. If False, ROOT will do an exit which attempts to do a clean shutdown.");
      desc.addUntracked<int>("DebugLevel",0)
 	  ->setComment("Sets ROOT's gDebug value.");
      descriptions.add("InitRootHandlers", desc);
    }

    void
    InitRootHandlers::enableWarnings_() {
      s_ignoreWarnings =false;
    }

    void
    InitRootHandlers::ignoreWarnings_() {
      s_ignoreWarnings = true;
    }

  }  // end of namespace service
}  // end of namespace edm
