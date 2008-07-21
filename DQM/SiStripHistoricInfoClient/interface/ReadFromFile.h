#include <memory>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondFormats/SiStripObjects/interface/SiStripSummary.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"



#include "TROOT.h"
#include "TPad.h"
#include "TSystem.h"
#include "TFile.h"
#include "TDirectory.h"
#include "TKey.h"
#include "TString.h"
#include "TProfile.h"
#include "TH1.h"
#include "TH2.h"
#include "TImage.h"
#include "TPaveText.h"
#include "TLatex.h"
#include "TStyle.h"
#include "TRFIOFile.h"
#include "TImageDump.h"

namespace edm {
    class ParameterSet;
    class Event;
    class EventId;
    class Timestamp;
    class DQMStore;
    class MonitorElement;
}

class ReadFromFile : public edm::EDAnalyzer {
   public:
      explicit ReadFromFile(const edm::ParameterSet&);
      ~ReadFromFile();
      
   private:
 
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endRun(const edm::Run& run , const edm::EventSetup& iSetup);
      void openRequestedFile(); 
      void scanTreeAndFillSummary(std::string top_dir,SiStripSummary* summary,std::string& histoName, std::vector<std::string>& Quantities);
      void writeToDB() const;
      uint32_t getRunNumber() const;
      uint32_t returnDetComponent(std::string histoName);

      DQMStore* dqmStore_;
         
      std::string ROOTFILE_DIR;  
      std::string FILE_NAME;  
      std::string ME_DIR;  
      
      bool doTracks;
      bool doNumberOfClusters;
      bool doClusterWidth;
      bool doClusterCharge;
      bool doClusterNoise;
      bool doClusterStoN;
   
      std::vector<SiStripSummary *> vSummary;
      std::vector<int> vReg;
      
      edm::ParameterSet iConfig_;
};

