#ifndef RPCMon_SS_Dbx_Global_H
#define RPCMon_SS_Dbx_Global_H

#include <string>

#include <FWCore/Framework/interface/Event.h>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include <FWCore/Framework/interface/EDAnalyzer.h>
#include <FWCore/Framework/interface/ESHandle.h>
#include <FWCore/Framework/interface/MakerMacros.h>
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DQMServices/Core/interface/DQMStore.h"

class RPCMon_SS_Dbx_Global : public edm::EDAnalyzer {

   public:
      explicit RPCMon_SS_Dbx_Global(const edm::ParameterSet&);
      ~RPCMon_SS_Dbx_Global();

   private:
      virtual void beginJob();
      void beginRun(const edm::Run& , const edm::EventSetup& );
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob();

      std::string  globalFolder_, digiLabel_;
      bool  saveRootFile_;
      std::string rootFileName_;
 DQMStore* dbe_;
 int   numberOfRings_;
      edm::InputTag rpcDigiCollectionTag_;
};

#endif
