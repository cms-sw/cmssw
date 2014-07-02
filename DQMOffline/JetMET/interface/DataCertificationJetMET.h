#ifndef DATACERTIFICATIONJETMET_H
#define DATACERTIFICATIONJETMET_H

// author: Kenichi Hatakeyama (Rockefeller U.)

// system include files
#include <memory>
#include <stdio.h>
#include <math.h>
#include <sstream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "DQMServices/Core/interface/DQMEDHarvester.h"
//
// class decleration
//

class DataCertificationJetMET : public DQMEDHarvester {
   public:
      explicit DataCertificationJetMET(const edm::ParameterSet&);
      ~DataCertificationJetMET();

   private:
      virtual void dqmEndJob(DQMStore::IBooker &, DQMStore::IGetter &) ;

      MonitorElement*  reportSummary;
      MonitorElement*  CertificationSummary;
      MonitorElement*  reportSummaryMap;
      MonitorElement*  CertificationSummaryMap;

   // ----------member data ---------------------------

   edm::ParameterSet conf_;
   edm::Service<TFileService> fs_;
   int verbose_;
   bool InMemory_;
   bool isData;
   std::string metFolder;
   std::string jetAlgo;

   std::string folderName;

   bool caloJetMeanTest;
   bool caloJetKSTest;
   bool pfJetMeanTest;
   bool pfJetKSTest;
   bool jptJetMeanTest;
   bool jptJetKSTest;
   bool caloMETMeanTest;
   bool caloMETKSTest;
   bool pfMETMeanTest;
   bool pfMETKSTest;
   bool tcMETMeanTest;
   bool tcMETKSTest;

   bool jetTests[5][2];  //one for each type of jet certification/test type
   bool metTests[5][2];  //one for each type of met certification/test type

};

#endif
