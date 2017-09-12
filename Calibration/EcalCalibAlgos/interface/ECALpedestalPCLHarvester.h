// -*- C++ -*-
//
// Package:    Calibration/EcalCalibAlgos
// Class:      ECALpedestalPCLHarvester
// 
/**\class ECALpedestalPCLHarvester ECALpedestalPCLHarvester.cc 

 Description: Fill DQM histograms with pedestals. Intended to be used on laser data from the TestEnablesEcalHcal dataset

 
*/
//
// Original Author:  Stefano Argiro
//         Created:  Wed, 22 Mar 2017 14:46:48 GMT
//
//


#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/DQMEDHarvester.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "CondFormats/EcalObjects/interface/EcalPedestals.h"
#include "CondFormats/EcalObjects/interface/EcalChannelStatus.h"

class ECALpedestalPCLHarvester : public  DQMEDHarvester {
   public:
      explicit ECALpedestalPCLHarvester(const edm::ParameterSet& ps);
      void endRun(edm::Run const& run, edm::EventSetup const & isetup) override;
      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);


   private:
     
      void dqmEndJob(DQMStore::IBooker& ibooker_, DQMStore::IGetter& igetter_) override ;

      void  dqmPlots(const EcalPedestals& newpeds, DQMStore::IBooker& ibooker);

      const EcalPedestals * currentPedestals_;
      const EcalPedestals * g6g1Pedestals_;
      const EcalChannelStatus * channelStatus_;
      bool  checkStatusCode(const DetId& id);
      bool  isGood(const DetId& id);

      bool  checkVariation(const EcalPedestalsMap& oldPedestals, const EcalPedestalsMap& newPedestals);
      std::vector<int> chStatusToExclude_;
      int minEntries_;


      int entriesEB_[EBDetId::kSizeForDenseIndexing];
      int entriesEE_[EEDetId::kSizeForDenseIndexing];
      bool   checkAnomalies_ ;    // whether or not to avoid creating sqlite file in case of many changed pedestals
      double nSigma_;             // threshold in sigmas to define a pedestal as changed
      double thresholdAnomalies_; // threshold (fraction of changed pedestals) to avoid creation of sqlite file 
      std::string dqmDir_;        // DQM directory where histograms are stored
      std::string labelG6G1_;    // DB label from which pedestals for G6 and G1 are to be copied
      float threshDiffEB_;       // if the new pedestals differs more than this from old, keep old
      float threshDiffEE_;         // same as above for EE. Stray channel protection

};
