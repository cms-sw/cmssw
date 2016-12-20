#ifndef CLUSTERCOUNT_H
#define CLUSTERCOUNT_H
// -*- C++ -*-
//
// Package:    ClusterCount
// Class:      ClusterCount
// 
/**\class ClusterCount ClusterCount.cc mytests/ClusterCount/src/ClusterCount.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Christophe DELAERE
//         Created:  Tue May 27 11:11:05 CEST 2008
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/SiStripCluster/interface/SiStripClusterCollection.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include <DataFormats/Common/interface/DetSetVector.h>

//
// class decleration
//

class ClusterCount : public DQMEDAnalyzer {

   public:

      explicit ClusterCount(const edm::ParameterSet&);
      ~ClusterCount();


   private:
      void bookHistograms(DQMStore::IBooker&, edm::Run const&,
                          edm::EventSetup const&) override;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);

      // ----------member data ---------------------------
      //      edm::InputTag clusterLabel_;
      edm::EDGetTokenT<edm::DetSetVector<SiStripCluster> > clusterToken_;

};

#endif

