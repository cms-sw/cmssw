#ifndef _DQMOFFLINE_HCAL_HCALRECHITSDQMCLIENT_H_
#define _DQMOFFLINE_HCAL_HCALRECHITSDQMCLIENT_H_

// -*- C++ -*-
//
// 
/*
 Description: This is a RecHits client meant to plot rechits quantities 
*/

//
// Originally create by: Hongxuan Liu 
//                        May 2010
//

#include <memory>
#include <unistd.h>
#include <FWCore/Framework/interface/EDAnalyzer.h>
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMEDHarvester.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "Geometry/Records/interface/HcalRecNumberingRecord.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"


#include <iostream>
#include <fstream>
#include <vector>

class MonitorElement;

class HcalRecHitsDQMClient : public DQMEDHarvester {
 
 private:
  std::string outputFile_;
  edm::ParameterSet conf_;

  bool verbose_;
  bool debug_;

  std::string dirName_;
  std::string dirNameJet_;
  std::string dirNameMET_;

  const HcalDDDRecConstants *hcons;
  int maxDepthHB_, maxDepthHE_, maxDepthHO_, maxDepthHF_, maxDepthAll_;
 
  int nChannels_[5]; // 0:any, 1:HB, 2:HE, 3:HO, 4: HF

 public:
  explicit HcalRecHitsDQMClient(const edm::ParameterSet& );
  virtual ~HcalRecHitsDQMClient();
  
  virtual void beginJob(void) override;
  virtual void beginRun(edm::Run const&, edm::EventSetup const&) override;
  virtual void dqmEndJob(DQMStore::IBooker &, DQMStore::IGetter &) override; //performed in the endJob

  int HcalRecHitsEndjob(const std::vector<MonitorElement*> &hcalMEs);

  float phifactor(int ieta);

};
 
#endif // _DQMOFFLINE_HCAL_HCALRECHITSDQMCLIENT_H_
