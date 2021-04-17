#ifndef MuonIdentification_MuonShowerDigiFiller_h
#define MuonIdentification_MuonShowerDigiFiller_h

// -*- C++ -*-
//
// Package:    MuonShowerDigiFiller
// Class:      MuonShowerDigiFiller
//
/**\class MuonShowerDigiFiller MuonShowerDigiFiller.h RecoMuon/MuonIdentification/interface/MuonShowerDigiFiller.h

 Description: Class filling shower information using DT and CSC digis

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Carlo Battilana, INFN BO
//         Created:  Sat Mar 23 14:36:22 CET 2019
//
//

// system include files

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "DataFormats/DTDigi/interface/DTDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCStripDigiCollection.h"

#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"

#include "DataFormats/MuonReco/interface/MuonChamberMatch.h"
#include "TrackingTools/TrackAssociator/interface/TAMuonChamberMatch.h"

//
// class decleration
//

class MuonShowerDigiFiller {
public:
  MuonShowerDigiFiller(const edm::ParameterSet&, edm::ConsumesCollector&& iC);

  void getES(const edm::EventSetup& iSetup);
  void getDigis(edm::Event& iEvent);

  void fill(reco::MuonChamberMatch& muChMatch) const;
  void fillDefault(reco::MuonChamberMatch& muChMatch) const;

private:
  double m_digiMaxDistanceX;

  edm::EDGetTokenT<DTDigiCollection> m_dtDigisToken;
  edm::EDGetTokenT<CSCStripDigiCollection> m_cscDigisToken;

  edm::ESGetToken<DTGeometry, MuonGeometryRecord> m_dtGeometryToken;
  edm::ESGetToken<CSCGeometry, MuonGeometryRecord> m_cscGeometryToken;

  edm::ESHandle<DTGeometry> m_dtGeometry;
  edm::ESHandle<CSCGeometry> m_cscGeometry;

  edm::Handle<DTDigiCollection> m_dtDigis;
  edm::Handle<CSCStripDigiCollection> m_cscDigis;
};

#endif
