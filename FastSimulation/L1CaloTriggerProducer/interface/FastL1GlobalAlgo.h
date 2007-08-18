#ifndef FastL1CaloSim_FastL1GlobalAlgo_h
#define FastL1CaloSim_FastL1GlobalAlgo_h
// -*- C++ -*-
//
// Package:    FastL1CaloSim
// Class:      FastL1GlobalAlgo
// 
/**\class FastL1GlobalAlgo

 Description: Global algorithm.

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Chi Nhan Nguyen
//         Created:  Mon Feb 19 13:25:24 CST 2007
// $Id: FastL1GlobalAlgo.h,v 1.5 2007/06/17 14:31:34 chinhan Exp $
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include <string>
#include <iostream>
#include <vector>

#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/Math/interface/LorentzVectorFwd.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/Math/interface/Vector3D.h"

#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"

#include "DataFormats/L1Trigger/interface/L1EmParticle.h"
#include "DataFormats/L1Trigger/interface/L1JetParticle.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticle.h"
#include "DataFormats/L1Trigger/interface/L1EtMissParticle.h"
#include "DataFormats/L1Trigger/interface/L1ParticleMap.h"

#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"

#include "FastSimulation/L1CaloTriggerProducer/interface/FastL1Region.h"
// No BitInfos for release versions
//#include "DataFormats/FastL1/interface/FastL1BitInfo.h"

#include "Geometry/CaloTopology/interface/CaloTowerConstituentsMap.h"
#include "DataFormats/Math/interface/Vector3D.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// class decleration
//
class FastL1GlobalAlgo {
   public:
      explicit FastL1GlobalAlgo(const edm::ParameterSet&);
      ~FastL1GlobalAlgo();

      void CaloTowersDump(edm::Event const& e);

      l1extra::L1EtMissParticle getMET() const { return m_MET; }
      l1extra::L1JetParticleCollection getTauJets() const { return m_TauJets; }
      l1extra::L1JetParticleCollection getCenJets() const { return m_CenJets; }
      l1extra::L1JetParticleCollection getForJets() const { return m_ForJets; }
      l1extra::L1EmParticleCollection getEgammas() const { return m_Egammas; }
      l1extra::L1EmParticleCollection getisoEgammas() const { return m_isoEgammas; }
      //FastL1BitInfoCollection getBitInfos() { return m_BitInfos; }

      //void FillBitInfos();
      void InitL1Regions();

     // ------------ Methods using Towers and RecHits ------------
      void FillMET(); // old version
      void FillMET(edm::Event const& e);
      void FillL1Regions(edm::Event const& e, const edm::EventSetup& c);
      void FillJets(const edm::EventSetup& e) { findJets(); };
      void FillEgammas(edm::Event const&);
 
      // ------------ Methods using Trigger Primitives------------
      void FillL1RegionsTP(edm::Event const& e, const edm::EventSetup& c);

 private:
      bool isMaxEtRgn_Window33(int rgnid);
      int isEMCand(CaloTowerDetId cid, l1extra::L1EmParticle* p,const edm::Event& e);
      bool isTauJet(int rgnid);
      void findJets();
      void addJet(int rgnId, bool taubit);
      void checkMapping();
      bool greaterEt(const reco::Candidate& a, const reco::Candidate& b);

      // ----------member data ---------------------------
      // output data
      l1extra::L1EtMissParticle m_MET;
      l1extra::L1JetParticleCollection m_TauJets;
      l1extra::L1JetParticleCollection m_CenJets;
      l1extra::L1JetParticleCollection m_ForJets;
      l1extra::L1EmParticleCollection m_Egammas;
      l1extra::L1EmParticleCollection m_isoEgammas;
      //FastL1BitInfoCollection m_BitInfos;

      std::vector<FastL1Region> m_Regions;
      FastL1RegionMap* m_RMap;

      FastL1Config m_L1Config;
};

#endif
