#ifndef RecoTauTag_FastL1GlobalAlgo_h
#define RecoTauTag_FastL1GlobalAlgo_h
// -*- C++ -*-
//
// Package:    L1CaloSim
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
// $Id: FastL1GlobalAlgo.h,v 1.20 2009/03/23 11:41:27 chinhan Exp $
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
#include "DataFormats/L1Trigger/interface/L1EtMissParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1ParticleMap.h"

#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"

#include "FastSimulation/L1CaloTriggerProducer/interface/FastL1Region.h"
#include "FastSimDataFormats/External/interface/FastL1BitInfo.h"

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

      //l1extra::L1EtMissParticle getMET() const { return m_MET; }
      l1extra::L1EtMissParticleCollection getMET() const { return m_METs; }
      l1extra::L1JetParticleCollection getTauJets() const { return m_TauJets; }
      l1extra::L1JetParticleCollection getCenJets() const { return m_CenJets; }
      l1extra::L1JetParticleCollection getForJets() const { return m_ForJets; }
      l1extra::L1EmParticleCollection getEgammas() const { return m_Egammas; }
      l1extra::L1EmParticleCollection getisoEgammas() const { return m_isoEgammas; }
      FastL1BitInfoCollection getBitInfos() { return m_BitInfos; }

      void FillBitInfos();
      void InitL1Regions();

     // ------------ Methods using Towers and RecHits ------------
      void FillMET(); // old version
      void FillMET(edm::Event const& e);
      void FillL1Regions(edm::Event const& e, const edm::EventSetup& c);
      void FillJets(const edm::EventSetup& e) { findJets(); };
      void FillEgammas(edm::Event const&);
 
      // ------------ Methods using Trigger Primitives------------
      void FillL1RegionsTP(edm::Event const& e, const edm::EventSetup& c);
      void FillEgammasTP(edm::Event const&);

      std::vector<FastL1Region> GetCaloRegions(){return m_Regions;}//KP
      

 private:
      bool isMaxEtRgn_Window33(int rgnid);
      //int isEMCand(CaloTowerDetId cid, l1extra::L1EmParticle p,const edm::Event& e);
      int isEMCand(CaloTowerDetId cid, l1extra::L1EmParticle *p,const edm::Event& e);
      bool isTauJet(int rgnid);
      bool TauIsolation(int rgnid);

      void findJets();
      void addJet(int rgnId, bool taubit);
      void checkMapping();
      bool greaterEt(const reco::Candidate& a, const reco::Candidate& b);

      double hcaletValue(const int ieta,const int compET);
      // ----------member data ---------------------------
      // output data
      //l1extra::L1EtMissParticle m_MET;
      l1extra::L1EtMissParticleCollection m_METs;
      l1extra::L1JetParticleCollection m_TauJets;
      l1extra::L1JetParticleCollection m_CenJets;
      l1extra::L1JetParticleCollection m_ForJets;
      l1extra::L1EmParticleCollection m_Egammas;
      l1extra::L1EmParticleCollection m_isoEgammas;
      FastL1BitInfoCollection m_BitInfos;

      std::vector<FastL1Region> m_Regions;
      FastL1RegionMap* m_RMap;
      bool m_DoBitInfo;

      bool m_GctIso;
      double m_IsolationEt;

      L1Config m_L1Config;
      double m_hcaluncomp[33][256];
};

#endif
