// -*- C++ -*-
//
// Package:    L1Trigger/L1TGloba
// Class:      L1TGlobalAnalyzer
//
/**\class L1TGlobalAnalyzer L1TGlobalAnalyzer.cc L1Trigger/L1TGlobal/plugins/L1TGlobalAnalyzer.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  James Brooke
//         Created:  Tue, 11 Mar 2014 14:55:45 GMT
//
// Modifying Author:  Brian Winer
//         Created: Tue, 10 Mar 2015 based off L1TCaloAnalyzer
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "DataFormats/L1TGlobal/interface/GlobalAlgBlk.h"
#include "DataFormats/L1TGlobal/interface/GlobalExtBlk.h"
#include "DataFormats/L1Trigger/interface/Muon.h"
#include "DataFormats/L1Trigger/interface/EGamma.h"
#include "DataFormats/L1Trigger/interface/Tau.h"
#include "DataFormats/L1Trigger/interface/Jet.h"
#include "DataFormats/L1Trigger/interface/EtSum.h"

#include "TH1F.h"
#include "TH2F.h"

//
// class declaration
//

namespace l1t {

  class L1TGlobalAnalyzer : public edm::one::EDAnalyzer<> {
  public:
    explicit L1TGlobalAnalyzer(const edm::ParameterSet&);
    ~L1TGlobalAnalyzer() override;

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  private:
    void beginJob() override;
    void analyze(const edm::Event&, const edm::EventSetup&) override;
    void endJob() override;

    //virtual void beginRun(edm::Run const&, edm::EventSetup const&) override;
    //virtual void endRun(edm::Run const&, edm::EventSetup const&) override;
    //virtual void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;
    //virtual void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;

    // ----------member data ---------------------------
    edm::EDGetToken m_gmuToken;
    edm::EDGetToken m_dmxEGToken;
    edm::EDGetToken m_dmxTauToken;
    edm::EDGetToken m_dmxJetToken;
    edm::EDGetToken m_dmxSumToken;
    edm::EDGetToken m_muToken;
    edm::EDGetToken m_egToken;
    edm::EDGetToken m_tauToken;
    edm::EDGetToken m_jetToken;
    edm::EDGetToken m_sumToken;

    edm::EDGetToken m_gtAlgToken;
    edm::EDGetToken m_emulGtAlgToken;
    edm::EDGetToken m_emulDxAlgToken;

    bool m_dogMUs;
    bool m_doDmxEGs;
    bool m_doDmxTaus;
    bool m_doDmxJets;
    bool m_doDmxSums;
    bool m_doMUs;
    bool m_doEGs;
    bool m_doTaus;
    bool m_doJets;
    bool m_doSums;

    bool m_doGtAlg;
    bool m_doEmulGtAlg;
    bool m_doEmulDxAlg;

    bool doText_;
    bool doHistos_;

    enum ObjectType {
      MU = 0x1,
      EG = 0x2,
      Tau = 0x3,
      Jet = 0x4,
      Sum = 0x5,
      DmxEG = 0x6,
      DmxTau = 0x7,
      DmxJet = 0x8,
      DmxSum = 0x9,
      GtAlg = 0xA,
      EmulGtAlg = 0xB,
      gMU = 0xC
    };

    std::vector<ObjectType> types_;
    std::vector<std::string> typeStr_;

    std::map<ObjectType, TFileDirectory> dirs_;
    std::map<ObjectType, TH1F*> het_;
    std::map<ObjectType, TH1F*> heta_;
    std::map<ObjectType, TH1F*> hphi_;
    std::map<ObjectType, TH1F*> hbx_;
    std::map<ObjectType, TH2F*> hetaphi_;

    TFileDirectory evtDispDir_;
    TFileDirectory algDir_;
    TFileDirectory dmxVGtDir_;
    TH1F* hAlgoBits_;
    TH1F* hEmulGtAlgoBits_;
    TH1F* hEmulDxAlgoBits_;
    TH2F* hAlgoBitsEmulGtVsHw_;
    TH2F* hAlgoBitsEmulDxVsHw_;
    TH2F* hGmtVsGTMUEt_;
    TH2F* hGmtVsGTMUEta_;
    TH2F* hGmtVsGTMUPhi_;
    TH2F* hDmxVsGTEGEt_;
    TH2F* hDmxVsGTEGEta_;
    TH2F* hDmxVsGTEGPhi_;
    TH2F* hDmxVsGTTauEt_;
    TH2F* hDmxVsGTTauEta_;
    TH2F* hDmxVsGTTauPhi_;
    TH2F* hDmxVsGTJetEt_;
    TH2F* hDmxVsGTJetEta_;
    TH2F* hDmxVsGTJetPhi_;
    TH2F* hDmxVsGTSumEt_ETT_;
    TH2F* hDmxVsGTSumEt_ETTem_;
    TH2F* hDmxVsGTSumEt_HTT_;
    TH2F* hDmxVsGTSumEt_ETM_;
    TH2F* hDmxVsGTSumPhi_ETM_;
    TH2F* hDmxVsGTSumEt_ETMHF_;
    TH2F* hDmxVsGTSumPhi_ETMHF_;
    TH2F* hDmxVsGTSumEt_HTM_;
    TH2F* hDmxVsGTSumPhi_HTM_;
    TH2F* hDmxVsGTSumEt_HFP0_;
    TH2F* hDmxVsGTSumEt_HFM0_;
    TH2F* hDmxVsGTSumEt_HFP1_;
    TH2F* hDmxVsGTSumEt_HFM1_;
  };

  //
  // constants, enums and typedefs
  //

  //
  // static data member definitions
  //

  //
  // constructors and destructor
  //
  L1TGlobalAnalyzer::L1TGlobalAnalyzer(const edm::ParameterSet& iConfig)
      : doText_(iConfig.getUntrackedParameter<bool>("doText", true)),
        doHistos_(iConfig.getUntrackedParameter<bool>("doHistos", true)) {
    //now do what ever initialization is needed

    // register what you consume and keep token for later access:
    edm::InputTag nullTag("None");

    edm::InputTag gmuTag = iConfig.getParameter<edm::InputTag>("gmuToken");
    m_gmuToken = consumes<l1t::MuonBxCollection>(gmuTag);
    m_dogMUs = !(gmuTag == nullTag);

    edm::InputTag dmxEGTag = iConfig.getParameter<edm::InputTag>("dmxEGToken");
    m_dmxEGToken = consumes<l1t::EGammaBxCollection>(dmxEGTag);
    m_doDmxEGs = !(dmxEGTag == nullTag);

    edm::InputTag dmxTauTag = iConfig.getParameter<edm::InputTag>("dmxTauToken");
    m_dmxTauToken = consumes<l1t::TauBxCollection>(dmxTauTag);
    m_doDmxTaus = !(dmxTauTag == nullTag);

    edm::InputTag dmxJetTag = iConfig.getParameter<edm::InputTag>("dmxJetToken");
    m_dmxJetToken = consumes<l1t::JetBxCollection>(dmxJetTag);
    m_doDmxJets = !(dmxJetTag == nullTag);

    edm::InputTag dmxSumTag = iConfig.getParameter<edm::InputTag>("dmxEtSumToken");
    m_dmxSumToken = consumes<l1t::EtSumBxCollection>(dmxSumTag);
    m_doDmxSums = !(dmxSumTag == nullTag);

    edm::InputTag muTag = iConfig.getParameter<edm::InputTag>("muToken");
    m_muToken = consumes<l1t::MuonBxCollection>(muTag);
    m_doMUs = !(muTag == nullTag);

    edm::InputTag egTag = iConfig.getParameter<edm::InputTag>("egToken");
    m_egToken = consumes<l1t::EGammaBxCollection>(egTag);
    m_doEGs = !(egTag == nullTag);

    edm::InputTag tauTag = iConfig.getParameter<edm::InputTag>("tauToken");
    m_tauToken = consumes<l1t::TauBxCollection>(tauTag);
    m_doTaus = !(tauTag == nullTag);

    edm::InputTag jetTag = iConfig.getParameter<edm::InputTag>("jetToken");
    m_jetToken = consumes<l1t::JetBxCollection>(jetTag);
    m_doJets = !(jetTag == nullTag);

    edm::InputTag sumTag = iConfig.getParameter<edm::InputTag>("etSumToken");
    m_sumToken = consumes<l1t::EtSumBxCollection>(sumTag);
    m_doSums = !(sumTag == nullTag);

    edm::InputTag gtAlgTag = iConfig.getParameter<edm::InputTag>("gtAlgToken");
    m_gtAlgToken = consumes<GlobalAlgBlkBxCollection>(gtAlgTag);
    m_doGtAlg = !(gtAlgTag == nullTag);

    edm::InputTag emulGtAlgTag = iConfig.getParameter<edm::InputTag>("emulGtAlgToken");
    m_emulGtAlgToken = consumes<GlobalAlgBlkBxCollection>(emulGtAlgTag);
    m_doEmulGtAlg = !(emulGtAlgTag == nullTag);

    edm::InputTag emulDxAlgTag = iConfig.getParameter<edm::InputTag>("emulDxAlgToken");
    m_emulDxAlgToken = consumes<GlobalAlgBlkBxCollection>(emulDxAlgTag);
    m_doEmulDxAlg = !(emulDxAlgTag == nullTag);

    types_.push_back(gMU);
    types_.push_back(DmxEG);
    types_.push_back(DmxTau);
    types_.push_back(DmxJet);
    types_.push_back(DmxSum);
    types_.push_back(MU);
    types_.push_back(EG);
    types_.push_back(Tau);
    types_.push_back(Jet);
    types_.push_back(Sum);

    typeStr_.push_back("gmtmu");
    typeStr_.push_back("dmxeg");
    typeStr_.push_back("dmxtau");
    typeStr_.push_back("dmxjet");
    typeStr_.push_back("dmxsum");
    typeStr_.push_back("mu");
    typeStr_.push_back("eg");
    typeStr_.push_back("tau");
    typeStr_.push_back("jet");
    typeStr_.push_back("sum");
  }

  L1TGlobalAnalyzer::~L1TGlobalAnalyzer() {
    // do anything here that needs to be done at desctruction time
    // (e.g. close files, deallocate resources etc.)
  }

  //
  // member functions
  //

  // ------------ method called for each event  ------------
  void L1TGlobalAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
    using namespace edm;

    std::stringstream text;

    // get gmt Muons
    if (m_dogMUs) {
      Handle<BXVector<l1t::Muon> > gmuons;
      iEvent.getByToken(m_gmuToken, gmuons);

      for (int ibx = gmuons->getFirstBX(); ibx <= gmuons->getLastBX(); ++ibx) {
        for (auto itr = gmuons->begin(ibx); itr != gmuons->end(ibx); ++itr) {
          hbx_.at(gMU)->Fill(ibx);
          het_.at(gMU)->Fill(itr->hwPt());
          heta_.at(gMU)->Fill(itr->hwEtaAtVtx());
          hphi_.at(gMU)->Fill(itr->hwPhiAtVtx());
          hetaphi_.at(gMU)->Fill(itr->hwEtaAtVtx(), itr->hwPhiAtVtx(), itr->hwPt());

          text << "Muon : "
               << " BX=" << ibx << " ipt=" << itr->hwPt() << " ieta=" << itr->hwEtaAtVtx()
               << " iphi=" << itr->hwPhiAtVtx() << std::endl;
        }
      }
    }

    // get EG
    if (m_doDmxEGs) {
      Handle<BXVector<l1t::EGamma> > dmxegs;
      iEvent.getByToken(m_dmxEGToken, dmxegs);

      for (int ibx = dmxegs->getFirstBX(); ibx <= dmxegs->getLastBX(); ++ibx) {
        for (auto itr = dmxegs->begin(ibx); itr != dmxegs->end(ibx); ++itr) {
          hbx_.at(DmxEG)->Fill(ibx);
          het_.at(DmxEG)->Fill(itr->hwPt());
          heta_.at(DmxEG)->Fill(itr->hwEta());
          hphi_.at(DmxEG)->Fill(itr->hwPhi());
          hetaphi_.at(DmxEG)->Fill(itr->hwEta(), itr->hwPhi(), itr->hwPt());

          text << "Dmx EG : "
               << " BX=" << ibx << " ipt=" << itr->hwPt() << " ieta=" << itr->hwEta() << " iphi=" << itr->hwPhi()
               << std::endl;
        }
      }
    }

    // get tau
    if (m_doDmxTaus) {
      Handle<BXVector<l1t::Tau> > dmxtaus;
      iEvent.getByToken(m_dmxTauToken, dmxtaus);

      for (int ibx = dmxtaus->getFirstBX(); ibx <= dmxtaus->getLastBX(); ++ibx) {
        for (auto itr = dmxtaus->begin(ibx); itr != dmxtaus->end(ibx); ++itr) {
          hbx_.at(DmxTau)->Fill(ibx);
          het_.at(DmxTau)->Fill(itr->hwPt());
          heta_.at(DmxTau)->Fill(itr->hwEta());
          hphi_.at(DmxTau)->Fill(itr->hwPhi());
          hetaphi_.at(DmxTau)->Fill(itr->hwEta(), itr->hwPhi(), itr->hwPt());

          text << "Dmx Tau : "
               << " BX=" << ibx << " ipt=" << itr->hwPt() << " ieta=" << itr->hwEta() << " iphi=" << itr->hwPhi()
               << std::endl;
        }
      }
    }

    // get jet
    if (m_doDmxJets) {
      Handle<BXVector<l1t::Jet> > dmxjets;
      iEvent.getByToken(m_dmxJetToken, dmxjets);

      for (int ibx = dmxjets->getFirstBX(); ibx <= dmxjets->getLastBX(); ++ibx) {
        for (auto itr = dmxjets->begin(ibx); itr != dmxjets->end(ibx); ++itr) {
          hbx_.at(DmxJet)->Fill(ibx);
          het_.at(DmxJet)->Fill(itr->hwPt());
          heta_.at(DmxJet)->Fill(itr->hwEta());
          hphi_.at(DmxJet)->Fill(itr->hwPhi());
          hetaphi_.at(DmxJet)->Fill(itr->hwEta(), itr->hwPhi(), itr->hwPt());

          text << "Dmx Jet : "
               << " BX=" << ibx << " ipt=" << itr->hwPt() << " ieta=" << itr->hwEta() << " iphi=" << itr->hwPhi()
               << std::endl;
        }
      }
    }

    // get sums
    if (m_doDmxSums) {
      Handle<BXVector<l1t::EtSum> > dmxsums;
      iEvent.getByToken(m_dmxSumToken, dmxsums);

      for (int ibx = dmxsums->getFirstBX(); ibx <= dmxsums->getLastBX(); ++ibx) {
        for (auto itr = dmxsums->begin(ibx); itr != dmxsums->end(ibx); ++itr) {
          hbx_.at(DmxSum)->Fill(ibx);
          het_.at(DmxSum)->Fill(itr->hwPt());
          heta_.at(DmxSum)->Fill(itr->hwEta());
          hphi_.at(DmxSum)->Fill(itr->hwPhi());
          hetaphi_.at(DmxSum)->Fill(itr->hwEta(), itr->hwPhi(), itr->hwPt());

          text << "Dmx Sum : "
               << " type=" << itr->getType() << " BX=" << ibx << " ipt=" << itr->hwPt() << " ieta=" << itr->hwEta()
               << " iphi=" << itr->hwPhi() << std::endl;
        }
      }
    }

    // get Muons
    if (m_doMUs) {
      Handle<BXVector<l1t::Muon> > muons;
      iEvent.getByToken(m_muToken, muons);

      for (int ibx = muons->getFirstBX(); ibx <= muons->getLastBX(); ++ibx) {
        for (auto itr = muons->begin(ibx); itr != muons->end(ibx); ++itr) {
          hbx_.at(MU)->Fill(ibx);
          het_.at(MU)->Fill(itr->hwPt());
          heta_.at(MU)->Fill(itr->hwEtaAtVtx());
          hphi_.at(MU)->Fill(itr->hwPhiAtVtx());
          hetaphi_.at(MU)->Fill(itr->hwEtaAtVtx(), itr->hwPhiAtVtx(), itr->hwPt());

          text << "Muon : "
               << " BX=" << ibx << " ipt=" << itr->hwPt() << " ieta=" << itr->hwEtaAtVtx()
               << " iphi=" << itr->hwPhiAtVtx() << std::endl;
        }
      }
    }

    // get EG
    if (m_doEGs) {
      Handle<BXVector<l1t::EGamma> > egs;
      iEvent.getByToken(m_egToken, egs);

      for (int ibx = egs->getFirstBX(); ibx <= egs->getLastBX(); ++ibx) {
        for (auto itr = egs->begin(ibx); itr != egs->end(ibx); ++itr) {
          hbx_.at(EG)->Fill(ibx);
          het_.at(EG)->Fill(itr->hwPt());
          heta_.at(EG)->Fill(itr->hwEta());
          hphi_.at(EG)->Fill(itr->hwPhi());
          hetaphi_.at(EG)->Fill(itr->hwEta(), itr->hwPhi(), itr->hwPt());

          text << "EG : "
               << " BX=" << ibx << " ipt=" << itr->hwPt() << " ieta=" << itr->hwEta() << " iphi=" << itr->hwPhi()
               << std::endl;
        }
      }
    }

    // get tau
    if (m_doTaus) {
      Handle<BXVector<l1t::Tau> > taus;
      iEvent.getByToken(m_tauToken, taus);

      for (int ibx = taus->getFirstBX(); ibx <= taus->getLastBX(); ++ibx) {
        for (auto itr = taus->begin(ibx); itr != taus->end(ibx); ++itr) {
          hbx_.at(Tau)->Fill(ibx);
          het_.at(Tau)->Fill(itr->hwPt());
          heta_.at(Tau)->Fill(itr->hwEta());
          hphi_.at(Tau)->Fill(itr->hwPhi());
          hetaphi_.at(Tau)->Fill(itr->hwEta(), itr->hwPhi(), itr->hwPt());

          text << "Tau : "
               << " BX=" << ibx << " ipt=" << itr->hwPt() << " ieta=" << itr->hwEta() << " iphi=" << itr->hwPhi()
               << std::endl;
        }
      }
    }

    // get jet
    if (m_doJets) {
      Handle<BXVector<l1t::Jet> > jets;
      iEvent.getByToken(m_jetToken, jets);

      for (int ibx = jets->getFirstBX(); ibx <= jets->getLastBX(); ++ibx) {
        for (auto itr = jets->begin(ibx); itr != jets->end(ibx); ++itr) {
          hbx_.at(Jet)->Fill(ibx);
          het_.at(Jet)->Fill(itr->hwPt());
          heta_.at(Jet)->Fill(itr->hwEta());
          hphi_.at(Jet)->Fill(itr->hwPhi());
          hetaphi_.at(Jet)->Fill(itr->hwEta(), itr->hwPhi(), itr->hwPt());

          text << "Jet : "
               << " BX=" << ibx << " ipt=" << itr->hwPt() << " ieta=" << itr->hwEta() << " iphi=" << itr->hwPhi()
               << std::endl;
        }
      }
    }

    // get sums
    if (m_doSums) {
      Handle<BXVector<l1t::EtSum> > sums;
      iEvent.getByToken(m_sumToken, sums);

      for (int ibx = sums->getFirstBX(); ibx <= sums->getLastBX(); ++ibx) {
        for (auto itr = sums->begin(ibx); itr != sums->end(ibx); ++itr) {
          hbx_.at(Sum)->Fill(ibx);
          het_.at(Sum)->Fill(itr->hwPt());
          heta_.at(Sum)->Fill(itr->hwEta());
          hphi_.at(Sum)->Fill(itr->hwPhi());
          hetaphi_.at(Sum)->Fill(itr->hwEta(), itr->hwPhi(), itr->hwPt());
          text << "Sum : "
               << " type=" << itr->getType() << " BX=" << ibx << " ipt=" << itr->hwPt() << " ieta=" << itr->hwEta()
               << " iphi=" << itr->hwPhi() << std::endl;
        }
      }
    }

    //Algorithm Bits
    if (m_doGtAlg) {
      Handle<BXVector<GlobalAlgBlk> > algs;
      iEvent.getByToken(m_gtAlgToken, algs);

      for (int ibx = algs->getFirstBX(); ibx <= algs->getLastBX(); ++ibx) {
        for (auto itr = algs->begin(ibx); itr != algs->end(ibx); ++itr) {
          for (int algBit = 0; algBit < 128;
               algBit++) {  //Fix Me: Should access size of algo vector...need method in GlobalAlgBlk class
            if (itr->getAlgoDecisionFinal(algBit)) {
              hAlgoBits_->Fill(algBit);
              text << "HW Fired Alg Bit : " << algBit << std::endl;
            }
          }
        }
      }
    }

    //Algorithm Bits (Emulation seeded by GT input)
    if (m_doEmulGtAlg) {
      Handle<BXVector<GlobalAlgBlk> > emulGtAlgs;
      iEvent.getByToken(m_emulGtAlgToken, emulGtAlgs);

      for (int ibx = emulGtAlgs->getFirstBX(); ibx <= emulGtAlgs->getLastBX(); ++ibx) {
        for (auto itr = emulGtAlgs->begin(ibx); itr != emulGtAlgs->end(ibx); ++itr) {
          for (int algBit = 0; algBit < 128;
               algBit++) {  //Fix Me: Should access size of algo vector...need method in GlobalAlgBlk class
            if (itr->getAlgoDecisionFinal(algBit)) {
              hEmulGtAlgoBits_->Fill(algBit);
              text << "Emul w/ GTInput Fired Alg Bit : " << algBit << std::endl;
            }
          }
        }
      }
    }

    //Algorithm Bits (Emulation seeded by Demux Output)
    if (m_doEmulDxAlg) {
      Handle<BXVector<GlobalAlgBlk> > emulDxAlgs;
      iEvent.getByToken(m_emulDxAlgToken, emulDxAlgs);

      for (int ibx = emulDxAlgs->getFirstBX(); ibx <= emulDxAlgs->getLastBX(); ++ibx) {
        for (auto itr = emulDxAlgs->begin(ibx); itr != emulDxAlgs->end(ibx); ++itr) {
          for (int algBit = 0; algBit < 128;
               algBit++) {  //Fix Me: Should access size of algo vector...need method in GlobalAlgBlk class
            if (itr->getAlgoDecisionFinal(algBit)) {
              hEmulDxAlgoBits_->Fill(algBit);
              text << "Emul w/ Demux output Fired Alg Bit : " << algBit << std::endl;
            }
          }
        }
      }
    }

    // Mu (gmt vs GT)
    if (m_doMUs & m_dogMUs) {
      Handle<BXVector<l1t::Muon> > mus;
      iEvent.getByToken(m_muToken, mus);

      Handle<BXVector<l1t::Muon> > gmtmus;
      iEvent.getByToken(m_gmuToken, gmtmus);

      for (int ibx = mus->getFirstBX(); ibx <= mus->getLastBX(); ++ibx) {
        // Cycle through all GT MUs
        for (unsigned int igtMU = 0; igtMU < mus->size(ibx); igtMU++) {
          double gtMUEt = mus->at(ibx, igtMU).hwPt();
          double gmtMUEt = 0.0;
          if (gmtmus->size(ibx) > igtMU)
            gmtMUEt = gmtmus->at(ibx, igtMU).hwPt();
          hGmtVsGTMUEt_->Fill(gtMUEt, gmtMUEt);

          double gtMUEta = mus->at(ibx, igtMU).hwEtaAtVtx();
          double gmtMUEta = 0.0;
          if (gmtmus->size(ibx) > igtMU)
            gmtMUEta = gmtmus->at(ibx, igtMU).hwEtaAtVtx();
          hGmtVsGTMUEta_->Fill(gtMUEta, gmtMUEta);

          double gtMUPhi = mus->at(ibx, igtMU).hwPhiAtVtx();
          double gmtMUPhi = 0.0;
          if (gmtmus->size(ibx) > igtMU)
            gmtMUPhi = gmtmus->at(ibx, igtMU).hwPhiAtVtx();
          hGmtVsGTMUPhi_->Fill(gtMUPhi, gmtMUPhi);
        }
        // if there are extra MUs in the dmx record them
        for (unsigned int igmtMU = mus->size(ibx); igmtMU < gmtmus->size(ibx); igmtMU++) {
          double gtMUEt = 0.0;  //no GT jet exists
          double gmtMUEt = gmtmus->at(ibx, igmtMU).hwPt();
          hGmtVsGTMUEt_->Fill(gtMUEt, gmtMUEt);

          double gtMUEta = 0.0;
          double gmtMUEta = gmtmus->at(ibx, igmtMU).hwEtaAtVtx();
          hGmtVsGTMUEta_->Fill(gtMUEta, gmtMUEta);

          double gtMUPhi = 0.0;
          double gmtMUPhi = gmtmus->at(ibx, igmtMU).hwPhiAtVtx();
          hGmtVsGTMUPhi_->Fill(gtMUPhi, gmtMUPhi);
        }
      }
    }

    // EG (Dmx vs GT)
    if (m_doEGs & m_doDmxEGs) {
      Handle<BXVector<l1t::EGamma> > egs;
      iEvent.getByToken(m_egToken, egs);

      Handle<BXVector<l1t::EGamma> > dmxegs;
      iEvent.getByToken(m_dmxEGToken, dmxegs);

      for (int ibx = egs->getFirstBX(); ibx <= egs->getLastBX(); ++ibx) {
        // Cycle through all GT egs
        for (unsigned int igtEG = 0; igtEG < egs->size(ibx); igtEG++) {
          double gtEGEt = egs->at(ibx, igtEG).hwPt();
          double dmxEGEt = 0.0;
          if (dmxegs->size(ibx) > igtEG)
            dmxEGEt = dmxegs->at(ibx, igtEG).hwPt();
          hDmxVsGTEGEt_->Fill(gtEGEt, dmxEGEt);

          double gtEGEta = egs->at(ibx, igtEG).hwEta();
          double dmxEGEta = 0.0;
          if (dmxegs->size(ibx) > igtEG)
            dmxEGEta = dmxegs->at(ibx, igtEG).hwEta();
          hDmxVsGTEGEta_->Fill(gtEGEta, dmxEGEta);

          double gtEGPhi = egs->at(ibx, igtEG).hwPhi();
          double dmxEGPhi = 0.0;
          if (dmxegs->size(ibx) > igtEG)
            dmxEGPhi = dmxegs->at(ibx, igtEG).hwPhi();
          hDmxVsGTEGPhi_->Fill(gtEGPhi, dmxEGPhi);
        }
        // if there are extra egs in the dmx record them
        for (unsigned int idmEG = egs->size(ibx); idmEG < dmxegs->size(ibx); idmEG++) {
          double gtEGEt = 0.0;  //no GT jet exists
          double dmxEGEt = dmxegs->at(ibx, idmEG).hwPt();
          hDmxVsGTEGEt_->Fill(gtEGEt, dmxEGEt);

          double gtEGEta = 0.0;
          double dmxEGEta = dmxegs->at(ibx, idmEG).hwEta();
          hDmxVsGTEGEta_->Fill(gtEGEta, dmxEGEta);

          double gtEGPhi = 0.0;
          double dmxEGPhi = dmxegs->at(ibx, idmEG).hwPhi();
          hDmxVsGTEGPhi_->Fill(gtEGPhi, dmxEGPhi);
        }
      }
    }

    // Tau (Dmx vs GT)
    if (m_doTaus & m_doDmxTaus) {
      Handle<BXVector<l1t::Tau> > taus;
      iEvent.getByToken(m_tauToken, taus);

      Handle<BXVector<l1t::Tau> > dmxtaus;
      iEvent.getByToken(m_dmxTauToken, dmxtaus);

      for (int ibx = taus->getFirstBX(); ibx <= taus->getLastBX(); ++ibx) {
        // Cycle through all GT taus
        for (unsigned int igtTau = 0; igtTau < taus->size(ibx); igtTau++) {
          double gtTauEt = taus->at(ibx, igtTau).hwPt();
          double dmxTauEt = 0.0;
          if (dmxtaus->size(ibx) > igtTau)
            dmxTauEt = dmxtaus->at(ibx, igtTau).hwPt();
          hDmxVsGTTauEt_->Fill(gtTauEt, dmxTauEt);

          double gtTauEta = taus->at(ibx, igtTau).hwEta();
          double dmxTauEta = 0.0;
          if (dmxtaus->size(ibx) > igtTau)
            dmxTauEta = dmxtaus->at(ibx, igtTau).hwEta();
          hDmxVsGTTauEta_->Fill(gtTauEta, dmxTauEta);

          double gtTauPhi = taus->at(ibx, igtTau).hwPhi();
          double dmxTauPhi = 0.0;
          if (dmxtaus->size(ibx) > igtTau)
            dmxTauPhi = dmxtaus->at(ibx, igtTau).hwPhi();
          hDmxVsGTTauPhi_->Fill(gtTauPhi, dmxTauPhi);
        }
        // if there are extra taus in the dmx record them
        for (unsigned int idmTau = taus->size(ibx); idmTau < dmxtaus->size(ibx); idmTau++) {
          double gtTauEt = 0.0;  //no GT jet exists
          double dmxTauEt = dmxtaus->at(ibx, idmTau).hwPt();
          hDmxVsGTTauEt_->Fill(gtTauEt, dmxTauEt);

          double gtTauEta = 0.0;
          double dmxTauEta = dmxtaus->at(ibx, idmTau).hwEta();
          hDmxVsGTTauEta_->Fill(gtTauEta, dmxTauEta);

          double gtTauPhi = 0.0;
          double dmxTauPhi = dmxtaus->at(ibx, idmTau).hwPhi();
          hDmxVsGTTauPhi_->Fill(gtTauPhi, dmxTauPhi);
        }
      }
    }

    // Jets (Dmx vs GT)
    if (m_doJets & m_doDmxJets) {
      Handle<BXVector<l1t::Jet> > jets;
      iEvent.getByToken(m_jetToken, jets);

      Handle<BXVector<l1t::Jet> > dmxjets;
      iEvent.getByToken(m_dmxJetToken, dmxjets);

      for (int ibx = jets->getFirstBX(); ibx <= jets->getLastBX(); ++ibx) {
        // Cycle through all GT jets
        for (unsigned int igtJ = 0; igtJ < jets->size(ibx); igtJ++) {
          double gtJetEt = jets->at(ibx, igtJ).hwPt();
          double dmxJetEt = 0.0;
          if (dmxjets->size(ibx) > igtJ)
            dmxJetEt = dmxjets->at(ibx, igtJ).hwPt();
          hDmxVsGTJetEt_->Fill(gtJetEt, dmxJetEt);

          double gtJetEta = jets->at(ibx, igtJ).hwEta();
          double dmxJetEta = 0.0;
          if (dmxjets->size(ibx) > igtJ)
            dmxJetEta = dmxjets->at(ibx, igtJ).hwEta();
          hDmxVsGTJetEta_->Fill(gtJetEta, dmxJetEta);

          double gtJetPhi = jets->at(ibx, igtJ).hwPhi();
          double dmxJetPhi = 0.0;
          if (dmxjets->size(ibx) > igtJ)
            dmxJetPhi = dmxjets->at(ibx, igtJ).hwPhi();
          hDmxVsGTJetPhi_->Fill(gtJetPhi, dmxJetPhi);
        }
        // if there are extra jets in the dmx record them
        for (unsigned int idmJ = jets->size(ibx); idmJ < dmxjets->size(ibx); idmJ++) {
          double gtJetEt = 0.0;  //no GT jet exists
          double dmxJetEt = dmxjets->at(ibx, idmJ).hwPt();
          hDmxVsGTJetEt_->Fill(gtJetEt, dmxJetEt);

          double gtJetEta = 0.0;
          double dmxJetEta = dmxjets->at(ibx, idmJ).hwEta();
          hDmxVsGTJetEta_->Fill(gtJetEta, dmxJetEta);

          double gtJetPhi = 0.0;
          double dmxJetPhi = dmxjets->at(ibx, idmJ).hwPhi();
          hDmxVsGTJetPhi_->Fill(gtJetPhi, dmxJetPhi);
        }
      }
    }

    // Sums (Dmx vs GT)
    if (m_doSums & m_doDmxSums) {
      Handle<BXVector<l1t::EtSum> > sums;
      iEvent.getByToken(m_sumToken, sums);

      Handle<BXVector<l1t::EtSum> > dmxSums;
      iEvent.getByToken(m_dmxSumToken, dmxSums);

      for (int ibx = sums->getFirstBX(); ibx <= sums->getLastBX(); ++ibx) {
        // Cycle through all GT sums
        for (unsigned int igtS = 0; igtS < sums->size(ibx); igtS++) {
          double gtSumEt = sums->at(ibx, igtS).hwPt();
          double dmxSumEt = 0.0;
          if (dmxSums->size(ibx) > igtS)
            dmxSumEt = dmxSums->at(ibx, igtS).hwPt();

          double gtSumPhi = sums->at(ibx, igtS).hwPhi();
          double dmxSumPhi = 0.0;
          if (dmxSums->size(ibx) > igtS)
            dmxSumPhi = dmxSums->at(ibx, igtS).hwPhi();

          if (sums->at(ibx, igtS).getType() == dmxSums->at(ibx, igtS).getType()) {
            switch (sums->at(ibx, igtS).getType()) {
              case l1t::EtSum::EtSumType::kTotalEt:
                hDmxVsGTSumEt_ETT_->Fill(gtSumEt, dmxSumEt);

                break;
              case l1t::EtSum::EtSumType::kTotalEtEm:
                hDmxVsGTSumEt_ETTem_->Fill(gtSumEt, dmxSumEt);

                break;
              case l1t::EtSum::EtSumType::kTotalHt:
                hDmxVsGTSumEt_HTT_->Fill(gtSumEt, dmxSumEt);

                break;
              case l1t::EtSum::EtSumType::kMissingEt:
                hDmxVsGTSumEt_ETM_->Fill(gtSumEt, dmxSumEt);
                hDmxVsGTSumPhi_ETM_->Fill(gtSumPhi, dmxSumPhi);
                break;
              case l1t::EtSum::EtSumType::kMissingEtHF:
                hDmxVsGTSumEt_ETMHF_->Fill(gtSumEt, dmxSumEt);
                hDmxVsGTSumPhi_ETMHF_->Fill(gtSumPhi, dmxSumPhi);
                break;
              case l1t::EtSum::EtSumType::kMissingHt:
                hDmxVsGTSumEt_HTM_->Fill(gtSumEt, dmxSumEt);
                hDmxVsGTSumPhi_HTM_->Fill(gtSumPhi, dmxSumPhi);
                break;
              case l1t::EtSum::EtSumType::kMinBiasHFP0:
                hDmxVsGTSumEt_HFP0_->Fill(gtSumEt, dmxSumEt);
                break;
              case l1t::EtSum::EtSumType::kMinBiasHFM0:
                hDmxVsGTSumEt_HFM0_->Fill(gtSumEt, dmxSumEt);
                break;
              case l1t::EtSum::EtSumType::kMinBiasHFP1:
                hDmxVsGTSumEt_HFP1_->Fill(gtSumEt, dmxSumEt);
                break;
              case l1t::EtSum::EtSumType::kMinBiasHFM1:
                hDmxVsGTSumEt_HFM1_->Fill(gtSumEt, dmxSumEt);
                break;
              default:
                break;
            }
          } else {
            text << "WARNING:  EtSum Types do not line up between DeMux and uGT " << std::endl;
          }
        }
        // if there are extra sumss in the dmx record them...should not be any...but let's check
        for (unsigned int idmS = sums->size(ibx); idmS < dmxSums->size(ibx); idmS++) {
          double gtSumEt = -1.0;  //no GT jet exists
          double dmxSumEt = dmxSums->at(ibx, idmS).hwPt();

          double gtSumPhi = -1.0;
          double dmxSumPhi = dmxSums->at(ibx, idmS).hwPhi();

          switch (dmxSums->at(ibx, idmS).getType()) {
            case l1t::EtSum::EtSumType::kTotalEt:
              hDmxVsGTSumEt_ETT_->Fill(gtSumEt, dmxSumEt);

              break;
            case l1t::EtSum::EtSumType::kTotalEtEm:
              hDmxVsGTSumEt_ETTem_->Fill(gtSumEt, dmxSumEt);

              break;
            case l1t::EtSum::EtSumType::kTotalHt:
              hDmxVsGTSumEt_HTT_->Fill(gtSumEt, dmxSumEt);

              break;
            case l1t::EtSum::EtSumType::kMissingEt:
              hDmxVsGTSumEt_ETM_->Fill(gtSumEt, dmxSumEt);
              hDmxVsGTSumPhi_ETM_->Fill(gtSumPhi, dmxSumPhi);
              break;
            case l1t::EtSum::EtSumType::kMissingEtHF:
              hDmxVsGTSumEt_ETMHF_->Fill(gtSumEt, dmxSumEt);
              hDmxVsGTSumPhi_ETMHF_->Fill(gtSumPhi, dmxSumPhi);
              break;
            case l1t::EtSum::EtSumType::kMissingHt:
              hDmxVsGTSumEt_HTM_->Fill(gtSumEt, dmxSumEt);
              hDmxVsGTSumPhi_HTM_->Fill(gtSumPhi, dmxSumPhi);
              break;
            default:
              break;
          }
        }
      }
    }

    //Algorithm Bits (Emulation vs HW)
    if (m_doGtAlg && m_doEmulGtAlg) {
      Handle<BXVector<GlobalAlgBlk> > hwalgs;
      iEvent.getByToken(m_gtAlgToken, hwalgs);

      Handle<BXVector<GlobalAlgBlk> > emulAlgs;
      iEvent.getByToken(m_emulGtAlgToken, emulAlgs);

      //for ( int ibx=hwalgs->getFirstBX(); ibx<=hwalgs->getLastBX(); ++ibx) {
      int ibx = 0;

      auto itr = hwalgs->begin(ibx);
      auto itr_emul = emulAlgs->begin(ibx);

      for (int algBit = 0; algBit < 128;
           algBit++) {  //Fix Me: Should access size of algo vector...need method in GlobalAlgBlk class
        bool hw = itr->getAlgoDecisionFinal(algBit);
        bool em = itr_emul->getAlgoDecisionFinal(algBit);
        if (hw & em) {
          hAlgoBitsEmulGtVsHw_->Fill(algBit, algBit);
        } else if (hw & !em) {
          hAlgoBitsEmulGtVsHw_->Fill(algBit, -1.0);
          text << "WARNING:  HW Fnd Alg Bit " << algBit << " but emulation did not " << std::endl;
        } else if (!hw & em) {
          hAlgoBitsEmulGtVsHw_->Fill(-1.0, algBit);
          text << "WARNING:  Emul. Fnd Alg Bit " << algBit << " but hardware did not " << std::endl;
        }
      }

      // }
    }

    //Algorithm Bits (Emulation vs HW)
    if (m_doGtAlg && m_doEmulDxAlg) {
      Handle<BXVector<GlobalAlgBlk> > hwalgs;
      iEvent.getByToken(m_gtAlgToken, hwalgs);

      Handle<BXVector<GlobalAlgBlk> > emulAlgs;
      iEvent.getByToken(m_emulDxAlgToken, emulAlgs);

      //for ( int ibx=hwalgs->getFirstBX(); ibx<=hwalgs->getLastBX(); ++ibx) {
      int ibx = 0;

      auto itr = hwalgs->begin(ibx);
      auto itr_emul = emulAlgs->begin(ibx);

      for (int algBit = 0; algBit < 128;
           algBit++) {  //Fix Me: Should access size of algo vector...need method in GlobalAlgBlk class
        bool hw = itr->getAlgoDecisionFinal(algBit);
        bool em = itr_emul->getAlgoDecisionFinal(algBit);
        if (hw & em) {
          hAlgoBitsEmulDxVsHw_->Fill(algBit, algBit);
        } else if (hw & !em) {
          hAlgoBitsEmulDxVsHw_->Fill(algBit, -1.0);
        } else if (!hw & em) {
          hAlgoBitsEmulDxVsHw_->Fill(-1.0, algBit);
        }
      }

      // }
    }

    if (doText_)
      edm::LogInfo("L1TGlobalEvents") << text.str();
  }

  // ------------ method called once each job just before starting event loop  ------------
  void L1TGlobalAnalyzer::beginJob() {
    edm::Service<TFileService> fs;

    auto itr = types_.cbegin();
    auto str = typeStr_.cbegin();

    for (; itr != types_.end(); ++itr, ++str) {
      if (*itr == Jet || *itr == DmxJet || *itr == Sum || *itr == DmxSum || *itr == DmxEG || *itr == EG ||
          *itr == DmxTau || *itr == Tau) {
        double etmax = 99.5;
        if (*itr == Jet || *itr == DmxJet || *itr == Sum || *itr == DmxSum)
          etmax = 499.5;

        dirs_.insert(std::pair<ObjectType, TFileDirectory>(*itr, fs->mkdir(*str)));

        het_.insert(std::pair<ObjectType, TH1F*>(*itr, dirs_.at(*itr).make<TH1F>("et", "", 500, -0.5, etmax)));

        hbx_.insert(std::pair<ObjectType, TH1F*>(*itr, dirs_.at(*itr).make<TH1F>("bx", "", 11, -5.5, 5.5)));

        heta_.insert(std::pair<ObjectType, TH1F*>(*itr, dirs_.at(*itr).make<TH1F>("eta", "", 229, -114.5, 114.5)));
        hphi_.insert(std::pair<ObjectType, TH1F*>(*itr, dirs_.at(*itr).make<TH1F>("phi", "", 144, -0.5, 143.5)));
        hetaphi_.insert(std::pair<ObjectType, TH2F*>(
            *itr, dirs_.at(*itr).make<TH2F>("etaphi", "", 229, -114.5, 114.5, 144, -0.5, 143.5)));
      } else if (*itr == MU || *itr == gMU) {
        double etmax = 511.5;
        dirs_.insert(std::pair<ObjectType, TFileDirectory>(*itr, fs->mkdir(*str)));

        het_.insert(std::pair<ObjectType, TH1F*>(*itr, dirs_.at(*itr).make<TH1F>("et", "", 512, -0.5, etmax)));

        hbx_.insert(std::pair<ObjectType, TH1F*>(*itr, dirs_.at(*itr).make<TH1F>("bx", "", 11, -5.5, 5.5)));

        heta_.insert(std::pair<ObjectType, TH1F*>(*itr, dirs_.at(*itr).make<TH1F>("eta", "", 549, -224.5, 224.5)));
        hphi_.insert(std::pair<ObjectType, TH1F*>(*itr, dirs_.at(*itr).make<TH1F>("phi", "", 576, -0.5, 575.5)));
        hetaphi_.insert(std::pair<ObjectType, TH2F*>(
            *itr, dirs_.at(*itr).make<TH2F>("etaphi", "", 549, -224.5, 224.5, 576, -0.5, 575.5)));
      }
    }

    algDir_ = fs->mkdir("Algorithms");
    hAlgoBits_ = algDir_.make<TH1F>("hAlgoBits", "Algorithm Bits", 100, -0.5, 99.5);
    hEmulGtAlgoBits_ = algDir_.make<TH1F>("hEmulGtAlgoBits", "GT Emulated Algorithm Bits", 100, -0.5, 99.5);
    hAlgoBitsEmulGtVsHw_ = algDir_.make<TH2F>(
        "hAlgoBitsEmulGtVsHw", "Algorithm Bits (GT) Emulation vs Hardware", 129, -1.5, 127.5, 129, -1.5, 127.5);
    hEmulDxAlgoBits_ = algDir_.make<TH1F>("hEmulDxAlgoBits", "Dx Emulated Algorithm Bits", 100, -0.5, 99.5);
    hAlgoBitsEmulDxVsHw_ = algDir_.make<TH2F>(
        "hAlgoBitsEmulDxVsHw", "Algorithm Bits (Dx) Emulation vs Hardware", 129, -1.5, 127.5, 129, -1.5, 127.5);

    dmxVGtDir_ = fs->mkdir("SourceVsGT");

    hGmtVsGTMUEt_ =
        dmxVGtDir_.make<TH2F>("hGmtVsGTMUEt", "Gmt MU Et versus GT MU Et", 512, -0.5, 511.5, 512, -0.5, 511.5);
    hGmtVsGTMUEta_ =
        dmxVGtDir_.make<TH2F>("hGmtVsGTMUEta", "Gmt MU Eta versus GT MU Eta", 549, -224.5, 224.5, 549, -224.5, 224.5);
    hGmtVsGTMUPhi_ =
        dmxVGtDir_.make<TH2F>("hGmtVsGTMUPhi", "Gmt MU Phi versus GT MU Phi", 576, -0.5, 575.5, 576, -0.5, 575.5);

    hDmxVsGTEGEt_ =
        dmxVGtDir_.make<TH2F>("hDmxVsGTEGEt", "Dmx EG Et versus GT EG Et", 500, -0.5, 499.5, 500, -0.5, 499.5);
    hDmxVsGTEGEta_ =
        dmxVGtDir_.make<TH2F>("hDmxVsGTEGEta", "Dmx EG Eta versus GT EG Eta", 229, -114.5, 114.5, 229, -114.5, 114.5);
    hDmxVsGTEGPhi_ =
        dmxVGtDir_.make<TH2F>("hDmxVsGTEGPhi", "Dmx EG Phi versus GT EG Phi", 144, -0.5, 143.5, 144, -0.5, 143.5);

    hDmxVsGTTauEt_ =
        dmxVGtDir_.make<TH2F>("hDmxVsGTTauEt", "Dmx Tau Et versus GT Tau Et", 500, -0.5, 499.5, 500, -0.5, 499.5);
    hDmxVsGTTauEta_ = dmxVGtDir_.make<TH2F>(
        "hDmxVsGTTauEta", "Dmx Tau Eta versus GT Tau Eta", 229, -114.5, 114.5, 229, -114.5, 114.5);
    hDmxVsGTTauPhi_ =
        dmxVGtDir_.make<TH2F>("hDmxVsGTTauPhi", "Dmx Tau Phi versus GT Tau Phi", 144, -0.5, 143.5, 144, -0.5, 143.5);

    hDmxVsGTJetEt_ =
        dmxVGtDir_.make<TH2F>("hDmxVsGTJetEt", "Dmx Jet Et versus GT Jet Et", 500, -0.5, 499.5, 500, -0.5, 499.5);
    hDmxVsGTJetEta_ = dmxVGtDir_.make<TH2F>(
        "hDmxVsGTJetEta", "Dmx Jet Eta versus GT Jet Eta", 229, -114.5, 114.5, 229, -114.5, 114.5);
    hDmxVsGTJetPhi_ =
        dmxVGtDir_.make<TH2F>("hDmxVsGTJetPhi", "Dmx Jet Phi versus GT Jet Phi", 144, -0.5, 143.5, 144, -0.5, 143.5);

    hDmxVsGTSumEt_ETT_ =
        dmxVGtDir_.make<TH2F>("hDmxVsGTSumEt_ETT", "Dmx ETT versus GT ETT", 256, -0.5, 2047.5, 256, -0.5, 2047.5);
    hDmxVsGTSumEt_ETTem_ =
        dmxVGtDir_.make<TH2F>("hDmxVsGTSumEt_ETTem", "Dmx ETTem versus GT ETTem", 256, -0.5, 2047.5, 256, -0.5, 2047.5);
    hDmxVsGTSumEt_HTT_ =
        dmxVGtDir_.make<TH2F>("hDmxVsGTSumEt_HTT", "Dmx HTT versus GT HTT", 256, -0.5, 2047.5, 256, -0.5, 2047.5);
    hDmxVsGTSumEt_ETM_ =
        dmxVGtDir_.make<TH2F>("hDmxVsGTSumEt_ETM", "Dmx ETM versus GT ETM", 500, -0.5, 499.5, 500, -0.5, 499.5);
    hDmxVsGTSumPhi_ETM_ = dmxVGtDir_.make<TH2F>(
        "hDmxVsGTSumPhi_ETM", "Dmx ETM Phi versus GT ETM Phi", 144, -0.5, 143.5, 144, -0.5, 143.5);
    hDmxVsGTSumEt_ETMHF_ =
        dmxVGtDir_.make<TH2F>("hDmxVsGTSumEt_ETMHF", "Dmx ETMHF versus GT ETMHF", 500, -0.5, 499.5, 500, -0.5, 499.5);
    hDmxVsGTSumPhi_ETMHF_ = dmxVGtDir_.make<TH2F>(
        "hDmxVsGTSumPhi_ETMHF", "Dmx ETMHF Phi versus GT ETMHF Phi", 144, -0.5, 143.5, 144, -0.5, 143.5);
    hDmxVsGTSumEt_HTM_ =
        dmxVGtDir_.make<TH2F>("hDmxVsGTSumEt_HTM", "Dmx HTM versus GT HTM", 500, -0.5, 499.5, 500, -0.5, 499.5);
    hDmxVsGTSumPhi_HTM_ = dmxVGtDir_.make<TH2F>(
        "hDmxVsGTSumPhi_HTM", "Dmx HTM Phi versus GT HTM Phi", 144, -0.5, 143.5, 144, -0.5, 143.5);

    hDmxVsGTSumEt_HFP0_ =
        dmxVGtDir_.make<TH2F>("hDmxVsGTSumEt_HFP0", "Dmx versus GT HFP0", 16, -0.5, 15.5, 16, -0.5, 15.5);
    hDmxVsGTSumEt_HFM0_ =
        dmxVGtDir_.make<TH2F>("hDmxVsGTSumEt_HFM0", "Dmx versus GT HFM0", 16, -0.5, 15.5, 16, -0.5, 15.5);
    hDmxVsGTSumEt_HFP1_ =
        dmxVGtDir_.make<TH2F>("hDmxVsGTSumEt_HFP1", "Dmx versus GT HFP1", 16, -0.5, 15.5, 16, -0.5, 15.5);
    hDmxVsGTSumEt_HFM1_ =
        dmxVGtDir_.make<TH2F>("hDmxVsGTSumEt_HFM1", "Dmx versus GT HFM1", 16, -0.5, 15.5, 16, -0.5, 15.5);
  }

  // ------------ method called once each job just after ending the event loop  ------------
  void L1TGlobalAnalyzer::endJob() {}

  // ------------ method called when starting to processes a run  ------------
  /*
void 
L1TGlobalAnalyzer::beginRun(edm::Run const&, edm::EventSetup const&)
{
}
*/

  // ------------ method called when ending the processing of a run  ------------
  /*
void 
L1TGlobalAnalyzer::endRun(edm::Run const&, edm::EventSetup const&)
{
}
*/

  // ------------ method called when starting to processes a luminosity block  ------------
  /*
void 
L1TGlobalAnalyzer::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}
*/

  // ------------ method called when ending the processing of a luminosity block  ------------
  /*
void 
L1TGlobalAnalyzer::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}
*/

  // ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
  void L1TGlobalAnalyzer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    //The following says we do not know what parameters are allowed so do no validation
    // Please change this to state exactly what you do use, even if it is no parameters
    edm::ParameterSetDescription desc;
    desc.setUnknown();
    descriptions.addDefault(desc);
  }

}  // namespace l1t

using namespace l1t;

//define this as a plug-in
DEFINE_FWK_MODULE(L1TGlobalAnalyzer);
