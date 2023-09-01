/**
 * \class GlobalBoard
 *
 *
 * Description: Global Trigger Logic board, see header file for details.
 *
 * Implementation:
 *    Class responsible for receiving the objects from the different subsystems 
 *    and for running the Global Trigger Logic (GTL) and Final Decision Logic (FDL).
 *    It is directly called by the L1TGlobalProducer.
 *    The emulator considers a single board for all algorithms. 
 *
 * \author: M. Fierro                    - HEPHY Vienna - ORCA version
 * \author: V. M. Ghete                  - HEPHY Vienna - CMSSW version
 * \author: V. Rekovic                   - add correlation with overlap removal cases
 *                                       - fractional prescales
 * \author: E. Fontanesi                 - extended for three-body correlation conditions
 *
 * \author: E. Fontanesi, E. Yigitbasi, A. Loeliger (original implementation by S. Dildick, 2021)   
 *                                       - fix for the muon shower triggers and check on all BXs
 * \author: E. Fontanesi                 - added 2Loose HMT for 2023 Run 3
 *                                       - added ZDC triggers for 2023 HI data-taking
 * $Date$
 * $Revision$
 *
 */

// this class header
#include "L1Trigger/L1TGlobal/interface/GlobalBoard.h"

// user include files
#include "DataFormats/L1TGlobal/interface/GlobalObjectMap.h"
#include "L1Trigger/L1TGlobal/interface/TriggerMenu.h"
#include "L1Trigger/L1TGlobal/interface/GlobalAlgorithm.h"
#include "L1Trigger/L1TGlobal/interface/MuonTemplate.h"
#include "L1Trigger/L1TGlobal/interface/MuonShowerTemplate.h"
#include "L1Trigger/L1TGlobal/interface/CaloTemplate.h"
#include "L1Trigger/L1TGlobal/interface/EnergySumTemplate.h"
#include "L1Trigger/L1TGlobal/interface/EnergySumZdcTemplate.h"
#include "L1Trigger/L1TGlobal/interface/ExternalTemplate.h"
#include "L1Trigger/L1TGlobal/interface/CorrelationTemplate.h"
#include "L1Trigger/L1TGlobal/interface/CorrelationThreeBodyTemplate.h"
#include "L1Trigger/L1TGlobal/interface/CorrelationWithOverlapRemovalTemplate.h"
#include "L1Trigger/L1TGlobal/interface/GlobalCondition.h"
#include "L1Trigger/L1TGlobal/interface/CorrCondition.h"
#include "L1Trigger/L1TGlobal/interface/CorrWithOverlapRemovalCondition.h"
#include "L1Trigger/L1TGlobal/interface/ConditionEvaluation.h"
#include "L1Trigger/L1TGlobal/interface/AlgorithmEvaluation.h"

// Conditions for uGT
#include "L1Trigger/L1TGlobal/interface/MuCondition.h"
#include "L1Trigger/L1TGlobal/interface/MuonShowerCondition.h"
#include "L1Trigger/L1TGlobal/interface/CaloCondition.h"
#include "L1Trigger/L1TGlobal/interface/EnergySumCondition.h"
#include "L1Trigger/L1TGlobal/interface/EnergySumZdcCondition.h"
#include "L1Trigger/L1TGlobal/interface/ExternalCondition.h"
#include "L1Trigger/L1TGlobal/interface/CorrCondition.h"
#include "L1Trigger/L1TGlobal/interface/CorrThreeBodyCondition.h"
#include "L1Trigger/L1TGlobal/interface/CorrWithOverlapRemovalCondition.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/MessageLogger/interface/MessageDrop.h"

// Constructor
l1t::GlobalBoard::GlobalBoard()
    : m_candL1Mu(new BXVector<const l1t::Muon*>),
      m_candL1MuShower(new BXVector<std::shared_ptr<l1t::MuonShower>>),
      m_candL1EG(new BXVector<const l1t::L1Candidate*>),
      m_candL1Tau(new BXVector<const l1t::L1Candidate*>),
      m_candL1Jet(new BXVector<const l1t::L1Candidate*>),
      m_candL1EtSum(new BXVector<const l1t::EtSum*>),
      m_candL1EtSumZdc(new BXVector<const l1t::EtSum*>),
      m_candL1External(new BXVector<const GlobalExtBlk*>),
      m_currentLumi(0),
      m_isDebugEnabled(edm::isDebugEnabled()) {
  m_uGtAlgBlk.reset();

  m_gtlAlgorithmOR.reset();
  m_gtlDecisionWord.reset();

  m_prescaleCounterAlgoTrig.clear();

  // Initialize cached IDs
  m_l1GtMenuCacheID = 0ULL;
  m_l1CaloGeometryCacheID = 0ULL;
  m_l1MuTriggerScalesCacheID = 0ULL;

  // Counter for number of events board sees
  m_boardEventCount = 0;

  // A single uGT GlobalBoard is taken into account in the emulator
  m_uGtBoardNumber = 0;
  m_uGtFinalBoard = true;
}

// Destructor
l1t::GlobalBoard::~GlobalBoard() {
  delete m_candL1Mu;
  delete m_candL1MuShower;
  delete m_candL1EG;
  delete m_candL1Tau;
  delete m_candL1Jet;
  delete m_candL1EtSum;
  delete m_candL1EtSumZdc;
  delete m_candL1External;
}

// Operations
void l1t::GlobalBoard::setBxFirst(int bx) { m_bxFirst_ = bx; }

void l1t::GlobalBoard::setBxLast(int bx) { m_bxLast_ = bx; }

void l1t::GlobalBoard::init(const int numberPhysTriggers,
                            const int nrL1Mu,
                            const int nrL1MuShower,
                            const int nrL1EG,
                            const int nrL1Tau,
                            const int nrL1Jet,
                            int bxFirst,
                            int bxLast) {
  setBxFirst(bxFirst);
  setBxLast(bxLast);

  m_candL1Mu->setBXRange(m_bxFirst_, m_bxLast_);
  m_candL1MuShower->setBXRange(m_bxFirst_, m_bxLast_);
  m_candL1EG->setBXRange(m_bxFirst_, m_bxLast_);
  m_candL1Tau->setBXRange(m_bxFirst_, m_bxLast_);
  m_candL1Jet->setBXRange(m_bxFirst_, m_bxLast_);
  m_candL1EtSum->setBXRange(m_bxFirst_, m_bxLast_);
  m_candL1EtSumZdc->setBXRange(m_bxFirst_, m_bxLast_);
  m_candL1External->setBXRange(m_bxFirst_, m_bxLast_);

  m_uGtAlgBlk.reset();

  LogDebug("L1TGlobal") << "\t Initializing Board with bxFirst = " << m_bxFirst_ << ", bxLast = " << m_bxLast_;
}

// receive data from Calorimeter
void l1t::GlobalBoard::receiveCaloObjectData(const edm::Event& iEvent,
                                             const edm::EDGetTokenT<BXVector<l1t::EGamma>>& egInputToken,
                                             const edm::EDGetTokenT<BXVector<l1t::Tau>>& tauInputToken,
                                             const edm::EDGetTokenT<BXVector<l1t::Jet>>& jetInputToken,
                                             const edm::EDGetTokenT<BXVector<l1t::EtSum>>& sumInputToken,
                                             const edm::EDGetTokenT<BXVector<l1t::EtSum>>& sumZdcInputToken,
                                             const bool receiveEG,
                                             const int nrL1EG,
                                             const bool receiveTau,
                                             const int nrL1Tau,
                                             const bool receiveJet,
                                             const int nrL1Jet,
                                             const bool receiveEtSums,
                                             const bool receiveEtSumsZdc) {
  if (m_verbosity) {
    LogDebug("L1TGlobal") << "\n**** Board receiving Calo Data ";
  }

  resetCalo();

  // get data from Calorimeter
  if (receiveEG) {
    edm::Handle<BXVector<l1t::EGamma>> egData;
    iEvent.getByToken(egInputToken, egData);

    if (!egData.isValid()) {
      if (m_verbosity) {
        edm::LogWarning("L1TGlobal") << "\nWarning: Input tag for the BXVector<l1t::EGamma> collection"
                                     << "\nrequested in configuration, but not found in the event.\n";
      }
    } else {
      // bx in EG data
      for (int i = egData->getFirstBX(); i <= egData->getLastBX(); ++i) {
        // Prevent from pushing back bx that is outside of allowed range
        if (i < m_bxFirst_ || i > m_bxLast_)
          continue;

        //Loop over EG in this bx
        int nObj = 0;
        for (std::vector<l1t::EGamma>::const_iterator eg = egData->begin(i); eg != egData->end(i); ++eg) {
          if (nObj < nrL1EG) {
            (*m_candL1EG).push_back(i, &(*eg));
          } else {
            edm::LogWarning("L1TGlobal") << " Too many EG (" << nObj << ") for uGT Configuration maxEG =" << nrL1EG;
          }
          LogDebug("L1TGlobal") << "EG  Pt " << eg->hwPt() << " Eta  " << eg->hwEta() << " Phi " << eg->hwPhi()
                                << "  Qual " << eg->hwQual() << "  Iso " << eg->hwIso();

          nObj++;
        }  //end loop over EG in bx
      }    //end loop over bx
    }      //end if over valid EG data
  }        //end if ReceiveEG data

  if (receiveTau) {
    edm::Handle<BXVector<l1t::Tau>> tauData;
    iEvent.getByToken(tauInputToken, tauData);

    if (!tauData.isValid()) {
      if (m_verbosity) {
        edm::LogWarning("L1TGlobal") << "\nWarning: Input tag for the BXVector<l1t::Tau> collection"
                                     << "\nrequested in configuration, but not found in the event.\n";
      }
    } else {
      // bx in tau data
      for (int i = tauData->getFirstBX(); i <= tauData->getLastBX(); ++i) {
        // Prevent from pushing back bx that is outside of allowed range
        if (i < m_bxFirst_ || i > m_bxLast_)
          continue;

        //Loop over tau in this bx
        int nObj = 0;
        for (std::vector<l1t::Tau>::const_iterator tau = tauData->begin(i); tau != tauData->end(i); ++tau) {
          if (nObj < nrL1Tau) {
            (*m_candL1Tau).push_back(i, &(*tau));
          } else {
            LogTrace("L1TGlobal") << " Too many Tau (" << nObj << ") for uGT Configuration maxTau =" << nrL1Tau;
          }

          LogDebug("L1TGlobal") << "tau  Pt " << tau->hwPt() << " Eta  " << tau->hwEta() << " Phi " << tau->hwPhi()
                                << "  Qual " << tau->hwQual() << "  Iso " << tau->hwIso();
          nObj++;

        }  //end loop over tau in bx
      }    //end loop over bx
    }      //end if over valid tau data
  }        //end if ReceiveTau data

  if (receiveJet) {
    edm::Handle<BXVector<l1t::Jet>> jetData;
    iEvent.getByToken(jetInputToken, jetData);

    if (!jetData.isValid()) {
      if (m_verbosity) {
        edm::LogWarning("L1TGlobal") << "\nWarning: Input tag for the BXVector<l1t::Jet> collection"
                                     << "\nrequested in configuration, but not found in the event.\n";
      }
    } else {
      // bx in jet data
      for (int i = jetData->getFirstBX(); i <= jetData->getLastBX(); ++i) {
        // Prevent from pushing back bx that is outside of allowed range
        if (i < m_bxFirst_ || i > m_bxLast_)
          continue;

        //Loop over jet in this bx
        int nObj = 0;
        for (std::vector<l1t::Jet>::const_iterator jet = jetData->begin(i); jet != jetData->end(i); ++jet) {
          if (nObj < nrL1Jet) {
            (*m_candL1Jet).push_back(i, &(*jet));
          } else {
            edm::LogWarning("L1TGlobal") << " Too many Jets (" << nObj << ") for uGT Configuration maxJet =" << nrL1Jet;
          }

          LogDebug("L1TGlobal") << "Jet  Pt " << jet->hwPt() << " Eta  " << jet->hwEta() << " Phi " << jet->hwPhi()
                                << "  Qual " << jet->hwQual() << "  Iso " << jet->hwIso();
          nObj++;
        }  //end loop over jet in bx
      }    //end loop over bx
    }      //end if over valid jet data
  }        //end if ReceiveJet data

  if (receiveEtSums) {
    edm::Handle<BXVector<l1t::EtSum>> etSumData;
    iEvent.getByToken(sumInputToken, etSumData);

    if (!etSumData.isValid()) {
      if (m_verbosity) {
        edm::LogWarning("L1TGlobal") << "\nWarning: Input tag for the BXVector<l1t::EtSum> collection"
                                     << "\nrequested in configuration, but not found in the event.\n";
      }
    } else {
      for (int i = etSumData->getFirstBX(); i <= etSumData->getLastBX(); ++i) {
        // Prevent from pushing back bx that is outside of allowed range
        if (i < m_bxFirst_ || i > m_bxLast_)
          continue;

        //Loop over EtSum objects in this bx
        for (std::vector<l1t::EtSum>::const_iterator etsum = etSumData->begin(i); etsum != etSumData->end(i); ++etsum) {
          (*m_candL1EtSum).push_back(i, &(*etsum));

          /*  In case we need to split these out
	          switch ( etsum->getType() ) {
		     case l1t::EtSum::EtSumType::kMissingEt:
		       {
			 //(*m_candETM).push_back(i,&(*etsum));
			 LogDebug("L1TGlobal") << "ETM:  Pt " << etsum->hwPt() <<  " Phi " << etsum->hwPhi();
		       }
		       break; 
		     case l1t::EtSum::EtSumType::kMissingHt:
		       {
			 //(*m_candHTM).push_back(i,&(*etsum));
			 LogDebug("L1TGlobal") << "HTM:  Pt " << etsum->hwPt() <<  " Phi " << etsum->hwPhi();
		       }
		       break; 		     
		     case l1t::EtSum::EtSumType::kTotalEt:
		       {
			 //(*m_candETT).push_back(i,&(*etsum));
			 LogDebug("L1TGlobal") << "ETT:  Pt " << etsum->hwPt();
		       }
		       break; 		     
		     case l1t::EtSum::EtSumType::kTotalHt:
		       {
			 //(*m_candHTT).push_back(i,&(*etsum));
			 LogDebug("L1TGlobal") << "HTT:  Pt " << etsum->hwPt();
		       }
		       break;
		     case l1t::EtSum::EtSumType::kTowerCount:
		       {
			 //(*m_candTowerCount).push_back(i,&(*etsum));
			 LogDebug("L1TGlobal") << "TowerCount: " << etsum->hwPt();
		       }
		       break;
		     default:
		       LogDebug("L1TGlobal") << "Default encounted ";
		       break;
		  }
*/

        }  //end loop over EtSum objects in bx
      }    //end loop over Bx
    }
  }

  if (receiveEtSumsZdc) {
    edm::Handle<BXVector<l1t::EtSum>> etSumData;
    iEvent.getByToken(sumZdcInputToken, etSumData);

    if (!etSumData.isValid()) {
      if (m_verbosity) {
        edm::LogWarning("L1TGlobal") << "\nWarning: Input tag for the ZDC Energy Sums collection"
                                     << "\nrequested in configuration, but not found in the event.\n";
      }
    } else {
      for (int i = etSumData->getFirstBX(); i <= etSumData->getLastBX(); ++i) {
        // Prevent from pushing back bx that is outside of allowed range
        if (i < m_bxFirst_ || i > m_bxLast_)
          continue;

        for (std::vector<l1t::EtSum>::const_iterator etsum = etSumData->begin(i); etsum != etSumData->end(i); ++etsum) {
          (*m_candL1EtSumZdc).push_back(i, &(*etsum));
        }
      }  //end loop over Bx
    }
  }
}

// receive data from Global Muon Trigger
void l1t::GlobalBoard::receiveMuonObjectData(const edm::Event& iEvent,
                                             const edm::EDGetTokenT<BXVector<l1t::Muon>>& muInputToken,
                                             const bool receiveMu,
                                             const int nrL1Mu) {
  if (m_verbosity) {
    LogDebug("L1TGlobal") << "\n**** GlobalBoard receiving muon data = ";
    //<< "\n     from input tag " << muInputTag << "\n"
  }

  resetMu();

  // get data from Global Muon Trigger
  if (receiveMu) {
    edm::Handle<BXVector<l1t::Muon>> muonData;
    iEvent.getByToken(muInputToken, muonData);

    if (!muonData.isValid()) {
      if (m_verbosity) {
        edm::LogWarning("L1TGlobal") << "\nWarning: Input tag for the BXVector<l1t::Muon> collection"
                                     << "\nrequested in configuration, but not found in the event.\n";
      }
    } else {
      // bx in muon data
      for (int i = muonData->getFirstBX(); i <= muonData->getLastBX(); ++i) {
        // Prevent from pushing back bx that is outside of allowed range
        if (i < m_bxFirst_ || i > m_bxLast_)
          continue;

        //Loop over Muons in this bx
        int nObj = 0;
        for (std::vector<l1t::Muon>::const_iterator mu = muonData->begin(i); mu != muonData->end(i); ++mu) {
          if (nObj < nrL1Mu) {
            (*m_candL1Mu).push_back(i, &(*mu));
          } else {
            edm::LogWarning("L1TGlobal") << " Too many Muons (" << nObj << ") for uGT Configuration maxMu =" << nrL1Mu;
          }

          LogDebug("L1TGlobal") << "Muon  Pt " << mu->hwPt() << " EtaAtVtx  " << mu->hwEtaAtVtx() << " PhiAtVtx "
                                << mu->hwPhiAtVtx() << "  Qual " << mu->hwQual() << "  Iso " << mu->hwIso();
          nObj++;
        }  //end loop over muons in bx
      }    //end loop over bx
    }      //end if over valid muon data
  }        //end if ReceiveMuon data
}

// receive muon shower data from Global Muon Trigger
void l1t::GlobalBoard::receiveMuonShowerObjectData(const edm::Event& iEvent,
                                                   const edm::EDGetTokenT<BXVector<l1t::MuonShower>>& muShowerInputToken,
                                                   const bool receiveMuShower,
                                                   const int nrL1MuShower) {
  // get data from Global Muon Trigger
  if (receiveMuShower) {
    edm::Handle<BXVector<l1t::MuonShower>> muonData;
    iEvent.getByToken(muShowerInputToken, muonData);

    if (!muonData.isValid()) {
      if (m_verbosity) {
        edm::LogWarning("L1TGlobal") << "\nWarning: Input tag for the BXVector<l1t::MuonShower> collection"
                                     << "\nrequested in configuration, but not found in the event.\n";
      }
    } else {
      // Loop over bx in muon data
      for (int i = muonData->getFirstBX(); i <= muonData->getLastBX(); ++i) {
        // Prevent from pushing back bx that is outside of allowed range
        if (i < m_bxFirst_ || i > m_bxLast_)
          continue;

        // Loop over Muon Showers in this bx
        int nObj = 0;
        for (std::vector<l1t::MuonShower>::const_iterator mu = muonData->begin(i); mu != muonData->end(i); ++mu) {
          if (nObj < nrL1MuShower) {
            /* NOTE: here the single object is split up into 5 separate MuonShower objects 
	       similarly to the description in the UTM library, where the conditions are four different objects.
	    */

            std::shared_ptr<l1t::MuonShower> musOneNominalInTime =
                std::make_shared<l1t::MuonShower>(false, false, false, false, false, false);
            std::shared_ptr<l1t::MuonShower> musOneTightInTime =
                std::make_shared<l1t::MuonShower>(false, false, false, false, false, false);
            std::shared_ptr<l1t::MuonShower> musTwoLooseDiffSectorsInTime =
                std::make_shared<l1t::MuonShower>(false, false, false, false, false, false);
            std::shared_ptr<l1t::MuonShower> musOutOfTime0 =
                std::make_shared<l1t::MuonShower>(false, false, false, false, false, false);
            std::shared_ptr<l1t::MuonShower> musOutOfTime1 =
                std::make_shared<l1t::MuonShower>(false, false, false, false, false, false);

            musOneNominalInTime->setOneNominalInTime(mu->isOneNominalInTime());
            musOneTightInTime->setOneTightInTime(mu->isOneTightInTime());
            musTwoLooseDiffSectorsInTime->setTwoLooseDiffSectorsInTime(mu->isTwoLooseDiffSectorsInTime());
            musOutOfTime0->setMusOutOfTime0(mu->musOutOfTime0());
            musOutOfTime1->setMusOutOfTime1(mu->musOutOfTime1());

            (*m_candL1MuShower).push_back(i, musOneNominalInTime);
            (*m_candL1MuShower).push_back(i, musOneTightInTime);
            (*m_candL1MuShower).push_back(i, musTwoLooseDiffSectorsInTime);
            (*m_candL1MuShower).push_back(i, musOutOfTime0);
            (*m_candL1MuShower).push_back(i, musOutOfTime1);

          } else {
            edm::LogWarning("L1TGlobal") << " Too many Muon Showers (" << nObj
                                         << ") for uGT Configuration maxMuShower =" << nrL1MuShower;
          }
          nObj++;
        }  //end loop over muon showers in bx
      }    //end loop over bx
    }      //end if over valid muon shower data
  }        //end if ReceiveMuonShower data
}

// receive data from Global External Conditions
void l1t::GlobalBoard::receiveExternalData(const edm::Event& iEvent,
                                           const edm::EDGetTokenT<BXVector<GlobalExtBlk>>& extInputToken,
                                           const bool receiveExt) {
  if (m_verbosity) {
    LogDebug("L1TGlobal") << "\n**** GlobalBoard receiving external data = ";
    //<< "\n     from input tag " << muInputTag << "\n"
  }

  resetExternal();

  // get data from Global Muon Trigger
  if (receiveExt) {
    edm::Handle<BXVector<GlobalExtBlk>> extData;
    iEvent.getByToken(extInputToken, extData);

    if (!extData.isValid()) {
      if (m_verbosity) {
        edm::LogWarning("L1TGlobal") << "\nWarning: Input tag for the BXVector<GlobalExtBlk> collection"
                                     << "\nrequested in configuration, but not found in the event.\n";
      }
    } else {
      // bx in muon data
      for (int i = extData->getFirstBX(); i <= extData->getLastBX(); ++i) {
        // Prevent from pushing back bx that is outside of allowed range
        if (i < m_bxFirst_ || i > m_bxLast_)
          continue;

        //Loop over ext in this bx
        for (std::vector<GlobalExtBlk>::const_iterator ext = extData->begin(i); ext != extData->end(i); ++ext) {
          (*m_candL1External).push_back(i, &(*ext));
        }  //end loop over ext in bx
      }    //end loop over bx
    }      //end if over valid ext data
  }        //end if ReceiveExt data
}

// run GTL
void l1t::GlobalBoard::runGTL(const edm::Event&,
                              const edm::EventSetup& evSetup,
                              const TriggerMenu* m_l1GtMenu,
                              const bool produceL1GtObjectMapRecord,
                              const int iBxInEvent,
                              std::unique_ptr<GlobalObjectMapRecord>& gtObjectMapRecord,
                              const unsigned int numberPhysTriggers,
                              const int nrL1Mu,
                              const int nrL1MuShower,
                              const int nrL1EG,
                              const int nrL1Tau,
                              const int nrL1Jet) {
  const std::vector<ConditionMap>& conditionMap = m_l1GtMenu->gtConditionMap();
  const AlgorithmMap& algorithmMap = m_l1GtMenu->gtAlgorithmMap();
  const GlobalScales& gtScales = m_l1GtMenu->gtScales();
  const std::string scaleSetName = gtScales.getScalesName();
  LogDebug("L1TGlobal") << " L1 Menu Scales -- Set Name: " << scaleSetName;

  // Reset AlgBlk for this bx
  m_uGtAlgBlk.reset();
  m_algInitialOr = false;
  m_algPrescaledOr = false;
  m_algIntermOr = false;
  m_algFinalOr = false;
  m_algFinalOrVeto = false;

  const std::vector<std::vector<MuonTemplate>>& corrMuon = m_l1GtMenu->corMuonTemplate();

  const std::vector<std::vector<CaloTemplate>>& corrCalo = m_l1GtMenu->corCaloTemplate();

  const std::vector<std::vector<EnergySumTemplate>>& corrEnergySum = m_l1GtMenu->corEnergySumTemplate();

  LogDebug("L1TGlobal") << "Size corrMuon " << corrMuon.size() << "\nSize corrCalo " << corrCalo.size()
                        << "\nSize corrSums " << corrEnergySum.size();

  // -----------------------------------------------------
  // Loop over condition maps (one map per condition chip),
  // then loop over conditions in the map and
  // save the results in temporary maps
  // -----------------------------------------------------
  // never happens in production but at first event...
  if (m_conditionResultMaps.size() != conditionMap.size()) {
    m_conditionResultMaps.clear();
    m_conditionResultMaps.resize(conditionMap.size());
  }

  int iChip = -1;

  for (std::vector<ConditionMap>::const_iterator itCondOnChip = conditionMap.begin();
       itCondOnChip != conditionMap.end();
       itCondOnChip++) {
    iChip++;

    AlgorithmEvaluation::ConditionEvaluationMap& cMapResults = m_conditionResultMaps[iChip];

    for (CItCond itCond = itCondOnChip->begin(); itCond != itCondOnChip->end(); itCond++) {
      // evaluate condition
      switch ((itCond->second)->condCategory()) {
        case CondMuon: {
          // BLW Not sure what to do with this for now
          const int ifMuEtaNumberBits = 0;

          MuCondition* muCondition = new MuCondition(itCond->second, this, nrL1Mu, ifMuEtaNumberBits);

          muCondition->setVerbosity(m_verbosity);

          muCondition->evaluateConditionStoreResult(iBxInEvent);

          cMapResults[itCond->first] = muCondition;

          if (m_verbosity && m_isDebugEnabled) {
            std::ostringstream myCout;
            muCondition->print(myCout);

            LogTrace("L1TGlobal") << myCout.str();
          }
          //delete muCondition;

        } break;
        case CondMuonShower: {
          MuonShowerCondition* muShowerCondition = new MuonShowerCondition(itCond->second, this, nrL1MuShower);

          muShowerCondition->setVerbosity(m_verbosity);

          muShowerCondition->evaluateConditionStoreResult(iBxInEvent);

          cMapResults[itCond->first] = muShowerCondition;

          if (m_verbosity && m_isDebugEnabled) {
            std::ostringstream myCout;
            muShowerCondition->print(myCout);

            edm::LogWarning("L1TGlobal") << "MuonShowerCondition " << myCout.str();
          }
          //delete muShowerCondition;

        } break;
        case CondCalo: {
          // BLW Not sure w hat to do with this for now
          const int ifCaloEtaNumberBits = 0;

          CaloCondition* caloCondition =
              new CaloCondition(itCond->second, this, nrL1EG, nrL1Jet, nrL1Tau, ifCaloEtaNumberBits);

          caloCondition->setVerbosity(m_verbosity);

          caloCondition->evaluateConditionStoreResult(iBxInEvent);

          cMapResults[itCond->first] = caloCondition;

          if (m_verbosity && m_isDebugEnabled) {
            std::ostringstream myCout;
            caloCondition->print(myCout);

            LogTrace("L1TGlobal") << myCout.str();
          }
          //                    delete caloCondition;

        } break;
        case CondEnergySum: {
          EnergySumCondition* eSumCondition = new EnergySumCondition(itCond->second, this);

          eSumCondition->setVerbosity(m_verbosity);
          eSumCondition->evaluateConditionStoreResult(iBxInEvent);

          cMapResults[itCond->first] = eSumCondition;

          if (m_verbosity && m_isDebugEnabled) {
            std::ostringstream myCout;
            eSumCondition->print(myCout);

            LogTrace("L1TGlobal") << myCout.str();
          }
          //                    delete eSumCondition;

        } break;
        case CondEnergySumZdc: {
          EnergySumZdcCondition* eSumZdcCondition = new EnergySumZdcCondition(itCond->second, this);

          eSumZdcCondition->setVerbosity(m_verbosity);
          eSumZdcCondition->evaluateConditionStoreResult(iBxInEvent);

          cMapResults[itCond->first] = eSumZdcCondition;

          if (m_verbosity && m_isDebugEnabled) {
            std::ostringstream myCout;
            eSumZdcCondition->print(myCout);

            LogTrace("L1TGlobal") << myCout.str();
          }
          //                    delete eSumZdcCondition;

        } break;

        case CondExternal: {
          ExternalCondition* extCondition = new ExternalCondition(itCond->second, this);

          extCondition->setVerbosity(m_verbosity);
          extCondition->evaluateConditionStoreResult(iBxInEvent);

          cMapResults[itCond->first] = extCondition;

          if (m_verbosity && m_isDebugEnabled) {
            std::ostringstream myCout;
            extCondition->print(myCout);

            LogTrace("L1TGlobal") << myCout.str();
          }
          //                    delete extCondition;

        } break;
        case CondCorrelation: {
          // get first the subconditions
          const CorrelationTemplate* corrTemplate = static_cast<const CorrelationTemplate*>(itCond->second);
          const GtConditionCategory cond0Categ = corrTemplate->cond0Category();
          const GtConditionCategory cond1Categ = corrTemplate->cond1Category();
          const int cond0Ind = corrTemplate->cond0Index();
          const int cond1Ind = corrTemplate->cond1Index();

          const GlobalCondition* cond0Condition = nullptr;
          const GlobalCondition* cond1Condition = nullptr;

          // maximum number of objects received for evaluation of l1t::Type1s condition
          int cond0NrL1Objects = 0;
          int cond1NrL1Objects = 0;
          LogDebug("L1TGlobal") << " cond0NrL1Objects" << cond0NrL1Objects << "  cond1NrL1Objects  "
                                << cond1NrL1Objects;

          switch (cond0Categ) {
            case CondMuon: {
              cond0Condition = &((corrMuon[iChip])[cond0Ind]);
            } break;
            case CondCalo: {
              cond0Condition = &((corrCalo[iChip])[cond0Ind]);
            } break;
            case CondEnergySum: {
              cond0Condition = &((corrEnergySum[iChip])[cond0Ind]);
            } break;
            default: {
              // do nothing, should not arrive here
            } break;
          }

          switch (cond1Categ) {
            case CondMuon: {
              cond1Condition = &((corrMuon[iChip])[cond1Ind]);
            } break;
            case CondCalo: {
              cond1Condition = &((corrCalo[iChip])[cond1Ind]);
            } break;
            case CondEnergySum: {
              cond1Condition = &((corrEnergySum[iChip])[cond1Ind]);
            } break;
            default: {
              // do nothing, should not arrive here
            } break;
          }

          CorrCondition* correlationCond = new CorrCondition(itCond->second, cond0Condition, cond1Condition, this);

          correlationCond->setVerbosity(m_verbosity);
          correlationCond->setScales(&gtScales);
          correlationCond->evaluateConditionStoreResult(iBxInEvent);

          cMapResults[itCond->first] = correlationCond;

          if (m_verbosity && m_isDebugEnabled) {
            std::ostringstream myCout;
            correlationCond->print(myCout);

            LogTrace("L1TGlobal") << myCout.str();
          }

          //  		delete correlationCond;

        } break;
        case CondCorrelationThreeBody: {
          // get first the subconditions
          const CorrelationThreeBodyTemplate* corrTemplate =
              static_cast<const CorrelationThreeBodyTemplate*>(itCond->second);
          const GtConditionCategory cond0Categ = corrTemplate->cond0Category();
          const GtConditionCategory cond1Categ = corrTemplate->cond1Category();
          const GtConditionCategory cond2Categ = corrTemplate->cond2Category();
          const int cond0Ind = corrTemplate->cond0Index();
          const int cond1Ind = corrTemplate->cond1Index();
          const int cond2Ind = corrTemplate->cond2Index();

          const GlobalCondition* cond0Condition = nullptr;
          const GlobalCondition* cond1Condition = nullptr;
          const GlobalCondition* cond2Condition = nullptr;

          // maximum number of objects received for evaluation of l1t::Type1s condition
          int cond0NrL1Objects = 0;
          int cond1NrL1Objects = 0;
          int cond2NrL1Objects = 0;
          LogDebug("L1TGlobal") << "  cond0NrL1Objects  " << cond0NrL1Objects << "  cond1NrL1Objects  "
                                << cond1NrL1Objects << "  cond2NrL1Objects  " << cond2NrL1Objects;
          if (cond0Categ == CondMuon) {
            cond0Condition = &((corrMuon[iChip])[cond0Ind]);
          } else {
            LogDebug("L1TGlobal") << "No muon0 to evaluate three-body correlation condition";
          }
          if (cond1Categ == CondMuon) {
            cond1Condition = &((corrMuon[iChip])[cond1Ind]);
          } else {
            LogDebug("L1TGlobal") << "No muon1 to evaluate three-body correlation condition";
          }
          if (cond2Categ == CondMuon) {
            cond2Condition = &((corrMuon[iChip])[cond2Ind]);
          } else {
            LogDebug("L1TGlobal") << "No muon2 to evaluate three-body correlation condition";
          }

          CorrThreeBodyCondition* correlationThreeBodyCond =
              new CorrThreeBodyCondition(itCond->second, cond0Condition, cond1Condition, cond2Condition, this);

          correlationThreeBodyCond->setVerbosity(m_verbosity);
          correlationThreeBodyCond->setScales(&gtScales);
          correlationThreeBodyCond->evaluateConditionStoreResult(iBxInEvent);
          cMapResults[itCond->first] = correlationThreeBodyCond;

          if (m_verbosity && m_isDebugEnabled) {
            std::ostringstream myCout;
            correlationThreeBodyCond->print(myCout);

            LogTrace("L1TGlobal") << myCout.str();
          }
          //              delete correlationThreeBodyCond;
        } break;

        case CondCorrelationWithOverlapRemoval: {
          // get first the subconditions
          const CorrelationWithOverlapRemovalTemplate* corrTemplate =
              static_cast<const CorrelationWithOverlapRemovalTemplate*>(itCond->second);
          const GtConditionCategory cond0Categ = corrTemplate->cond0Category();
          const GtConditionCategory cond1Categ = corrTemplate->cond1Category();
          const GtConditionCategory cond2Categ = corrTemplate->cond2Category();
          const int cond0Ind = corrTemplate->cond0Index();
          const int cond1Ind = corrTemplate->cond1Index();
          const int cond2Ind = corrTemplate->cond2Index();

          const GlobalCondition* cond0Condition = nullptr;
          const GlobalCondition* cond1Condition = nullptr;
          const GlobalCondition* cond2Condition = nullptr;

          // maximum number of objects received for evaluation of l1t::Type1s condition
          int cond0NrL1Objects = 0;
          int cond1NrL1Objects = 0;
          int cond2NrL1Objects = 0;
          LogDebug("L1TGlobal") << " cond0NrL1Objects" << cond0NrL1Objects << "  cond1NrL1Objects  " << cond1NrL1Objects
                                << " cond2NrL1Objects  " << cond2NrL1Objects;

          switch (cond0Categ) {
            case CondMuon: {
              cond0Condition = &((corrMuon[iChip])[cond0Ind]);
            } break;
            case CondCalo: {
              cond0Condition = &((corrCalo[iChip])[cond0Ind]);
            } break;
            case CondEnergySum: {
              cond0Condition = &((corrEnergySum[iChip])[cond0Ind]);
            } break;
            default: {
              // do nothing, should not arrive here
            } break;
          }

          switch (cond1Categ) {
            case CondMuon: {
              cond1Condition = &((corrMuon[iChip])[cond1Ind]);
            } break;
            case CondCalo: {
              cond1Condition = &((corrCalo[iChip])[cond1Ind]);
            } break;
            case CondEnergySum: {
              cond1Condition = &((corrEnergySum[iChip])[cond1Ind]);
            } break;
            default: {
              // do nothing, should not arrive here
            } break;
          }

          switch (cond2Categ) {
            case CondMuon: {
              cond2Condition = &((corrMuon[iChip])[cond2Ind]);
            } break;
            case CondCalo: {
              cond2Condition = &((corrCalo[iChip])[cond2Ind]);
            } break;
            case CondEnergySum: {
              cond2Condition = &((corrEnergySum[iChip])[cond2Ind]);
            } break;
            default: {
              // do nothing, should not arrive here
            } break;
          }

          CorrWithOverlapRemovalCondition* correlationCondWOR =
              new CorrWithOverlapRemovalCondition(itCond->second, cond0Condition, cond1Condition, cond2Condition, this);

          correlationCondWOR->setVerbosity(m_verbosity);
          correlationCondWOR->setScales(&gtScales);
          correlationCondWOR->evaluateConditionStoreResult(iBxInEvent);

          cMapResults[itCond->first] = correlationCondWOR;

          if (m_verbosity && m_isDebugEnabled) {
            std::ostringstream myCout;
            correlationCondWOR->print(myCout);

            LogTrace("L1TGlobal") << myCout.str();
          }

          //  		delete correlationCondWOR;

        } break;
        case CondNull: {
          // do nothing

        } break;
        default: {
          // do nothing

        } break;
      }
    }
  }

  // -----------------------
  // Loop over algorithm map
  // -----------------------
  // Empty vector for object maps - filled during loop
  std::vector<GlobalObjectMap> objMapVec;
  if (produceL1GtObjectMapRecord && (iBxInEvent == 0))
    objMapVec.reserve(numberPhysTriggers);

  for (CItAlgo itAlgo = algorithmMap.begin(); itAlgo != algorithmMap.end(); itAlgo++) {
    AlgorithmEvaluation gtAlg(itAlgo->second);
    gtAlg.evaluateAlgorithm((itAlgo->second).algoChipNumber(), m_conditionResultMaps);

    int algBitNumber = (itAlgo->second).algoBitNumber();
    bool algResult = gtAlg.gtAlgoResult();

    LogDebug("L1TGlobal") << " ===> for iBxInEvent = " << iBxInEvent << ":\t algBitName = " << itAlgo->first
                          << ",\t algBitNumber = " << algBitNumber << ",\t algResult = " << algResult;

    if (algResult) {
      //            m_gtlAlgorithmOR.set(algBitNumber);
      m_uGtAlgBlk.setAlgoDecisionInitial(algBitNumber, algResult);
      m_algInitialOr = true;
    }

    if (m_verbosity && m_isDebugEnabled) {
      std::ostringstream myCout;
      (itAlgo->second).print(myCout);
      gtAlg.print(myCout);

      LogTrace("L1TGlobal") << myCout.str();
    }

    // object maps only for BxInEvent = 0
    if (produceL1GtObjectMapRecord && (iBxInEvent == 0)) {
      std::vector<L1TObjectTypeInCond> otypes;
      for (auto iop = gtAlg.operandTokenVector().begin(); iop != gtAlg.operandTokenVector().end(); ++iop) {
        //cout << "INFO:  operand name:  " << iop->tokenName << "\n";
        int found = 0;
        L1TObjectTypeInCond otype;
        for (auto imap = conditionMap.begin(); imap != conditionMap.end(); imap++) {
          auto match = imap->find(iop->tokenName);

          if (match != imap->end()) {
            found = 1;
            //cout << "DEBUG: found match for " << iop->tokenName << " at " << match->first << "\n";

            otype = match->second->objectType();

            for (auto itype = otype.begin(); itype != otype.end(); itype++) {
              //cout << "type:  " << *itype << "\n";
            }
          }
        }
        if (!found) {
          edm::LogWarning("L1TGlobal") << "\n Failed to find match for operand token " << iop->tokenName << "\n";
        } else {
          otypes.push_back(otype);
        }
      }

      // set object map
      GlobalObjectMap objMap;

      objMap.setAlgoName(itAlgo->first);
      objMap.setAlgoBitNumber(algBitNumber);
      objMap.setAlgoGtlResult(algResult);
      objMap.swapOperandTokenVector(gtAlg.operandTokenVector());
      objMap.swapCombinationVector(gtAlg.gtAlgoCombinationVector());
      // gtAlg is empty now...
      objMap.swapObjectTypeVector(otypes);

      if (m_verbosity && m_isDebugEnabled) {
        std::ostringstream myCout1;
        objMap.print(myCout1);

        LogTrace("L1TGlobal") << myCout1.str();
      }

      objMapVec.push_back(objMap);
    }
  }

  // object maps only for BxInEvent = 0
  if (produceL1GtObjectMapRecord && (iBxInEvent == 0)) {
    gtObjectMapRecord->swapGtObjectMap(objMapVec);
  }

  // loop over condition maps (one map per condition chip)
  // then loop over conditions in the map
  // delete the conditions created with new, zero pointer, do not clear map, keep the vector as is...
  for (std::vector<AlgorithmEvaluation::ConditionEvaluationMap>::iterator itCondOnChip = m_conditionResultMaps.begin();
       itCondOnChip != m_conditionResultMaps.end();
       itCondOnChip++) {
    for (AlgorithmEvaluation::ItEvalMap itCond = itCondOnChip->begin(); itCond != itCondOnChip->end(); itCond++) {
      delete itCond->second;
      itCond->second = nullptr;
    }
  }
}

// -------
// Run GTL
// -------
void l1t::GlobalBoard::runFDL(const edm::Event& iEvent,
                              const int iBxInEvent,
                              const int totalBxInEvent,
                              const unsigned int numberPhysTriggers,
                              const std::vector<double>& prescaleFactorsAlgoTrig,
                              const std::vector<unsigned int>& triggerMaskAlgoTrig,
                              const std::vector<int>& triggerMaskVetoAlgoTrig,
                              const bool algorithmTriggersUnprescaled,
                              const bool algorithmTriggersUnmasked) {
  if (m_verbosity) {
    LogDebug("L1TGlobal") << "\n**** GlobalBoard apply Final Decision Logic ";
  }

  // update and clear prescales at the beginning of the luminosity segment
  if (m_prescaleCounterAlgoTrig.empty() or
      (m_currentLumi != iEvent.luminosityBlock() and m_resetPSCountersEachLumiSec)) {
    m_prescaleCounterAlgoTrig.clear();
    m_prescaleCounterAlgoTrig.reserve(totalBxInEvent);
    auto const& prescaleCountersAlgoTrig =
        m_semiRandomInitialPSCounters ? prescaleCountersWithSemirandomInitialCounter(prescaleFactorsAlgoTrig, iEvent)
                                      : prescaleCounters(prescaleFactorsAlgoTrig);
    for (int iBxInEvent = 0; iBxInEvent < totalBxInEvent; ++iBxInEvent) {
      m_prescaleCounterAlgoTrig.push_back(prescaleCountersAlgoTrig);
    }

    m_currentLumi = iEvent.luminosityBlock();
  }

  // Copy Algorithm bits to Prescaled word
  // Prescaling and Masking done below if requested.
  m_uGtAlgBlk.copyInitialToInterm();

  // -------------------------------------------
  //      Apply Prescales or skip if turned off
  // -------------------------------------------
  if (!algorithmTriggersUnprescaled) {
    // iBxInEvent is ... -2 -1 0 1 2 ... while counters are 0 1 2 3 4 ...
    int const inBxInEvent = totalBxInEvent / 2 + iBxInEvent;

    bool temp_algPrescaledOr = false;
    bool alreadyReported = false;
    for (unsigned int iBit = 0; iBit < numberPhysTriggers; ++iBit) {
      bool const bitValue = m_uGtAlgBlk.getAlgoDecisionInitial(iBit);
      if (bitValue) {
        // Make sure algo bit in range, warn otherwise
        if (iBit < prescaleFactorsAlgoTrig.size()) {
          if (prescaleFactorsAlgoTrig.at(iBit) != 1) {
            bool const triggered = m_prescaleCounterAlgoTrig.at(inBxInEvent).at(iBit).accept();
            if (triggered) {
              temp_algPrescaledOr = true;
            } else {
              // change bit to false in prescaled word and final decision word
              m_uGtAlgBlk.setAlgoDecisionInterm(iBit, false);
            }  //if Prescale counter reached zero
          }    //if prescale factor is not 1 (ie. no prescale)
          else {
            temp_algPrescaledOr = true;
          }
        }  // require bit in range
        else if (!alreadyReported) {
          alreadyReported = true;
          edm::LogWarning("L1TGlobal") << "\nWarning: algoBit >= prescaleFactorsAlgoTrig.size() in bx " << iBxInEvent;
        }
      }  //if algo bit is set true
    }    //loop over alg bits

    m_algPrescaledOr = temp_algPrescaledOr;  //temp

  } else {
    // Since not Prescaling just take OR of Initial Work
    m_algPrescaledOr = m_algInitialOr;

  }  //if we are going to apply prescales.

  // Copy Algorithm bits fron Prescaled word to Final Word
  // Masking done below if requested.
  m_uGtAlgBlk.copyIntermToFinal();

  if (!algorithmTriggersUnmasked) {
    bool temp_algFinalOr = false;
    bool alreadyReported = false;
    for (unsigned int iBit = 0; iBit < numberPhysTriggers; ++iBit) {
      const bool bitValue = m_uGtAlgBlk.getAlgoDecisionInterm(iBit);

      if (bitValue) {
        //bool isMasked = ( triggerMaskAlgoTrig.at(iBit) == 0 );
        bool isMasked = false;
        if (iBit < triggerMaskAlgoTrig.size())
          isMasked = (triggerMaskAlgoTrig.at(iBit) == 0);
        else if (!alreadyReported) {
          alreadyReported = true;
          edm::LogWarning("L1TGlobal") << "\nWarning: algoBit >= triggerMaskAlgoTrig.size() in bx " << iBxInEvent;
        }

        bool const passMask = (bitValue && !isMasked);

        if (passMask)
          temp_algFinalOr = true;
        else
          m_uGtAlgBlk.setAlgoDecisionFinal(iBit, false);

        // Check if veto mask is true, if it is, set the event veto flag.
        if (triggerMaskVetoAlgoTrig.at(iBit) == 1)
          m_algFinalOrVeto = true;
      }
    }

    m_algIntermOr = temp_algFinalOr;

  } else {
    m_algIntermOr = m_algPrescaledOr;

  }  ///if we are masking.

  // --------------------------
  // Set FinalOR for this board
  // --------------------------
  m_algFinalOr = (m_algIntermOr & !m_algFinalOrVeto);
}

// Fill DAQ Record
void l1t::GlobalBoard::fillAlgRecord(int iBxInEvent,
                                     std::unique_ptr<GlobalAlgBlkBxCollection>& uGtAlgRecord,
                                     int prescaleSet,
                                     int menuUUID,
                                     int firmwareUUID) {
  if (m_verbosity) {
    LogDebug("L1TGlobal") << "\n**** GlobalBoard fill DAQ Records for bx= " << iBxInEvent;
  }

  // Set header information
  m_uGtAlgBlk.setbxInEventNr((iBxInEvent & 0xF));
  m_uGtAlgBlk.setPreScColumn(prescaleSet);
  m_uGtAlgBlk.setL1MenuUUID(menuUUID);
  m_uGtAlgBlk.setL1FirmwareUUID(firmwareUUID);

  m_uGtAlgBlk.setFinalORVeto(m_algFinalOrVeto);
  m_uGtAlgBlk.setFinalORPreVeto(m_algIntermOr);
  m_uGtAlgBlk.setFinalOR(m_algFinalOr);

  uGtAlgRecord->push_back(iBxInEvent, m_uGtAlgBlk);
}

// clear GTL
void l1t::GlobalBoard::reset() {
  resetMu();
  resetMuonShower();
  resetCalo();
  resetExternal();

  m_uGtAlgBlk.reset();

  m_gtlDecisionWord.reset();
  m_gtlAlgorithmOR.reset();
}

// clear muon
void l1t::GlobalBoard::resetMu() {
  m_candL1Mu->clear();
  m_candL1Mu->setBXRange(m_bxFirst_, m_bxLast_);
}

// clear muon shower
void l1t::GlobalBoard::resetMuonShower() {
  m_candL1MuShower->clear();
  m_candL1MuShower->setBXRange(m_bxFirst_, m_bxLast_);
}

// clear calo
void l1t::GlobalBoard::resetCalo() {
  m_candL1EG->clear();
  m_candL1Tau->clear();
  m_candL1Jet->clear();
  m_candL1EtSum->clear();
  m_candL1EtSumZdc->clear();

  m_candL1EG->setBXRange(m_bxFirst_, m_bxLast_);
  m_candL1Tau->setBXRange(m_bxFirst_, m_bxLast_);
  m_candL1Jet->setBXRange(m_bxFirst_, m_bxLast_);
  m_candL1EtSum->setBXRange(m_bxFirst_, m_bxLast_);
  m_candL1EtSumZdc->setBXRange(m_bxFirst_, m_bxLast_);
}

void l1t::GlobalBoard::resetExternal() {
  m_candL1External->clear();
  m_candL1External->setBXRange(m_bxFirst_, m_bxLast_);
}

// print Global Muon Trigger data received
void l1t::GlobalBoard::printGmtData(const int iBxInEvent) const {
  LogTrace("L1TGlobal") << "\nl1t::L1GlobalTrigger: uGMT data received for BxInEvent = " << iBxInEvent;

  int nrL1Mu = m_candL1Mu->size(iBxInEvent);
  LogTrace("L1TGlobal") << "Number of GMT muons = " << nrL1Mu << "\n";
}

// initialize prescale counters to zero
std::vector<l1t::GlobalBoard::PrescaleCounter> l1t::GlobalBoard::prescaleCounters(
    std::vector<double> const& prescaleFactorsAlgoTrig) {
  std::vector<PrescaleCounter> out;
  out.reserve(prescaleFactorsAlgoTrig.size());
  for (size_t iAlgo = 0; iAlgo < prescaleFactorsAlgoTrig.size(); ++iAlgo) {
    out.emplace_back(prescaleFactorsAlgoTrig[iAlgo]);
  }
  return out;
}

// initialises prescale counters with a semi-random value in the range [0, prescale*10^precision - 1]
std::vector<l1t::GlobalBoard::PrescaleCounter> l1t::GlobalBoard::prescaleCountersWithSemirandomInitialCounter(
    std::vector<double> const& prescaleFactorsAlgoTrig, edm::Event const& iEvent) {
  // pick a (semi)random number seeding based on run, lumi, event numbers,
  // this leads to different (semi)random numbers for different streams,
  // reminder: different streams have different initial event number
  std::srand(iEvent.id().run());
  std::srand(std::rand() + iEvent.id().luminosityBlock());
  std::srand(std::rand() + iEvent.id().event());
  int const semirandom = std::rand();

  std::vector<PrescaleCounter> out;
  out.reserve(prescaleFactorsAlgoTrig.size());

  for (size_t iAlgo = 0; iAlgo < prescaleFactorsAlgoTrig.size(); ++iAlgo) {
    out.emplace_back(prescaleFactorsAlgoTrig[iAlgo]);
    // initialise trigger_counter to a (semi)random integer
    // between 0 and prescale_count - 1 (both inclusive)
    // (this only changes the behaviour of triggers with PS > 1)
    auto& prescaleCounter = out.back();
    if (prescaleCounter.prescale_count > 0) {
      prescaleCounter.trigger_counter = semirandom % prescaleCounter.prescale_count;
    }
  }

  return out;
}

// return decision of PrescalCounter, and update its internal counter
bool l1t::GlobalBoard::PrescaleCounter::accept() {
  trigger_counter += m_singlestep;

  if (prescale_count == 0 or trigger_counter < prescale_count)
    return false;

  trigger_counter -= prescale_count;

  return true;
}
