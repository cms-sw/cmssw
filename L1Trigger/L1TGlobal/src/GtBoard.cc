/**
 * \class GtBoard
 *
 *
 * Description: Global Trigger Logic board, see header file for details.
 *
 * Implementation:
 *    <TODO: enter implementation details>
 *
 * \author: M. Fierro            - HEPHY Vienna - ORCA version
 * \author: Vasile Mihai Ghete   - HEPHY Vienna - CMSSW version
 *
 * $Date$
 * $Revision$
 *
 */

// this class header
#include "L1Trigger/L1TGlobal/interface/GtBoard.h"

// system include files
#include <ext/hash_map>

// user include files
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMap.h"

#include "CondFormats/L1TObjects/interface/L1GtFwd.h"

#include "L1Trigger/L1TGlobal/interface/TriggerMenu.h"
#include "CondFormats/DataRecord/interface/L1TGlobalTriggerMenuRcd.h"

#include "CondFormats/L1TObjects/interface/L1GtCondition.h"
#include "CondFormats/L1TObjects/interface/L1GtAlgorithm.h"

#include "L1Trigger/L1TGlobal/interface/MuonTemplate.h"
#include "L1Trigger/L1TGlobal/interface/CaloTemplate.h"
#include "L1Trigger/L1TGlobal/interface/EnergySumTemplate.h"
#include "CondFormats/L1TObjects/interface/L1GtJetCountsTemplate.h"
#include "CondFormats/L1TObjects/interface/L1GtHfBitCountsTemplate.h"
#include "CondFormats/L1TObjects/interface/L1GtHfRingEtSumsTemplate.h"
#include "CondFormats/L1TObjects/interface/L1GtCastorTemplate.h"
#include "CondFormats/L1TObjects/interface/L1GtBptxTemplate.h"
#include "CondFormats/L1TObjects/interface/L1GtExternalTemplate.h"
#include "CondFormats/L1TObjects/interface/L1GtCorrelationTemplate.h"

#include "CondFormats/L1TObjects/interface/L1MuTriggerScales.h"
#include "CondFormats/DataRecord/interface/L1MuTriggerScalesRcd.h"
#include "CondFormats/L1TObjects/interface/L1CaloGeometry.h"
#include "CondFormats/DataRecord/interface/L1CaloGeometryRecord.h"

#include "L1Trigger/GlobalTrigger/interface/L1GtAlgorithmEvaluation.h"
#include "L1Trigger/L1TGlobal/interface/ConditionEvaluation.h"
#include "L1Trigger/L1TGlobal/interface/AlgorithmEvaluation.h"


//#include "L1Trigger/GlobalTrigger/interface/L1CaloCondition.h"
#include "L1Trigger/GlobalTrigger/interface/L1GtJetCountsCondition.h"
#include "L1Trigger/GlobalTrigger/interface/L1GtHfBitCountsCondition.h"
#include "L1Trigger/GlobalTrigger/interface/L1GtHfRingEtSumsCondition.h"
#include "L1Trigger/GlobalTrigger/interface/L1GtCastorCondition.h"
#include "L1Trigger/GlobalTrigger/interface/L1GtBptxCondition.h"
#include "L1Trigger/GlobalTrigger/interface/L1GtExternalCondition.h"
#include "L1Trigger/GlobalTrigger/interface/L1GtCorrelationCondition.h"


// Conditions for uGt
#include "L1Trigger/L1TGlobal/interface/MuCondition.h"
#include "L1Trigger/L1TGlobal/interface/CaloCondition.h"
#include "L1Trigger/L1TGlobal/interface/EnergySumCondition.h"

//   *** Comment out what do we do with this.
#include "L1Trigger/GlobalTrigger/interface/L1GtEtaPhiConversions.h"


#include "FWCore/Utilities/interface/Exception.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/MessageLogger/interface/MessageDrop.h"

#include "FWCore/Framework/interface/ESHandle.h"

// forward declarations

// constructor
l1t::GtBoard::GtBoard() :
    m_candL1Mu( new BXVector<const l1t::Muon*>),
    m_candL1EG( new BXVector<const l1t::L1Candidate*>),
    m_candL1Tau( new BXVector<const l1t::L1Candidate*>),
    m_candL1Jet( new BXVector<const l1t::L1Candidate*>),
    m_candL1EtSum( new BXVector<const l1t::EtSum*>),
    m_isDebugEnabled(edm::isDebugEnabled())
{

    m_uGtAlgBlk.reset();
    m_uGtExtBlk.reset();
    
    m_gtlAlgorithmOR.reset();
    m_gtlDecisionWord.reset();

    // initialize cached IDs
    m_l1GtMenuCacheID = 0ULL;
    m_l1CaloGeometryCacheID = 0ULL;
    m_l1MuTriggerScalesCacheID = 0ULL;

    // Counter for number of events board sees
    m_boardEventCount=0;
    
    // Need to expand use with more than one uGt GtBoard for now assume 1
    m_uGtBoardNumber = 0;
    m_uGtFinalBoard = true;
   
/*   Do we need this?
    // pointer to conversion - actually done in the event loop (cached)
    m_gtEtaPhiConversions = new L1GtEtaPhiConversions();
    m_gtEtaPhiConversions->setVerbosity(m_verbosity);
*/

}

// destructor
l1t::GtBoard::~GtBoard() {

    reset();
    delete m_candL1Mu;
    delete m_candL1EG;
    delete m_candL1Tau;
    delete m_candL1Jet;
    delete m_candL1EtSum;

//    delete m_gtEtaPhiConversions;

}

// operations
void l1t::GtBoard::setBxFirst(int bx){

  m_bxFirst_ = bx;

}

void l1t::GtBoard::setBxLast(int bx){

  m_bxLast_ = bx;

}

void l1t::GtBoard::init(const int numberPhysTriggers, const int nrL1Mu, const int nrL1EG, const int nrL1Tau, const int nrL1Jet,
			   int bxFirst, int bxLast) {

  setBxFirst(bxFirst);
  setBxLast(bxLast);

  m_candL1Mu->setBXRange( m_bxFirst_, m_bxLast_ );
  m_candL1EG->setBXRange( m_bxFirst_, m_bxLast_ );
  m_candL1Tau->setBXRange( m_bxFirst_, m_bxLast_ );
  m_candL1Jet->setBXRange( m_bxFirst_, m_bxLast_ );
  m_candL1EtSum->setBXRange( m_bxFirst_, m_bxLast_ );

  m_uGtAlgBlk.reset();
  m_uGtExtBlk.reset();
  
  LogDebug("l1t|Global") << "\t Initializing Board with bxFirst = " << m_bxFirst_ << ", bxLast = " << m_bxLast_ << std::endl;

  //m_candL1Mu->resizeAll(nrL1Mu);

    // FIXME move from bitset to std::vector<bool> to be able to use
    // numberPhysTriggers from EventSetup

    //m_gtlAlgorithmOR.reserve(numberPhysTriggers);
    //m_gtlAlgorithmOR.assign(numberPhysTriggers, false);

    //m_gtlDecisionWord.reserve(numberPhysTriggers);
    //m_gtlDecisionWord.assign(numberPhysTriggers, false);

}



// receive data from Calorimeter
void l1t::GtBoard::receiveCaloObjectData(edm::Event& iEvent,
        const edm::InputTag& caloInputTag, 
        const bool receiveEG, const int nrL1EG,
	const bool receiveTau, const int nrL1Tau,	
	const bool receiveJet, const int nrL1Jet,
	const bool receiveEtSums) {

    if (m_verbosity) {
        LogDebug("l1t|Global")
                << "\n**** Board receiving Calo Data "
                <<  "\n     from input tag " << caloInputTag << "\n"
                << std::endl;

    }

    resetCalo();
    
    // get data from Calorimeter
    if (receiveEG) {
        edm::Handle<BXVector<l1t::EGamma>> egData;
        iEvent.getByLabel(caloInputTag, egData);

        if (!egData.isValid()) {
            if (m_verbosity) {
                edm::LogWarning("l1t|Global")
                        << "\nWarning: BXVector<l1t::EGamma> with input tag "
                        << caloInputTag
                        << "\nrequested in configuration, but not found in the event.\n"
                        << std::endl;
            }
        } else {
           // bx in EG data
           for(int i = egData->getFirstBX(); i <= egData->getLastBX(); ++i) {
  
              //Loop over EG in this bx
              for(std::vector<l1t::EGamma>::const_iterator eg = egData->begin(i); eg != egData->end(i); ++eg) {

	        (*m_candL1EG).push_back(i,&(*eg));
	        LogDebug("l1t|Global") << "EG  Pt " << eg->hwPt() << " Eta  " << eg->hwEta() << " Phi " << eg->hwPhi() << "  Qual " << eg->hwQual() <<"  Iso " << eg->hwIso() << std::endl;
              } //end loop over EG in bx
	   } //end loop over bx   

        } //end if over valid EG data

    } //end if ReveiveEG data


    if (receiveTau) {
        edm::Handle<BXVector<l1t::Tau>> tauData;
        iEvent.getByLabel(caloInputTag, tauData);

        if (!tauData.isValid()) {
            if (m_verbosity) {
                edm::LogWarning("l1t|Global")
                        << "\nWarning: BXVector<l1t::Tau> with input tag "
                        << caloInputTag
                        << "\nrequested in configuration, but not found in the event.\n"
                        << std::endl;
            }
        } else {
           // bx in tau data
           for(int i = tauData->getFirstBX(); i <= tauData->getLastBX(); ++i) {
  
              //Loop over tau in this bx
              for(std::vector<l1t::Tau>::const_iterator tau = tauData->begin(i); tau != tauData->end(i); ++tau) {

	        (*m_candL1Tau).push_back(i,&(*tau));
	        LogDebug("l1t|Global") << "tau  Pt " << tau->hwPt() << " Eta  " << tau->hwEta() << " Phi " << tau->hwPhi() << "  Qual " << tau->hwQual() <<"  Iso " << tau->hwIso() << std::endl;
              } //end loop over tau in bx
	   } //end loop over bx   

        } //end if over valid tau data

    } //end if ReveiveTau data


    if (receiveJet) {
        edm::Handle<BXVector<l1t::Jet>> jetData;
        iEvent.getByLabel(caloInputTag, jetData);

        if (!jetData.isValid()) {
            if (m_verbosity) {
                edm::LogWarning("l1t|Global")
                        << "\nWarning: BXVector<l1t::Jet> with input tag "
                        << caloInputTag
                        << "\nrequested in configuration, but not found in the event.\n"
                        << std::endl;
            }
        } else {
           // bx in jet data
           for(int i = jetData->getFirstBX(); i <= jetData->getLastBX(); ++i) {
  
              //Loop over jet in this bx
              for(std::vector<l1t::Jet>::const_iterator jet = jetData->begin(i); jet != jetData->end(i); ++jet) {

	        (*m_candL1Jet).push_back(i,&(*jet));
	        LogDebug("l1t|Global") << "Jet  Pt " << jet->hwPt() << " Eta  " << jet->hwEta() << " Phi " << jet->hwPhi() << "  Qual " << jet->hwQual() <<"  Iso " << jet->hwIso() << std::endl;
              } //end loop over jet in bx
	   } //end loop over bx   

        } //end if over valid jet data

    } //end if ReveiveJet data


    if(receiveEtSums) {
        edm::Handle<BXVector<l1t::EtSum>> etSumData;
        iEvent.getByLabel(caloInputTag, etSumData);

        if(!etSumData.isValid()) {
            if (m_verbosity) {
                edm::LogWarning("l1t|Global")
                        << "\nWarning: BXVector<l1t::EtSum> with input tag "
                        << caloInputTag
                        << "\nrequested in configuration, but not found in the event.\n"
                        << std::endl;
            }
	} else {

           for(int i = etSumData->getFirstBX(); i <= etSumData->getLastBX(); ++i) {

              //Loop over jet in this bx
              for(std::vector<l1t::EtSum>::const_iterator etsum = etSumData->begin(i); etsum != etSumData->end(i); ++etsum) {
	                    
                  (*m_candL1EtSum).push_back(i,&(*etsum));
		  
/*  In case we need to split these out
	          switch ( etsum->getType() ) {
		     case l1t::EtSum::EtSumType::kMissingEt:
		       (*m_candETM).push_back(i,&(*etsum));
		       LogDebug("l1t|Global") << "ETM:  Pt " << etsum->hwPt() <<  " Phi " << etsum->hwPhi()  << std::endl;
		       break; 
		     case l1t::EtSum::EtSumType::kMissingHt:
		       (*m_candHTM.push_back(i,&(*etsum);
		       LogDebug("l1t|Global") << "HTM:  Pt " << etsum->hwPt() <<  " Phi " << etsum->hwPhi()  << std::endl;
		       break; 		     
		     case l1t::EtSum::EtSumType::kTotalEt:
		       (*m_candETT.push_back(i,&(*etsum);
		       LogDebug("l1t|Global") << "ETT:  Pt " << etsum->hwPt() << std::endl;
		       break; 		     
		     case l1t::EtSum::EtSumType::kTotalHt:
		       (*m_candHTT.push_back(i,&(*etsum);
		       LogDebug("l1t|Global") << "HTT:  Pt " << etsum->hwPt() << std::endl;
		       break; 		     
		  }
*/
	      
	      } //end loop over jet in bx
	   } //end loop over Bx
	
	}    

      
    }

}


// receive data from Global Muon Trigger
void l1t::GtBoard::receiveMuonObjectData(edm::Event& iEvent,
    const edm::InputTag& muInputTag, const bool receiveMu,
    const int nrL1Mu) {

    if (m_verbosity) {
        LogDebug("l1t|Global")
                << "\n**** GtBoard receiving muon data = "
                << "\n     from input tag " << muInputTag << "\n"
                << std::endl;
    }

    resetMu();

    // get data from Global Muon Trigger
    if (receiveMu) {
        edm::Handle<BXVector<l1t::Muon>> muonData;
        iEvent.getByLabel(muInputTag, muonData);

        if (!muonData.isValid()) {
            if (m_verbosity) {
                edm::LogWarning("l1t|Global")
                        << "\nWarning: BXVector<l1t::Muon> with input tag "
                        << muInputTag
                        << "\nrequested in configuration, but not found in the event.\n"
                        << std::endl;
            }
        } else {
           // bx in muon data
           for(int i = muonData->getFirstBX(); i <= muonData->getLastBX(); ++i) {
  
              //Loop over Muons in this bx
              for(std::vector<l1t::Muon>::const_iterator mu = muonData->begin(i); mu != muonData->end(i); ++mu) {

	        (*m_candL1Mu).push_back(i,&(*mu));
	        LogDebug("l1t|Global") << "Muon  Pt " << mu->hwPt() << " Eta  " << mu->hwEta() << " Phi " << mu->hwPhi() << "  Qual " << mu->hwQual() <<"  Iso " << mu->hwIso() << std::endl;
              } //end loop over muons in bx
	   } //end loop over bx   

        } //end if over valid muon data

    } //end if ReveiveMuon data

    if (m_verbosity && m_isDebugEnabled) {
//  *** Needs fixing
//        printGmtData(iBxInEvent);
    }

}

// run GTL
void l1t::GtBoard::runGTL(
        edm::Event& iEvent, const edm::EventSetup& evSetup,
        const bool produceL1GtObjectMapRecord,
        const int iBxInEvent,
        std::auto_ptr<L1GlobalTriggerObjectMapRecord>& gtObjectMapRecord,
        const unsigned int numberPhysTriggers,
        const int nrL1Mu,
        const int nrL1EG,
	const int nrL1Tau,
	const int nrL1Jet,
        const int nrL1JetCounts) {


	// get / update the trigger menu from the EventSetup
    // local cache & check on cacheIdentifier
    unsigned long long l1GtMenuCacheID = evSetup.get<L1TGlobalTriggerMenuRcd>().cacheIdentifier();

    if (m_l1GtMenuCacheID != l1GtMenuCacheID) {

        edm::ESHandle< TriggerMenu> l1GtMenu;
        evSetup.get< L1TGlobalTriggerMenuRcd>().get(l1GtMenu) ;
        m_l1GtMenu =  l1GtMenu.product();
       (const_cast<TriggerMenu*>(m_l1GtMenu))->buildGtConditionMap();

        m_l1GtMenuCacheID = l1GtMenuCacheID;
    }

    const std::vector<ConditionMap>& conditionMap = m_l1GtMenu->gtConditionMap();
    const AlgorithmMap& algorithmMap = m_l1GtMenu->gtAlgorithmMap();

    // Reset AlgBlk for this bx
     m_uGtAlgBlk.reset();
     m_uGtExtBlk.reset();
     m_algInitialOr=false;
     m_algPrescaledOr=false;
     m_algFinalOr=false;
     
     
    /*
    const std::vector<std::vector<MuonTemplate> >& corrMuon =
            m_l1GtMenu->corMuonTemplate();

      // Comment out for now
    const std::vector<std::vector<CaloTemplate> >& corrCalo =
            m_l1GtMenu->corCaloTemplate();

    const std::vector<std::vector<EnergySumTemplate> >& corrEnergySum =
            m_l1GtMenu->corEnergySumTemplate();

    LogDebug("l1t|Global") << "Size corrMuon " << corrMuon.size() 
                           << "\nSize corrCalo " << corrCalo.size() 
			   << "\nSize corrSums " << corrEnergySum.size() << std::endl;
    */

    // conversion needed for correlation conditions
    // done in the condition loop when the first correlation template is in the menu
    bool convertScale = false;

    // get / update the calorimeter geometry from the EventSetup
    // local cache & check on cacheIdentifier
    unsigned long long l1CaloGeometryCacheID =
            evSetup.get<L1CaloGeometryRecord>().cacheIdentifier();


    if (m_l1CaloGeometryCacheID != l1CaloGeometryCacheID) {

        edm::ESHandle<L1CaloGeometry> l1CaloGeometry;
        evSetup.get<L1CaloGeometryRecord>().get(l1CaloGeometry) ;
        m_l1CaloGeometry =  l1CaloGeometry.product();

        m_l1CaloGeometryCacheID = l1CaloGeometryCacheID;
        convertScale = true;

    }

    // get / update the eta and phi muon trigger scales from the EventSetup
    // local cache & check on cacheIdentifier
    unsigned long long l1MuTriggerScalesCacheID =
            evSetup.get<L1MuTriggerScalesRcd>().cacheIdentifier();

    if (m_l1MuTriggerScalesCacheID != l1MuTriggerScalesCacheID) {

        edm::ESHandle< L1MuTriggerScales> l1MuTriggerScales;
        evSetup.get< L1MuTriggerScalesRcd>().get(l1MuTriggerScales);
        m_l1MuTriggerScales = l1MuTriggerScales.product();

        m_l1MuTriggerScalesCacheID = l1MuTriggerScalesCacheID;
        convertScale = true;
    }

    if (convertScale) {

/*  Comment out for now
        m_gtEtaPhiConversions->setVerbosity(m_verbosity);
        m_gtEtaPhiConversions->convertL1Scales(m_l1CaloGeometry,
                m_l1MuTriggerScales, ifCaloEtaNumberBits, ifMuEtaNumberBits);

        // print the conversions if DEBUG and verbosity enabled

        if (m_verbosity && m_isDebugEnabled) {
            std::ostringstream myCout;
            m_gtEtaPhiConversions->print(myCout);

            LogTrace("l1t|Global") << myCout.str() << std::endl;
        }

        // set convertScale to false to avoid executing the conversion
        // more than once - in case the scales change it will be set to true
        // in the cache check
*/
        convertScale = false;
    }


    // loop over condition maps (one map per condition chip)
    // then loop over conditions in the map
    // save the results in temporary maps

    // never happens in production but at first event...
    if (m_conditionResultMaps.size() != conditionMap.size()) {
        m_conditionResultMaps.clear();
        m_conditionResultMaps.resize(conditionMap.size());
    }
    
    int iChip = -1;

    for (std::vector<ConditionMap>::const_iterator
    		itCondOnChip = conditionMap.begin(); itCondOnChip != conditionMap.end(); itCondOnChip++) {

        iChip++;

// blw was this commented out before?
        //AlgorithmEvaluation::ConditionEvaluationMap cMapResults;
        // AlgorithmEvaluation::ConditionEvaluationMap cMapResults((*itCondOnChip).size()); // hash map

       AlgorithmEvaluation::ConditionEvaluationMap& cMapResults =
               m_conditionResultMaps[iChip];



        for (CItCond itCond = itCondOnChip->begin(); itCond != itCondOnChip->end(); itCond++) {

            // evaluate condition
            switch ((itCond->second)->condCategory()) {
                case CondMuon: {

                    // BLW Not sure what to do with this for now
		    const int ifMuEtaNumberBits = 0;
		    
                    MuCondition* muCondition = new MuCondition(itCond->second, this,
                            nrL1Mu, ifMuEtaNumberBits);

                    muCondition->setVerbosity(m_verbosity);
                  // BLW Comment out for now
		  //  muCondition->setGtCorrParDeltaPhiNrBins(
                  //          (m_gtEtaPhiConversions->gtObjectNrBinsPhi(Mu)) / 2
                  //                  + 1);
		  // BLW Need to pass the bx we are working on.
                    muCondition->evaluateConditionStoreResult(iBxInEvent);

                   // BLW COmment out for now 
		    cMapResults[itCond->first] = muCondition;

                    if (m_verbosity && m_isDebugEnabled) {
                        std::ostringstream myCout;
                        muCondition->print(myCout);

                        LogTrace("l1t|Global") << myCout.str() << std::endl;
                    }		    
                    //delete muCondition;

                }
                    break;
                case CondCalo: {

                    // BLW Not sure w hat to do with this for now
		    const int ifCaloEtaNumberBits = 0;

                    CaloCondition* caloCondition = new CaloCondition(
                            itCond->second, this,
                            nrL1EG,
                            nrL1Jet,
                            nrL1Tau,
                            ifCaloEtaNumberBits);

                    caloCondition->setVerbosity(m_verbosity);

                    // BLW COmment out for Now
                    //caloCondition->setGtCorrParDeltaPhiNrBins(
                    //        (m_gtEtaPhiConversions->gtObjectNrBinsPhi(
                    //                ((itCond->second)->objectType())[0])) / 2
                    //                + 1);
                    caloCondition->evaluateConditionStoreResult(iBxInEvent);
                    
		    // BLW Comment out for now
                    cMapResults[itCond->first] = caloCondition;

                    if (m_verbosity && m_isDebugEnabled) {
                        std::ostringstream myCout;
                        caloCondition->print(myCout);

                        LogTrace("l1t|Global") << myCout.str() << std::endl;
                    }
                    //                    delete caloCondition;
		    
                }
                    break;
                case CondEnergySum: {

                    EnergySumCondition* eSumCondition = new EnergySumCondition(
                            itCond->second, this);

                    eSumCondition->setVerbosity(m_verbosity);
                    eSumCondition->evaluateConditionStoreResult(iBxInEvent);

                    cMapResults[itCond->first] = eSumCondition;

                    if (m_verbosity && m_isDebugEnabled) {
                        std::ostringstream myCout;
                        eSumCondition->print(myCout);

                        LogTrace("l1t|Global") << myCout.str() << std::endl;
                    }
                    //                    delete eSumCondition;

                }
                    break;
/*  Don't access conditions for now		
                case CondJetCounts: {
                    L1GtJetCountsCondition* jcCondition = new L1GtJetCountsCondition(
                            itCond->second, this, nrL1JetCounts);

                    jcCondition->setVerbosity(m_verbosity);
                    jcCondition->evaluateConditionStoreResult();

                    cMapResults[itCond->first] = jcCondition;

                    if (m_verbosity && m_isDebugEnabled) {
                        std::ostringstream myCout;
                        jcCondition->print(myCout);

                        LogTrace("l1t|Global") << myCout.str() << std::endl;
                    }

                    //                  delete jcCondition;
                }
                    break;
*/
                case CondExternal: {
  
 /*  Don't access conditions for now 
                    bool externalCondResult = true;

                    // FIXME need a solution to read External with real value

                    L1GtExternalCondition* externalCondition = new L1GtExternalCondition(
                            itCond->second, externalCondResult);

                    externalCondition->setVerbosity(m_verbosity);
                    externalCondition->evaluateConditionStoreResult();

                    cMapResults[itCond->first] = externalCondition;

                    if (m_verbosity && m_isDebugEnabled) {
                        std::ostringstream myCout;
                        externalCondition->print(myCout);

                        LogTrace("l1t|Global") << myCout.str() << std::endl;
                    }

                    //                  delete externalCondition;
*/
                }
                    break;
                case CondCorrelation: {


/*  Don't access conditions for now
                    // get first the sub-conditions
                    const L1GtCorrelationTemplate* corrTemplate =
                        static_cast<const L1GtCorrelationTemplate*>(itCond->second);
                    const L1GtConditionCategory cond0Categ = corrTemplate->cond0Category();
                    const L1GtConditionCategory cond1Categ = corrTemplate->cond1Category();
                    const int cond0Ind = corrTemplate->cond0Index();
                    const int cond1Ind = corrTemplate->cond1Index();

                    const L1GtCondition* cond0Condition = 0;
                    const L1GtCondition* cond1Condition = 0;

                    // maximum number of objects received for evaluation of l1t::Type1s condition
                    int cond0NrL1Objects = 0;
                    int cond1NrL1Objects = 0;

                    int cond0EtaBits = 0;
                    int cond1EtaBits = 0;

                    switch (cond0Categ) {
                        case CondMuon: {
                            cond0Condition = &((corrMuon[iChip])[cond0Ind]);
                            cond0NrL1Objects = nrL1Mu;
                            cond0EtaBits = ifMuEtaNumberBits;
                        }
                            break;
                        case CondCalo: {
                            cond0Condition = &((corrCalo[iChip])[cond0Ind]);

                            switch ((cond0Condition->objectType())[0]) {
                                case NoIsoEG:
                                    cond0NrL1Objects= nrL1NoIsoEG;
                                    break;
                                case IsoEG:
                                    cond0NrL1Objects = nrL1IsoEG;
                                    break;
                                case CenJet:
                                    cond0NrL1Objects = nrL1CenJet;
                                    break;
                                case ForJet:
                                    cond0NrL1Objects = nrL1ForJet;
                                    break;
                                case TauJet:
                                    cond0NrL1Objects = nrL1TauJet;
                                    break;
                                default:
                                    cond0NrL1Objects = 0;
                                    break;
                            }

                            cond0EtaBits = ifCaloEtaNumberBits;
                        }
                            break;
                        case CondEnergySum: {
                            cond0Condition = &((corrEnergySum[iChip])[cond0Ind]);
                            cond0NrL1Objects = 1;
                        }
                            break;
                        default: {
                            // do nothing, should not arrive here
                        }
                            break;
                    }

                    switch (cond1Categ) {
                        case CondMuon: {
                            cond1Condition = &((corrMuon[iChip])[cond1Ind]);
                            cond1NrL1Objects = nrL1Mu;
                            cond1EtaBits = ifMuEtaNumberBits;
                        }
                            break;
                        case CondCalo: {
                            cond1Condition = &((corrCalo[iChip])[cond1Ind]);

                            switch ((cond1Condition->objectType())[0]) {
                                case NoIsoEG:
                                    cond1NrL1Objects= nrL1NoIsoEG;
                                    break;
                                case IsoEG:
                                    cond1NrL1Objects = nrL1IsoEG;
                                    break;
                                case CenJet:
                                    cond1NrL1Objects = nrL1CenJet;
                                    break;
                                case ForJet:
                                    cond1NrL1Objects = nrL1ForJet;
                                    break;
                                case TauJet:
                                    cond1NrL1Objects = nrL1TauJet;
                                    break;
                                default:
                                    cond1NrL1Objects = 0;
                                    break;
                            }

                             cond1EtaBits = ifCaloEtaNumberBits;
                        }
                            break;
                        case CondEnergySum: {
                            cond1Condition = &((corrEnergySum[iChip])[cond1Ind]);
                            cond1NrL1Objects = 1;
                        }
                            break;
                        default: {
                            // do nothing, should not arrive here
                        }
                            break;
                    }

                    L1GtCorrelationCondition* correlationCond =
                        new L1GtCorrelationCondition(itCond->second,
                            cond0Condition, cond1Condition,
                            cond0NrL1Objects, cond1NrL1Objects,
                            cond0EtaBits, cond1EtaBits,
                            this, this, m_gtEtaPhiConversions);

                    correlationCond->setVerbosity(m_verbosity);
                    correlationCond->evaluateConditionStoreResult();

                    cMapResults[itCond->first] = correlationCond;

                    if (m_verbosity && m_isDebugEnabled) {
                        std::ostringstream myCout;
                        correlationCond->print(myCout);

                        LogTrace("l1t|Global") << myCout.str() << std::endl;
                    }

                    //                  delete correlationCond;
*/
                }
                    break;
                case CondNull: {

                    // do nothing

                }
                    break;
                default: {
                   // do nothing

                }
                    break;
            }

        }

    }

    // loop over algorithm map
    /// DMP Start debugging here
    // empty vector for object maps - filled during loop
    std::vector<L1GlobalTriggerObjectMap> objMapVec;
    if (produceL1GtObjectMapRecord && (iBxInEvent == 0)) objMapVec.reserve(numberPhysTriggers);

    for (CItAlgo itAlgo = algorithmMap.begin(); itAlgo != algorithmMap.end(); itAlgo++) {
        AlgorithmEvaluation gtAlg(itAlgo->second);
        gtAlg.evaluateAlgorithm((itAlgo->second).algoChipNumber(), m_conditionResultMaps);

        int algBitNumber = (itAlgo->second).algoBitNumber();
        bool algResult = gtAlg.gtAlgoResult();

	LogDebug("l1t|Global") << " ===> for iBxInEvent = " << iBxInEvent << ":\t algBitName = " << itAlgo->first << ",\t algBitNumber = " << algBitNumber << ",\t algResult = " << algResult << std::endl;

        if (algResult) {
//            m_gtlAlgorithmOR.set(algBitNumber);
              m_uGtAlgBlk.setAlgoDecisionInitial(algBitNumber,algResult);
	      m_algInitialOr = true;	    
        }

        if (m_verbosity && m_isDebugEnabled) {
            std::ostringstream myCout;
            ( itAlgo->second ).print(myCout);
            gtAlg.print(myCout);

            LogTrace("l1t|Global") << myCout.str() << std::endl;
        }


        // object maps only for BxInEvent = 0
        if (produceL1GtObjectMapRecord && (iBxInEvent == 0)) {

            // set object map
            L1GlobalTriggerObjectMap objMap;

            objMap.setAlgoName(itAlgo->first);
            objMap.setAlgoBitNumber(algBitNumber);
            objMap.setAlgoGtlResult(algResult);
            objMap.swapOperandTokenVector(gtAlg.operandTokenVector());
            objMap.swapCombinationVector(gtAlg.gtAlgoCombinationVector());
	    // gtAlg is empty now...

            if (m_verbosity && m_isDebugEnabled) {
                std::ostringstream myCout1;
                objMap.print(myCout1);

                LogTrace("l1t|Global") << myCout1.str() << std::endl;
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
    for (std::vector<AlgorithmEvaluation::ConditionEvaluationMap>::iterator
            itCondOnChip  = m_conditionResultMaps.begin();
            itCondOnChip != m_conditionResultMaps.end(); itCondOnChip++) {

        for (AlgorithmEvaluation::ItEvalMap
                itCond  = itCondOnChip->begin();
                itCond != itCondOnChip->end(); itCond++) {

            delete itCond->second;
            itCond->second = 0;
        }
    }

}


// run GTL
void l1t::GtBoard::runFDL(edm::Event& iEvent, 
        const int iBxInEvent,
        const bool algorithmTriggersUnprescaled,
        const bool algorithmTriggersUnmasked ){


    if (m_verbosity) {
        LogDebug("l1t|Global")
                << "\n**** GtBoard apply Final Decision Logic "
                << std::endl;

    }

/* Nothing with prescales right now.
    // prescale counters are reset at the beginning of the luminosity segment
    if (m_firstEv) {

        m_firstEv = false;
    }

    // TODO FIXME find the beginning of the luminosity segment
    if (m_firstEvLumiSegment) {

        m_firstEvLumiSegment = false;

    }
*/

    // Copy Algorithm bits to Prescaled word 
    // Prescaling and Masking done below if requested.
    m_uGtAlgBlk.copyInitialToPrescaled();
    

    // -------------------------------------------
    //      Apply Prescales or skip if turned off
    // -------------------------------------------
    if (!algorithmTriggersUnprescaled){
/*
	for (unsigned int iBit = 0; iBit < numberPhysTriggers; ++iBit) {

            if (prescaleFactorsAlgoTrig.at(iBit) != 1) {

        	bool bitValue = algoDecisionWord.at( iBit );
        	if (bitValue) {

                    (m_prescaleCounterAlgoTrig.at(inBxInEvent).at(iBit))--;
                    if (m_prescaleCounterAlgoTrig.at(inBxInEvent).at(iBit) == 0) {

                	// bit already true in algoDecisionWord, just reset counter
                	m_prescaleCounterAlgoTrig.at(inBxInEvent).at(iBit) =
                            prescaleFactorsAlgoTrig.at(iBit);
                    } else {

                	// change bit to false in prescaled word and final decision word
                	algoDecisionWord[iBit] = false;

                    } //if Prescale counter reached zero
        	} //if algo bit is set true
            } //if prescale factor is not 1 (ie. no prescale)
	} //loop over alg bits
*/
	m_algPrescaledOr = m_algInitialOr; //temp
	 
    } else {
        
	// Since not Prescaling just take OR of Initial Work
	m_algPrescaledOr = m_algInitialOr;
	
    }//if we are going to apply prescales.
      
      
      
    // Copy Algorithm bits fron Prescaled word to Final Word 
    // Masking done below if requested.
    m_uGtAlgBlk.copyPrescaledToFinal();
    
    if(!algorithmTriggersUnmasked) {
    
 
/*
       masking the bits goes here.
*/       
        m_algFinalOr = m_algPrescaledOr;
	
    } else {
      
         m_algFinalOr = m_algPrescaledOr;
     
    } ///if we are masking.	



}

// Fill DAQ Record
void l1t::GtBoard::fillAlgRecord(int iBxInEvent, 
				    std::auto_ptr<GlobalAlgBlkBxCollection>& uGtAlgRecord,
				    cms_uint64_t orbNr,
				    int bxNr
				    ) 
{

    if (m_verbosity) {
        LogDebug("l1t|Global")
                << "\n**** GtBoard fill DAQ Records for bx= " << iBxInEvent
                << std::endl;

    }

// Set header information
    m_uGtAlgBlk.setOrbitNr((unsigned int)(orbNr & 0xFFFFFFFF));
    m_uGtAlgBlk.setbxNr((bxNr & 0xFFFF));
    m_uGtAlgBlk.setbxInEventNr((iBxInEvent & 0xF));

// Set the header information and Final OR
    int finalOR = 0x0;     
    if(m_algFinalOr) finalOR  = (finalOR | 0x2);  
    if(m_uGtFinalBoard) {
       finalOR = (finalOR | 0x8);
       if( (finalOR >>1) & 0x1 ) finalOR = (finalOR | 0x1);
    }
    m_uGtAlgBlk.setFinalOR(finalOR);
    

    uGtAlgRecord->push_back(iBxInEvent, m_uGtAlgBlk);

}

// Fill DAQ Record
void l1t::GtBoard::fillExtRecord(int iBxInEvent, 
				    std::auto_ptr<GlobalExtBlkBxCollection>& uGtExtRecord,
				    cms_uint64_t orbNr,
				    int bxNr
				    ) 
{

    if (m_verbosity) {
        LogDebug("l1t|Global")
                << "\n**** Board fill DAQ Records for bx= " << iBxInEvent
                << std::endl;

    }
// Set header information
    m_uGtExtBlk.setOrbitNr((unsigned int)(orbNr & 0xFFFFFFFF));
    m_uGtExtBlk.setbxNr((bxNr & 0xFFFF));
    m_uGtExtBlk.setbxInEventNr((iBxInEvent & 0xF));

    uGtExtRecord->push_back(iBxInEvent, m_uGtExtBlk);

}


// clear GTL
void l1t::GtBoard::reset() {

  resetMu();
  resetCalo();

  m_uGtAlgBlk.reset();
  m_uGtExtBlk.reset();

  m_gtlDecisionWord.reset();
  m_gtlAlgorithmOR.reset();
  
   

}

// clear muon
void l1t::GtBoard::resetMu() {

  m_candL1Mu->clear();
  m_candL1Mu->setBXRange( m_bxFirst_, m_bxLast_ );

}

// clear calo
void l1t::GtBoard::resetCalo() {

  m_candL1EG->clear();
  m_candL1Tau->clear();
  m_candL1Jet->clear();
  m_candL1EtSum->clear();

  m_candL1EG->setBXRange( m_bxFirst_, m_bxLast_ );
  m_candL1Tau->setBXRange( m_bxFirst_, m_bxLast_ );
  m_candL1Jet->setBXRange( m_bxFirst_, m_bxLast_ );
  m_candL1EtSum->setBXRange( m_bxFirst_, m_bxLast_ );

}

// print Global Muon Trigger data received by GTL
void l1t::GtBoard::printGmtData(const int iBxInEvent) const {

    LogTrace("l1t|Global")
            << "\nl1t::L1GlobalTrigger: GMT data received for BxInEvent = "
            << iBxInEvent << std::endl;

    int nrL1Mu = m_candL1Mu->size(iBxInEvent);
    LogTrace("l1t|Global")
            << "Number of GMT muons = " << nrL1Mu << "\n"
            << std::endl;
/*
    for (std::vector<const L1MuGMTCand*>::const_iterator iter =
            m_candL1Mu->begin(); iter != m_candL1Mu->end(); iter++) {

        LogTrace("l1t|Global") << *(*iter) << std::endl;

    }
*/
    LogTrace("l1t|Global") << std::endl;

}
