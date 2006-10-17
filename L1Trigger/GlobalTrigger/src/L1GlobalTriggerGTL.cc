/**
 * \class L1GlobalTriggerGTL
 * 
 * 
 * 
 * Description: Global Trigger Logic board, see header file for details 
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
#include "L1Trigger/GlobalTrigger/interface/L1GlobalTriggerGTL.h"

// system include files
#include <vector>

// user include files
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"

#include "L1Trigger/GlobalTrigger/interface/L1GlobalTrigger.h"
#include "L1Trigger/GlobalTrigger/interface/L1GlobalTriggerSetup.h"
#include "L1Trigger/GlobalTrigger/interface/L1GlobalTriggerConfig.h"

#include "L1Trigger/GlobalTrigger/interface/L1GlobalTriggerConditions.h"
#include "L1Trigger/GlobalTrigger/interface/L1GlobalTriggerMuonTemplate.h"
#include "L1Trigger/GlobalTrigger/interface/L1GlobalTriggerCaloTemplate.h"
#include "L1Trigger/GlobalTrigger/interface/L1GlobalTriggerEsumsTemplate.h"
#include "L1Trigger/GlobalTrigger/interface/L1GlobalTriggerJetCountsTemplate.h"

#include "L1Trigger/GlobalMuonTrigger/interface/L1MuGlobalMuonTrigger.h"

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// forward declarations

// constructor
L1GlobalTriggerGTL::L1GlobalTriggerGTL(const L1GlobalTrigger& gt) 
    : 
    m_GT(gt), 
    glt_muonCand( new GMTVector(L1GlobalTriggerReadoutRecord::NumberL1Muons) ) {

    glt_algorithmOR.reset();
    glt_decision.reset(); 
    
    for ( int i = 0; i < 9; i++) {
        glt_cond[i].reset();
        glt_algos.push_back( particleBlock() );
        glt_particleConditions.push_back( new conditions( L1GlobalTriggerSetup::MaxItem ) );
    }

    glt_muonCand->reserve(L1GlobalTriggerReadoutRecord::NumberL1Muons);
    
}

// destructor
L1GlobalTriggerGTL::~L1GlobalTriggerGTL() { 
    
    reset();
    glt_muonCand->clear();
    delete glt_muonCand;
    
    for (conditionContainer::iterator iter = glt_particleConditions.begin(); 
            iter != glt_particleConditions.end(); iter++ ) {
        (*iter)->clear();
        delete *iter;
    }
}

// operations

// receive input data

void L1GlobalTriggerGTL::receiveData(edm::Event& iEvent) {

    reset(); 
    
    // disabling Global Muon input 

    const L1GlobalTriggerConfig* gtConf = m_GT.gtSetup()->gtConfig();

    if ( gtConf != 0 ) { 
        if ( gtConf->getInputMask()[1] ) {

            LogDebug("L1GlobalTriggerGTL") 
                << "\n**** Global Muon input disabled! \n  inputMask[1] = " 
                << gtConf->getInputMask()[1]
                << "     All candidates empty." << "\n**** \n"
                << std::endl;

            // set all muon candidates empty                
            for ( unsigned int iMuon = 0; iMuon < L1GlobalTriggerReadoutRecord::NumberL1Muons; iMuon++ ) {
        
                L1GlobalTriggerGTL::MuonDataWord dataword = 0; 
                (*glt_muonCand)[iMuon] = new L1MuGMTCand( dataword );
            }
            return;
            
        }
    } else {
        throw cms::Exception("L1GtConfiguration")
         << "**** No configuration file exists! \n";
    }

    LogDebug("L1GlobalTriggerGTL") 
        << "**** L1GlobalTriggerGTL receiving muon data" 
        << std::endl;
    

    // get data from Global Muon Trigger
    // the GLT receives 4 * 26 bits from the Global Muon Trigger

    edm::Handle<std::vector<L1MuGMTCand> > muonData;
    iEvent.getByLabel("gmt", muonData);

    for ( unsigned int iMuon = 0; iMuon < L1GlobalTriggerReadoutRecord::NumberL1Muons; iMuon++ ) {

        L1GlobalTriggerGTL::MuonDataWord dataword = 0; 
        unsigned int nMuon = 0;

        for ( std::vector<L1MuGMTCand>::const_iterator itMuon = muonData->begin(); 
            itMuon != muonData->end(); itMuon++ ) {
            
            // retrieving info for BX = 0 only !!
            if ( (*itMuon).bx() == 0 ) {
                if ( nMuon == iMuon ) {
                    dataword = (*itMuon).getDataWord();
                    break;
                }
                nMuon++;
            }
        }
        (*glt_muonCand)[iMuon] = new L1MuGMTCand( dataword );
    }
    
    printGmtData();    

}

// run GTL
void L1GlobalTriggerGTL::run() {
        
//    LogDebug ("Trace") << "**** L1GlobalTriggerGTL run " << std::endl;

    // try xml conditions
    const L1GlobalTriggerConfig* gtConf = m_GT.gtSetup()->gtConfig();
     
    if (gtConf != 0) {
        unsigned int chipnr;
        LogDebug("L1GlobalTriggerGTL") 
            << "\n***** Result of the XML-conditions \n" 
            << std::endl;

        for (chipnr = 0; chipnr < L1GlobalTriggerConfig::NumberConditionChips; chipnr++) { 
            LogTrace("L1GlobalTriggerGTL") 
                << "\n---------Chip " << chipnr + 1 << " ----------\n" 
                << std::endl; 
    
            for (L1GlobalTriggerConfig::ConditionsMap::const_iterator 
                itxml = gtConf->conditionsmap[chipnr].begin(); 
                itxml != gtConf->conditionsmap[chipnr].end(); itxml++) {

                std::string condName = itxml->first;

                LogTrace("L1GlobalTriggerGTL") 
                    << "\n===============================================\n" 
                    << "Evaluating condition: " << condName 
                    << "\n" 
                    << std::endl;

                bool condResult = itxml->second->blockCondition_sr(); 
                
                LogTrace("L1GlobalTriggerGTL")
                    << condName << " result: " << condResult 
                    << std::endl;
            
            }
        }
        
        LogTrace("L1GlobalTriggerGTL") 
            << "\n---------- Prealgorithms: evaluation ---------\n" 
            << std::endl;
        for (L1GlobalTriggerConfig::ConditionsMap::const_iterator 
            itxml  = gtConf->prealgosmap.begin(); 
            itxml != gtConf->prealgosmap.end(); itxml++) {
                                
            std::string prealgoName = itxml->first;
            bool prealgoResult = itxml->second->blockCondition_sr(); 
            std::string prealgoExpression = itxml->second->getNumericExpression();
            
            LogTrace("L1GlobalTriggerGTL")
                << "  " << prealgoName << " : " << prealgoResult 
                << " = " << prealgoExpression
                << std::endl;

        }
        LogTrace("L1GlobalTriggerGTL") 
            << "\n---------- End of prealgorithm list ---------\n" 
            << std::endl;

        LogTrace("L1GlobalTriggerGTL") 
            << "\n---------- Algorithms: evaluation ----------\n" 
            << std::endl;
        for (L1GlobalTriggerConfig::ConditionsMap::const_iterator 
            itxml  = gtConf->algosmap.begin(); 
            itxml != gtConf->algosmap.end(); itxml++) {
            
            std::string algoName = itxml->first;
            bool algoResult = itxml->second->blockCondition_sr(); 
            std::string algoExpression = itxml->second->getNumericExpression();
            
            LogTrace("L1GlobalTriggerGTL")
                << "  " << algoName << " : " << algoResult 
                << " = " << algoExpression
                << std::endl;

        }
        LogTrace("L1GlobalTriggerGTL") 
            << "\n---------- End of algorithm list ----------\n" 
            << std::endl;

        // set the pins
        if (gtConf->getVersion() == L1GlobalTriggerConfig::VERSION_PROTOTYPE) {
            // prototype version uses algos
            for (L1GlobalTriggerConfig::ConditionsMap::const_iterator 
                itxml  = gtConf->algosmap.begin(); 
                itxml != gtConf->algosmap.end(); itxml++) {

                std::string algoName = itxml->first;
    
                if (itxml->second->getLastResult()) {
                    if (itxml->second->getOutputPin() > 0) {
                        glt_algorithmOR.set( itxml->second->getOutputPin()-1);
                    }
                }

                edm::LogVerbatim("L1GlobalTriggerGTL")
                    << " Bit " << itxml->second->getOutputPin()-1
                    << " " << algoName << ": "  
                    << std::endl;
            }

        } else {
            // final version use prealgos
            for (L1GlobalTriggerConfig::ConditionsMap::const_iterator 
                itxml  = gtConf->prealgosmap.begin(); 
                itxml != gtConf->prealgosmap.end(); itxml++) {

                std::string prealgoName = itxml->first;
                
                // algo( i ) = prealgo( i+1 ), i = 0, MaxNumberAlgorithms
                int prealgoNumber = itxml->second->getAlgoNumber();               
    
                if (itxml->second->getLastResult()) {
                    if (prealgoNumber > 0) {
                        glt_algorithmOR.set( prealgoNumber-1);
                        
                    }
                }
                
                edm::LogVerbatim("L1GlobalTriggerGTL")
                    << " Bit " << prealgoNumber-1
                    << " " << prealgoName << ": " << glt_algorithmOR[ prealgoNumber-1 ]
                    << std::endl;

            }
        }
    }
    
}

// clear GTL
void L1GlobalTriggerGTL::reset() {
    
    GMTVector::iterator iter;
    for ( iter = glt_muonCand->begin(); iter < glt_muonCand->end(); iter++ ) {
        if (*iter) {
            delete (*iter);
            *iter = 0;
        }
    }

    glt_decision.reset();
    
    // TODO get rid of 9 hardwired!!!!!
    for (int i = 0; i < 9; i++) {
        glt_cond[i].reset();
    }

    glt_algorithmOR.reset();

}

// print Global Muon Trigger data

void L1GlobalTriggerGTL::printGmtData() const {
    
    edm::LogVerbatim("L1GlobalTriggerGTL") 
        << "\nMuon data received by GTL:" << std::endl;
    
    for ( GMTVector::iterator iter = glt_muonCand->begin(); 
            iter < glt_muonCand->end(); iter++ ) {

        LogTrace("L1GlobalTriggerGTL") 
            << "\nIterator value = " << (*iter) << std::endl;

        (*iter)->print();

    }

    edm::LogVerbatim("L1GlobalTriggerGTL") << std::endl;

}


const std::vector<L1GlobalTriggerGTL::MuonDataWord> L1GlobalTriggerGTL::getMuons() const { 

    std::vector<L1GlobalTriggerGTL::MuonDataWord> muon(L1GlobalTriggerReadoutRecord::NumberL1Muons);

    for (unsigned int i = 0; i < L1GlobalTriggerReadoutRecord::NumberL1Muons; i++) {
        muon[i] = (*glt_muonCand)[i]->getDataWord(); 
    }

    return muon;
}
