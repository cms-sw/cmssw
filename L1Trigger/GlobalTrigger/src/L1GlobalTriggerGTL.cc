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
 * $Date:$
 * $Revision:$
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

// run GTL
void L1GlobalTriggerGTL::run() {
        
    LogDebug ("Trace") << "**** L1GlobalTriggerGTL run " << std::endl;

    // try xml conditions
    const L1GlobalTriggerConfig* gtConf = m_GT.gtSetup()->gtConfig();
     
    if (gtConf != NULL) {
        unsigned int chipnr;
        edm::LogInfo("L1GlobalTriggerGTL") 
            << "***** Result of the XML-conditions " 
            << std::endl;

        for (chipnr = 0; chipnr < L1GlobalTriggerConfig::max_chips; chipnr++) { 
            edm::LogInfo("L1GlobalTriggerGTL") 
                << "---------Chip " << chipnr + 1 << " ----------" 
                << std::endl; 
    
            for (L1GlobalTriggerConfig::ConditionsMap::const_iterator 
                itxml = gtConf->conditionsmap[chipnr].begin(); 
                itxml != gtConf->conditionsmap[chipnr].end(); itxml++) {

            	edm::LogInfo("L1GlobalTriggerGTL") 
                    << itxml->first << ": " << itxml->second->blockCondition_sr() 
                    << std::endl;
            
            }
        }
        
        edm::LogInfo("L1GlobalTriggerGTL") 
            << "---------- Prealgos ---------" 
            << std::endl;
        for (L1GlobalTriggerConfig::ConditionsMap::const_iterator 
            itxml = gtConf->prealgosmap.begin(); 
            itxml!= gtConf->prealgosmap.end(); itxml++) {
                
            edm::LogInfo("L1GlobalTriggerGTL") 
                << itxml->first << ": " << itxml->second->blockCondition_sr() 
                << " = " << itxml->second->getNumericExpression() << std::endl;
                
        }

        edm::LogInfo("L1GlobalTriggerGTL") 
            << "---------- Algos ----------" 
            << std::endl;
        for (L1GlobalTriggerConfig::ConditionsMap::const_iterator 
            itxml  = gtConf->algosmap.begin(); 
            itxml != gtConf->algosmap.end(); itxml++) {
            
            edm::LogInfo("L1GlobalTriggerGTL") 
                << itxml->first << ": " << itxml->second->blockCondition_sr() 
                << " = " << itxml->second->getNumericExpression() 
                << std::endl;

        }

        // set the pins
        if (gtConf->getVersion() == L1GlobalTriggerConfig::VERSION_PROTOTYPE) {
            // prototype version uses algos
            for (L1GlobalTriggerConfig::ConditionsMap::const_iterator 
                itxml  = gtConf->algosmap.begin(); 
                itxml != gtConf->algosmap.end(); itxml++) {

                if (itxml->second->getLastResult()) {
                    if (itxml->second->getOutputPin() > 0) {
                        glt_algorithmOR.set( itxml->second->getOutputPin()-1);
                    }
                }
            }
        } else {
            // final version use prealgos
            for (L1GlobalTriggerConfig::ConditionsMap::const_iterator 
                itxml  = gtConf->prealgosmap.begin(); 
                itxml != gtConf->prealgosmap.end(); itxml++) {

                if (itxml->second->getLastResult()) {
                    if (itxml->second->getOutputPin() > 0) {
                        glt_algorithmOR.set( itxml->second->getOutputPin()-1);
                    }
                }
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

void L1GlobalTriggerGTL::print() const {
    
    edm::LogInfo("L1GlobalTriggerGTL") << " muon data :" << std::endl;
    
    for ( GMTVector::iterator iter = glt_muonCand->begin(); 
            iter < glt_muonCand->end(); iter++ ) {

        edm::LogInfo("L1GlobalTriggerGTL") << " iter = " << (*iter) << std::endl;
        (*iter)->print();
    }

    edm::LogInfo("L1GlobalTriggerGTL") << std::endl;

}

// receive input data

void L1GlobalTriggerGTL::receiveData(edm::Event& iEvent) {

    reset(); 
    
    // disabling Global Muon input 

    const L1GlobalTriggerConfig* gtConf = m_GT.gtSetup()->gtConfig();

    if ( gtConf != NULL) { 
        if ( !gtConf->inputMask()[1] ) {
            return;
        }
    }

    edm::LogInfo("L1GlobalTriggerGTL") 
        << "**** L1GlobalTriggerGTL receiving muon data" 
        << std::endl;
    

    // TODO add the code
    
    // get data from Global Muon Trigger
    // the GLT receives 4 * 26 bits from the Global Muon Trigger

}


const std::vector<L1GlobalTriggerGTL::MuonDataWord> L1GlobalTriggerGTL::getMuons() const { 

    std::vector<L1GlobalTriggerGTL::MuonDataWord> muon(L1GlobalTriggerReadoutRecord::NumberL1Muons);

    for (unsigned int i = 0; i < L1GlobalTriggerReadoutRecord::NumberL1Muons; i++) {
        muon[i] = (*glt_muonCand)[i]->getDataWord(); 
    }

    return muon;
}

void L1GlobalTriggerGTL::defineConditions() {

    
    // muon particle condition blocks
    edm::LogInfo("L1GlobalTriggerGTL") 
        << "**** L1GlobalTriggerGTL: Building Muon Particle Condition Blocks" 
        << std::endl;

    for (particleBlock::const_iterator iter = (glt_algos[0]).begin(); 
    iter != (glt_algos[0]).end(); 
    iter++ ) {

    // TODO test here
    edm::LogInfo("L1GlobalTriggerGTL") 
        << "**** ITERATOR " << *iter 
        << std::endl;
        
//        (*glt_particleConditions[0])[(*iter)] = new L1GlobalTriggerMuonTemplate( (*iter) );
    }
    
    // non-isolated electron particle blocks
    edm::LogInfo("L1GlobalTriggerGTL") 
        << "**** L1GlobalTriggerGTL: Building Non Isolated Electron Particle Condition Blocks" 
        << std::endl;
    
    for (particleBlock::const_iterator iter = (glt_algos[1]).begin(); 
        iter != (glt_algos[1]).end();  
        iter++ ) {
        
//        (*glt_particleConditions[1])[(*iter)] = new L1GlobalTriggerCaloTemplate( (*iter) ); 
    }
            
    // isolated electron particle blocks
   
    edm::LogInfo("L1GlobalTriggerGTL") 
        << "**** L1GlobalTriggerGTL: Building Isolated Electron Particle Condition Blocks" 
        << std::endl;
    
    for (particleBlock::const_iterator iter = (glt_algos[2]).begin(); 
        iter != (glt_algos[2]).end(); 
        iter++ ) {

//        (*glt_particleConditions[2])[(*iter)] = new L1GlobalTriggerCaloTemplate( (*iter) ); 
    }
        
    // central jet particle blocks
    edm::LogInfo("L1GlobalTriggerGTL") 
        << "**** L1GlobalTriggerGTL: Building Central Jet Particle Condition Blocks" 
        << std::endl;
    
    for (particleBlock::const_iterator iter = (glt_algos[3]).begin(); 
        iter != (glt_algos[3]).end(); 
        iter++ ) {
        
//        (*glt_particleConditions[3])[(*iter)] = new L1GlobalTriggerCaloTemplate( (*iter) ); 
    }
        
    // forward jet particle blocks
    edm::LogInfo("L1GlobalTriggerGTL") 
        << "**** L1GlobalTriggerGTL: Building Forward Jet Particle Condition Blocks" 
        << std::endl;
    
    for (particleBlock::const_iterator iter = (glt_algos[4]).begin(); 
        iter != (glt_algos[4]).end(); 
        iter++ ) {
    
//        (*glt_particleConditions[4])[(*iter)] = new L1GlobalTriggerCaloTemplate( (*iter) ); 
    }
        
    // tau jet particle blocks
    edm::LogInfo("L1GlobalTriggerGTL") 
        << "**** L1GlobalTriggerGTL: Building Tau Jet Particle Condition Blocks" 
        << std::endl;
    
    for (particleBlock::const_iterator iter = (glt_algos[5]).begin(); 
        iter != (glt_algos[5]).end(); 
        iter++ ) {
        
//        (*glt_particleConditions[5])[(*iter)] = new L1GlobalTriggerCaloTemplate( (*iter) ); 
    }
 
    // total Et condition blocks
    edm::LogInfo("L1GlobalTriggerGTL") 
        << "**** L1GlobalTriggerGTL: Building Total Et Condition Blocks" 
        << std::endl;
    
    for (particleBlock::const_iterator iter = (glt_algos[6]).begin(); 
        iter != (glt_algos[6]).end(); 
        iter++ ) { 
        
//        (*glt_particleConditions[6])[(*iter)] = new L1GlobalTriggerEsumsTemplate( (*iter) );
    }

    // missing Et condition blocks
    edm::LogInfo("L1GlobalTriggerGTL") 
        << "**** L1GlobalTriggerGTL: Building Missing Et Condition Blocks" 
        << std::endl;
    
    for (particleBlock::const_iterator iter = (glt_algos[7]).begin();
        iter != (glt_algos[7]).end(); 
        iter++ ) {         
        
//        (*glt_particleConditions[7])[(*iter)] = new L1GlobalTriggerEsumsTemplate( (*iter) );
    }
    
    // total hadron Et condition blocks
    edm::LogInfo("L1GlobalTriggerGTL") 
        << "**** L1GlobalTriggerGTL: Building TotalHadron Et Condition Blocks" 
        << std::endl;
    
    for (particleBlock::const_iterator iter = (glt_algos[8]).begin();
        iter != (glt_algos[8]).end(); 
        iter++ ) {         
        
//        (*glt_particleConditions[8])[(*iter)] = new L1GlobalTriggerEsumsTemplate( (*iter) );
    }
    
    // jet counts condition blocks
    edm::LogInfo("L1GlobalTriggerGTL") 
        << "**** L1GlobalTriggerGTL: Building Jet Counts Condition Blocks" 
        << std::endl;

    for (particleBlock::const_iterator iter = (glt_algos[9]).begin(); 
        iter != (glt_algos[9]).end(); 
        iter++ ) {
        
//        (*glt_particleConditions[9])[(*iter)] = new L1GlobalTriggerJetCountsTemplate( (*iter) );
    }
}


