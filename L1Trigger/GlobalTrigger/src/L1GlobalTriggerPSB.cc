/**
 * \class L1GlobalTriggerPSB
 * 
 * 
 * 
 * Description: Pipelined Synchronising Buffer, see header file for details 
 * Implementation:
 *    <TODO: enter implementation details>
 *    <TODO: this class has to be changed to follow the hardware>
 *   
 * \author: M. Fierro            - HEPHY Vienna - ORCA version 
 * \author: Vasile Mihai Ghete   - HEPHY Vienna - CMSSW version 
 * 
 * $Date:$
 * $Revision:$
 *
 */

// this class header
#include "L1Trigger/GlobalTrigger/interface/L1GlobalTriggerPSB.h"

// system include files
#include <bitset>

// user include files
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"

#include "DataFormats/L1GlobalTrigger/interface/L1TriggerObject.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctCand.h"

#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctEmCand.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctJetCand.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctEtSums.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctJetCounts.h"

#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctCollections.h"

#include "L1Trigger/GlobalTrigger/interface/L1GlobalTrigger.h"
#include "L1Trigger/GlobalTrigger/interface/L1GlobalTriggerSetup.h"
#include "L1Trigger/GlobalTrigger/interface/L1GlobalTriggerConfig.h"

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// forward declarations

// constructor
L1GlobalTriggerPSB::L1GlobalTriggerPSB(L1GlobalTrigger& gt) 
    : m_GT(gt), 
    glt_electronList        ( new CaloVector(L1GlobalTriggerReadoutRecord::NumberL1Electrons) ),
    glt_isolatedElectronList( new CaloVector(L1GlobalTriggerReadoutRecord::NumberL1IsolatedElectrons) ),
    glt_centralJetList      ( new CaloVector(L1GlobalTriggerReadoutRecord::NumberL1CentralJets) ),
    glt_forwardJetList      ( new CaloVector(L1GlobalTriggerReadoutRecord::NumberL1ForwardJets) ),
    glt_tauJetList          ( new CaloVector(L1GlobalTriggerReadoutRecord::NumberL1TauJets) ),
    glt_missingEtList(0),
    glt_totalEtList(0),
    glt_totalHtList(0),
    glt_jetCountsList(0) { 

    glt_electronList->reserve(L1GlobalTriggerReadoutRecord::NumberL1Electrons);
    glt_isolatedElectronList->reserve(L1GlobalTriggerReadoutRecord::NumberL1IsolatedElectrons);
    glt_centralJetList->reserve(L1GlobalTriggerReadoutRecord::NumberL1CentralJets);
    glt_forwardJetList->reserve(L1GlobalTriggerReadoutRecord::NumberL1ForwardJets);
    glt_tauJetList->reserve(L1GlobalTriggerReadoutRecord::NumberL1TauJets);

}

// destructor
L1GlobalTriggerPSB::~L1GlobalTriggerPSB() { 

    reset();
    glt_electronList->clear();
    glt_isolatedElectronList->clear();
    glt_centralJetList->clear();
    glt_forwardJetList->clear();
    glt_tauJetList->clear();

    delete glt_electronList;
    delete glt_isolatedElectronList;
    delete glt_centralJetList;
    delete glt_forwardJetList;
    delete glt_tauJetList;

    if ( glt_missingEtList ) delete glt_missingEtList; 
    if ( glt_totalEtList )   delete glt_totalEtList;
    if ( glt_totalHtList )   delete glt_totalHtList;

    if ( glt_jetCountsList )  delete glt_jetCountsList;
        
}

// operations

// receive input data

void L1GlobalTriggerPSB::receiveData(edm::Event& iEvent) {

    reset();
    
    // disabling calorimeter input

    const L1GlobalTriggerConfig* gtConf = m_GT.gtSetup()->gtConfig();

    if ( gtConf != 0 ) { 
        if ( gtConf->getInputMask()[0] ) {

            LogDebug("L1GlobalTriggerPSB") 
                << "\n**** Calorimeter input disabled! \n     inputMask[0] = " 
                << gtConf->getInputMask()[0] 
                << "     All candidates empty." << "\n**** \n"
                << std::endl;
                
            // empty electrons
            for ( unsigned int i = 0; i < L1GlobalTriggerReadoutRecord::NumberL1Electrons; i++ ) {
        
                L1GlobalTriggerPSB::CaloDataWord dataword = 0; 
                
                bool isolation = false;
                (*glt_electronList)[i] = new L1GctEmCand( dataword, isolation );
                
            }
        
            // empty isolated electrons
            for ( unsigned int i = 0; i < L1GlobalTriggerReadoutRecord::NumberL1IsolatedElectrons; i++ ) {
        
                L1GlobalTriggerPSB::CaloDataWord dataword = 0; 
                
                bool isolation = true;
                (*glt_isolatedElectronList)[i] = new L1GctEmCand( dataword, isolation );
        
            }
        
            // empty central jets 
            for ( unsigned int i = 0; i < L1GlobalTriggerReadoutRecord::NumberL1CentralJets; i++ ) {
        
                L1GlobalTriggerPSB::CaloDataWord dataword = 0; 
                
                bool isTau = false;
                bool isFor = false;
                (*glt_centralJetList)[i] = new L1GctJetCand( dataword, isTau, isFor );
        
            }
            
            // empty forward jets 
            for ( unsigned int i = 0; i < L1GlobalTriggerReadoutRecord::NumberL1ForwardJets; i++ ) {
        
                L1GlobalTriggerPSB::CaloDataWord dataword = 0; 
                
                bool isTau = false;
                bool isFor = true;
                (*glt_forwardJetList)[i] = new L1GctJetCand( dataword, isTau, isFor );
        
            }
        
            // empty tau jets 
            for ( unsigned int i = 0; i < L1GlobalTriggerReadoutRecord::NumberL1TauJets; i++ ) {
        
                L1GlobalTriggerPSB::CaloDataWord dataword = 0; 
                
                bool isTau = true;
                bool isFor = false;
                (*glt_tauJetList)[i] = new L1GctJetCand( dataword, isTau, isFor );
        
            }

            // null values for energy sums and jet counts
            glt_missingEtList = new L1GctEtMiss(); 
            glt_totalEtList   = new L1GctEtTotal(); 
            glt_totalHtList   = new L1GctEtHad(); 
            
            glt_jetCountsList = new L1GctJetCounts(); 

            return;

        }
    } else {
        throw cms::Exception("L1GtConfiguration")
         << "**** No configuration file exists! \n";
    }

    // reading data
    
    LogDebug("L1GlobalTriggerPSB") 
        << "\n**** L1GlobalTriggerPSB receiving calorimeter data\n" 
        << std::endl;

    edm::Handle<L1GctEmCandCollection> emCands;
    edm::Handle<L1GctEmCandCollection> isoEmCands;
    
    edm::Handle<L1GctJetCandCollection> cenJets;
    edm::Handle<L1GctJetCandCollection> forJets;
    edm::Handle<L1GctJetCandCollection> tauJets;
    
    edm::Handle<L1GctEtMiss>  missEt;
    edm::Handle<L1GctEtTotal> totalEt;
    edm::Handle<L1GctEtHad>   totalHt;
    
    edm::Handle<L1GctJetCounts> jetCounts;
 
    iEvent.getByLabel("gct", "nonIsoEm", emCands);
    iEvent.getByLabel("gct", "isoEm",    isoEmCands);

    iEvent.getByLabel("gct", "cenJets", cenJets);
    iEvent.getByLabel("gct", "forJets", forJets);
    iEvent.getByLabel("gct", "tauJets", tauJets);

    iEvent.getByLabel("gct", missEt);
    iEvent.getByLabel("gct", totalEt);
    iEvent.getByLabel("gct", totalHt);
  
    // TODO FIXME un-comment when the "jet counts" collection is written     
//    iEvent.getByLabel("gct", jetCounts);
    
    // electrons
    for ( unsigned int i = 0; i < L1GlobalTriggerReadoutRecord::NumberL1Electrons; i++ ) {

        L1GlobalTriggerPSB::CaloDataWord dataword = 0; 
        unsigned int nElec = 0;

        for (L1GctEmCandCollection::const_iterator it = emCands->begin(); 
            it != emCands->end(); it++) {
            
//            // retrieving info for BX = 0 only !!
//            if ( (*it).bx() == 0 ) {
                if ( nElec == i ) {
                    dataword = (*it).raw();                                    
                    break;
                }
                nElec++;
//            } // TODO un-comment / change according to bunch-cross treatment in GCT 
        }

        bool isolation = false;
        (*glt_electronList)[i] = new L1GctEmCand( dataword, isolation );
        
    }

    // isolated electrons
    for ( unsigned int i = 0; i < L1GlobalTriggerReadoutRecord::NumberL1IsolatedElectrons; i++ ) {

        L1GlobalTriggerPSB::CaloDataWord dataword = 0; 
        unsigned int nElec = 0;

        for (L1GctEmCandCollection::const_iterator it = isoEmCands->begin(); 
            it != isoEmCands->end(); it++) {
            
//            // retrieving info for BX = 0 only !!
//            if ( (*it).bx() == 0 ) {
                if ( nElec == i ) {
                    dataword = (*it).raw();
                    break;
                }
                nElec++;
//            } // TODO un-comment / change according to bunch-cross treatment in GCT 
        }
                    
        bool isolation = true;
        (*glt_isolatedElectronList)[i] = new L1GctEmCand( dataword, isolation );

    }

    // central jets 
    for ( unsigned int i = 0; i < L1GlobalTriggerReadoutRecord::NumberL1CentralJets; i++ ) {

        L1GlobalTriggerPSB::CaloDataWord dataword = 0; 
        unsigned int nJet = 0;

        for (L1GctJetCandCollection::const_iterator it = cenJets->begin(); 
            it != cenJets->end(); it++) {
            
//            // retrieving info for BX = 0 only !!
//            if ( (*it).bx() == 0 ) {
                if ( nJet == i ) {
                    dataword = (*it).raw();
                    break;
                }
                nJet++;
//            } // TODO un-comment / change according to bunch-cross treatment in GCT 
        }

        bool isTau = false;
        bool isFor = false;
        (*glt_centralJetList)[i] = new L1GctJetCand( dataword, isTau, isFor );

    }
    
    // forward jets 
    for ( unsigned int i = 0; i < L1GlobalTriggerReadoutRecord::NumberL1ForwardJets; i++ ) {

        L1GlobalTriggerPSB::CaloDataWord dataword = 0; 
        unsigned int nJet = 0;

        for (L1GctJetCandCollection::const_iterator it = forJets->begin(); 
            it != forJets->end(); it++) {
            
//            // retrieving info for BX = 0 only !!
//            if ( (*it).bx() == 0 ) {
                if ( nJet == i ) {
                    dataword = (*it).raw();
                    break;
                }
                nJet++;
//            } // TODO un-comment / change according to bunch-cross treatment in GCT 
        }
        
        bool isTau = false;
        bool isFor = true;
        (*glt_forwardJetList)[i] = new L1GctJetCand( dataword, isTau, isFor );

    }

    // tau jets 
    for ( unsigned int i = 0; i < L1GlobalTriggerReadoutRecord::NumberL1TauJets; i++ ) {

        L1GlobalTriggerPSB::CaloDataWord dataword = 0; 
        unsigned int nJet = 0;

        for (L1GctJetCandCollection::const_iterator it = tauJets->begin(); 
            it != tauJets->end(); it++) {
            
//            // retrieving info for BX = 0 only !!
//            if ( (*it).bx() == 0 ) {
                if ( nJet == i ) {
                    dataword = (*it).raw();
                    break;
                }
                nJet++;
//            } // TODO un-comment / change according to bunch-cross treatment in GCT 
        }
        
        bool isTau = true;
        bool isFor = false;
        (*glt_tauJetList)[i] = new L1GctJetCand( dataword, isTau, isFor );

    }
    
    glt_missingEtList = new L1GctEtMiss(  (*missEt).raw()  ); 
    glt_totalEtList   = new L1GctEtTotal( (*totalEt).raw() ); 
    glt_totalHtList   = new L1GctEtHad(   (*totalHt).raw() ); 
    
    
// TODO FIXME comment empty constructor and un-comment next when jet counts are written 
     glt_jetCountsList = new L1GctJetCounts();
//    glt_jetCountsList = new L1GctJetCounts( (*jetCounts) ); 
    
    printGctData();    

}


// clear PSB

void L1GlobalTriggerPSB::reset() {
    
    CaloVector::iterator iter;
    for ( iter = glt_electronList->begin(); iter < glt_electronList->end(); iter++ ) {
        if (*iter) {
            delete (*iter);
            *iter = 0;
        }
    }
    for ( iter = glt_isolatedElectronList->begin(); iter < glt_isolatedElectronList->end(); iter++ ) {
        if (*iter) {
            delete (*iter);
            *iter = 0;
        }
    }
    for ( iter = glt_centralJetList->begin(); iter < glt_centralJetList->end(); iter++ ) {
        if (*iter) {
             delete (*iter);
            *iter = 0;
        }
    }
    for ( iter = glt_forwardJetList->begin(); iter < glt_forwardJetList->end(); iter++ ) {
        if (*iter) {
             delete (*iter);
            *iter = 0;
        }
    }
    for ( iter = glt_tauJetList->begin(); iter < glt_tauJetList->end(); iter++ ) {
        if (*iter) {
            delete (*iter);
            *iter = 0;
        }
    }

    if ( glt_missingEtList ) { delete glt_missingEtList; glt_missingEtList = 0; }
    if ( glt_totalEtList )   { delete glt_totalEtList; glt_totalEtList = 0; }
    if ( glt_totalHtList )   { delete glt_totalHtList; glt_totalHtList = 0; }

    if ( glt_jetCountsList ) { delete glt_jetCountsList; glt_jetCountsList = 0; }

}

// print Global Calorimeter Trigger data
// use int to bitset conversion to print 
void L1GlobalTriggerPSB::printGctData() const {

    edm::LogInfo("L1GlobalTriggerPSB") 
        << "\n L1GlobalTrigger Calorimeter input data\n" << std::endl;

    CaloVector::const_iterator iter;

    edm::LogVerbatim("L1GlobalTriggerPSB") << "   GCT Non Isolated Electrons " << std::endl;
    for ( iter = glt_electronList->begin(); iter < glt_electronList->end(); iter++ ) {
//        edm::LogVerbatim("L1GlobalTriggerPSB") 
//            << std::bitset<L1GlobalTriggerReadoutRecord::NumberCaloBits>( (*iter)->rank() ) 
//            << std::bitset<L1GlobalTriggerReadoutRecord::NumberCaloBits>( (*iter)->etaIndex() ) 
//            << std::bitset<L1GlobalTriggerReadoutRecord::NumberCaloBits>( (*iter)->phiIndex() ) 
//            << std::endl;

        edm::LogVerbatim("L1GlobalTriggerPSB") 
            << "Rank = " << (*iter)->rank()
            << " Eta index = " << (*iter)->etaIndex() 
            << " Phi index = " << (*iter)->phiIndex()  
            << std::endl;
    }

    edm::LogVerbatim("L1GlobalTriggerPSB") << "   GCT Isolated Electrons " << std::endl;
    for ( iter = glt_isolatedElectronList->begin(); iter < glt_isolatedElectronList->end(); iter++ ) {
        edm::LogVerbatim("L1GlobalTriggerPSB") 
            << "Rank = " << (*iter)->rank()
            << " Eta index = " << (*iter)->etaIndex() 
            << " Phi index = " << (*iter)->phiIndex()  
            << std::endl;
    }

    edm::LogVerbatim("L1GlobalTriggerPSB") << "   GCT Central Jets " << std::endl;
    for ( iter = glt_centralJetList->begin(); iter < glt_centralJetList->end(); iter++ ) {
        edm::LogVerbatim("L1GlobalTriggerPSB") 
            << "Rank = " << (*iter)->rank()
            << " Eta index = " << (*iter)->etaIndex() 
            << " Phi index = " << (*iter)->phiIndex()  
            << std::endl;
    }

    edm::LogVerbatim("L1GlobalTriggerPSB") << "   GCT Forward Jets " << std::endl;
    for ( iter = glt_forwardJetList->begin(); iter < glt_forwardJetList->end(); iter++ ) {
        edm::LogVerbatim("L1GlobalTriggerPSB") 
            << "Rank = " << (*iter)->rank()
            << " Eta index = " << (*iter)->etaIndex() 
            << " Phi index = " << (*iter)->phiIndex()  
            << std::endl;
    }

    edm::LogVerbatim("L1GlobalTriggerPSB") << "   GCT Tau Jets " << std::endl;
    for ( iter = glt_tauJetList->begin(); iter < glt_tauJetList->end(); iter++ ) {
        edm::LogVerbatim("L1GlobalTriggerPSB") 
            << "Rank = " << (*iter)->rank()
            << " Eta index = " << (*iter)->etaIndex() 
            << " Phi index = " << (*iter)->phiIndex()  
            << std::endl;
    }

    edm::LogVerbatim("L1GlobalTriggerPSB") << "   GCT Missing Transverse Energy " << std::endl;
    if ( glt_missingEtList ) {
//        edm::LogVerbatim("L1GlobalTriggerPSB") << std::bitset<L1GlobalTriggerReadoutRecord::NumberMissingEtBits>(glt_missingEtList->et()) << std::endl;
//        edm::LogVerbatim("L1GlobalTriggerPSB") << std::bitset<L1GlobalTriggerReadoutRecord::NumberMissingEtBits>(glt_missingEtList->phi()) << std::endl;
        edm::LogVerbatim("L1GlobalTriggerPSB") << "ET  = " << glt_missingEtList->et() << std::endl;
        edm::LogVerbatim("L1GlobalTriggerPSB") << "phi = " << glt_missingEtList->phi() << std::endl;
    }

    edm::LogVerbatim("L1GlobalTriggerPSB") << "   GCT Total Transverse Energy " << std::endl;
    if ( glt_totalEtList )   {
//        edm::LogVerbatim("L1GlobalTriggerPSB") << std::bitset<L1GlobalTriggerReadoutRecord::NumberCaloBits>(glt_totalEtList->et()) << std::endl;
        edm::LogVerbatim("L1GlobalTriggerPSB") <<  "ET  = " << glt_totalEtList->et() << std::endl;
    }

    edm::LogVerbatim("L1GlobalTriggerPSB") << "   GCT Total Hadron Transverse Energy " << std::endl;
    if ( glt_totalHtList )   {
//        edm::LogVerbatim("L1GlobalTriggerPSB") << std::bitset<L1GlobalTriggerReadoutRecord::NumberCaloBits>(glt_totalHtList->et()) << std::endl;
        edm::LogVerbatim("L1GlobalTriggerPSB") <<  "ET  = " << glt_totalHtList->et() << std::endl;
    }

    edm::LogVerbatim("L1GlobalTriggerPSB") << "   GCT Jet Counts " << std::endl;
    if ( glt_jetCountsList ) {
        edm::LogVerbatim("L1GlobalTriggerPSB") << "To  be done" << std::endl; // TODO fix it when jet counts are available    
    }


// TODO ask GCT for a print function?    
//    CaloVector::iterator iter;
//    for ( iter = glt_electronList->begin(); iter < glt_electronList->end(); iter++ )     
//        (*iter)->print();
//    for ( iter = glt_isolatedElectronList->begin(); iter < glt_isolatedElectronList->end(); iter++ )     
//        (*iter)->print();
//    for ( iter = glt_centralJetList->begin(); iter < glt_centralJetList->end(); iter++ )     
//        (*iter)->print();
//    for ( iter = glt_forwardJetList->begin(); iter< glt_forwardJetList->end(); iter++ )     
//        (*iter)->print();
//    for ( iter = glt_tauJetList->begin(); iter < glt_tauJetList->end(); iter++ )     
//        (*iter)->print();
//
//    glt_missingEtList->print();
//    glt_totalEtList->print();
//    glt_totalHtList->print();
//    glt_jetCountsList->print();

}



// static data members
