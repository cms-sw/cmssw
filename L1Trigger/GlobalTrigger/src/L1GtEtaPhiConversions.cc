/**
 * \class L1GtEtaPhiConversions
 * 
 * 
 * Description: convert eta and phi between various L1 trigger objects.
 * 
 * Implementation:
 *    <TODO: enter implementation details>
 *   
 * \author: Vasile Mihai Ghete   - HEPHY Vienna 
 * 
 * $Date$
 * $Revision$
 *
 */

// this class header
#include "L1Trigger/GlobalTrigger/interface/L1GtEtaPhiConversions.h"

// system include files
#include <iostream>
#include <iomanip>

// user include files

//   base class

//
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMapFwd.h"

#include "CondFormats/L1TObjects/interface/L1MuTriggerScales.h"
#include "CondFormats/L1TObjects/interface/L1CaloGeometry.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/MessageLogger/interface/MessageDrop.h"

// constructor
L1GtEtaPhiConversions::L1GtEtaPhiConversions() {

    // do nothing

}

// destructor
L1GtEtaPhiConversions::~L1GtEtaPhiConversions() {

    // do nothing

}

// methods

// perform all conversions
void L1GtEtaPhiConversions::convert(const L1CaloGeometry* l1CaloGeometry,
        const L1MuTriggerScales* l1MuTriggerScales,
        const int ifCaloEtaNumberBits, const int ifMuEtaNumberBits) {
    
    // no bullet-proof method, depends on binning... 
    
    //
    // convert phi scale for muon (finer) to phi scale for calo (coarser)
    //
    
    unsigned int nrMuPhiBins = 144; // FIXME ask Ivan for size() ... 
    //unsigned int nrMuPhiBins = l1MuTriggerScales->getPhiScale()->size();

    unsigned int nrGctEmJetPhiBins = l1CaloGeometry->numberGctEmJetPhiBins();
    
    if (edm::isDebugEnabled() ) {
        
        LogTrace("L1GtEtaPhiConversions") << "\n nrGctEmJetPhiBins = "
                << nrGctEmJetPhiBins << "\n" << std::endl;

        for (unsigned int iCalo = 0; iCalo < nrGctEmJetPhiBins; ++iCalo) {
            double phiCaloLowEdge =
                    l1CaloGeometry->emJetPhiBinLowEdge(iCalo);
            double phiCaloHighEdge =
                    l1CaloGeometry->emJetPhiBinHighEdge(iCalo);

            LogTrace("L1GtEtaPhiConversions") << "Bin " << iCalo
                    << "\t phiCaloLowEdge = " << phiCaloLowEdge
                    << "\t phiCaloHighEdge = " << phiCaloHighEdge
                    << std::endl;

        }

        LogTrace("L1GtEtaPhiConversions") << "\n nrMuPhiBins = "
                << nrMuPhiBins << "\n" << std::endl;

        for (unsigned int iBin = 0; iBin < nrMuPhiBins; ++iBin) {
            double phiMuLowEdge = l1MuTriggerScales->getPhiScale()->getLowEdge(iBin);
            double phiMuHighEdge = l1MuTriggerScales->getPhiScale()->getHighEdge(iBin);

            LogTrace("L1GtEtaPhiConversions") << "Bin " << iBin
                    << "\t phiMuLowEdge = " << phiMuLowEdge
                    << "\t phiMuHighEdge = " << phiMuHighEdge << std::endl;

        }

        LogTrace("L1GtEtaPhiConversions") << "\n"
                << l1MuTriggerScales->getPhiScale()->print() << "\n" << std::endl;

    }
    
    // 
    m_lutPhiMuCalo.clear(); 
    m_lutPhiMuCalo.resize(nrMuPhiBins); 
    
    for (unsigned int phiMuInd = 0; phiMuInd < nrMuPhiBins; ++phiMuInd) {
        
        double phiMuLowEdge = l1MuTriggerScales->getPhiScale()->getLowEdge(phiMuInd);
        double phiMuHighEdge = l1MuTriggerScales->getPhiScale()->getHighEdge(phiMuInd);

        for (unsigned int iCalo = nrGctEmJetPhiBins; iCalo <= nrGctEmJetPhiBins; --iCalo) {
            
            double phiCaloLowEdge = l1CaloGeometry->emJetPhiBinLowEdge(iCalo);
            double phiCaloHighEdge = l1CaloGeometry->emJetPhiBinHighEdge(iCalo);
            
            if (phiMuLowEdge >= phiCaloLowEdge) {
                m_lutPhiMuCalo[phiMuInd] = iCalo%nrGctEmJetPhiBins;
                                    
                LogTrace("L1GtEtaPhiConversions") 
                        << " phiMuLowEdge[" << phiMuInd << "] = " << phiMuLowEdge                            
                        << " phiMuHighEdge[" << phiMuInd << "] = " << phiMuHighEdge                            
                        << "\n phiCaloLowEdge[" << iCalo << "] = " << phiCaloLowEdge
                        << " phiCaloHighEdge[" << iCalo << "] = " << phiCaloHighEdge
                        << std::endl;

                break;                    
            }                     
        }
        
    }
        
    if (edm::isDebugEnabled() ) {
        LogTrace("L1GtEtaPhiConversions") << std::endl;
        for (unsigned int iBin = 0; iBin < m_lutPhiMuCalo.size(); ++iBin) {
            LogTrace("L1GtEtaPhiConversions") << "Mu phiIndex = " << iBin
                    << " converted to index " << m_lutPhiMuCalo[iBin]
                    << std::endl;

        }
        LogTrace("L1GtEtaPhiConversions") << std::endl;
    }
    
    //
    // convert calo eta scale for CenJet/TauJet & IsoEG/NoIsoEG to a common 
    // central / forward eta scale
    //
    
    unsigned int nrGctCentralEtaBinsPerHalf = 
        l1CaloGeometry->numberGctCentralEtaBinsPerHalf();

    unsigned int nrGctForwardEtaBinsPerHalf =
            l1CaloGeometry->numberGctForwardEtaBinsPerHalf();
    
    unsigned int nrGctTotalEtaBinsPerHalf = nrGctCentralEtaBinsPerHalf
            + nrGctForwardEtaBinsPerHalf;
    

    m_lutEtaCenCaloCommon.clear(); 
    m_lutEtaCenCaloCommon.resize(nrGctTotalEtaBinsPerHalf); 
    
    for (unsigned int cenInd = 0; cenInd < nrGctCentralEtaBinsPerHalf; ++cenInd) {
        
        // FIXME 
        
    }
        
    if (edm::isDebugEnabled() ) {
        LogTrace("L1GtEtaPhiConversions") << std::endl;
        LogTrace("L1GtEtaPhiConversions") << std::endl;
    }


}

// print all the performed conversions
void L1GtEtaPhiConversions::print(std::ostream& myCout) const {

    myCout << "\n  L1GtEtaPhiConversions print...\n" << std::endl;
    // FIXME

}

