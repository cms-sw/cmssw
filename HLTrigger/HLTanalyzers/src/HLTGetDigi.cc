/** \class HLTGetDigi
 *
 * See header file for documentation
 *
 *  $Date: 2011/10/12 09:00:40 $
 *  $Revision: 1.8 $
 *
 *  \author various
 *
 */

#include "HLTrigger/HLTanalyzers/interface/HLTGetDigi.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

// system include files
#include <memory>
#include <vector>
#include <map>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalDigi/interface/EBDataFrame.h"
#include "DataFormats/EcalDigi/interface/EEDataFrame.h"
#include "DataFormats/EcalDigi/interface/ESDataFrame.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "DataFormats/HcalDigi/interface/HBHEDataFrame.h"
#include "DataFormats/HcalDigi/interface/HODataFrame.h"
#include "DataFormats/HcalDigi/interface/HFDataFrame.h"
#include "DataFormats/MuonData/interface/MuonDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCWireDigi.h"
#include "DataFormats/CSCDigi/interface/CSCWireDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCStripDigi.h"
#include "DataFormats/CSCDigi/interface/CSCStripDigiCollection.h"
#include "DataFormats/DTDigi/interface/DTDigi.h"
#include "DataFormats/DTDigi/interface/DTDigiCollection.h"
#include "DataFormats/RPCDigi/interface/RPCDigi.h"
#include "DataFormats/RPCDigi/interface/RPCDigiCollection.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctEtSums.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctCollections.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctJetCounts.h"
#include "DataFormats/L1CaloTrigger/interface/L1CaloCollections.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerEvmReadoutRecord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMapRecord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMap.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTReadoutCollection.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTCand.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTExtendedCand.h"
#include "CondFormats/L1TObjects/interface/L1GtParameters.h"
#include "CondFormats/DataRecord/interface/L1GtParametersRcd.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTCand.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTExtendedCand.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTReadoutCollection.h"

using namespace edm;
using namespace std;

//
// constructors and destructor
//
HLTGetDigi::HLTGetDigi(const edm::ParameterSet& ps)
{
  EBdigiCollection_ = ps.getParameter<edm::InputTag>("EBdigiCollection");
  EEdigiCollection_ = ps.getParameter<edm::InputTag>("EEdigiCollection");
  ESdigiCollection_ = ps.getParameter<edm::InputTag>("ESdigiCollection");
  HBHEdigiCollection_ = ps.getParameter<edm::InputTag>("HBHEdigiCollection");
  HOdigiCollection_   = ps.getParameter<edm::InputTag>("HOdigiCollection");
  HFdigiCollection_   = ps.getParameter<edm::InputTag>("HFdigiCollection");
  PXLdigiCollection_ = ps.getParameter<edm::InputTag>("SiPixeldigiCollection");
  SSTdigiCollection_ = ps.getParameter<edm::InputTag>("SiStripdigiCollection");
  CSCStripdigiCollection_ = ps.getParameter<edm::InputTag>("CSCStripdigiCollection");
  CSCWiredigiCollection_ = ps.getParameter<edm::InputTag>("CSCWiredigiCollection");
  DTdigiCollection_ = ps.getParameter<edm::InputTag>("DTdigiCollection");
  RPCdigiCollection_ = ps.getParameter<edm::InputTag>("RPCdigiCollection");

  GctCaloEmLabel_ = ps.getParameter<edm::InputTag>("L1CaloEmCollection");
  GctCaloRegionLabel_ = ps.getParameter<edm::InputTag>("L1CaloRegionCollection");

  GctIsoEmLabel_ = ps.getParameter<edm::InputTag>("GctIsoEmCollection");
  GctNonIsoEmLabel_ = ps.getParameter<edm::InputTag>("GctNonIsoEmCollection");

  GctCenJetLabel_ = ps.getParameter<edm::InputTag>("GctCenJetCollection");
  GctForJetLabel_ = ps.getParameter<edm::InputTag>("GctForJetCollection");
  GctTauJetLabel_ = ps.getParameter<edm::InputTag>("GctTauJetCollection");
  GctJetCountsLabel_ = ps.getParameter<edm::InputTag>("GctJetCounts");

  GctEtHadLabel_ = ps.getParameter<edm::InputTag>("GctEtHadCollection");
  GctEtMissLabel_ = ps.getParameter<edm::InputTag>("GctEtMissCollection");
  GctEtTotalLabel_ = ps.getParameter<edm::InputTag>("GctEtTotalCollection");

  GtEvmRRLabel_ = ps.getParameter<edm::InputTag>("GtEvmReadoutRecord");
  GtObjectMapLabel_ = ps.getParameter<edm::InputTag>("GtObjectMapRecord");
  GtRRLabel_ = ps.getParameter<edm::InputTag>("GtReadoutRecord");

  GmtCandsLabel_ = ps.getParameter<edm::InputTag>("GmtCands");
  GmtReadoutCollection_ = ps.getParameter<edm::InputTag>("GmtReadoutCollection");
  
  //--- Define which digis we want ---//
  getEcalDigis_    = ps.getUntrackedParameter<bool>("getEcal",true) ; 
  getEcalESDigis_  = ps.getUntrackedParameter<bool>("getEcalES",true) ; 
  getHcalDigis_    = ps.getUntrackedParameter<bool>("getHcal",true) ; 
  getPixelDigis_   = ps.getUntrackedParameter<bool>("getPixels",true) ; 
  getStripDigis_   = ps.getUntrackedParameter<bool>("getStrips",true) ; 
  getCSCDigis_     = ps.getUntrackedParameter<bool>("getCSC",true) ; 
  getDTDigis_      = ps.getUntrackedParameter<bool>("getDT",true) ; 
  getRPCDigis_     = ps.getUntrackedParameter<bool>("getRPC",true) ; 
  getGctEmDigis_   = ps.getUntrackedParameter<bool>("getGctEm",true) ; 
  getGctJetDigis_  = ps.getUntrackedParameter<bool>("getGctJet",true) ; 
  getGctJetCounts_ = ps.getUntrackedParameter<bool>("getGctJetCounts",true) ; 
  getGctEtDigis_   = ps.getUntrackedParameter<bool>("getGctEt",true) ;
  getL1Calo_       = ps.getUntrackedParameter<bool>("getL1Calo",true) ;
  getGtEvmRR_      = ps.getUntrackedParameter<bool>("getGtEvmRR",true) ;
  getGtObjectMap_  = ps.getUntrackedParameter<bool>("getGtObjectMap",true) ;
  getGtRR_         = ps.getUntrackedParameter<bool>("getGtReadoutRecord",true) ;
  getGmtCands_     = ps.getUntrackedParameter<bool>("getGmtCands",true) ;
  getGmtRC_        = ps.getUntrackedParameter<bool>("getGmtReadout",true) ;
  
}

HLTGetDigi::~HLTGetDigi()
{ }

//
// member functions
//

// ------------ method called to produce the data  ------------
void
HLTGetDigi::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
    using namespace edm;

    //--- L1 GCT and GT Digis ---//
    edm::Handle<L1GctEtHad> GctEtHad ; 
    edm::Handle<L1GctEtMiss> GctEtMiss ; 
    edm::Handle<L1GctEtTotal> GctEtTotal ; 

    const L1GctEtHad*   etHad   = 0 ; 
    const L1GctEtMiss*  etMiss  = 0 ; 
    const L1GctEtTotal* etTotal = 0 ;

    if (getGctEtDigis_) {
        iEvent.getByLabel(GctEtHadLabel_,GctEtHad) ; 
        iEvent.getByLabel(GctEtMissLabel_,GctEtMiss) ; 
        iEvent.getByLabel(GctEtTotalLabel_,GctEtTotal) ; 
        etHad = GctEtHad.product() ; 
        etMiss = GctEtMiss.product() ; 
        etTotal = GctEtTotal.product() ; 

        LogDebug("DigiInfo") << "Value of L1GctEtHad::et(): " << etHad->et() ; 
        LogDebug("DigiInfo") << "Value of L1GctEtMiss::et(): " << etMiss->et() << ", phi(): " << etMiss->phi() ; 
        LogDebug("DigiInfo") << "Value of L1GctEtTotal::et(): " << etTotal->et() ; 
    }

    edm::Handle<L1GctEmCandCollection> GctIsoEM ; 
    edm::Handle<L1GctEmCandCollection> GctNonIsoEM ; 

    const L1GctEmCandCollection* isoEMdigis = 0 ; 
    const L1GctEmCandCollection* nonIsoEMdigis = 0 ; 
    if (getGctEmDigis_) {
        iEvent.getByLabel(GctIsoEmLabel_,GctIsoEM) ;
        isoEMdigis = GctIsoEM.product() ; 
        iEvent.getByLabel(GctNonIsoEmLabel_,GctNonIsoEM) ; 
        nonIsoEMdigis = GctNonIsoEM.product() ; 
        LogDebug("DigiInfo") << "total # Gct Iso EM digis: " << isoEMdigis->size() ; 
        LogDebug("DigiInfo") << "total # Gct non-Iso EM digis: " << nonIsoEMdigis->size() ;
    }
    
    edm::Handle<L1GctJetCandCollection> GctCenJets ; 
    edm::Handle<L1GctJetCandCollection> GctForJets ; 
    edm::Handle<L1GctJetCandCollection> GctTauJets ; 
    edm::Handle<L1GctJetCounts> GctJetCounts ; 

    const L1GctJetCandCollection* cenJetDigis = 0 ; 
    const L1GctJetCandCollection* forJetDigis = 0 ; 
    const L1GctJetCandCollection* tauJetDigis = 0 ;
    std::auto_ptr<L1GctJetCounts> newCounts( new L1GctJetCounts() ) ;
    L1GctJetCounts* counts = newCounts.get() ; 
        
    if (getGctJetDigis_) {
        iEvent.getByLabel(GctCenJetLabel_,GctCenJets) ;
        cenJetDigis = GctCenJets.product() ; 
        iEvent.getByLabel(GctForJetLabel_,GctForJets) ;
        forJetDigis = GctForJets.product() ; 
        iEvent.getByLabel(GctTauJetLabel_,GctTauJets) ;
        tauJetDigis = GctTauJets.product() ; 
        LogDebug("DigiInfo") << "total # Gct central Jet digis: " << cenJetDigis->size() ; 
        LogDebug("DigiInfo") << "total # Gct forward Jet digis: " << forJetDigis->size() ;
        LogDebug("DigiInfo") << "total # Gct tau Jet digis: " << tauJetDigis->size() ;
    }

    if (getGctJetCounts_) {
        iEvent.getByLabel(GctJetCountsLabel_,GctJetCounts) ; 
        *counts = *GctJetCounts.product() ;
    }

    edm::Handle<L1CaloEmCollection> GctCaloEm ; 
    edm::Handle<L1CaloRegionCollection> GctCaloRegion ; 
    
    const L1CaloEmCollection* caloEm = 0 ; 
    const L1CaloRegionCollection* caloRegion = 0 ; 

    if (getL1Calo_) {
        iEvent.getByLabel(GctCaloEmLabel_,GctCaloEm) ; 
        iEvent.getByLabel(GctCaloRegionLabel_,GctCaloRegion) ; 

        caloEm = GctCaloEm.product() ; 
        caloRegion = GctCaloRegion.product() ; 

        LogDebug("DigiInfo") << "Calo EM size: " << caloEm->size() ; 
        LogDebug("DigiInfo") << "Calo region size: " << caloRegion->size() ; 
    }

    edm::Handle<L1GlobalTriggerEvmReadoutRecord> gtEvmRR ;
    edm::Handle<L1GlobalTriggerObjectMapRecord> gtMap ;
    edm::Handle<L1GlobalTriggerReadoutRecord> gtRR ;

    edm::ESHandle< L1GtParameters > l1GtPar ;
    iSetup.get< L1GtParametersRcd >().get( l1GtPar ) ;
    int nBx = l1GtPar->gtTotalBxInEvent() ; 
    
    std::auto_ptr<L1GlobalTriggerEvmReadoutRecord> newGtEvm( new L1GlobalTriggerEvmReadoutRecord(nBx) ) ; 
    std::auto_ptr<L1GlobalTriggerObjectMapRecord> newGtMap( new L1GlobalTriggerObjectMapRecord() ) ; 
    std::auto_ptr<L1GlobalTriggerReadoutRecord> newGtRR( new L1GlobalTriggerReadoutRecord(nBx) ) ; 
    L1GlobalTriggerEvmReadoutRecord* evm = newGtEvm.get() ; 
    L1GlobalTriggerObjectMapRecord* map = newGtMap.get() ; 
    L1GlobalTriggerReadoutRecord* rr = newGtRR.get() ; 

    if (getGtEvmRR_) {
        iEvent.getByLabel(GtEvmRRLabel_, gtEvmRR) ;
        *evm = *gtEvmRR.product() ;
    }
    if (getGtObjectMap_) {
        iEvent.getByLabel(GtObjectMapLabel_, gtMap) ;
        *map = *gtMap.product() ;
    }
    if (getGtRR_) {
        iEvent.getByLabel(GtRRLabel_, gtRR) ;
        *rr = *gtRR.product() ;
    }
    
    edm::Handle< std::vector<L1MuGMTCand> > GmtCands ;
    edm::Handle<L1MuGMTReadoutCollection> GmtMuCollection ; 
    std::auto_ptr<std::vector<L1MuGMTCand> > cands( new std::vector<L1MuGMTCand> ) ;
    std::auto_ptr<L1MuGMTReadoutCollection> muCollection(new L1MuGMTReadoutCollection(nBx)) ;
 
    if (getGmtCands_) {
        iEvent.getByLabel(GmtCandsLabel_,GmtCands) ; 
        *cands = *GmtCands.product() ; 
    }
    if (getGmtRC_) {
        iEvent.getByLabel(GmtReadoutCollection_,GmtMuCollection) ;
        *muCollection = *GmtMuCollection.product() ;
        std::vector<L1MuGMTExtendedCand> muons = muCollection->getRecord().getGMTCands() ;
        LogDebug("DigiInfo") << "GMT muons present: " << muons.size() ;
    }
    
    edm::Handle< DetSetVector<PixelDigi> >  input;
    auto_ptr<DetSetVector<PixelDigi> > NewPixelDigi(new DetSetVector<PixelDigi> );
    DetSetVector<PixelDigi>* tt = NewPixelDigi.get();
    if (getPixelDigis_) {
        iEvent.getByLabel(PXLdigiCollection_, input);
        *tt = *input.product();
    }

    edm::Handle< edm::DetSetVector<SiStripDigi> >  input2;
    auto_ptr<DetSetVector<SiStripDigi> > NewSiDigi(new DetSetVector<SiStripDigi> );
    DetSetVector<SiStripDigi>* uu = NewSiDigi.get();
    if (getStripDigis_) {
        iEvent.getByLabel(SSTdigiCollection_, input2);
        *uu = *input2.product();
    }

    Handle<EBDigiCollection> EcalDigiEB;
    Handle<EEDigiCollection> EcalDigiEE;
    Handle<ESDigiCollection> EcalDigiES;
    const EBDigiCollection* EBdigis = 0;
    const EEDigiCollection* EEdigis = 0;
    const ESDigiCollection* ESdigis = 0; 

    if (getEcalDigis_) {
        iEvent.getByLabel( EBdigiCollection_, EcalDigiEB );
        EBdigis = EcalDigiEB.product();
        LogDebug("DigiInfo") << "total # EBdigis: " << EBdigis->size() ;
     
        iEvent.getByLabel( EEdigiCollection_, EcalDigiEE );
        EEdigis = EcalDigiEE.product();
        LogDebug("DigiInfo") << "total # EEdigis: " << EEdigis->size() ;
    }

    if (getEcalESDigis_) {
        iEvent.getByLabel( ESdigiCollection_, EcalDigiES );
        ESdigis = EcalDigiES.product();
        LogDebug("DigiInfo") << "total # ESdigis: " << ESdigis->size() ;
    }
        
    Handle<HBHEDigiCollection> HcalDigiHBHE ; 
    Handle<HODigiCollection> HcalDigiHO ; 
    Handle<HFDigiCollection> HcalDigiHF ; 
    const HBHEDigiCollection* HBHEdigis = 0 ;
    const HODigiCollection* HOdigis = 0 ;
    const HFDigiCollection* HFdigis = 0 ; 

    if (getHcalDigis_) {
        iEvent.getByLabel( HBHEdigiCollection_, HcalDigiHBHE );
        HBHEdigis = HcalDigiHBHE.product();
        LogDebug("DigiInfo") << "total # HBHEdigis: " << HBHEdigis->size() ;
     
        iEvent.getByLabel( HOdigiCollection_, HcalDigiHO );
        HOdigis = HcalDigiHO.product();
        LogDebug("DigiInfo") << "total # HOdigis: " << HOdigis->size() ;
    
        iEvent.getByLabel( HFdigiCollection_, HcalDigiHF );
        HFdigis = HcalDigiHF.product();
        LogDebug("DigiInfo") << "total # HFdigis: " << HFdigis->size() ;
    }
    
    Handle<CSCStripDigiCollection> CSCDigiStrip ; 
    Handle<CSCWireDigiCollection> CSCDigiWire ; 

    if (getCSCDigis_) {
        iEvent.getByLabel( CSCStripdigiCollection_, CSCDigiStrip );
        iEvent.getByLabel( CSCWiredigiCollection_,  CSCDigiWire );

        int numDigis = 0 ; 
        for (CSCStripDigiCollection::DigiRangeIterator iter=CSCDigiStrip->begin();
             iter!=CSCDigiStrip->end(); iter++) {
            for ( vector<CSCStripDigi>::const_iterator digiIter = (*iter).second.first;
                  digiIter != (*iter).second.second; digiIter++) numDigis++ ;
        }
        LogDebug("DigiInfo") << "total # CSCstripdigis: " << numDigis ;
        numDigis = 0 ; 
        for (CSCWireDigiCollection::DigiRangeIterator iter=CSCDigiWire->begin();
             iter!=CSCDigiWire->end(); iter++) {
            for ( vector<CSCWireDigi>::const_iterator digiIter = (*iter).second.first;
                  digiIter != (*iter).second.second; digiIter++) numDigis++ ;
        }
        LogDebug("DigiInfo") << "total # CSCwiredigis: " << numDigis ;
    }
    
    Handle<DTDigiCollection> DTDigiHandle ; 

    if (getDTDigis_) {
        iEvent.getByLabel( DTdigiCollection_, DTDigiHandle );
    
        int numDigis = 0 ; 
        for (DTDigiCollection::DigiRangeIterator iter=DTDigiHandle->begin();
             iter!=DTDigiHandle->end(); iter++) {
            for ( vector<DTDigi>::const_iterator digiIter = (*iter).second.first;
                  digiIter != (*iter).second.second; digiIter++) numDigis++ ;
        }
        LogDebug("DigiInfo") << "total # DTdigis: " << numDigis ;
    }
        
    Handle<RPCDigiCollection> RPCDigiHandle ; 

    if (getRPCDigis_) { 
        iEvent.getByLabel( RPCdigiCollection_, RPCDigiHandle );

        int numDigis = 0 ; 
        for (RPCDigiCollection::DigiRangeIterator iter=RPCDigiHandle->begin();
             iter!=RPCDigiHandle->end(); iter++) {
            for ( vector<RPCDigi>::const_iterator digiIter = (*iter).second.first;
                  digiIter != (*iter).second.second; digiIter++) numDigis++ ;
        }
        LogDebug("DigiInfo") << "total # RPCdigis: " << numDigis ;
    }
    
    LogDebug("DigiInfo") << "***--------------- End of Event -----------------***" ;  
    
}
