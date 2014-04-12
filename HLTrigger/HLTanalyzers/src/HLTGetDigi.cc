/** \class HLTGetDigi
 *
 * See header file for documentation
 *
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
#include "FWCore/Utilities/interface/InputTag.h"

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
  
  //--- Declare consums ---//
  if (getEcalDigis_) {
    EBdigiToken_ = consumes<EBDigiCollection>(EBdigiCollection_);
    EEdigiToken_ = consumes<EEDigiCollection>(EEdigiCollection_);
   }
  if (getEcalESDigis_) {
    ESdigiToken_ = consumes<ESDigiCollection>(ESdigiCollection_);
  }
  if (getHcalDigis_) {
    HBHEdigiToken_ = consumes<HBHEDigiCollection>(HBHEdigiCollection_);
    HOdigiToken_ = consumes<HODigiCollection>(HOdigiCollection_);
    HFdigiToken_ = consumes<HFDigiCollection>(HFdigiCollection_);
  }
  if (getPixelDigis_) {
    PXLdigiToken_ = consumes<edm::DetSetVector<PixelDigi> >(PXLdigiCollection_);
  }
  if (getStripDigis_) {
    SSTdigiToken_ = consumes<edm::DetSetVector<SiStripDigi> >(SSTdigiCollection_);
  }
  if (getCSCDigis_) {
    CSCStripdigiToken_ = consumes<CSCStripDigiCollection>(CSCStripdigiCollection_);
    CSCWiredigiToken_ = consumes<CSCWireDigiCollection>(CSCWiredigiCollection_);
  }
  if (getDTDigis_) {
    DTdigiToken_ = consumes<DTDigiCollection>(DTdigiCollection_);
  }
  if (getRPCDigis_) {
    RPCdigiToken_ = consumes<RPCDigiCollection>(RPCdigiCollection_);
  }
  if (getGctEmDigis_) {
    GctIsoEmToken_ = consumes<L1GctEmCandCollection>(GctIsoEmLabel_);
    GctNonIsoEmToken_ = consumes<L1GctEmCandCollection>(GctNonIsoEmLabel_);
  }
  if (getGctJetDigis_) {
    GctCenJetToken_ = consumes<L1GctJetCandCollection>(GctCenJetLabel_);
    GctForJetToken_ = consumes<L1GctJetCandCollection>(GctForJetLabel_);
    GctTauJetToken_ = consumes<L1GctJetCandCollection>(GctTauJetLabel_);
  }
  if (getGctJetCounts_) {
    GctJetCountsToken_ = consumes<L1GctJetCounts>(GctJetCountsLabel_);
  }
  if (getGctEtDigis_) {
    GctEtHadToken_ = consumes<L1GctEtHad>(GctEtHadLabel_);
    GctEtMissToken_ = consumes<L1GctEtMiss>(GctEtMissLabel_);
    GctEtTotalToken_ = consumes<L1GctEtTotal>(GctEtTotalLabel_);
  }
  if (getL1Calo_) {
    GctCaloEmToken_ = consumes<L1CaloEmCollection>(GctCaloEmLabel_);
    GctCaloRegionToken_ = consumes<L1CaloRegionCollection>(GctCaloRegionLabel_);
  }
  if (getGtEvmRR_) {
    GtEvmRRToken_ = consumes<L1GlobalTriggerEvmReadoutRecord>(GtEvmRRLabel_);
  }
  if (getGtObjectMap_) {
    GtObjectMapToken_ = consumes<L1GlobalTriggerObjectMapRecord>(GtObjectMapLabel_);
  }
  if (getGtRR_) {
    GtRRToken_ = consumes<L1GlobalTriggerReadoutRecord>(GtRRLabel_);
  }
  if (getGmtCands_) {
    GmtCandsToken_ = consumes<std::vector<L1MuGMTCand> >(GmtCandsLabel_);
  }
  if (getGmtRC_) {
    GmtReadoutToken_ = consumes<L1MuGMTReadoutCollection>(GmtReadoutCollection_);
  }

}

HLTGetDigi::~HLTGetDigi()
{ }

void
HLTGetDigi::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("EEdigiCollection",edm::InputTag("ecalDigis","eeDigis"));
  desc.add<edm::InputTag>("HBHEdigiCollection",edm::InputTag("hcalDigis"));
  desc.add<edm::InputTag>("GctIsoEmCollection",edm::InputTag("gctDigis","isoEm"));
  desc.add<edm::InputTag>("ESdigiCollection",edm::InputTag("ecalPreshowerDigis"));
  desc.add<edm::InputTag>("GctEtHadCollection",edm::InputTag("gctDigis"));
  desc.add<edm::InputTag>("CSCStripdigiCollection",edm::InputTag("muonCSCDigis","MuonCSCStripDigi"));
  desc.add<edm::InputTag>("GmtCands",edm::InputTag("gmtDigis"));
  desc.add<edm::InputTag>("GctEtTotalCollection",edm::InputTag("gctDigis"));
  desc.add<edm::InputTag>("SiStripdigiCollection",edm::InputTag("siStripDigis"));
  desc.add<edm::InputTag>("GctJetCounts",edm::InputTag("gctDigis"));
  desc.add<edm::InputTag>("DTdigiCollection",edm::InputTag("muonDTDigis"));
  desc.add<edm::InputTag>("GctTauJetCollection ",edm::InputTag("gctDigis","tauJets"));
  desc.add<edm::InputTag>("L1CaloRegionCollection",edm::InputTag("rctDigis"));
  desc.add<edm::InputTag>("GtObjectMapRecord",edm::InputTag("gtDigis"));
  desc.add<edm::InputTag>("GmtReadoutCollection",edm::InputTag("gmtDigis"));
  desc.add<edm::InputTag>("HOdigiCollection",edm::InputTag("hcalDigis"));
  desc.add<edm::InputTag>("RPCdigiCollection",edm::InputTag("muonRPCDigis"));
  desc.add<edm::InputTag>("CSCWiredigiCollection",edm::InputTag("muonCSCDigis","MuonCSCWireDigi"));
  desc.add<edm::InputTag>("GctForJetCollection",edm::InputTag("gctDigis","tauJets"));
  desc.add<edm::InputTag>("HFdigiCollection",edm::InputTag("hcalDigis"));
  desc.add<edm::InputTag>("SiPixeldigiCollection",edm::InputTag("siPixelDigis"));
  desc.add<edm::InputTag>("GctNonIsoEmCollection",edm::InputTag("gctDigis","nonIsoEm"));
  desc.add<edm::InputTag>("GtEvmReadoutRecord",edm::InputTag("gtDigis"));
  desc.add<edm::InputTag>("L1CaloEmCollection",edm::InputTag("rctDigis"));
  desc.add<edm::InputTag>("GctCenJetCollection",edm::InputTag("gctDigis","cenJets"));
  desc.add<edm::InputTag>("GtReadoutRecord",edm::InputTag("gtDigis"));
  desc.add<edm::InputTag>("GctEtMissCollection",edm::InputTag("gctDigis"));
  desc.add<edm::InputTag>("EBdigiCollection",edm::InputTag("ecalDigis","ebDigis"));
  desc.addUntracked<bool>("getGctEt",true);
  desc.addUntracked<bool>("getGtReadoutRecord",true);
  desc.addUntracked<bool>("getGtEvmRR",true);
  desc.addUntracked<bool>("getGctEm",true);
  desc.addUntracked<bool>("getPixels",true);
  desc.addUntracked<bool>("getGctJet",true);
  desc.addUntracked<bool>("getHcal",true);
  desc.addUntracked<bool>("getGctJetCounts",true);
  desc.addUntracked<bool>("getL1Calo",false);
  desc.addUntracked<bool>("getStrips",true);
  desc.addUntracked<bool>("getDT",true);
  desc.addUntracked<bool>("getGtObjectMap",true);
  desc.addUntracked<bool>("getGmtCands",true);
  desc.addUntracked<bool>("getRPC",true);
  desc.addUntracked<bool>("getEcal",true);
  desc.addUntracked<bool>("getGmtReadout",true);
  desc.addUntracked<bool>("getEcalES",true);
  desc.addUntracked<bool>("getCSC",true);
  descriptions.add("hltGetDigi",desc);
}

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
        iEvent.getByToken(GctEtHadToken_,GctEtHad) ; 
        iEvent.getByToken(GctEtMissToken_,GctEtMiss) ; 
        iEvent.getByToken(GctEtTotalToken_,GctEtTotal) ; 
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
        iEvent.getByToken(GctIsoEmToken_,GctIsoEM) ;
        isoEMdigis = GctIsoEM.product() ; 
        iEvent.getByToken(GctNonIsoEmToken_,GctNonIsoEM) ; 
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
        iEvent.getByToken(GctCenJetToken_,GctCenJets) ;
        cenJetDigis = GctCenJets.product() ; 
        iEvent.getByToken(GctForJetToken_,GctForJets) ;
        forJetDigis = GctForJets.product() ; 
        iEvent.getByToken(GctTauJetToken_,GctTauJets) ;
        tauJetDigis = GctTauJets.product() ; 
        LogDebug("DigiInfo") << "total # Gct central Jet digis: " << cenJetDigis->size() ; 
        LogDebug("DigiInfo") << "total # Gct forward Jet digis: " << forJetDigis->size() ;
        LogDebug("DigiInfo") << "total # Gct tau Jet digis: " << tauJetDigis->size() ;
    }

    if (getGctJetCounts_) {
        iEvent.getByToken(GctJetCountsToken_,GctJetCounts) ; 
        *counts = *GctJetCounts.product() ;
    }

    edm::Handle<L1CaloEmCollection> GctCaloEm ; 
    edm::Handle<L1CaloRegionCollection> GctCaloRegion ; 
    
    const L1CaloEmCollection* caloEm = 0 ; 
    const L1CaloRegionCollection* caloRegion = 0 ; 

    if (getL1Calo_) {
        iEvent.getByToken(GctCaloEmToken_,GctCaloEm) ; 
        iEvent.getByToken(GctCaloRegionToken_,GctCaloRegion) ; 

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
        iEvent.getByToken(GtEvmRRToken_, gtEvmRR) ;
        *evm = *gtEvmRR.product() ;
    }
    if (getGtObjectMap_) {
        iEvent.getByToken(GtObjectMapToken_, gtMap) ;
        *map = *gtMap.product() ;
    }
    if (getGtRR_) {
        iEvent.getByToken(GtRRToken_, gtRR) ;
        *rr = *gtRR.product() ;
    }
    
    edm::Handle< std::vector<L1MuGMTCand> > GmtCands ;
    edm::Handle<L1MuGMTReadoutCollection> GmtMuCollection ; 
    std::auto_ptr<std::vector<L1MuGMTCand> > cands( new std::vector<L1MuGMTCand> ) ;
    std::auto_ptr<L1MuGMTReadoutCollection> muCollection(new L1MuGMTReadoutCollection(nBx)) ;
 
    if (getGmtCands_) {
        iEvent.getByToken(GmtCandsToken_,GmtCands) ; 
        *cands = *GmtCands.product() ; 
    }
    if (getGmtRC_) {
        iEvent.getByToken(GmtReadoutToken_,GmtMuCollection) ;
        *muCollection = *GmtMuCollection.product() ;
        std::vector<L1MuGMTExtendedCand> muons = muCollection->getRecord().getGMTCands() ;
        LogDebug("DigiInfo") << "GMT muons present: " << muons.size() ;
    }
    
    edm::Handle< DetSetVector<PixelDigi> >  input;
    auto_ptr<DetSetVector<PixelDigi> > NewPixelDigi(new DetSetVector<PixelDigi> );
    DetSetVector<PixelDigi>* tt = NewPixelDigi.get();
    if (getPixelDigis_) {
        iEvent.getByToken(PXLdigiToken_, input);
        *tt = *input.product();
    }

    edm::Handle< edm::DetSetVector<SiStripDigi> >  input2;
    auto_ptr<DetSetVector<SiStripDigi> > NewSiDigi(new DetSetVector<SiStripDigi> );
    DetSetVector<SiStripDigi>* uu = NewSiDigi.get();
    if (getStripDigis_) {
        iEvent.getByToken(SSTdigiToken_, input2);
        *uu = *input2.product();
    }

    Handle<EBDigiCollection> EcalDigiEB;
    Handle<EEDigiCollection> EcalDigiEE;
    Handle<ESDigiCollection> EcalDigiES;
    const EBDigiCollection* EBdigis = 0;
    const EEDigiCollection* EEdigis = 0;
    const ESDigiCollection* ESdigis = 0; 

    if (getEcalDigis_) {
        iEvent.getByToken( EBdigiToken_, EcalDigiEB );
        EBdigis = EcalDigiEB.product();
        LogDebug("DigiInfo") << "total # EBdigis: " << EBdigis->size() ;
     
        iEvent.getByToken( EEdigiToken_, EcalDigiEE );
        EEdigis = EcalDigiEE.product();
        LogDebug("DigiInfo") << "total # EEdigis: " << EEdigis->size() ;
    }

    if (getEcalESDigis_) {
        iEvent.getByToken( ESdigiToken_, EcalDigiES );
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
        iEvent.getByToken( HBHEdigiToken_, HcalDigiHBHE );
        HBHEdigis = HcalDigiHBHE.product();
        LogDebug("DigiInfo") << "total # HBHEdigis: " << HBHEdigis->size() ;
     
        iEvent.getByToken( HOdigiToken_, HcalDigiHO );
        HOdigis = HcalDigiHO.product();
        LogDebug("DigiInfo") << "total # HOdigis: " << HOdigis->size() ;
    
        iEvent.getByToken( HFdigiToken_, HcalDigiHF );
        HFdigis = HcalDigiHF.product();
        LogDebug("DigiInfo") << "total # HFdigis: " << HFdigis->size() ;
    }
    
    Handle<CSCStripDigiCollection> CSCDigiStrip ; 
    Handle<CSCWireDigiCollection> CSCDigiWire ; 

    if (getCSCDigis_) {
        iEvent.getByToken( CSCStripdigiToken_, CSCDigiStrip );
        iEvent.getByToken( CSCWiredigiToken_,  CSCDigiWire );

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
        iEvent.getByToken( DTdigiToken_, DTDigiHandle );
    
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
        iEvent.getByToken( RPCdigiToken_, RPCDigiHandle );

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
