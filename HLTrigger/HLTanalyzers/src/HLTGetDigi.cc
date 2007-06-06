/** \class HLTGetDigi
 *
 * See header file for documentation
 *
 *  $Date: 2007/04/20 06:58:26 $
 *  $Revision: 1.1 $
 *
 *  \author various
 *
 */

#include "HLTrigger/HLTanalyzers/interface/HLTGetDigi.h"

#include "DataFormats/Common/interface/Handle.h"

// system include files
#include <memory>
#include <vector>
#include <map>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

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

  //--- Define which digis we want ---//
  getEcalDigis_   = ps.getUntrackedParameter<bool>("getEcal",true) ; 
  getEcalESDigis_ = ps.getUntrackedParameter<bool>("getEcalES",true) ; 
  getHcalDigis_   = ps.getUntrackedParameter<bool>("getHcal",true) ; 
  getPixelDigis_  = ps.getUntrackedParameter<bool>("getPixels",true) ; 
  getStripDigis_  = ps.getUntrackedParameter<bool>("getStrips",true) ; 
  getCSCDigis_    = ps.getUntrackedParameter<bool>("getCSC",true) ; 
  getDTDigis_     = ps.getUntrackedParameter<bool>("getDT",true) ; 
  getRPCDigis_    = ps.getUntrackedParameter<bool>("getRPC",true) ; 
  
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
    const CSCStripDigiCollection* CSCstripdigis ; 
    const CSCWireDigiCollection* CSCwiredigis ; 

    if (getCSCDigis_) {
        iEvent.getByLabel( CSCStripdigiCollection_, CSCDigiStrip );
        CSCstripdigis = CSCDigiStrip.product();
        iEvent.getByLabel( CSCWiredigiCollection_, CSCDigiWire );
        CSCwiredigis = CSCDigiWire.product();

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
    const DTDigiCollection* DTdigis ; 

    if (getDTDigis_) {
        iEvent.getByLabel( DTdigiCollection_, DTDigiHandle );
        DTdigis = DTDigiHandle.product();
    
        int numDigis = 0 ; 
        for (DTDigiCollection::DigiRangeIterator iter=DTDigiHandle->begin();
             iter!=DTDigiHandle->end(); iter++) {
            for ( vector<DTDigi>::const_iterator digiIter = (*iter).second.first;
                  digiIter != (*iter).second.second; digiIter++) numDigis++ ;
        }
        LogDebug("DigiInfo") << "total # DTdigis: " << numDigis ;
    }
        
    Handle<RPCDigiCollection> RPCDigiHandle ; 
    const RPCDigiCollection* RPCdigis ; 

    if (getRPCDigis_) { 
        iEvent.getByLabel( RPCdigiCollection_, RPCDigiHandle );
        RPCdigis = RPCDigiHandle.product();

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
