/** \class EcalTPSkimmer
 *   produce a subset of TP information
 *
 *  $Id: EcalTPSkimmer.cc,v 1.2 2010/10/02 16:13:54 ferriff Exp $
 *  $Date: 2010/10/02 16:13:54 $
 *  $Revision: 1.2 $
 *  \author Federico Ferri, CEA/Saclay Irfu/SPP
 *
 **/

#include "RecoLocalCalo/EcalRecProducers/plugins/EcalTPSkimmer.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDetId/interface/EcalTrigTowerDetId.h"

#include "CondFormats/DataRecord/interface/EcalChannelStatusRcd.h"
#include "CondFormats/EcalObjects/interface/EcalChannelStatus.h"

#include "Geometry/Records/interface/IdealGeometryRecord.h"

EcalTPSkimmer::EcalTPSkimmer(const edm::ParameterSet& ps)
{
        skipModule_         = ps.getParameter<bool>("skipModule");

        doBarrel_           = ps.getParameter<bool>("doBarrel");
        doEndcap_           = ps.getParameter<bool>("doEndcap");

        chStatusToSelectTP_ = ps.getParameter<std::vector<uint32_t> >("chStatusToSelectTP");

        tpOutputCollection_ = ps.getParameter<std::string>("tpOutputCollection");
        tpInputCollection_  = ps.getParameter<edm::InputTag>("tpInputCollection");

        produces< EcalTrigPrimDigiCollection >(tpOutputCollection_);
}

EcalTPSkimmer::~EcalTPSkimmer()
{
}

void
EcalTPSkimmer::produce(edm::Event& evt, const edm::EventSetup& es)
{
        insertedTP_.clear();
        
        using namespace edm;

        es.get<IdealGeometryRecord>().get(ttMap_);

        // collection of rechits to put in the event
        std::auto_ptr< EcalTrigPrimDigiCollection > tpOut( new EcalTrigPrimDigiCollection );
        
        if ( skipModule_ ) {
                evt.put( tpOut, tpOutputCollection_ );
                return;
        }

        edm::ESHandle<EcalChannelStatus> chStatus;
        es.get<EcalChannelStatusRcd>().get(chStatus);

        edm::Handle<EcalTrigPrimDigiCollection> tpIn;
        evt.getByLabel(tpInputCollection_, tpIn);

        if ( ! tpIn.isValid() ) {
                edm::LogError("EcalTPSkimmer") << "Can't get the product " << tpInputCollection_.instance()
                        << " with label " << tpInputCollection_.label();
                return;
        }

        if ( doBarrel_ ) {
                EcalChannelStatusMap::const_iterator chit;
                uint16_t code = 0;
                for ( int i = 0; i < EBDetId::kSizeForDenseIndexing; ++i )
                {
                        if ( ! EBDetId::validDenseIndex( i ) ) continue;
                        EBDetId id = EBDetId::detIdFromDenseIndex( i );
                        chit = chStatus->find( id );
                        // check if the channel status means TP to be kept
                        if ( chit != chStatus->end() ) {
                                code = (*chit).getStatusCode() & 0x001F;
                                if ( std::find( chStatusToSelectTP_.begin(), chStatusToSelectTP_.end(), code ) != chStatusToSelectTP_.end() ) {
                                        // retrieve the TP DetId
                                        EcalTrigTowerDetId ttDetId( ((EBDetId)id).tower() );
                                        // insert the TP if not done already
                                        if ( ! alreadyInserted( ttDetId ) ) insertTP( ttDetId, tpIn, *tpOut );
                                }
                        } else {
                                edm::LogError("EcalDetIdToBeRecoveredProducer") << "No channel status found for xtal "
                                        << id.rawId()
                                        << "! something wrong with EcalChannelStatus in your DB? ";
                        }
                }
        }

        if ( doEndcap_ ) {
                EcalChannelStatusMap::const_iterator chit;
                uint16_t code = 0;
                for ( int i = 0; i < EEDetId::kSizeForDenseIndexing; ++i )
                {
                        if ( ! EEDetId::validDenseIndex( i ) ) continue;
                        EEDetId id = EEDetId::detIdFromDenseIndex( i );
                        chit = chStatus->find( id );
                        // check if the channel status means TP to be kept
                        if ( chit != chStatus->end() ) {
                                code = (*chit).getStatusCode() & 0x001F;
                                if ( std::find( chStatusToSelectTP_.begin(), chStatusToSelectTP_.end(), code ) != chStatusToSelectTP_.end() ) {
                                        // retrieve the TP DetId
                                        EcalTrigTowerDetId ttDetId = ttMap_->towerOf( id );
                                        // insert the TP if not done already
                                        if ( ! alreadyInserted( ttDetId ) ) insertTP( ttDetId, tpIn, *tpOut );
                                }
                        } else {
                                edm::LogError("EcalDetIdToBeRecoveredProducer") << "No channel status found for xtal "
                                        << id.rawId()
                                        << "! something wrong with EcalChannelStatus in your DB? ";
                        }
                }
        }

        // put the collection of reconstructed hits in the event   
        LogInfo("EcalTPSkimmer") << "total # of TP inserted: " << tpOut->size();

        evt.put( tpOut, tpOutputCollection_ );
}


bool EcalTPSkimmer::alreadyInserted( EcalTrigTowerDetId ttId )
{
        return ( insertedTP_.find( ttId ) != insertedTP_.end() );
}


void EcalTPSkimmer::insertTP( EcalTrigTowerDetId ttId, edm::Handle<EcalTrigPrimDigiCollection> &tpIn, EcalTrigPrimDigiCollection &tpOut )
{
        EcalTrigPrimDigiCollection::const_iterator tpIt = tpIn->find( ttId );
        if ( tpIt != tpIn->end() ) {
                tpOut.push_back( *tpIt );
                insertedTP_.insert( ttId );
        }
}


#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE( EcalTPSkimmer );

