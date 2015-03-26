#include "RecoLocalCalo/EcalRecProducers/plugins/EcalDetIdToBeRecoveredProducer.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/EcalDetId/interface/EcalDetIdCollections.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

#include "Geometry/EcalMapping/interface/EcalMappingRcd.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "CondFormats/DataRecord/interface/EcalChannelStatusRcd.h"

#include <set>

#include <sys/types.h>
#include <signal.h>

EcalDetIdToBeRecoveredProducer::EcalDetIdToBeRecoveredProducer(const edm::ParameterSet& ps)
{
        // SRP collections
        ebSrFlagToken_ = consumes<EBSrFlagCollection>(ps.getParameter<edm::InputTag>("ebSrFlagCollection"));
        eeSrFlagToken_ = consumes<EESrFlagCollection>(ps.getParameter<edm::InputTag>("eeSrFlagCollection"));

        // Integrity for xtal data
        ebIntegrityGainErrorsToken_ = consumes<EBDetIdCollection>(ps.getParameter<edm::InputTag>("ebIntegrityGainErrors"));
        ebIntegrityGainSwitchErrorsToken_ = consumes<EBDetIdCollection>(ps.getParameter<edm::InputTag>("ebIntegrityGainSwitchErrors"));
        ebIntegrityChIdErrorsToken_ = consumes<EBDetIdCollection>(ps.getParameter<edm::InputTag>("ebIntegrityChIdErrors"));

        // Integrity for xtal data - EE specific (to be rivisited towards EB+EE common collection)
        eeIntegrityGainErrorsToken_ = consumes<EEDetIdCollection>(ps.getParameter<edm::InputTag>("eeIntegrityGainErrors"));
        eeIntegrityGainSwitchErrorsToken_ = consumes<EEDetIdCollection>(ps.getParameter<edm::InputTag>("eeIntegrityGainSwitchErrors"));
        eeIntegrityChIdErrorsToken_ = consumes<EEDetIdCollection>(ps.getParameter<edm::InputTag>("eeIntegrityChIdErrors"));

        // Integrity Errors
        integrityTTIdErrorsToken_ = consumes<EcalElectronicsIdCollection>(ps.getParameter<edm::InputTag>("integrityTTIdErrors"));
        integrityBlockSizeErrorsToken_ = consumes<EcalElectronicsIdCollection>(ps.getParameter<edm::InputTag>("integrityBlockSizeErrors"));

        // output collections
        ebDetIdCollection_ = ps.getParameter<std::string>("ebDetIdToBeRecovered");
        eeDetIdCollection_ = ps.getParameter<std::string>("eeDetIdToBeRecovered");
        ttDetIdCollection_ = ps.getParameter<std::string>("ebFEToBeRecovered");
        scDetIdCollection_ = ps.getParameter<std::string>("eeFEToBeRecovered");

        produces< std::set<EBDetId> >( ebDetIdCollection_ );
        produces< std::set<EEDetId> >( eeDetIdCollection_ );
        produces< std::set<EcalTrigTowerDetId> >( ttDetIdCollection_ );
        produces< std::set<EcalScDetId> >( scDetIdCollection_ );
}


EcalDetIdToBeRecoveredProducer::~EcalDetIdToBeRecoveredProducer()
{
}


void EcalDetIdToBeRecoveredProducer::beginRun(edm::Run const& run, const edm::EventSetup& es)
{
        edm::ESHandle< EcalElectronicsMapping > pEcalMapping;
        es.get<EcalMappingRcd>().get(pEcalMapping);
        ecalMapping_ = pEcalMapping.product();

        edm::ESHandle< EcalChannelStatusMap > pChStatus;
        es.get<EcalChannelStatusRcd>().get(pChStatus);
        chStatus_ = pChStatus.product();

        es.get<IdealGeometryRecord>().get(ttMap_);
}

// fuction return true if "coll" have "item"
template <typename CollT, typename ItemT>
bool include(const CollT& coll, const ItemT& item)
{
  typename CollT::const_iterator res = std::find( coll.begin(), coll.end(), item );
  return ( res != coll.end() );
}

void EcalDetIdToBeRecoveredProducer::produce(edm::Event& ev, const edm::EventSetup& es)
{
        std::vector< edm::Handle<EBDetIdCollection> > ebDetIdColls;
        std::vector< edm::Handle<EEDetIdCollection> > eeDetIdColls;
        std::vector< edm::Handle<EcalElectronicsIdCollection> > ttColls;

        std::auto_ptr< std::set<EBDetId> > ebDetIdToRecover( new std::set<EBDetId> ); // isolated channels to be recovered
        std::auto_ptr< std::set<EEDetId> > eeDetIdToRecover( new std::set<EEDetId> ); // isolated channels to be recovered
        std::auto_ptr< std::set<EcalTrigTowerDetId> > ebTTDetIdToRecover( new std::set<EcalTrigTowerDetId> ); // tt to be recovered
        std::auto_ptr< std::set<EcalScDetId> > eeSCDetIdToRecover( new std::set<EcalScDetId> ); // sc to be recovered

        /*
         * get collections
         */

        // Selective Readout Flags
        edm::Handle<EBSrFlagCollection> ebSrFlags;
        ev.getByToken(ebSrFlagToken_, ebSrFlags );

        edm::Handle<EESrFlagCollection> eeSrFlags;
        ev.getByToken(eeSrFlagToken_, eeSrFlags );

        // Integrity errors
        edm::Handle<EBDetIdCollection> ebIntegrityGainErrors;
        ev.getByToken( ebIntegrityGainErrorsToken_, ebIntegrityGainErrors );

	ebDetIdColls.push_back( ebIntegrityGainErrors );
       
        edm::Handle<EBDetIdCollection> ebIntegrityGainSwitchErrors;
        ev.getByToken( ebIntegrityGainSwitchErrorsToken_, ebIntegrityGainSwitchErrors );       
	ebDetIdColls.push_back( ebIntegrityGainSwitchErrors );

        edm::Handle<EBDetIdCollection> ebIntegrityChIdErrors;
        ev.getByToken( ebIntegrityChIdErrorsToken_, ebIntegrityChIdErrors );
	ebDetIdColls.push_back( ebIntegrityChIdErrors );
       
        edm::Handle<EEDetIdCollection> eeIntegrityGainErrors;
        ev.getByToken( eeIntegrityGainErrorsToken_, eeIntegrityGainErrors );
	eeDetIdColls.push_back( eeIntegrityGainErrors );
       
        edm::Handle<EEDetIdCollection> eeIntegrityGainSwitchErrors;
        ev.getByToken( eeIntegrityGainSwitchErrorsToken_, eeIntegrityGainSwitchErrors );
	eeDetIdColls.push_back( eeIntegrityGainSwitchErrors );
		


        edm::Handle<EEDetIdCollection> eeIntegrityChIdErrors;
        ev.getByToken( eeIntegrityChIdErrorsToken_, eeIntegrityChIdErrors );
	eeDetIdColls.push_back( eeIntegrityChIdErrors );
 
 
 

        edm::Handle<EcalElectronicsIdCollection> integrityTTIdErrors;
        ev.getByToken( integrityTTIdErrorsToken_, integrityTTIdErrors );
	ttColls.push_back( integrityTTIdErrors );

        edm::Handle<EcalElectronicsIdCollection> integrityBlockSizeErrors;
        ev.getByToken( integrityBlockSizeErrorsToken_, integrityBlockSizeErrors );
	ttColls.push_back( integrityBlockSizeErrors );
               
        /*
         *  get regions of interest from SRP
         */
        // -- Barrel
        EBDetIdCollection ebSrpDetId;
        EcalTrigTowerDetIdCollection ebSrpTTDetId;
        for ( EBSrFlagCollection::const_iterator it = ebSrFlags->begin(); it != ebSrFlags->end(); ++it ) {
                const int flag = it->value();
                if ( flag == EcalSrFlag::SRF_FULL || ( flag == EcalSrFlag::SRF_FORCED_MASK) ) {
                        const EcalTrigTowerDetId ttId = it->id();
                        ebSrpTTDetId.push_back( ttId );
                
                        const std::vector<DetId> vid = ttMap_->constituentsOf( ttId );
                
                        for ( std::vector<DetId>::const_iterator itId = vid.begin(); itId != vid.end(); ++itId ) {
                                ebSrpDetId.push_back( *itId );
                        }
                }
        }
        // -- Endcap
        EEDetIdCollection eeSrpDetId;
        //EcalTrigTowerDetIdCollection eeSrpTTDetId;
        for ( EESrFlagCollection::const_iterator it = eeSrFlags->begin(); it != eeSrFlags->end(); ++it ) {
                const int flag = it->value();
                if ( flag == EcalSrFlag::SRF_FULL || ( flag == EcalSrFlag::SRF_FORCED_MASK) ) {
                        //EcalTrigTowerDetId ttId = it->id();
                        //eeSrpTTDetId.push_back( ttId );
                        const EcalScDetId scId( it->id() );
                        // not clear how to get the vector of DetId constituents of a SC...
                        //////////EcalElectronicsId eId( scId.rawId() );
                        //std::vector<DetId> vid = ecalMapping_->dccTowerConstituents( eId.dccId(), eId.towerId() );
                        std::vector<DetId> vid;
                        for(int dx=1; dx<=5; ++dx){
                                for(int dy=1; dy<=5; ++dy){
                                        const int ix = (scId.ix()-1)*5 + dx;
                                        const int iy = (scId.iy()-1)*5 + dy;
                                        const int iz = scId.zside();
                                        if(EEDetId::validDetId(ix, iy, iz)){
                                                vid.push_back(EEDetId(ix, iy, iz));
                                        }
                                }
                        }
                        ////eeSrpDetId.insert( interestingDetId.end(), vid.begin(), vid.end() );
                        //std::vector<DetId> vid = ttMap_->constituentsOf( ttId );
                        for ( std::vector<DetId>::const_iterator itId = vid.begin(); itId != vid.end(); ++itId ) {
                                eeSrpDetId.push_back( *itId );
                        }
                }
        }
        // SRP switched off: get the list from the DB
        if ( ebSrFlags->size() == 0 ) {
        }
        // SRP switched off: get the list from the DB
        if ( eeSrFlags->size() == 0 ) {
        }

        /*
         *  get OR of integrity error collections
         *  in interesting regions flagged by SRP
         *  and insert them in the list of DetId to recover
         */
        // -- Barrel
        for ( std::vector<edm::Handle<EBDetIdCollection> >::const_iterator it = ebDetIdColls.begin(); it != ebDetIdColls.end(); ++it )
        {
                const EBDetIdCollection * idc = it->product();
                for ( EBDetIdCollection::const_iterator jt = idc->begin(); jt != idc->end(); ++jt )
                  if (include(ebSrpDetId, *jt))
                    ebDetIdToRecover->insert( *jt );
        }
        // -- Endcap
        for ( std::vector<edm::Handle<EEDetIdCollection> >::const_iterator it = eeDetIdColls.begin(); it != eeDetIdColls.end(); ++it )
        {
                const EEDetIdCollection * idc = it->product();
                for ( EEDetIdCollection::const_iterator jt = idc->begin(); jt != idc->end(); ++jt )
                  if (include(eeSrpDetId, *jt))
                    eeDetIdToRecover->insert( *jt );
        }

        /* 
         * find isolated dead channels (from DB info)           --> chStatus 10, 11, 12
         * and group of dead channels w/ trigger(from DB info)  --> chStatus 13
         * in interesting regions flagged by SRP
         */
        // -- Barrel
        for ( EBDetIdCollection::const_iterator itId = ebSrpDetId.begin(); itId != ebSrpDetId.end(); ++itId ) {
                EcalChannelStatusMap::const_iterator chit = chStatus_->find( *itId );
                if ( chit != chStatus_->end() ) {
                        const int flag = (*chit).getStatusCode();
                        if ( flag >= 10 && flag <= 12) { // FIXME -- avoid hardcoded values...
                                ebDetIdToRecover->insert( *itId );
                        } else if ( flag == 13 || flag == 14 ) { // FIXME -- avoid hardcoded values...
                                ebTTDetIdToRecover->insert( (*itId).tower() );
                        }
                } else {
                        edm::LogError("EcalDetIdToBeRecoveredProducer") << "No channel status found for xtal "
                                << (*itId).rawId()
                                << "! something wrong with EcalChannelStatus in your DB? ";
                }
        }
        // -- Endcap
        for ( EEDetIdCollection::const_iterator itId = eeSrpDetId.begin(); itId != eeSrpDetId.end(); ++itId ) {
                EcalChannelStatusMap::const_iterator chit = chStatus_->find( *itId );
                if ( chit != chStatus_->end() ) {
                        int flag = (*chit).getStatusCode() ;
                        if ( flag >= 10 && flag <= 12) { // FIXME -- avoid hardcoded values...
                                eeDetIdToRecover->insert( *itId );
                        } else if ( flag == 13 || flag == 14 ) { // FIXME -- avoid hardcoded values...
                                eeSCDetIdToRecover->insert( EcalScDetId(1+((*itId).ix()-1)/5,1+((*itId).iy()-1)/5,(*itId).zside()) );
                        }
                } else {
                        edm::LogError("EcalDetIdToBeRecoveredProducer") << "No channel status found for xtal "
                                << (*itId).rawId()
                                << "! something wrong with EcalChannelStatus in your DB? ";
                }
        }
        
        
        // loop over electronics id associated with TT and SC
        for (size_t t = 0; t < ttColls.size(); ++t) {
          const EcalElectronicsIdCollection& coll = *(ttColls[t]);

          for (size_t i = 0; i < coll.size(); ++i)
          {
            const EcalElectronicsId elId = coll[i];
            const EcalSubdetector subdet = elId.subdet();
            const DetId detId = ecalMapping_->getDetId(elId);
            
            if (subdet == EcalBarrel) { // elId pointing to TT
              // get list of crystals corresponding to TT
              const EcalTrigTowerDetId ttId( ttMap_->towerOf(detId) );
              const std::vector<DetId>& vid = ttMap_->constituentsOf(ttId);
              
              for (size_t j = 0; j < vid.size(); ++j) {
                const EBDetId ebdi(vid[j]);
                if (include(ebSrpDetId, ebdi)) {
                  ebDetIdToRecover->insert(ebdi);
                  ebTTDetIdToRecover->insert(ebdi.tower());
                }
              }
            }
            else if (subdet == EcalEndcap) { // elId pointing to SC
              // extract list of crystals corresponding to SC
              const EcalScDetId scId(detId);
              std::vector<DetId> vid;
              for(int dx=1; dx<=5; ++dx) {
                for(int dy=1; dy<=5; ++dy) {
                  const int ix = (scId.ix()-1)*5 + dx;
                  const int iy = (scId.iy()-1)*5 + dy;
                  const int iz = scId.zside();
                  if(EEDetId::validDetId(ix, iy, iz))
                    vid.push_back(EEDetId(ix, iy, iz));
                }
              }
              
              for (size_t j = 0; j < vid.size(); ++j) {
                const EEDetId eedi(vid[i]);
                if (include(eeSrpDetId, eedi)) {
                  eeDetIdToRecover->insert(eedi);
                  eeSCDetIdToRecover->insert(EcalScDetId(eedi));
                }
              }
            }
            else
              edm::LogWarning("EcalDetIdToBeRecoveredProducer")
                << "Incorrect EcalSubdetector = " << subdet
                << " in EcalElectronicsIdCollection collection ";
          }
        }

        // return the collections
        ev.put( ebDetIdToRecover, ebDetIdCollection_ );
        ev.put( eeDetIdToRecover, eeDetIdCollection_ );
        ev.put( ebTTDetIdToRecover, ttDetIdCollection_ );
        ev.put( eeSCDetIdToRecover, scDetIdCollection_ );
}

void EcalDetIdToBeRecoveredProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("ebIntegrityChIdErrors",edm::InputTag("ecalDigis","EcalIntegrityChIdErrors"));
  desc.add<std::string>("ebDetIdToBeRecovered","ebDetId");
  desc.add<edm::InputTag>("integrityTTIdErrors",edm::InputTag("ecalDigis","EcalIntegrityTTIdErrors"));
  desc.add<edm::InputTag>("eeIntegrityGainErrors",edm::InputTag("ecalDigis","EcalIntegrityGainErrors"));
  desc.add<std::string>("ebFEToBeRecovered","ebFE");
  desc.add<edm::InputTag>("ebIntegrityGainErrors",edm::InputTag("ecalDigis","EcalIntegrityGainErrors"));
  desc.add<std::string>("eeDetIdToBeRecovered","eeDetId");
  desc.add<edm::InputTag>("eeIntegrityGainSwitchErrors",edm::InputTag("ecalDigis","EcalIntegrityGainSwitchErrors"));
  desc.add<edm::InputTag>("eeIntegrityChIdErrors",edm::InputTag("ecalDigis","EcalIntegrityChIdErrors"));
  desc.add<edm::InputTag>("ebIntegrityGainSwitchErrors",edm::InputTag("ecalDigis","EcalIntegrityGainSwitchErrors"));
  desc.add<edm::InputTag>("ebSrFlagCollection",edm::InputTag("ecalDigis"));
  desc.add<std::string>("eeFEToBeRecovered","eeFE");
  desc.add<edm::InputTag>("integrityBlockSizeErrors",edm::InputTag("ecalDigis","EcalIntegrityBlockSizeErrors"));
  desc.add<edm::InputTag>("eeSrFlagCollection",edm::InputTag("ecalDigis"));
  descriptions.add("ecalDetIdToBeRecovered",desc);
}


#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE( EcalDetIdToBeRecoveredProducer );
