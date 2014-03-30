#include "RecoLocalCalo/EcalRecProducers/plugins/EcalRecHitWorkerRecover.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "CondFormats/DataRecord/interface/EcalIntercalibConstantsRcd.h"
#include "CondFormats/DataRecord/interface/EcalTimeCalibConstantsRcd.h"
#include "CondFormats/DataRecord/interface/EcalADCToGeVConstantRcd.h"
#include "CondFormats/DataRecord/interface/EcalChannelStatusRcd.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalDetId/interface/EcalScDetId.h"
#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/EcalMapping/interface/EcalMappingRcd.h"

#include "CondFormats/EcalObjects/interface/EcalTimeCalibConstants.h"
#include "CalibCalorimetry/EcalLaserCorrection/interface/EcalLaserDbRecord.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/EDProducer.h"


EcalRecHitWorkerRecover::EcalRecHitWorkerRecover(const edm::ParameterSet&ps, edm::ConsumesCollector&  c) :
  EcalRecHitWorkerBaseClass(ps,c)
{
        rechitMaker_ = new EcalRecHitSimpleAlgo();
        // isolated channel recovery
        singleRecoveryMethod_    = ps.getParameter<std::string>("singleChannelRecoveryMethod");
        singleRecoveryThreshold_ = ps.getParameter<double>("singleChannelRecoveryThreshold");
        killDeadChannels_        = ps.getParameter<bool>("killDeadChannels");
        recoverEBIsolatedChannels_ = ps.getParameter<bool>("recoverEBIsolatedChannels");
        recoverEEIsolatedChannels_ = ps.getParameter<bool>("recoverEEIsolatedChannels");
        recoverEBVFE_ = ps.getParameter<bool>("recoverEBVFE");
        recoverEEVFE_ = ps.getParameter<bool>("recoverEEVFE");
        recoverEBFE_ = ps.getParameter<bool>("recoverEBFE");
        recoverEEFE_ = ps.getParameter<bool>("recoverEEFE");

	dbStatusToBeExcludedEE_ = ps.getParameter<std::vector<int> >("dbStatusToBeExcludedEE");
	dbStatusToBeExcludedEB_ = ps.getParameter<std::vector<int> >("dbStatusToBeExcludedEB");

        logWarningEtThreshold_EB_FE_ = ps.getParameter<double>("logWarningEtThreshold_EB_FE");
        logWarningEtThreshold_EE_FE_ = ps.getParameter<double>("logWarningEtThreshold_EE_FE");

        tpDigiToken_ = 
          c.consumes<EcalTrigPrimDigiCollection>(ps.getParameter<edm::InputTag>("triggerPrimitiveDigiCollection"));
}


void EcalRecHitWorkerRecover::set(const edm::EventSetup& es)
{
 
        es.get<EcalLaserDbRecord>().get(laser);
        es.get<CaloTopologyRecord>().get(caloTopology_);
        ecalScale_.setEventSetup( es );
        es.get<EcalMappingRcd>().get(pEcalMapping_);
        ecalMapping_ = pEcalMapping_.product();
        // geometry...
        es.get<EcalBarrelGeometryRecord>().get("EcalBarrel",pEBGeom_);
        es.get<EcalEndcapGeometryRecord>().get("EcalEndcap",pEEGeom_);
	es.get<CaloGeometryRecord>().get(caloGeometry_);
	es.get<EcalChannelStatusRcd>().get(chStatus_);
        geo_ = caloGeometry_.product();
        ebGeom_ = pEBGeom_.product();
        eeGeom_ = pEEGeom_.product();
        es.get<IdealGeometryRecord>().get(ttMap_);
        recoveredDetIds_EB_.clear();
        recoveredDetIds_EE_.clear();
	tpgscale_.setEventSetup(es);
}


bool
EcalRecHitWorkerRecover::run( const edm::Event & evt, 
                const EcalUncalibratedRecHit& uncalibRH,
                EcalRecHitCollection & result )
{
        DetId detId=uncalibRH.id();
	uint32_t flags = (0xF & uncalibRH.flags()); 

        // get laser coefficient
        //float lasercalib = laser->getLaserCorrection( detId, evt.time());

        // killDeadChannels_ = true, means explicitely kill dead channels even if the recovered energies are computed in the code
        // if you don't want to store the recovered energies in the rechit you can produce LogWarnings if logWarningEtThreshold_EB(EE)_FE>0 
	// logWarningEtThreshold_EB(EE)_FE_<0 will not compute the recovered energies at all (faster)

        if ( killDeadChannels_ ) {
                if (    (flags == EcalRecHitWorkerRecover::EB_single && !recoverEBIsolatedChannels_)
                     || (flags == EcalRecHitWorkerRecover::EE_single && !recoverEEIsolatedChannels_)
                     || (flags == EcalRecHitWorkerRecover::EB_VFE && !recoverEBVFE_)
                     || (flags == EcalRecHitWorkerRecover::EE_VFE && !recoverEEVFE_)
                     ) {
                        EcalRecHit hit( detId, 0., 0., EcalRecHit::kDead );
                        hit.setFlag( EcalRecHit::kDead)  ;
                        insertRecHit( hit, result); // insert trivial rechit with kDead flag
                        return true;
                } 
                if ( flags == EcalRecHitWorkerRecover::EB_FE && !recoverEBFE_) {
                        EcalTrigTowerDetId ttDetId( ((EBDetId)detId).tower() );
                        std::vector<DetId> vid = ttMap_->constituentsOf( ttDetId );
                        for ( std::vector<DetId>::const_iterator dit = vid.begin(); dit != vid.end(); ++dit ) {
                                EcalRecHit hit( (*dit), 0., 0., EcalRecHit::kDead );
                                hit.setFlag( EcalRecHit::kDead ) ;
                                insertRecHit( hit, result ); // insert trivial rechit with kDead flag
                        }
			if(logWarningEtThreshold_EB_FE_<0)return true; // if you don't want log warning just return true
                }
                if ( flags == EcalRecHitWorkerRecover::EE_FE && !recoverEEFE_) {
                        EEDetId id( detId );
                        EcalScDetId sc( 1+(id.ix()-1)/5, 1+(id.iy()-1)/5, id.zside() );
                        std::vector<DetId> eeC;
                        for(int dx=1; dx<=5; ++dx){
                                for(int dy=1; dy<=5; ++dy){
                                        int ix = (sc.ix()-1)*5 + dx;
                                        int iy = (sc.iy()-1)*5 + dy;
                                        int iz = sc.zside();
                                        if(EEDetId::validDetId(ix, iy, iz)){
                                                eeC.push_back(EEDetId(ix, iy, iz));
                                        }
                                }
                        }
                        for ( size_t i = 0; i < eeC.size(); ++i ) {
                                EcalRecHit hit( eeC[i], 0., 0., EcalRecHit::kDead );
                                hit.setFlag( EcalRecHit::kDead ) ;
                                insertRecHit( hit, result ); // insert trivial rechit with kDead flag
                        }
		   	if(logWarningEtThreshold_EE_FE_<0)   return true; // if you don't want log warning just return true
                }
        }

        if ( flags == EcalRecHitWorkerRecover::EB_single ) {
                    // recover as single dead channel
                    ebDeadChannelCorrector.setCaloTopology(caloTopology_.product());

                    // channel recovery. Accepted new RecHit has the flag AcceptRecHit=TRUE
                    bool AcceptRecHit=true;
                    EcalRecHit hit = ebDeadChannelCorrector.correct( detId, result, singleRecoveryMethod_, singleRecoveryThreshold_, &AcceptRecHit);

                    if ( hit.energy() != 0 and AcceptRecHit == true ) {
                        hit.setFlag( EcalRecHit::kNeighboursRecovered ) ;
                    } else {
                        // recovery failed
                        hit.setFlag( EcalRecHit::kDead ) ;
                    }
                    insertRecHit( hit, result );
                
        } else if ( flags == EcalRecHitWorkerRecover::EE_single ) {
                    // recover as single dead channel
		    eeDeadChannelCorrector.setCaloTopology(caloTopology_.product());

                    // channel recovery. Accepted new RecHit has the flag AcceptRecHit=TRUE
                    bool AcceptRecHit=true;
                    EcalRecHit hit = eeDeadChannelCorrector.correct( detId, result, singleRecoveryMethod_, singleRecoveryThreshold_, &AcceptRecHit);
                    if ( hit.energy() != 0 and AcceptRecHit == true ) {
                        hit.setFlag( EcalRecHit::kNeighboursRecovered ) ;
                    } else {
                       // recovery failed
                       hit.setFlag( EcalRecHit::kDead ) ;
                    }
                    insertRecHit( hit, result );
                
        } else if ( flags == EcalRecHitWorkerRecover::EB_VFE ) {
                // recover as dead VFE
                EcalRecHit hit( detId, 0., 0.);
                hit.setFlag( EcalRecHit::kDead ) ;
                // recovery not implemented
                insertRecHit( hit, result );
        } else if ( flags == EcalRecHitWorkerRecover::EB_FE ) {
                // recover as dead TT

                EcalTrigTowerDetId ttDetId( ((EBDetId)detId).tower() );
                edm::Handle<EcalTrigPrimDigiCollection> pTPDigis;
                evt.getByToken(tpDigiToken_, pTPDigis);
                const EcalTrigPrimDigiCollection * tpDigis = 0;               
		tpDigis = pTPDigis.product();
           
                EcalTrigPrimDigiCollection::const_iterator tp = tpDigis->find( ttDetId );
                // recover the whole trigger tower
                if ( tp != tpDigis->end() ) {
                        //std::vector<DetId> vid = ecalMapping_->dccTowerConstituents( ecalMapping_->DCCid( ttDetId ), ecalMapping_->iTT( ttDetId ) );
                        std::vector<DetId> vid = ttMap_->constituentsOf( ttDetId );
                        float tpEt  = ecalScale_.getTPGInGeV( tp->compressedEt(), tp->id() );
                        float tpEtThreshEB = logWarningEtThreshold_EB_FE_;
                        if(tpEt>tpEtThreshEB){
                                edm::LogWarning("EnergyInDeadEB_FE")<<"TP energy in the dead TT = "<<tpEt<<" at "<<ttDetId;
                        }
                        if ( !killDeadChannels_ || recoverEBFE_ ) {  
                                // democratic energy sharing
                                
                                for ( std::vector<DetId>::const_iterator dit = vid.begin(); dit != vid.end(); ++dit ) {
				        if (alreadyInserted(*dit)) continue;
				        float theta = ebGeom_->getGeometry(*dit)->getPosition().theta();
                                        float tpEt  = ecalScale_.getTPGInGeV( tp->compressedEt(), tp->id() );
                                        if(checkChannelStatus(*dit, dbStatusToBeExcludedEB_)){
						EcalRecHit hit( *dit, tpEt /((float)vid.size()) / sin(theta), 0.);
						hit.setFlag( EcalRecHit::kTowerRecovered ) ;
						if ( tp->compressedEt() == 0xFF ) hit.setFlag( EcalRecHit::kTPSaturated );
                                        	if ( tp->sFGVB() ) hit.setFlag( EcalRecHit::kL1SpikeFlag );
                                        	insertRecHit( hit, result );
                                	}
				}
                } else {
                        // tp not found => recovery failed
                        std::vector<DetId> vid = ttMap_->constituentsOf( ttDetId );
                        for ( std::vector<DetId>::const_iterator dit = vid.begin(); dit != vid.end(); ++dit ) {
			  if (alreadyInserted(*dit)) continue;
			  EcalRecHit hit( *dit,0., 0. );
                                hit.setFlag( EcalRecHit::kDead ) ;
                                insertRecHit( hit, result );
			}
                        }
                }
        } else if ( flags == EcalRecHitWorkerRecover::EE_FE ) {
                        // Structure for recovery:
                        // ** SC --> EEDetId constituents (eeC) --> associated Trigger Towers (aTT) --> EEDetId constituents (aTTC)
                        // ** energy for a SC EEDetId = [ sum_aTT(energy) - sum_aTTC(energy) ] / N_eeC
                        // .. i.e. the total energy of the TTs covering the SC minus 
                        // .. the energy of the recHits in the TTs but not in the SC
                        //std::vector<DetId> vid = ecalMapping_->dccTowerConstituents( ecalMapping_->DCCid( ttDetId ), ecalMapping_->iTT( ttDetId ) );
			// due to lack of implementation of the EcalTrigTowerDetId ix,iy methods in EE we compute Et recovered energies (in EB we compute E)

                        EEDetId eeId( detId );
                        EcalScDetId sc( (eeId.ix()-1)/5+1, (eeId.iy()-1)/5+1, eeId.zside() );
                        std::set<DetId> eeC;
                        for(int dx=1; dx<=5; ++dx){
                                for(int dy=1; dy<=5; ++dy){
                                        int ix = (sc.ix()-1)*5 + dx;
                                        int iy = (sc.iy()-1)*5 + dy;
                                        int iz = sc.zside();
                                        if(EEDetId::validDetId(ix, iy, iz)){
					  EEDetId id(ix, iy, iz);
					  if (checkChannelStatus(id,dbStatusToBeExcludedEE_)){
                                                eeC.insert(id);
					  } // check status
                                        }
                                }
                        }
                        
                        edm::Handle<EcalTrigPrimDigiCollection> pTPDigis;
                        evt.getByToken(tpDigiToken_, pTPDigis);
                        const EcalTrigPrimDigiCollection * tpDigis = 0;
			tpDigis = pTPDigis.product();

                        // associated trigger towers
                        std::set<EcalTrigTowerDetId> aTT;
                        for ( std::set<DetId>::const_iterator it = eeC.begin(); it!=eeC.end(); ++it ) {
                                aTT.insert( ttMap_->towerOf( *it ) );
                        }
                        // associated trigger towers: total energy
                        float totE = 0;
                        // associated trigger towers: EEDetId constituents
                        std::set<DetId> aTTC;
                        bool atLeastOneTPSaturated = false;
                        for ( std::set<EcalTrigTowerDetId>::const_iterator it = aTT.begin(); it != aTT.end(); ++it ) {
                                // add the energy of this trigger tower
                                EcalTrigPrimDigiCollection::const_iterator itTP = tpDigis->find( *it );
                                if ( itTP != tpDigis->end() ) {

					std::vector<DetId> v = ttMap_->constituentsOf( *it );

					// from the constituents, remove dead channels
					std::vector<DetId>::iterator ttcons = v.begin();
					while (ttcons != v.end()){
					  if (!checkChannelStatus(*ttcons,dbStatusToBeExcludedEE_)){
						ttcons=v.erase(ttcons);
					  } else {
					    ++ttcons;
					  }
					}// while 

                                        if ( itTP->compressedEt() == 0xFF ){ // In the case of a saturated trigger tower, a fraction
					  atLeastOneTPSaturated = true; //of the saturated energy is put in: number of xtals in dead region/total xtals in TT *63.75
                                        	
					  //Alternative recovery algorithm that I will now investigate.
					  //Estimate energy sums the energy in the working channels, then decides how much energy
					  //to put here depending on that. Duncan 20101203
					  
					  totE += estimateEnergy(itTP->id().ietaAbs(), &result, eeC, v);
					  
					  /* 
					     These commented out lines use
					     64GeV*fraction of the TT overlapping the dead FE
					    
					  int count = 0;
					  for (std::vector<DetId>::const_iterator idsit = v.begin(); idsit != v.end(); ++ idsit){
					  std::set<DetId>::const_iterator itFind = eeC.find(*idsit);
					  if (itFind != eeC.end())
					  ++count;
					  }
					  //std::cout << count << ", " << v.size() << std::endl;
					  totE+=((float)count/(float)v.size())* ((it->ietaAbs()>26)?2*ecalScale_.getTPGInGeV( itTP->compressedEt(), itTP->id() ):ecalScale_.getTPGInGeV( itTP->compressedEt(), itTP->id() ));*/
                                        }
                                        else {totE += ((it->ietaAbs()>26)?2:1)*ecalScale_.getTPGInGeV( itTP->compressedEt(), itTP->id() );}
                                        
                                
					// get the trigger tower constituents
					
					if (itTP->compressedEt() == 0){ // If there's no energy in TT, the constituents are removed from the recovery.
						 for (size_t i = 0 ; i < v.size(); ++i)
							 eeC.erase(v[i]);
					}
					else if (itTP->compressedEt()!=0xFF){ //If it's saturated the energy has already been determined, so we do not want to subtract any channels
					 for ( size_t j = 0; j < v.size(); ++j ) {
						 aTTC.insert( v[j] );
					 }
					}
                                    
                                }
                        }
                        // remove crystals of dead SC
                        // (this step is not needed if sure that SC crystals are not 
                        // in the recHit collection)
			
			for ( std::set<DetId>::const_iterator it = eeC.begin(); it != eeC.end(); ++it ) {
			  aTTC.erase(*it);
                        }
                        // compute the total energy for the dead SC
                        const EcalRecHitCollection * hits = &result;
                        for ( std::set<DetId>::const_iterator it = aTTC.begin(); it != aTTC.end(); ++it ) {
                                EcalRecHitCollection::const_iterator jt = hits->find( *it );
                                if ( jt != hits->end() ) {
				  float energy = jt->energy(); // Correct conversion to Et
				  float eta = geo_->getPosition(jt->id()).eta();
				  float pf = 1.0/cosh(eta);
				  //float theta = eeGeom_->getGeometry( *it )->getPosition().theta();
				  // use Et instead of E, consistent with the Et estimation of the associated TT
				  totE -= energy*pf;
                                }
                        }
                        

			float scEt = totE;
			float scEtThreshEE = logWarningEtThreshold_EE_FE_;
			if(scEt>scEtThreshEE){
				edm::LogWarning("EnergyInDeadEE_FE")<<"TP energy in the dead TT = "<<scEt<<" at "<<sc;
			}

                        // assign the energy to the SC crystals
			if ( !killDeadChannels_ || recoverEEFE_ ) { // if eeC is empty, i.e. there are no hits 
			                                            // in the tower, nothing is returned. No negative values from noise.
			  for ( std::set<DetId>::const_iterator it = eeC.begin(); it != eeC.end(); ++it ) {

			    float eta = geo_->getPosition(*it).eta(); //Convert back to E from Et for the recovered hits
			    float pf = 1.0/cosh(eta);
			    EcalRecHit hit( *it, totE / ((float)eeC.size()*pf), 0);
			    
			    if (atLeastOneTPSaturated) hit.setFlag(EcalRecHit::kTPSaturated );                            
			    hit.setFlag(EcalRecHit::kTowerRecovered); 
			    insertRecHit( hit, result );
			    
			  }// for
                        }// if 
        }
        return true;
}

float EcalRecHitWorkerRecover::estimateEnergy(int ieta, EcalRecHitCollection* hits, const std::set<DetId>& sId, const std::vector<DetId>& vId  ) {
	
	float xtalE=0;
	int count = 0;
	for (std::vector<DetId>::const_iterator vIdit = vId.begin(); vIdit != vId.end(); ++ vIdit){
		std::set<DetId>::const_iterator sIdit = sId.find(*vIdit);
		if (sIdit==sId.end()){
			float energy = hits->find(*vIdit)->energy();
			float eta = geo_->getPosition(*vIdit).eta();
			float pf = 1.0/cosh(eta);
			xtalE += energy*pf;
			count++;
		}
	}
	
	if (count==0) {                    // If there are no overlapping crystals return saturated value.

	  double etsat = tpgscale_.getTPGInGeV(0xFF,
					       ttMap_->towerOf(*vId.begin())); // get saturation value for the first
                                                                               // constituent, for the others it's the same 

	  return etsat/cosh(ieta)*(ieta>26?2:1); // account for duplicated TT in EE for ieta>26
	}
	else return xtalE*((vId.size()/(float)count) - 1)*(ieta>26?2:1);
	
	
}


void EcalRecHitWorkerRecover::insertRecHit( const EcalRecHit &hit, EcalRecHitCollection &collection )
{
        // skip already inserted DetId's and raise a log warning
        if ( alreadyInserted( hit.id() ) ) {
	  edm::LogWarning("EcalRecHitWorkerRecover") << "DetId already recovered! Skipping...";
                return;
        }
        EcalRecHitCollection::iterator it = collection.find( hit.id() );
        if ( it == collection.end() ) {
                // insert the hit in the collection
                collection.push_back( hit );
        } else {
                // overwrite existing recHit
                *it = hit;
        }
        if ( hit.id().subdetId() == EcalBarrel ) {
                recoveredDetIds_EB_.insert( hit.id() );
        } else if ( hit.id().subdetId() == EcalEndcap ) {
                recoveredDetIds_EE_.insert( hit.id() );
        } else {
                edm::LogError("EcalRecHitWorkerRecover::InvalidDetId") << "Invalid DetId " << hit.id().rawId();
        }
}


bool EcalRecHitWorkerRecover::alreadyInserted( const DetId & id )
{
        bool res = false;
        if ( id.subdetId() == EcalBarrel ) {
                res = ( recoveredDetIds_EB_.find( id ) != recoveredDetIds_EB_.end() );
        } else if ( id.subdetId() == EcalEndcap ) {
                res = ( recoveredDetIds_EE_.find( id ) != recoveredDetIds_EE_.end() );
        } else {
                edm::LogError("EcalRecHitWorkerRecover::InvalidDetId") << "Invalid DetId " << id.rawId();
        }
        return res;
}

// In the future, this will be used to calibrate the TT energy. There is a dependance on
// eta at lower energies that can be corrected for here after more validation.
float EcalRecHitWorkerRecover::recCheckCalib(float eTT, int ieta){
	 
	 return eTT;
	 
}

// return false is the channel has status  in the list of statusestoexclude
// true otherwise (channel ok)
bool EcalRecHitWorkerRecover::checkChannelStatus(const DetId& id, 
						 const std::vector<int>& statusestoexclude){
  

  if (!chStatus_.isValid())     
    edm::LogError("ObjectNotFound") << "Channel Status not set"; 
  
  
  EcalChannelStatus::const_iterator chIt = chStatus_->find( id );
  uint16_t dbStatus = 0;
  if ( chIt != chStatus_->end() ) {
    dbStatus = chIt->getStatusCode();
  } else {
    edm::LogError("ObjectNotFound") << "No channel status found for xtal " 
				    << id.rawId() 
				    << "! something wrong with EcalChannelStatus in your DB? ";
  }
  
  for (std::vector<int>::const_iterator status = statusestoexclude.begin();
       status!= statusestoexclude.end(); ++status){
    
    if ( *status == dbStatus) return false;
    
  }

  return true;
}

#include "FWCore/Framework/interface/MakerMacros.h"
#include "RecoLocalCalo/EcalRecProducers/interface/EcalRecHitWorkerFactory.h"
DEFINE_EDM_PLUGIN( EcalRecHitWorkerFactory, EcalRecHitWorkerRecover, "EcalRecHitWorkerRecover" );
