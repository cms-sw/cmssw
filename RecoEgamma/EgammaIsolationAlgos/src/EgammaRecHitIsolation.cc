//*****************************************************************************
// File:      EgammaRecHitIsolation.cc
// ----------------------------------------------------------------------------
// OrigAuth:  Matthias Mozer, hacked by Sam Harper (ie the ugly stuff is mine)
// Institute: IIHE-VUB, RAL
//=============================================================================
//*****************************************************************************
//C++ includes
#include <vector>
#include <functional>

//ROOT includes
#include <Math/VectorUtil.h>

//CMSSW includes
#include "RecoEgamma/EgammaIsolationAlgos/interface/EgammaRecHitIsolation.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"
#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/CaloRecHit/interface/CaloClusterFwd.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"
#include "DataFormats/Math/interface/deltaPhi.h"

using namespace std;

EgammaRecHitIsolation::EgammaRecHitIsolation (double extRadius,
                                              double intRadius,
                                              double etaSlice,
                                              double etLow,
                                              double eLow,
                                              edm::ESHandle<CaloGeometry> theCaloGeom,
                                              CaloRecHitMetaCollectionV* caloHits,
                                              const EcalSeverityLevelAlgo* sl,
                                              DetId::Detector detector):  // not used anymore, kept for compatibility
    extRadius_(extRadius),
    intRadius_(intRadius),
    etaSlice_(etaSlice),
    etLow_(etLow),
    eLow_(eLow),
    theCaloGeom_(theCaloGeom) ,  
    caloHits_(caloHits),
    sevLevel_(sl),
    useNumCrystals_(false),
    vetoClustered_(false),
    ecalBarHits_(0),
    chStatus_(0),
    severityLevelCut_(-1),
    //severityRecHitThreshold_(0),
    //spId_(EcalSeverityLevelAlgo::kSwissCross),
    //spIdThreshold_(0),
    v_chstatus_(0)
{
    //set up the geometry and selector
    const CaloGeometry* caloGeom = theCaloGeom_.product();
    subdet_[0] = caloGeom->getSubdetectorGeometry(DetId::Ecal,EcalBarrel);
    subdet_[1] = caloGeom->getSubdetectorGeometry(DetId::Ecal,EcalEndcap);

}

EgammaRecHitIsolation::~EgammaRecHitIsolation ()
{
}

double EgammaRecHitIsolation::getSum_(const reco::Candidate* emObject,bool returnEt) const
{

    double energySum = 0.;
    if (caloHits_){
        //Take the SC position
        reco::SuperClusterRef sc = emObject->get<reco::SuperClusterRef>();
        math::XYZPoint theCaloPosition = sc.get()->position();
        GlobalPoint pclu (theCaloPosition.x () ,
                theCaloPosition.y () ,
                theCaloPosition.z () );
        double etaclus = pclu.eta();
        double phiclus = pclu.phi();
        double r2 = intRadius_*intRadius_;



        std::vector< std::pair<DetId, float> >::const_iterator rhIt;


        for(int subdetnr=0; subdetnr<=1 ; subdetnr++){  // look in barrel and endcap
            CaloSubdetectorGeometry::DetIdSet chosen = subdet_[subdetnr]->getCells(pclu,extRadius_);// select cells around cluster
            CaloRecHitMetaCollectionV::const_iterator j=caloHits_->end();
            for (CaloSubdetectorGeometry::DetIdSet::const_iterator  i = chosen.begin ();i!= chosen.end ();++i){//loop selected cells

                j=caloHits_->find(*i); // find selected cell among rechits
                if( j!=caloHits_->end()){ // add rechit only if available 
                    const  GlobalPoint & position = theCaloGeom_.product()->getPosition(*i);
                    double eta = position.eta();
                    double phi = position.phi();
                    double etaDiff = eta - etaclus;
                    double phiDiff= reco::deltaPhi(phi,phiclus);
                    double energy = j->energy();

                    if(useNumCrystals_) {
                        if( fabs(etaclus) < 1.479 ) {  // Barrel num crystals, crystal width = 0.0174
                            if ( fabs(etaDiff) < 0.0174*etaSlice_) continue;  
                            if ( sqrt(etaDiff*etaDiff + phiDiff*phiDiff) < 0.0174*intRadius_) continue; 
                        } else {                       // Endcap num crystals, crystal width = 0.00864*fabs(sinh(eta))
                            if ( fabs(etaDiff) < 0.00864*fabs(sinh(eta))*etaSlice_) continue;  
                            if ( sqrt(etaDiff*etaDiff + phiDiff*phiDiff) < 0.00864*fabs(sinh(eta))*intRadius_) continue; 
                        }
                    } else {
                        if ( fabs(etaDiff) < etaSlice_) continue;  // jurassic strip cut
                        if ( etaDiff*etaDiff + phiDiff*phiDiff < r2) continue; // jurassic exclusion cone cut
                    }

                    //Check if RecHit is in SC
                    if(vetoClustered_) {

                        //Loop over basic clusters:
                        bool isClustered = false;
                        for(reco::CaloCluster_iterator bcIt = sc->clustersBegin();bcIt != sc->clustersEnd(); ++bcIt) {
                            for(rhIt = (*bcIt)->hitsAndFractions().begin();rhIt != (*bcIt)->hitsAndFractions().end(); ++rhIt) {
                                if( rhIt->first == *i ) isClustered = true;
                                if( isClustered ) break;
                            }
                            if( isClustered ) break;
                        } //end loop over basic clusters

                        if(isClustered) continue;
                    }  //end if removeClustered

                    //Severity level check
                    //make sure we have a barrel rechit                                     
                    //call the severity level method                                        
                    //passing the EBDetId                                                   
                    //the rechit collection in order to calculate the swiss crss            
                    //and the EcalChannelRecHitRcd                                          
                    //only consider rechits with ET >                                       
                    //the SpikeId method (currently kE1OverE9 or kSwissCross)               
                    //cut value for above                                                   
                    //then if the severity level is too high, we continue to the next rechit

                    if( severityLevelCut_!=-1 && ecalBarHits_ && 
                        sevLevel_->severityLevel(EBDetId(j->detid()), *ecalBarHits_) >= severityLevelCut_) 
                      continue;                    
                    //                            *chStatus_,                           
                    //        severityRecHitThreshold_,             
                    //        spId_,                                
                    //        spIdThreshold_                        
                    //    ) >= severityLevelCut_) continue;         




                    //Check based on flags to protect from recovered channels from non-read towers
                    //Assumption is that v_chstatus_ is empty unless doFlagChecks() has been called
                    std::vector<int>::const_iterator vit = std::find( v_chstatus_.begin(), v_chstatus_.end(),  ((EcalRecHit*)(&*j))->recoFlag() );
                    if ( vit != v_chstatus_.end() ) continue; // the recHit has to be excluded from the iso sum


                    double et = energy*position.perp()/position.mag();
                    if ( fabs(et) > etLow_ && fabs(energy) > eLow_){ //Changed energy --> fabs(energy)
                        if(returnEt) energySum+=et;
                        else energySum+=energy;
                    }

                }                //End if not end of list
            }                    //End loop over rechits
        }                        //End loop over barrel/endcap
    }                            //End if caloHits_
    return energySum;
}



double EgammaRecHitIsolation::getSum_(const reco::SuperCluster* sc,bool returnEt) const
{

    double energySum = 0.;
    if (caloHits_){
        //Take the SC position
 
        math::XYZPoint theCaloPosition = sc->position();
        GlobalPoint pclu (theCaloPosition.x () ,
                theCaloPosition.y () ,
                theCaloPosition.z () );
        double etaclus = pclu.eta();
        double phiclus = pclu.phi();
        double r2 = intRadius_*intRadius_;

        std::vector< std::pair<DetId, float> >::const_iterator rhIt;


        for(int subdetnr=0; subdetnr<=1 ; subdetnr++){  // look in barrel and endcap
            CaloSubdetectorGeometry::DetIdSet chosen = subdet_[subdetnr]->getCells(pclu,extRadius_);// select cells around cluster
            CaloRecHitMetaCollectionV::const_iterator j=caloHits_->end();
            for (CaloSubdetectorGeometry::DetIdSet::const_iterator  i = chosen.begin ();i!= chosen.end ();++i){//loop selected cells

                j=caloHits_->find(*i); // find selected cell among rechits
                if( j!=caloHits_->end()){ // add rechit only if available 
                    const  GlobalPoint & position = theCaloGeom_.product()->getPosition(*i);
                    double eta = position.eta();
                    double phi = position.phi();
                    double etaDiff = eta - etaclus;
                    double phiDiff= reco::deltaPhi(phi,phiclus);
                    double energy = j->energy();

                    if(useNumCrystals_) {
                        if( fabs(etaclus) < 1.479 ) {  // Barrel num crystals, crystal width = 0.0174
                            if ( fabs(etaDiff) < 0.0174*etaSlice_) continue;  
                            if ( sqrt(etaDiff*etaDiff + phiDiff*phiDiff) < 0.0174*intRadius_) continue; 
                        } else {                       // Endcap num crystals, crystal width = 0.00864*fabs(sinh(eta))
                            if ( fabs(etaDiff) < 0.00864*fabs(sinh(eta))*etaSlice_) continue;  
                            if ( sqrt(etaDiff*etaDiff + phiDiff*phiDiff) < 0.00864*fabs(sinh(eta))*intRadius_) continue; 
                        }
                    } else {
                        if ( fabs(etaDiff) < etaSlice_) continue;  // jurassic strip cut
                        if ( etaDiff*etaDiff + phiDiff*phiDiff < r2) continue; // jurassic exclusion cone cut
                    }

                    //Check if RecHit is in SC
                    if(vetoClustered_) {

                        //Loop over basic clusters:
                        bool isClustered = false;
                        for(reco::CaloCluster_iterator bcIt = sc->clustersBegin();bcIt != sc->clustersEnd(); ++bcIt) {
                            for(rhIt = (*bcIt)->hitsAndFractions().begin();rhIt != (*bcIt)->hitsAndFractions().end(); ++rhIt) {
                                if( rhIt->first == *i ) isClustered = true;
                                if( isClustered ) break;
                            }
                            if( isClustered ) break;
                        } //end loop over basic clusters

                        if(isClustered) continue;
                    }  //end if removeClustered

                    //Severity level check
                    //make sure we have a barrel rechit                                     
                    //call the severity level method                                        
                    //passing the EBDetId                                                   
                    //the rechit collection in order to calculate the swiss crss            
                    //and the EcalChannelRecHitRcd                                          
                    //only consider rechits with ET >                                       
                    //the SpikeId method (currently kE1OverE9 or kSwissCross)               
                    //cut value for above                                                   
                    //then if the severity level is too high, we continue to the next rechit

                    if( severityLevelCut_!=-1 && ecalBarHits_ && 
                        sevLevel_->severityLevel(EBDetId(j->detid()), *ecalBarHits_) >= severityLevelCut_) 
                      continue;                    
                    //                            *chStatus_,                           
                    //        severityRecHitThreshold_,             
                    //        spId_,                                
                    //        spIdThreshold_                        
                    //    ) >= severityLevelCut_) continue;         




                    //Check based on flags to protect from recovered channels from non-read towers
                    //Assumption is that v_chstatus_ is empty unless doFlagChecks() has been called
                    std::vector<int>::const_iterator vit = std::find( v_chstatus_.begin(), v_chstatus_.end(),  ((EcalRecHit*)(&*j))->recoFlag() );
                    if ( vit != v_chstatus_.end() ) continue; // the recHit has to be excluded from the iso sum


                    double et = energy*position.perp()/position.mag();
                    if ( fabs(et) > etLow_ && fabs(energy) > eLow_){ //Changed energy --> fabs(energy)
                        if(returnEt) energySum+=et;
                        else energySum+=energy;
                    }

                }                //End if not end of list
            }                    //End loop over rechits
        }                        //End loop over barrel/endcap
    }                            //End if caloHits_
    return energySum;
}

