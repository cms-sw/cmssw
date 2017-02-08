#include "RecoLocalCalo/EcalRecProducers/plugins/EcalUncalibRecHitWorkerWeights.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CondFormats/DataRecord/interface/EcalGainRatiosRcd.h"
#include "CondFormats/DataRecord/interface/EcalPedestalsRcd.h"
#include "CondFormats/DataRecord/interface/EcalWeightXtalGroupsRcd.h"
#include "CondFormats/DataRecord/interface/EcalTBWeightsRcd.h"

#include <FWCore/ParameterSet/interface/ConfigurationDescriptions.h>
#include <FWCore/ParameterSet/interface/ParameterSetDescription.h>

EcalUncalibRecHitWorkerWeights::EcalUncalibRecHitWorkerWeights(const edm::ParameterSet&ps, edm::ConsumesCollector& c) :
  EcalUncalibRecHitWorkerBaseClass(ps,c)
{
}

void
EcalUncalibRecHitWorkerWeights::set(const edm::EventSetup& es)
{
        es.get<EcalGainRatiosRcd>().get(gains);
        es.get<EcalPedestalsRcd>().get(peds);
        es.get<EcalWeightXtalGroupsRcd>().get(grps);
        es.get<EcalTBWeightsRcd>().get(wgts);
}


bool
EcalUncalibRecHitWorkerWeights::run( const edm::Event & evt,
                const EcalDigiCollection::const_iterator & itdg,
                EcalUncalibratedRecHitCollection & result )
{
        DetId detid(itdg->id());

        const EcalPedestals::Item * aped = 0;
        const EcalMGPAGainRatio * aGain = 0;
        const EcalXtalGroupId * gid = 0;
        EcalTBWeights::EcalTDCId tdcid(1);

        if (detid.subdetId()==EcalEndcap) {
                unsigned int hashedIndex = EEDetId(detid).hashedIndex();
                aped  = &peds->endcap(hashedIndex);
                aGain = &gains->endcap(hashedIndex);
                gid   = &grps->endcap(hashedIndex);
        } else {
                unsigned int hashedIndex = EBDetId(detid).hashedIndex();
                aped  = &peds->barrel(hashedIndex);
                aGain = &gains->barrel(hashedIndex);
                gid   = &grps->barrel(hashedIndex);
        }

        pedVec[0] = aped->mean_x12;
        pedVec[1] = aped->mean_x6;
        pedVec[2] = aped->mean_x1;
        pedRMSVec[0] = aped->rms_x12;
        pedRMSVec[1] = aped->rms_x6;
        pedRMSVec[2] = aped->rms_x1;
        gainRatios[0] = 1.;
        gainRatios[1] = aGain->gain12Over6();
        gainRatios[2] = aGain->gain6Over1()*aGain->gain12Over6();

        // now lookup the correct weights in the map
        EcalTBWeights::EcalTBWeightMap const & wgtsMap = wgts->getMap();
        EcalTBWeights::EcalTBWeightMap::const_iterator wit;
        wit = wgtsMap.find( std::make_pair(*gid,tdcid) );
        if( wit == wgtsMap.end() ) {
                edm::LogError("EcalUncalibRecHitError") << "No weights found for EcalGroupId: " 
                        << gid->id() << " and  EcalTDCId: " << tdcid
                        << "\n  skipping digi with id: " << detid.rawId();

                return false;
        }
        const EcalWeightSet& wset = wit->second; // this is the EcalWeightSet

        const EcalWeightSet::EcalWeightMatrix& mat1 = wset.getWeightsBeforeGainSwitch();
        const EcalWeightSet::EcalWeightMatrix& mat2 = wset.getWeightsAfterGainSwitch();
//        const EcalWeightSet::EcalChi2WeightMatrix& mat3 = wset.getChi2WeightsBeforeGainSwitch();
//        const EcalWeightSet::EcalChi2WeightMatrix& mat4 = wset.getChi2WeightsAfterGainSwitch();

        weights[0] = &mat1;
        weights[1] = &mat2;

//        chi2mat[0] = &mat3;
//        chi2mat[1] = &mat4;
/*
        if (detid.subdetId()==EcalEndcap) {
                result.push_back(uncalibMaker_endcap_.makeRecHit(*itdg, pedVec, gainRatios, weights, chi2mat));
        } else {
                result.push_back(uncalibMaker_barrel_.makeRecHit(*itdg, pedVec, gainRatios, weights, chi2mat));
        }
*/
        if (detid.subdetId()==EcalEndcap) {
                result.push_back(uncalibMaker_endcap_.makeRecHit(*itdg, pedVec, pedRMSVec, gainRatios, weights, testbeamEEShape));
        } else {
                result.push_back(uncalibMaker_barrel_.makeRecHit(*itdg, pedVec, pedRMSVec, gainRatios, weights, testbeamEBShape));
        }
        return true;
}

edm::ParameterSetDescription
EcalUncalibRecHitWorkerWeights::getAlgoDescription() {

  edm::ParameterSetDescription psd;
  return psd;
}

#include "FWCore/Framework/interface/MakerMacros.h"
#include "RecoLocalCalo/EcalRecProducers/interface/EcalUncalibRecHitWorkerFactory.h"
DEFINE_EDM_PLUGIN( EcalUncalibRecHitWorkerFactory, EcalUncalibRecHitWorkerWeights, "EcalUncalibRecHitWorkerWeights" );
#include "RecoLocalCalo/EcalRecProducers/interface/EcalUncalibRecHitFillDescriptionWorkerFactory.h"
DEFINE_EDM_PLUGIN( EcalUncalibRecHitFillDescriptionWorkerFactory, EcalUncalibRecHitWorkerWeights, "EcalUncalibRecHitWorkerWeights");
