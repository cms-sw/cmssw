#include "RecoLocalCalo/EcalRecProducers/plugins/EcalUncalibRecHitWorkerGlobal.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CondFormats/DataRecord/interface/EcalGainRatiosRcd.h"
#include "CondFormats/DataRecord/interface/EcalPedestalsRcd.h"
#include "CondFormats/DataRecord/interface/EcalWeightXtalGroupsRcd.h"
#include "CondFormats/DataRecord/interface/EcalTBWeightsRcd.h"
#include "CondFormats/DataRecord/interface/EcalTimeCalibConstantsRcd.h"

EcalUncalibRecHitWorkerGlobal::EcalUncalibRecHitWorkerGlobal(const edm::ParameterSet&ps) :
        EcalUncalibRecHitWorkerBaseClass(ps)
{
        // ratio method parameters
        EBtimeFitParameters_ = ps.getParameter<std::vector<double> >("EBtimeFitParameters"); 
        EEtimeFitParameters_ = ps.getParameter<std::vector<double> >("EEtimeFitParameters"); 
        EBamplitudeFitParameters_ = ps.getParameter<std::vector<double> >("EBamplitudeFitParameters");
        EEamplitudeFitParameters_ = ps.getParameter<std::vector<double> >("EEamplitudeFitParameters");
        EBtimeFitLimits_.first  = ps.getParameter<double>("EBtimeFitLimits_Lower");
        EBtimeFitLimits_.second = ps.getParameter<double>("EBtimeFitLimits_Upper");
        EEtimeFitLimits_.first  = ps.getParameter<double>("EEtimeFitLimits_Lower");
        EEtimeFitLimits_.second = ps.getParameter<double>("EEtimeFitLimits_Upper");
        outOfTimeThresh_ = ps.getParameter<double>("outOfTimeThreshold");
        amplitudeThreshEB_ = ps.getParameter<double>("amplitudeThresholdEB");
        amplitudeThreshEE_ = ps.getParameter<double>("amplitudeThresholdEE");
        // leading edge parameters
        ebPulseShape_ = ps.getParameter<std::vector<double> >("ebPulseShape");
        eePulseShape_ = ps.getParameter<std::vector<double> >("eePulseShape");
}

void
EcalUncalibRecHitWorkerGlobal::set(const edm::EventSetup& es)
{
        // common setup
        es.get<EcalGainRatiosRcd>().get(gains);
        es.get<EcalPedestalsRcd>().get(peds);

        // for the weights method
        es.get<EcalWeightXtalGroupsRcd>().get(grps);
        es.get<EcalTBWeightsRcd>().get(wgts);

        // for the ratio method

        // for the leading edge method
        es.get<EcalTimeCalibConstantsRcd>().get(itime);
}


// check saturation: 5 samples with gainId = 0
template < class C >
int EcalUncalibRecHitWorkerGlobal::isSaturated(const C & dataFrame)
{
        //bool saturated_ = 0;
        int cnt;
        for (int j = 0; j < C::MAXSAMPLES - 5; ++j) {
                cnt = 0;
                for (int i = j; i < (j + 5) && i < C::MAXSAMPLES; ++i) {
                        if ( dataFrame.sample(i).gainId() == 0 ) ++cnt;
                }
                if ( cnt == 5 ) return j-1 ; // the last unsaturated sample
        }
        return -1; // no saturation found
}



bool
EcalUncalibRecHitWorkerGlobal::run( const edm::Event & evt,
                const EcalDigiCollection::const_iterator & itdg,
                EcalUncalibratedRecHitCollection & result )
{
        DetId detid(itdg->id());

        // intelligence for recHit computation
        EcalUncalibratedRecHit uncalibRecHit;
        
        
        const EcalPedestals::Item * aped = 0;
        const EcalMGPAGainRatio * aGain = 0;
        const EcalXtalGroupId * gid = 0;

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
        gainRatios[0] = 1.;
        gainRatios[1] = aGain->gain12Over6();
        gainRatios[2] = aGain->gain6Over1()*aGain->gain12Over6();


        // === amplitude computation ===
        int leadingSample = -1;
        if (detid.subdetId()==EcalEndcap) {
                leadingSample = ((EcalDataFrame)(*itdg)).lastUnsaturatedSample();
        } else {
                leadingSample = ((EcalDataFrame)(*itdg)).lastUnsaturatedSample();
        }

        if ( leadingSample >= 0 ) { // saturation
                if ( leadingSample != 4 ) {
                        // all samples different from the fifth are not reliable for the amplitude estimation
                        // put by default the energy at the saturation threshold and flag as saturated
                        float sratio = 1;
                        if ( detid.subdetId()==EcalBarrel) {
                                sratio = ebPulseShape_[5] / ebPulseShape_[4];
                        } else {
                                sratio = eePulseShape_[5] / eePulseShape_[4];
                        }
                        uncalibRecHit = EcalUncalibratedRecHit( (*itdg).id(), 4095*12*sratio, 0, 0, 0);
                        uncalibRecHit.setRecoFlag( EcalUncalibratedRecHit::kSaturated );
                } else {
                        // compute the right bin of the pulse shape using time calibration constants
                        // -- for the moment this is not used
                        EcalTimeCalibConstantMap::const_iterator it = itime->find( detid );
                        EcalTimeCalibConstant itimeconst = 0;
                        if( it != itime->end() ) {
                                itimeconst = (*it);
                        } else {
                                edm::LogError("EcalRecHitError") << "No time intercalib const found for xtal "
                                        << detid.rawId()
                                        << "! something wrong with EcalTimeCalibConstants in your DB? ";
                        }
                        // float clockToNsConstant = 25.;
                        // reconstruct the rechit
                        if (detid.subdetId()==EcalEndcap) {
                                leadingEdgeMethod_endcap_.setPulseShape( eePulseShape_ );
                                // float mult = (float)eePulseShape_.size() / (float)(*itdg).size();
                                // bin (or some analogous mapping) will be used instead of the leadingSample
                                //int bin  = (int)(( (mult * leadingSample + mult/2) * clockToNsConstant + itimeconst ) / clockToNsConstant);
                                // bin is not uset for the moment
                                leadingEdgeMethod_endcap_.setLeadingEdgeSample( leadingSample );
                                uncalibRecHit = leadingEdgeMethod_endcap_.makeRecHit(*itdg, pedVec, gainRatios, 0, 0);
                                uncalibRecHit.setRecoFlag( EcalUncalibratedRecHit::kLeadingEdgeRecovered );
                                leadingEdgeMethod_endcap_.setLeadingEdgeSample( -1 );
                        } else {
                                leadingEdgeMethod_barrel_.setPulseShape( ebPulseShape_ );
                                // float mult = (float)ebPulseShape_.size() / (float)(*itdg).size();
                                // bin (or some analogous mapping) will be used instead of the leadingSample
                                //int bin  = (int)(( (mult * leadingSample + mult/2) * clockToNsConstant + itimeconst ) / clockToNsConstant);
                                // bin is not uset for the moment
                                leadingEdgeMethod_barrel_.setLeadingEdgeSample( leadingSample );
                                uncalibRecHit = leadingEdgeMethod_barrel_.makeRecHit(*itdg, pedVec, gainRatios, 0, 0);
                                uncalibRecHit.setRecoFlag( EcalUncalibratedRecHit::kLeadingEdgeRecovered );
                                leadingEdgeMethod_barrel_.setLeadingEdgeSample( -1 );
                        }
                }
        } else {
                // weights method
                EcalTBWeights::EcalTDCId tdcid(1);
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
                const EcalWeightSet::EcalChi2WeightMatrix& mat3 = wset.getChi2WeightsBeforeGainSwitch();
                const EcalWeightSet::EcalChi2WeightMatrix& mat4 = wset.getChi2WeightsAfterGainSwitch();

                weights[0] = &mat1;
                weights[1] = &mat2;

                chi2mat[0] = &mat3;
                chi2mat[1] = &mat4;

                // get uncalibrated recHit
                if (detid.subdetId()==EcalEndcap) {
                        uncalibRecHit = weightsMethod_endcap_.makeRecHit(*itdg, pedVec, gainRatios, weights, chi2mat);
                } else {
                        uncalibRecHit = weightsMethod_barrel_.makeRecHit(*itdg, pedVec, gainRatios, weights, chi2mat);
                }

                // === time computation ===
                // ratio method
                if (detid.subdetId()==EcalEndcap) {
                                ratioMethod_endcap_.init( *itdg, pedVec, gainRatios );
                                ratioMethod_endcap_.computeTime( EEtimeFitParameters_, EEtimeFitLimits_ );
                                EcalUncalibRecHitRatioMethodAlgo<EEDataFrame>::CalculatedRecHit crh = ratioMethod_endcap_.getCalculatedRecHit();
                                uncalibRecHit.setJitter( crh.timeMax - 5 );
                                // FIXME: set the error?
                                // it time too different from 5, reconstruct the amplitude 
                                // with the ratioMethod and store it in the RecHit flags
                                if ( uncalibRecHit.amplitude() > amplitudeThreshEE_
                                     && fabs(crh.timeMax-5) > outOfTimeThresh_ ) {
                                        uncalibRecHit.setRecoFlag( EcalUncalibratedRecHit::kOutOfTime );
                                        uncalibRecHit.setOutOfTimeEnergy( crh.amplitudeMax );
                                }
                } else {
                                ratioMethod_barrel_.init( *itdg, pedVec, gainRatios );
                                ratioMethod_barrel_.computeTime( EBtimeFitParameters_, EBtimeFitLimits_ );
                                EcalUncalibRecHitRatioMethodAlgo<EBDataFrame>::CalculatedRecHit crh = ratioMethod_barrel_.getCalculatedRecHit();
                                uncalibRecHit.setJitter( crh.timeMax - 5 );
                                // FIXME: set the error?
                                // it time too different from 5, reconstruct the amplitude
                                // with the ratioMethod and store it in the RecHit flags
                                if ( uncalibRecHit.amplitude() > amplitudeThreshEB_ && 
                                     fabs(crh.timeMax-5) > outOfTimeThresh_ ) {
                                        uncalibRecHit.setRecoFlag( EcalUncalibratedRecHit::kOutOfTime );
                                        uncalibRecHit.setOutOfTimeEnergy( crh.amplitudeMax );
                                }
                }
        }


        // put the recHit in the collection
        if (detid.subdetId()==EcalEndcap) {
                result.push_back( uncalibRecHit );
        } else {
                result.push_back( uncalibRecHit );
        }
        return true;
}

#include "FWCore/Framework/interface/MakerMacros.h"
#include "RecoLocalCalo/EcalRecProducers/interface/EcalUncalibRecHitWorkerFactory.h"
DEFINE_EDM_PLUGIN( EcalUncalibRecHitWorkerFactory, EcalUncalibRecHitWorkerGlobal, "EcalUncalibRecHitWorkerGlobal" );
