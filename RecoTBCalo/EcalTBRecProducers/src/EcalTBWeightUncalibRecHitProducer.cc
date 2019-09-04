/** \class EcalTBWeightUncalibRecHitProducer
 *   produce ECAL uncalibrated rechits from dataframes
 *
  *
  *  $Alex Zabi$
  *  Modification to detect first sample to switch gain.
  *  used for amplitude recontruction at high energy
  *  Add TDC convention option (P. Meridiani)
  *
  */
#include "RecoTBCalo/EcalTBRecProducers/interface/EcalTBWeightUncalibRecHitProducer.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalDigi/interface/EcalMGPASample.h"
#include "DataFormats/Common/interface/Handle.h"

#include <iostream>
#include <iomanip>
#include <cmath>

#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CondFormats/EcalObjects/interface/EcalPedestals.h"
#include "CondFormats/DataRecord/interface/EcalPedestalsRcd.h"

#include "DataFormats/EcalRecHit/interface/EcalUncalibratedRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

#include "CondFormats/EcalObjects/interface/EcalXtalGroupId.h"
#include "CondFormats/EcalObjects/interface/EcalWeightXtalGroups.h"
#include "CondFormats/DataRecord/interface/EcalWeightXtalGroupsRcd.h"

#include "CondFormats/EcalObjects/interface/EcalWeight.h"
#include "CondFormats/EcalObjects/interface/EcalWeightSet.h"
#include "CondFormats/EcalObjects/interface/EcalTBWeights.h"
#include "CondFormats/DataRecord/interface/EcalTBWeightsRcd.h"

#include "CondFormats/EcalObjects/interface/EcalMGPAGainRatio.h"
#include "CondFormats/EcalObjects/interface/EcalGainRatios.h"
#include "CondFormats/DataRecord/interface/EcalGainRatiosRcd.h"

#include "CLHEP/Matrix/Matrix.h"
#include "CLHEP/Matrix/SymMatrix.h"
#include <vector>

#define DEBUG
EcalTBWeightUncalibRecHitProducer::EcalTBWeightUncalibRecHitProducer(const edm::ParameterSet& ps)
    : testbeamEEShape(EEShape(
          false)),  // Shapes have been updated in 2018 such as to be able to fetch shape from the DB if EBShape(true)//EEShape(true) are used
      testbeamEBShape(EBShape(
          false))  // use false as argument if you would rather prefer to use Phase I hardcoded shapes (18.05.2018 K. Theofilatos)
{
  EBdigiCollection_ = ps.getParameter<edm::InputTag>("EBdigiCollection");
  EEdigiCollection_ = ps.getParameter<edm::InputTag>("EEdigiCollection");
  tdcRecInfoCollection_ = ps.getParameter<edm::InputTag>("tdcRecInfoCollection");
  EBhitCollection_ = ps.getParameter<std::string>("EBhitCollection");
  EEhitCollection_ = ps.getParameter<std::string>("EEhitCollection");
  nbTimeBin_ = ps.getParameter<int>("nbTimeBin");
  use2004OffsetConvention_ = ps.getUntrackedParameter<bool>("use2004OffsetConvention", false);
  produces<EBUncalibratedRecHitCollection>(EBhitCollection_);
  produces<EEUncalibratedRecHitCollection>(EEhitCollection_);
}

EcalTBWeightUncalibRecHitProducer::~EcalTBWeightUncalibRecHitProducer() {}

void EcalTBWeightUncalibRecHitProducer::produce(edm::Event& evt, const edm::EventSetup& es) {
  using namespace edm;

  Handle<EBDigiCollection> pEBDigis;
  const EBDigiCollection* EBdigis = nullptr;
  if (!EBdigiCollection_.label().empty() || !EBdigiCollection_.instance().empty()) {
    //     evt.getByLabel( digiProducer_, EBdigiCollection_, pEBDigis);
    evt.getByLabel(EBdigiCollection_, pEBDigis);
    if (!pEBDigis.isValid()) {
      edm::LogError("EcalUncalibRecHitError") << "Error! can't get the product " << EBdigiCollection_;
    } else {
      EBdigis = pEBDigis.product();  // get a ptr to the produc
#ifdef DEBUG
      LogDebug("EcalUncalibRecHitInfo") << "total # EBdigis: " << EBdigis->size();
#endif
    }
  }

  Handle<EEDigiCollection> pEEDigis;
  const EEDigiCollection* EEdigis = nullptr;

  if (!EEdigiCollection_.label().empty() || !EEdigiCollection_.instance().empty()) {
    //     evt.getByLabel( digiProducer_, EEdigiCollection_, pEEDigis);
    evt.getByLabel(EEdigiCollection_, pEEDigis);
    if (!pEEDigis.isValid()) {
      edm::LogError("EcalUncalibRecHitError") << "Error! can't get the product " << EEdigiCollection_;
    } else {
      EEdigis = pEEDigis.product();  // get a ptr to the produc
#ifdef DEBUG
      LogDebug("EcalUncalibRecHitInfo") << "total # EEdigis: " << EEdigis->size();
      //	 std::cout << "total # EEdigis: " << EEdigis->size() << std::endl;
#endif
    }
  }

  if (!EBdigis && !EEdigis)
    return;

  Handle<EcalTBTDCRecInfo> pRecTDC;
  const EcalTBTDCRecInfo* recTDC = nullptr;

  //     evt.getByLabel( digiProducer_, EBdigiCollection_, pEBDigis);
  evt.getByLabel(tdcRecInfoCollection_, pRecTDC);
  if (pRecTDC.isValid()) {
    recTDC = pRecTDC.product();  // get a ptr to the product
  }

  // fetch map of groups of xtals
  edm::ESHandle<EcalWeightXtalGroups> pGrp;
  es.get<EcalWeightXtalGroupsRcd>().get(pGrp);
  const EcalWeightXtalGroups* grp = pGrp.product();
  if (!grp)
    return;

  const EcalXtalGroupsMap& grpMap = grp->getMap();

  // Gain Ratios
  edm::ESHandle<EcalGainRatios> pRatio;
  es.get<EcalGainRatiosRcd>().get(pRatio);
  const EcalGainRatioMap& gainMap = pRatio.product()->getMap();  // map of gain ratios

  // fetch TB weights
#ifdef DEBUG
  LogDebug("EcalUncalibRecHitDebug") << "Fetching EcalTBWeights from DB ";
  //   std::cout  <<"Fetching EcalTBWeights from DB " ;
#endif
  edm::ESHandle<EcalTBWeights> pWgts;
  es.get<EcalTBWeightsRcd>().get(pWgts);
  const EcalTBWeights* wgts = pWgts.product();

  if (!wgts)
    return;

#ifdef DEBUG
  LogDebug("EcalUncalibRecHitDebug") << "EcalTBWeightMap.size(): " << std::setprecision(3) << wgts->getMap().size();
  //   std::cout << "EcalTBWeightMap.size(): " << std::setprecision(3) << wgts->getMap().size() ;
#endif

  // fetch the pedestals from the cond DB via EventSetup
#ifdef DEBUG
  LogDebug("EcalUncalibRecHitDebug") << "fetching pedestals....";
  //   std::cout << "fetching pedestals....";
#endif
  edm::ESHandle<EcalPedestals> pedHandle;
  es.get<EcalPedestalsRcd>().get(pedHandle);
  const EcalPedestalsMap& pedMap = pedHandle.product()->getMap();  // map of pedestals
#ifdef DEBUG
  LogDebug("EcalUncalibRecHitDebug") << "done.";
  //   std::cout << "done." ;
#endif
  // collection of reco'ed ampltudes to put in the event

  auto EBuncalibRechits = std::make_unique<EBUncalibratedRecHitCollection>();
  auto EEuncalibRechits = std::make_unique<EEUncalibratedRecHitCollection>();

  EcalPedestalsMapIterator pedIter;  // pedestal iterator

  EcalGainRatioMap::const_iterator gainIter;  // gain iterator

  EcalXtalGroupsMap::const_iterator git;  // group iterator

  EcalTBWeights::EcalTBWeightMap::const_iterator wit;  //weights iterator
  // loop over EB digis
  //Getting the TDC bin
  EcalTBWeights::EcalTDCId tdcid(int(nbTimeBin_ / 2) + 1);

  if (recTDC)
    if (recTDC->offset() == -999.) {
      edm::LogError("EcalUncalibRecHitError") << "TDC bin completely out of range. Returning";
      return;
    }

  if (EBdigis) {
    for (unsigned int idig = 0; idig < EBdigis->size(); ++idig) {
      EBDataFrame itdg = (*EBdigis)[idig];

      //     counter_++; // verbosity counter

      // find pedestals for this channel
#ifdef DEBUG
      LogDebug("EcalUncalibRecHitDebug") << "looking up pedestal for crystal: " << EBDetId(itdg.id());
#endif
      pedIter = pedMap.find(itdg.id().rawId());
      if (pedIter == pedMap.end()) {
        edm::LogError("EcalUncalibRecHitError")
            << "error!! could not find pedestals for channel: " << EBDetId(itdg.id())
            << "\n  no uncalib rechit will be made for this digi!";
        continue;
      }
      const EcalPedestals::Item& aped = (*pedIter);
      double pedVec[3];
      double pedRMSVec[3];
      pedVec[0] = aped.mean_x12;
      pedVec[1] = aped.mean_x6;
      pedVec[2] = aped.mean_x1;
      pedRMSVec[0] = aped.rms_x12;
      pedRMSVec[1] = aped.rms_x6;
      pedRMSVec[2] = aped.rms_x1;

      // find gain ratios
#ifdef DEBUG
      LogDebug("EcalUncalibRecHitDebug") << "looking up gainRatios for crystal: " << EBDetId(itdg.id());
#endif
      gainIter = gainMap.find(itdg.id().rawId());
      if (gainIter == gainMap.end()) {
        edm::LogError("EcalUncalibRecHitError")
            << "error!! could not find gain ratios for channel: " << EBDetId(itdg.id())
            << "\n  no uncalib rechit will be made for this digi!";
        continue;
      }
      const EcalMGPAGainRatio& aGain = (*gainIter);
      double gainRatios[3];
      gainRatios[0] = 1.;
      gainRatios[1] = aGain.gain12Over6();
      gainRatios[2] = aGain.gain6Over1() * aGain.gain12Over6();

      // lookup group ID for this channel
      git = grpMap.find(itdg.id().rawId());
      if (git == grpMap.end()) {
        edm::LogError("EcalUncalibRecHitError")
            << "No group id found for this crystal. something wrong with EcalWeightXtalGroups in your DB?"
            << "\n  no uncalib rechit will be made for digi with id: " << EBDetId(itdg.id());
        continue;
      }
      const EcalXtalGroupId& gid = (*git);

      //GAIN SWITCHING DETECTION ///////////////////////////////////////////////////////////////////////////////////////////////////
      double sampleGainRef = 1;
      int sampleSwitch = 999;
      for (int sample = 0; sample < itdg.size(); ++sample) {
        double gainSample = itdg.sample(sample).gainId();
        if (gainSample != sampleGainRef) {
          sampleGainRef = gainSample;
          sampleSwitch = sample;
        }
      }  //loop sample
      ///////////////////////////////////////////////////////////////////////////////////////////////////

      if (recTDC) {
        int tdcBin = 0;
        if (recTDC->offset() <= 0.)
          tdcBin = 1;
        else if (recTDC->offset() >= 1.)
          tdcBin = nbTimeBin_;
        else
          tdcBin = int(recTDC->offset() * float(nbTimeBin_)) + 1;

        if (tdcBin < 1 || tdcBin > nbTimeBin_) {
          edm::LogError("EcalUncalibRecHitError")
              << "TDC bin out of range " << tdcBin << " offset " << recTDC->offset();
          continue;
        }

        // In case gain switching happens at the sample 4 (5th sample)
        // (sample 5 (6th sample) in 2004 TDC convention) an extra
        // set of weights has to be used. This set of weights is assigned to
        // TDC values going from 25 and up.
        if (use2004OffsetConvention_ && sampleSwitch == 5)
          tdcid = EcalTBWeights::EcalTDCId(tdcBin + 25);
        else if (!use2004OffsetConvention_ && sampleSwitch == 4)
          tdcid = EcalTBWeights::EcalTDCId(tdcBin + 25);
        else
          tdcid = EcalTBWeights::EcalTDCId(tdcBin);
      }  //check TDC

      // now lookup the correct weights in the map
      wit = wgts->getMap().find(std::make_pair(gid, tdcid));
      if (wit == wgts->getMap().end()) {  // no weights found for this group ID
        edm::LogError("EcalUncalibRecHitError")
            << "No weights found for EcalGroupId: " << gid.id() << " and  EcalTDCId: " << tdcid
            << "\n  skipping digi with id: " << EBDetId(itdg.id());
        continue;
      }
      const EcalWeightSet& wset = wit->second;  // this is the EcalWeightSet

      // EcalWeightMatrix is vec<vec:double>>

#ifdef DEBUG
      LogDebug("EcalUncalibRecHitDebug") << "accessing matrices of weights...";
#endif
      const EcalWeightSet::EcalWeightMatrix& mat1 = wset.getWeightsBeforeGainSwitch();
      const EcalWeightSet::EcalWeightMatrix& mat2 = wset.getWeightsAfterGainSwitch();
      //const EcalWeightSet::EcalChi2WeightMatrix& mat3 = wset.getChi2WeightsBeforeGainSwitch();
      //const EcalWeightSet::EcalChi2WeightMatrix& mat4 = wset.getChi2WeightsAfterGainSwitch();
      const EcalWeightSet::EcalWeightMatrix* weights[2];
      weights[0] = &mat1;
      weights[1] = &mat2;
      // 	 weights.push_back(clmat1);
      // 	 weights.push_back(clmat2);
      // 	 LogDebug("EcalUncalibRecHitDebug") << "weights before switch:\n" << clmat1 ;
      // 	 LogDebug("EcalUncalibRecHitDebug") << "weights after switch:\n" << clmat2 ;

      // build CLHEP chi2  matrices
      //const EcalWeightSet::EcalChi2WeightMatrix* chi2mat[2];
      // chi2mat[0]=&mat3;
      // chi2mat[1]=&mat4;

      EcalUncalibratedRecHit aHit = EBalgo_.makeRecHit(itdg, pedVec, pedRMSVec, gainRatios, weights, testbeamEBShape);
      //EBalgo_.makeRecHit(itdg, pedVec, gainRatios, weights, chi2mat);
      EBuncalibRechits->push_back(aHit);
#ifdef DEBUG
      if (aHit.amplitude() > 0.) {
        LogDebug("EcalUncalibRecHitDebug") << "processed EBDataFrame with id: " << EBDetId(itdg.id()) << "\n"
                                           << "uncalib rechit amplitude: " << aHit.amplitude();
      }
#endif
    }
  }
  // put the collection of reconstructed hits in the event
  evt.put(std::move(EBuncalibRechits), EBhitCollection_);

  if (EEdigis) {
    for (unsigned int idig = 0; idig < EEdigis->size(); ++idig) {
      EEDataFrame itdg = (*EEdigis)[idig];

      //     counter_++; // verbosity counter

      // find pedestals for this channel
#ifdef DEBUG
      LogDebug("EcalUncalibRecHitDebug") << "looking up pedestal for crystal: " << EEDetId(itdg.id());
      //	 std::cout << "looking up pedestal for crystal: " << EEDetId(itdg.id()) ;
#endif
      pedIter = pedMap.find(itdg.id().rawId());
      if (pedIter == pedMap.end()) {
        edm::LogError("EcalUncalibRecHitError")
            << "error!! could not find pedestals for channel: " << EEDetId(itdg.id())
            << "\n  no uncalib rechit will be made for this digi!";
        continue;
      }
      const EcalPedestals::Item& aped = (*pedIter);
      double pedVec[3];
      double pedRMSVec[3];
      pedVec[0] = aped.mean_x12;
      pedVec[1] = aped.mean_x6;
      pedVec[2] = aped.mean_x1;
      pedRMSVec[0] = aped.rms_x12;
      pedRMSVec[1] = aped.rms_x6;
      pedRMSVec[2] = aped.rms_x1;

      // find gain ratios
#ifdef DEBUG
      LogDebug("EcalUncalibRecHitDebug") << "looking up gainRatios for crystal: " << EEDetId(itdg.id());
      //	 std::cout << "looking up gainRatios for crystal: " << EEDetId(itdg.id()) ;
#endif
      gainIter = gainMap.find(itdg.id().rawId());
      if (gainIter == gainMap.end()) {
        edm::LogError("EcalUncalibRecHitError")
            << "error!! could not find gain ratios for channel: " << EEDetId(itdg.id())
            << "\n  no uncalib rechit will be made for this digi!";
        continue;
      }
      const EcalMGPAGainRatio& aGain = (*gainIter);
      double gainRatios[3];
      gainRatios[0] = 1.;
      gainRatios[1] = aGain.gain12Over6();
      gainRatios[2] = aGain.gain6Over1() * aGain.gain12Over6();

      // lookup group ID for this channel
      git = grpMap.find(itdg.id().rawId());
      if (git == grpMap.end()) {
        edm::LogError("EcalUncalibRecHitError")
            << "No group id found for this crystal. something wrong with EcalWeightXtalGroups in your DB?"
            << "\n  no uncalib rechit will be made for digi with id: " << EEDetId(itdg.id());
        continue;
      }
      const EcalXtalGroupId& gid = (*git);

      //GAIN SWITCHING DETECTION ///////////////////////////////////////////////////////////////////////////////////////////////////
      double sampleGainRef = 1;
      int sampleSwitch = 999;
      for (int sample = 0; sample < itdg.size(); ++sample) {
        double gainSample = itdg.sample(sample).gainId();
        if (gainSample != sampleGainRef) {
          sampleGainRef = gainSample;
          sampleSwitch = sample;
        }
      }  //loop sample
      ///////////////////////////////////////////////////////////////////////////////////////////////////

      if (recTDC) {
        int tdcBin = 0;
        if (recTDC->offset() <= 0.)
          tdcBin = 1;
        else if (recTDC->offset() >= 1.)
          tdcBin = nbTimeBin_;
        else
          tdcBin = int(recTDC->offset() * float(nbTimeBin_)) + 1;

        if (tdcBin < 1 || tdcBin > nbTimeBin_) {
          edm::LogError("EcalUncalibRecHitError")
              << "TDC bin out of range " << tdcBin << " offset " << recTDC->offset();
          continue;
        }

        // In case gain switching happens at the sample 4 (5th sample)
        // (sample 5 (6th sample) in 2004 TDC convention) an extra
        // set of weights has to be used. This set of weights is assigned to
        // TDC values going from 25 and up.
        if (use2004OffsetConvention_ && sampleSwitch == 5)
          tdcid = EcalTBWeights::EcalTDCId(tdcBin + 25);
        else if (!use2004OffsetConvention_ && sampleSwitch == 4)
          tdcid = EcalTBWeights::EcalTDCId(tdcBin + 25);
        else
          tdcid = EcalTBWeights::EcalTDCId(tdcBin);
      }  //check TDC

      // now lookup the correct weights in the map
      wit = wgts->getMap().find(std::make_pair(gid, tdcid));
      if (wit == wgts->getMap().end()) {  // no weights found for this group ID
        edm::LogError("EcalUncalibRecHitError")
            << "No weights found for EcalGroupId: " << gid.id() << " and  EcalTDCId: " << tdcid
            << "\n  skipping digi with id: " << EEDetId(itdg.id());
        continue;
      }
      const EcalWeightSet& wset = wit->second;  // this is the EcalWeightSet

      // EcalWeightMatrix is vec<vec:double>>

#ifdef DEBUG
      LogDebug("EcalUncalibRecHitDebug") << "accessing matrices of weights...";
      //	 std::cout << "accessing matrices of weights...";
#endif
      const EcalWeightSet::EcalWeightMatrix& mat1 = wset.getWeightsBeforeGainSwitch();
      const EcalWeightSet::EcalWeightMatrix& mat2 = wset.getWeightsAfterGainSwitch();
      //const EcalWeightSet::EcalChi2WeightMatrix& mat3 = wset.getChi2WeightsBeforeGainSwitch();
      //const EcalWeightSet::EcalChi2WeightMatrix& mat4 = wset.getChi2WeightsAfterGainSwitch();
      const EcalWeightSet::EcalWeightMatrix* weights[2];
      weights[0] = &mat1;
      weights[1] = &mat2;
      // 	 weights.push_back(clmat1);
      // 	 weights.push_back(clmat2);
      // 	 LogDebug("EcalUncalibRecHitDebug") << "weights before switch:\n" << clmat1 ;
      // 	 LogDebug("EcalUncalibRecHitDebug") << "weights after switch:\n" << clmat2 ;

      // build CLHEP chi2  matrices
      //const EcalWeightSet::EcalChi2WeightMatrix* chi2mat[2];
      //chi2mat[0]=&mat3;
      //chi2mat[1]=&mat4;

      EcalUncalibratedRecHit aHit = EEalgo_.makeRecHit(itdg, pedVec, pedRMSVec, gainRatios, weights, testbeamEEShape);
      //EEalgo_.makeRecHit(itdg, pedVec, gainRatios, weights, chi2mat);
      EEuncalibRechits->push_back(aHit);
#ifdef DEBUG
      if (aHit.amplitude() > 0.) {
        LogDebug("EcalUncalibRecHitDebug") << "processed EEDataFrame with id: " << EEDetId(itdg.id()) << "\n"
                                           << "uncalib rechit amplitude: " << aHit.amplitude()

            // 	   std::cout << "processed EEDataFrame with id: "
            //                   << EEDetId(itdg.id()) << "\n"
            //                   << "uncalib rechit amplitude: " << aHit.amplitude() << std::endl;
            ;
      }
#endif
    }
  }
  // put the collection of reconstructed hits in the event
  evt.put(std::move(EEuncalibRechits), EEhitCollection_);
}

// HepMatrix
// EcalTBWeightUncalibRecHitProducer::makeMatrixFromVectors(const std::vector< std::vector<EcalWeight> >& vecvec) {
//   int nrow = vecvec.size();
//   int ncol = (vecvec[0]).size();
//   HepMatrix clmat(nrow,ncol);
//   //LogDebug("EcalUncalibRecHitDebug") << "created HepMatrix(" << nrow << "," << ncol << ")" ;
//   for(int irow=0;irow<nrow;++irow) {
//     for(int icol=0;icol<ncol;++icol) {
//         clmat[irow][icol] = ((vecvec[irow])[icol]).value();
//     }
//   }
//   return clmat;
// }

// HepMatrix
// EcalTBWeightUncalibRecHitProducer::makeDummySymMatrix(int size)
// {
//   HepMatrix clmat(10,10);
//   //LogDebug("EcalUncalibRecHitDebug") << "created HepMatrix(" << nrow << "," << ncol << ")" ;
//   for(int irow=0; irow<size; ++irow) {
//     for(int icol=0 ; icol<size; ++icol) {
//       clmat[irow][icol] = irow+icol;
//     }
//   }
//   return clmat;
// }
