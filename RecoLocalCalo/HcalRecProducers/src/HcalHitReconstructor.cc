#include "HcalHitReconstructor.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/Common/interface/EDCollection.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "CalibFormats/HcalObjects/interface/HcalCoderDb.h"
#include "CalibFormats/HcalObjects/interface/HcalDbRecord.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalSeverityLevelComputer.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalSeverityLevelComputerRcd.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalDbASCIIIO.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "Geometry/Records/interface/HcalRecNumberingRecord.h"
#include "CondFormats/DataRecord/interface/HcalFrontEndMapRcd.h"
#include "CondFormats/DataRecord/interface/HcalOOTPileupCorrectionRcd.h"
#include "CondFormats/DataRecord/interface/HcalOOTPileupCompatibilityRcd.h"
#include "CondFormats/DataRecord/interface/HBHENegativeEFilterRcd.h"
#include "CondFormats/HcalObjects/interface/HcalFrontEndMap.h"
#include "CondFormats/HcalObjects/interface/OOTPileupCorrectionColl.h"
#include "CondFormats/HcalObjects/interface/OOTPileupCorrData.h"
#include <iostream>
#include <fstream>

/*  Hcal Hit reconstructor allows for CaloRecHits with status words */

HcalHitReconstructor::HcalHitReconstructor(edm::ParameterSet const& conf)
    : reco_(conf.getParameter<bool>("correctForTimeslew"),
            conf.getParameter<bool>("correctForPhaseContainment"),
            conf.getParameter<double>("correctionPhaseNS"),
            consumesCollector()),
      det_(DetId::Hcal),
      inputLabel_(conf.getParameter<edm::InputTag>("digiLabel")),
      correctTiming_(conf.getParameter<bool>("correctTiming")),
      setNoiseFlags_(conf.getParameter<bool>("setNoiseFlags")),
      setHSCPFlags_(conf.getParameter<bool>("setHSCPFlags")),
      setSaturationFlags_(conf.getParameter<bool>("setSaturationFlags")),
      setTimingTrustFlags_(conf.getParameter<bool>("setTimingTrustFlags")),
      setPulseShapeFlags_(conf.getParameter<bool>("setPulseShapeFlags")),
      setNegativeFlags_(false),
      dropZSmarkedPassed_(conf.getParameter<bool>("dropZSmarkedPassed")),
      firstAuxTS_(conf.getParameter<int>("firstAuxTS")),
      firstSample_(conf.getParameter<int>("firstSample")),
      samplesToAdd_(conf.getParameter<int>("samplesToAdd")),
      tsFromDB_(conf.getParameter<bool>("tsFromDB")),
      useLeakCorrection_(conf.getParameter<bool>("useLeakCorrection")),
      dataOOTCorrectionName_(""),
      dataOOTCorrectionCategory_("Data"),
      mcOOTCorrectionName_(""),
      mcOOTCorrectionCategory_("MC"),
      setPileupCorrection_(nullptr) {
  // register for data access
  tok_ho_ = consumes<HODigiCollection>(inputLabel_);
  tok_hf_ = consumes<HFDigiCollection>(inputLabel_);
  tok_calib_ = consumes<HcalCalibDigiCollection>(inputLabel_);

  std::string subd = conf.getParameter<std::string>("Subdetector");
  //Set all FlagSetters to 0
  /* Important to do this!  Otherwise, if the setters are turned off,
     the "if (XSetter_) delete XSetter_;" commands can crash
  */

  recoParamsFromDB_ = conf.getParameter<bool>("recoParamsFromDB");
  //  recoParamsFromDB_ = false ; //  trun off for now.

  // std::cout<<"  HcalHitReconstructor   recoParamsFromDB_ "<<recoParamsFromDB_<<std::endl;

  if (conf.existsAs<bool>("setNegativeFlags"))
    setNegativeFlags_ = conf.getParameter<bool>("setNegativeFlags");

  hfdigibit_ = nullptr;

  hfS9S1_ = nullptr;
  hfS8S1_ = nullptr;
  hfPET_ = nullptr;
  saturationFlagSetter_ = nullptr;
  HFTimingTrustFlagSetter_ = nullptr;
  digiTimeFromDB_ = false;  // only need for HF

  if (setSaturationFlags_) {
    const edm::ParameterSet& pssat = conf.getParameter<edm::ParameterSet>("saturationParameters");
    saturationFlagSetter_ = new HcalADCSaturationFlag(pssat.getParameter<int>("maxADCvalue"));
  }

  if (!strcasecmp(subd.c_str(), "HO")) {
    subdet_ = HcalOuter;
    // setPileupCorrection_ = &HcalSimpleRecAlgo::setHOPileupCorrection;
    setPileupCorrection_ = nullptr;
    produces<HORecHitCollection>();
  } else if (!strcasecmp(subd.c_str(), "HF")) {
    subdet_ = HcalForward;
    // setPileupCorrection_ = &HcalSimpleRecAlgo::setHFPileupCorrection;
    setPileupCorrection_ = nullptr;
    digiTimeFromDB_ = conf.getParameter<bool>("digiTimeFromDB");

    if (setTimingTrustFlags_) {
      const edm::ParameterSet& pstrust = conf.getParameter<edm::ParameterSet>("hfTimingTrustParameters");
      HFTimingTrustFlagSetter_ = new HFTimingTrustFlag(pstrust.getParameter<int>("hfTimingTrustLevel1"),
                                                       pstrust.getParameter<int>("hfTimingTrustLevel2"));
    }

    if (setNoiseFlags_) {
      const edm::ParameterSet& psdigi = conf.getParameter<edm::ParameterSet>("digistat");
      const edm::ParameterSet& psTimeWin = conf.getParameter<edm::ParameterSet>("HFInWindowStat");
      hfdigibit_ = new HcalHFStatusBitFromDigis(psdigi, psTimeWin);

      const edm::ParameterSet& psS9S1 = conf.getParameter<edm::ParameterSet>("S9S1stat");
      hfS9S1_ = new HcalHF_S9S1algorithm(psS9S1.getParameter<std::vector<double> >("short_optimumSlope"),
                                         psS9S1.getParameter<std::vector<double> >("shortEnergyParams"),
                                         psS9S1.getParameter<std::vector<double> >("shortETParams"),
                                         psS9S1.getParameter<std::vector<double> >("long_optimumSlope"),
                                         psS9S1.getParameter<std::vector<double> >("longEnergyParams"),
                                         psS9S1.getParameter<std::vector<double> >("longETParams"),
                                         psS9S1.getParameter<int>("HcalAcceptSeverityLevel"),
                                         psS9S1.getParameter<bool>("isS8S1"));

      const edm::ParameterSet& psS8S1 = conf.getParameter<edm::ParameterSet>("S8S1stat");
      hfS8S1_ = new HcalHF_S9S1algorithm(psS8S1.getParameter<std::vector<double> >("short_optimumSlope"),
                                         psS8S1.getParameter<std::vector<double> >("shortEnergyParams"),
                                         psS8S1.getParameter<std::vector<double> >("shortETParams"),
                                         psS8S1.getParameter<std::vector<double> >("long_optimumSlope"),
                                         psS8S1.getParameter<std::vector<double> >("longEnergyParams"),
                                         psS8S1.getParameter<std::vector<double> >("longETParams"),
                                         psS8S1.getParameter<int>("HcalAcceptSeverityLevel"),
                                         psS8S1.getParameter<bool>("isS8S1"));

      const edm::ParameterSet& psPET = conf.getParameter<edm::ParameterSet>("PETstat");
      hfPET_ = new HcalHF_PETalgorithm(psPET.getParameter<std::vector<double> >("short_R"),
                                       psPET.getParameter<std::vector<double> >("shortEnergyParams"),
                                       psPET.getParameter<std::vector<double> >("shortETParams"),
                                       psPET.getParameter<std::vector<double> >("long_R"),
                                       psPET.getParameter<std::vector<double> >("longEnergyParams"),
                                       psPET.getParameter<std::vector<double> >("longETParams"),
                                       psPET.getParameter<int>("HcalAcceptSeverityLevel"),
                                       psPET.getParameter<std::vector<double> >("short_R_29"),
                                       psPET.getParameter<std::vector<double> >("long_R_29"));
    }
    produces<HFRecHitCollection>();
  } else if (!strcasecmp(subd.c_str(), "ZDC")) {
    det_ = DetId::Calo;
    subdet_ = HcalZDCDetId::SubdetectorId;
    produces<ZDCRecHitCollection>();
  } else if (!strcasecmp(subd.c_str(), "CALIB")) {
    subdet_ = HcalOther;
    subdetOther_ = HcalCalibration;
    produces<HcalCalibRecHitCollection>();
  } else {
    edm::LogWarning("Configuration") << "HcalHitReconstructor is not associated with a specific subdetector!"
                                     << std::endl;
  }

  // If no valid OOT pileup correction name specified,
  // disable the correction
  if (conf.existsAs<std::string>("dataOOTCorrectionName"))
    dataOOTCorrectionName_ = conf.getParameter<std::string>("dataOOTCorrectionName");
  if (conf.existsAs<std::string>("dataOOTCorrectionCategory"))
    dataOOTCorrectionCategory_ = conf.getParameter<std::string>("dataOOTCorrectionCategory");
  if (conf.existsAs<std::string>("mcOOTCorrectionName"))
    mcOOTCorrectionName_ = conf.getParameter<std::string>("mcOOTCorrectionName");
  if (conf.existsAs<std::string>("mcOOTCorrectionCategory"))
    mcOOTCorrectionCategory_ = conf.getParameter<std::string>("mcOOTCorrectionCategory");
  if (dataOOTCorrectionName_.empty() && mcOOTCorrectionName_.empty())
    setPileupCorrection_ = nullptr;

  // ES tokens
  htopoToken_ = esConsumes<HcalTopology, HcalRecNumberingRecord, edm::Transition::BeginRun>();
  if (tsFromDB_ || recoParamsFromDB_)
    paramsToken_ = esConsumes<HcalRecoParams, HcalRecoParamsRcd, edm::Transition::BeginRun>();
  if (digiTimeFromDB_)
    digiTimeToken_ = esConsumes<HcalFlagHFDigiTimeParams, HcalFlagHFDigiTimeParamsRcd, edm::Transition::BeginRun>();
  conditionsToken_ = esConsumes<HcalDbService, HcalDbRecord>();
  qualToken_ = esConsumes<HcalChannelQuality, HcalChannelQualityRcd>(edm::ESInputTag("", "withTopo"));
  sevToken_ = esConsumes<HcalSeverityLevelComputer, HcalSeverityLevelComputerRcd>();
}

HcalHitReconstructor::~HcalHitReconstructor() {
  delete hfdigibit_;

  delete hfS9S1_;
  delete hfS8S1_;
  delete hfPET_;
  delete saturationFlagSetter_;
  delete HFTimingTrustFlagSetter_;
}

void HcalHitReconstructor::beginRun(edm::Run const& r, edm::EventSetup const& es) {
  const HcalTopology& htopo = es.getData(htopoToken_);

  if (tsFromDB_ || recoParamsFromDB_) {
    const HcalRecoParams& p = es.getData(paramsToken_);
    paramTS_ = std::make_unique<HcalRecoParams>(p);
    paramTS_->setTopo(&htopo);

    // std::cout<<" skdump in HcalHitReconstructor::beginRun   dupm RecoParams "<<std::endl;
    // std::ofstream skfile("skdumpRecoParamsNewFormat.txt");
    // HcalDbASCIIIO::dumpObject(skfile, (*paramTS_) );
  }

  if (digiTimeFromDB_) {
    const HcalFlagHFDigiTimeParams& p = es.getData(digiTimeToken_);
    hFDigiTimeParams_ = std::make_unique<HcalFlagHFDigiTimeParams>(p);
    hFDigiTimeParams_->setTopo(&htopo);
  }

  reco_.beginRun(es);
}

void HcalHitReconstructor::endRun(edm::Run const& r, edm::EventSetup const& es) { reco_.endRun(); }

void HcalHitReconstructor::produce(edm::Event& e, const edm::EventSetup& eventSetup) {
  // get conditions
  const HcalDbService* conditions = &eventSetup.getData(conditionsToken_);
  const HcalChannelQuality* myqual = &eventSetup.getData(qualToken_);
  const HcalSeverityLevelComputer* mySeverity = &eventSetup.getData(sevToken_);

  if (useLeakCorrection_)
    reco_.setLeakCorrection();

  // GET THE BEAM CROSSING INFO HERE, WHEN WE UNDERSTAND HOW THINGS WORK.
  // Then, call "setBXInfo" method of the reco_ object.
  // Also remember to call SetBXInfo in the negative energy flag setter.

  if (det_ == DetId::Hcal) {
    //  HO ------------------------------------------------------------------
    if (subdet_ == HcalOuter) {
      edm::Handle<HODigiCollection> digi;
      e.getByToken(tok_ho_, digi);

      // create empty output
      auto rec = std::make_unique<HORecHitCollection>();
      rec->reserve(digi->size());
      // run the algorithm
      HODigiCollection::const_iterator i;

      // Vote on majority TS0 CapId
      int favorite_capid = 0;
      if (correctTiming_) {
        long capid_votes[4] = {0, 0, 0, 0};
        for (i = digi->begin(); i != digi->end(); i++) {
          capid_votes[(*i)[0].capid()]++;
        }
        for (int k = 0; k < 4; k++)
          if (capid_votes[k] > capid_votes[favorite_capid])
            favorite_capid = k;
      }

      for (i = digi->begin(); i != digi->end(); i++) {
        HcalDetId cell = i->id();
        DetId detcell = (DetId)cell;
        // firstSample & samplesToAdd
        if (tsFromDB_ || recoParamsFromDB_) {
          const HcalRecoParam* param_ts = paramTS_->getValues(detcell.rawId());
          if (tsFromDB_) {
            firstSample_ = param_ts->firstSample();
            samplesToAdd_ = param_ts->samplesToAdd();
          }
          if (recoParamsFromDB_) {
            bool correctForTimeslew = param_ts->correctForTimeslew();
            bool correctForPhaseContainment = param_ts->correctForPhaseContainment();
            float phaseNS = param_ts->correctionPhaseNS();
            useLeakCorrection_ = param_ts->useLeakCorrection();
            correctTiming_ = param_ts->correctTiming();
            firstAuxTS_ = param_ts->firstAuxTS();
            int pileupCleaningID = param_ts->pileupCleaningID();
            reco_.setRecoParams(
                correctForTimeslew, correctForPhaseContainment, useLeakCorrection_, pileupCleaningID, phaseNS);
          }
        }

        int first = firstSample_;
        int toadd = samplesToAdd_;

        if (first >= i->size() || first < 0)
          edm::LogWarning("Configuration")
              << "HcalHitReconstructor: illegal firstSample" << first << "  in subdet " << subdet_ << std::endl;

        // check on cells to be ignored and dropped: (rof,20.Feb.09)
        const HcalChannelStatus* mydigistatus = myqual->getValues(detcell.rawId());
        if (mySeverity->dropChannel(mydigistatus->getValue()))
          continue;
        if (dropZSmarkedPassed_)
          if (i->zsMarkAndPass())
            continue;

        const HcalCalibrations& calibrations = conditions->getHcalCalibrations(cell);
        const HcalQIECoder* channelCoder = conditions->getHcalCoder(cell);
        const HcalQIEShape* shape = conditions->getHcalShape(channelCoder);
        HcalCoderDb coder(*channelCoder, *shape);

        rec->push_back(reco_.reconstruct(*i, first, toadd, coder, calibrations));

        // Set auxiliary flag
        int auxflag = 0;
        int fTS = firstAuxTS_;
        if (fTS < 0)
          fTS = 0;  //silly protection against negative time slice values
        for (int xx = fTS; xx < fTS + 4 && xx < i->size(); ++xx)
          auxflag += (i->sample(xx).adc())
                     << (7 *
                         (xx - fTS));  // store the time slices in the first 28 bits of aux, a set of 4 7-bit adc values
        // bits 28 and 29 are reserved for capid of the first time slice saved in aux
        auxflag += ((i->sample(fTS).capid()) << 28);
        (rec->back()).setAux(auxflag);
        // (rec->back()).setFlags(0);  Don't want to do this because the algorithm
        //                             can already set some flags
        // Fill Presample ADC flag
        if (fTS > 0)
          (rec->back()).setFlagField((i->sample(fTS - 1).adc()), HcalCaloFlagLabels::PresampleADC, 7);

        if (setSaturationFlags_)
          saturationFlagSetter_->setSaturationFlag(rec->back(), *i);
        if (correctTiming_)
          HcalTimingCorrector::Correct(rec->back(), *i, favorite_capid);
      }
      // return result
      e.put(std::move(rec));

      // HF -------------------------------------------------------------------
    } else if (subdet_ == HcalForward) {
      edm::Handle<HFDigiCollection> digi;
      e.getByToken(tok_hf_, digi);

      ///////////////////////////////////////////////////////////////// HF
      // create empty output
      auto rec = std::make_unique<HFRecHitCollection>();
      rec->reserve(digi->size());
      // run the algorithm
      HFDigiCollection::const_iterator i;

      // Vote on majority TS0 CapId
      int favorite_capid = 0;
      if (correctTiming_) {
        long capid_votes[4] = {0, 0, 0, 0};
        for (i = digi->begin(); i != digi->end(); i++) {
          capid_votes[(*i)[0].capid()]++;
        }
        for (int k = 0; k < 4; k++)
          if (capid_votes[k] > capid_votes[favorite_capid])
            favorite_capid = k;
      }

      for (i = digi->begin(); i != digi->end(); i++) {
        HcalDetId cell = i->id();
        DetId detcell = (DetId)cell;

        if (tsFromDB_ || recoParamsFromDB_) {
          const HcalRecoParam* param_ts = paramTS_->getValues(detcell.rawId());
          if (tsFromDB_) {
            firstSample_ = param_ts->firstSample();
            samplesToAdd_ = param_ts->samplesToAdd();
          }
          if (recoParamsFromDB_) {
            bool correctForTimeslew = param_ts->correctForTimeslew();
            bool correctForPhaseContainment = param_ts->correctForPhaseContainment();
            float phaseNS = param_ts->correctionPhaseNS();
            useLeakCorrection_ = param_ts->useLeakCorrection();
            correctTiming_ = param_ts->correctTiming();
            firstAuxTS_ = param_ts->firstAuxTS();
            int pileupCleaningID = param_ts->pileupCleaningID();
            reco_.setRecoParams(
                correctForTimeslew, correctForPhaseContainment, useLeakCorrection_, pileupCleaningID, phaseNS);
          }
        }

        int first = firstSample_;
        int toadd = samplesToAdd_;

        if (first >= i->size() || first < 0)
          edm::LogWarning("Configuration")
              << "HcalHitReconstructor: illegal firstSample" << first << "  in subdet " << subdet_ << std::endl;

        // check on cells to be ignored and dropped: (rof,20.Feb.09)
        const HcalChannelStatus* mydigistatus = myqual->getValues(detcell.rawId());
        if (mySeverity->dropChannel(mydigistatus->getValue()))
          continue;
        if (dropZSmarkedPassed_)
          if (i->zsMarkAndPass())
            continue;

        const HcalCalibrations& calibrations = conditions->getHcalCalibrations(cell);
        const HcalQIECoder* channelCoder = conditions->getHcalCoder(cell);
        const HcalQIEShape* shape = conditions->getHcalShape(channelCoder);
        HcalCoderDb coder(*channelCoder, *shape);

        // Set HFDigiTime flag values from digiTimeFromDB_
        if (digiTimeFromDB_ && hfdigibit_) {
          const HcalFlagHFDigiTimeParam* hfDTparam = hFDigiTimeParams_->getValues(detcell.rawId());
          hfdigibit_->resetParamsFromDB(hfDTparam->HFdigiflagFirstSample(),
                                        hfDTparam->HFdigiflagSamplesToAdd(),
                                        hfDTparam->HFdigiflagExpectedPeak(),
                                        hfDTparam->HFdigiflagMinEThreshold(),
                                        hfDTparam->HFdigiflagCoefficients());
        }

        //std::cout << "TOADDHF " << toadd << " " << first << " " << std::endl;
        rec->push_back(reco_.reconstruct(*i, first, toadd, coder, calibrations));

        // Set auxiliary flag
        int auxflag = 0;
        int fTS = firstAuxTS_;
        if (fTS < 0)
          fTS = 0;  // silly protection against negative time slice values
        for (int xx = fTS; xx < fTS + 4 && xx < i->size(); ++xx)
          auxflag += (i->sample(xx).adc())
                     << (7 *
                         (xx - fTS));  // store the time slices in the first 28 bits of aux, a set of 4 7-bit adc values
        // bits 28 and 29 are reserved for capid of the first time slice saved in aux
        auxflag += ((i->sample(fTS).capid()) << 28);
        (rec->back()).setAux(auxflag);

        // (rec->back()).setFlags(0);  Don't want to do this because the algorithm
        //                             can already set some flags

        // Fill Presample ADC flag
        if (fTS > 0)
          (rec->back()).setFlagField((i->sample(fTS - 1).adc()), HcalCaloFlagLabels::PresampleADC, 7);

        // This calls the code for setting the HF noise bit determined from digi shape
        if (setNoiseFlags_)
          hfdigibit_->hfSetFlagFromDigi(rec->back(), *i, coder, calibrations);
        if (setSaturationFlags_)
          saturationFlagSetter_->setSaturationFlag(rec->back(), *i);
        if (setTimingTrustFlags_)
          HFTimingTrustFlagSetter_->setHFTimingTrustFlag(rec->back(), *i);
        if (correctTiming_)
          HcalTimingCorrector::Correct(rec->back(), *i, favorite_capid);
      }  // for (i=digi->begin(); i!=digi->end(); i++) -- loop on all HF digis

      // The following flags require the full set of rechits
      // These need to be set consecutively, so an energy check should be the first
      // test performed on these hits (to minimize the loop time)
      if (setNoiseFlags_) {
        // Step 1:  Set PET flag  (short fibers of |ieta|==29)
        // Neighbor/partner channels that are flagged by Pulse Shape algorithm (HFDigiTime)
        // won't be considered in these calculations
        for (HFRecHitCollection::iterator i = rec->begin(); i != rec->end(); ++i) {
          int depth = i->id().depth();
          int ieta = i->id().ieta();
          // Short fibers and all channels at |ieta|=29 use PET settings in Algo 3
          if (depth == 2 || abs(ieta) == 29)
            hfPET_->HFSetFlagFromPET(*i, *rec, myqual, mySeverity);
        }

        // Step 2:  Set S8S1 flag (short fibers or |ieta|==29)
        for (HFRecHitCollection::iterator i = rec->begin(); i != rec->end(); ++i) {
          int depth = i->id().depth();
          int ieta = i->id().ieta();
          // Short fibers and all channels at |ieta|=29 use PET settings in Algo 3
          if (depth == 2 || abs(ieta) == 29)
            hfS8S1_->HFSetFlagFromS9S1(*i, *rec, myqual, mySeverity);
        }

        // Set 3:  Set S9S1 flag (long fibers)
        for (HFRecHitCollection::iterator i = rec->begin(); i != rec->end(); ++i) {
          int depth = i->id().depth();
          int ieta = i->id().ieta();
          // Short fibers and all channels at |ieta|=29 use PET settings in Algo 3
          if (depth == 1 && abs(ieta) != 29)
            hfS9S1_->HFSetFlagFromS9S1(*i, *rec, myqual, mySeverity);
        }
      }

      // return result
      e.put(std::move(rec));
    } else if (subdet_ == HcalOther && subdetOther_ == HcalCalibration) {
      edm::Handle<HcalCalibDigiCollection> digi;
      e.getByToken(tok_calib_, digi);

      // create empty output
      auto rec = std::make_unique<HcalCalibRecHitCollection>();
      rec->reserve(digi->size());
      // run the algorithm
      int first = firstSample_;
      int toadd = samplesToAdd_;

      HcalCalibDigiCollection::const_iterator i;
      for (i = digi->begin(); i != digi->end(); i++) {
        HcalCalibDetId cell = i->id();
        //	HcalDetId cellh = i->id();
        DetId detcell = (DetId)cell;
        // check on cells to be ignored and dropped: (rof,20.Feb.09)
        const HcalChannelStatus* mydigistatus = myqual->getValues(detcell.rawId());
        if (mySeverity->dropChannel(mydigistatus->getValue()))
          continue;
        if (dropZSmarkedPassed_)
          if (i->zsMarkAndPass())
            continue;

        const HcalCalibrations& calibrations = conditions->getHcalCalibrations(cell);
        const HcalQIECoder* channelCoder = conditions->getHcalCoder(cell);
        const HcalQIEShape* shape = conditions->getHcalShape(channelCoder);
        HcalCoderDb coder(*channelCoder, *shape);

        // firstSample & samplesToAdd
        if (tsFromDB_) {
          const HcalRecoParam* param_ts = paramTS_->getValues(detcell.rawId());
          first = param_ts->firstSample();
          toadd = param_ts->samplesToAdd();
        }
        rec->push_back(reco_.reconstruct(*i, first, toadd, coder, calibrations));

        /*
	  // Flag setting not available for calibration rechits
	// Set auxiliary flag
	int auxflag=0;
        int fTS = firstAuxTS_;
	for (int xx=fTS; xx<fTS+4 && xx<i->size();++xx)
	  auxflag+=(i->sample(xx).adc())<<(7*(xx-fTS)); // store the time slices in the first 28 bits of aux, a set of 4 7-bit adc values
	// bits 28 and 29 are reserved for capid of the first time slice saved in aux
	auxflag+=((i->sample(fTS).capid())<<28);
	(rec->back()).setAux(auxflag);

	(rec->back()).setFlags(0); // Not yet implemented for HcalCalibRecHit
	*/
      }
      // return result
      e.put(std::move(rec));
    }
  }
  //DL  delete myqual;
}  // void HcalHitReconstructor::produce(...)
