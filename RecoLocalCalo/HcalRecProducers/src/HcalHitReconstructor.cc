#include "HcalHitReconstructor.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/Common/interface/EDCollection.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "CalibFormats/HcalObjects/interface/HcalCoderDb.h"
#include "CalibFormats/HcalObjects/interface/HcalCalibrations.h"
#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "CalibFormats/HcalObjects/interface/HcalDbRecord.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalSeverityLevelComputer.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalSeverityLevelComputerRcd.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalDbASCIIIO.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include <iostream>
#include <fstream>


/*  Hcal Hit reconstructor allows for CaloRecHits with status words */

HcalHitReconstructor::HcalHitReconstructor(edm::ParameterSet const& conf):
  reco_(conf.getParameter<bool>("correctForTimeslew"),
	conf.getParameter<bool>("correctForPhaseContainment"),
	conf.getParameter<double>("correctionPhaseNS")),
  det_(DetId::Hcal),
  inputLabel_(conf.getParameter<edm::InputTag>("digiLabel")),
  correctTiming_(conf.getParameter<bool>("correctTiming")),
  setNoiseFlags_(conf.getParameter<bool>("setNoiseFlags")),
  setHSCPFlags_(conf.getParameter<bool>("setHSCPFlags")),
  setSaturationFlags_(conf.getParameter<bool>("setSaturationFlags")),
  setTimingTrustFlags_(conf.getParameter<bool>("setTimingTrustFlags")),
  setPulseShapeFlags_(conf.getParameter<bool>("setPulseShapeFlags")),
  dropZSmarkedPassed_(conf.getParameter<bool>("dropZSmarkedPassed")),
  firstAuxTS_(conf.getParameter<int>("firstAuxTS")),
  firstSample_(conf.getParameter<int>("firstSample")),
  samplesToAdd_(conf.getParameter<int>("samplesToAdd")),
  tsFromDB_(conf.getParameter<bool>("tsFromDB")),
  useLeakCorrection_( conf.getParameter<bool>("useLeakCorrection")),
  paramTS(0),
  theTopology(0)
{

  std::string subd=conf.getParameter<std::string>("Subdetector");
  //Set all FlagSetters to 0
  /* Important to do this!  Otherwise, if the setters are turned off,
     the "if (XSetter_) delete XSetter_;" commands can crash
  */

  recoParamsFromDB_ = conf.getParameter<bool>("recoParamsFromDB");
  //  recoParamsFromDB_ = false ; //  trun off for now.

  // std::cout<<"  HcalHitReconstructor   recoParamsFromDB_ "<<recoParamsFromDB_<<std::endl;

  hbheFlagSetter_             = 0;
  hbheHSCPFlagSetter_         = 0;
  hbhePulseShapeFlagSetter_   = 0;
  hbheTimingShapedFlagSetter_ = 0;
  hfdigibit_                  = 0;

  hfS9S1_                     = 0;
  hfS8S1_                     = 0;
  hfPET_                      = 0;
  saturationFlagSetter_       = 0;
  HFTimingTrustFlagSetter_    = 0;
  digiTimeFromDB_             = false; // only need for HF
  
  if (setSaturationFlags_)
    {
      const edm::ParameterSet& pssat      = conf.getParameter<edm::ParameterSet>("saturationParameters");
      saturationFlagSetter_ = new HcalADCSaturationFlag(pssat.getParameter<int>("maxADCvalue"));
    }

  if (!strcasecmp(subd.c_str(),"HBHE")) {
    subdet_=HcalBarrel;
    bool timingShapedCutsFlags = conf.getParameter<bool>("setTimingShapedCutsFlags");
    if (timingShapedCutsFlags)
      {
	const edm::ParameterSet& psTshaped = conf.getParameter<edm::ParameterSet>("timingshapedcutsParameters");
	hbheTimingShapedFlagSetter_ = new HBHETimingShapedFlagSetter(psTshaped.getParameter<std::vector<double> >("tfilterEnvelope"),
								     psTshaped.getParameter<bool>("ignorelowest"),
								     psTshaped.getParameter<bool>("ignorehighest"),
								     psTshaped.getParameter<double>("win_offset"),
								     psTshaped.getParameter<double>("win_gain"));
      }
      
    if (setNoiseFlags_)
      {
	const edm::ParameterSet& psdigi    =conf.getParameter<edm::ParameterSet>("flagParameters");
	hbheFlagSetter_=new HBHEStatusBitSetter(psdigi.getParameter<double>("nominalPedestal"),
						psdigi.getParameter<double>("hitEnergyMinimum"),
						psdigi.getParameter<int>("hitMultiplicityThreshold"),
						psdigi.getParameter<std::vector<edm::ParameterSet> >("pulseShapeParameterSets")
	 );
      } // if (setNoiseFlags_)
    if (setHSCPFlags_)
      {
	const edm::ParameterSet& psHSCP = conf.getParameter<edm::ParameterSet>("hscpParameters");
	hbheHSCPFlagSetter_ = new HBHETimeProfileStatusBitSetter(psHSCP.getParameter<double>("r1Min"),
								 psHSCP.getParameter<double>("r1Max"),
								 psHSCP.getParameter<double>("r2Min"),
								 psHSCP.getParameter<double>("r2Max"),
								 psHSCP.getParameter<double>("fracLeaderMin"),
								 psHSCP.getParameter<double>("fracLeaderMax"),
								 psHSCP.getParameter<double>("slopeMin"),
								 psHSCP.getParameter<double>("slopeMax"),
								 psHSCP.getParameter<double>("outerMin"),
								 psHSCP.getParameter<double>("outerMax"),
								 psHSCP.getParameter<double>("TimingEnergyThreshold"));
      } // if (setHSCPFlags_) 
    if (setPulseShapeFlags_)
      {
        const edm::ParameterSet &psPulseShape = conf.getParameter<edm::ParameterSet>("pulseShapeParameters");
        hbhePulseShapeFlagSetter_ = new HBHEPulseShapeFlagSetter(
								 psPulseShape.getParameter<double>("MinimumChargeThreshold"),
								 psPulseShape.getParameter<double>("TS4TS5ChargeThreshold"),
								 psPulseShape.getParameter<unsigned int>("TrianglePeakTS"),
								 psPulseShape.getParameter<std::vector<double> >("LinearThreshold"),
								 psPulseShape.getParameter<std::vector<double> >("LinearCut"),
								 psPulseShape.getParameter<std::vector<double> >("RMS8MaxThreshold"),
								 psPulseShape.getParameter<std::vector<double> >("RMS8MaxCut"),
								 psPulseShape.getParameter<std::vector<double> >("LeftSlopeThreshold"),
								 psPulseShape.getParameter<std::vector<double> >("LeftSlopeCut"),
								 psPulseShape.getParameter<std::vector<double> >("RightSlopeThreshold"),
								 psPulseShape.getParameter<std::vector<double> >("RightSlopeCut"),
								 psPulseShape.getParameter<std::vector<double> >("RightSlopeSmallThreshold"),
								 psPulseShape.getParameter<std::vector<double> >("RightSlopeSmallCut"),
								 psPulseShape.getParameter<std::vector<double> >("TS4TS5LowerThreshold"),
								 psPulseShape.getParameter<std::vector<double> >("TS4TS5LowerCut"),
								 psPulseShape.getParameter<std::vector<double> >("TS4TS5UpperThreshold"),
								 psPulseShape.getParameter<std::vector<double> >("TS4TS5UpperCut"),
								 psPulseShape.getParameter<bool>("UseDualFit"),
                         psPulseShape.getParameter<bool>("TriangleIgnoreSlow"));
      }  // if (setPulseShapeFlags_)

    produces<HBHERecHitCollection>();
  } else if (!strcasecmp(subd.c_str(),"HO")) {
    subdet_=HcalOuter;
    produces<HORecHitCollection>();
  } else if (!strcasecmp(subd.c_str(),"HF")) {
    subdet_=HcalForward;
    digiTimeFromDB_=conf.getParameter<bool>("digiTimeFromDB");

    if (setTimingTrustFlags_) {
      
      const edm::ParameterSet& pstrust      = conf.getParameter<edm::ParameterSet>("hfTimingTrustParameters");
      HFTimingTrustFlagSetter_=new HFTimingTrustFlag(pstrust.getParameter<int>("hfTimingTrustLevel1"),
						     pstrust.getParameter<int>("hfTimingTrustLevel2"));
    }

    if (setNoiseFlags_)
      {
	const edm::ParameterSet& psdigi    =conf.getParameter<edm::ParameterSet>("digistat");
	const edm::ParameterSet& psTimeWin =conf.getParameter<edm::ParameterSet>("HFInWindowStat");
	hfdigibit_=new HcalHFStatusBitFromDigis(psdigi,psTimeWin);

	const edm::ParameterSet& psS9S1   = conf.getParameter<edm::ParameterSet>("S9S1stat");
	hfS9S1_   = new HcalHF_S9S1algorithm(psS9S1.getParameter<std::vector<double> >("short_optimumSlope"),
					     psS9S1.getParameter<std::vector<double> >("shortEnergyParams"),
					     psS9S1.getParameter<std::vector<double> >("shortETParams"),
					     psS9S1.getParameter<std::vector<double> >("long_optimumSlope"),
					     psS9S1.getParameter<std::vector<double> >("longEnergyParams"),
					     psS9S1.getParameter<std::vector<double> >("longETParams"),
					     psS9S1.getParameter<int>("HcalAcceptSeverityLevel"),
					     psS9S1.getParameter<bool>("isS8S1")
					     );

	const edm::ParameterSet& psS8S1   = conf.getParameter<edm::ParameterSet>("S8S1stat");
	hfS8S1_   = new HcalHF_S9S1algorithm(psS8S1.getParameter<std::vector<double> >("short_optimumSlope"),
					     psS8S1.getParameter<std::vector<double> >("shortEnergyParams"),
					     psS8S1.getParameter<std::vector<double> >("shortETParams"),
					     psS8S1.getParameter<std::vector<double> >("long_optimumSlope"),
					     psS8S1.getParameter<std::vector<double> >("longEnergyParams"),
					     psS8S1.getParameter<std::vector<double> >("longETParams"),
					     psS8S1.getParameter<int>("HcalAcceptSeverityLevel"),
					     psS8S1.getParameter<bool>("isS8S1")
					     );

	const edm::ParameterSet& psPET    = conf.getParameter<edm::ParameterSet>("PETstat");
	hfPET_    = new HcalHF_PETalgorithm(psPET.getParameter<std::vector<double> >("short_R"),
					    psPET.getParameter<std::vector<double> >("shortEnergyParams"),
					    psPET.getParameter<std::vector<double> >("shortETParams"),
					    psPET.getParameter<std::vector<double> >("long_R"),
					    psPET.getParameter<std::vector<double> >("longEnergyParams"),
					    psPET.getParameter<std::vector<double> >("longETParams"),
					    psPET.getParameter<int>("HcalAcceptSeverityLevel"),
					    psPET.getParameter<std::vector<double> >("short_R_29"),
					    psPET.getParameter<std::vector<double> >("long_R_29")
					    );
      }
    produces<HFRecHitCollection>();
  } else if (!strcasecmp(subd.c_str(),"ZDC")) {
    det_=DetId::Calo;
    subdet_=HcalZDCDetId::SubdetectorId;
    produces<ZDCRecHitCollection>();
  } else if (!strcasecmp(subd.c_str(),"CALIB")) {
    subdet_=HcalOther;
    subdetOther_=HcalCalibration;
    produces<HcalCalibRecHitCollection>();
  } else {
     std::cout << "HcalHitReconstructor is not associated with a specific subdetector!" << std::endl;
  }       
  
}

HcalHitReconstructor::~HcalHitReconstructor() {
  if (hbheFlagSetter_)        delete hbheFlagSetter_;
  if (hfdigibit_)             delete hfdigibit_;
  if (hbheHSCPFlagSetter_)    delete hbheHSCPFlagSetter_;
  if (hbhePulseShapeFlagSetter_) delete hbhePulseShapeFlagSetter_;
  if (hfS9S1_)                delete hfS9S1_;
  if (hfPET_)                 delete hfPET_;
  if (theTopology)            delete theTopology;
  if (paramTS)            delete paramTS;
}

void HcalHitReconstructor::beginRun(edm::Run&r, edm::EventSetup const & es){

  if ( tsFromDB_== true || recoParamsFromDB_ == true )
    {
      edm::ESHandle<HcalRecoParams> p;
      es.get<HcalRecoParamsRcd>().get(p);
      paramTS = new HcalRecoParams(*p.product());

      edm::ESHandle<HcalTopology> htopo;
      es.get<IdealGeometryRecord>().get(htopo);
      theTopology=new HcalTopology(*htopo);
      paramTS->setTopo(theTopology);
      


      // std::cout<<" skdump in HcalHitReconstructor::beginRun   dupm RecoParams "<<std::endl;
      // std::ofstream skfile("skdumpRecoParamsNewFormat.txt");
      // HcalDbASCIIIO::dumpObject(skfile, (*paramTS) );
    }

  if (digiTimeFromDB_==true)
    {
      edm::ESHandle<HcalFlagHFDigiTimeParams> p;
      es.get<HcalFlagHFDigiTimeParamsRcd>().get(p);
      HFDigiTimeParams = p.product();

      if (theTopology==0) {
	edm::ESHandle<HcalTopology> htopo;
	es.get<IdealGeometryRecord>().get(htopo);
	theTopology=new HcalTopology(*htopo);
      }
      HFDigiTimeParams->setTopo(theTopology);

    }
  reco_.beginRun(es);
}

void HcalHitReconstructor::endRun(edm::Run&r, edm::EventSetup const & es){
  if (tsFromDB_==true)
    {
      if (paramTS) delete paramTS; paramTS=0;
    }
  if (digiTimeFromDB_==true)
    {
      //DLif (HFDigiTimeParams) delete HFDigiTimeParams;
    }
  reco_.endRun();
}

void HcalHitReconstructor::produce(edm::Event& e, const edm::EventSetup& eventSetup)
{

  // get conditions
  edm::ESHandle<HcalTopology> topo;
  eventSetup.get<IdealGeometryRecord>().get(topo);


  edm::ESHandle<HcalDbService> conditions;
  eventSetup.get<HcalDbRecord>().get(conditions);
  // HACK related to HB- corrections
  if(e.isRealData()) reco_.setForData();    
  if(useLeakCorrection_) reco_.setLeakCorrection();

  edm::ESHandle<HcalChannelQuality> p;
  eventSetup.get<HcalChannelQualityRcd>().get(p);
  //DLHcalChannelQuality* myqual = new HcalChannelQuality(*p.product());
  const HcalChannelQuality* myqual = p.product();
  if (!myqual->topo()) myqual->setTopo(topo.product());


  edm::ESHandle<HcalSeverityLevelComputer> mycomputer;
  eventSetup.get<HcalSeverityLevelComputerRcd>().get(mycomputer);
  const HcalSeverityLevelComputer* mySeverity = mycomputer.product();
  
  if (det_==DetId::Hcal) {

    // HBHE -------------------------------------------------------------------
    if (subdet_==HcalBarrel || subdet_==HcalEndcap) {
      edm::Handle<HBHEDigiCollection> digi;
      
      e.getByLabel(inputLabel_,digi);
      
      // create empty output
      std::auto_ptr<HBHERecHitCollection> rec(new HBHERecHitCollection);
      rec->reserve(digi->size());
      // run the algorithm
      if (setNoiseFlags_) hbheFlagSetter_->Clear();
      HBHEDigiCollection::const_iterator i;
      std::vector<HBHEDataFrame> HBDigis;
      std::vector<int> RecHitIndex;

      // Vote on majority TS0 CapId
      int favorite_capid = 0; 
      if (correctTiming_) {
        long capid_votes[4] = {0,0,0,0};
        for (i=digi->begin(); i!=digi->end(); i++) {
          capid_votes[(*i)[0].capid()]++;
        }
        for (int k = 0; k < 4; k++)
          if (capid_votes[k] > capid_votes[favorite_capid])
            favorite_capid = k;
      }

      for (i=digi->begin(); i!=digi->end(); i++) {
	HcalDetId cell = i->id();
	DetId detcell=(DetId)cell;

        if(tsFromDB_ || recoParamsFromDB_) {
          const HcalRecoParam* param_ts = paramTS->getValues(detcell.rawId());
	  if(tsFromDB_) {
	    firstSample_  = param_ts->firstSample();
	    samplesToAdd_ = param_ts->samplesToAdd();
	  }
          if(recoParamsFromDB_) {
             bool correctForTimeslew=param_ts->correctForTimeslew();
             bool correctForPhaseContainment= param_ts->correctForPhaseContainment();
             float phaseNS=param_ts->correctionPhaseNS();
             useLeakCorrection_= param_ts->useLeakCorrection();
             correctTiming_ = param_ts->correctTiming();
             firstAuxTS_ = param_ts->firstAuxTS();
             int pileupCleaningID = param_ts->pileupCleaningID();

	     /*	     
	     int sub     = cell.subdet();
	     int depth   = cell.depth();
	     int inteta  = cell.ieta();
	     int intphi  = cell.iphi();

	     std::cout << "HcalHitReconstructor::produce  cell:" 
		       << " sub, ieta, iphi, depth = " 
		       << sub << "  " << inteta << "  " << intphi 
		       << "  " << depth << std::endl
		       << "    first, toadd = " << firstSample_ << ", "
		       << samplesToAdd_ << std::endl
		       << "    correctForTimeslew " << correctForTimeslew
		       << std::endl
		       << "    correctForPhaseContainment " 
		       <<  correctForPhaseContainment << std::endl
		       << "    phaseNS " <<  phaseNS << std::endl
		       << "    useLeakCorrection  " << useLeakCorrection_ 
		       << std::endl 
		       << "    correctTiming " << correctTiming_ << std::endl
		       << "    firstAuxTS " << firstAuxTS_  << std::endl
		       << "    pileupCleaningID "  << pileupCleaningID
		       << std::endl;
	     */

             reco_.setRecoParams(correctForTimeslew,correctForPhaseContainment,useLeakCorrection_,pileupCleaningID,phaseNS);
          }
        }

        int first = firstSample_;
        int toadd = samplesToAdd_;

	// check on cells to be ignored and dropped: (rof,20.Feb.09)
	const HcalChannelStatus* mydigistatus=myqual->getValues(detcell.rawId());
	if (mySeverity->dropChannel(mydigistatus->getValue() ) ) continue;
	if (dropZSmarkedPassed_)
	  if (i->zsMarkAndPass()) continue;

	const HcalCalibrations& calibrations=conditions->getHcalCalibrations(cell);
	const HcalQIECoder* channelCoder = conditions->getHcalCoder (cell);
	const HcalQIEShape* shape = conditions->getHcalShape (channelCoder);
	HcalCoderDb coder (*channelCoder, *shape);

	rec->push_back(reco_.reconstruct(*i,first,toadd,coder,calibrations));

	// Set auxiliary flag
	int auxflag=0;
        int fTS = firstAuxTS_;
	if (fTS<0) fTS=0; // silly protection against time slice <0
	for (int xx=fTS; xx<fTS+4 && xx<i->size();++xx)
	  auxflag+=(i->sample(xx).adc())<<(7*(xx-fTS)); // store the time slices in the first 28 bits of aux, a set of 4 7-bit adc values
	// bits 28 and 29 are reserved for capid of the first time slice saved in aux
	auxflag+=((i->sample(fTS).capid())<<28);
	(rec->back()).setAux(auxflag);

	(rec->back()).setFlags(0);  // this sets all flag bits to 0
	// Set presample flag
	if (fTS>0)
	  (rec->back()).setFlagField((i->sample(fTS-1).adc()), HcalCaloFlagLabels::PresampleADC,7);

	if (hbheTimingShapedFlagSetter_!=0)
	  hbheTimingShapedFlagSetter_->SetTimingShapedFlags(rec->back());
	if (setNoiseFlags_)
	  hbheFlagSetter_->SetFlagsFromDigi(&(*topo),rec->back(),*i,coder,calibrations,first,toadd);
	if (setPulseShapeFlags_ == true)
	  hbhePulseShapeFlagSetter_->SetPulseShapeFlags(rec->back(), *i, coder, calibrations);
	if (setSaturationFlags_)
	  saturationFlagSetter_->setSaturationFlag(rec->back(),*i);
	if (correctTiming_)
	  HcalTimingCorrector::Correct(rec->back(), *i, favorite_capid);
	if (setHSCPFlags_ && i->id().ietaAbs()<16)
	  {
	    double DigiEnergy=0;
            for(int j=0; j!=i->size(); DigiEnergy += i->sample(j++).nominal_fC());
            if(DigiEnergy > hbheHSCPFlagSetter_->EnergyThreshold())
              {
                HBDigis.push_back(*i);
                RecHitIndex.push_back(rec->size()-1);
              }
	    
	  } // if (set HSCPFlags_ && |ieta|<16)
      } // loop over HBHE digis


      if (setNoiseFlags_) hbheFlagSetter_->SetFlagsFromRecHits(&(*topo),*rec);
      if (setHSCPFlags_)  hbheHSCPFlagSetter_->hbheSetTimeFlagsFromDigi(rec.get(), HBDigis, RecHitIndex);
      // return result
      e.put(rec);

      //  HO ------------------------------------------------------------------
    } else if (subdet_==HcalOuter) {
      edm::Handle<HODigiCollection> digi;
      e.getByLabel(inputLabel_,digi);
      
      // create empty output
      std::auto_ptr<HORecHitCollection> rec(new HORecHitCollection);
      rec->reserve(digi->size());
      // run the algorithm
      HODigiCollection::const_iterator i;

      // Vote on majority TS0 CapId
      int favorite_capid = 0; 
      if (correctTiming_) {
        long capid_votes[4] = {0,0,0,0};
        for (i=digi->begin(); i!=digi->end(); i++) {
          capid_votes[(*i)[0].capid()]++;
        }
        for (int k = 0; k < 4; k++)
          if (capid_votes[k] > capid_votes[favorite_capid])
            favorite_capid = k;
      }

      for (i=digi->begin(); i!=digi->end(); i++) {
	HcalDetId cell = i->id();
	DetId detcell=(DetId)cell;
        // firstSample & samplesToAdd
        if(tsFromDB_ || recoParamsFromDB_) {
          const HcalRecoParam* param_ts = paramTS->getValues(detcell.rawId());
	  if(tsFromDB_) {
	    firstSample_  = param_ts->firstSample();
	    samplesToAdd_ = param_ts->samplesToAdd();
	  }
          if(recoParamsFromDB_) {
             bool correctForTimeslew=param_ts->correctForTimeslew();
             bool correctForPhaseContainment= param_ts->correctForPhaseContainment();
             float phaseNS=param_ts->correctionPhaseNS();
             useLeakCorrection_= param_ts->useLeakCorrection();
             correctTiming_ = param_ts->correctTiming();
             firstAuxTS_ = param_ts->firstAuxTS();
             int pileupCleaningID = param_ts->pileupCleaningID();
             reco_.setRecoParams(correctForTimeslew,correctForPhaseContainment,useLeakCorrection_,pileupCleaningID,phaseNS);
          }
        }

        int first = firstSample_;
        int toadd = samplesToAdd_;

	// check on cells to be ignored and dropped: (rof,20.Feb.09)
	const HcalChannelStatus* mydigistatus=myqual->getValues(detcell.rawId());
	if (mySeverity->dropChannel(mydigistatus->getValue() ) ) continue;
	if (dropZSmarkedPassed_)
	  if (i->zsMarkAndPass()) continue;

	const HcalCalibrations& calibrations=conditions->getHcalCalibrations(cell);
	const HcalQIECoder* channelCoder = conditions->getHcalCoder (cell);
	const HcalQIEShape* shape = conditions->getHcalShape (channelCoder);
	HcalCoderDb coder (*channelCoder, *shape);

	rec->push_back(reco_.reconstruct(*i,first,toadd,coder,calibrations));

	// Set auxiliary flag
	int auxflag=0;
        int fTS = firstAuxTS_;
	if (fTS<0) fTS=0; //silly protection against negative time slice values
	for (int xx=fTS; xx<fTS+4 && xx<i->size();++xx)
	  auxflag+=(i->sample(xx).adc())<<(7*(xx-fTS)); // store the time slices in the first 28 bits of aux, a set of 4 7-bit adc values
	// bits 28 and 29 are reserved for capid of the first time slice saved in aux
	auxflag+=((i->sample(fTS).capid())<<28);
	(rec->back()).setAux(auxflag);

	(rec->back()).setFlags(0);
	// Fill Presample ADC flag
	if (fTS>0)
	  (rec->back()).setFlagField((i->sample(fTS-1).adc()), HcalCaloFlagLabels::PresampleADC,7);

	if (setSaturationFlags_)
	  saturationFlagSetter_->setSaturationFlag(rec->back(),*i);
	if (correctTiming_)
	  HcalTimingCorrector::Correct(rec->back(), *i, favorite_capid);
      }
      // return result
      e.put(rec);    

      // HF -------------------------------------------------------------------
    } else if (subdet_==HcalForward) {
      edm::Handle<HFDigiCollection> digi;
      e.getByLabel(inputLabel_,digi);


      ///////////////////////////////////////////////////////////////// HF
      // create empty output
      std::auto_ptr<HFRecHitCollection> rec(new HFRecHitCollection);
      rec->reserve(digi->size());
      // run the algorithm
      HFDigiCollection::const_iterator i;

      // Vote on majority TS0 CapId
      int favorite_capid = 0; 
      if (correctTiming_) {
        long capid_votes[4] = {0,0,0,0};
        for (i=digi->begin(); i!=digi->end(); i++) {
          capid_votes[(*i)[0].capid()]++;
        }
        for (int k = 0; k < 4; k++)
          if (capid_votes[k] > capid_votes[favorite_capid])
            favorite_capid = k;
      }

      for (i=digi->begin(); i!=digi->end(); i++) {
	HcalDetId cell = i->id();
	DetId detcell=(DetId)cell;

        if(tsFromDB_ || recoParamsFromDB_) {
          const HcalRecoParam* param_ts = paramTS->getValues(detcell.rawId());
	  if(tsFromDB_) {
	    firstSample_  = param_ts->firstSample();
	    samplesToAdd_ = param_ts->samplesToAdd();
	  }
          if(recoParamsFromDB_) {
             bool correctForTimeslew=param_ts->correctForTimeslew();
             bool correctForPhaseContainment= param_ts->correctForPhaseContainment();
             float phaseNS=param_ts->correctionPhaseNS();
             useLeakCorrection_= param_ts->useLeakCorrection();
             correctTiming_ = param_ts->correctTiming();
             firstAuxTS_ = param_ts->firstAuxTS();
             int pileupCleaningID = param_ts->pileupCleaningID();
             reco_.setRecoParams(correctForTimeslew,correctForPhaseContainment,useLeakCorrection_,pileupCleaningID,phaseNS);
          }
        }

        int first = firstSample_;
        int toadd = samplesToAdd_;

	// check on cells to be ignored and dropped: (rof,20.Feb.09)
	const HcalChannelStatus* mydigistatus=myqual->getValues(detcell.rawId());
	if (mySeverity->dropChannel(mydigistatus->getValue() ) ) continue;
	if (dropZSmarkedPassed_)
	  if (i->zsMarkAndPass()) continue;

	const HcalCalibrations& calibrations=conditions->getHcalCalibrations(cell);
	const HcalQIECoder* channelCoder = conditions->getHcalCoder (cell);
	const HcalQIEShape* shape = conditions->getHcalShape (channelCoder);
	HcalCoderDb coder (*channelCoder, *shape);

	// Set HFDigiTime flag values from digiTimeFromDB_
	if (digiTimeFromDB_==true && hfdigibit_!=0)
	  {
	    const HcalFlagHFDigiTimeParam* hfDTparam = HFDigiTimeParams->getValues(detcell.rawId());
	    hfdigibit_->resetParamsFromDB(hfDTparam->HFdigiflagFirstSample(),
					  hfDTparam->HFdigiflagSamplesToAdd(),
					  hfDTparam->HFdigiflagExpectedPeak(),
					  hfDTparam->HFdigiflagMinEThreshold(),
					  hfDTparam->HFdigiflagCoefficients()
					  );
	  }

	//std::cout << "TOADDHF " << toadd << " " << first << " " << std::endl;
	rec->push_back(reco_.reconstruct(*i,first,toadd,coder,calibrations));

	// Set auxiliary flag
	int auxflag=0;
        int fTS = firstAuxTS_;
	if (fTS<0) fTS=0; // silly protection against negative time slice values
	for (int xx=fTS; xx<fTS+4 && xx<i->size();++xx)
	  auxflag+=(i->sample(xx).adc())<<(7*(xx-fTS)); // store the time slices in the first 28 bits of aux, a set of 4 7-bit adc values
	// bits 28 and 29 are reserved for capid of the first time slice saved in aux
	auxflag+=((i->sample(fTS).capid())<<28);
	(rec->back()).setAux(auxflag);

	// Clear flags
	(rec->back()).setFlags(0);

	// Fill Presample ADC flag
	if (fTS>0)
	  (rec->back()).setFlagField((i->sample(fTS-1).adc()), HcalCaloFlagLabels::PresampleADC,7);

	// This calls the code for setting the HF noise bit determined from digi shape
	if (setNoiseFlags_) 
	  hfdigibit_->hfSetFlagFromDigi(rec->back(),*i,coder,calibrations);
	if (setSaturationFlags_)
	  saturationFlagSetter_->setSaturationFlag(rec->back(),*i);
	if (setTimingTrustFlags_)
	  HFTimingTrustFlagSetter_->setHFTimingTrustFlag(rec->back(),*i);
	if (correctTiming_)
	  HcalTimingCorrector::Correct(rec->back(), *i, favorite_capid);
      } // for (i=digi->begin(); i!=digi->end(); i++) -- loop on all HF digis

      // The following flags require the full set of rechits
      // These need to be set consecutively, so an energy check should be the first 
      // test performed on these hits (to minimize the loop time)
      if (setNoiseFlags_) 
	{
	  // Step 1:  Set PET flag  (short fibers of |ieta|==29)
	  // Neighbor/partner channels that are flagged by Pulse Shape algorithm (HFDigiTime)
	  // won't be considered in these calculations
	  for (HFRecHitCollection::iterator i = rec->begin();i!=rec->end();++i)
	    {
	      int depth=i->id().depth();
	      int ieta=i->id().ieta();
	      // Short fibers and all channels at |ieta|=29 use PET settings in Algo 3
	      if (depth==2 || abs(ieta)==29 ) 
		hfPET_->HFSetFlagFromPET(*i,*rec,myqual,mySeverity);
	    }

	  // Step 2:  Set S8S1 flag (short fibers or |ieta|==29)
	  for (HFRecHitCollection::iterator i = rec->begin();i!=rec->end();++i)
	    {
	      int depth=i->id().depth();
	      int ieta=i->id().ieta();
	      // Short fibers and all channels at |ieta|=29 use PET settings in Algo 3
	      if (depth==2 || abs(ieta)==29 ) 
		hfS8S1_->HFSetFlagFromS9S1(*i,*rec,myqual,mySeverity);
	    }

	  // Set 3:  Set S9S1 flag (long fibers)
	  for (HFRecHitCollection::iterator i = rec->begin();i!=rec->end();++i)
	    {
	      int depth=i->id().depth();
	      int ieta=i->id().ieta();
	      // Short fibers and all channels at |ieta|=29 use PET settings in Algo 3
	      if (depth==1 && abs(ieta)!=29 ) 
		hfS9S1_->HFSetFlagFromS9S1(*i,*rec,myqual, mySeverity);
	    }
	}

      // return result
      e.put(rec);     
    } else if (subdet_==HcalOther && subdetOther_==HcalCalibration) {
      edm::Handle<HcalCalibDigiCollection> digi;
      e.getByLabel(inputLabel_,digi);
      
      // create empty output
      std::auto_ptr<HcalCalibRecHitCollection> rec(new HcalCalibRecHitCollection);
      rec->reserve(digi->size());
      // run the algorithm
      int first = firstSample_;
      int toadd = samplesToAdd_;

      HcalCalibDigiCollection::const_iterator i;
      for (i=digi->begin(); i!=digi->end(); i++) {
	HcalCalibDetId cell = i->id();
	//	HcalDetId cellh = i->id();
	DetId detcell=(DetId)cell;
	// check on cells to be ignored and dropped: (rof,20.Feb.09)
	const HcalChannelStatus* mydigistatus=myqual->getValues(detcell.rawId());
	if (mySeverity->dropChannel(mydigistatus->getValue() ) ) continue;
	if (dropZSmarkedPassed_)
	  if (i->zsMarkAndPass()) continue;

	const HcalCalibrations& calibrations=conditions->getHcalCalibrations(cell);
	const HcalQIECoder* channelCoder = conditions->getHcalCoder (cell);
	const HcalQIEShape* shape = conditions->getHcalShape (channelCoder);
	HcalCoderDb coder (*channelCoder, *shape);

	// firstSample & samplesToAdd
        if(tsFromDB_) {
	  const HcalRecoParam* param_ts = paramTS->getValues(detcell.rawId());
	  first = param_ts->firstSample();    
	  toadd = param_ts->samplesToAdd();    
	}
	rec->push_back(reco_.reconstruct(*i,first,toadd,coder,calibrations));

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
      e.put(rec);     
    }
  } 
  //DL  delete myqual;
} // void HcalHitReconstructor::produce(...)
