#include "L1Trigger/RegionalCaloTrigger/interface/L1RCTProducer.h" 

#include "DataFormats/L1CaloTrigger/interface/L1CaloCollections.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"

// default scales
#include "CondFormats/L1TObjects/interface/L1CaloEcalScale.h"
#include "CondFormats/DataRecord/interface/L1CaloEcalScaleRcd.h"
#include "CondFormats/L1TObjects/interface/L1CaloHcalScale.h"
#include "CondFormats/DataRecord/interface/L1CaloHcalScaleRcd.h"

// debug scales
#include "CalibCalorimetry/EcalTPGTools/interface/EcalTPGScale.h"
#include "CalibFormats/CaloTPG/interface/CaloTPGTranscoder.h"
#include "CalibFormats/CaloTPG/interface/CaloTPGRecord.h"

#include "CondFormats/L1TObjects/interface/L1CaloEtScale.h"
#include "CondFormats/DataRecord/interface/L1EmEtScaleRcd.h"

#include "CondFormats/L1TObjects/interface/L1RCTParameters.h"
#include "CondFormats/DataRecord/interface/L1RCTParametersRcd.h"
#include "CondFormats/L1TObjects/interface/L1RCTChannelMask.h"
#include "CondFormats/DataRecord/interface/L1RCTChannelMaskRcd.h"

#include "L1Trigger/RegionalCaloTrigger/interface/L1RCT.h"
#include "L1Trigger/RegionalCaloTrigger/interface/L1RCTLookupTables.h" 

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <vector>
using std::vector;

#include <iostream>
using std::cout;
using std::endl;

L1RCTProducer::L1RCTProducer(const edm::ParameterSet& conf) : 
  rctLookupTables(new L1RCTLookupTables),
  rct(new L1RCT(rctLookupTables)),
  useEcal(conf.getParameter<bool>("useEcal")),
  useHcal(conf.getParameter<bool>("useHcal")),
  ecalDigisLabel(conf.getParameter<edm::InputTag>("ecalDigisLabel")),
  hcalDigisLabel(conf.getParameter<edm::InputTag>("hcalDigisLabel")),
  useDebugTpgScales(conf.getParameter<bool>("useDebugTpgScales")),
  useHcalCosmicTiming(conf.getParameter<bool>("useHcalCosmicTiming")),
  useEcalCosmicTiming(conf.getParameter<bool>("useEcalCosmicTiming")),
  preSamples(conf.getParameter<unsigned>("preSamples")),
  postSamples(conf.getParameter<unsigned>("postSamples")),
  useMCAsInput(conf.getParameter<bool>("UseMCAsInput")),
  hfShift(conf.getParameter<int>("HFShift")),
  hbShift(conf.getParameter<int>("HBShift"))
{
  produces<L1CaloEmCollection>();
  produces<L1CaloRegionCollection>();
}

L1RCTProducer::~L1RCTProducer()
{
  if(rct != 0) delete rct;
  if(rctLookupTables != 0) delete rctLookupTables;
}

void L1RCTProducer::beginJob(const edm::EventSetup& eventSetup)
{
}

void L1RCTProducer::produce(edm::Event& event, const edm::EventSetup& eventSetup)
{

  // Refresh configuration information every event
  // Hopefully, this does not take too much time
  // There should be a call back function in future to
  // handle changes in configuration

  // parameters to configure RCT (thresholds, etc)
  edm::ESHandle<L1RCTParameters> rctParameters;
  eventSetup.get<L1RCTParametersRcd>().get(rctParameters);
  const L1RCTParameters* r = rctParameters.product();

  // list of RCT channels to mask
  edm::ESHandle<L1RCTChannelMask> channelMask;
  eventSetup.get<L1RCTChannelMaskRcd>().get(channelMask);
  const L1RCTChannelMask* c = channelMask.product();
  
  // energy scale to convert eGamma output
  edm::ESHandle<L1CaloEtScale> emScale;
  eventSetup.get<L1EmEtScaleRcd>().get(emScale);
  const L1CaloEtScale* s = emScale.product();

  rctLookupTables->setRCTParameters(r);
  rctLookupTables->setChannelMask(c);
  rctLookupTables->setL1CaloEtScale(s);

  // use these dummies to get the delete right when using old-style
  // scales to create set of L1CaloXcalScales
  L1CaloEcalScale* dummyE(0);
  L1CaloHcalScale* dummyH(0);

  if (useDebugTpgScales) // generate new-style scales from tpg scales
    {

      std::cout << "Using old-style TPG scales!" << endl;

      // old version of hcal energy scale to convert input
      edm::ESHandle<CaloTPGTranscoder> transcoder;
      eventSetup.get<CaloTPGRecord>().get(transcoder);
      const CaloTPGTranscoder* h_tpg = transcoder.product();

      // old version of ecal energy scale to convert input
      EcalTPGScale* e_tpg = new EcalTPGScale();
      e_tpg->setEventSetup(eventSetup);

      L1CaloEcalScale* ecalScale = new L1CaloEcalScale();
      L1CaloHcalScale* hcalScale = new L1CaloHcalScale();

      // generate L1CaloXcalScales from old-style scales (thanks, werner!)
      // ECAL
      //std::cout << "ECAL Pos " << L1CaloEcalScale::nBinRank << std::endl ;
      for( unsigned short ieta = 1 ; ieta <= L1CaloEcalScale::nBinEta; ++ieta )
	{
	  for( unsigned short irank = 0 ; irank < L1CaloEcalScale::nBinRank; ++irank )
	    {
	      //std::cout << ieta << " " << irank ;
	      EcalSubdetector subdet = ( ieta <= 17 ) ? EcalBarrel : EcalEndcap ;
	      double etGeVPos =
		e_tpg->getTPGInGeV
		( irank, EcalTrigTowerDetId(1, // +ve eta
					    subdet,
					    ieta,
					    1 )); // dummy phi value
	      ecalScale->setBin( irank, ieta, 1, etGeVPos ) ;
	      //std::cout << etGeVPos << ", " ;
	    }
	  //std::cout << std::endl ;
	}
      //std::cout << std::endl ;
      
      //std::cout << "ECAL Neg" << std::endl ;
      for( unsigned short ieta = 1 ; ieta <= L1CaloEcalScale::nBinEta; ++ieta )
	{
	  for( unsigned short irank = 0 ; irank < L1CaloEcalScale::nBinRank; ++irank )
	    {
	      EcalSubdetector subdet = ( ieta <= 17 ) ? EcalBarrel : EcalEndcap ;
	      
	      //std::cout << ieta << " " << irank ;
	      double etGeVNeg =
		e_tpg->getTPGInGeV
		( irank,
		  EcalTrigTowerDetId(-1, // -ve eta
				     subdet,
				     ieta,
				     2 )); // dummy phi value
	      ecalScale->setBin( irank, ieta, -1, etGeVNeg ) ;
	      //std::cout << etGeVNeg << ", " ;
	    }
	  //std::cout << std::endl ;
	}
      //std::cout << std::endl ;

      // HCAL
      //std::cout << "HCAL" << std::endl ;
      for( unsigned short ieta = 1 ; ieta <= L1CaloHcalScale::nBinEta; ++ieta )
	{
	  for( unsigned short irank = 0 ; irank < L1CaloHcalScale::nBinRank; ++irank )
	    {
	      double etGeV = h_tpg->hcaletValue( ieta, irank ) ;

	      hcalScale->setBin( irank, ieta, 1, etGeV ) ;
	      hcalScale->setBin( irank, ieta, -1, etGeV ) ;
	      //std::cout << etGeV << ", " ;
	    }
	  //std::cout << std::endl ;
	}
      //std::cout << std::endl ;

      // set the input scales
      rctLookupTables->setEcalScale(ecalScale);
      rctLookupTables->setHcalScale(hcalScale);

      dummyE = ecalScale;
      dummyH = hcalScale;

      delete e_tpg;

    }
  else
    {

      // get energy scale to convert input from ECAL
      edm::ESHandle<L1CaloEcalScale> ecalScale;
      eventSetup.get<L1CaloEcalScaleRcd>().get(ecalScale);
      const L1CaloEcalScale* e = ecalScale.product();
      
      // get energy scale to convert input from HCAL
      edm::ESHandle<L1CaloHcalScale> hcalScale;
      eventSetup.get<L1CaloHcalScaleRcd>().get(hcalScale);
      const L1CaloHcalScale* h = hcalScale.product();

      // set scales
      rctLookupTables->setEcalScale(e);
      rctLookupTables->setHcalScale(h);

    }

  edm::Handle<EcalTrigPrimDigiCollection> ecal;
  edm::Handle<HcalTrigPrimDigiCollection> hcal;
  
  EcalTrigPrimDigiCollection::const_iterator ecal_it;
  HcalTrigPrimDigiCollection::const_iterator hcal_it;

  if (useEcal) { event.getByLabel(ecalDigisLabel, ecal); }
  if (useHcal) { event.getByLabel(hcalDigisLabel, hcal); }

  unsigned nSamples = preSamples + postSamples + 1;
  //std::cout << "pre " << preSamples << " post " << postSamples
  //	    << " total " << nSamples << std::endl;

  bool tooLittleDataEcal = false;
  bool tooLittleDataHcal = false;

  std::vector<EcalTrigPrimDigiCollection> ecalColl(nSamples);
  std::vector<HcalTrigPrimDigiCollection> hcalColl(nSamples);
  if (ecal.isValid()) 
    { 
      // loop through all ecal digis
      for (ecal_it = ecal->begin(); ecal_it != ecal->end(); ecal_it++)
	{
	  short zside = ecal_it->id().zside();
	  unsigned short ietaAbs = ecal_it->id().ietaAbs();
	  short iphi = ecal_it->id().iphi();
	  /*
	  if (ecal_it->compressedEt() > 0)
	    {
	      std::cout << "[Producer] ecal tower energy is " 
			<< ecal_it->compressedEt() << std::endl;
	    }
	  */
	  // loop through requested time samples for each digi
	  unsigned short digiSize = ecal_it->size();
	  // (size of each digi must be no less than nSamples)
	  unsigned short nSOI = (unsigned short) ( ecal_it->
						   sampleOfInterest() );
	  if (digiSize < nSamples || nSOI < preSamples
	      || ((digiSize - nSOI) < (nSamples - preSamples)))
	    {
	      // log error -- should not happen!
	      if (tooLittleDataEcal == false)
		{
		  edm::LogWarning ("TooLittleData")
		    << "ECAL data should have at least " << nSamples
		    << " time samples per digi, current digi has " 
		    << digiSize << ".  Insufficient data to process "
		    << "requested bx's.  Filling extra with zeros";
		  tooLittleDataEcal = true;
		}
	      unsigned short preLoopsZero = (unsigned short) (preSamples) 
		- nSOI;
	      unsigned short postLoopsZero = (unsigned short) (postSamples)
		- (digiSize - nSOI - 1);
	      
	      // fill extra bx's at beginning with zeros
	      for (int sample = 0; sample < preLoopsZero; sample++)
		{
		  // fill first few with zeros
		  EcalTriggerPrimitiveDigi
		    ecalDigi(EcalTrigTowerDetId((int) zside, EcalTriggerTower,
						(int) ietaAbs, (int) iphi));
		  ecalDigi.setSize(1);
		  ecalDigi.setSample(0, EcalTriggerPrimitiveSample(0,false,0));
		  ecalColl[sample].push_back(ecalDigi);
		}

	      // loop through existing data
	      for (int sample = preLoopsZero; 
		   sample < (preLoopsZero + digiSize); sample++)
		{
		  // go through data
		  EcalTriggerPrimitiveDigi
		    ecalDigi(EcalTrigTowerDetId((int) zside, EcalTriggerTower,
						(int) ietaAbs, (int) iphi));
		  ecalDigi.setSize(1);

		  if (useEcalCosmicTiming && iphi >= 1 && iphi <= 36)
		    {
		      if (nSOI < (preSamples + 1))
			{
			  edm::LogWarning ("TooLittleData")
			    << "ECAL data needs at least one presample "
			    << "more than the number requested "
			    << "to use ecal cosmic timing mod!  "
			    //<< "Setting data "
			    //<< "for this time slice to zero and "
			    << "reverting to useEcalCosmicTiming = false "
			    << "for rest of job.";
			  //ecalDigi.setSample(0, EcalTriggerPrimitiveSample
			  //		     (0, false, 0));
			  useEcalCosmicTiming = false;
			}
		      else
			{
			  // take data from one crossing earlier
			  ecalDigi.setSample(0, EcalTriggerPrimitiveSample
					     (ecal_it->sample(nSOI + sample - 
							      preSamples - 
							      1).raw()));
			}
		    }
		  //else
		  if ((!useEcalCosmicTiming) || (iphi >=37 && iphi <= 72))
		    {
		      ecalDigi.setSample(0, EcalTriggerPrimitiveSample
					 (ecal_it->sample(nSOI + sample - 
							  preSamples).raw()));
		    }
		  ecalColl[sample].push_back(ecalDigi);
		  /*
		  if (ecal_it->sample(sample).compressedEt() > 0)
		    {
		      std::cout << "[Producer] ecal tower energy is "
		      << ecal_it->sample(sample).compressedEt()
		      << " in sample " << sample << std::endl;
		    }
		  */
		}
	      
	      // fill extra bx's at end with zeros
	      for (int sample = (preLoopsZero + digiSize); 
		   sample < nSamples; sample++)
		{
		  // fill zeros!
		  EcalTriggerPrimitiveDigi
		    ecalDigi(EcalTrigTowerDetId((int) zside, EcalTriggerTower,
						(int) ietaAbs, (int) iphi));
		  ecalDigi.setSize(1);
		  ecalDigi.setSample(0, EcalTriggerPrimitiveSample(0,false,0));
		  ecalColl[sample].push_back(ecalDigi);
		}
	    }
	  else
	    {
	      for (unsigned short sample = 0; sample < nSamples; sample++)
		{
		  /*
		  if (ecal_it->sample(sample).compressedEt() > 0)
		    {
		      std::cout << "[Producer] ecal tower energy is "
				<< ecal_it->sample(sample).compressedEt()
				<< " in sample " << sample << std::endl;
		    }
		  */
		  // put each time sample into its own digi
		  short zside = ecal_it->id().zside();
		  unsigned short ietaAbs = ecal_it->id().ietaAbs();
		  short iphi = ecal_it->id().iphi();
		  EcalTriggerPrimitiveDigi
		    ecalDigi(EcalTrigTowerDetId((int) zside, EcalTriggerTower,
						(int) ietaAbs, (int) iphi));
		  ecalDigi.setSize(1);

		  if (useEcalCosmicTiming && iphi >= 1 && iphi <=36)
		    {
		      if (nSOI < (preSamples + 1))
			{
			  edm::LogWarning ("TooLittleData")
			    << "ECAL data needs at least one presample "
			    << "more than the number requested "
			    << "to use ecal cosmic timing mod!  "
			    //<< "Setting data "
			    //<< "for this time slice to zero and "
			    << "reverting to useEcalCosmicTiming = false "
			    << "for rest of job.";
			  //ecalDigi.setSample(0, EcalTriggerPrimitiveSample
			  //	     (0, false, 0));
			  useEcalCosmicTiming = false;
			}
		      else
			{
			  ecalDigi.setSample(0, EcalTriggerPrimitiveSample
					     (ecal_it->sample
					      (ecal_it->sampleOfInterest() + 
					       sample - preSamples - 
					       1).raw()));
			}
		    }
		  //else
		  if ((!useEcalCosmicTiming) || (iphi >=37 && iphi <= 72))
		    {
		      ecalDigi.setSample(0, EcalTriggerPrimitiveSample
					 (ecal_it->sample
					  (ecal_it->sampleOfInterest() + 
					   sample - preSamples).raw()));
		    }
		  // push back each digi into correct "time sample" of coll
		  ecalColl[sample].push_back(ecalDigi);
		}
	    }
	}
    }
  if (hcal.isValid()) 
    { 
      // loop through all hcal digis
      for (hcal_it = hcal->begin(); hcal_it != hcal->end(); hcal_it++)
	{
	  short ieta = hcal_it->id().ieta();
	  short iphi = hcal_it->id().iphi();
	  // loop through time samples for each digi
	  unsigned short digiSize = hcal_it->size();
	  // (size of each digi must be no less than nSamples)
	  unsigned short nSOI = (unsigned short) (hcal_it->presamples());
	  if (digiSize < nSamples || nSOI < preSamples
	      || ((digiSize - nSOI) < (nSamples - preSamples)))
	    {
	      // log error -- should not happen!
	      if (tooLittleDataHcal == false)
		{
		  edm::LogWarning ("TooLittleData")
		    << "HCAL data should have at least " << nSamples
		    << " time samples per digi, current digi has " 
		    << digiSize << ".  Insufficient data to process "
		    << "requested bx's.  Filling extra with zeros";
		  tooLittleDataHcal = true;
		}
	      unsigned short preLoopsZero = (unsigned short) (preSamples) 
		- nSOI;
	      unsigned short postLoopsZero = (unsigned short) (postSamples)
		- (digiSize - nSOI - 1);
	      
	      // fill extra bx's at beginning with zeros
	      for (int sample = 0; sample < preLoopsZero; sample++)
		{
		  // fill first few with zeros
		  HcalTriggerPrimitiveDigi
		    hcalDigi(HcalTrigTowerDetId((int) ieta, (int) iphi));
		  hcalDigi.setSize(1);
		  hcalDigi.setPresamples(0);
		  hcalDigi.setSample(0, HcalTriggerPrimitiveSample(0,false,0,0));
		  hcalColl[sample].push_back(hcalDigi);
		}

	      // loop through existing data
	      for (int sample = preLoopsZero; 
		   sample < (preLoopsZero + digiSize); sample++)
		{
		  // go through data
		  HcalTriggerPrimitiveDigi
		    hcalDigi(HcalTrigTowerDetId((int) ieta, (int) iphi));
		  hcalDigi.setSize(1);
		  hcalDigi.setPresamples(0);

		  // for cosmics, hcal data from upper half of det
		  // comes 1 bx before data from bottom half
		  // SHOULDN'T ever go out of bounds (typically >=1 presample)
		  // but if 0 presamples, sets useHcalCosmicTiming = false
		  /*
		  std::cout << "[producer] useHcalCosmicTiming=" 
			    << useHcalCosmicTiming
			    << " iphi=" << iphi << " hcal presamples="
			    << hcal_it->presamples()
			    << std::endl;
		  */
		  if (useHcalCosmicTiming && iphi >= 1 && iphi <= 36)
		    {
		      if (nSOI < (preSamples + 1))
			{
			  edm::LogWarning ("TooLittleData")
			    << "HCAL data needs at least one presample "
			    << "more than the number requested "
			    << "to use hcal cosmic timing mod!  "
			    //<< "Setting data "
			    //<< "for this time slice to zero and "
			    << "reverting to useHcalCosmicTiming = false "
			    << "for rest of job.";
			  //hcalDigi.setSample(0, HcalTriggerPrimitiveSample
			  //		     (0, false, 0, 0));
			  useHcalCosmicTiming = false;
			}
		      else
			{
			  hcalDigi.setSample(0, HcalTriggerPrimitiveSample
					     (hcal_it->sample(hcal_it->
							      presamples() + 
							      sample - 
							      preSamples - 
							      1).raw()));
			}
		    }
		  //else
		  if ((!useHcalCosmicTiming) || (iphi >= 37 && iphi <= 72))
		    {

		      //If it is MC keep as it is
//		      if(useMCAsInput)
//			{
			  hcalDigi.setSample(0, HcalTriggerPrimitiveSample
					     (hcal_it->sample(hcal_it->
							      presamples() + 
							      sample - 
							      preSamples).raw()));
/*			}
		      else //It is data 
			{
			  if(ieta>-29 && ieta<29) 
			    hcalDigi.setSample(0, HcalTriggerPrimitiveSample
					       (hcal_it->sample(hcal_it->
							      presamples() + 
								sample - 
								preSamples+hbShift).raw()));
			  if(ieta<=-29 || ieta>=29)
			    hcalDigi.setSample(0, HcalTriggerPrimitiveSample
					       (hcal_it->sample(hcal_it->
							      presamples() + 
								sample - 
								preSamples+hfShift).raw()));

			}
*/
		    }
		  hcalColl[sample].push_back(hcalDigi);
		}
	      
	      // fill extra bx's at end with zeros
	      for (int sample = (preLoopsZero + digiSize); 
		   sample < nSamples; sample++)
		{
		  // fill zeros!
		  HcalTriggerPrimitiveDigi
		    hcalDigi(HcalTrigTowerDetId((int) ieta, (int) iphi));
		  hcalDigi.setSize(1);
		  hcalDigi.setPresamples(0);
		  hcalDigi.setSample(0, HcalTriggerPrimitiveSample(0,false,0,0));
		  hcalColl[sample].push_back(hcalDigi);
		}
	    }
	  else
	    {
	      for (unsigned short sample = 0; sample < nSamples; sample++)
		{
		  // put each (relevant) time sample into its own digi
		  HcalTriggerPrimitiveDigi hcalDigi(HcalTrigTowerDetId(
						    (int) ieta, (int) iphi));
		  hcalDigi.setSize(1);
		  hcalDigi.setPresamples(0);

		  // for cosmics, hcal data from upper half of det
		  // comes 1 bx before data from bottom half
		  // SHOULDN'T ever go out of bounds (typically >=1 presample)
		  // but if 0 presamples, sets useHcalCosmicTiming = false
		  /*
		  std::cout << "[producer] useHcalCosmicTiming=" 
			    << useHcalCosmicTiming
			    << " iphi=" << iphi << " hcal presamples="
			    << hcal_it->presamples() << " no extra zeros"
			    << std::endl;
		  */
		  if (useHcalCosmicTiming && iphi >= 1 && iphi <= 36)
		    {
		      if (nSOI < (preSamples + 1))
			{
			  edm::LogWarning ("TooLittleData")
			    << "HCAL data needs at least one presample "
			    << "more than the number requested "
			    << "to use hcal cosmic timing mod!  "
			    //<< "Setting data "
			    //<< "for this time slice to zero and "
			    << "reverting to useHcalCosmicTiming = false "
			    << "for rest of job.";
			  //hcalDigi.setSample(0, HcalTriggerPrimitiveSample
			  //		     (0, false, 0, 0));
			  useHcalCosmicTiming = false;
			}
		      else
			{
			  hcalDigi.setSample(0, HcalTriggerPrimitiveSample
					     (hcal_it->sample(hcal_it->
							      presamples() + 
							      sample - 
							      preSamples - 
							      1).raw()));
			}
		    }
		  //else
		  if ((!useHcalCosmicTiming) || (iphi >= 37 && iphi <= 72))
		    {
//		      hcalDigi.setSample(0, HcalTriggerPrimitiveSample
//					 (hcal_it->sample(hcal_it->
//							  presamples() + 
//							  sample - 
//							  preSamples).raw()));
		      //If it is MC keep as it is
		      if(useMCAsInput)
			{
			  hcalDigi.setSample(0, HcalTriggerPrimitiveSample
					     (hcal_it->sample(hcal_it->
							      presamples() + 
							      sample - 
							      preSamples).raw()));
			}
		      else //It is data 
			{
			  if(ieta>-29 && ieta<29) 
			    hcalDigi.setSample(0, HcalTriggerPrimitiveSample
					       (hcal_it->sample(hcal_it->
							      presamples() + 
								sample - 
								preSamples+hbShift).raw()));
			  if(ieta<=-29 || ieta>=29)
			    hcalDigi.setSample(0, HcalTriggerPrimitiveSample
					       (hcal_it->sample(hcal_it->
							      presamples() + 
								sample - 
								preSamples+hfShift).raw()));

			}
		    }
		  hcalColl[sample].push_back(hcalDigi);  
		}
	    }
	}
    }

  std::auto_ptr<L1CaloEmCollection> rctEmCands (new L1CaloEmCollection);
  std::auto_ptr<L1CaloRegionCollection> rctRegions (new L1CaloRegionCollection);

  // loop through and process each bx
  for (unsigned short sample = 0; sample < nSamples; sample++)
    {
      rct->digiInput(ecalColl[sample], hcalColl[sample]);
      rct->processEvent();

      // Stuff to create
      
      for (int j = 0; j<18; j++)
	{
	  L1CaloEmCollection isolatedEGObjects = rct->getIsolatedEGObjects(j);
	  L1CaloEmCollection nonisolatedEGObjects = rct->getNonisolatedEGObjects(j);
	  for (int i = 0; i<4; i++) 
	    {
	      isolatedEGObjects.at(i).setBx(sample - preSamples);
	      nonisolatedEGObjects.at(i).setBx(sample - preSamples);
	      rctEmCands->push_back(isolatedEGObjects.at(i));
	      rctEmCands->push_back(nonisolatedEGObjects.at(i));
	    }
	}
      
      
      for (int i = 0; i < 18; i++)
	{
	  vector<L1CaloRegion> regions = rct->getRegions(i);
	  for (int j = 0; j < 22; j++)
	    {
	      regions.at(j).setBx(sample - preSamples);
	      rctRegions->push_back(regions.at(j));
	    }
	}
    }
  
  //putting stuff back into event
  event.put(rctEmCands);
  event.put(rctRegions);

  if (dummyE != 0) delete dummyE;
  if (dummyH != 0) delete dummyH;

}
