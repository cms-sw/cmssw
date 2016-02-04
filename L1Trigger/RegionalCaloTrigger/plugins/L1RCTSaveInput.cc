#include <memory>
#include <string>
#include <iostream>
#include <fstream>
using std::ostream;
using std::cout;
using std::cerr;
using std::endl;

#include <iomanip>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "L1Trigger/RegionalCaloTrigger/interface/L1RCTSaveInput.h"

#include "DataFormats/L1CaloTrigger/interface/L1CaloCollections.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"

#include "CondFormats/L1TObjects/interface/L1CaloHcalScale.h"
#include "CondFormats/DataRecord/interface/L1CaloHcalScaleRcd.h"
#include "CondFormats/L1TObjects/interface/L1CaloEcalScale.h"
#include "CondFormats/DataRecord/interface/L1CaloEcalScaleRcd.h"

// also include old-style scales -- same as L1RCTProducer?
#include "CalibFormats/CaloTPG/interface/CaloTPGTranscoder.h"
#include "CalibFormats/CaloTPG/interface/CaloTPGRecord.h"
#include "CalibCalorimetry/EcalTPGTools/interface/EcalTPGScale.h"

#include "CondFormats/L1TObjects/interface/L1CaloEtScale.h"
#include "CondFormats/DataRecord/interface/L1EmEtScaleRcd.h"

#include "CondFormats/L1TObjects/interface/L1RCTParameters.h"
#include "CondFormats/DataRecord/interface/L1RCTParametersRcd.h"
#include "CondFormats/L1TObjects/interface/L1RCTChannelMask.h"
#include "CondFormats/DataRecord/interface/L1RCTChannelMaskRcd.h"

#include "L1Trigger/RegionalCaloTrigger/interface/L1RCT.h"
#include "L1Trigger/RegionalCaloTrigger/interface/L1RCTLookupTables.h" 

L1RCTSaveInput::L1RCTSaveInput(const edm::ParameterSet& conf) :
  fileName(conf.getUntrackedParameter<std::string>("rctTestInputFile")),
  rctLookupTables(new L1RCTLookupTables),
  rct(new L1RCT(rctLookupTables)),
  useEcal(conf.getParameter<bool>("useEcal")),
  useHcal(conf.getParameter<bool>("useHcal")),
  ecalDigisLabel(conf.getParameter<edm::InputTag>("ecalDigisLabel")),
  hcalDigisLabel(conf.getParameter<edm::InputTag>("hcalDigisLabel")),
  useDebugTpgScales(conf.getParameter<bool>("useDebugTpgScales"))
{
  ofs.open(fileName.c_str(), std::ios::app);
  if(!ofs)
    {
      std::cerr << "Could not create " << fileName << std::endl;
      exit(1);
    }
}

L1RCTSaveInput::~L1RCTSaveInput()
{
  if(rct != 0) delete rct;
  if(rctLookupTables != 0) delete rctLookupTables;
}

void
L1RCTSaveInput::analyze(const edm::Event& event,
			const edm::EventSetup& eventSetup)
{
  edm::ESHandle<L1RCTParameters> rctParameters;
  eventSetup.get<L1RCTParametersRcd>().get(rctParameters);
  const L1RCTParameters* r = rctParameters.product();
  edm::ESHandle<L1RCTChannelMask> channelMask;
  eventSetup.get<L1RCTChannelMaskRcd>().get(channelMask);
  const L1RCTChannelMask* c = channelMask.product();
  edm::ESHandle<L1CaloEtScale> emScale;
  eventSetup.get<L1EmEtScaleRcd>().get(emScale);
  const L1CaloEtScale* s = emScale.product();

  rctLookupTables->setRCTParameters(r);
  rctLookupTables->setChannelMask(c);
  rctLookupTables->setL1CaloEtScale(s);

  // use dummies to get delete right when creating new scales from old
  L1CaloEcalScale* dummyE(0);
  L1CaloHcalScale* dummyH(0);

  if (useDebugTpgScales)
    {
      // use old-style scales
      edm::ESHandle<CaloTPGTranscoder> transcoder;
      eventSetup.get<CaloTPGRecord>().get(transcoder);
      const CaloTPGTranscoder* h_tpg = transcoder.product();

      EcalTPGScale* e_tpg = new EcalTPGScale();
      e_tpg->setEventSetup(eventSetup);

      L1CaloEcalScale* ecalScale = new L1CaloEcalScale();
      L1CaloHcalScale* hcalScale = new L1CaloHcalScale();

      // create input scales, werner's code
      // ECAL
      std::cout << "ECAL Pos " << L1CaloEcalScale::nBinRank << std::endl ;
      for( unsigned short ieta = 1 ; ieta <= L1CaloEcalScale::nBinEta; ++ieta )
        {
          for( unsigned short irank = 0 ; irank < L1CaloEcalScale::nBinRank; ++irank )
            {
              std::cout << ieta << " " << irank ;
              EcalSubdetector subdet = ( ieta <= 17 ) ? EcalBarrel : EcalEndcap ;
              double etGeVPos =
                e_tpg->getTPGInGeV
                ( irank, EcalTrigTowerDetId(1, // +ve eta
                                            subdet,
                                            ieta,
                                            1 )); // dummy phi value
              ecalScale->setBin( irank, ieta, 1, etGeVPos ) ;
	      std::cout << etGeVPos << ", " ;
            }
	  std::cout << std::endl ;
        }
      std::cout << std::endl ;

      std::cout << "ECAL Neg" << std::endl ;
      for( unsigned short ieta = 1 ; ieta <= L1CaloEcalScale::nBinEta; ++ieta )
        {
          for( unsigned short irank = 0 ; irank < L1CaloEcalScale::nBinRank; ++irank )
            {
              EcalSubdetector subdet = ( ieta <= 17 ) ? EcalBarrel : EcalEndcap ;

	      std::cout << ieta << " " << irank ;
              double etGeVNeg =
                e_tpg->getTPGInGeV
                ( irank,
                  EcalTrigTowerDetId(-1, // -ve eta
                                     subdet,
                                     ieta,
                                     2 )); // dummy phi value
              ecalScale->setBin( irank, ieta, -1, etGeVNeg ) ;
	      std::cout << etGeVNeg << ", " ;
            }
	  std::cout << std::endl ;
        }
      std::cout << std::endl ;

      // HCAL
      std::cout << "HCAL" << std::endl ;
      for( unsigned short ieta = 1 ; ieta <= L1CaloHcalScale::nBinEta; ++ieta )
        {
          for( unsigned short irank = 0 ; irank < L1CaloHcalScale::nBinRank; ++irank )
            {
              double etGeV = h_tpg->hcaletValue( ieta, irank ) ;

              hcalScale->setBin( irank, ieta, 1, etGeV ) ;
              hcalScale->setBin( irank, ieta, -1, etGeV ) ;
	      std::cout << etGeV << ", " ;
	      std::cout << std::endl ;
	    }
	  std::cout << std::endl ;
	}

      // set the input scales
      rctLookupTables->setEcalScale(ecalScale);
      rctLookupTables->setHcalScale(hcalScale);
      
      dummyE = ecalScale;
      dummyH = hcalScale;
      
      delete e_tpg;
      
    }
  else
    {
      edm::ESHandle<L1CaloHcalScale> hcalScale;
      eventSetup.get<L1CaloHcalScaleRcd>().get(hcalScale);
      const L1CaloHcalScale* h = hcalScale.product();
      edm::ESHandle<L1CaloEcalScale> ecalScale;
      eventSetup.get<L1CaloEcalScaleRcd>().get(ecalScale);
      const L1CaloEcalScale* e = ecalScale.product();

      rctLookupTables->setHcalScale(h);
      rctLookupTables->setEcalScale(e);

    }

  edm::Handle<EcalTrigPrimDigiCollection> ecal;
  edm::Handle<HcalTrigPrimDigiCollection> hcal;
  event.getByLabel(ecalDigisLabel, ecal); 
  event.getByLabel(hcalDigisLabel, hcal); 
  EcalTrigPrimDigiCollection ecalColl;
  HcalTrigPrimDigiCollection hcalColl;
  if (ecal.isValid()) { ecalColl = *ecal; }
  if (hcal.isValid()) { hcalColl = *hcal; }
  rct->digiInput(ecalColl, hcalColl);
  static int nEvents = 0;
  if(nEvents == 0)
    {
      ofs
	<< "Crate = 0-17" << std::endl
	<< "Card = 0-7 within the crate" << std::endl
	<< "Tower = 0-31 covers 4 x 8 covered by the card" << std::endl
	<< "EMAddr(0:8) = EMFGBit(0:0)+CompressedEMET(1:8)" << std::endl
	<< "HDAddr(0:8) = HDFGBit(0:0)+CompressedHDET(1:8) - note: HDFGBit(0:0) is not part of the hardware LUT address" << std::endl
	<< "LutOut(0:17)= LinearEMET(0:6)+HoEFGVetoBit(7:7)+LinearJetET(8:16)+ActivityBit(17:17)" << std::endl
	<< "Event" << "\t"
	<< "Crate" << "\t"
	<< "Card" << "\t"
	<< "Tower" << "\t"
	<< "EMAddr" << "\t"
	<< "HDAddr" << "\t"
	<< "LUTOut"
	<< std::endl;
    }
  if(nEvents < 64)
    {
      for(unsigned short iCrate = 0; iCrate < 18; iCrate++)
	{
	  for(unsigned short iCard = 0; iCard < 7; iCard++)
	    {
	      // tower numbered from 0-31
	      for(unsigned short iTower = 0; iTower < 32; iTower++)
		{
		  unsigned short ecal = rct->ecalCompressedET(iCrate, iCard, iTower);
		  unsigned short hcal = rct->hcalCompressedET(iCrate, iCard, iTower);
		  unsigned short fgbit = rct->ecalFineGrainBit(iCrate, iCard, iTower);
		  unsigned short mubit = rct->hcalFineGrainBit(iCrate, iCard, iTower);
		  unsigned long lutOutput = rctLookupTables->lookup(ecal, hcal, fgbit, iCrate, iCard, iTower);
		  ofs
		    << std::hex 
		    << nEvents << "\t"
		    << iCrate << "\t"
		    << iCard << "\t"
		    << iTower << "\t"
		    << ecal * 2 + fgbit << "\t"
		    << hcal * 2 + mubit << "\t"
		    << lutOutput
		    << std::dec 
		    << std::endl;
		}
	    }
	}
    }
  nEvents++;
}
