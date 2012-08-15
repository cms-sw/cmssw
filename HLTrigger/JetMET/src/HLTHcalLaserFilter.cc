// -*- C++ -*-
//
// Class:      HLTHcalLaserFilter
// 
/**\class HLTHcalLaserFilter

 Description: HLT filter module for rejecting events with HCAL laser firing

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Alex Mott
//         Created:  Wed Aug 15 10:37:03 EST 2012
//
//

#include "HLTrigger/JetMET/interface/HLTHcalLaserFilter.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include <iostream>

HLTHcalLaserFilter::HLTHcalLaserFilter(const edm::ParameterSet& iConfig) :
  hcalDigiCollection_(iConfig.getParameter<edm::InputTag>("hcalDigiCollection")),
  maxTotalCalibCharge_(iConfig.getParameter<double>("maxTotalCalibCharge")),
  maxCalibCountTS45_(iConfig.getParameter<int>("maxCalibCountTS45")),
  maxCalibCountgt15TS45_(iConfig.getParameter<int>("maxCalibCountgt15TS45")),
  maxCalibChargeTS45_(iConfig.getParameter<double>("maxCalibChargeTS45")),
  maxCalibChargegt15TS45_(iConfig.getParameter<double>("maxCalibChargegt15TS45"))
{
}


HLTHcalLaserFilter::~HLTHcalLaserFilter(){}

void
HLTHcalLaserFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("hcalDigiCollection",edm::InputTag("hltHcalDigis"));
  desc.add<double>("maxTotalCalibCharge",-1);
  desc.add<int>("maxCalibCountTS45",-1);
  desc.add<int>("maxCalibCountgt15TS45",-1);
  desc.add<double>("maxCalibChargeTS45",-1);
  desc.add<double>("maxCalibChargegt15TS45",-1);

  descriptions.add("hltHcalLaserFilter",desc);
}

//
// member functions
//

bool HLTHcalLaserFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  edm::Handle<HcalCalibDigiCollection> hCalib;
  iEvent.getByLabel("hcalDigis", hCalib);

  // Set up potential filter variables
  double totalCalibCharge=0;
  int calibCountTS45=0;
  int calibCountgt15TS45=0;
  double calibChargeTS45=0;
  double calibChargegt15TS45=0;

  const float adc2fC[128]={-0.5,0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5,10.5,11.5,12.5,
			   13.5,15.,17.,19.,21.,23.,25.,27.,29.5,32.5,35.5,38.5,42.,46.,50.,54.5,59.5,
			   64.5,59.5,64.5,69.5,74.5,79.5,84.5,89.5,94.5,99.5,104.5,109.5,114.5,119.5,
			   124.5,129.5,137.,147.,157.,167.,177.,187.,197.,209.5,224.5,239.5,254.5,272.,
			   292.,312.,334.5,359.5,384.5,359.5,384.5,409.5,434.5,459.5,484.5,509.5,534.5,
			   559.5,584.5,609.5,634.5,659.5,684.5,709.5,747.,797.,847.,897.,947.,997.,
			   1047.,1109.5,1184.5,1259.5,1334.5,1422.,1522.,1622.,1734.5,1859.5,1984.5,
			   1859.5,1984.5,2109.5,2234.5,2359.5,2484.5,2609.5,2734.5,2859.5,2984.5,
			   3109.5,3234.5,3359.5,3484.5,3609.5,3797.,4047.,4297.,4547.,4797.,5047.,
			   5297.,5609.5,5984.5,6359.5,6734.5,7172.,7672.,8172.,8734.5,9359.5,9984.5};

  if(hCalib.isValid() == true)
    {
      // loop over calibration channels
      
      //for timing reasons, we abort within the loop if a field ever goes out of bounds
      for(HcalCalibDigiCollection::const_iterator digi = hCalib->begin(); digi != hCalib->end(); digi++)
	{
	  if(digi->id().hcalSubdet() == 0)
	    continue;
	  
	  HcalCalibDetId myid=(HcalCalibDetId)digi->id();
	  if ( myid.calibFlavor()==HcalCalibDetId::HOCrosstalk)
	    continue; // ignore HOCrosstalk channels
	  
	  // Add this digi to total calibration charge
	  for(int i = 0; i < (int)digi->size(); i++)
	    totalCalibCharge = totalCalibCharge + adc2fC[digi->sample(i).adc()&0xff];

	  if(maxTotalCalibCharge_ >= 0 && totalCalibCharge > maxTotalCalibCharge_) return false;

	  // Compute charge in TS4 + TS5  
	  if (digi->size()>5)
	    {
	      double sumCharge=adc2fC[digi->sample(4).adc()&0xff]+adc2fC[digi->sample(5).adc()&0xff];
	      
	      ++calibCountTS45;
	      calibChargeTS45+=sumCharge;

	      if(maxCalibCountTS45_  >= 0 && calibCountTS45 > maxCalibCountTS45_) return false;
	      if(maxCalibChargeTS45_ >= 0 && calibChargeTS45 > maxCalibChargeTS45_) return false;

	      // Increment when sumCharge > 15 fC (other thresholds could be defined)
	      if (sumCharge>15)
		{
		  ++calibCountgt15TS45;
		  calibChargegt15TS45+=sumCharge;
		  if(maxCalibCountgt15TS45_  >= 0 && calibCountgt15TS45 > maxCalibCountgt15TS45_) return false;
		  if(maxCalibChargegt15TS45_ >= 0 && calibChargegt15TS45 > maxCalibChargegt15TS45_) return false;

		}
	    } // if (digi->size()>5)
	  
	} // loop on calibration digis:  for (HcalCalibDigiCollection::...)
    } // if (hCalib.isValid()==true)
  
  return true;
}
