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

#include <iostream>

HLTHcalLaserFilter::HLTHcalLaserFilter(const edm::ParameterSet& iConfig)
    : m_theCalibToken(consumes(iConfig.getParameter<edm::InputTag>("hcalDigiCollection"))),
      timeSlices_(iConfig.getParameter<std::vector<int> >("timeSlices")),
      thresholdsfC_(iConfig.getParameter<std::vector<double> >("thresholdsfC")),
      CalibCountFilterValues_(iConfig.getParameter<std::vector<int> >("CalibCountFilterValues")),
      CalibChargeFilterValues_(iConfig.getParameter<std::vector<double> >("CalibChargeFilterValues")),
      maxTotalCalibCharge_(iConfig.getParameter<double>("maxTotalCalibCharge")),
      maxAllowedHFcalib_(iConfig.getParameter<int>("maxAllowedHFcalib"))

{}

void HLTHcalLaserFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("hcalDigiCollection", edm::InputTag("hltHcalDigis"));
  desc.add<double>("maxTotalCalibCharge", -1);

  std::vector<int> dummy_vint;
  std::vector<double> dummy_vdouble;

  desc.add<std::vector<int> >("timeSlices", dummy_vint);
  desc.add<std::vector<double> >("thresholdsfC", dummy_vdouble);
  desc.add<std::vector<int> >("CalibCountFilterValues", dummy_vint);
  desc.add<std::vector<double> >("CalibChargeFilterValues", dummy_vdouble);
  desc.add<int>("maxAllowedHFcalib", -1);
  descriptions.add("hltHcalLaserFilter", desc);
}

//
// member functions
//

bool HLTHcalLaserFilter::filter(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  edm::Handle<HcalCalibDigiCollection> hCalib;
  iEvent.getByToken(m_theCalibToken, hCalib);

  int numHFcalib = 0;

  // Set up potential filter variables
  double totalCalibCharge = 0;

  // Track multiplicity and total charge for each fC threshold
  std::vector<int> CalibCount;
  std::vector<double> CalibCharge;
  for (unsigned int i = 0; i < thresholdsfC_.size(); ++i) {
    CalibCount.push_back(0);
    CalibCharge.push_back(0);
  }

  static constexpr float adc2fC[128] = {
      -0.5,   0.5,    1.5,    2.5,    3.5,    4.5,    5.5,    6.5,    7.5,    8.5,    9.5,    10.5,   11.5,
      12.5,   13.5,   15.,    17.,    19.,    21.,    23.,    25.,    27.,    29.5,   32.5,   35.5,   38.5,
      42.,    46.,    50.,    54.5,   59.5,   64.5,   59.5,   64.5,   69.5,   74.5,   79.5,   84.5,   89.5,
      94.5,   99.5,   104.5,  109.5,  114.5,  119.5,  124.5,  129.5,  137.,   147.,   157.,   167.,   177.,
      187.,   197.,   209.5,  224.5,  239.5,  254.5,  272.,   292.,   312.,   334.5,  359.5,  384.5,  359.5,
      384.5,  409.5,  434.5,  459.5,  484.5,  509.5,  534.5,  559.5,  584.5,  609.5,  634.5,  659.5,  684.5,
      709.5,  747.,   797.,   847.,   897.,   947.,   997.,   1047.,  1109.5, 1184.5, 1259.5, 1334.5, 1422.,
      1522.,  1622.,  1734.5, 1859.5, 1984.5, 1859.5, 1984.5, 2109.5, 2234.5, 2359.5, 2484.5, 2609.5, 2734.5,
      2859.5, 2984.5, 3109.5, 3234.5, 3359.5, 3484.5, 3609.5, 3797.,  4047.,  4297.,  4547.,  4797.,  5047.,
      5297.,  5609.5, 5984.5, 6359.5, 6734.5, 7172.,  7672.,  8172.,  8734.5, 9359.5, 9984.5};

  if (hCalib.isValid() == true) {
    // loop over calibration channels

    //for timing reasons, we abort within the loop if a field ever goes out of bounds
    for (auto const& digi : *hCalib) {
      if (digi.id().hcalSubdet() == 0)
        continue;

      HcalCalibDetId myid = (HcalCalibDetId)digi.id();
      if (myid.hcalSubdet() == HcalBarrel || myid.hcalSubdet() == HcalEndcap) {
        if (myid.calibFlavor() == HcalCalibDetId::HOCrosstalk)
          continue;  // ignore HOCrosstalk channels

        // Add this digi to total calibration charge
        for (int i = 0; i < (int)digi.size(); i++)
          totalCalibCharge = totalCalibCharge + adc2fC[digi.sample(i).adc() & 0xff];

        if (maxTotalCalibCharge_ >= 0 && totalCalibCharge > maxTotalCalibCharge_)
          return false;

        // Compute total charge found in the provided subset of timeslices
        double sumCharge = 0;
        unsigned int NTS = timeSlices_.size();
        int digisize = (int)digi.size();  // gives value of largest time slice

        for (unsigned int ts = 0; ts < NTS; ++ts)  // loop over provided timeslices
        {
          if (timeSlices_[ts] < 0 || timeSlices_[ts] > digisize)
            continue;
          sumCharge += adc2fC[digi.sample(timeSlices_[ts]).adc() & 0xff];
        }

        // Check multiplicity and charge against filter settings for each charge threshold
        for (unsigned int thresh = 0; thresh < thresholdsfC_.size(); ++thresh) {
          if (sumCharge > thresholdsfC_[thresh]) {
            ++CalibCount[thresh];
            CalibCharge[thresh] += sumCharge;
            // FilterValues must be >=0 in order for filter to be applied
            if (CalibCount[thresh] >= CalibCountFilterValues_[thresh] && CalibCountFilterValues_[thresh] >= 0) {
              //std::cout <<"Number of channels > "<<thresholdsfC_[thresh]<<" = "<<CalibCount[thresh]<<"; vetoing!"<<std::endl;
              return false;
            }
            if (CalibCharge[thresh] >= CalibChargeFilterValues_[thresh] && CalibChargeFilterValues_[thresh] >= 0) {
              //std::cout <<"FILTERED BY HBHE"<<std::endl;
              return false;
            }
          }  //if (sumCharge > thresholdsfC_[thresh])
        }    //for (unsigned int thresh=0;thresh<thresholdsfC_.size();++thresh)
      }      // if HB or HE Calib
      else if (myid.hcalSubdet() == HcalForward && maxAllowedHFcalib_ >= 0) {
        ++numHFcalib;
        //std::cout <<"numHFcalib = "<<numHFcalib<<"  Max allowed = "<<maxAllowedHFcalib_<<std::endl;
        if (numHFcalib > maxAllowedHFcalib_) {
          //std::cout <<"FILTERED BY HF; "<<maxAllowedHFcalib_<<std::endl;
          return false;
        }
      }
    }  // loop on calibration digis:  for (HcalCalibDigiCollection::...)

    /*
      for (unsigned int thresh=0;thresh<thresholdsfC_.size();++thresh)
	{
	  std::cout <<"Thresh = "<<thresholdsfC_[thresh]<<"  Num channels found = "<<CalibCount[thresh]<<std::endl;
	}
      */
  }  // if (hCalib.isValid()==true)
  //std::cout <<"UNFILTERED"<<std::endl;
  return true;
}
