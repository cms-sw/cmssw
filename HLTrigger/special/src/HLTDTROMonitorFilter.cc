/*
 *  See header file for a description of this class.
 *
 *  $Date: 2009/05/20 16:12:45 $
 *  $Revision: 1.1 $
 *  \author G. Cerminara - INFN Torino
 */

#include "HLTrigger/special/interface/HLTDTROMonitorFilter.h"

#include <FWCore/Framework/interface/Event.h>

#include <FWCore/Framework/interface/ESHandle.h>
#include <FWCore/Framework/interface/EventSetup.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>

#include <DataFormats/FEDRawData/interface/FEDRawData.h>
#include <DataFormats/FEDRawData/interface/FEDNumbering.h>
#include <EventFilter/DTRawToDigi/interface/DTDDUWords.h>

using namespace edm;


HLTDTROMonitorFilter::HLTDTROMonitorFilter(const edm::ParameterSet& pset){
  inputLabel = pset.getParameter<InputTag>("inputLabel");


}

HLTDTROMonitorFilter::~HLTDTROMonitorFilter(){}


bool HLTDTROMonitorFilter::filter(edm::Event& event, const edm::EventSetup& setup) {
  // get the raw data
  event.getByLabel(inputLabel, rawdata);
  

  // Loop over the DT FEDs
  static int FEDIDmin = FEDNumbering::getDTFEDIds().first;
  static int FEDIDMax = FEDNumbering::getDTFEDIds().second;

  // Definitions
  static const int wordSize_32 = 4;
  static const int wordSize_64 = 8;

  for (int dduID=FEDIDmin; dduID<=FEDIDMax; ++dduID) {  // loop over all feds
    const FEDRawData& feddata = rawdata->FEDData(dduID);
    const int datasize = feddata.size();    
    if (datasize){ // check the FED payload
      const unsigned int* index32 = reinterpret_cast<const unsigned int*>(feddata.data());
      const int numberOf32Words = datasize/wordSize_32;
      
      const unsigned char* index8 = reinterpret_cast<const unsigned char*>(index32);

      // Check Status Words (1 x ROS)
      for (int rosId = 0; rosId < 12; rosId++ ) {
	int wordIndex8 = numberOf32Words*wordSize_32 - 3*wordSize_64 + rosId; 
	DTDDUFirstStatusWord statusWord(index8[wordIndex8]);
	// check the error bit
	if(statusWord.errorFromROS() != 0) return true;
      }
    }
  }

  // check the event error flag 
  return false;
}
