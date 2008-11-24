
/*
 *  See header file for a description of this class.
 *
 *  $Date: $
 *  $Revision: $
 *  \author G. Cerminara - INFN Torino
 */

#include "DQM/DTMonitorModule/interface/DTTimeEvolutionHisto.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace std;
using namespace edm;




DTTimeEvolutionHisto::DTTimeEvolutionHisto(DQMStore *dbe, const string& name,
					   const string& title,
					   int nbins,
					   int lsPrescale,
					   bool sliding,
					   int mode) : valueLastTimeSlot(0),
						       nEventsInLastTimeSlot(0),
						       theLSPrescale(lsPrescale),
						       doSlide(sliding),
						       nLSinTimeSlot(0),
						       firstLSinTimeSlot(0),
						       theMode(mode) {
  // set the number of bins to be booked
  nBookedBins = nbins;
  if(sliding) nBookedBins++;
  if(!sliding && theMode == 0)
    LogWarning("DTDQM|DTMonitorModule|DTMonitorClient|DTTimeEvolutionHisto")
      << "[DTTimeEvolutionHisto]***Error: wrong configuration" << endl;


  stringstream realTitle; realTitle << title << "/" <<  theLSPrescale << " LS";

  // book the ME
  histo = dbe->book1D(name, realTitle.str(), nBookedBins, 0, nBookedBins);

  // set the axis label
  if(sliding) {
    histo->setBinLabel(1,"avg. previous",1);
  } else {
    // loop over bins and 
    for(int bin =1; bin != nBookedBins; ++bin) {    
      stringstream label; label << "LS " << ((bin-1)*theLSPrescale)+1 << "-" << bin*theLSPrescale;
      histo->setBinLabel(bin, label.str(),1);

    }

  }

}




DTTimeEvolutionHisto::DTTimeEvolutionHisto(DQMStore *dbe, const string& name) : valueLastTimeSlot(0),
										nEventsInLastTimeSlot(0),
										theLSPrescale(-1),
										doSlide(false),
										nLSinTimeSlot(0),
										firstLSinTimeSlot(0),
										theMode(0) { // FIXME: set other memebers to sensible values
  LogVerbatim("DTDQM|DTMonitorModule|DTMonitorClient|DTTimeEvolutionHisto")
    << "[DTTimeEvolutionHisto] Retrieve ME with name: " << name << endl;
  histo = dbe->get(name);
}


DTTimeEvolutionHisto::~DTTimeEvolutionHisto(){}



void DTTimeEvolutionHisto::setTimeSlotValue(float value, int timeSlot) {
  //   LogVerbatim("DTDQM|DTMonitorModule|DTMonitorClient|DTTimeEvolutionHisto")
  //   << "[DTTimeEvolutionHisto] ME name: " <<  histo->getName() << endl;
  if(!doSlide) {
    //     LogVerbatim("DTDQM|DTMonitorModule|DTMonitorClient|DTTimeEvolutionHisto")
    //     << "        fill bin: " << timeSlot << " with value: " << value << endl;
    histo->Fill(timeSlot,value);
  } else {
    for(int bin = 1; bin != nBookedBins; ++bin) {
      float value = histo->getBinContent(bin);
      //       LogVerbatim("DTDQM|DTMonitorModule|DTMonitorClient|DTTimeEvolutionHisto")
      //  << "        bin: " << bin << " has value: " << value << endl;
      if(bin == 1) { // average of previous time slots (fixme)
	histo->setBinContent(bin, (value + histo->getBinContent(bin+1))/2.);
      } else if(bin != nBookedBins) {
	histo->setBinContent(bin, histo->getBinContent(bin+1));
	histo->setBinLabel(bin, histo->getTH1F()->GetXaxis()->GetBinLabel(bin+1),1);
      }
    }
    histo->setBinContent(nBookedBins, value);
  }
}


void DTTimeEvolutionHisto::accumulateValueTimeSlot(float value) {
  valueLastTimeSlot += value;
}



void DTTimeEvolutionHisto::updateTimeSlot(int ls, int nEventsInLS) {
  
  if(doSlide) { // sliding bins
    // count LS in this time-slot
    nLSinTimeSlot++;
    nEventsInLastTimeSlot += nEventsInLS;
    
    if(nLSinTimeSlot == 1) firstLSinTimeSlot = ls;

    if(nLSinTimeSlot%theLSPrescale ==0) { // update the value of the slot and reset the counters
      LogVerbatim("DTDQM|DTMonitorModule|DTMonitorClient|DTTimeEvolutionHisto")
	<< "[DTTimeEvolutionHisto] Update time-slot, # entries: " << valueLastTimeSlot 
	<< " # events: " << nEventsInLastTimeSlot << endl;
      // set the bin content
      float value = 0;
      if(theMode == 0) {
	if(nEventsInLastTimeSlot != 0) value = valueLastTimeSlot/(float)nEventsInLastTimeSlot;
      } else if(theMode == 1) {
	value = valueLastTimeSlot;
      } else if(theMode == 2) {
	value = nEventsInLastTimeSlot;
      }
      setTimeSlotValue(value, nBookedBins);
      LogVerbatim("DTDQM|DTMonitorModule|DTMonitorClient|DTTimeEvolutionHisto")
	<< "       updated value: " << histo->getBinContent(nBookedBins) << endl;

      // set the bin label
      stringstream binLabel; binLabel << "LS " << firstLSinTimeSlot;
      if(nLSinTimeSlot > 1) binLabel << "-" << ls;
      histo->setBinLabel(nBookedBins,binLabel.str(),1);

      // reset the counters for the time slot
      nLSinTimeSlot = 0; 
      nEventsInLastTimeSlot = 0;
      valueLastTimeSlot = 0;
    }


  } else {
    int binN = (int)ls/(int)theLSPrescale;
    // set the bin content
    float value = 0;
    if(theMode == 1) {
      value = valueLastTimeSlot;
    } else if(theMode == 2) {
      value = nEventsInLS;
    }
    LogVerbatim("DTDQM|DTMonitorModule|DTMonitorClient|DTTimeEvolutionHisto")
      << "[DTTimeEvolutionHisto] Update time-slot: "<< binN << " with value: " << value << endl;
    setTimeSlotValue(value,binN);
  }
}




void DTTimeEvolutionHisto::normalizeTo(const MonitorElement *histForNorm) {
  if(histo == 0) {
    LogWarning("DTDQM|DTMonitorModule|DTMonitorClient|DTTimeEvolutionHisto")
      << "[DTTimeEvolutionHisto]***Error: pointer to ME is NULL" << endl;
    return;
  }
  int nBins = histo->getNbinsX();
  if(histForNorm->getNbinsX() != nBins) {
    LogWarning("DTDQM|DTMonitorModule|DTMonitorClient|DTTimeEvolutionHisto")
      << "[DTTimeEvolutionHisto]***Error: normalizing histos with != # of bins" << endl;
    return;
  }
  for(int bin = 1; bin <= nBins; ++bin) { // loop over bins
    if(histForNorm->getBinContent(bin) != 0) {
      double normValue = histo->getBinContent(bin)/histForNorm->getBinContent(bin);
      LogVerbatim("DTDQM|DTMonitorModule|DTMonitorClient|DTTimeEvolutionHisto")
	<< "[DTTimeEvolutionHisto] Normalizing bin: " << bin << " to: " <<  histo->getBinContent(bin) << " / " << histForNorm->getBinContent(bin)
	<< " = " << normValue << endl;
      histo->setBinContent(bin, normValue);
    } else {
      histo->setBinContent(bin, 0);
    }
  }
}
