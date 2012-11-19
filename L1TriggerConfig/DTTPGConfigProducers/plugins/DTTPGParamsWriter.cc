/*
 *  See header file for a description of this class.
 *
 *  $Date: 2011/09/21 07:44:17 $
 *  $Revision: 1.3 $
 *  \author C. Battilana CIEMAT
 */

#include "L1TriggerConfig/DTTPGConfigProducers/plugins/DTTPGParamsWriter.h"
#include "CalibMuon/DTCalibration/interface/DTCalibDBUtils.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"

/* C++ Headers */
#include <vector> 
#include <iostream>
#include <string>
#include <sstream>
#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string_regex.hpp>

using namespace std;
using namespace edm;


// Constructor
DTTPGParamsWriter::DTTPGParamsWriter(const ParameterSet& pset) {

  debug_ = pset.getUntrackedParameter<bool>("debug", false);
  inputFileName_ = pset.getUntrackedParameter<string>("inputFile");
  // Create the object to be written to DB
  phaseMap_ = new DTTPGParameters();
  
  if(debug_)
    cout << "[DTTPGParamsWriter]Constructor called!" << endl;

}



// Destructor
DTTPGParamsWriter::~DTTPGParamsWriter(){

  if(debug_)
    cout << "[DTTPGParamsWriter]Destructor called!" << endl;

}

// Do the job
void DTTPGParamsWriter::analyze(const Event & event, const EventSetup& eventSetup) {

  if(debug_)
    cout << "[DTTPGParamsWriter]Reading data from file." << endl;

  std::ifstream inputFile_(inputFileName_.c_str());  
  int nLines=0;
  std::string line;

  while(std::getline(inputFile_, line)) {
    DTChamberId chId;
    float fine;
    int coarse;
    pharseLine(line,chId,fine,coarse);
    phaseMap_->set(chId,coarse,fine,DTTimeUnits::ns);
    if (debug_) {
      float fineDB;
      int coarseDB;
      phaseMap_->get(chId,coarseDB,fineDB,DTTimeUnits::ns);
      std::cout << "[DTTPGParamsWriter] Read data for chamber " << chId 
		<< ". File params -> fine: " << fine << " coarse: " << coarse 
		<< ". DB params -> fine: " << fineDB << " coarse: " << coarseDB << std::endl;
    }
    nLines++;
  }
  if (debug_) {
    std::cout << "[DTTPGParamsWriter] # of entries written the the DB: " << nLines << std::endl;
  }
  if (nLines!=250) {
    std::cout << "[DTTPGParamsWriter] # of DB entries != 250. Check you input file!" << std::endl;
  }
  

  inputFile_.close();


}

void DTTPGParamsWriter::pharseLine(std::string &line, DTChamberId& chId, float &fine, int  &coarse) {

  std::vector<std::string> elements;
  boost::algorithm::split(elements,line,boost::algorithm::is_any_of(string(" \t\n")));  // making string conversion explicit (needed to cope with -Warray-bounds in slc5_ia32_gcc434  
  if (elements.size() != 5) {
    throw cms::Exception("DTTPGParamsWriter") << "wrong number of entries in line : " << line << " pleas check your input file syntax!";
  } else {
    chId   = DTChamberId(atoi(elements[0].c_str()),atoi(elements[1].c_str()),atoi(elements[2].c_str()));
    fine   = atof(elements[3].c_str());
    coarse = atoi(elements[4].c_str());
  }

}

// Write objects to DB
void DTTPGParamsWriter::endJob() {
  if(debug_) 
	cout << "[DTTPGParamsWriter] Writing ttrig object to DB!" << endl;

  string delayRecord = "DTTPGParametersRcd";
  DTCalibDBUtils::writeToDB(delayRecord, phaseMap_);

}  
