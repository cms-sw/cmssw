///
/// \class L1TGlobalPrescalesVetosESProducer
///
/// Description: Produces L1T Trigger Menu Condition Format
///
/// Implementation:
///    Dummy producer for L1T uGT Trigger Menu
///


// system include files
#include <memory>
#include "boost/shared_ptr.hpp"
#include <iostream>
#include <fstream>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>

#include "tmEventSetup/tmEventSetup.hh"
#include "tmEventSetup/esTriggerMenu.hh"
#include "tmEventSetup/esAlgorithm.hh"
#include "tmEventSetup/esCondition.hh"
#include "tmEventSetup/esObject.hh"
#include "tmEventSetup/esCut.hh"
#include "tmEventSetup/esScale.hh"
#include "tmGrammar/Algorithm.hh"

#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESProducts.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

#include "L1Trigger/L1TGlobal/interface/PrescalesVetosHelper.h"

#include "CondFormats/L1TObjects/interface/L1TGlobalPrescalesVetos.h"
#include "CondFormats/DataRecord/interface/L1TGlobalPrescalesVetosRcd.h"

using namespace std;
using namespace edm;
using namespace l1t;
//
// class declaration
//

class L1TGlobalPrescalesVetosESProducer : public edm::ESProducer {
public:
  L1TGlobalPrescalesVetosESProducer(const edm::ParameterSet&);
  ~L1TGlobalPrescalesVetosESProducer();

  typedef boost::shared_ptr<L1TGlobalPrescalesVetos> ReturnType;

  ReturnType produce(const L1TGlobalPrescalesVetosRcd&);

private:

  PrescalesVetosHelper data_;

  unsigned int m_numberPhysTriggers;
  std::string m_prescalesFile;

  std::vector<std::vector<int> > m_initialPrescaleFactorsAlgoTrig;
  std::vector<unsigned int> m_initialTriggerMaskAlgoTrig;
  std::vector<int> m_initialTriggerMaskVetoAlgoTrig;
  
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
L1TGlobalPrescalesVetosESProducer::L1TGlobalPrescalesVetosESProducer(const edm::ParameterSet& conf) :
  data_(new L1TGlobalPrescalesVetos())
{
  //the following line is needed to tell the framework what
  // data is being produced
  setWhatProduced(this);
  //setWhatProduced(this, conf.getParameter<std::string>("label"));


  // directory in /data/Luminosity for the trigger menu
  std::string menuDir = conf.getParameter<std::string>("TriggerMenuLuminosity");
  //std::string menuDir = "startup";

  // prescale CSV file
  std::string prescaleFileName = conf.getParameter<std::string>("PrescaleCSVFile");

  edm::FileInPath f1("L1Trigger/L1TGlobal/data/Luminosity/" +
		     menuDir + "/" + prescaleFileName);

  m_prescalesFile = f1.fullPath();

  unsigned int temp_numberPhysTriggers = 512;
 
  // Get prescale factors from CSV file for now
  std::fstream inputPrescaleFile;
  inputPrescaleFile.open(m_prescalesFile);

  std::vector<std::vector<int> > vec;
  std::vector<std::vector<int> > prescale_vec;

  std::vector<unsigned int> temp_triggerMask;
  std::vector<int> temp_triggerVetoMask;

  if( inputPrescaleFile ){
    std::string prefix1("#");
    std::string prefix2("-1");

    std::string line; 

    bool first = true;

    while( getline(inputPrescaleFile,line) ){

      if( !line.compare(0, prefix1.size(), prefix1) ) continue;
      //if( !line.compare(0, prefix2.size(), prefix2) ) continue;

      istringstream split(line);
      int value;
      int col = 0;
      char sep;

      while( split >> value ){
	if( first ){
	  // Each new value read on line 1 should create a new inner vector
	  vec.push_back(std::vector<int>());
	}

	vec[col].push_back(value);
	++col;

	// read past the separator
	split>>sep;
      }

      // Finished reading line 1 and creating as many inner
      // vectors as required
      first = false;
    }


    int NumPrescaleSets = 0;

    int maskColumn = -1;
    int maskVetoColumn = -1;
    for( int iCol=0; iCol<int(vec.size()); iCol++ ){
      if( vec[iCol].size() > 0 ){
	int firstRow = vec[iCol][0];

	if( firstRow > 0 ) NumPrescaleSets++;
	else if( firstRow==-2 ) maskColumn = iCol;
	else if( firstRow==-3 ) maskVetoColumn = iCol;
      }
    }

    // Fill default values for mask and veto mask
    for( unsigned int iBit = 0; iBit < temp_numberPhysTriggers; ++iBit ){
      unsigned int inputDefaultMask = 1;
      unsigned int inputDefaultVetoMask = 0;
      temp_triggerMask.push_back(inputDefaultMask);
      temp_triggerVetoMask.push_back(inputDefaultVetoMask);
    }

    // Fill non-trivial mask and veto mask
    if( maskColumn>=0 || maskVetoColumn>=0 ){
      for( int iBit=1; iBit<int(vec[0].size()); iBit++ ){
	unsigned int algoBit = vec[0][iBit];
	// algoBit must be less than the number of triggers
	if( algoBit < temp_numberPhysTriggers ){
	  if( maskColumn>=0 ){
	    unsigned int triggerMask = vec[maskColumn][iBit];
	    temp_triggerMask[algoBit] = triggerMask;
	  }
	  if( maskVetoColumn>=0 ){
	    unsigned int triggerVetoMask = vec[maskVetoColumn][iBit];
	    temp_triggerVetoMask[algoBit] = triggerVetoMask;
	  }
	}
      }
    }


    if( NumPrescaleSets > 0 ){
      // Fill default prescale set
      for( int iSet=0; iSet<NumPrescaleSets; iSet++ ){
	prescale_vec.push_back(std::vector<int>());
	for( unsigned int iBit = 0; iBit < temp_numberPhysTriggers; ++iBit ){
	  int inputDefaultPrescale = 1;
	  prescale_vec[iSet].push_back(inputDefaultPrescale);
	}
      }

      // Fill non-trivial prescale set
      for( int iBit=1; iBit<int(vec[0].size()); iBit++ ){
	unsigned int algoBit = vec[0][iBit];
	// algoBit must be less than the number of triggers
	if( algoBit < temp_numberPhysTriggers ){
	  for( int iSet=0; iSet<int(vec.size()); iSet++ ){
	    int useSet = -1;
	    if( vec[iSet].size() > 0 ){
	      useSet = vec[iSet][0];
	    }
	    useSet -= 1;
	      
	    if( useSet<0 ) continue;

	    int prescale = vec[iSet][iBit];
	    prescale_vec[useSet][algoBit] = prescale;
	  }
	}
	else{
	  LogTrace("L1TGlobalPrescalesVetosESProducer")
	    << "\nPrescale file has algo bit: " << algoBit
	    << "\nThis is larger than the number of triggers: " << m_numberPhysTriggers
	    << "\nSomething is wrong. Ignoring."
	    << std::endl;
	}
      }
    }

  }
  else {
    LogTrace("L1TGlobalPrescalesVetosESProducer")
      << "\nCould not find file: " << m_prescalesFile
      << "\nFilling the prescale vectors with prescale 1"
      << "\nSetting prescale set to 1"
      << std::endl;

    for( int col=0; col < 1; col++ ){
      prescale_vec.push_back(std::vector<int>());
      for( unsigned int iBit = 0; iBit < temp_numberPhysTriggers; ++iBit ){
	int inputDefaultPrescale = 1;
	prescale_vec[col].push_back(inputDefaultPrescale);
      }
    }
  }

  inputPrescaleFile.close();

  m_initialPrescaleFactorsAlgoTrig = prescale_vec;
  m_initialTriggerMaskAlgoTrig = temp_triggerMask;
  m_initialTriggerMaskVetoAlgoTrig = temp_triggerVetoMask;

}


L1TGlobalPrescalesVetosESProducer::~L1TGlobalPrescalesVetosESProducer()
{

   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
L1TGlobalPrescalesVetosESProducer::ReturnType
L1TGlobalPrescalesVetosESProducer::produce(const L1TGlobalPrescalesVetosRcd& iRecord)
{
  
  // configure the helper class parameters via its set funtions, e.g.:
  data_.setBxMaskDefault(0);
  data_.setPrescaleFactorTable(m_initialPrescaleFactorsAlgoTrig);
  data_.setTriggerMaskVeto(m_initialTriggerMaskVetoAlgoTrig);

  // write the condition format to the event setup via the helper:
  using namespace edm::es;
  boost::shared_ptr<L1TGlobalPrescalesVetos> pMenu = boost::shared_ptr< L1TGlobalPrescalesVetos >(data_.getWriteInstance());
  return pMenu;
}

//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(L1TGlobalPrescalesVetosESProducer);
