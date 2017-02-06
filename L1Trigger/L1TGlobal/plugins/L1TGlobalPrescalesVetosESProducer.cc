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

#include "L1Trigger/L1TCommon/interface/XmlConfigReader.h"
#include "L1Trigger/L1TCommon/interface/Mask.h"
#include "L1Trigger/L1TCommon/interface/Setting.h"
#include "L1Trigger/L1TCommon/interface/TrigSystem.h"

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
  int m_verbosity;
  int m_bx_mask_default;

  std::vector<std::vector<int> > m_initialPrescaleFactorsAlgoTrig;
  std::vector<unsigned int> m_initialTriggerMaskAlgoTrig;
  std::vector<int> m_initialTriggerMaskVetoAlgoTrig;
  std::map<int, std::vector<int> > m_initialTriggerAlgoBxMaskAlgoTrig;
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

  m_numberPhysTriggers = 512;
  
  // directory in /data/Luminosity for the trigger menu
  std::string menuDir = conf.getParameter<std::string>("TriggerMenuLuminosity");
  //std::string menuDir = "startup";

  m_verbosity = conf.getParameter<int>("Verbosity");
  m_bx_mask_default = conf.getParameter<int>("AlgoBxMaskDefault");

  // XML files
  std::string prescalesFileName = conf.getParameter<std::string>("PrescaleXMLFile");
  std::string algobxmaskFileName = conf.getParameter<std::string>("AlgoBxMaskXMLFile");
  std::string finormaskFileName = conf.getParameter<std::string>("FinOrMaskXMLFile");
  std::string vetomaskFileName = conf.getParameter<std::string>("VetoMaskXMLFile");

  // Full path
  edm::FileInPath f1_prescale("L1Trigger/L1TGlobal/data/Luminosity/" + menuDir + "/" + prescalesFileName);
  std::string m_prescaleFile = f1_prescale.fullPath();

  edm::FileInPath f1_mask_algobx("L1Trigger/L1TGlobal/data/Luminosity/" + menuDir + "/" + algobxmaskFileName);
  std::string m_algobxmaskFile = f1_mask_algobx.fullPath();
  
  edm::FileInPath f1_mask_finor("L1Trigger/L1TGlobal/data/Luminosity/" + menuDir + "/" + finormaskFileName);
  std::string m_finormaskFile = f1_mask_finor.fullPath();

  edm::FileInPath f1_mask_veto("L1Trigger/L1TGlobal/data/Luminosity/" + menuDir + "/" + vetomaskFileName);
  std::string m_vetomaskFile = f1_mask_veto.fullPath();

  // XML payloads
  std::string xmlPayload_prescale;
  std::string xmlPayload_mask_algobx;
  std::string xmlPayload_mask_finor;
  std::string xmlPayload_mask_veto;

  std::vector<std::vector<int> > prescales;
  std::vector<unsigned int> triggerMasks;
  std::vector<int> triggerVetoMasks;
  std::map<int, std::vector<int> > triggerAlgoBxMaskAlgoTrig;

  // Prescales
  std::ifstream input_prescale;
  input_prescale.open( m_prescaleFile );
  if( !input_prescale ){
    LogTrace("L1TGlobalPrescalesVetosESProducer")
      << "\nCould not find file: " << m_prescaleFile
      << "\nFilling the prescale vectors with prescale 1"
      << std::endl;

    for( int col=0; col < 1; col++ ){
      prescales.push_back(std::vector<int>());
      for( unsigned int iBit = 0; iBit < m_numberPhysTriggers; ++iBit ){
	int inputDefaultPrescale = 1;
	prescales[col].push_back(inputDefaultPrescale);
      }
    }
  }
  else {
    while( !input_prescale.eof() ){
        string tmp;
        getline( input_prescale, tmp, '\n' );
        xmlPayload_prescale.append( tmp );
    }

    XmlConfigReader xmlReader_prescale;
    l1t::TrigSystem ts_prescale;
    ts_prescale.addProcRole("uGtProcessor", "uGtProcessor");

    // run the parser 
    xmlReader_prescale.readDOMFromString( xmlPayload_prescale ); // initialize it
    xmlReader_prescale.readRootElement( ts_prescale, "uGT" ); // extract all of the relevant context
    ts_prescale.setConfigured();

    std::map<string, l1t::Setting> settings_prescale = ts_prescale.getSettings("uGtProcessor");
    std::vector<l1t::TableRow> tRow_prescale = settings_prescale["prescales"].getTableRows();

    unsigned int numColumns_prescale = 0;
    if( tRow_prescale.size()>0 ){
      std::vector<std::string> firstRow_prescale = tRow_prescale[0].getRow();
      numColumns_prescale = firstRow_prescale.size();
    }
    
    int NumPrescaleSets = numColumns_prescale - 1;
    int NumAlgos_prescale = tRow_prescale.size();

    if( NumPrescaleSets > 0 ){
      // Fill default prescale set
      for( int iSet=0; iSet<NumPrescaleSets; iSet++ ){
	prescales.push_back(std::vector<int>());
	for( int iBit = 0; iBit < NumAlgos_prescale; ++iBit ){
	  int inputDefaultPrescale = 1;
	  prescales[iSet].push_back(inputDefaultPrescale);
	}
      }

      for( auto it=tRow_prescale.begin(); it!=tRow_prescale.end(); it++ ){
	unsigned int algoBit = it->getRowValue<unsigned int>("algo/prescale-index");
	for( int iSet=0; iSet<NumPrescaleSets; iSet++ ){
	  int prescale = it->getRowValue<unsigned int>(std::to_string(iSet));
	  prescales[iSet][algoBit] = prescale;
	}
      }
    }

  }
  input_prescale.close();




  ///////////////////////////////////////////////////////////////////////////////////////////////////////////

  // finor mask
  // Setting mask to default 1 (unmask)
  for( unsigned int iAlg=0; iAlg < m_numberPhysTriggers; iAlg++ )
    triggerMasks.push_back(1);

  
  std::ifstream input_mask_finor;
  input_mask_finor.open( m_finormaskFile );
  if( !input_mask_finor ){
    LogTrace("L1TGlobalPrescalesVetosESProducer")
      << "\nCould not find file: " << m_finormaskFile
      << "\nFilling the finor mask vectors with 1 (unmask)"
      << std::endl;
  }
  else {
    while( !input_mask_finor.eof() ){
        string tmp;
        getline( input_mask_finor, tmp, '\n' );
        xmlPayload_mask_finor.append( tmp );
    }

    XmlConfigReader xmlReader_mask_finor;
    l1t::TrigSystem ts_mask_finor;
    ts_mask_finor.addProcRole("uGtProcessor", "uGtProcessor");

    // run the parser 
    xmlReader_mask_finor.readDOMFromString( xmlPayload_mask_finor ); // initialize it
    xmlReader_mask_finor.readRootElement( ts_mask_finor, "uGT" ); // extract all of the relevant context
    ts_mask_finor.setConfigured();

    std::map<string, l1t::Setting> settings_mask_finor = ts_mask_finor.getSettings("uGtProcessor");

    std::vector<l1t::TableRow> tRow_mask_finor = settings_mask_finor["finorMask"].getTableRows();

    for( auto it=tRow_mask_finor.begin(); it!=tRow_mask_finor.end(); it++ ){
      unsigned int algoBit = it->getRowValue<unsigned int>("algo");
      unsigned int mask = it->getRowValue<unsigned int>("mask");
      if( algoBit < m_numberPhysTriggers ) triggerMasks[algoBit] = mask;
    }
  }
  input_mask_finor.close();

  
  ///////////////////////////////////////////////////////////////////////////////////////////////////////////

  // veto mask
  // Setting veto mask to default 0 (no veto)
  for( unsigned int iAlg=0; iAlg < m_numberPhysTriggers; iAlg++ )
    triggerVetoMasks.push_back(0);
  
  std::ifstream input_mask_veto;
  input_mask_veto.open( m_vetomaskFile );
  if( !input_mask_veto ){
    LogTrace("L1TGlobalPrescalesVetosESProducer")
      << "\nCould not find file: " << m_vetomaskFile
      << "\nFilling the veto mask vectors with 1 (unmask)"
      << std::endl;
  }
  else {
    while( !input_mask_veto.eof() ){
        string tmp;
        getline( input_mask_veto, tmp, '\n' );
        xmlPayload_mask_veto.append( tmp );
    }

    XmlConfigReader xmlReader_mask_veto;
    l1t::TrigSystem ts_mask_veto;
    ts_mask_veto.addProcRole("uGtProcessor", "uGtProcessor");

    // run the parser 
    xmlReader_mask_veto.readDOMFromString( xmlPayload_mask_veto ); // initialize it
    xmlReader_mask_veto.readRootElement( ts_mask_veto, "uGT" ); // extract all of the relevant context
    ts_mask_veto.setConfigured();

    std::map<string, l1t::Setting> settings_mask_veto = ts_mask_veto.getSettings("uGtProcessor");
    std::vector<l1t::TableRow> tRow_mask_veto = settings_mask_veto["vetoMask"].getTableRows();

    for( auto it=tRow_mask_veto.begin(); it!=tRow_mask_veto.end(); it++ ){
      unsigned int algoBit = it->getRowValue<unsigned int>("algo");
      unsigned int veto = it->getRowValue<unsigned int>("veto");
      if( algoBit < m_numberPhysTriggers ) triggerVetoMasks[algoBit] = int(veto);
    }
  }
  input_mask_veto.close();

  ///////////////////////////////////////////////////////////////////////////////////////////////////////////

  // Algo bx mask
  std::ifstream input_mask_algobx;
  input_mask_algobx.open( m_algobxmaskFile );
  if( !input_mask_algobx ){
    LogTrace("L1TGlobalPrescalesVetosESProducer")
      << "\nCould not find file: " << m_algobxmaskFile
      << "\nNot filling map"
      << std::endl;
  }
  else {
    while( !input_mask_algobx.eof() ){
        string tmp;
        getline( input_mask_algobx, tmp, '\n' );
        xmlPayload_mask_algobx.append( tmp );
    }

    XmlConfigReader xmlReader_mask_algobx;
    l1t::TrigSystem ts_mask_algobx;
    ts_mask_algobx.addProcRole("uGtProcessor", "uGtProcessor");

    // run the parser 
    xmlReader_mask_algobx.readDOMFromString( xmlPayload_mask_algobx ); // initialize it
    xmlReader_mask_algobx.readRootElement( ts_mask_algobx, "uGT" ); // extract all of the relevant context
    ts_mask_algobx.setConfigured();

    std::map<string, l1t::Setting> settings_mask_algobx = ts_mask_algobx.getSettings("uGtProcessor");
    std::vector<l1t::TableRow> tRow_mask_algobx = settings_mask_algobx["algorithmBxMask"].getTableRows();

    unsigned int numCol_mask_algobx = 0;
    if( tRow_mask_algobx.size()>0 ){
      std::vector<std::string> firstRow_mask_algobx = tRow_mask_algobx[0].getRow();
      numCol_mask_algobx = firstRow_mask_algobx.size();
    }
    
    int NumAlgoBitsInMask = numCol_mask_algobx - 1;
    if( NumAlgoBitsInMask > 0 ){
      for( auto it=tRow_mask_algobx.begin(); it!=tRow_mask_algobx.end(); it++ ){
	int bx = it->getRowValue<unsigned int>("bx/algo");
	std::vector<int> maskedBits;
	for( int iBit=0; iBit<NumAlgoBitsInMask; iBit++ ){
	  unsigned int maskBit = it->getRowValue<unsigned int>(std::to_string(iBit));
	  if( int(maskBit)!=m_bx_mask_default ) maskedBits.push_back(iBit);
	}
	if( maskedBits.size()>0 ) triggerAlgoBxMaskAlgoTrig[bx] = maskedBits;
      }
    }

  }
  input_mask_algobx.close();


  // Set prescales to zero if masked
  for( unsigned int iSet=0; iSet < prescales.size(); iSet++ ){
    for( unsigned int iBit=0; iBit < prescales[iSet].size(); iBit++ ){
      // Add protection in case prescale table larger than trigger mask size
      if( iBit >= triggerMasks.size() ){
	    edm::LogWarning("L1TGlobal")
	      << "\nWarning: algoBit in prescale table >= triggerMasks.size() "
	      << "\nWarning: no information on masking bit or not, setting as unmasked "
	      << std::endl;
      }
      else {
	prescales[iSet][iBit] *= triggerMasks[iBit];
      }
    }
  }

  m_initialPrescaleFactorsAlgoTrig = prescales;
  m_initialTriggerMaskAlgoTrig = triggerMasks;
  m_initialTriggerMaskVetoAlgoTrig = triggerVetoMasks;
  m_initialTriggerAlgoBxMaskAlgoTrig = triggerAlgoBxMaskAlgoTrig;

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
  data_.setBxMaskDefault(m_bx_mask_default);
  data_.setPrescaleFactorTable(m_initialPrescaleFactorsAlgoTrig);
  data_.setTriggerMaskVeto(m_initialTriggerMaskVetoAlgoTrig);
  data_.setTriggerAlgoBxMask(m_initialTriggerAlgoBxMaskAlgoTrig);

  if( m_verbosity ){
    LogDebug("L1TGlobal") << " ====> Prescales table <=== " << std::endl;
    for( unsigned int ix=0; ix < m_initialPrescaleFactorsAlgoTrig.size(); ix++ ){
      LogDebug("L1TGlobal") << " Prescale set = " << ix << std::endl;
      for( unsigned int iy=0; iy < m_initialPrescaleFactorsAlgoTrig[ix].size(); iy++ ){
	LogDebug("L1TGlobal") << "\t Algo " << iy << ": " << m_initialPrescaleFactorsAlgoTrig[ix][iy] << std::endl;
      }
    }

    LogDebug("L1TGlobal") << " ====> Trigger mask veto <=== " << std::endl;
    for( unsigned int ix=0; ix < m_initialTriggerMaskVetoAlgoTrig.size(); ix++ ){
      LogDebug("L1TGlobal") << "\t Algo " << ix << ": " << m_initialTriggerMaskVetoAlgoTrig[ix] << std::endl;
    }

    LogDebug("L1TGlobal") << " ====> Algo bx mask <=== " << std::endl;
    if( m_initialTriggerAlgoBxMaskAlgoTrig.size()==0 ) LogDebug("L1TGlobal") << "\t(empty map)" << std::endl;
    for( auto& it: m_initialTriggerAlgoBxMaskAlgoTrig ){
      LogDebug("L1TGlobal") << " bx = " << it.first << " : iAlg =";
      std::vector<int> masked = it.second;
      for( unsigned int iAlg=0; iAlg < masked.size(); iAlg++ ){
	LogDebug("L1TGlobal") << " " << masked[iAlg];
      }
      LogDebug("L1TGlobal") << " " << std::endl;
    }
  }

  // write the condition format to the event setup via the helper:
  using namespace edm::es;
  boost::shared_ptr<L1TGlobalPrescalesVetos> pMenu = boost::shared_ptr< L1TGlobalPrescalesVetos >(data_.getWriteInstance());
  return pMenu;
}

//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(L1TGlobalPrescalesVetosESProducer);
