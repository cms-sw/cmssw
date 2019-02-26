#include "CondTools/Ecal/interface/EcalLaserHandler.h"

#include "CondTools/Ecal/interface/EcalMappingElectronicsHandler.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"

#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EcalElectronicsId.h"
#include "DataFormats/EcalDetId/interface/EcalTriggerElectronicsId.h"

#include "DataFormats/Provenance/interface/Timestamp.h"
#include<iostream>

EcalMappingElectronicsHandler::EcalMappingElectronicsHandler(const edm::ParameterSet & ps):           
  txtFileSource_(ps.getUntrackedParameter<std::string>("txtFile")),
  m_name(ps.getUntrackedParameter<std::string>("name","EcalMappingElectronicsHandler")),
  since_(ps.getUntrackedParameter<long long>("since",1)) 
{
  std::cout << "EcalMappingElectronics Source handler constructor\n" << std::endl;
}

EcalMappingElectronicsHandler::~EcalMappingElectronicsHandler()
{
}


void EcalMappingElectronicsHandler::getNewObjects()
{

  std::cout << "------- Ecal - > getNewObjects\n";
  EcalMappingElectronics *payload = new EcalMappingElectronics ;
  std::unique_ptr<EcalMappingElectronics> mapping = std::unique_ptr<EcalMappingElectronics>( new EcalMappingElectronics() );
  //Filling map reading from file 
  edm::LogInfo("EcalMappingElectronicsHandler") << "Reading mapping from file " << edm::FileInPath(txtFileSource_).fullPath().c_str() ;
  
  std::ifstream f(edm::FileInPath(txtFileSource_).fullPath().c_str());
  if (!f.good())
    {
      edm::LogError("EcalMappingElectronicsHandler") << "File not found";
      throw cms::Exception("FileNotFound");
    }
  
  // uint32_t detid, elecid, triggerid;
  
  int ix, iy, iz, CL;
  // int dccid, towerid, stripid, xtalid;
  // int tccid, tower, ipseudostrip, xtalinps;
  int dccid, towerid, pseudostrip_in_SC, xtal_in_pseudostrip;
  int tccid, tower, pseudostrip_in_TCC, pseudostrip_in_TT;
  
  while ( ! f.eof()) 
    {
      // f >> detid >> elecid >> triggerid; 
      f >> ix >> iy >> iz >> CL >> dccid >> towerid >> pseudostrip_in_SC >> xtal_in_pseudostrip >> tccid >> tower >> 
	pseudostrip_in_TCC >> pseudostrip_in_TT ;
      
      //       if (!EEDetId::validDetId(ix,iy,iz))
      // 	  continue;
      
      EEDetId detid(ix,iy,iz,EEDetId::XYMODE);
      // std::cout << " dcc tower ps_in_SC xtal_in_ps " << dccid << " " << towerid << " " << pseudostrip_in_SC << " " << xtal_in_pseudostrip << std::endl;
      EcalElectronicsId elecid(dccid,towerid, pseudostrip_in_SC, xtal_in_pseudostrip);
      // std::cout << " tcc tt ps_in_TT xtal_in_ps " << tccid << " " << tower << " " << pseudostrip_in_TT << " " << xtal_in_pseudostrip << std::endl;
      EcalTriggerElectronicsId triggerid(tccid, tower, pseudostrip_in_TT, xtal_in_pseudostrip);
      EcalMappingElement aElement;
      aElement.electronicsid = elecid.rawId();
      aElement.triggerid = triggerid.rawId();
      (*payload).setValue(detid, aElement);
    }
  
  f.close();
  edm::LogInfo("EcalMappingElectronicsHandler") << "Reading completed ready to insert in DB";  
  //Filling completed transferring to DB
  m_to_transfer.push_back(std::make_pair(payload,since_));
  //  delete payload;
}



