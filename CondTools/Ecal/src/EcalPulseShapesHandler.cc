#include "CondTools/Ecal/interface/EcalPulseShapesHandler.h"

#include <algorithm> 
#include<iostream>
#include<fstream>

popcon::EcalPulseShapesHandler::EcalPulseShapesHandler(const edm::ParameterSet & ps)
  :    m_name(ps.getUntrackedParameter<std::string>("name","EcalPulseShapesHandler")) {

	std::cout << "EcalPulseShape Source handler constructor\n" << std::endl;
        m_firstRun=static_cast<unsigned int>(atoi( ps.getParameter<std::string>("firstRun").c_str()));
        m_filename=ps.getParameter<std::string>("inputFileName");
        m_EBPulseShapeTemplate=ps.getParameter<std::vector<double> >("EBPulseShapeTemplate");
        m_EEPulseShapeTemplate=ps.getParameter<std::vector<double> >("EEPulseShapeTemplate");
        
}

popcon::EcalPulseShapesHandler::~EcalPulseShapesHandler()
{
}


bool popcon::EcalPulseShapesHandler::checkPulseShape( EcalPulseShapes::Item* item ){
  // true means all is standard and OK
  bool result=true;
  for(int s=0; s<EcalPulseShape::TEMPLATESAMPLES; ++s) {
    if(item->pdfval[s] > 1 || item->pdfval[s] < 0) result=false;
  }
  return result; 
}

void popcon::EcalPulseShapesHandler::fillSimPulseShape( EcalPulseShapes::Item* item, bool isbarrel ){ 
  for(int s=0; s<EcalPulseShape::TEMPLATESAMPLES; ++s) {
    item->pdfval[s] = isbarrel ? m_EBPulseShapeTemplate[s] : m_EEPulseShapeTemplate[s];
  }
}

void popcon::EcalPulseShapesHandler::getNewObjects()
{

  std::cout << "------- Ecal - > getNewObjects\n";

  // create the object pukse shapes
  EcalPulseShapes* pulseshapes = new EcalPulseShapes();

  // read the templates from a text file
  std::ifstream inputfile;
  inputfile.open(m_filename.c_str());
  float templatevals[EcalPulseShape::TEMPLATESAMPLES];
  unsigned int rawId;
  int isbarrel;
  std::string line;

  // keep track of bad crystals
  int nEBbad(0), nEEbad(0);
  int nEBgood(0), nEEgood(0);
  std::vector<EBDetId> ebgood;
  std::vector<EEDetId> eegood;

  // fill with the measured shapes only for data
  if(m_firstRun > 1) {
    while (std::getline(inputfile, line)) {
      std::istringstream linereader(line);
      linereader >> isbarrel >> rawId;
      // std::cout << "Inserting template for crystal with rawId = " << rawId << " (isbarrel = " << isbarrel << ") " << std::endl;
      for(int s=0; s<EcalPulseShape::TEMPLATESAMPLES; ++s) {
        linereader >> templatevals[s];
        // std::cout << templatevals[s] << "\t";
      }
      // std::cout << std::endl;

      if (!linereader) {
        std::cout << "Wrong format of the text file. Exit." << std::endl;
        return;
      }
      EcalPulseShapes::Item item;
      for(int s=0; s<EcalPulseShape::TEMPLATESAMPLES; ++s) item.pdfval[s] = templatevals[s];
    
      if(isbarrel) {
        EBDetId ebdetid(rawId);
        if(!checkPulseShape(&item) ) nEBbad++;
        else {
          nEBgood++;
          ebgood.push_back(ebdetid);
          pulseshapes->insert(std::make_pair(ebdetid.rawId(),item));
        }
      } else {
        EEDetId eedetid(rawId);
        if(!checkPulseShape(&item) ) nEEbad++;
        else {
          nEEgood++;
          eegood.push_back(eedetid);
          pulseshapes->insert(std::make_pair(eedetid.rawId(),item));      
        }
      }
    }
  }    

  // now fill the bad crystals and simulation with the simulation values (from TB)
  std::cout << "Filled the DB with the good measured ECAL templates. Now filling the others with the TB values" << std::endl;
  for(int iEta=-EBDetId::MAX_IETA; iEta<=EBDetId::MAX_IETA ;++iEta) {
    if(iEta==0) continue;
    for(int iPhi=EBDetId::MIN_IPHI; iPhi<=EBDetId::MAX_IPHI; ++iPhi) {
      if (EBDetId::validDetId(iEta,iPhi)) {
        EBDetId ebdetid(iEta,iPhi,EBDetId::ETAPHIMODE);

        std::vector<EBDetId>::iterator it = find(ebgood.begin(),ebgood.end(),ebdetid);
        if(it == ebgood.end()) {
          EcalPulseShapes::Item item;
          fillSimPulseShape(&item,true);
          pulseshapes->insert(std::make_pair(ebdetid.rawId(),item));        
        }
      }
    }
  }
 
  for(int iZ=-1; iZ<2; iZ+=2) {
    for(int iX=EEDetId::IX_MIN; iX<=EEDetId::IX_MAX ;++iX) {
      for(int iY=EEDetId::IY_MIN; iY<=EEDetId::IY_MAX; ++iY) {
        if (EEDetId::validDetId(iX,iY,iZ)) {
          EEDetId eedetid(iX,iY,iZ);
          
          std::vector<EEDetId>::iterator it = find(eegood.begin(),eegood.end(),eedetid);
          if(it == eegood.end()) {
            EcalPulseShapes::Item item;
            fillSimPulseShape(&item,false);
            pulseshapes->insert(std::make_pair(eedetid.rawId(),item));
          }
        }
      }
    }
  }

  std::cout << "Inserted the pulse shapes into the new item object" << std::endl;
  
  unsigned int irun=m_firstRun;
  Time_t snc= (Time_t) irun ;
  
  m_to_transfer.push_back(std::make_pair((EcalPulseShapes*)pulseshapes,snc));
  
  std::cout << "Ecal - > end of getNewObjects -----------" << std::endl;
  std::cout << "N. bad shapes for EB = " << nEBbad << std::endl;
  std::cout << "N. bad shapes for EE = " << nEEbad << std::endl;
  std::cout << "Written the object" << std::endl;
  
}
