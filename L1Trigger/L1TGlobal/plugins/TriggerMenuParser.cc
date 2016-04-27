/**
 * \class TriggerMenuParser
 *
 *
 * Description: Xerces-C XML parser for the L1 Trigger menu.
 *
 * Implementation:
 *    <TODO: enter implementation details>
 *
 * \author: Vasile Mihai Ghete - HEPHY Vienna
 *
 * $Date$
 * $Revision$
 *
 */

// this class header
#include "TriggerMenuParser.h"

// system include files
#include <string>
#include <vector>

#include <iostream>
#include <fstream>
#include <iomanip>

#include <boost/cstdint.hpp>

#include "L1Trigger/L1TGlobal/interface/GlobalCondition.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/MessageLogger/interface/MessageDrop.h"

#include "tmEventSetup/tmEventSetup.hh"
#include "tmEventSetup/esTriggerMenu.hh"
#include "tmEventSetup/esAlgorithm.hh"
#include "tmEventSetup/esCondition.hh"
#include "tmEventSetup/esObject.hh"
#include "tmEventSetup/esCut.hh"
#include "tmEventSetup/esScale.hh"
#include "tmGrammar/Algorithm.hh"



// constructor
l1t::TriggerMenuParser::TriggerMenuParser() :
    m_triggerMenuInterface("NULL"),
    m_triggerMenuName("NULL"), m_triggerMenuImplementation(0x0), m_scaleDbKey("NULL")

{

    // menu names, scale key initialized to NULL due to ORACLE treatment of strings

    // empty

}

// destructor
l1t::TriggerMenuParser::~TriggerMenuParser() {

    clearMaps();

}

// set the number of condition chips in GTL
void l1t::TriggerMenuParser::setGtNumberConditionChips(
    const unsigned int& numberConditionChipsValue) {

    m_numberConditionChips = numberConditionChipsValue;

}

// set the number of pins on the GTL condition chips
void l1t::TriggerMenuParser::setGtPinsOnConditionChip(const unsigned int& pinsOnConditionChipValue) {

    m_pinsOnConditionChip = pinsOnConditionChipValue;

}

// set the correspondence "condition chip - GTL algorithm word"
// in the hardware
void l1t::TriggerMenuParser::setGtOrderConditionChip(
    const std::vector<int>& orderConditionChipValue) {

    m_orderConditionChip = orderConditionChipValue;

}

// set the number of physics trigger algorithms
void l1t::TriggerMenuParser::setGtNumberPhysTriggers(
        const unsigned int& numberPhysTriggersValue) {

    m_numberPhysTriggers = numberPhysTriggersValue;

}


// set the condition maps
void l1t::TriggerMenuParser::setGtConditionMap(const std::vector<ConditionMap>& condMap) {
    m_conditionMap = condMap;
}

// set the trigger menu name
void l1t::TriggerMenuParser::setGtTriggerMenuInterface(const std::string& menuInterface) {
    m_triggerMenuInterface = menuInterface;
}

// set the trigger menu uuid
void l1t::TriggerMenuParser::setGtTriggerMenuUUID(const int uuid) {
    m_triggerMenuUUID = uuid;
}

void l1t::TriggerMenuParser::setGtTriggerMenuName(const std::string& menuName) {
    m_triggerMenuName = menuName;
}

void l1t::TriggerMenuParser::setGtTriggerMenuImplementation(const unsigned long& menuImplementation) {
    m_triggerMenuImplementation = menuImplementation;
}

// set menu associated scale key
void l1t::TriggerMenuParser::setGtScaleDbKey(const std::string& scaleKey) {
    m_scaleDbKey = scaleKey;
}

// set the vectors containing the conditions
void l1t::TriggerMenuParser::setVecMuonTemplate(
        const std::vector<std::vector<MuonTemplate> >& vecMuonTempl) {

    m_vecMuonTemplate = vecMuonTempl;
}

void l1t::TriggerMenuParser::setVecCaloTemplate(
        const std::vector<std::vector<CaloTemplate> >& vecCaloTempl) {

    m_vecCaloTemplate = vecCaloTempl;
}

void l1t::TriggerMenuParser::setVecEnergySumTemplate(
        const std::vector<std::vector<EnergySumTemplate> >& vecEnergySumTempl) {

    m_vecEnergySumTemplate = vecEnergySumTempl;
}



void l1t::TriggerMenuParser::setVecExternalTemplate(
        const std::vector<std::vector<ExternalTemplate> >& vecExternalTempl) {

    m_vecExternalTemplate = vecExternalTempl;
}


void l1t::TriggerMenuParser::setVecCorrelationTemplate(
        const std::vector<std::vector<CorrelationTemplate> >& vecCorrelationTempl) {

    m_vecCorrelationTemplate = vecCorrelationTempl;
}

// set the vectors containing the conditions for correlation templates
//
void l1t::TriggerMenuParser::setCorMuonTemplate(
        const std::vector<std::vector<MuonTemplate> >& corMuonTempl) {

    m_corMuonTemplate = corMuonTempl;
}

void l1t::TriggerMenuParser::setCorCaloTemplate(
        const std::vector<std::vector<CaloTemplate> >& corCaloTempl) {

    m_corCaloTemplate = corCaloTempl;
}

void l1t::TriggerMenuParser::setCorEnergySumTemplate(
        const std::vector<std::vector<EnergySumTemplate> >& corEnergySumTempl) {

    m_corEnergySumTemplate = corEnergySumTempl;
}




// set the algorithm map (by algorithm names)
void l1t::TriggerMenuParser::setGtAlgorithmMap(const AlgorithmMap& algoMap) {
    m_algorithmMap = algoMap;
}

// set the algorithm map (by algorithm aliases)
void l1t::TriggerMenuParser::setGtAlgorithmAliasMap(const AlgorithmMap& algoMap) {
    m_algorithmAliasMap = algoMap;
}



std::map<std::string, unsigned int> l1t::TriggerMenuParser::getExternalSignals(const L1TUtmTriggerMenu* utmMenu) {

   using namespace tmeventsetup;
   const esTriggerMenu* menu = reinterpret_cast<const esTriggerMenu*> (utmMenu);
   const std::map<std::string, esCondition>& condMap = menu->getConditionMap();

  std::map<std::string, unsigned int> extBitMap;
  
  //loop over the algorithms
  for (std::map<std::string, esCondition>::const_iterator cit = condMap.begin();
       cit != condMap.end(); cit++)
  {
      const esCondition& condition = cit->second;
      if(condition.getType() == esConditionType::Externals ) {
      
             // Get object for External conditions
             const std::vector<esObject>& objects = condition.getObjects();
             for (size_t jj = 0; jj < objects.size(); jj++) {   

                 const esObject object = objects.at(jj);
                 if(object.getType() == esObjectType::EXT) {
              
                   unsigned int channelID = object.getExternalChannelId();
		   std::string  name = object.getExternalSignalName();
		   
		   if (extBitMap.count(name) == 0) extBitMap.insert(std::map<std::string, unsigned int>::value_type(name,channelID));
                }
             }   
         
      }
        
  }   
/*   
  for (std::map<std::string, unsigned int>::const_iterator cit = extBitMap.begin();
       cit != extBitMap.end(); cit++) {
       std::cout << " Ext Map:  Name " << cit->first << " Bit " << cit->second << std::endl;
  } 
*/
  return  extBitMap;
  
}

// parse def.xml file
void l1t::TriggerMenuParser::parseCondFormats(const L1TUtmTriggerMenu* utmMenu) {


    // resize the vector of condition maps
    // the number of condition chips should be correctly set before calling parseXmlFile
    m_conditionMap.resize(m_numberConditionChips);

    m_vecMuonTemplate.resize(m_numberConditionChips);
    m_vecCaloTemplate.resize(m_numberConditionChips);
    m_vecEnergySumTemplate.resize(m_numberConditionChips);
    m_vecExternalTemplate.resize(m_numberConditionChips);

    m_vecCorrelationTemplate.resize(m_numberConditionChips);
    m_corMuonTemplate.resize(m_numberConditionChips);
    m_corCaloTemplate.resize(m_numberConditionChips);
    m_corEnergySumTemplate.resize(m_numberConditionChips);

  using namespace tmeventsetup;
  using namespace Algorithm;
  
  const esTriggerMenu* menu = reinterpret_cast<const esTriggerMenu*> (utmMenu);

  //get the meta data
  m_triggerMenuDescription = menu->getComment();
  m_triggerMenuDate = menu->getDatetime();
  m_triggerMenuImplementation =  0; // FIXME: ( getMmHashN(menu->getFirmwareUuid()) & 0xFFFFFFFF); //make sure we only have 32 bits 
  m_triggerMenuName = menu->getName();
  m_triggerMenuInterface = menu->getVersion(); //BLW: correct descriptor?
  m_triggerMenuUUID = 0; // FIXME:  ( getMmHashN(menu->getName()) & 0xFFFFFFFF); //make sure we only have 32 bits 

  const std::map<std::string, esAlgorithm>& algoMap = menu->getAlgorithmMap();
  const std::map<std::string, esCondition>& condMap = menu->getConditionMap();
  const std::map<std::string, esScale>& scaleMap = menu->getScaleMap();

  // parse the scales
  m_gtScales.setScalesName( menu->getScaleSetName() );
  parseScales(scaleMap);


  //loop over the algorithms
  for (std::map<std::string, esAlgorithm>::const_iterator cit = algoMap.begin();
       cit != algoMap.end(); cit++)
  {
    //condition chip (artifact)  TO DO: Update
    int chipNr = 0;
  
    //get algorithm
    const esAlgorithm& algo = cit->second;

    //parse the algorithm
    parseAlgorithm(algo,chipNr); //blw

    //get conditions for this algorithm
    const std::vector<std::string>& rpn_vec = algo.getRpnVector();
    for (size_t ii = 0; ii < rpn_vec.size(); ii++)
    {
      const std::string& token = rpn_vec.at(ii);
      if (isGate(token)) continue;
//      long hash = getHash(token);
      const esCondition& condition = condMap.find(token)->second;
     
     
      //check to see if this condtion already exists
      if ((m_conditionMap[chipNr]).count(condition.getName()) == 0) {
     	  
	  // parse Calo Conditions (EG, Jets, Taus)      
	  if(condition.getType() == esConditionType::SingleEgamma || 
             condition.getType() == esConditionType::DoubleEgamma ||
	     condition.getType() == esConditionType::TripleEgamma ||
	     condition.getType() == esConditionType::QuadEgamma   ||
	     condition.getType() == esConditionType::SingleTau    ||
	     condition.getType() == esConditionType::DoubleTau    ||
	     condition.getType() == esConditionType::TripleTau    ||
	     condition.getType() == esConditionType::QuadTau      ||
	     condition.getType() == esConditionType::SingleJet    ||
	     condition.getType() == esConditionType::DoubleJet    ||
	     condition.getType() == esConditionType::TripleJet    ||
	     condition.getType() == esConditionType::QuadJet      ) 
	  {
             parseCalo(condition,chipNr,false); //blw 

	  // parse Energy Sums	 
	  } else if(condition.getType() == esConditionType::TotalEt ||
                    condition.getType() == esConditionType::TotalHt ||
		    condition.getType() == esConditionType::MissingEt ||
		    condition.getType() == esConditionType::MissingHt )
		    //condition.getType() == esConditionType::MissingEt2 ||
		    //condition.getType() == esConditionType::MinBias )
	  {
             parseEnergySum(condition,chipNr,false); 	

	  //parse Muons	 	
	  } else if(condition.getType() == esConditionType::SingleMuon    ||
	            condition.getType() == esConditionType::DoubleMuon    ||
	            condition.getType() == esConditionType::TripleMuon    ||
	            condition.getType() == esConditionType::QuadMuon      )       
	  {
             parseMuon(condition,chipNr,false);
             
	     
	  //parse Correlation Conditions	 	
	  } else if(condition.getType() == esConditionType::MuonMuonCorrelation    ||
	            condition.getType() == esConditionType::MuonEsumCorrelation    ||
	            condition.getType() == esConditionType::CaloMuonCorrelation    ||
	            condition.getType() == esConditionType::CaloCaloCorrelation    ||
		    condition.getType() == esConditionType::CaloEsumCorrelation    ||
		    condition.getType() == esConditionType::InvariantMass )       
	  {
             parseCorrelation(condition,chipNr);

	  //parse Muons	 	
	  } else if(condition.getType() == esConditionType::Externals      )       
	  {
             parseExternal(condition,chipNr);
	     	    
	  }      
      
      }//if condition is a new one
    }//loop over conditions
  }//loop over algorithms

  return;


}



//

void l1t::TriggerMenuParser::setGtTriggerMenuInterfaceDate(const std::string& val) {

    m_triggerMenuInterfaceDate = val;

}

void l1t::TriggerMenuParser::setGtTriggerMenuInterfaceAuthor(const std::string& val) {

    m_triggerMenuInterfaceAuthor = val;

}

void l1t::TriggerMenuParser::setGtTriggerMenuInterfaceDescription(const std::string& val) {

    m_triggerMenuInterfaceDescription = val;

}


void l1t::TriggerMenuParser::setGtTriggerMenuDate(const std::string& val) {

    m_triggerMenuDate = val;

}

void l1t::TriggerMenuParser::setGtTriggerMenuAuthor(const std::string& val) {

    m_triggerMenuAuthor = val;

}

void l1t::TriggerMenuParser::setGtTriggerMenuDescription(const std::string& val) {

    m_triggerMenuDescription = val;

}

void l1t::TriggerMenuParser::setGtAlgorithmImplementation(const std::string& val) {

    m_algorithmImplementation = val;

}



// methods for conditions and algorithms

// clearMaps - delete all conditions and algorithms in
// the maps and clear the maps.
void l1t::TriggerMenuParser::clearMaps() {

    // loop over condition maps (one map per condition chip)
    // then loop over conditions in the map
    for (std::vector<ConditionMap>::iterator itCondOnChip = m_conditionMap.begin(); itCondOnChip
        != m_conditionMap.end(); itCondOnChip++) {

        // the conditions in the maps are deleted in L1uGtTriggerMenu, not here

        itCondOnChip->clear();

    }

    // the algorithms in the maps are deleted in L1uGtTriggerMenu, not here
    m_algorithmMap.clear();

}

// insertConditionIntoMap - safe insert of condition into condition map.
// if the condition name already exists, do not insert it and return false
bool l1t::TriggerMenuParser::insertConditionIntoMap(GlobalCondition& cond, const int chipNr) {

    std::string cName = cond.condName();
    LogTrace("TriggerMenuParser")
    << "    Trying to insert condition \"" << cName << "\" in the condition map." ;

    // no condition name has to appear twice!
    if ((m_conditionMap[chipNr]).count(cName) != 0) {
        LogTrace("TriggerMenuParser") << "      Condition " << cName
            << " already exists - not inserted!" << std::endl;
        return false;
    }

    (m_conditionMap[chipNr])[cName] = &cond;
     LogTrace("TriggerMenuParser")
     << "      OK - condition inserted!"
    << std::endl;


    return true;

}

// insert an algorithm into algorithm map
bool l1t::TriggerMenuParser::insertAlgorithmIntoMap(const GlobalAlgorithm& alg) {

    std::string algName = alg.algoName();
    std::string algAlias = alg.algoAlias();
    //LogTrace("TriggerMenuParser")
    //<< "    Trying to insert algorithm \"" << algName << "\" in the algorithm map." ;

    // no algorithm name has to appear twice!
    if (m_algorithmMap.count(algName) != 0) {
        LogTrace("TriggerMenuParser") << "      Algorithm \"" << algName
            << "\"already exists in the algorithm map- not inserted!" << std::endl;
        return false;
    }

    if (m_algorithmAliasMap.count(algAlias) != 0) {
        LogTrace("TriggerMenuParser") << "      Algorithm alias \"" << algAlias
            << "\"already exists in the algorithm alias map- not inserted!" << std::endl;
        return false;
    }

    // bit number less than zero or greater than maximum number of algorithms
    int bitNumber = alg.algoBitNumber();
    if ((bitNumber < 0) || (bitNumber >= static_cast<int>(m_numberPhysTriggers))) {
        LogTrace("TriggerMenuParser") << "      Bit number " << bitNumber
            << " outside allowed range [0, " << m_numberPhysTriggers
            << ") - algorithm not inserted!" << std::endl;
        return false;
    }

    // maximum number of algorithms
    if (m_algorithmMap.size() >= m_numberPhysTriggers) {
        LogTrace("TriggerMenuParser") << "      More than maximum allowed "
            << m_numberPhysTriggers << " algorithms in the algorithm map - not inserted!"
            << std::endl;
        return false;
    }

    
    // chip number outside allowed values
    int chipNr = alg.algoChipNumber(static_cast<int>(m_numberConditionChips),
        static_cast<int>(m_pinsOnConditionChip), m_orderConditionChip);

    if ((chipNr < 0) || (chipNr > static_cast<int>(m_numberConditionChips))) {
        LogTrace("TriggerMenuParser") << "      Chip number " << chipNr
            << " outside allowed range [0, " << m_numberConditionChips
            << ") - algorithm not inserted!" << std::endl;
        return false;
    }

    // output pin outside allowed values
    int outputPin = alg.algoOutputPin(static_cast<int>(m_numberConditionChips),
        static_cast<int>(m_pinsOnConditionChip), m_orderConditionChip);

    if ((outputPin < 0) || (outputPin > static_cast<int>(m_pinsOnConditionChip))) {
        LogTrace("TriggerMenuParser") << "      Output pin " << outputPin
            << " outside allowed range [0, " << m_pinsOnConditionChip
            << "] - algorithm not inserted!" << std::endl;
        return false;
    }

    // no two algorithms on the same chip can have the same output pin
    for (CItAlgo itAlgo = m_algorithmMap.begin(); itAlgo != m_algorithmMap.end(); itAlgo++) {

        int iPin = (itAlgo->second).algoOutputPin( static_cast<int>(m_numberConditionChips),
            static_cast<int>(m_pinsOnConditionChip), m_orderConditionChip);
        std::string iName = itAlgo->first;
        int iChip = (itAlgo->second).algoChipNumber(static_cast<int>(m_numberConditionChips),
            static_cast<int>(m_pinsOnConditionChip), m_orderConditionChip);

        if ( (outputPin == iPin) && (chipNr == iChip)) {
            LogTrace("TriggerMenuParser") << "      Output pin " << outputPin
                << " is the same as for algorithm " << iName
                << "\n      from the same chip number " << chipNr << " - algorithm not inserted!"
                << std::endl;
            return false;
        }

    }

    // insert algorithm
    m_algorithmMap[algName] = alg;
    m_algorithmAliasMap[algAlias] = alg;

    //LogTrace("TriggerMenuParser")
    //<< "      OK - algorithm inserted!"
    //<< std::endl;

    return true;

}


template <typename T> std::string l1t::TriggerMenuParser::l1t2string( T data ){
  std::stringstream ss;
  ss << data;
  return ss.str();
}
int l1t::TriggerMenuParser::l1tstr2int( const std::string data ){ 
  std::stringstream ss;
  ss << data;
  int value;
  ss >> value;
  return value;
}


/**
 * parseScales Parse Et, Eta, and Phi Scales
 *
 * @return "true" if succeeded, "false" if an error occurred.
 *
 */

bool l1t::TriggerMenuParser::parseScales(std::map<std::string, tmeventsetup::esScale> scaleMap) {
	
    using namespace tmeventsetup;
 
//  Setup ScaleParameter to hold information from parsing
    GlobalScales::ScaleParameters muScales; 
    GlobalScales::ScaleParameters egScales; 
    GlobalScales::ScaleParameters tauScales;
    GlobalScales::ScaleParameters jetScales;
    GlobalScales::ScaleParameters ettScales;
    GlobalScales::ScaleParameters etmScales;
    GlobalScales::ScaleParameters httScales;
    GlobalScales::ScaleParameters htmScales; 
 
 
// Start by parsing the Scale Map
    for (std::map<std::string, esScale>::const_iterator cit = scaleMap.begin();
       cit != scaleMap.end(); cit++)
  {
     const esScale& scale = cit->second;
 
    GlobalScales::ScaleParameters *scaleParam;
    if      (scale.getObjectType() == esObjectType::Muon)   scaleParam = &muScales;
    else if (scale.getObjectType() == esObjectType::Egamma) scaleParam = &egScales;
    else if (scale.getObjectType() == esObjectType::Tau)    scaleParam = &tauScales;
    else if (scale.getObjectType() == esObjectType::Jet)    scaleParam = &jetScales;
    else if (scale.getObjectType() == esObjectType::ETT)    scaleParam = &ettScales;
    else if (scale.getObjectType() == esObjectType::ETM)    scaleParam = &etmScales;
    else if (scale.getObjectType() == esObjectType::HTT)    scaleParam = &httScales;
    else if (scale.getObjectType() == esObjectType::HTM)    scaleParam = &htmScales;
    else scaleParam = 0;
    
    if(scaleParam != 0) {	
        switch(scale.getScaleType()) {
	    case esScaleType::EtScale: {
	        scaleParam->etMin  = scale.getMinimum();
		scaleParam->etMax  = scale.getMaximum();
		scaleParam->etStep = scale.getStep();
		
		//Get bin edges
		const std::vector<esBin> binsV = scale.getBins();
		for(unsigned int i=0; i<binsV.size(); i++) {
		   const esBin& bin = binsV.at(i); 
		   std::pair<double, double> binLimits(bin.minimum, bin.maximum);
		   scaleParam->etBins.push_back(binLimits);
		}
		
		// If this is an energy sum fill dummy values for eta and phi
		// There are no scales for these in the XML so the other case statements will not be seen....do it here.
		if(scale.getObjectType() == esObjectType::ETT || scale.getObjectType() == esObjectType::HTT || 
		   scale.getObjectType() == esObjectType::ETM || scale.getObjectType() == esObjectType::HTM ) {
		   
	           scaleParam->etaMin  = -1.;
		   scaleParam->etaMax  = -1.;
		   scaleParam->etaStep = -1.;		   
		   if(scale.getObjectType() == esObjectType::ETT || scale.getObjectType() == esObjectType::HTT) {
	              scaleParam->phiMin  = -1.;
		      scaleParam->phiMax  = -1.;
		      scaleParam->phiStep = -1.;		   		   
		   }
		}   
	    }
		break;
	    case esScaleType::EtaScale: {
	        scaleParam->etaMin  = scale.getMinimum();
		scaleParam->etaMax  = scale.getMaximum();
		scaleParam->etaStep = scale.getStep();
		
		//Get bin edges
		const std::vector<esBin> binsV = scale.getBins();
		scaleParam->etaBins.resize(pow(2,scale.getNbits()));
		for(unsigned int i=0; i<binsV.size(); i++) {
		   const esBin& bin = binsV.at(i); 
		   std::pair<double, double> binLimits(bin.minimum, bin.maximum);		   
		   scaleParam->etaBins.at(bin.hw_index) = binLimits;
		}
	    }
		break;
	    case esScaleType::PhiScale: {
	        scaleParam->phiMin  = scale.getMinimum();
		scaleParam->phiMax  = scale.getMaximum();
		scaleParam->phiStep = scale.getStep();
		
		//Get bin edges
		const std::vector<esBin> binsV = scale.getBins();
		scaleParam->phiBins.resize(pow(2,scale.getNbits()));
		for(unsigned int i=0; i<binsV.size(); i++) {
		   const esBin& bin = binsV.at(i); 
		   std::pair<double, double> binLimits(bin.minimum, bin.maximum);
		   scaleParam->phiBins.at(bin.hw_index) = binLimits;
		}
	    }
		break;				
	    default:
	    
	        break;			
	} //end switch 
    } //end valid scale	
  } //end loop over scaleMap

  // put the ScaleParameters into the class
  m_gtScales.setMuonScales(muScales);
  m_gtScales.setEGScales(egScales);
  m_gtScales.setTauScales(tauScales);
  m_gtScales.setJetScales(jetScales);
  m_gtScales.setETTScales(ettScales);
  m_gtScales.setETMScales(etmScales);
  m_gtScales.setHTTScales(httScales);
  m_gtScales.setHTMScales(htmScales);
  


// Setup the LUT for the Scale Conversions
  bool hasPrecision = false;
  std::map<std::string, unsigned int> precisions;
  getPrecisions(precisions, scaleMap);
  for (std::map<std::string, unsigned int>::const_iterator cit = precisions.begin(); cit != precisions.end(); cit++)
  {
    //std::cout << cit->first << " = " << cit->second << "\n";
    hasPrecision = true;
  }


  if (hasPrecision)
  {

    //Start with the Cal - Muon Eta LUTS
    //----------------------------------
    parseCalMuEta_LUTS(scaleMap, "EG",  "MU");  
    parseCalMuEta_LUTS(scaleMap, "JET", "MU");
    parseCalMuEta_LUTS(scaleMap, "TAU", "MU");
    
    //Now the Cal - Muon Phi LUTS
    //-------------------------------------
    parseCalMuPhi_LUTS(scaleMap, "EG",  "MU");
    parseCalMuPhi_LUTS(scaleMap, "JET", "MU");
    parseCalMuPhi_LUTS(scaleMap, "TAU", "MU");
    parseCalMuPhi_LUTS(scaleMap, "HTM", "MU");
    parseCalMuPhi_LUTS(scaleMap, "ETM", "MU");

    // Now the Pt LUTs  (??? more combinations needed ??)
    // ---------------
    parsePt_LUTS(scaleMap, "EG",  precisions["PRECISION-EG-MU-MassPt"] );
    parsePt_LUTS(scaleMap, "MU",  precisions["PRECISION-EG-MU-MassPt"] );
    parsePt_LUTS(scaleMap, "JET", precisions["PRECISION-EG-JET-MassPt"] );
    parsePt_LUTS(scaleMap, "TAU", precisions["PRECISION-EG-TAU-MassPt"] );
   
    // Now the Delta Eta/Cosh LUTs (must be done in groups)
    // ----------------------------------------------------
    parseDeltaEta_Cosh_LUTS(scaleMap,"EG","EG", precisions["PRECISION-EG-EG-Delta"], precisions["PRECISION-EG-EG-Math"]);
    parseDeltaEta_Cosh_LUTS(scaleMap,"EG","JET",precisions["PRECISION-EG-JET-Delta"],precisions["PRECISION-EG-JET-Math"]);
    parseDeltaEta_Cosh_LUTS(scaleMap,"EG","TAU",precisions["PRECISION-EG-TAU-Delta"],precisions["PRECISION-EG-TAU-Math"]);    
    parseDeltaEta_Cosh_LUTS(scaleMap,"EG","MU", precisions["PRECISION-EG-MU-Delta"], precisions["PRECISION-EG-MU-Math"]);

    parseDeltaEta_Cosh_LUTS(scaleMap,"JET","JET",precisions["PRECISION-JET-JET-Delta"],precisions["PRECISION-JET-JET-Math"]);
    parseDeltaEta_Cosh_LUTS(scaleMap,"JET","TAU",precisions["PRECISION-JET-TAU-Delta"],precisions["PRECISION-JET-TAU-Math"]);    
    parseDeltaEta_Cosh_LUTS(scaleMap,"JET","MU", precisions["PRECISION-JET-MU-Delta"], precisions["PRECISION-JET-MU-Math"]);

    parseDeltaEta_Cosh_LUTS(scaleMap,"TAU","TAU",precisions["PRECISION-TAU-TAU-Delta"],precisions["PRECISION-TAU-TAU-Math"]);    
    parseDeltaEta_Cosh_LUTS(scaleMap,"TAU","MU", precisions["PRECISION-TAU-MU-Delta"], precisions["PRECISION-TAU-MU-Math"]);

    parseDeltaEta_Cosh_LUTS(scaleMap,"MU","MU",  precisions["PRECISION-MU-MU-Delta"], precisions["PRECISION-MU-MU-Math"]);


    // Now the Delta Phi/Cos LUTs (must be done in groups)
    // ----------------------------------------------------
    parseDeltaPhi_Cos_LUTS(scaleMap,"EG","EG", precisions["PRECISION-EG-EG-Delta"], precisions["PRECISION-EG-EG-Math"]);
    parseDeltaPhi_Cos_LUTS(scaleMap,"EG","JET",precisions["PRECISION-EG-JET-Delta"],precisions["PRECISION-EG-JET-Math"]);
    parseDeltaPhi_Cos_LUTS(scaleMap,"EG","TAU",precisions["PRECISION-EG-TAU-Delta"],precisions["PRECISION-EG-TAU-Math"]);
    parseDeltaPhi_Cos_LUTS(scaleMap,"EG","ETM",precisions["PRECISION-EG-ETM-Delta"],precisions["PRECISION-EG-ETM-Math"]);
    parseDeltaPhi_Cos_LUTS(scaleMap,"EG","HTM",precisions["PRECISION-EG-HTM-Delta"],precisions["PRECISION-EG-HTM-Math"]);
    parseDeltaPhi_Cos_LUTS(scaleMap,"EG","MU", precisions["PRECISION-EG-MU-Delta"], precisions["PRECISION-EG-MU-Math"]);

    parseDeltaPhi_Cos_LUTS(scaleMap,"JET","JET",precisions["PRECISION-JET-JET-Delta"],precisions["PRECISION-JET-JET-Math"]);
    parseDeltaPhi_Cos_LUTS(scaleMap,"JET","TAU",precisions["PRECISION-JET-TAU-Delta"],precisions["PRECISION-JET-TAU-Math"]);
    parseDeltaPhi_Cos_LUTS(scaleMap,"JET","ETM",precisions["PRECISION-JET-ETM-Delta"],precisions["PRECISION-JET-ETM-Math"]);
    parseDeltaPhi_Cos_LUTS(scaleMap,"JET","HTM",precisions["PRECISION-JET-HTM-Delta"],precisions["PRECISION-JET-HTM-Math"]);
    parseDeltaPhi_Cos_LUTS(scaleMap,"JET","MU", precisions["PRECISION-JET-MU-Delta"], precisions["PRECISION-JET-MU-Math"]);

    parseDeltaPhi_Cos_LUTS(scaleMap,"TAU","TAU",precisions["PRECISION-TAU-TAU-Delta"],precisions["PRECISION-TAU-TAU-Math"]);
    parseDeltaPhi_Cos_LUTS(scaleMap,"TAU","ETM",precisions["PRECISION-TAU-ETM-Delta"],precisions["PRECISION-TAU-ETM-Math"]);
    parseDeltaPhi_Cos_LUTS(scaleMap,"TAU","HTM",precisions["PRECISION-TAU-HTM-Delta"],precisions["PRECISION-TAU-HTM-Math"]);
    parseDeltaPhi_Cos_LUTS(scaleMap,"TAU","MU", precisions["PRECISION-TAU-MU-Delta"], precisions["PRECISION-TAU-MU-Math"]);

    parseDeltaPhi_Cos_LUTS(scaleMap,"MU","ETM",precisions["PRECISION-MU-ETM-Delta"],precisions["PRECISION-MU-ETM-Math"]);
    parseDeltaPhi_Cos_LUTS(scaleMap,"MU","HTM",precisions["PRECISION-MU-HTM-Delta"],precisions["PRECISION-MU-HTM-Math"]);
    parseDeltaPhi_Cos_LUTS(scaleMap,"MU","MU", precisions["PRECISION-MU-MU-Delta"], precisions["PRECISION-MU-MU-Math"]);

    //m_gtScales.dumpAllLUTs(std::cout);
    //m_gtScales.print(std::cout);

  }

  

 
    
    return true;
}

void l1t::TriggerMenuParser::parseCalMuEta_LUTS(std::map<std::string, tmeventsetup::esScale> scaleMap, std::string obj1, std::string obj2) 
{

    using namespace tmeventsetup;
    
    // First Delta Eta for this set
    std::string scLabel1 = obj1; 
    scLabel1 += "-ETA";
    std::string scLabel2 = obj2; 
    scLabel2 += "-ETA";    
    const esScale* scale1 = &scaleMap.find(scLabel1)->second;
    const esScale* scale2 = &scaleMap.find(scLabel2)->second;   

    std::vector<long long> lut_cal_2_mu_eta;
    getCaloMuonEtaConversionLut(lut_cal_2_mu_eta, scale1, scale2);

    std::string lutName = obj1;
    lutName += "-";
    lutName += obj2;
    m_gtScales.setLUT_CalMuEta(lutName,lut_cal_2_mu_eta);
        

}

void l1t::TriggerMenuParser::parseCalMuPhi_LUTS(std::map<std::string, tmeventsetup::esScale> scaleMap, std::string obj1, std::string obj2) 
{

    using namespace tmeventsetup;
    
    // First Delta Eta for this set
    std::string scLabel1 = obj1; 
    scLabel1 += "-PHI";
    std::string scLabel2 = obj2; 
    scLabel2 += "-PHI";    
    const esScale* scale1 = &scaleMap.find(scLabel1)->second;
    const esScale* scale2 = &scaleMap.find(scLabel2)->second;   

    std::vector<long long> lut_cal_2_mu_phi;
    getCaloMuonPhiConversionLut(lut_cal_2_mu_phi, scale1, scale2);

    std::string lutName = obj1;
    lutName += "-";
    lutName += obj2;
    m_gtScales.setLUT_CalMuPhi(lutName,lut_cal_2_mu_phi);
        

}

void l1t::TriggerMenuParser::parsePt_LUTS(std::map<std::string, tmeventsetup::esScale> scaleMap, std::string obj1, unsigned int prec) 
{

    using namespace tmeventsetup;
    
    // First Delta Eta for this set
    std::string scLabel1 = obj1; 
    scLabel1 += "-ET";
    const esScale* scale1 = &scaleMap.find(scLabel1)->second;

    std::vector<long long> lut_pt;
    getLut(lut_pt, scale1, prec);
    m_gtScales.setLUT_Pt(scLabel1,lut_pt,prec);
        

}		
	
void l1t::TriggerMenuParser::parseDeltaEta_Cosh_LUTS(std::map<std::string, tmeventsetup::esScale> scaleMap, std::string obj1, std::string obj2, unsigned int prec1, unsigned int prec2) 
{

    using namespace tmeventsetup;
    
    // First Delta Eta for this set
    std::string scLabel1 = obj1; 
    scLabel1 += "-ETA";
    std::string scLabel2 = obj2; 
    scLabel2 += "-ETA";    
    const esScale* scale1 = &scaleMap.find(scLabel1)->second;
    const esScale* scale2 = &scaleMap.find(scLabel2)->second;   
    std::vector<double> val_delta_eta;
    std::vector<long long> lut_delta_eta;
    size_t n = getDeltaVector(val_delta_eta, scale1, scale2);
    setLut(lut_delta_eta, val_delta_eta, prec1);
    std::string lutName = obj1;
    lutName += "-";
    lutName += obj2;
    m_gtScales.setLUT_DeltaEta(lutName,lut_delta_eta,prec1);
        
    // Second Get the Cosh for this delta Eta Set
    std::vector<long long> lut_cosh;
    applyCosh(val_delta_eta, n);
    setLut(lut_cosh, val_delta_eta, prec2);
    m_gtScales.setLUT_Cosh(lutName,lut_cosh,prec2);

}	

void l1t::TriggerMenuParser::parseDeltaPhi_Cos_LUTS(std::map<std::string, tmeventsetup::esScale> scaleMap, std::string obj1, std::string obj2, unsigned int prec1, unsigned int prec2) 
{

    using namespace tmeventsetup;
    
    // First Delta phi for this set
    std::string scLabel1 = obj1; 
    scLabel1 += "-PHI";
    std::string scLabel2 = obj2; 
    scLabel2 += "-PHI";    
    const esScale* scale1 = &scaleMap.find(scLabel1)->second;
    const esScale* scale2 = &scaleMap.find(scLabel2)->second;   
    std::vector<double> val_delta_phi;
    std::vector<long long> lut_delta_phi;
    size_t n = getDeltaVector(val_delta_phi, scale1, scale2);
    setLut(lut_delta_phi, val_delta_phi, prec1);
    std::string lutName = obj1;
    lutName += "-";
    lutName += obj2;
    m_gtScales.setLUT_DeltaPhi(lutName,lut_delta_phi,prec1);
        
    // Second Get the Cosh for this delta phi Set
    std::vector<long long> lut_cos;
    applyCos(val_delta_phi, n);
    setLut(lut_cos, val_delta_phi, prec2);
    m_gtScales.setLUT_Cos(lutName,lut_cos,prec2);

}	

/**
 * parseMuon Parse a muon condition and insert an entry to the conditions map
 *
 * @param node The corresponding node.
 * @param name The name of the condition.
 * @param chipNr The number of the chip this condition is located.
 *
 * @return "true" if succeeded, "false" if an error occurred.
 *
 */

bool l1t::TriggerMenuParser::parseMuon(tmeventsetup::esCondition condMu,
        unsigned int chipNr, const bool corrFlag) {

    using namespace tmeventsetup;

    // get condition, particle name (must be muon) and type name
    std::string condition = "muon";
    std::string particle = "muon";//l1t2string( condMu.objectType() );
    std::string type = l1t2string( condMu.getType() );
    std::string name = l1t2string( condMu.getName() );
    int nrObj = -1;



    GtConditionType cType = l1t::TypeNull; 

    if (condMu.getType() == esConditionType::SingleMuon) {
	type = "1_s";
	cType = l1t::Type1s;
	nrObj = 1;
    } else if (condMu.getType() == esConditionType::DoubleMuon) {
	type = "2_s";
	cType = l1t::Type2s;
	nrObj = 2;	
    } else if (condMu.getType() == esConditionType::TripleMuon) {
	type = "3";
	cType = l1t::Type3s;
	nrObj = 3;
    } else if (condMu.getType() == esConditionType::QuadMuon) {
	type = "4";
	cType = l1t::Type4s;
	nrObj = 4;
    } else {
        edm::LogError("TriggerMenuParser") << "Wrong type for muon-condition ("
            << type << ")" << std::endl;
        return false;
    }

    if (nrObj < 0) {
        edm::LogError("TriggerMenuParser") << "Unknown type for muon-condition (" << type
            << ")" << "\nCan not determine number of trigger objects. " << std::endl;
        return false;
    }

    LogDebug("TriggerMenuParser")
      << "\n ****************************************** "
      << "\n      parseMuon  "
      << "\n condition = " << condition
      << "\n particle  = " << particle
      << "\n type      = " << type
      << "\n name      = " << name
      << std::endl;



//     // get values

    // temporary storage of the parameters
    std::vector<MuonTemplate::ObjectParameter> objParameter(nrObj);
    
    // Do we need this?
    MuonTemplate::CorrelationParameter corrParameter;

    // need at least two values for deltaPhi
    std::vector<boost::uint64_t> tmpValues((nrObj > 2) ? nrObj : 2);
    tmpValues.reserve( nrObj );

    if( int(condMu.getObjects().size())!=nrObj ){
      edm::LogError("TriggerMenuParser") << " condMu objects: nrObj = " << nrObj
					    << "condMu.getObjects().size() = " 
					    << condMu.getObjects().size()
					    << std::endl;
      return false;
    }


//  Look for cuts on the objects in the condition
     unsigned int chargeCorrelation = 1;
     const std::vector<esCut>& cuts = condMu.getCuts();      
     for (size_t jj = 0; jj < cuts.size(); jj++)
      {
        const esCut cut = cuts.at(jj);
	if(cut.getCutType() == esCutType::ChargeCorrelation) { 
	   if( cut.getData()=="ls" )      chargeCorrelation = 2;
	   else if( cut.getData()=="os" ) chargeCorrelation = 4;
	   else chargeCorrelation = 1; //ignore correlation
        }
      }
   
    //set charge correlation parameter
    corrParameter.chargeCorrelation = chargeCorrelation;//tmpValues[0];


    int cnt = 0;


// BLW TO DO: These needs to the added to the object rather than the whole condition.
    int relativeBx = 0;
    bool gEq = false;
    
// Loop over objects and extract the cuts on the objects
    const std::vector<esObject>& objects = condMu.getObjects();
    for (size_t jj = 0; jj < objects.size(); jj++) {   

       const esObject object = objects.at(jj);
       gEq =  (object.getComparisonOperator() == esComparisonOperator::GE);

//  BLW TO DO: This needs to be added to the Object Parameters   
       relativeBx = object.getBxOffset();

//  Loop over the cuts for this object
        int upperThresholdInd = -1; 
	int lowerThresholdInd = 0;
        int cntEta = 0;
        unsigned int etaWindow1Lower=-1, etaWindow1Upper=-1, etaWindow2Lower=-1, etaWindow2Upper=-1;
	int cntPhi = 0;
	unsigned int phiWindow1Lower=-1, phiWindow1Upper=-1, phiWindow2Lower=-1, phiWindow2Upper=-1;
        int isolationLUT = 0xF; //default is to ignore unless specified.
	int charge = -1; //default value is to ignore unless specified
	int qualityLUT = 0xFFFF; //default is to ignore unless specified.		
	
        const std::vector<esCut>& cuts = object.getCuts();
        for (size_t kk = 0; kk < cuts.size(); kk++)
        {
          const esCut cut = cuts.at(kk); 
	 
	  switch(cut.getCutType()){
	     case esCutType::Threshold:
	       lowerThresholdInd = cut.getMinimum().index;
	       upperThresholdInd = cut.getMaximum().index;
	       break;
	       
	     case esCutType::Eta: {
	       
                 if(cntEta == 0) {
		    etaWindow1Lower = cut.getMinimum().index;
		    etaWindow1Upper = cut.getMaximum().index;
		 } else if(cntEta == 1) {
		    etaWindow2Lower = cut.getMinimum().index;
		    etaWindow2Upper = cut.getMaximum().index;
                 } else {
        	   edm::LogError("TriggerMenuParser") << "Too Many Eta Cuts for muon-condition ("
        	       << particle << ")" << std::endl;
        	   return false;
		 }
		 cntEta++; 

	       } break;
	       
	     case esCutType::Phi: {

                if(cntPhi == 0) {
		    phiWindow1Lower = cut.getMinimum().index;
		    phiWindow1Upper = cut.getMaximum().index;
		 } else if(cntPhi == 1) {
		    phiWindow2Lower = cut.getMinimum().index;
		    phiWindow2Upper = cut.getMaximum().index;
                 } else {
        	   edm::LogError("TriggerMenuParser") << "Too Many Phi Cuts for muon-condition ("
        	       << particle << ")" << std::endl;
        	   return false;
		 }
		 cntPhi++; 

	       }break;
	       
	     case esCutType::Charge:
	       if( cut.getData()=="positive" ) charge = 0;
               else if( cut.getData()=="negative" ) charge = 1;
	       else charge = -1;
	       break;
	     case esCutType::Quality:
	     
                qualityLUT = l1tstr2int(cut.getData());
	     
	       break;
	     case esCutType::Isolation: {

                isolationLUT = l1tstr2int(cut.getData());
		       
	       } break;
	     default:
	       break; 	       	       	       	       
	  } //end switch 
	  
        } //end loop over cuts


// Set the parameter cuts
	objParameter[cnt].ptHighThreshold = upperThresholdInd;
	objParameter[cnt].ptLowThreshold  = lowerThresholdInd;

	objParameter[cnt].etaWindow1Lower     = etaWindow1Lower;
	objParameter[cnt].etaWindow1Upper     = etaWindow1Upper;
	objParameter[cnt].etaWindow2Lower = etaWindow2Lower;
	objParameter[cnt].etaWindow2Upper = etaWindow2Upper;

	objParameter[cnt].phiWindow1Lower     = phiWindow1Lower;
	objParameter[cnt].phiWindow1Upper     = phiWindow1Upper;
	objParameter[cnt].phiWindow2Lower = phiWindow2Lower;
	objParameter[cnt].phiWindow2Upper = phiWindow2Upper;

// BLW TO DO: Do we need these anymore?  Drop them?   
        objParameter[cnt].enableMip = false;//tmpMip[i];
        objParameter[cnt].enableIso = false;//tmpEnableIso[i];
        objParameter[cnt].requestIso = false;//tmpRequestIso[i];

        objParameter[cnt].charge = charge;
        objParameter[cnt].qualityLUT = qualityLUT;
        objParameter[cnt].isolationLUT = isolationLUT;


        cnt++;
    } //end loop over objects	


    // object types - all muons
    std::vector<GlobalObject> objType(nrObj, gtMu);



    // now create a new CondMuonition
    MuonTemplate muonCond(name);

    muonCond.setCondType(cType);
    muonCond.setObjectType(objType);
    muonCond.setCondGEq(gEq);
    muonCond.setCondChipNr(chipNr);
    muonCond.setCondRelativeBx(relativeBx);

    muonCond.setConditionParameter(objParameter, corrParameter);

    if (edm::isDebugEnabled()) {
        std::ostringstream myCoutStream;
        muonCond.print(myCoutStream);
        LogTrace("TriggerMenuParser") << myCoutStream.str() << "\n" << std::endl;
    }

    // insert condition into the map and into muon template vector
    if ( !insertConditionIntoMap(muonCond, chipNr)) {
        edm::LogError("TriggerMenuParser")
                << "    Error: duplicate condition (" << name << ")"
                << std::endl;
        return false;
    }
    else {
        LogDebug("TriggerMenuParser") << "Added Condition " << name << " to the ConditionMap" << std::endl;
        if (corrFlag) {
	    
            (m_corMuonTemplate[chipNr]).push_back(muonCond);
        }
        else {
	    LogDebug("TriggerMenuParser") << "Added Condition " << name << " to the vecMuonTemplate vector" << std::endl;
            (m_vecMuonTemplate[chipNr]).push_back(muonCond);
        }

    }

    //
    return true;
}


bool l1t::TriggerMenuParser::parseMuonCorr(const tmeventsetup::esObject* corrMu,
        unsigned int chipNr) {


//    XERCES_CPP_NAMESPACE_USE
    using namespace tmeventsetup;

    // get condition, particle name (must be muon) and type name
    std::string condition = "muon";
    std::string particle = "muon";//l1t2string( condMu.objectType() );
    std::string type = l1t2string( corrMu->getType() );
    std::string name = l1t2string( corrMu->getName() );
    int nrObj = 1;
    type = "1_s";
    GtConditionType cType = l1t::Type1s;



    if (nrObj < 0) {
        edm::LogError("TriggerMenuParser") << "Unknown type for muon-condition (" << type
            << ")" << "\nCan not determine number of trigger objects. " << std::endl;
        return false;
    }

    LogDebug("TriggerMenuParser")
      << "\n ****************************************** "
      << "\n      parseMuon  "
      << "\n condition = " << condition
      << "\n particle  = " << particle
      << "\n type      = " << type
      << "\n name      = " << name
      << std::endl;



//     // get values

    // temporary storage of the parameters
    std::vector<MuonTemplate::ObjectParameter> objParameter(nrObj);
    
    // Do we need this?
    MuonTemplate::CorrelationParameter corrParameter;

    // need at least two values for deltaPhi
    std::vector<boost::uint64_t> tmpValues((nrObj > 2) ? nrObj : 2);
    tmpValues.reserve( nrObj );


// BLW TO DO: How do we deal with these in the new format    
//    std::string str_chargeCorrelation = l1t2string( condMu.requestedChargeCorr() );
    std::string str_chargeCorrelation = "ig";
    unsigned int chargeCorrelation = 0;
    if( str_chargeCorrelation=="ig" )      chargeCorrelation = 1;
    else if( str_chargeCorrelation=="ls" ) chargeCorrelation = 2;
    else if( str_chargeCorrelation=="os" ) chargeCorrelation = 4;

    //getXMLHexTextValue("1", dst);
    corrParameter.chargeCorrelation = chargeCorrelation;//tmpValues[0];



 // BLW TO DO: These needs to the added to the object rather than the whole condition.
   int relativeBx = 0;
   bool gEq = false;


   //const esObject* object = condMu;
   gEq =  (corrMu->getComparisonOperator() == esComparisonOperator::GE);

 //  BLW TO DO: This needs to be added to the Object Parameters   
   relativeBx = corrMu->getBxOffset();

 //  Loop over the cuts for this object
    int upperThresholdInd = -1;
    int lowerThresholdInd = 0;
    int cntEta = 0;
    unsigned int etaWindow1Lower=-1, etaWindow1Upper=-1, etaWindow2Lower=-1, etaWindow2Upper=-1;
    int cntPhi = 0;
    unsigned int phiWindow1Lower=-1, phiWindow1Upper=-1, phiWindow2Lower=-1, phiWindow2Upper=-1;
    int isolationLUT = 0xF; //default is to ignore unless specified.
    int charge = -1;       //defaut is to ignore unless specified
    int qualityLUT = 0xFFFF; //default is to ignore unless specified.		

    const std::vector<esCut>& cuts = corrMu->getCuts();
    for (size_t kk = 0; kk < cuts.size(); kk++)
    {
      const esCut cut = cuts.at(kk); 

      switch(cut.getCutType()){
	 case esCutType::Threshold:
	   lowerThresholdInd = cut.getMinimum().index;
	   upperThresholdInd = cut.getMaximum().index;
	   break;

	 case esCutType::Eta: {

             if(cntEta == 0) {
		etaWindow1Lower = cut.getMinimum().index;
		etaWindow1Upper = cut.getMaximum().index;
	     } else if(cntEta == 1) {
		etaWindow2Lower = cut.getMinimum().index;
		etaWindow2Upper = cut.getMaximum().index;
             } else {
               edm::LogError("TriggerMenuParser") << "Too Many Eta Cuts for muon-condition ("
        	   << particle << ")" << std::endl;
               return false;
	     }
	     cntEta++; 

	   } break;

	 case esCutType::Phi: {

            if(cntPhi == 0) {
		phiWindow1Lower = cut.getMinimum().index;
		phiWindow1Upper = cut.getMaximum().index;
	     } else if(cntPhi == 1) {
		phiWindow2Lower = cut.getMinimum().index;
		phiWindow2Upper = cut.getMaximum().index;
             } else {
               edm::LogError("TriggerMenuParser") << "Too Many Phi Cuts for muon-condition ("
        	   << particle << ")" << std::endl;
               return false;
	     }
	     cntPhi++; 

	   }break;

	 case esCutType::Charge:
	   if( cut.getData()=="positive" ) charge = 0;
           else if( cut.getData()=="negative" ) charge = 1;
	   else charge = -1; 
	   break;
	 case esCutType::Quality:

            qualityLUT = l1tstr2int(cut.getData());

	   break;
	 case esCutType::Isolation: {

            isolationLUT = l1tstr2int(cut.getData());

	   } break;
	 default:
	   break; 	       	       	       	       
      } //end switch 

    } //end loop over cuts


 // Set the parameter cuts
    objParameter[0].ptHighThreshold = upperThresholdInd;
    objParameter[0].ptLowThreshold  = lowerThresholdInd;

    objParameter[0].etaWindow1Lower     = etaWindow1Lower;
    objParameter[0].etaWindow1Upper     = etaWindow1Upper;
    objParameter[0].etaWindow2Lower = etaWindow2Lower;
    objParameter[0].etaWindow2Upper = etaWindow2Upper;

    objParameter[0].phiWindow1Lower     = phiWindow1Lower;
    objParameter[0].phiWindow1Upper     = phiWindow1Upper;
    objParameter[0].phiWindow2Lower = phiWindow2Lower;
    objParameter[0].phiWindow2Upper = phiWindow2Upper;

 // BLW TO DO: Do we need these anymore?  Drop them?   
    objParameter[0].enableMip = false;//tmpMip[i];
    objParameter[0].enableIso = false;//tmpEnableIso[i];
    objParameter[0].requestIso = false;//tmpRequestIso[i];

    objParameter[0].charge = charge;
    objParameter[0].qualityLUT = qualityLUT;
    objParameter[0].isolationLUT = isolationLUT;


    // object types - all muons
    std::vector<GlobalObject> objType(nrObj, gtMu);

    // now create a new CondMuonition
    MuonTemplate muonCond(name);

    muonCond.setCondType(cType);
    muonCond.setObjectType(objType);
    muonCond.setCondGEq(gEq);
    muonCond.setCondChipNr(chipNr);
    muonCond.setCondRelativeBx(relativeBx);
    muonCond.setConditionParameter(objParameter, corrParameter);

    if (edm::isDebugEnabled()) {
        std::ostringstream myCoutStream;
        muonCond.print(myCoutStream);
        LogTrace("TriggerMenuParser") << myCoutStream.str() << "\n" << std::endl;
    }

/*
    // insert condition into the map and into muon template vector
    if ( !insertConditionIntoMap(muonCond, chipNr)) {
        edm::LogError("TriggerMenuParser")
                << "    Error: duplicate condition (" << name << ")"
                << std::endl;
        return false;
    }
    else {
        LogDebug("TriggerMenuParser") << "Added Condition " << name << " to the ConditionMap" << std::endl;
            (m_corMuonTemplate[chipNr]).push_back(muonCond);
    }
*/
    (m_corMuonTemplate[chipNr]).push_back(muonCond);


    //
    return true;
}


/**
 * parseCalo Parse a calo condition and insert an entry to the conditions map
 *
 * @param node The corresponding node.
 * @param name The name of the condition.
 * @param chipNr The number of the chip this condition is located.
 *
 * @return "true" if succeeded, "false" if an error occurred.
 *
 */

bool l1t::TriggerMenuParser::parseCalo(tmeventsetup::esCondition condCalo,
        unsigned int chipNr, const bool corrFlag) {


//    XERCES_CPP_NAMESPACE_USE
    using namespace tmeventsetup;
    
    // get condition, particle name and type name

    std::string condition = "calo";
    std::string particle = "test-fix" ;
    std::string type = l1t2string( condCalo.getType() );
    std::string name = l1t2string( condCalo.getName() );

    LogDebug("TriggerMenuParser")
      << "\n ****************************************** " 
      << "\n      (in parseCalo) " 
      << "\n condition = " << condition 
      << "\n particle  = " << particle 
      << "\n type      = " << type 
      << "\n name      = " << name 
      << std::endl;


    GtConditionType cType = l1t::TypeNull; 

    // determine object type type
    // BLW TO DO:  Can this object type wait and be done later in the parsing. Or done differently completely..
    GlobalObject caloObjType;
    int nrObj = -1;

    if (condCalo.getType() == esConditionType::SingleEgamma) {
        caloObjType = gtEG;
	type = "1_s";
	cType= l1t::Type1s;
	nrObj = 1;
    } else if (condCalo.getType() == esConditionType::DoubleEgamma) {
        caloObjType = gtEG;
	type = "2_s";
	cType= l1t::Type2s;
	nrObj = 2;	
    } else if (condCalo.getType() == esConditionType::TripleEgamma) {
        caloObjType = gtEG;
	cType= l1t::Type3s;
	type = "3";
	nrObj = 3;
    } else if (condCalo.getType() == esConditionType::QuadEgamma) {
        caloObjType = gtEG;
	cType= l1t::Type4s;
	type = "4";
	nrObj = 4;
    } else if (condCalo.getType() == esConditionType::SingleJet) {
        caloObjType = gtJet;
	cType= l1t::Type1s;
	type = "1_s";
	nrObj = 1;
    } else if (condCalo.getType() == esConditionType::DoubleJet) {
        caloObjType = gtJet;
	cType= l1t::Type2s;
	type = "2_s";
	nrObj = 2;	
    } else if (condCalo.getType() == esConditionType::TripleJet) {
        caloObjType = gtJet;
	cType= l1t::Type3s;
	type = "3";
	nrObj = 3;
    } else if (condCalo.getType() == esConditionType::QuadJet) {
        caloObjType = gtJet;
	cType= l1t::Type4s;
	type = "4";
	nrObj = 4;			
    } else if (condCalo.getType() == esConditionType::SingleTau) {
        caloObjType = gtTau;
	cType= l1t::Type1s;
	type = "1_s";
	nrObj = 1;
    } else if (condCalo.getType() == esConditionType::DoubleTau) {
        caloObjType = gtTau;
	cType= l1t::Type2s;
	type = "2_s";
	nrObj = 2;	
    } else if (condCalo.getType() == esConditionType::TripleTau) {
        caloObjType = gtTau;
	cType= l1t::Type3s;
	type = "3";
	nrObj = 3;
    } else if (condCalo.getType() == esConditionType::QuadTau) {
        caloObjType = gtTau;
	cType= l1t::Type4s;
	type = "4";
	nrObj = 4;		
    } else {
        edm::LogError("TriggerMenuParser") << "Wrong particle for calo-condition ("
            << particle << ")" << std::endl;
        return false;
    }

//    std::string str_etComparison = l1t2string( condCalo.comparison_operator() );

    if (nrObj < 0) {
        edm::LogError("TriggerMenuParser") << "Unknown type for calo-condition (" << type
            << ")" << "\nCan not determine number of trigger objects. " << std::endl;
        return false;
    }

    // get values

    // temporary storage of the parameters
    std::vector<CaloTemplate::ObjectParameter> objParameter(nrObj);

    //BLW TO DO:  Can this be dropped?
    CaloTemplate::CorrelationParameter corrParameter;

    // need at least one value for deltaPhiRange
    std::vector<boost::uint64_t> tmpValues((nrObj > 1) ? nrObj : 1);
    tmpValues.reserve( nrObj );


    if( int(condCalo.getObjects().size())!=nrObj ){
      edm::LogError("TriggerMenuParser") << " condCalo objects: nrObj = " << nrObj
						    << "condCalo.getObjects().size() = " 
						    << condCalo.getObjects().size()
						    << std::endl;
      return false;
    }


//    std::string str_condCalo = "";
//    boost::uint64_t tempUIntH, tempUIntL;
//    boost::uint64_t dst;
    int cnt = 0;

// BLW TO DO: These needs to the added to the object rather than the whole condition.
    int relativeBx = 0;
    bool gEq = false;
    
// Loop over objects and extract the cuts on the objects
    const std::vector<esObject>& objects = condCalo.getObjects();
    for (size_t jj = 0; jj < objects.size(); jj++) {   

       const esObject object = objects.at(jj);
       gEq =  (object.getComparisonOperator() == esComparisonOperator::GE);

//  BLW TO DO: This needs to be added to the Object Parameters   
       relativeBx = object.getBxOffset();

//  Loop over the cuts for this object
        int upperThresholdInd = -1;
	int lowerThresholdInd = 0;
        int cntEta = 0;
        unsigned int etaWindow1Lower=-1, etaWindow1Upper=-1, etaWindow2Lower=-1, etaWindow2Upper=-1;
	int cntPhi = 0;
	unsigned int phiWindow1Lower=-1, phiWindow1Upper=-1, phiWindow2Lower=-1, phiWindow2Upper=-1;
        int isolationLUT = 0xF; //default is to ignore isolation unless specified.
	int qualityLUT   = 0xF; //default is to ignore quality unless specified.	
		
	
        const std::vector<esCut>& cuts = object.getCuts();
        for (size_t kk = 0; kk < cuts.size(); kk++)
        {
          const esCut cut = cuts.at(kk); 
	 
	  switch(cut.getCutType()){
	     case esCutType::Threshold:
	       lowerThresholdInd = cut.getMinimum().index;
	       upperThresholdInd = cut.getMaximum().index;
	       break;
	     case esCutType::Eta: {
	       
                 if(cntEta == 0) {
		    etaWindow1Lower = cut.getMinimum().index;
		    etaWindow1Upper = cut.getMaximum().index;
		 } else if(cntEta == 1) {
		    etaWindow2Lower = cut.getMinimum().index;
		    etaWindow2Upper = cut.getMaximum().index;
                 } else {
        	   edm::LogError("TriggerMenuParser") << "Too Many Eta Cuts for calo-condition ("
        	       << particle << ")" << std::endl;
        	   return false;
		 }
		 cntEta++; 

	       } break;
	       
	     case esCutType::Phi: {

                if(cntPhi == 0) {
		    phiWindow1Lower = cut.getMinimum().index;
		    phiWindow1Upper = cut.getMaximum().index;
		 } else if(cntPhi == 1) {
		    phiWindow2Lower = cut.getMinimum().index;
		    phiWindow2Upper = cut.getMaximum().index;
                 } else {
        	   edm::LogError("TriggerMenuParser") << "Too Many Phi Cuts for calo-condition ("
        	       << particle << ")" << std::endl;
        	   return false;
		 }
		 cntPhi++; 

	       }break;
	       
	     case esCutType::Charge: {

       	         edm::LogError("TriggerMenuParser") << "No charge cut for calo-condition ("
        	       << particle << ")" << std::endl;
        	   return false;

	       }break;
	     case esCutType::Quality: {
             
	       qualityLUT = l1tstr2int(cut.getData());

	       }break;
	     case esCutType::Isolation: {

               isolationLUT = l1tstr2int(cut.getData());
		       
	       } break;
	     default:
	       break; 	       	       	       	       
	  } //end switch 
	  
        } //end loop over cuts

// Fill the object parameters
	objParameter[cnt].etHighThreshold = upperThresholdInd;
	objParameter[cnt].etLowThreshold  = lowerThresholdInd;
	objParameter[cnt].etaWindow1Lower     = etaWindow1Lower;
	objParameter[cnt].etaWindow1Upper     = etaWindow1Upper;
	objParameter[cnt].etaWindow2Lower = etaWindow2Lower;
	objParameter[cnt].etaWindow2Upper = etaWindow2Upper;
	objParameter[cnt].phiWindow1Lower     = phiWindow1Lower;
	objParameter[cnt].phiWindow1Upper     = phiWindow1Upper;
	objParameter[cnt].phiWindow2Lower = phiWindow2Lower;
	objParameter[cnt].phiWindow2Upper = phiWindow2Upper;
        objParameter[cnt].isolationLUT       = isolationLUT;
        objParameter[cnt].qualityLUT         = qualityLUT; //TO DO: Must add 

      // Output for debugging
      LogDebug("TriggerMenuParser") 
	<< "\n      Calo ET high thresholds (hex) for calo object " << caloObjType << " " << cnt << " = "
	<< std::hex << objParameter[cnt].etLowThreshold << " - " << objParameter[cnt].etHighThreshold 
	<< "\n      etaWindow Lower / Upper for calo object " << cnt << " = 0x"
	<< objParameter[cnt].etaWindow1Lower << " / 0x" << objParameter[cnt].etaWindow1Upper
	<< "\n      etaWindowVeto Lower / Upper for calo object " << cnt << " = 0x"
	<< objParameter[cnt].etaWindow2Lower << " / 0x" << objParameter[cnt].etaWindow2Upper
	<< "\n      phiWindow Lower / Upper for calo object " << cnt << " = 0x"
	<< objParameter[cnt].phiWindow1Lower << " / 0x" << objParameter[cnt].phiWindow1Upper
	<< "\n      phiWindowVeto Lower / Upper for calo object " << cnt << " = 0x"
	<< objParameter[cnt].phiWindow2Lower << " / 0x" << objParameter[cnt].phiWindow2Upper
	<< "\n      Isolation LUT for calo object " << cnt << " = 0x"
	<< objParameter[cnt].isolationLUT
	<< "\n      Quality LUT for calo object " << cnt << " = 0x"
	<< objParameter[cnt].qualityLUT << std::dec
	<< std::endl;

      cnt++;
    } //end loop over objects



    // object types - all same caloObjType
    std::vector<GlobalObject> objType(nrObj, caloObjType);


    

    // now create a new calo condition
    CaloTemplate caloCond(name);

    caloCond.setCondType(cType);
    caloCond.setObjectType(objType);
    
    //BLW TO DO: This needs to be added to the object rather than the whole condition
    caloCond.setCondGEq(gEq);
    caloCond.setCondChipNr(chipNr);
    
    //BLW TO DO: This needs to be added to the object rather than the whole condition
    caloCond.setCondRelativeBx(relativeBx);

    caloCond.setConditionParameter(objParameter, corrParameter);

    if (edm::isDebugEnabled() ) {

        std::ostringstream myCoutStream;
        caloCond.print(myCoutStream);
        LogTrace("TriggerMenuParser") << myCoutStream.str() << "\n" << std::endl;

    }


    // insert condition into the map
    if ( !insertConditionIntoMap(caloCond, chipNr)) {

        edm::LogError("TriggerMenuParser")
                << "    Error: duplicate condition (" << name << ")"
                << std::endl;

        return false;
    }
    else {

        if (corrFlag) {
            (m_corCaloTemplate[chipNr]).push_back(caloCond);
       }
        else {
            (m_vecCaloTemplate[chipNr]).push_back(caloCond);
        }

    }


    //
    return true;
}



/**
 * parseCalo Parse a calo condition and insert an entry to the conditions map
 *
 * @param node The corresponding node.
 * @param name The name of the condition.
 * @param chipNr The number of the chip this condition is located.
 *
 * @return "true" if succeeded, "false" if an error occurred.
 *
 */

bool l1t::TriggerMenuParser::parseCaloCorr(const tmeventsetup::esObject* corrCalo,
        unsigned int chipNr) {


//    XERCES_CPP_NAMESPACE_USE
    using namespace tmeventsetup;
    
    // get condition, particle name and type name

    std::string condition = "calo";
    std::string particle = "test-fix" ;
    std::string type = l1t2string( corrCalo->getType() );
    std::string name = l1t2string( corrCalo->getName() );

    LogDebug("TriggerMenuParser")
      << "\n ****************************************** " 
      << "\n      (in parseCalo) " 
      << "\n condition = " << condition 
      << "\n particle  = " << particle 
      << "\n type      = " << type 
      << "\n name      = " << name 
      << std::endl;


    // determine object type type
    // BLW TO DO:  Can this object type wait and be done later in the parsing. Or done differently completely..
    GlobalObject caloObjType;
    int nrObj = 1;
    type = "1_s";
    GtConditionType cType = l1t::Type1s;


    if (corrCalo->getType() == esObjectType::Egamma) {
        caloObjType = gtEG;
    } else if (corrCalo->getType() == esObjectType::Jet) {
        caloObjType = gtJet;
    } else if (corrCalo->getType() == esObjectType::Tau) {
        caloObjType = gtTau;
    } else {
        edm::LogError("TriggerMenuParser") << "Wrong particle for calo-condition ("
            << particle << ")" << std::endl;
        return false;
    }


//    std::string str_etComparison = l1t2string( condCalo.comparison_operator() );

    if (nrObj < 0) {
        edm::LogError("TriggerMenuParser") << "Unknown type for calo-condition (" << type
            << ")" << "\nCan not determine number of trigger objects. " << std::endl;
        return false;
    }

    // get values

    // temporary storage of the parameters
    std::vector<CaloTemplate::ObjectParameter> objParameter(nrObj);

    //BLW TO DO:  Can this be dropped?
    CaloTemplate::CorrelationParameter corrParameter;

    // need at least one value for deltaPhiRange
    std::vector<boost::uint64_t> tmpValues((nrObj > 1) ? nrObj : 1);
    tmpValues.reserve( nrObj );



// BLW TO DO: These needs to the added to the object rather than the whole condition.
    int relativeBx = 0;
    bool gEq = false;
    

    gEq =  (corrCalo->getComparisonOperator() == esComparisonOperator::GE);

//  BLW TO DO: This needs to be added to the Object Parameters   
    relativeBx = corrCalo->getBxOffset();

//  Loop over the cuts for this object
     int upperThresholdInd = -1;
     int lowerThresholdInd = 0;
     int cntEta = 0;
     unsigned int etaWindow1Lower=-1, etaWindow1Upper=-1, etaWindow2Lower=-1, etaWindow2Upper=-1;
     int cntPhi = 0;
     unsigned int phiWindow1Lower=-1, phiWindow1Upper=-1, phiWindow2Lower=-1, phiWindow2Upper=-1;
     int isolationLUT = 0xF; //default is to ignore isolation unless specified.
     int qualityLUT   = 0xF; //default is to ignore quality unless specified.	


     const std::vector<esCut>& cuts = corrCalo->getCuts();
     for (size_t kk = 0; kk < cuts.size(); kk++)
     {
       const esCut cut = cuts.at(kk); 

       switch(cut.getCutType()){
	  case esCutType::Threshold:
	    lowerThresholdInd = cut.getMinimum().index;
	    upperThresholdInd = cut.getMaximum().index;
	    break;
	  case esCutType::Eta: {

              if(cntEta == 0) {
		 etaWindow1Lower = cut.getMinimum().index;
		 etaWindow1Upper = cut.getMaximum().index;
	      } else if(cntEta == 1) {
		 etaWindow2Lower = cut.getMinimum().index;
		 etaWindow2Upper = cut.getMaximum().index;
              } else {
        	edm::LogError("TriggerMenuParser") << "Too Many Eta Cuts for calo-condition ("
        	    << particle << ")" << std::endl;
        	return false;
	      }
	      cntEta++; 

	    } break;

	  case esCutType::Phi: {

             if(cntPhi == 0) {
		 phiWindow1Lower = cut.getMinimum().index;
		 phiWindow1Upper = cut.getMaximum().index;
	      } else if(cntPhi == 1) {
		 phiWindow2Lower = cut.getMinimum().index;
		 phiWindow2Upper = cut.getMaximum().index;
              } else {
        	edm::LogError("TriggerMenuParser") << "Too Many Phi Cuts for calo-condition ("
        	    << particle << ")" << std::endl;
        	return false;
	      }
	      cntPhi++; 

	    }break;

	  case esCutType::Charge: {

       	      edm::LogError("TriggerMenuParser") << "No charge cut for calo-condition ("
        	    << particle << ")" << std::endl;
        	return false;

	    }break;
	  case esCutType::Quality: {

	    qualityLUT = l1tstr2int(cut.getData());

	    }break;
	  case esCutType::Isolation: {

            isolationLUT = l1tstr2int(cut.getData());

	    } break;
	  default:
	    break; 	       	       	       	       
       } //end switch 

     } //end loop over cuts

// Fill the object parameters
     objParameter[0].etLowThreshold  = lowerThresholdInd;
     objParameter[0].etHighThreshold = upperThresholdInd;
     objParameter[0].etaWindow1Lower = etaWindow1Lower;
     objParameter[0].etaWindow1Upper = etaWindow1Upper;
     objParameter[0].etaWindow2Lower = etaWindow2Lower;
     objParameter[0].etaWindow2Upper = etaWindow2Upper;
     objParameter[0].phiWindow1Lower = phiWindow1Lower;
     objParameter[0].phiWindow1Upper = phiWindow1Upper;
     objParameter[0].phiWindow2Lower = phiWindow2Lower;
     objParameter[0].phiWindow2Upper = phiWindow2Upper;
     objParameter[0].isolationLUT    = isolationLUT;
     objParameter[0].qualityLUT      = qualityLUT; //TO DO: Must add 

   // Output for debugging
   LogDebug("TriggerMenuParser") 
     << "\n      Calo ET high threshold (hex) for calo object " << caloObjType << " "  << " = "
     << std::hex << objParameter[0].etLowThreshold << " - " << objParameter[0].etHighThreshold 
     << "\n      etaWindow Lower / Upper for calo object "  << " = 0x"
     << objParameter[0].etaWindow1Lower << " / 0x" << objParameter[0].etaWindow1Upper
     << "\n      etaWindowVeto Lower / Upper for calo object "  << " = 0x"
     << objParameter[0].etaWindow2Lower << " / 0x" << objParameter[0].etaWindow2Upper
     << "\n      phiWindow Lower / Upper for calo object "  << " = 0x"
     << objParameter[0].phiWindow1Lower << " / 0x" << objParameter[0].phiWindow1Upper
     << "\n      phiWindowVeto Lower / Upper for calo object "  << " = 0x"
     << objParameter[0].phiWindow2Lower << " / 0x" << objParameter[0].phiWindow2Upper
     << "\n      Isolation LUT for calo object "  << " = 0x"
     << objParameter[0].isolationLUT
     << "\n      Quality LUT for calo object "  << " = 0x"
     << objParameter[0].qualityLUT << std::dec
     << std::endl;





    // object types - all same caloObjType
    std::vector<GlobalObject> objType(nrObj, caloObjType);


    

    // now create a new calo condition
    CaloTemplate caloCond(name);

    caloCond.setCondType(cType);
    caloCond.setObjectType(objType);
    
    //BLW TO DO: This needs to be added to the object rather than the whole condition
    caloCond.setCondGEq(gEq);
    caloCond.setCondChipNr(chipNr);
    
    //BLW TO DO: This needs to be added to the object rather than the whole condition
    caloCond.setCondRelativeBx(relativeBx);

    caloCond.setConditionParameter(objParameter, corrParameter);

    if (edm::isDebugEnabled() ) {

        std::ostringstream myCoutStream;
        caloCond.print(myCoutStream);
        LogTrace("TriggerMenuParser") << myCoutStream.str() << "\n" << std::endl;

    }

/*
    // insert condition into the map
    if ( !insertConditionIntoMap(caloCond, chipNr)) {

        edm::LogError("TriggerMenuParser")
                << "    Error: duplicate condition (" << name << ")"
                << std::endl;

        return false;
    }
    else {
            (m_corCaloTemplate[chipNr]).push_back(caloCond);
    }
*/
    (m_corCaloTemplate[chipNr]).push_back(caloCond);

    //
    return true;
}



/**
 * parseEnergySum Parse an "energy sum" condition and insert an entry to the conditions map
 *
 * @param node The corresponding node.
 * @param name The name of the condition.
 * @param chipNr The number of the chip this condition is located.
 *
 * @return "true" if succeeded, "false" if an error occurred.
 *
 */

bool l1t::TriggerMenuParser::parseEnergySum(tmeventsetup::esCondition condEnergySum,
        unsigned int chipNr, const bool corrFlag) {


//    XERCES_CPP_NAMESPACE_USE
     using namespace tmeventsetup;
     
    // get condition, particle name and type name

    std::string condition = "calo";
    std::string type = l1t2string( condEnergySum.getType() );
    std::string name = l1t2string( condEnergySum.getName() );

    LogDebug("TriggerMenuParser")
      << "\n ****************************************** " 
      << "\n      (in parseEnergySum) " 
      << "\n condition = " << condition 
      << "\n type      = " << type 
      << "\n name      = " << name 
      << std::endl;



    // determine object type type
    GlobalObject energySumObjType;
    GtConditionType cType;

    if( condEnergySum.getType() == esConditionType::MissingEt ){
      energySumObjType = GlobalObject::gtETM;
      cType = TypeETM;
    }
    else if( condEnergySum.getType() == esConditionType::TotalEt ){
      energySumObjType = GlobalObject::gtETT;
      cType = TypeETT;
    }
    else if( condEnergySum.getType() == esConditionType::TotalHt ){
      energySumObjType = GlobalObject::gtHTT;
      cType = TypeHTT;
    }
    else if( condEnergySum.getType() == esConditionType::MissingHt ){
      energySumObjType = GlobalObject::gtHTM;
      cType = TypeHTM;
    }
/*    else if( condEnergySum.getType() == esConditionType::MissingEt2 ){
      energySumObjType = GlobalObject::gtETM2;
      cType = TypeETM2;
    }
    else if( condEnergySum.getType() == esConditionType::MinBias ){
      energySumObjType = GlobalObject::gtMinBias;
      cType = TypeMinBias;
    }  */      
    else {
      edm::LogError("TriggerMenuParser")
	<< "Wrong type for energy-sum condition (" << type
	<< ")" << std::endl;
      return false;
    }



    // global object
    int nrObj = 1;

//    std::string str_etComparison = l1t2string( condEnergySum.comparison_operator() );

    // get values

    // temporary storage of the parameters
    std::vector<EnergySumTemplate::ObjectParameter> objParameter(nrObj);


    int cnt = 0;

// BLW TO DO: These needs to the added to the object rather than the whole condition.
    int relativeBx = 0;
    bool gEq = false;
    
//    l1t::EnergySumsObjectRequirement objPar = condEnergySum.objectRequirement();

// Loop over objects and extract the cuts on the objects
    const std::vector<esObject>& objects = condEnergySum.getObjects();
    for (size_t jj = 0; jj < objects.size(); jj++) {   

       const esObject object = objects.at(jj);
       gEq =  (object.getComparisonOperator() == esComparisonOperator::GE);

//  BLW TO DO: This needs to be added to the Object Parameters   
       relativeBx = object.getBxOffset();

//  Loop over the cuts for this object
        int lowerThresholdInd = 0;
	int upperThresholdInd = -1;
	int cntPhi = 0;
	unsigned int phiWindow1Lower=-1, phiWindow1Upper=-1, phiWindow2Lower=-1, phiWindow2Upper=-1;
		
	
        const std::vector<esCut>& cuts = object.getCuts();
        for (size_t kk = 0; kk < cuts.size(); kk++)
        {
          const esCut cut = cuts.at(kk); 
	 
	  switch(cut.getCutType()){
	     case esCutType::Threshold:
	       lowerThresholdInd = cut.getMinimum().index;
	       upperThresholdInd = cut.getMaximum().index;
	       break;

	     case esCutType::Eta: 
	       break;
	       
	     case esCutType::Phi: {

                if(cntPhi == 0) {
		    phiWindow1Lower = cut.getMinimum().index;
		    phiWindow1Upper = cut.getMaximum().index;
		 } else if(cntPhi == 1) {
		    phiWindow2Lower = cut.getMinimum().index;
		    phiWindow2Upper = cut.getMaximum().index;
                 } else {
        	   edm::LogError("TriggerMenuParser") << "Too Many Phi Cuts for esum-condition ("
        	       << type << ")" << std::endl;
        	   return false;
		 }
		 cntPhi++; 

	       }
	       break;
	       
	     default:
	       break; 	       	       	       	       
	  } //end switch 
	  
        } //end loop over cuts



    // Fill the object parameters
    objParameter[cnt].etLowThreshold   = lowerThresholdInd;
    objParameter[cnt].etHighThreshold = upperThresholdInd;
    objParameter[cnt].phiWindow1Lower = phiWindow1Lower;
    objParameter[cnt].phiWindow1Upper = phiWindow1Upper;
    objParameter[cnt].phiWindow2Lower = phiWindow2Lower;
    objParameter[cnt].phiWindow2Upper = phiWindow2Upper;

      
    // Output for debugging
    LogDebug("TriggerMenuParser") 
      << "\n      EnergySum ET high threshold (hex) for energy sum object " << cnt << " = "
      << std::hex << objParameter[cnt].etLowThreshold << " - " << objParameter[cnt].etHighThreshold 
      << "\n      phiWindow Lower / Upper for calo object " << cnt << " = 0x"
      << objParameter[cnt].phiWindow1Lower << " / 0x" << objParameter[cnt].phiWindow1Upper
      << "\n      phiWindowVeto Lower / Upper for calo object " << cnt << " = 0x"
      << objParameter[cnt].phiWindow2Lower << " / 0x" << objParameter[cnt].phiWindow2Upper << std::dec
      << std::endl;

      cnt++;
    } //end loop over objects
    
    // object types - all same energySumObjType
    std::vector<GlobalObject> objType(nrObj, energySumObjType);

    // now create a new energySum condition

    EnergySumTemplate energySumCond(name);

    energySumCond.setCondType(cType);
    energySumCond.setObjectType(objType);
    energySumCond.setCondGEq(gEq);
    energySumCond.setCondChipNr(chipNr);
    energySumCond.setCondRelativeBx(relativeBx);

    energySumCond.setConditionParameter(objParameter);

    if (edm::isDebugEnabled() ) {

        std::ostringstream myCoutStream;
        energySumCond.print(myCoutStream);
        LogTrace("TriggerMenuParser") << myCoutStream.str() << "\n" << std::endl;

    }

    // insert condition into the map
    if ( !insertConditionIntoMap(energySumCond, chipNr)) {

        edm::LogError("TriggerMenuParser")
                << "    Error: duplicate condition (" << name << ")"
                << std::endl;

        return false;
    }
    else {

        if (corrFlag) {
            (m_corEnergySumTemplate[chipNr]).push_back(energySumCond);

        }
        else {
            (m_vecEnergySumTemplate[chipNr]).push_back(energySumCond);
        }

    }



    //
    return true;
}


/**
 * parseEnergySum Parse an "energy sum" condition and insert an entry to the conditions map
 *
 * @param node The corresponding node.
 * @param name The name of the condition.
 * @param chipNr The number of the chip this condition is located.
 *
 * @return "true" if succeeded, "false" if an error occurred.
 *
 */

bool l1t::TriggerMenuParser::parseEnergySumCorr(const tmeventsetup::esObject* corrESum,
        unsigned int chipNr) {


//    XERCES_CPP_NAMESPACE_USE
     using namespace tmeventsetup;
     
    // get condition, particle name and type name

    std::string condition = "calo";
    std::string type = l1t2string( corrESum->getType() );
    std::string name = l1t2string( corrESum->getName() );

    LogDebug("TriggerMenuParser")
      << "\n ****************************************** " 
      << "\n      (in parseEnergySum) " 
      << "\n condition = " << condition 
      << "\n type      = " << type 
      << "\n name      = " << name 
      << std::endl;



    // determine object type type
    GlobalObject energySumObjType;
    GtConditionType cType;

    if( corrESum->getType()== esObjectType::ETM ){
      energySumObjType = GlobalObject::gtETM;
      cType = TypeETM;
    }
    else if( corrESum->getType()== esObjectType::HTM ){
      energySumObjType = GlobalObject::gtHTM;
      cType = TypeHTM;
    }
/*    else if( corrESum->getType()== esObjectType::ETM2 ){
      energySumObjType = GlobalObject::gtETM2;
      cType = TypeETM2;
    } */
    else {
      edm::LogError("TriggerMenuParser")
	<< "Wrong type for energy-sum correclation condition (" << type
	<< ")" << std::endl;
      return false;
    }



    // global object
    int nrObj = 1;

//    std::string str_etComparison = l1t2string( condEnergySum.comparison_operator() );

    // get values

    // temporary storage of the parameters
    std::vector<EnergySumTemplate::ObjectParameter> objParameter(nrObj);


    int cnt = 0;

// BLW TO DO: These needs to the added to the object rather than the whole condition.
    int relativeBx = 0;
    bool gEq = false;
    
//    l1t::EnergySumsObjectRequirement objPar = condEnergySum.objectRequirement();


   gEq =  (corrESum->getComparisonOperator() == esComparisonOperator::GE);

//  BLW TO DO: This needs to be added to the Object Parameters   
   relativeBx = corrESum->getBxOffset();

//  Loop over the cuts for this object
    int lowerThresholdInd = 0;
    int upperThresholdInd = -1;
    int cntPhi = 0;
    unsigned int phiWindow1Lower=-1, phiWindow1Upper=-1, phiWindow2Lower=-1, phiWindow2Upper=-1;


    const std::vector<esCut>& cuts = corrESum->getCuts();
    for (size_t kk = 0; kk < cuts.size(); kk++)
    {
      const esCut cut = cuts.at(kk); 

      switch(cut.getCutType()){
	 case esCutType::Threshold:
	   lowerThresholdInd = cut.getMinimum().index;
	   upperThresholdInd = cut.getMaximum().index;
	   break;

	 case esCutType::Eta: 
	   break;

	 case esCutType::Phi: {

            if(cntPhi == 0) {
		phiWindow1Lower = cut.getMinimum().index;
		phiWindow1Upper = cut.getMaximum().index;
	     } else if(cntPhi == 1) {
		phiWindow2Lower = cut.getMinimum().index;
		phiWindow2Upper = cut.getMaximum().index;
             } else {
               edm::LogError("TriggerMenuParser") << "Too Many Phi Cuts for esum-condition ("
        	   << type << ")" << std::endl;
               return false;
	     }
	     cntPhi++; 

	   }
	   break;

	 default:
	   break; 	       	       	       	       
      } //end switch 

    } //end loop over cuts



    // Fill the object parameters
    objParameter[0].etLowThreshold  = lowerThresholdInd;
    objParameter[0].etHighThreshold = upperThresholdInd;
    objParameter[0].phiWindow1Lower = phiWindow1Lower;
    objParameter[0].phiWindow1Upper = phiWindow1Upper;
    objParameter[0].phiWindow2Lower = phiWindow2Lower;
    objParameter[0].phiWindow2Upper = phiWindow2Upper;

      
    // Output for debugging
    LogDebug("TriggerMenuParser") 
      << "\n      EnergySum ET high threshold (hex) for energy sum object " << cnt << " = "
      << std::hex << objParameter[0].etLowThreshold << " - " << objParameter[0].etLowThreshold 
      << "\n      phiWindow Lower / Upper for calo object " << cnt << " = 0x"
      << objParameter[0].phiWindow1Lower << " / 0x" << objParameter[0].phiWindow1Upper
      << "\n      phiWindowVeto Lower / Upper for calo object " << cnt << " = 0x"
      << objParameter[0].phiWindow2Lower << " / 0x" << objParameter[0].phiWindow2Upper << std::dec
      << std::endl;

    
    // object types - all same energySumObjType
    std::vector<GlobalObject> objType(nrObj, energySumObjType);

    // now create a new energySum condition

    EnergySumTemplate energySumCond(name);

    energySumCond.setCondType(cType);
    energySumCond.setObjectType(objType);
    energySumCond.setCondGEq(gEq);
    energySumCond.setCondChipNr(chipNr);
    energySumCond.setCondRelativeBx(relativeBx);

    energySumCond.setConditionParameter(objParameter);

    if (edm::isDebugEnabled() ) {

        std::ostringstream myCoutStream;
        energySumCond.print(myCoutStream);
        LogTrace("TriggerMenuParser") << myCoutStream.str() << "\n" << std::endl;

    }
/*
    // insert condition into the map
    if ( !insertConditionIntoMap(energySumCond, chipNr)) {

        edm::LogError("TriggerMenuParser")
                << "    Error: duplicate condition (" << name << ")"
                << std::endl;

        return false;
    }
    else {

       (m_corEnergySumTemplate[chipNr]).push_back(energySumCond);

    }
*/
   (m_corEnergySumTemplate[chipNr]).push_back(energySumCond);


    //
    return true;
}



/**
 * parseExternal Parse an External condition and
 * insert an entry to the conditions map
 *
 * @param node The corresponding node.
 * @param name The name of the condition.
 * @param chipNr The number of the chip this condition is located.
 *
 * @return "true" if succeeded, "false" if an error occurred.
 *
 */

bool l1t::TriggerMenuParser::parseExternal(tmeventsetup::esCondition condExt,
        unsigned int chipNr) {


    using namespace tmeventsetup;

    
    // get condition, particle name and type name
    std::string condition = "ext";     
    std::string particle = "test-fix";
    std::string type = l1t2string( condExt.getType() );
    std::string name = l1t2string( condExt.getName() );


    LogDebug("TriggerMenuParser")
      << "\n ****************************************** " 
      << "\n      (in parseExternal) " 
      << "\n condition = " << condition 
      << "\n particle  = " << particle 
      << "\n type      = " << type 
      << "\n name      = " << name 
      << std::endl;


    // object type and condition type
    // object type - irrelevant for External conditions
    GtConditionType cType = TypeExternal;

    int relativeBx = 0;    
    unsigned int channelID = 0;

    // Get object for External conditions
    const std::vector<esObject>& objects = condExt.getObjects();
    for (size_t jj = 0; jj < objects.size(); jj++) {   

       const esObject object = objects.at(jj);
       if(object.getType() == esObjectType::EXT) {
          relativeBx = object.getBxOffset();
          channelID = object.getExternalChannelId();
       }
    }   


    // set the boolean value for the ge_eq mode - irrelevant for External conditions
    bool gEq = false;

    // now create a new External condition
    ExternalTemplate externalCond(name);

    externalCond.setCondType(cType);
    externalCond.setCondGEq(gEq);
    externalCond.setCondChipNr(chipNr);
    externalCond.setCondRelativeBx(relativeBx);
    externalCond.setExternalChannel(channelID);

    LogTrace("TriggerMenuParser") 
             << externalCond << "\n" << std::endl;

    // insert condition into the map
    if ( !insertConditionIntoMap(externalCond, chipNr)) {

        edm::LogError("TriggerMenuParser")
            << "    Error: duplicate condition (" << name
            << ")" << std::endl;

        return false;
    } else {

        (m_vecExternalTemplate[chipNr]).push_back(externalCond);

    }
         
    return true;
}


/**
 * parseCorrelation Parse a correlation condition and
 * insert an entry to the conditions map
 *
 * @param node The corresponding node.
 * @param name The name of the condition.
 * @param chipNr The number of the chip this condition is located.
 *
 * @return "true" if succeeded, "false" if an error occurred.
 *
 */

bool l1t::TriggerMenuParser::parseCorrelation(
        tmeventsetup::esCondition corrCond,
        unsigned int chipNr) {

    using namespace tmeventsetup;

    std::string condition = "corr";
    std::string particle = "test-fix" ;
    std::string type = l1t2string( corrCond.getType() );
    std::string name = l1t2string( corrCond.getName() );

    LogDebug("TriggerMenuParser") << " ****************************************** " << std::endl
     << "     (in parseCorrelation) " << std::endl
     << " condition = " << condition << std::endl
     << " particle  = " << particle << std::endl
     << " type      = " << type << std::endl
     << " name      = " << name << std::endl;


   

    // create a new correlation condition
    CorrelationTemplate correlationCond(name);

    // check that the condition does not exist already in the map
    if ( !insertConditionIntoMap(correlationCond, chipNr)) {

        edm::LogError("TriggerMenuParser")
                << "    Error: duplicate correlation condition (" << name << ")"
                << std::endl;

        return false;
    }


// Define some of the quantities to store the parased information

    // condition type BLW  (Do we change this to the type of correlation condition?)
    GtConditionType cType = l1t::Type2cor;

    // two objects (for sure)
    const int nrObj = 2;

    // object types and greater equal flag - filled in the loop
    int intGEq[nrObj] = { -1, -1 };
    std::vector<GlobalObject> objType(nrObj);   //BLW do we want to define these as a different type?
    std::vector<GtConditionCategory> condCateg(nrObj);   //BLW do we want to change these categories

    // correlation flag and index in the cor*vector
    const bool corrFlag = true;
    int corrIndexVal[nrObj] = { -1, -1 };


    // Storage of the correlation selection
    CorrelationTemplate::CorrelationParameter corrParameter;
    corrParameter.chargeCorrelation = 1; //ignore charge correlation

// Get the correlation Cuts on the legs
      int cutType = 0;  
      const std::vector<esCut>& cuts = corrCond.getCuts();      
      for (size_t jj = 0; jj < cuts.size(); jj++)
      {
        const esCut cut = cuts.at(jj);

	if(cut.getCutType() == esCutType::ChargeCorrelation) { 
	   if( cut.getData()=="ls" )      corrParameter.chargeCorrelation = 2;
	   else if( cut.getData()=="os" ) corrParameter.chargeCorrelation = 4;
	   else corrParameter.chargeCorrelation = 1; //ignore charge correlation
        } else {

// 
//  Unitl utm has method to calculate these, do the integer value calculation with precision.
//
          double minV = cut.getMinimum().value;
	  double maxV = cut.getMaximum().value;
	  
	  //Scale down very large numbers out of xml
	  if(maxV > 1.0e6) maxV = 1.0e6;
	  
	  if(cut.getCutType() == esCutType::DeltaEta) {
	     //std::cout << "DeltaEta Cut minV = " << minV << " Max = " << maxV << " precMin = " << cut.getMinimum().index << " precMax = " << cut.getMaximum().index << std::endl;
	     corrParameter.minEtaCutValue = (long long)(minV * pow(10.,cut.getMinimum().index)); 
	     corrParameter.maxEtaCutValue = (long long)(maxV * pow(10.,cut.getMaximum().index)); 
	     corrParameter.precEtaCut     = cut.getMinimum().index;	     
	     cutType = cutType | 0x1;
	  } else if (cut.getCutType() == esCutType::DeltaPhi) {
	     //std::cout << "DeltaPhi Cut minV = " << minV << " Max = " << maxV << " precMin = " << cut.getMinimum().index << " precMax = " << cut.getMaximum().index << std::endl;
	     corrParameter.minPhiCutValue = (long long)(minV * pow(10.,cut.getMinimum().index));
	     corrParameter.maxPhiCutValue = (long long)(maxV * pow(10.,cut.getMaximum().index));
	     corrParameter.precPhiCut     = cut.getMinimum().index;
	     cutType = cutType | 0x2;
	  } else if (cut.getCutType() == esCutType::DeltaR) {
	     //std::cout << "DeltaR Cut minV = " << minV << " Max = " << maxV << " precMin = " << cut.getMinimum().index << " precMax = " << cut.getMaximum().index << std::endl;
	     corrParameter.minDRCutValue = (long long)(minV * pow(10.,cut.getMinimum().index));
	     corrParameter.maxDRCutValue = (long long)(maxV * pow(10.,cut.getMaximum().index));
	     corrParameter.precDRCut     = cut.getMinimum().index;
	     cutType = cutType | 0x4; 
	  } else if (cut.getCutType() == esCutType::Mass) {
	     //std::cout << "Mass Cut minV = " << minV << " Max = " << maxV << " precMin = " << cut.getMinimum().index << " precMax = " << cut.getMaximum().index << std::endl;	  
	     corrParameter.minMassCutValue = (long long)(minV * pow(10.,cut.getMinimum().index));
	     corrParameter.maxMassCutValue = (long long)(maxV * pow(10.,cut.getMaximum().index));
	     corrParameter.precMassCut     = cut.getMinimum().index;
	     cutType = cutType | 0x8; 
          }
	}  

      }
      corrParameter.corrCutType = cutType;

// Get the two objects that form the legs
      const std::vector<esObject>& objects = corrCond.getObjects();
      if(objects.size() != 2) {
            edm::LogError("TriggerMenuParser")
                    << "incorrect number of objects for the correlation condition " << name << " corrFlag " << corrFlag << std::endl;
            return false;      
      }
      
// loop over legs      
      for (size_t jj = 0; jj < objects.size(); jj++)
      {
        const esObject object = objects.at(jj);
/*      std::cout << "      obj name = " << object->getName() << "\n";
        std::cout << "      obj type = " << object->getType() << "\n";
        std::cout << "      obj op = " << object->getComparisonOperator() << "\n";
        std::cout << "      obj bx = " << object->getBxOffset() << "\n";
*/

// check the leg type
        if(object.getType() == esObjectType::Muon) {
	  // we have a muon  

/*
          //BLW Hold on to this code we may need to go back to it at some point.
	  // Now we are putting ALL leg conditions into the vector (so there are duplicates)
	  // This is potentially a place to slim down the code.  Note: We currently evaluate the
	  // conditions every time, so even if we put the condition in the vector once, we would 
	  // still evaluate it multiple times.  This is a place for optimization.
	   	                  
              parseMuonCorr(&object,chipNr);
	      corrIndexVal[jj] = (m_corMuonTemplate[chipNr]).size() - 1;	     
	      
          } else {
	     LogDebug("TriggerMenuParser") << "Not Adding Correlation Muon Condition to Map...looking for the condition in Muon Cor Vector" << std::endl;
	     bool found = false;
	     int index = 0;
	     while(!found && index<(int)((m_corMuonTemplate[chipNr]).size()) ) {
	         if( (m_corMuonTemplate[chipNr]).at(index).condName() == object.getName() ) {
		    LogDebug("TriggerMenuParser") << "Found condition " << object.getName() << " in vector at index " << index << std::endl;
		    found = true;
		 } else {
		    index++;		 
		 }
	     }	  
	     if(found) {
	        corrIndexVal[jj] = index;
	     } else {
	       edm::LogError("TriggerMenuParser") << "FAILURE: Condition " << object.getName() << " is in map but not in cor. vector " << std::endl;
	     }
	     
	  }
*/
          parseMuonCorr(&object,chipNr);
	  corrIndexVal[jj] = (m_corMuonTemplate[chipNr]).size() - 1;	     
	  
          //Now set some flags for this subCondition
	  intGEq[jj] = (object.getComparisonOperator() == esComparisonOperator::GE);
          objType[jj] = gtMu;
          condCateg[jj] = CondMuon;
	  
        } else if(object.getType() == esObjectType::Egamma ||
	          object.getType() == esObjectType::Jet    ||
		  object.getType() == esObjectType::Tau ) {
	  
	  // we have an Calo object
          parseCaloCorr(&object,chipNr);
	  corrIndexVal[jj] = (m_corCaloTemplate[chipNr]).size() - 1;

          //Now set some flags for this subCondition
	  intGEq[jj] = (object.getComparisonOperator() == esComparisonOperator::GE);
          switch(object.getType()) {
	     case esObjectType::Egamma: { 
	      objType[jj] = gtEG;
	     }
	        break;
	     case esObjectType::Jet: { 
	      objType[jj] = gtJet;
	     }
	        break;
	     case esObjectType::Tau: { 
	      objType[jj] = gtTau;
	     }
	        break;
	      default: {
	      }
	        break;	
          }		 
          condCateg[jj] = CondCalo;
          	     
          
	  
	  
        } else if(object.getType() == esObjectType::ETM  ||
	        //  object.getType() == esObjectType::ETM2 ||
	          object.getType() == esObjectType::HTM ) {
	 
	  // we have Energy Sum
          parseEnergySumCorr(&object,chipNr);
          corrIndexVal[jj] = (m_corEnergySumTemplate[chipNr]).size() - 1;

          //Now set some flags for this subCondition
	  intGEq[jj] = (object.getComparisonOperator() == esComparisonOperator::GE);
          switch(object.getType()) {
	     case esObjectType::ETM: { 
	      objType[jj] = GlobalObject::gtETM;
	     }
	        break;
	     case esObjectType::HTM: { 
	      objType[jj] = GlobalObject::gtHTM;
	     }
	        break;
/*	     case esObjectType::ETM2: { 
	      objType[jj] = GlobalObject::gtETM2;
	     }
	        break; */		
	      default: {
	      }
	        break;			
          }		 
          condCateg[jj] = CondEnergySum;
	  	              

	} else {
	
          edm::LogError("TriggerMenuParser")
                  << "Illegal Object Type "
                  << " for the correlation condition " << name << std::endl;
          return false;	     

	}  //if block on leg types

      }  //loop over legs
    

    // get greater equal flag for the correlation condition
    bool gEq = true;
    if (intGEq[0] != intGEq[1]) {
        edm::LogError("TriggerMenuParser")
                << "Inconsistent GEq flags for sub-conditions "
                << " for the correlation condition " << name << std::endl;
        return false;

    }
    else {
        gEq = (intGEq[0] != 0);

    }
    

   // fill the correlation condition
    correlationCond.setCondType(cType);
    correlationCond.setObjectType(objType);
    correlationCond.setCondGEq(gEq);
    correlationCond.setCondChipNr(chipNr);

    correlationCond.setCond0Category(condCateg[0]);
    correlationCond.setCond1Category(condCateg[1]);

    correlationCond.setCond0Index(corrIndexVal[0]);
    correlationCond.setCond1Index(corrIndexVal[1]);

    correlationCond.setCorrelationParameter(corrParameter);

    if (edm::isDebugEnabled() ) {

        std::ostringstream myCoutStream;
        correlationCond.print(myCoutStream);
        LogTrace("TriggerMenuParser") << myCoutStream.str() << "\n"
                << std::endl;

    }

    // insert condition into the map
    // condition is not duplicate, check was done at the beginning

    (m_vecCorrelationTemplate[chipNr]).push_back(correlationCond);
    
    
    //
    return true;
}


/**
 * workAlgorithm - parse the algorithm and insert it into algorithm map.
 *
 * @param node The corresponding node to the algorithm.
 * @param name The name of the algorithm.
 * @param chipNr The number of the chip the conditions for that algorithm are located on.
 *
 * @return "true" on success, "false" if an error occurred.
 *
 */

bool l1t::TriggerMenuParser::parseAlgorithm( tmeventsetup::esAlgorithm algorithm,
    unsigned int chipNr) {

  
  using namespace tmeventsetup;
  //using namespace Algorithm;
  

    // get alias
    std::string algAlias = algorithm.getName();
    std::string algName  = algorithm.getName();

    if (algAlias == "") {
        algAlias = algName;
        LogDebug("TriggerMenuParser")
                << "\n    No alias defined for algorithm. Alias set to algorithm name."
                << "\n    Algorithm name:  " << algName << "\n    Algorithm alias: " << algAlias
                << std::endl;
    } else {
      //LogDebug("TriggerMenuParser") 
      LogDebug("TriggerMenuParser")  << "\n    Alias defined for algorithm."
			      << "\n    Algorithm name:  " << algName << "\n    Algorithm alias: " << algAlias
			      << std::endl;
    }

    // get the logical expression
    std::string logExpression = algorithm.getExpressionInCondition();

    LogDebug("TriggerMenuParser")
      << "      Logical expression: " << logExpression
      << "      Chip number:        " << chipNr
      << std::endl;

    // determine output pin
    int outputPin = algorithm.getIndex();


    //LogTrace("TriggerMenuParser")
    LogDebug("TriggerMenuParser")  << "      Output pin:         " << outputPin
			    << std::endl;


    // compute the bit number from chip number, output pin and order of the chips
    // pin numbering start with 1, bit numbers with 0
    int bitNumber = outputPin;// + (m_orderConditionChip[chipNr] -1)*m_pinsOnConditionChip -1;

    //LogTrace("TriggerMenuParser")
    LogDebug("TriggerMenuParser")  << "      Bit number:         " << bitNumber
			    << std::endl;

    // create a new algorithm and insert it into algorithm map
    GlobalAlgorithm alg(algName, logExpression, bitNumber);
    alg.setAlgoChipNumber(static_cast<int>(chipNr));
    alg.setAlgoAlias(algAlias);

    if (edm::isDebugEnabled() ) {

        std::ostringstream myCoutStream;
        alg.print(myCoutStream);
        LogTrace("TriggerMenuParser") << myCoutStream.str() << "\n" << std::endl;

    }

    // insert algorithm into the map
    if ( !insertAlgorithmIntoMap(alg)) {  
        return false;
    }

    return true;

}


// static class members

