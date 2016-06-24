/** \class HLTPrescaleProvider
 *
 * See header file for documentation
 *
 *
 *  \author Martin Grunewald
 *
 */

#include "HLTrigger/HLTcore/interface/HLTPrescaleProvider.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <cassert>
#include <sstream>

static const bool useL1EventSetup(true);
static const bool useL1GtTriggerMenuLite(false);
static const unsigned char countMax(2);

bool HLTPrescaleProvider::init(const edm::Run& iRun,
                               const edm::EventSetup& iSetup,
                               const std::string& processName,
                               bool& changed) {

  count_[0]=0;
  count_[1]=0;
  count_[2]=0;
  count_[3]=0;
  count_[4]=0;

  const bool result(hltConfigProvider_.init(iRun, iSetup, processName, changed));

  const unsigned int l1tType(hltConfigProvider_.l1tType());
  if (l1tType==1) {
    /// L1 GTA V3: https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideL1TriggerL1GtUtils#Version_3
    l1GtUtils_.getL1GtRunCache(iRun,iSetup,useL1EventSetup,useL1GtTriggerMenuLite);
  } else if (l1tType==2) {
    l1tGlobalUtil_.retrieveL1Setup(iSetup);
  } else {
    edm::LogError("HLTPrescaleProvider")
      << " Unknown L1T Type " << l1tType << " - prescales will not be avaiable!";
  }

  return result;
}

int HLTPrescaleProvider::prescaleSet(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
 const unsigned int l1tType(hltConfigProvider_.l1tType());
 if (l1tType==1) {
  // return hltPrescaleTable_.set();
  l1GtUtils_.getL1GtRunCache(iEvent,iSetup,useL1EventSetup,useL1GtTriggerMenuLite);
  int errorTech(0);
  const int psfsiTech(l1GtUtils_.prescaleFactorSetIndex(iEvent,L1GtUtils::TechnicalTrigger,errorTech));
  int errorPhys(0);
  const int psfsiPhys(l1GtUtils_.prescaleFactorSetIndex(iEvent,L1GtUtils::AlgorithmTrigger,errorPhys));
  assert(psfsiTech==psfsiPhys);
  if ( (errorTech==0) && (errorPhys==0) &&
       (psfsiTech>=0) && (psfsiPhys>=0) && (psfsiTech==psfsiPhys) ) {
    return psfsiPhys;
  } else {
    /// error - notify user!
    if (count_[0]<countMax) {
      count_[0] += 1;
      edm::LogError("HLTPrescaleProvider")
	<< " Using processName '" << hltConfigProvider_.processName() <<"':"
	<< " Error in determining HLT prescale set index from L1 data using L1GtUtils:"
	<< " Tech/Phys error = " << errorTech << "/" << errorPhys
	<< " Tech/Phys psfsi = " << psfsiTech << "/" << psfsiPhys;
    }
    return -1;
  }
 } else if (l1tType==2) {
   l1tGlobalUtil_.retrieveL1Event(iEvent,iSetup);
   return static_cast<int>(l1tGlobalUtil_.prescaleColumn());
 } else {
   if (count_[0]<countMax) {
     count_[0] += 1;
     edm::LogError("HLTPrescaleProvider")
       << " Using processName '" << hltConfigProvider_.processName() <<"':"
       << " Unknown L1T Type " << l1tType << " - can not determine prescale set index!";
   }
   return -1;
 }

}

unsigned int HLTPrescaleProvider::prescaleValue(const edm::Event& iEvent,
                                                const edm::EventSetup& iSetup,
                                                const std::string& trigger) {
  const int set(prescaleSet(iEvent,iSetup));
  if (set<0) {
    return 1;
  } else {
    return hltConfigProvider_.prescaleValue(static_cast<unsigned int>(set),trigger);
  }
}

std::pair<int,int>
HLTPrescaleProvider::prescaleValues(const edm::Event& iEvent,
                                    const edm::EventSetup& iSetup,
                                    const std::string& trigger) {

  // start with setting both L1T and HLT prescale values to 0
  std::pair<int,int> result(std::pair<int,int>(0,0));

  // get HLT prescale (possible if HLT prescale set index is correctly found)
  const int set(prescaleSet(iEvent,iSetup));
  if (set<0) {
    result.second = -1;
  } else {
    result.second = static_cast<int>(hltConfigProvider_.prescaleValue(static_cast<unsigned int>(set),trigger));
  }

  // get L1T prescale - works only for those hlt trigger paths with
  // exactly one L1GT seed module which has exactly one L1T name as seed

 const unsigned int l1tType(hltConfigProvider_.l1tType());
 if (l1tType==1){
  const unsigned int nL1GTSeedModules(hltConfigProvider_.hltL1GTSeeds(trigger).size());
  if (nL1GTSeedModules==0) {
    // no L1 seed module on path hence no L1 seed hence formally no L1 prescale
    result.first=1;
  } else if (nL1GTSeedModules==1) {
    l1GtUtils_.getL1GtRunCache(iEvent,iSetup,useL1EventSetup,useL1GtTriggerMenuLite);
    const std::string l1tname(hltConfigProvider_.hltL1GTSeeds(trigger).at(0).second);
    int               l1error(0);
    result.first = l1GtUtils_.prescaleFactor(iEvent,l1tname,l1error);
    if (l1error!=0) {
      if (count_[1]<countMax) {
	count_[1] += 1;
	edm::LogError("HLTPrescaleProvider")
	  << " Error in determining L1T prescale for HLT path: '"	<< trigger
	  << "' with L1T seed: '" << l1tname
	  << "' using L1GtUtils: error code = " << l1error << "." << std::endl
	  << " Note: for this method ('prescaleValues'), only a single L1T name (and not a bit number) is allowed as seed!"
	  << std::endl
	  << " For seeds being complex logical expressions, try the new method 'prescaleValuesInDetail'."<< std::endl;
      }
      result.first = -1;
    }
  } else {
    /// error - can't handle properly multiple L1GTSeed modules
    if (count_[2]<countMax) {
      count_[2] += 1;
      std::string dump("'"+hltConfigProvider_.hltL1GTSeeds(trigger).at(0).second+"'");
      for (unsigned int i=1; i!=nL1GTSeedModules; ++i) {
	dump += " * '"+hltConfigProvider_.hltL1GTSeeds(trigger).at(i).second+"'";
      }
      edm::LogError("HLTPrescaleProvider")
	<< " Error in determining L1T prescale for HLT path: '" << trigger
	<< "' has multiple L1GTSeed modules, " << nL1GTSeedModules
	<< ", with L1 seeds: " << dump
	<< ". (Note: at most one L1GTSeed module is allowed for a proper determination of the L1T prescale!)";
    }
    result.first = -1;
  }
 } else if (l1tType==2){
  const unsigned int nL1TSeedModules(hltConfigProvider_.hltL1TSeeds(trigger).size());
  if (nL1TSeedModules==0) {
    // no L1 seed module on path hence no L1 seed hence formally no L1 prescale
    result.first=1;
  } else if (nL1TSeedModules==1) {
    l1tGlobalUtil_.retrieveL1Event(iEvent,iSetup);
    const std::string l1tname(hltConfigProvider_.hltL1GTSeeds(trigger).at(0).second);
    bool l1error(!l1tGlobalUtil_.getPrescaleByName(l1tname,result.first));
    if (l1error) {
      if (count_[1]<countMax) {
	count_[1] += 1;
	edm::LogError("HLTPrescaleProvider")
	  << " Error in determining L1T prescale for HLT path: '"	<< trigger
	  << "' with L1T seed: '" << l1tname
	  << "' using L1TGlobalUtil: error cond = " << l1error << "." << std::endl
	  << " Note: for this method ('prescaleValues'), only a single L1T name (and not a bit number) is allowed as seed!"
	  << std::endl
	  << " For seeds being complex logical expressions, try the new method 'prescaleValuesInDetail'."<< std::endl;
      }
      result.first = -1;
    }
  } else {
    /// error - can't handle properly multiple L1TSeed modules
    if (count_[2]<countMax) {
      count_[2] += 1;
      std::string dump("'"+hltConfigProvider_.hltL1TSeeds(trigger).at(0)+"'");
      for (unsigned int i=1; i!=nL1TSeedModules; ++i) {
	dump += " * '"+hltConfigProvider_.hltL1TSeeds(trigger).at(i)+"'";
      }
      edm::LogError("HLTPrescaleProvider")
	<< " Error in determining L1T prescale for HLT path: '" << trigger
	<< "' has multiple L1TSeed modules, " << nL1TSeedModules
	<< ", with L1T seeds: " << dump
	<< ". (Note: at most one L1TSeed module is allowed for a proper determination of the L1T prescale!)";
    }
    result.first = -1;
  }
 } else {
   if (count_[1]<countMax) {
     count_[1] += 1;
     edm::LogError("HLTPrescaleProvider")
       << " Unknown L1T Type " << l1tType << " - can not determine L1T prescale! ";
   }
   result.first = -1;
 }

  return result;
}

std::pair<std::vector<std::pair<std::string,int> >,int>
HLTPrescaleProvider::prescaleValuesInDetail(const edm::Event& iEvent,
                                            const edm::EventSetup& iSetup,
                                            const std::string& trigger) {

  std::pair<std::vector<std::pair<std::string,int> >,int> result;
  result.first.clear();

  // get HLT prescale (possible if HLT prescale set index is correctly found)
  const int set(prescaleSet(iEvent,iSetup));
  if (set<0) {
    result.second = -1;
  } else {
    result.second = static_cast<int>(hltConfigProvider_.prescaleValue(static_cast<unsigned int>(set),trigger));
  }

  // get L1T prescale - works only for those hlt trigger paths with
  // exactly one L1GT seed module

 const unsigned int l1tType(hltConfigProvider_.l1tType());
 if (l1tType==1){
  const unsigned int nL1GTSeedModules(hltConfigProvider_.hltL1GTSeeds(trigger).size());
  if (nL1GTSeedModules==0) {
    // no L1 seed module on path hence no L1 seed hence formally no L1 prescale
    result.first.clear();
  } else if (nL1GTSeedModules==1) {
    l1GtUtils_.getL1GtRunCache(iEvent,iSetup,useL1EventSetup,useL1GtTriggerMenuLite);
    const std::string l1tname(hltConfigProvider_.hltL1GTSeeds(trigger).at(0).second);
    L1GtUtils::LogicalExpressionL1Results l1Logical(l1tname, l1GtUtils_);
    l1Logical.logicalExpressionRunUpdate(iEvent.getRun(),iSetup,l1tname);
    const std::vector<std::pair<std::string, int> >& errorCodes(l1Logical.errorCodes(iEvent));
    result.first = l1Logical.prescaleFactors();
    int               l1error(l1Logical.isValid() ? 0 : 1);
    for (unsigned int i=0; i<errorCodes.size(); ++i) {
      l1error += std::abs(errorCodes[i].second);
    }
    if (l1error!=0) {
      if (count_[3]<countMax) {
	count_[3] += 1;
	std::ostringstream message;
	message
	  << " Error in determining L1T prescales for HLT path: '" << trigger
	  << "' with complex L1T seed: '" << l1tname
	  << "' using L1GtUtils: " << std::endl
	  << " isValid=" << l1Logical.isValid()
	  << " l1tname/error/prescale " << errorCodes.size()
	  << std::endl;
	for (unsigned int i=0; i< errorCodes.size(); ++i) {
	  message << " " << i << ":" << errorCodes[i].first << "/" << errorCodes[i].second << "/"
		  << result.first[i].second;
	}
	message << ".";
	edm::LogError("HLTPrescaleProvider") << message.str();
      }
      result.first.clear();
    }
  } else {
    /// error - can't handle properly multiple L1GTSeed modules
    if (count_[4]<countMax) {
      count_[4] += 1;
      std::string dump("'"+hltConfigProvider_.hltL1GTSeeds(trigger).at(0).second+"'");
      for (unsigned int i=1; i!=nL1GTSeedModules; ++i) {
	dump += " * '"+hltConfigProvider_.hltL1GTSeeds(trigger).at(i).second+"'";
      }
      edm::LogError("HLTPrescaleProvider")
	<< " Error in determining L1T prescale for HLT path: '" << trigger
	<< "' has multiple L1GTSeed modules, " << nL1GTSeedModules
	<< ", with L1 seeds: " << dump
	<< ". (Note: at most one L1GTSeed module is allowed for a proper determination of the L1T prescale!)";
    }
    result.first.clear();
  }
 } else if (l1tType==2){
  const unsigned int nL1TSeedModules(hltConfigProvider_.hltL1TSeeds(trigger).size());
  if (nL1TSeedModules==0) {
    // no L1 seed module on path hence no L1 seed hence formally no L1 prescale
    result.first.clear();
  } else if (nL1TSeedModules==1) {
    l1tGlobalUtil_.retrieveL1Event(iEvent,iSetup);
    std::string l1tname(hltConfigProvider_.hltL1TSeeds(trigger).at(0));
    GlobalLogicParser l1tGlobalLogicParser = GlobalLogicParser(l1tname);
    const std::vector<GlobalLogicParser::OperandToken> l1tSeeds = l1tGlobalLogicParser.expressionSeedsOperandList();
    int l1error(0);
    int l1tPrescale(-1);
    for (unsigned int i=0; i<l1tSeeds.size(); ++i) {
      const string& l1tSeed = l1tSeeds[i].tokenName;
      if (!l1tGlobalUtil_.getPrescaleByName(l1tSeed,l1tPrescale)) {
	l1error += 1;
      }
      result.first.push_back(std::pair<std::string,int>(l1tSeed,l1tPrescale));
    }
    if (l1error!=0) {
      if (count_[3]<countMax) {
	count_[3] += 1;
	string l1name = l1tname;
	std::ostringstream message;
	message
	  << " Error in determining L1T prescales for HLT path: '" << trigger
	  << "' with complex L1T seed: '" << l1tname
	  << "' using L1TGlobalUtil: " << std::endl
	  << " isValid=" << l1tGlobalLogicParser.checkLogicalExpression(l1name)
	  << " l1tname/error/prescale " << l1tSeeds.size()
	  << std::endl;
	for (unsigned int i=0; i<l1tSeeds.size(); ++i) {
	  const string& l1tSeed = l1tSeeds[i].tokenName;
	  message << " " << i << ":" << l1tSeed << "/" << l1tGlobalUtil_.getPrescaleByName(l1tSeed,l1tPrescale) << "/"
		  << result.first[i].second;
	}
	message << ".";
	edm::LogError("HLTPrescaleProvider") << message.str();
      }
      result.first.clear();
    }
  } else {
    /// error - can't handle properly multiple L1TSeed modules
    if (count_[4]<countMax) {
      count_[4] += 1;
      std::string dump("'"+hltConfigProvider_.hltL1TSeeds(trigger).at(0)+"'");
      for (unsigned int i=1; i!=nL1TSeedModules; ++i) {
	dump += " * '"+hltConfigProvider_.hltL1TSeeds(trigger).at(i)+"'";
      }
      edm::LogError("HLTPrescaleProvider")
	<< " Error in determining L1T prescale for HLT path: '" << trigger
	<< "' has multiple L1TSeed modules, " << nL1TSeedModules
	<< ", with L1T seeds: " << dump
	<< ". (Note: at most one L1TSeed module is allowed for a proper determination of the L1T prescale!)";
    }
    result.first.clear();
  }
 } else {
   if (count_[3]<countMax) {
     count_[3] += 1;
     edm::LogError("HLTPrescaleProvider")
       << " Unknown L1T Type " << l1tType << " - can not determine L1T prescale! ";
   }
   result.first.clear();
 }

  return result;
}
