///////////////////////////////////////////////////////////////////////////////
//
// PrescaleService
// ---------------
//
///////////////////////////////////////////////////////////////////////////////


#include "FWCore/PrescaleService/interface/PrescaleService.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/ParameterSet/interface/Registry.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <iostream>
#include <set>
#include <algorithm>


namespace edm {
  namespace service {

    // constructor
    PrescaleService::PrescaleService(ParameterSet const& iPS,ActivityRegistry& iReg)
      : lvl1Labels_(iPS.getParameter<std::vector<std::string> >("lvl1Labels"))
      , lvl1Default_(findDefaultIndex(iPS.getParameter<std::string>("lvl1DefaultLabel"), lvl1Labels_))
      , vpsetPrescales_(iPS.getParameterSetVector("prescaleTable"))
      , prescaleTable_()
    {
      iReg.watchPostBeginJob(this, &PrescaleService::postBeginJob);

      // Sanity check
      if (lvl1Default_ >= lvl1Labels_.size()) {
        throw cms::Exception("InvalidLvl1Index")
          <<"lvl1Default_ '" << lvl1Default_ << "' exceeds number of prescale columns " << lvl1Labels_.size() << "!";
      }

      // Check and store Prescale Table
      for (unsigned int iVPSet=0; iVPSet < vpsetPrescales_.size(); ++iVPSet) {
        const ParameterSet& psetPrescales = vpsetPrescales_[iVPSet];
        const std::string pathName = psetPrescales.getParameter<std::string>("pathName");
	if (prescaleTable_.find(pathName)!=prescaleTable_.end()) {
	  throw cms::Exception("PrescaleServiceConfigError")
	    << " Path '" << pathName << "' found more than once!";
	}
	std::vector<unsigned int> prescales = psetPrescales.getParameter<std::vector<unsigned int> >("prescales");
	if (prescales.size()!=lvl1Labels_.size()) {
	  throw cms::Exception("PrescaleServiceConfigError")
	    << " Path '" << pathName << "' has " << prescales.size() << " prescales, instead of expected " << lvl1Labels_.size() << "!";
	}
	prescaleTable_[pathName] = prescales;
      }

    }
      
    // destructor
    PrescaleService::~PrescaleService() {
    }

    // member functions

    void PrescaleService::postBeginJob() {

      // Acesss to Process ParameterSet needed - can't be done in c'tor
      const ParameterSet prcPS = getProcessParameterSet();

      // Label of HLTPrescaler on each path, keyed on pathName
      std::map<std::string,std::string> path2module;
      // Name of path for each HLTPrescaler, keyed on moduleLabel
      std::map<std::string,std::string> module2path;

      // Check process config:
      // * each path contains at most one HLTPrescaler instance
      // * each HLTPrescaler instance is part of at most one path
      // * each HLTPrescaler instance is part of at least one ptah

      // Find all HLTPrescaler instances
      const std::vector<std::string> allModules=prcPS.getParameter<std::vector<std::string> >("@all_modules");
      for(unsigned int i = 0; i < allModules.size(); ++i) {
        ParameterSet const& pset  = prcPS.getParameterSet(allModules[i]);
	const std::string moduleLabel = pset.getParameter<std::string>("@module_label");
	const std::string moduleType  = pset.getParameter<std::string>("@module_type");
        if (moduleType == "HLTPrescaler") module2path[moduleLabel]="";
      }
      // Check all modules on all paths
      const std::vector<std::string> allPaths = prcPS.getParameter<std::vector<std::string> >("@paths");
      for (unsigned int iP = 0; iP < allPaths.size(); ++iP) {
        const std::string& pathName = allPaths[iP];
        std::vector<std::string> modules = prcPS.getParameter<std::vector<std::string> >(pathName);
        for (unsigned int iM = 0; iM < modules.size(); ++iM) {
          const std::string& moduleLabel = modules[iM];
	  if (module2path.find(moduleLabel)!=module2path.end()) {
            if (path2module.find(pathName)==path2module.end()) {
	      path2module[pathName]=moduleLabel;
	    } else {
              throw cms::Exception("PrescaleServiceConfigError")
		<< "Path '" << pathName << "' with (>1) HLTPrescalers: " << path2module[pathName] << "+" << moduleLabel << "!";
	    }
	    if (module2path[moduleLabel]=="") {
	      module2path[moduleLabel]=pathName;
	    } else {
              throw cms::Exception("PrescaleServiceConfigError")
                << " HLTPrescaler '" << moduleLabel << "' on (>1) Paths: " << module2path[moduleLabel] << "+" << pathName << "!";
	    }
	  }
	}
      }
      // Check all HLTPrescaler instances are on a path
      for (std::map<std::string,std::string>::const_iterator it = module2path.begin(); it!=module2path.end(); ++it) {
	if (it->second=="") {
	  throw cms::Exception("PrescaleServiceConfigError")
	    << " HLTPrescaler '" << it->first << "' not found on any path!";
	}
      }

      // Check paths stored Prescale Table: each path is actually in the process config
      for (std::map<std::string, std::vector<unsigned int> >::const_iterator it = prescaleTable_.begin(); it!=prescaleTable_.end(); ++it) {
	if (path2module.find(it->first)==path2module.end()) {
          throw cms::Exception("PrescaleServiceConfigError")
            << " Path '"<< it->first << "' is unknown or does not contain any HLTPrescaler!";
        }
      }

    }
    
    // const method
    unsigned int PrescaleService::getPrescale(std::string const& prescaledPath) const
    {
      return getPrescale(lvl1Default_, prescaledPath);
    }
    
    // const method
    unsigned int PrescaleService::getPrescale(unsigned int lvl1Index, std::string const& prescaledPath) const
    {
      if (lvl1Index >= lvl1Labels_.size()) {
        throw cms::Exception("InvalidLvl1Index")
          << "lvl1Index '" << lvl1Index << "' exceeds number of prescale columns " << lvl1Labels_.size() << "!";
      }
      PrescaleTable_t::const_iterator it = prescaleTable_.find(prescaledPath);
      return (it == prescaleTable_.end()) ? 1 : it->second[lvl1Index];
    }
    
    // static method
    unsigned int PrescaleService::findDefaultIndex(std::string const & label, std::vector<std::string> const & labels) {
      for (unsigned int i = 0; i < labels.size(); ++i) {
        if (labels[i] == label) {
          return i;
        }
      }
      return labels.size();
    }
    
    // static method
    void PrescaleService::fillDescriptions(edm::ConfigurationDescriptions & descriptions) {
      edm::ParameterSetDescription desc;

      std::vector<std::string> defaultVector;
      defaultVector.push_back(std::string("default"));
      desc.add<std::vector<std::string> >("lvl1Labels", defaultVector);

      // This default vector<ParameterSet> will be used when
      // the configuration does not include this parameter and
      // it also gets written into the generated cfi file.
      std::vector<edm::ParameterSet> defaultVPSet;
      edm::ParameterSet pset0;
      pset0.addParameter<std::string>("pathName", std::string("HLTPath"));
      std::vector<unsigned> defaultVectorU;
      defaultVectorU.push_back(1u);
      pset0.addParameter<std::vector<unsigned> >("prescales", defaultVectorU);
      defaultVPSet.push_back(pset0);

      edm::ParameterSetDescription validator;
      validator.add<std::string>("pathName");
      validator.add<std::vector<unsigned int> >("prescales");

      desc.addVPSet("prescaleTable", validator, defaultVPSet);

      desc.add<std::string>("lvl1DefaultLabel", std::string("default"));
      desc.add<bool>       ("forceDefault",     false);

      descriptions.add("PrescaleService", desc);
    }

  } // namespace service
} // namespace edm
