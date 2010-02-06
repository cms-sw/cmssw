////////////////////////////////////////////////////////////////////////////////
//
// PrescaleService
// ---------------
//
//            04/25/2008 Philipp Schieferdecker <philipp.schieferdecker@cern.ch>
////////////////////////////////////////////////////////////////////////////////


#include "FWCore/PrescaleService/interface/PrescaleService.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/ParameterSet/interface/Registry.h"

#include <set>
#include <algorithm>


namespace edm {
  namespace service {
    
    ////////////////////////////////////////////////////////////////////////////////
    // construction/destruction
    ////////////////////////////////////////////////////////////////////////////////

    //______________________________________________________________________________
    PrescaleService::PrescaleService(ParameterSet const& iPS,ActivityRegistry& iReg)
      : configured_(false)
      , lvl1Labels_(iPS.getParameter<std::vector<std::string> >("lvl1Labels"))
      , nLvl1Index_(lvl1Labels_.size())
      , iLvl1IndexDefault_(0)
      , vpsetPrescales_(iPS.getParameter<std::vector<ParameterSet> >("prescaleTable"))
      , prescaleTable_()
    {
      std::string lvl1DefaultLabel=
	iPS.getUntrackedParameter<std::string>("lvl1DefaultLabel","");
      for (unsigned int i = 0; i < lvl1Labels_.size(); ++i) {
	if (lvl1Labels_[i] == lvl1DefaultLabel) iLvl1IndexDefault_ = i;
      }

      iReg.watchPostBeginJob(this, &PrescaleService::postBeginJob);
      iReg.watchPostEndJob(this, &PrescaleService::postEndJob);
      
      iReg.watchPreProcessEvent(this, &PrescaleService::preEventProcessing);
      iReg.watchPostProcessEvent(this, &PrescaleService::postEventProcessing);
      
      iReg.watchPreModule(this, &PrescaleService::preModule);
      iReg.watchPostModule(this, &PrescaleService::postModule);
    }
      
    //______________________________________________________________________________
    PrescaleService::~PrescaleService() {
    }

    ////////////////////////////////////////////////////////////////////////////////
    // implementation of member functions
    ////////////////////////////////////////////////////////////////////////////////

    void PrescaleService::reconfigure(ParameterSet const& iPS) {
      vpsetPrescales_.clear();
      prescaleTable_.clear();
      iLvl1IndexDefault_ = 0;
      lvl1Labels_ = iPS.getParameter<std::vector<std::string> >("lvl1Labels");
      nLvl1Index_ = lvl1Labels_.size();
      vpsetPrescales_ = iPS.getParameter<std::vector<ParameterSet> >("prescaleTable");
      std::string lvl1DefaultLabel=
	iPS.getUntrackedParameter<std::string>("lvl1DefaultLabel","");
      for (unsigned int i=0; i < lvl1Labels_.size(); ++i) {
	if (lvl1Labels_[i] == lvl1DefaultLabel) iLvl1IndexDefault_ = i;
      }
      configure();
    }

    void PrescaleService::postBeginJob() {
      if (!configured_) {
        configure();
      }
    }

    //______________________________________________________________________________
    void PrescaleService::configure()
    {
      configured_ = true;

      ParameterSet prcPS = getProcessParameterSet();
      
      // find all HLTPrescaler modules
      std::set<std::string> prescalerModules;
      std::vector<std::string> allModules=prcPS.getParameter<std::vector<std::string> >("@all_modules");
      for(unsigned int i = 0; i < allModules.size(); ++i) {
	ParameterSet pset  = prcPS.getParameter<ParameterSet>(allModules[i]);
	std::string moduleLabel = pset.getParameter<std::string>("@module_label");
	std::string moduleType  = pset.getParameter<std::string>("@module_type");
	if (moduleType == "HLTPrescaler") prescalerModules.insert(moduleLabel);
      }
      
      // find all paths with an HLTPrescaler and check for <=1
      std::set<std::string> prescaledPathSet;
      std::vector<std::string> allPaths = prcPS.getParameter<std::vector<std::string> >("@paths");
      for (unsigned int iP = 0; iP < allPaths.size(); ++iP) {
	std::string pathName = allPaths[iP];
	std::vector<std::string> modules = prcPS.getParameter<std::vector<std::string> >(pathName);
	for (unsigned int iM = 0; iM < modules.size(); ++iM) {
	  std::string moduleLabel = modules[iM];
	  if (prescalerModules.erase(moduleLabel)>0) {
	    std::set<std::string>::const_iterator itPath=prescaledPathSet.find(pathName);
	    if (itPath==prescaledPathSet.end()) {
	      prescaledPathSet.insert(pathName);
	    } else {
	      throw cms::Exception("DuplicatePrescaler")
	        <<"path '"<<pathName<<"' has more than one HLTPrescaler!";
	    }
	  }
	}
      }

      std::vector<std::string> prescaledPaths;
      for (unsigned int iVPSet=0; iVPSet < vpsetPrescales_.size(); ++iVPSet) {
	ParameterSet psetPrescales = vpsetPrescales_[iVPSet];
	std::string pathName = psetPrescales.getParameter<std::string>("pathName");
	if (prescaledPathSet.erase(pathName) > 0) {
	  std::vector<unsigned int> prescales =
	    psetPrescales.getParameter<std::vector<unsigned int> >("prescales");
	  if (prescales.size()!=nLvl1Index_) {
	    throw cms::Exception("PrescaleTableMismatch")
	      << "path '" << pathName << "' has unexpected number of prescales";
	  }
	  prescaleTable_[pathName] = prescales;
	}
	else {
	  throw cms::Exception("PrescaleTableUnknownPath")
	    <<"path '"<<pathName<<"' is invalid or does not "
	    <<"contain any HLTPrescaler";
	}
      }      
    }

    //______________________________________________________________________________
    unsigned int PrescaleService::getPrescale(std::string const& prescaledPath)
    {
      return getPrescale(iLvl1IndexDefault_, prescaledPath);
    }
    
    //______________________________________________________________________________
    unsigned int PrescaleService::getPrescale(unsigned int lvl1Index,
					      std::string const& prescaledPath)
    {
      if (lvl1Index >= nLvl1Index_) {
	throw cms::Exception("InvalidLvl1Index")
	  <<"lvl1Index '"<<lvl1Index<<"' exceeds number of prescale columns";
      }

      if (!configured_) {
        configure();
      }
      
      PrescaleTable_t::const_iterator it = prescaleTable_.find(prescaledPath);
      return (it == prescaleTable_.end()) ? 1 : it->second[lvl1Index];
    }
    

  } // namespace service
} // namespace edm
