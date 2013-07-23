#ifndef HLTriggerOffline_Higgs_HLTHiggsValidator_H
#define HLTriggerOffline_Higgs_HLTHiggsValidator_H

/** \class HLTHiggsValidator
 *  Generate histograms for trigger efficiencies Higgs related
 *  Documentation available on the CMS TWiki:
 *  https://twiki.cern.ch/twiki/bin/view/CMS/HiggsWGHLTValidate
 *
 *  $Date: 2012/03/23 11:50:56 $
 *  $Revision: 1.7 $
 *  \author  J. Duarte Campderros (based and adapted on J. Klukas,
 *           M. Vander Donckt and J. Alcaraz code from the 
 *           HLTriggerOffline/Muon package)
 */

//#include "FWCore/PluginManager/interface/ModuleDef.h"
//#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DQMStore.h"

#include "HLTriggerOffline/Higgs/interface/HLTHiggsSubAnalysis.h"

#include <vector>
#include <cstring>


class EVTColContainer;

class HLTHiggsValidator : public edm::EDAnalyzer 
{
	public:
		//! Constructor
	      	HLTHiggsValidator(const edm::ParameterSet &);
	      	~HLTHiggsValidator();

	private:
		// concrete analyzer methods
	      	virtual void beginJob();
	      	virtual void beginRun(const edm::Run &iRun, const edm::EventSetup & iSetup);
	      	virtual void analyze(const edm::Event & iEvent, const edm::EventSetup & iSetup);
		virtual void endRun(const edm::Run & iRun, const edm::EventSetup & iSetup);
		virtual void endJob();

		//! Input from configuration file
		edm::ParameterSet _pset;
		//! the names of the subanalysis
		std::vector<std::string> _analysisnames;
		
		//! The instances of the class which do the real work
		std::vector<HLTHiggsSubAnalysis> _analyzers;
				
		//! The container with all the collections needed
		EVTColContainer * _collections;
		
		// Access to the DQM
		DQMStore * _dbe;      	
};

#endif
