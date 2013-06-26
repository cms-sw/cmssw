#ifndef _PhysicsTools_HepMCCandAlgos_ModelFilter_h_
#define _PhysicsTools_HepMCCandAlgos_ModelFilter_h_

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"



/** 
    The ModelFilter class will select events in a "soup" MC
    (like the SUSY signal MC) from the comments of LHEEventProduct
    that match "modelTag". The user can require the value of that
    parameter to lie between a min and max value. 
 */

namespace edm {

  class ModelFilter : public edm::EDFilter 
  {
  public:
	explicit ModelFilter(const edm::ParameterSet&);
	~ModelFilter();
	
	static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
	std::vector<std::string> split(std::string fstring, std::string splitter);
	typedef std::vector<std::string>::const_iterator comments_const_iterator;
	
  private:
	virtual void beginJob() ;
	virtual bool filter(edm::Event&, const edm::EventSetup&) override;
	virtual void endJob() ;
    
	edm::InputTag inputTagSource_;
	std::string modelTag_;
	std::vector<double> parameterMins_;
	std::vector<double> parameterMaxs_;
  };


}

#endif /*ModelFilter_h_*/
