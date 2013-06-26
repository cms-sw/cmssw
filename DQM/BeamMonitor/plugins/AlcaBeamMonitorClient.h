#ifndef DQM_BeamMonitor_AlcaBeamMonitorClient_h
#define DQM_BeamMonitor_AlcaBeamMonitorClient_h

/** \class AlcaBeamMonitorClient
 * *
 *  $Date: 2010/09/24 06:36:05 $
 *  $Revision: 1.2 $
 *  \author  Lorenzo Uplegger/FNAL
 *   
 */
// C++
#include <map>
#include <vector>
#include <string>
// CMS
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DataFormats/Provenance/interface/LuminosityBlockID.h"
//#include "DataFormats/VertexReco/interface/Vertex.h"
//#include "DataFormats/VertexReco/interface/VertexFwd.h"

class AlcaBeamMonitorClient : public edm::EDAnalyzer {
 public:
  AlcaBeamMonitorClient( const edm::ParameterSet& );
  ~AlcaBeamMonitorClient();

 protected:

  void beginJob 	   (void);
  void beginRun 	   (const edm::Run& iRun,  	       const edm::EventSetup& iSetup);
  void analyze  	   (const edm::Event& iEvent, 	       const edm::EventSetup& iSetup);
  void beginLuminosityBlock(const edm::LuminosityBlock& iLumi, const edm::EventSetup& iSetup);
  void endLuminosityBlock  (const edm::LuminosityBlock& iLumi, const edm::EventSetup& iSetup);
  void endRun		   (const edm::Run& iRun,              const edm::EventSetup& iSetup);
  void endJob		   (const edm::LuminosityBlock& iLumi, const edm::EventSetup& iSetup);
  
 private:
  //                x,y,z,sigmax(y,z)... [run,lumi]          Histo name      
  typedef std::map<std::string,std::map<std::string,std::map<std::string,MonitorElement*> > > HistosContainer;

  //                x,y,z,sigmax(y,z)... [run,lumi]          Histo name      
  typedef std::map<std::string,std::map<std::string,std::map<std::string,int> > > PositionContainer;

  //Parameters
  edm::ParameterSet parameters_;
  std::string       monitorName_;

  //Service variables
  int         numberOfValuesToSave_;
  DQMStore*   dbe_;

  //Containers
  HistosContainer    					      histosMap_;
  std::vector<std::string>                                    varNamesV_; //x,y,z,sigmax(y,z)
  std::multimap<std::string,std::string>                      histoByCategoryNames_; //run, lumi
  std::map<edm::LuminosityBlockNumber_t,std::vector<double> > valuesMap_;
  PositionContainer    			                      positionsMap_;
};

#endif

