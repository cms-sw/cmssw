#ifndef DQM_BeamMonitor_AlcaBeamMonitor_h
#define DQM_BeamMonitor_AlcaBeamMonitor_h

/** \class AlcaBeamMonitor
 * *
 *  $Date: 2010/08/11 21:58:56 $
 *  $Revision: 1.1 $
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
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

class BeamFitter;
class PVFitter;

class AlcaBeamMonitor : public edm::EDAnalyzer {
 public:
  AlcaBeamMonitor( const edm::ParameterSet& );
  ~AlcaBeamMonitor();

 protected:

  void beginJob 	   (void);
  void beginRun 	   (const edm::Run& iRun,  	       const edm::EventSetup& iSetup);
  void analyze  	   (const edm::Event& iEvent, 	       const edm::EventSetup& iSetup);
  void beginLuminosityBlock(const edm::LuminosityBlock& iLumi, const edm::EventSetup& iSetup);
  void endLuminosityBlock  (const edm::LuminosityBlock& iLumi, const edm::EventSetup& iSetup);
  void endRun		   (const edm::Run& iRun,              const edm::EventSetup& iSetup);
  void endJob		   (const edm::LuminosityBlock& iLumi, const edm::EventSetup& iSetup);
  
 private:
  //Typedefs
  //                x,y,z,sigmax(y,z)...       lumi      
  typedef std::map<std::string,std::map<edm::LuminosityBlockNumber_t,reco::BeamSpot> >  BeamSpotContainer;
  
  //                x,y,z,sigmax(y,z)... [run,lumi]          Histo name      
  typedef std::map<std::string,std::map<std::string,std::map<std::string,MonitorElement*> > > HistosContainer;

  //Parameters
  edm::ParameterSet parameters_;
  std::string       monitorName_;
  edm::InputTag     primaryVertexLabel_;
  edm::InputTag     beamSpotLabel_;
  edm::InputTag     trackLabel_;
  edm::InputTag     scalerLabel_;

  //Service variables
  int         firstLumi_,lastLumi_;
  int         numberOfLumis_;
  DQMStore*   dbe_;
  BeamFitter* theBeamFitter_;
  PVFitter*   thePVFitter_;
  
  // MonitorElements:
  MonitorElement* h_d0_phi0;

  //Containers
  BeamSpotContainer  							      beamSpotsMap_;
  HistosContainer    							      histosMap_;
  std::vector<std::string>                                                    varNamesV_; //x,y,z,sigmax(y,z)
  std::multimap<std::string,std::string>                                      histoByCategoryNames_; //run, lumi
  std::map<edm::LuminosityBlockNumber_t,std::vector<reco::VertexCollection> > verticesMap_;
  
};

#endif

