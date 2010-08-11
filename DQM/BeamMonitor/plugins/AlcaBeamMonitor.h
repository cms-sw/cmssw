#ifndef DQM_BeamMonitor_AlcaBeamMonitor_h
#define DQM_BeamMonitor_AlcaBeamMonitor_h

/** \class AlcaBeamMonitor
 * *
 *  $Date: 2010/07/06 23:37:27 $
 *  $Revision: 1.4 $
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

// class declaration
//

class AlcaBeamMonitor : public edm::EDAnalyzer {
 public:
  AlcaBeamMonitor( const edm::ParameterSet& );
  ~AlcaBeamMonitor();

 protected:
   
  // BeginJob
  void beginJob();

  // BeginRun
  void beginRun(const edm::Run& r, const edm::EventSetup& c);
  
  void analyze(const edm::Event& e, const edm::EventSetup& c) ;
  
  void beginLuminosityBlock(const edm::LuminosityBlock& lumiSeg, 
			    const edm::EventSetup& context) ;
  
  void endLuminosityBlock(const edm::LuminosityBlock& lumiSeg, 
			  const edm::EventSetup& c);
  // EndRun
  void endRun(const edm::Run& r, const edm::EventSetup& c);
  // Endjob
  void endJob(const edm::LuminosityBlock& lumiSeg, const edm::EventSetup& c);
  
 private:
  //                x,y,z,sigmax(y,z)... [PV,BF,DB...]         lumi      
//  typedef std::map<std::string,std::map<std::string,std::map<edm::LuminosityBlockNumber_t,Result> > > ResultsContainer;
  typedef std::map<std::string,std::map<edm::LuminosityBlockNumber_t,reco::BeamSpot> >  BeamSpotContainer;

  BeamSpotContainer  beamSpotsMap_;

  //Parameters
  edm::ParameterSet parameters_;
  std::string       monitorName_;
  edm::InputTag     primaryVertexLabel_; // primary vertex
  edm::InputTag     beamSpotLabel_; // primary vertex
  edm::InputTag     trackLabel_;
  edm::InputTag     scalerLabel_;

  int firstLumi_,lastLumi_;
  int numberOfLumis_;
  
  DQMStore*   dbe_;
  BeamFitter* theBeamFitter_;
  PVFitter*   thePVFitter_;
  
  // MonitorElements:
  MonitorElement* h_d0_phi0;

  //                x,y,z,sigmax(y,z)... [run,lumi]          Histo name      
  typedef std::map<std::string,std::map<std::string,std::map<std::string,MonitorElement*> > > HistosContainer;
  std::vector<std::string> varNamesV_; //x,y,z,sigmax(y,z)
  std::multimap<std::string,std::string> histoByCategoryNames_; //run, lumi
  
  HistosContainer histosMap_;
  
  std::map<edm::LuminosityBlockNumber_t,std::vector<reco::VertexCollection> > verticesMap_;
//  TH1F service 
  
  //
  std::time_t tmpTime;
  std::time_t startTime;
  std::time_t refTime;
  edm::TimeValue_t ftimestamp;
  
};

#endif

