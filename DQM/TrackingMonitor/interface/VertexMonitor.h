#ifndef VertexMonitor_H
#define VertexMonitor_H
// -*- C++ -*-
//
// 
/**\class VertexMonitor VertexMonitor.cc
Monitoring source for general quantities related to vertex
*/

// system include files
#include <memory> 

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/MonitorElement.h"

#include "DataFormats/VertexReco/interface/Vertex.h"

class GetLumi;

class DQMStore;

class VertexMonitor
{
   public:
  VertexMonitor(const edm::ParameterSet&,const edm::InputTag&,const edm::InputTag&, std::string pvLabel);
       virtual ~VertexMonitor();
       static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

       virtual void beginJob(DQMStore * dqmStore_);
       virtual void analyze(const edm::Event&, const edm::EventSetup&);
       
       virtual void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&);
       virtual void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&);
       
       // ----------member data ---------------------------
       
       edm::ParameterSet conf_;
       
       edm::InputTag     primaryVertexInputTag_;
       edm::InputTag     selectedPrimaryVertexInputTag_;
       std::string       label_;

       GetLumi* lumiDetails_;

       MonitorElement* NumberOfPVtx;
       MonitorElement* NumberOfPVtxVsBXlumi;
       MonitorElement* NumberOfPVtxVsGoodPVtx;
       MonitorElement* NumberOfGoodPVtx;
       MonitorElement* NumberOfGoodPVtxVsBXlumi;
       MonitorElement* FractionOfGoodPVtx;
       MonitorElement* FractionOfGoodPVtxVsBXlumi;
       MonitorElement* FractionOfGoodPVtxVsGoodPVtx;
       MonitorElement* FractionOfGoodPVtxVsPVtx;
       MonitorElement* NumberOfFakePVtx;
       MonitorElement* NumberOfFakePVtxVsBXlumi;
       MonitorElement* NumberOfFakePVtxVsGoodPVtx;
       MonitorElement* NumberOfBADndofPVtx;
       MonitorElement* NumberOfBADndofPVtxVsBXlumi;
       MonitorElement* NumberOfBADndofPVtxVsGoodPVtx;

       MonitorElement* Chi2oNDFVsGoodPVtx;
       MonitorElement* Chi2oNDFVsBXlumi;
       MonitorElement* Chi2ProbVsGoodPVtx;
       MonitorElement* Chi2ProbVsBXlumi;

       MonitorElement* GoodPVtxSumPt;
       MonitorElement* GoodPVtxSumPtVsBXlumi;
       MonitorElement* GoodPVtxSumPtVsGoodPVtx;
	
       MonitorElement* GoodPVtxNumberOfTracks;
       MonitorElement* GoodPVtxNumberOfTracksVsBXlumi;
       MonitorElement* GoodPVtxNumberOfTracksVsGoodPVtx;
       MonitorElement* GoodPVtxNumberOfTracksVsGoodPVtxNdof;	

       MonitorElement* GoodPVtxChi2oNDFVsGoodPVtx;
       MonitorElement* GoodPVtxChi2oNDFVsBXlumi;
       MonitorElement* GoodPVtxChi2ProbVsGoodPVtx;
       MonitorElement* GoodPVtxChi2ProbVsBXlumi;

       bool doAllPlots_; 
       bool doPlotsVsBXlumi_;
       bool doPlotsVsGoodPVtx_;
       
       std::string histname;  //for naming the histograms according to algorithm used

};
#endif
