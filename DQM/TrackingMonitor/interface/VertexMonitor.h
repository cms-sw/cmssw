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
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DQMStore.h"

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

class GetLumi;

class VertexMonitor
{
   public:
       VertexMonitor(const edm::ParameterSet&,const edm::InputTag&,const edm::InputTag&, std::string pvLabel);
       VertexMonitor(const edm::ParameterSet&,const edm::InputTag&,const edm::InputTag&, std::string pvLabel,edm::ConsumesCollector& iC);

       virtual ~VertexMonitor();
       static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

       virtual void initHisto(DQMStore::IBooker & ibooker);
       virtual void analyze(const edm::Event&, const edm::EventSetup&);
       
       // ----------member data ---------------------------
       
       edm::ParameterSet conf_;
       
       edm::InputTag     primaryVertexInputTag_;
       edm::InputTag     selectedPrimaryVertexInputTag_;
       std::string       label_;

       edm::EDGetTokenT<reco::VertexCollection> pvToken_;
       edm::EDGetTokenT<reco::VertexCollection> selpvToken_;
       

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
