// -*- C++ -*-
//
// Package:    JetVertexAssociation
// Class:      JetVertexAssociation
//
/**\class JetVertexAssociation JetVertexAssociation.cc JetMETCorrections/JetVertexAssociation/src/JetVertexAssociation.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Natalia Ilina
// Modified by Eduardo Luiggi
//
//         Created:  Tue Oct 31 10:52:41 CET 2006
//
//

/**
  * 'JetVertexAssociation' represents the association of the jet with the signal vertex
  *
  * Parameters of the method: JV_deltaZ, JV_alpha_threshold(alpha_0 or beta_0),
  *                           JV_cone_size, JV_type_Algo ("1" - alpha, "2" - beta) - (the details are in CMS NOTE 2006/091),
  *
  * Output: <pair<double, bool> >.
  *                    The first - variable alpha(beta) for the jet,
  *                    the second - "true" for jet from signal vertex, "false" for jet from pile-up.
  **/

#include <memory>
#include <iostream>
#include <iomanip>
#include <cmath>


#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "JetMETCorrections/JetVertexAssociation/interface/JetVertexAssociation.h"
#include "JetMETCorrections/JetVertexAssociation/interface/JetVertexMain.h"

using namespace std;
using namespace reco;
namespace cms{

  JetVertexAssociation::JetVertexAssociation(const edm::ParameterSet& iConfig): m_algo(iConfig),
                                                                                jet_token(consumes<CaloJetCollection>(edm::InputTag(iConfig.getParameter<std::string>("JET_ALGO")))),
                                                                                track_token(consumes<TrackCollection>(edm::InputTag(iConfig.getParameter<std::string>("TRACK_ALGO")))),
                                                                                vertex_token(consumes<VertexCollection>(edm::InputTag(iConfig.getParameter<std::string>("VERTEX_ALGO")))) {



    produces<ResultCollection1>("Var");
    produces<ResultCollection2>("JetType");


  }

  void JetVertexAssociation::produce(edm::Event& iEvent, const edm::EventSetup& iSetup){

   edm::Handle<CaloJetCollection> jets;
   iEvent.getByToken(jet_token, jets);

   edm::Handle<TrackCollection> tracks;
   iEvent.getByToken(track_token, tracks);

   edm::Handle<VertexCollection> vertexes;
   iEvent.getByToken(vertex_token, vertexes);

   double SIGNAL_V_Z = 0.;
   double SIGNAL_V_Z_ERROR = 0.;
   double ptmax = -100.;

   VertexCollection::const_iterator vert = vertexes->begin ();
   if(vertexes->size() > 0 )   {
        for (; vert != vertexes->end (); vert++) {

                SIGNAL_V_Z = vert->z();
                double pt = 0.;
                reco::Vertex::trackRef_iterator tr = vert->tracks_begin();
                for (; tr != vert->tracks_end(); tr++)  pt += (*tr)->pt();
                if( pt >= ptmax ){

	                  ptmax = pt;
		          SIGNAL_V_Z = vert->z();
                          SIGNAL_V_Z_ERROR = vert->zError();

		}

	}
   }

   pair<double, bool> result;
   std::auto_ptr<ResultCollection1> result1 (new ResultCollection1) ;
   std::auto_ptr<ResultCollection2> result2 (new ResultCollection2) ;

   CaloJetCollection::const_iterator jet = jets->begin ();

   if(jets->size() > 0 )   {
        for (; jet != jets->end (); jet++) {
	     result = m_algo.Main(*jet, tracks, SIGNAL_V_Z, SIGNAL_V_Z_ERROR);
             result1->push_back(result.first);
             result2->push_back(result.second);

	}
   }

   iEvent.put(result1, "Var");
   iEvent.put(result2, "JetType");

  }
}
