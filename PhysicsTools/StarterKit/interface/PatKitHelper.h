#ifndef PhysicsTools_StarterKit_interface_PatKitHelper_h
#define PhysicsTools_StarterKit_interface_PatKitHelper_h




//-------------------------------------------------------------------------------------
//
// Original Author:  Salvatore Rappoccio
//         Created:  Mon Jul  7 10:37:27 CDT 2008
// $Id: PatKitHelper.h,v 1.1 2008/07/07 20:06:54 srappocc Exp $
//
// Revision History:
//       -  Sal Rappoccio, Mon Jul  7 10:37:27 CDT 2008
//          Creation of object to make SK more inline with Framework advice.
//          This includes removing PatAnalyzerKit as a base class, and anything that
//          needs that functionality should use this class instead of deriving from
//          PatAnalyzerKit. 
//-------------------------------------------------------------------------------------
#include "PhysicsTools/StarterKit/interface/PhysicsHistograms.h"

#include "PhysicsTools/UtilAlgos/interface/TFileService.h"
#include "PhysicsTools/UtilAlgos/interface/TFileDirectory.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/EDProducer.h"


namespace pat {

  class PatKitHelper {
  public:

    PatKitHelper(edm::ParameterSet const & parameters);
    ~PatKitHelper();

    // Pull out a struct for the axis limits from the config file
    PhysicsHistograms::KinAxisLimits getAxisLimits( std::string name );
						    

    // Book histograms
    void bookHistos( edm::EDProducer * producer );

    // Get handles
    void getHandles( edm::Event  & event,
		     edm::Handle<std::vector<pat::Muon> > &     muonHandle,
		     edm::Handle<std::vector<pat::Electron> > & electronHandle,
		     edm::Handle<std::vector<pat::Tau> > &      tauHandle,
		     edm::Handle<std::vector<pat::Jet> > &      jetHandle,
		     edm::Handle<std::vector<pat::MET> > &      METHandle,
		     edm::Handle<std::vector<pat::Photon> > &   photonHandle
		     );

    
    // fill histograms
    void fillHistograms( edm::Event & event,
			 edm::Handle<std::vector<pat::Muon> > &     muonHandle,
			 edm::Handle<std::vector<pat::Electron> > & electronHandle,
			 edm::Handle<std::vector<pat::Tau> > &      tauHandle,
			 edm::Handle<std::vector<pat::Jet> > &      jetHandle,
			 edm::Handle<std::vector<pat::MET> > &      METHandle,
			 edm::Handle<std::vector<pat::Photon> > &   photonHandle
			 );
    
    
    // Function to add ntuple variables to the EDProducer
    void addNtupleVar ( edm::EDProducer * prod, std::string name, std::string type );

    // Save ntuple variables to event evt
    void saveNtuple (  edm::Event & event,
		       const std::vector<pat::PhysVarHisto*> & ntvars);
    
    // Helper function template to write objects to event
    template <class T>
      void saveNtupleVar(  edm::Event & event,
			   std::string name, T value);

    // Helper function template to write vectors of objects to event
    template <class T>
      void saveNtupleVec(  edm::Event & event,
			   std::string name, const std::vector<T> & invec);

    
    // verbose switch
    int verboseLevel_;

    // Keep a version of the parameter set in question
    edm::ParameterSet         parameters_;

    // Here is where the histograms go
    PhysicsHistograms  *      physHistos_;

    // File service for histograms
    edm::Service<TFileService> fs_;
    
    // List of ntuple variables
    std::vector< pat::PhysVarHisto* > ntVars_ ;

    
    
    // run and event numbers
    pat::PhysVarHisto *  h_runNumber_;
    pat::PhysVarHisto *  h_eventNumber_;
    
  };

}


#endif
