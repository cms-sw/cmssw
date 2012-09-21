// -*- C++ -*-
//
// Package:    TestIsoSimTracks
// Class:      IsolatedParticles
// 
/*


 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Sergey Petrushanko
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/Common/interface/OrphanHandle.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/CaloTowers/interface/CaloTower.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
//#include "DataFormats/DTRecHit/interface/DTRecHitCollection.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"

#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/Track/interface/SimTrack.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertex.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"
#include "SimDataFormats/CaloHit/interface/PCaloHit.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"

// calorimeter info
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"

//#include "Geometry/DTGeometry/interface/DTLayer.h"
//#include "Geometry/DTGeometry/interface/DTGeometry.h"
//#include "Geometry/Records/interface/MuonGeometryRecord.h"

#include "DataFormats/GeometrySurface/interface/Cylinder.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"

#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"


#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "TrackPropagation/SteppingHelixPropagator/interface/SteppingHelixPropagator.h"

#include <boost/regex.hpp>

#include "TrackingTools/TrackAssociator/interface/TrackDetectorAssociator.h"
//-ap #include "TrackingTools/TrackAssociator/interface/TimerStack.h"
#include "Utilities/Timing/interface/TimerStack.h"

#include "TH1F.h"
#include <TFile.h>

class TestIsoSimTracks : public edm::EDAnalyzer {
 public:
   explicit TestIsoSimTracks(const edm::ParameterSet&);
   virtual ~TestIsoSimTracks(){};
   
   virtual void analyze (const edm::Event&, const edm::EventSetup&);
   void endJob(void);

 private:
  TFile* m_Hfile;
      struct{
        TH1F* eta;
        TH1F* phi;
        TH1F* p;
        TH1F* pt;
        TH1F* isomult;
      } IsoHists;  
   TrackDetectorAssociator trackAssociator_;
   TrackAssociatorParameters trackAssociatorParameters_;

   edm::InputTag simTracksTag_;
   edm::InputTag simVerticesTag_;
};

TestIsoSimTracks::TestIsoSimTracks(const edm::ParameterSet& iConfig) :
   simTracksTag_(iConfig.getParameter<edm::InputTag>("simTracksTag")),
   simVerticesTag_(iConfig.getParameter<edm::InputTag>("simVerticesTag"))
{
   // Fill data labels
   //std::vector<std::string> labels = iConfig.getParameter<std::vector<std::string> >("labels");
   //boost::regex regExp1 ("([^\\s,]+)[\\s,]+([^\\s,]+)$");
   //boost::regex regExp2 ("([^\\s,]+)[\\s,]+([^\\s,]+)[\\s,]+([^\\s,]+)$");
   //boost::smatch matches;
	
   m_Hfile=new TFile("IsoHists.root","RECREATE");
    IsoHists.eta = new TH1F("Eta","Track eta",100,-5.,5.);
    IsoHists.phi = new TH1F("Phi","Track phi",100,-3.5,3.5);
    IsoHists.p = new TH1F("Momentum","Track momentum",100,0.,20.);
    IsoHists.pt = new TH1F("pt","Track pt",100,0.,10.);
    IsoHists.isomult = new TH1F("IsoMult","Iso Mult",10,-0.5,9.5);

   //for(std::vector<std::string>::const_iterator label = labels.begin(); label != labels.end(); label++) {
   //   if (boost::regex_match(*label,matches,regExp1))
//	trackAssociator_.addDataLabels(matches[1],matches[2]);
 //     else if (boost::regex_match(*label,matches,regExp2))
//	trackAssociator_.addDataLabels(matches[1],matches[2],matches[3]);
 //     else
//	edm::LogError("ConfigurationError") << "Failed to parse label:\n" << *label << "Skipped.\n";
 //  }
   
   // trackAssociator_.addDataLabels("EBRecHitCollection","ecalrechit","EcalRecHitsEB");
   // trackAssociator_.addDataLabels("CaloTowerCollection","towermaker");
   // trackAssociator_.addDataLabels("DTRecSegment4DCollection","recseg4dbuilder");

   // Load TrackDetectorAssociator parameters
   edm::ParameterSet parameters = iConfig.getParameter<edm::ParameterSet>("TrackAssociatorParameters");
   trackAssociatorParameters_.loadParameters( parameters );
   trackAssociator_.useDefaultPropagator();
}

void TestIsoSimTracks::analyze( const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;

// mine! b

     std::vector<GlobalPoint> AllTracks;
     std::vector<GlobalPoint> AllTracks1;
     
// mine! e

   // get list of tracks and their vertices
   Handle<SimTrackContainer> simTracks;
   iEvent.getByLabel<SimTrackContainer>(simTracksTag_, simTracks);
   
   Handle<SimVertexContainer> simVertices;
   iEvent.getByLabel<SimVertexContainer>(simVerticesTag_, simVertices);
   if (! simVertices.isValid() ) throw cms::Exception("FatalError") << "No vertices found\n";
   
   // loop over simulated tracks
   std::cout << "Number of simulated tracks found in the event: " << simTracks->size() << std::endl;
   for(SimTrackContainer::const_iterator tracksCI = simTracks->begin(); 
       tracksCI != simTracks->end(); tracksCI++){
      
      // skip low Pt tracks
      if (tracksCI->momentum().Pt() < 0.7) {
//	 std::cout << "Skipped low Pt track (Pt: " << tracksCI->momentum().perp() << ")" <<std::endl;
	 continue;
      }
      
      // get vertex
      int vertexIndex = tracksCI->vertIndex();
      // uint trackIndex = tracksCI->genpartIndex();
      
      SimVertex vertex(math::XYZVectorD(0.,0.,0.),0);
      if (vertexIndex >= 0) vertex = (*simVertices)[vertexIndex];
      
      // skip tracks originated away from the IP
//      if (vertex.position().rho() > 50) {
//	 std::cout << "Skipped track originated away from IP: " <<vertex.position().rho()<<std::endl;
//	 continue;
//      }
      
      std::cout << "\n-------------------------------------------------------\n Track (pt,eta,phi): " << tracksCI->momentum().Pt() << " , " <<
	tracksCI->momentum().eta() << " , " << tracksCI->momentum().phi() << std::endl;
      
      // Simply get ECAL energy of the crossed crystals
//      std::cout << "ECAL energy of crossed crystals: " << 
//	trackAssociator_.getEcalEnergy(iEvent, iSetup,
//				       trackAssociator_.getFreeTrajectoryState(iSetup, *tracksCI, vertex) )
//	  << " GeV" << std::endl;
				       
//      std::cout << "Details:\n" <<std::endl;
      TrackDetMatchInfo info = trackAssociator_.associate(iEvent, iSetup,
							  trackAssociator_.getFreeTrajectoryState(iSetup, *tracksCI, vertex),
							  trackAssociatorParameters_);
//      std::cout << "ECAL, if track reach ECAL:     " << info.isGoodEcal << std::endl;
//      std::cout << "ECAL, number of crossed cells: " << info.crossedEcalRecHits.size() << std::endl;
//      std::cout << "ECAL, energy of crossed cells: " << info.ecalEnergy() << " GeV" << std::endl;
//      std::cout << "ECAL, number of cells in the cone: " << info.ecalRecHits.size() << std::endl;
//      std::cout << "ECAL, energy in the cone: " << info.ecalConeEnergy() << " GeV" << std::endl;
//      std::cout << "ECAL, trajectory point (z,R,eta,phi): " << info.trkGlobPosAtEcal.z() << ", "
//	<< info.trkGlobPosAtEcal.R() << " , "	<< info.trkGlobPosAtEcal.eta() << " , " 
//	<< info.trkGlobPosAtEcal.phi()<< std::endl;

// mine! b

	  double rfa =     sqrt (info.trkGlobPosAtEcal.x()*info.trkGlobPosAtEcal.x() +
	                         info.trkGlobPosAtEcal.y()*info.trkGlobPosAtEcal.y() + 
				 info.trkGlobPosAtEcal.z()*info.trkGlobPosAtEcal.z()) /
		           sqrt ( tracksCI->momentum().x()*tracksCI->momentum().x() +
			          tracksCI->momentum().y()*tracksCI->momentum().y() +
				  tracksCI->momentum().z()*tracksCI->momentum().z());

          if (info.isGoodEcal==1 && fabs(info.trkGlobPosAtEcal.eta()) < 2.6){
 	   AllTracks.push_back(GlobalPoint(info.trkGlobPosAtEcal.x()/rfa, info.trkGlobPosAtEcal.y()/rfa, info.trkGlobPosAtEcal.z()/rfa));
	    if (tracksCI->momentum().Pt() > 2. && fabs(info.trkGlobPosAtEcal.eta()) < 2.1) 
	     {				 
	     AllTracks1.push_back(GlobalPoint(info.trkGlobPosAtEcal.x()/rfa, info.trkGlobPosAtEcal.y()/rfa, info.trkGlobPosAtEcal.z()/rfa));
	     }
	  }

// mine! e   
   
//      std::cout << "HCAL, if track reach HCAL:      " << info.isGoodHcal << std::endl;
//      std::cout << "HCAL, number of crossed towers: " << info.crossedTowers.size() << std::endl;
//      std::cout << "HCAL, energy of crossed towers: " << info.hcalEnergy() << " GeV" << std::endl;
//      std::cout << "HCAL, number of towers in the cone: " << info.towers.size() << std::endl;
//      std::cout << "HCAL, energy in the cone: " << info.hcalConeEnergy() << " GeV" << std::endl;
//      std::cout << "HCAL, trajectory point (z,R,eta,phi): " << info.trkGlobPosAtHcal.z() << ", "
//	<< info.trkGlobPosAtHcal.R() << " , "	<< info.trkGlobPosAtHcal.eta() << " , "
//	<< info.trkGlobPosAtHcal.phi()<< std::endl;

   }

// mine! b

  std::cout << " NUMBER of tracks  " << AllTracks.size() << "  and candidates for iso tracks  " << AllTracks1.size() <<std::endl;

  double imult=0.;
      
  for (unsigned int ia1=0; ia1<AllTracks1.size(); ia1++) 
  {

    double delta_min=3.141592;
  
   for (unsigned int ia=0; ia<AllTracks.size(); ia++) 
   {
     double delta_phi = fabs(AllTracks1[ia1].phi() - AllTracks[ia].phi());
     if (delta_phi > 3.141592) delta_phi = 6.283184 - delta_phi;
     double delta_eta = fabs(AllTracks1[ia1].eta() - AllTracks[ia].eta());
     double delta_actual = sqrt( delta_phi*delta_phi + delta_eta*delta_eta );

     if (delta_actual < delta_min && delta_actual != 0.) delta_min = delta_actual;

   }    
    
    if (delta_min > 0.5) {
    
    std::cout << "FIND ISOLATED TRACK " << AllTracks1[ia1].mag() << "  " << AllTracks1[ia1].eta()<< "  "<< AllTracks1[ia1].phi()<< std::endl;    

    IsoHists.eta->Fill(AllTracks1[ia1].eta());
    IsoHists.phi->Fill(AllTracks1[ia1].phi());
    IsoHists.p->Fill(AllTracks1[ia1].mag());
    IsoHists.pt->Fill(AllTracks1[ia1].perp());
    imult = imult+1.;

    }
  }
    IsoHists.isomult->Fill(imult);

// mine! e

}


void TestIsoSimTracks::endJob(void) {

    m_Hfile->cd();
    IsoHists.eta->Write();
    IsoHists.phi->Write();
    IsoHists.p->Write();
    IsoHists.pt->Write();
    IsoHists.isomult->Write();
    m_Hfile->Close();

}



//define this as a plug-in
DEFINE_FWK_MODULE(TestIsoSimTracks);
