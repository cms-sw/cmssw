
#include "JetMETCorrections/JetVertexAssociation/interface/JetVertexMain.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <cmath>
using namespace reco;
using namespace edm;

JetVertexMain::JetVertexMain(const ParameterSet & parameters) {

  deltaZ = parameters.getParameter<double>("JV_deltaZ");
  threshold = parameters.getParameter<double>("JV_alpha_threshold");
  cone_size = parameters.getParameter<double>("JV_cone_size");
  Algo =  parameters.getParameter<int>("JV_type_Algo");

}


pair<double,bool> JetVertexMain::Main(const reco::CaloJet& jet, edm::Handle<TrackCollection> tracks, double signal_vert_Z){

  pair<double, bool> parameter; 
 
  double jet_et = jet.et();
  double jet_phi = jet.phi();
  double jet_eta = jet.eta();

  //  cout<<"JET: "<<jet_et<<endl;
  double Pt_jets_X = 0. ;
  double Pt_jets_Y = 0. ;
  double Pt_jets_X_tot = 0. ;
  double Pt_jets_Y_tot = 0. ;

  TrackCollection::const_iterator track = tracks->begin ();

  if (tracks->size() > 0 )   { 
   for (; track != tracks->end (); track++) {
     double Vertex_Z = track->vz();
     double track_eta = track->eta();
     double track_phi = track->phi();

     if (DeltaR(track_eta,jet_eta, track_phi, jet_phi) < cone_size) {
		
		  Pt_jets_X_tot += track->px();
		  Pt_jets_Y_tot += track->py();  
		  if (fabs(Vertex_Z-signal_vert_Z) < deltaZ) {
		    
		      Pt_jets_X += track->px();
		      Pt_jets_Y += track->py();
		       
		    }
		}
   

   }
  }

  double Var = -1;
  
  if (Algo == 1) Var =  Track_Pt(Pt_jets_X, Pt_jets_Y)/jet_et;
  else if (Algo == 2) {
      if (Track_Pt(Pt_jets_X_tot, Pt_jets_Y_tot)!=0)  Var =  Track_Pt(Pt_jets_X, Pt_jets_Y)/Track_Pt(Pt_jets_X_tot, Pt_jets_Y_tot);
      else  cout << "[Jets] JetVertexAssociation: Warning! problems for  Algo = 2: possible division by zero .." << endl;
  }
  else {
    
      Var =  Track_Pt(Pt_jets_X, Pt_jets_Y)/jet_et;
      cout << "[Jets] JetVertexAssociation: Warning! Algo = " << Algo << " not found; using Algo = 1" << endl;
  }

  //  cout<<"Var = "<<Var<<endl;

  if (Var >= threshold) parameter = pair<double, bool>(Var, true);
  else  parameter = pair<double, bool>(Var, false);

  return parameter;

}

double JetVertexMain::DeltaR(double eta1, double eta2, double phi1, double phi2){

  double dphi = fabs(phi1-phi2);
  if(dphi > 3.1415) dphi = 6.283 - dphi;  
  double deta = fabs(eta1-eta2);
  return sqrt(dphi*dphi + deta*deta);

}

double JetVertexMain::Track_Pt(double px, double py){

  return sqrt(px*px+py*py);

}
