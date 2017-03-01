#ifndef MCPhotonAnalyzer_H
#define MCPhotonAnalyzer_H
#include "RecoEgamma/EgammaMCTools/interface/PhotonMCTruthFinder.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include <map>
#include <vector>


// forward declarations
class TFile;
class TH1F;
class TH2F;
class TProfile;
class TTree;
class SimVertex;
class SimTrack;


class MCPhotonAnalyzer : public edm::EDAnalyzer
{

   public:
   
      //
      explicit MCPhotonAnalyzer( const edm::ParameterSet& ) ;
      virtual ~MCPhotonAnalyzer();
                                   
      
      virtual void analyze( const edm::Event&, const edm::EventSetup& ) ;
      virtual void beginJob() ;
      virtual void endJob() ;

   private:
 
   
      float etaTransformation( float a, float b);
      float phiNormalization( float& a);

      
      //
      PhotonMCTruthFinder*  thePhotonMCTruthFinder_;
            
      const TrackerGeometry* trackerGeom;
      
      std::string fOutputFileName_ ;
      TFile*      fOutputFile_ ;
      


      
      int nEvt_;
      int nMatched_;

      /// global variable for the MC photon

      double mcPhi_;
      double mcEta_;

      std::string HepMCLabel;
      std::string SimTkLabel;
      std::string SimVtxLabel;
      std::string SimHitLabel;


      // all photons
      TH1F* h_MCPhoE_;
      TH1F* h_MCPhoEta_;
      TH1F* h_MCPhoEta1_;
      TH1F* h_MCPhoEta2_;
      TH1F* h_MCPhoEta3_;
      TH1F* h_MCPhoEta4_;
      TH1F* h_MCPhoPhi_;
      // Conversion
      TH1F* h_MCConvPhoE_;
      TH1F* h_MCConvPhoEta_;
      TH1F* h_MCConvPhoPhi_;
      TH1F* h_MCConvPhoR_;
      TH1F* h_MCConvPhoREta1_;
      TH1F* h_MCConvPhoREta2_;
      TH1F* h_MCConvPhoREta3_;
      TH1F* h_MCConvPhoREta4_;
      TH1F* h_convFracEta1_;
      TH1F* h_convFracEta2_;
      TH1F* h_convFracEta3_;
      TH1F* h_convFracEta4_;


      /// Conversions with two tracks
      TH1F* h_MCConvPhoTwoTracksE_;
      TH1F* h_MCConvPhoTwoTracksEta_;
      TH1F* h_MCConvPhoTwoTracksPhi_;
      TH1F* h_MCConvPhoTwoTracksR_;
      /// Conversions with one track
      TH1F* h_MCConvPhoOneTrackE_;
      TH1F* h_MCConvPhoOneTrackEta_;
      TH1F* h_MCConvPhoOneTrackPhi_;
      TH1F* h_MCConvPhoOneTrackR_;

      TH1F* h_MCEleE_;
      TH1F* h_MCEleEta_;
      TH1F* h_MCElePhi_;
      TH1F* h_BremFrac_;      
      TH1F* h_BremEnergy_;      
      TH2F* h_EleEvsPhoE_;
      TH2F* h_bremEvsEleE_;

      TProfile* p_BremVsR_;
      TProfile* p_BremVsEta_;

      TProfile* p_BremVsConvR_;
      TProfile* p_BremVsConvEta_;

      TH2F* h_bremFracVsConvR_;

};

#endif
