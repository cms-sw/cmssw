#ifndef MCPizeroAnalyzer_H
#define MCPizeroAnalyzer_H
#include "RecoEgamma/EgammaMCTools/interface/PizeroMCTruthFinder.h"

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


class MCPizeroAnalyzer : public edm::EDAnalyzer
{

   public:

      //
      explicit MCPizeroAnalyzer( const edm::ParameterSet& ) ;
      virtual ~MCPizeroAnalyzer();


      virtual void analyze( const edm::Event&, const edm::EventSetup& ) ;
      virtual void beginJob() ;
      virtual void endJob() ;

   private:


      float etaTransformation( float a, float b);
      float phiNormalization( float& a);


      //
      PizeroMCTruthFinder*  thePizeroMCTruthFinder_;



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


      TH1F* h_MCPizE_;
      TH1F* h_MCPizEta_;
      TH1F* h_MCPizUnEta_;
      TH1F* h_MCPiz1ConEta_;
      TH1F* h_MCPiz2ConEta_;
      TH1F* h_MCPizPhi_;
      TH1F* h_MCPizMass1_;
      TH1F* h_MCPizMass2_;

      TH1F* h_MCEleE_;
      TH1F* h_MCEleEta_;
      TH1F* h_MCElePhi_;
      TH1F* h_BremFrac_;
      TH1F* h_BremEnergy_;

      TH2F* h_EleEvsPhoE_;

      TH1F* h_MCPhoE_;
      TH1F* h_MCPhoEta_;
      TH1F* h_MCPhoPhi_;
      TH1F* h_MCConvPhoE_;
      TH1F* h_MCConvPhoEta_;
      TH1F* h_MCConvPhoPhi_;
      TH1F* h_MCConvPhoR_;



};

#endif
