#ifndef MCElectronAnalyzer_H
#define MCElectronAnalyzer_H
#include "RecoEgamma/EgammaMCTools/interface/ElectronMCTruthFinder.h"

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


class MCElectronAnalyzer : public edm::EDAnalyzer
{

   public:

      //
      explicit MCElectronAnalyzer( const edm::ParameterSet& ) ;
      virtual ~MCElectronAnalyzer();


      virtual void analyze( const edm::Event&, const edm::EventSetup& ) ;
      virtual void beginJob() ;
      virtual void endJob() ;

   private:


      float etaTransformation( float a, float b);
      float phiNormalization( float& a);


      //
      ElectronMCTruthFinder*  theElectronMCTruthFinder_;

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


      TH1F* h_MCEleE_;
      TH1F* h_MCEleEta_;
      TH1F* h_MCElePhi_;
      TH1F* h_BremFrac_;
      TH1F* h_BremEnergy_;

      TProfile* p_BremVsR_;
      TProfile* p_BremVsEta_;



};

#endif
