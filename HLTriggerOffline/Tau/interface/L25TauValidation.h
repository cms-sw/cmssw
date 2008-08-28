// Original Author:  Eduardo Luigi
//         Created:  Sun Jan 20 20:10:02 CST 2008



// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "DataFormats/BTauReco/interface/IsolatedTauTagInfo.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include <string>
#include "TH1F.h"


//Include DQM core
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"


 typedef math::XYZTLorentzVectorD   LV;
 typedef std::vector<LV>            LVColl;




//
// class decleration
//


class L25TauValidation : public edm::EDAnalyzer {
   public:
      explicit L25TauValidation(const edm::ParameterSet&);
      ~L25TauValidation();

   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      bool match(const LV& recoJet, const LVColl& matchingObject);       
   
      //Parameters to read
      
      edm::InputTag jetTagSrc_;
      edm::InputTag jetMCTagSrc_;
      edm::InputTag caloJets_;

      //Output file
      std::string tT_;
      std::string outFile_;

      //Monitor elements main
      MonitorElement* jetEt;
      MonitorElement* jetEta;
      MonitorElement* nL2EcalIsoJets;
      MonitorElement* nL25Jets;
      MonitorElement* nPxlTrksInL25Jet;
      MonitorElement* nQPxlTrksInL25Jet;
      MonitorElement* signalLeadTrkPt;
      MonitorElement* l25IsoJetEta;
      MonitorElement* l25IsoJetEt;
      MonitorElement* l25EtaEff;
      MonitorElement* l25EtEff;



               
      int nTracksInIsolationRing_;
      float rMatch_;
      float rSig_;
      float rIso_;
      float minPtIsoRing_;
      float ptLeadTk_;
      float mcMatch_;
      bool signal_;
      //Histogram Limits
      double EtMin_;
      double EtMax_;
      int NBins_;



};



