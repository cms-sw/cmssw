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
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/BTauReco/interface/IsolatedTauTagInfo.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include <string>


//Include DQM core
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"


 typedef math::XYZTLorentzVectorD   LV;
 typedef std::vector<LV>            LVColl;




//
// class decleration
//


class HLTTauDQMTrkPlotter  {
   public:
       HLTTauDQMTrkPlotter(const edm::ParameterSet&,int,int,int,double,bool,double);
      ~HLTTauDQMTrkPlotter();
      void analyze(const edm::Event&, const edm::EventSetup&,const LVColl&);

   private:
      std::pair<bool,LV> match(const LV& recoJet, const LVColl& matchingObject);       
      bool matchJet(const reco::Jet& ,const reco::CaloJetCollection&); 
   
      //Parameters to read
      edm::InputTag jetTagSrc_;
      edm::InputTag isolJets_;
     
      //Output file
      std::string folder_;
      std::string type_;
      double mcMatch_;
      //Monitor elements main
      MonitorElement* jetEt;
      MonitorElement* jetEta;
      MonitorElement* jetPhi;

      MonitorElement* isoJetEt;
      MonitorElement* isoJetEta;
      MonitorElement* isoJetPhi;

      MonitorElement* nPxlTrksInL25Jet;
      MonitorElement* nQPxlTrksInL25Jet;
      MonitorElement* signalLeadTrkPt;
      MonitorElement* hasLeadTrack;

      MonitorElement* EtEffNum;
      MonitorElement* EtEffDenom;
      MonitorElement* EtaEffNum;
      MonitorElement* EtaEffDenom;
      MonitorElement* PhiEffNum;
      MonitorElement* PhiEffDenom;



      DQMStore* store;
              
      bool doRef_;

      //Histogram Limits

      double EtMax_;
      int NPtBins_;
      int NEtaBins_;
      int NPhiBins_;

};



