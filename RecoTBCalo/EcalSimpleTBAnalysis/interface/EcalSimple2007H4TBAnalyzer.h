#ifndef ECALSIMPLE2007H4TBANALYZER_H
#define ECALSIMPLE2007H4TBANALYZER_H

/**\class EcalSimple2007H4TBAnalyzer

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// $Id: EcalSimple2007H4TBAnalyzer.h,v 1.2 2010/01/04 15:09:12 ferriff Exp $
//



// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"

#include <string>
//#include "TTree.h"
#include "TH1.h"
#include "TGraph.h"
#include "TH2.h"
#include<fstream>
#include<map>
//#include<stl_pair>



class EcalSimple2007H4TBAnalyzer : public edm::EDAnalyzer {
   public:
      explicit EcalSimple2007H4TBAnalyzer( const edm::ParameterSet& );
      ~EcalSimple2007H4TBAnalyzer();


      virtual void analyze( edm::Event const &, edm::EventSetup const & );
      virtual void beginRun(edm::Run const &, edm::EventSetup const&);
      virtual void endJob();
 private:
      std::string rootfile_;
      std::string digiCollection_;
      std::string digiProducer_;
      std::string hitCollection_;
      std::string hitProducer_;
      std::string hodoRecInfoCollection_;
      std::string hodoRecInfoProducer_;
      std::string tdcRecInfoCollection_;
      std::string tdcRecInfoProducer_;
      std::string eventHeaderCollection_;
      std::string eventHeaderProducer_;

      // Amplitude vs TDC offset
      TH2F* h_ampltdc;

      TH2F* h_Shape_;
      
      // Reconstructed energies
      TH1F* h_tableIsMoving;
      TH1F* h_e1x1;
      TH1F* h_e3x3; 
      TH1F* h_e5x5; 

      TH1F* h_e1x1_center;
      TH1F* h_e3x3_center; 
      TH1F* h_e5x5_center; 
      
      TH1F* h_e1e9;
      TH1F* h_e1e25;
      TH1F* h_e9e25;

      TH1F* h_S6; 
      TH1F* h_bprofx; 
      TH1F* h_bprofy; 
      
      TH1F* h_qualx; 
      TH1F* h_qualy; 
      
      TH1F* h_slopex; 
      TH1F* h_slopey; 
      
      TH2F* h_mapx[25]; 
      TH2F* h_mapy[25]; 

      TH2F* h_e1e9_mapx;
      TH2F* h_e1e9_mapy;

      TH2F* h_e1e25_mapx;
      TH2F* h_e1e25_mapy;

      TH2F* h_e9e25_mapx;
      TH2F* h_e9e25_mapy;

      EEDetId xtalInBeam_;
      EBDetId xtalInBeamTmp;
      EEDetId Xtals5x5[25];

      const CaloGeometry* theTBGeometry_;
};



#endif
