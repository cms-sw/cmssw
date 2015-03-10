#ifndef EcalXMLDelaysPlotter_H
#define EcalXMLDelaysPlotter_H
/**\class EcalXMLDelaysPlotter

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>

*/

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalDetId/interface/EcalTrigTowerDetId.h"
#include "Geometry/EcalMapping/interface/EcalElectronicsMapping.h"
#include "Geometry/EcalMapping/interface/EcalMappingRcd.h"

#include <string>
#include "TProfile.h"
#include "TProfile2D.h"

#include "TGraphErrors.h"
#include "TGraph.h"
#include "TH1F.h"
#include "TH3F.h"
#include <vector>

#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"


//#include<fstream>
//#include<map>
//#include<stl_pair>

//

class EcalXMLDelaysPlotter : public edm::EDAnalyzer {
   public:
      explicit EcalXMLDelaysPlotter( const edm::ParameterSet& );
      ~EcalXMLDelaysPlotter();


      virtual void analyze(  edm::Event const&,   edm::EventSetup const& );
      virtual void beginJob();
      virtual void beginRun(edm::EventSetup const&);
      virtual void endJob();
      void readinXMLs();
      double timecorr(const CaloSubdetectorGeometry *geometry_p, DetId id);
 private:
      
      std::string rootfile_;
      std::string txtFileName_;
      std::string txtFileForChGroups_;
      //std::string pndiodeProducer_;
      std::vector<double> sMAves_;
      std::vector<double> sMCorr_;

      const EcalElectronicsMapping* ecalElectronicsMap_;
 
      int ievt_;
	  
      static const int numXtals = 15480;
 
	  
	  //Correct for Timing 
      bool corrtimeEcal;
      bool corrtimeBH;
      bool bhplus_;
      double EBradius_;
	  
      double allave_;
	  
      bool writetxtfiles_;

      double XMLTTVals_[54][68];
      double EETTVals_[2];
      double ETT_[54][68];

      TProfile2D *EBXMLProfile_;
      TProfile2D *EEPXMLProfile_;
      TProfile2D *EEMXMLProfile_;
      TProfile2D *EEPXMLProfileCh_;
      TProfile2D *EEMXMLProfileCh_;
};



#endif
