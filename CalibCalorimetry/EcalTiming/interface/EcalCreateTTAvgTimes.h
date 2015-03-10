// -*- C++ -*-
//
/**\class EcalCreateTTAvgTimes

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Seth COOPER
//         Created:  Wed Sep 30 16:29:33 CEST 2009
// $Id: EcalCreateTTAvgTimes.h,v 1.1 2010/01/08 21:34:04 scooper Exp $
//
//

// system include files
#include <memory>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"

#include "Geometry/EcalMapping/interface/EcalElectronicsMapping.h"
#include "Geometry/EcalMapping/interface/EcalMappingRcd.h"

#include "TFile.h"
#include "TTree.h"
#include "TH1.h"
#include "TH2.h"

//
// class decleration
//

class EcalCreateTTAvgTimes : public edm::EDAnalyzer {
   public:
      explicit EcalCreateTTAvgTimes(const edm::ParameterSet&);
      ~EcalCreateTTAvgTimes();


   private:
      virtual void beginJob(const edm::EventSetup& c);
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob();
      const std::pair<double,double> computeWeightedMeanAndSigma(std::vector<float>&, std::vector<float>&);
      const std::pair<double,double> computeUnweightedMeanAndSigma(std::vector<float>&);
      
      std::string intToString(int num);
      // ----------member data ---------------------------
      std::string inputFile_;
      bool subtractTowerAvgForOfflineCalibs_;

      const EcalElectronicsMapping* ecalElectronicsMap_;
      
};

