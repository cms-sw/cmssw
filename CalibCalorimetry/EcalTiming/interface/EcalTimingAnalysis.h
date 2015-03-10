#ifndef EcalTimingAnalysis_H
#define EcalTimingAnalysis_H
/**\class EcalTimingAnalysis

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
#include "TTree.h"
#include <vector>

#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"


//#include<fstream>
//#include<map>
//#include<stl_pair>

//

class EcalTimingAnalysis : public edm::EDAnalyzer {
   public:
      explicit EcalTimingAnalysis( const edm::ParameterSet& );
      ~EcalTimingAnalysis();


      virtual void analyze(  edm::Event const&,   edm::EventSetup const& );
      virtual void beginJob();
      virtual void beginRun(edm::EventSetup const&);
      virtual void endJob();
      double timecorr(const CaloSubdetectorGeometry *geometry_p, DetId id);
      double myTheta(const CaloSubdetectorGeometry *geometry_p, DetId id);
 private:
      
      std::string rootfile_;
      std::string hitCollection_;
      std::string hitCollectionEE_;
      std::string hitProducer_;
      std::string hitProducerEE_;
	  std::string rhitCollection_;
      std::string rhitCollectionEE_;
      std::string rhitProducer_;
      std::string rhitProducerEE_;
      std::string digiProducer_;
      std::string gtRecordCollectionTag_;
      float ampl_thr_;
      float ampl_thrEE_;
	  double mintime_;
	  double maxtime_;
      int min_num_ev_;
      int max_num_ev_;
      int sm_;
      std::string txtFileName_;
      std::string txtFileForChGroups_;
      //std::string pndiodeProducer_;
      std::vector<double> sMAves_;
      std::vector<double> sMCorr_;
      
      TProfile* amplProfileConv_[54][4];
      TProfile* absoluteTimingConv_[54][4];

      TProfile* amplProfileAll_[54][4];
      TProfile* absoluteTimingAll_[54][4];
      
      TProfile* Chi2ProfileConv_[54][4];
      TH1F* timeCry[54][4];
      
      TProfile* relativeTimingBlueConv_[54];

      TGraphErrors* ttTiming_[54];
      TGraphErrors* ttTimingAll_;
      TGraphErrors* ttTimingRel_[54];
      TGraphErrors* ttTimingAllRel_;
      TGraphErrors* ttTimingAllSMChng_;
      
      TGraph* lasershiftVsTime_[54];
      TH2F* lasershiftVsTimehist_[54];
      TH1F* lasershiftLM_[54];
      TH1F* lasershift_;
      
      TProfile2D* ttTimingEtaPhi_;
      TProfile2D* chTimingEtaPhi_;
	    
      TProfile* ttTimingEta_;
      TProfile* chTimingEta_;
	  
      TProfile* ttTimingEtaEEP_;
	  
      TProfile* ttTimingEtaEEM_;
	  
      TProfile2D* chTimingEtaPhiEEP_;
      TProfile2D* chTimingEtaPhiEEM_;
      
      TProfile2D* ttTimingEtaPhiEEP_;
      TProfile2D* ttTimingEtaPhiEEM_;
      
      TH1F* timeCry1[54]; 
      TH1F* timeCry2[54]; 
      TH1F* timeRelCry1[54]; 
      TH1F* timeRelCry2[54]; 
      
      TH1F* aveRelXtalTime_;
      TH1F* aveRelXtalTimebyDCC_[54];
      TH2F* aveRelXtalTimeVsAbsTime_;
      
      TProfile2D* fullAmpProfileEB_;
      TProfile2D* fullAmpProfileEEP_;
      TProfile2D* fullAmpProfileEEM_;
      
      double timerunstart_;
      double timerunlength_;
	  
      TH1F* lasersPerEvt;

      const EcalElectronicsMapping* ecalElectronicsMap_;
 
      int ievt_;
      int numGoodEvtsPerSM_[54];
	  
      static const int numXtals = 15480;
  
      //Allows for running the job on a file
      bool fromfile_;
      std::string fromfilename_;   
	  
	  //Correct for Timing 
      bool corrtimeEcal;
      bool corrtimeBH;
      bool bhplus_;
      double EBradius_;
	  bool splash09cor_;
	  TTree* eventTimingInfoTree_;
	  
	  struct TTreeMembers {
	    int numEBcrys_;
	    int numEEcrys_;
	    int cryHashesEB_[61200];
	    int cryHashesEE_[14648];
	    float cryTimesEB_[61200];
	    float cryTimesEE_[14648];
	    float cryUTimesEB_[61200];
	    float cryUTimesEE_[14648];
	    float cryTimeErrorsEB_[61200];
	    float cryTimeErrorsEE_[14648];
	    float cryAmpsEB_[61200];
	    float cryAmpsEE_[14648];
	    float cryETEB_[61200];
	    float cryETEE_[14648];
            float kswisskEB_[61200];
	    int numTriggers_;
	    int numTechTriggers_;
	    int triggers_[200];
	    int techtriggers_[200];
	    float absTime_;
	    int lumiSection_;
	    int bx_;
	    int orbit_;
	    int run_;
	    float correctionToSample5EB_;
	    float correctionToSample5EEP_;
	    float correctionToSample5EEM_;
	  } TTreeMembers_; 
      double allave_;
      double allshift_;
      double timeerrthr_; 
      int minxtals_;
      bool writetxtfiles_;
      bool timingTree_;	  
      bool correctAVE_;
      std::map<int, double> eta2zmap_;

};



#endif
