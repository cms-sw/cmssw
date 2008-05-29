/** \class MuonSegmentEff
 *
 * Class for RPC Monitoring using RPCDigi and DT and CSCS egments.
 *
 *  $Date: 2008/05/22 12:52:45 $
 *  $Revision: 1.14 $
 *
 * \author Camilo Carrillo (Uniandes)
 *
 */

#include <FWCore/Framework/interface/Frameworkfwd.h>
#include <FWCore/Framework/interface/EDAnalyzer.h>
#include <FWCore/Framework/interface/Event.h>
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include <DataFormats/MuonDetId/interface/RPCDetId.h>


#include<string>
#include<map>
#include<fstream>

class RPCDetId;
class TFile;
class TH1F;
class TFile;
class TCanvas;
class TH2F;
class TString;

class MuonSegmentEff : public edm::EDAnalyzer {
   public:
      explicit MuonSegmentEff(const edm::ParameterSet&);
      ~MuonSegmentEff();
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      std::map<std::string, MonitorElement*> bookDetUnitSeg(RPCDetId & detId);

   private:

      std::vector<std::map<RPCDetId, int> > counter;
      std::vector<int> totalcounter;
      std::ofstream ofrej;
      std::ofstream ofeff;
      bool incldt;
      bool incldtMB4;
      bool inclcsc;
      bool prodImages;
      double MinimalResidual;
      double MinimalResidualRB4;
      double MinCosAng;
      double MaxD;
      double MaxDrb4;
      double MaxStripToCountInAverage;
      double MaxStripToCountInAverageRB4;
      std::string muonRPCDigis;
      std::string cscSegments;
      std::string dt4DSegments;
      std::string rejected;
      std::string rollseff;
      

      //Giuseppe
      std::map<std::string, std::map<std::string, MonitorElement*> >  meCollection;
      bool EffSaveRootFile;
      int  EffSaveRootFileEventsInterval;
      std::string EffRootFileName;
      std::string nameInLog;
      DQMStore * dbe;
      std::vector<std::string> _idList;

      //GLOBAL
      std::string GlobalRootLabel;
      TFile* fOutputFile;

      TH1F* hGlobalRes;
      TH1F* hGlobalResLa1;
      TH1F* hGlobalResLa2;
      TH1F* hGlobalResLa3;
      TH1F* hGlobalResLa4;
      TH1F* hGlobalResLa5;
      TH1F* hGlobalResLa6;
      
      TH1F* hGlobalResClu1;
      TH1F* hGlobalResClu2;
      TH1F* hGlobalResClu3;
      TH1F* hGlobalResClu4;

      TH1F* hGlobalResClu1La1;
      TH1F* hGlobalResClu1La2;
      TH1F* hGlobalResClu1La3;
      TH1F* hGlobalResClu1La4;
      TH1F* hGlobalResClu1La5;
      TH1F* hGlobalResClu1La6;
      
      TH1F* hGlobalResClu3La1;
      TH1F* hGlobalResClu3La2;
      TH1F* hGlobalResClu3La3;
      TH1F* hGlobalResClu3La4;
      TH1F* hGlobalResClu3La5;
      TH1F* hGlobalResClu3La6;
      
      TCanvas * Ca2;

      TH1F* hGlobalResY;
      
      //wheel-2
      TH1F* OGlobWm2;
      TH1F* PGlobWm2;
      TH1F* EffGlobWm2;
      TH1F* EffGlobm2s1;  TH1F* EffGlobm2s2;  TH1F* EffGlobm2s3;  TH1F* EffGlobm2s4;  TH1F* EffGlobm2s5;  TH1F* EffGlobm2s6; 
      TH1F* EffGlobm2s7;  TH1F* EffGlobm2s8;  TH1F* EffGlobm2s9;  TH1F* EffGlobm2s10;  TH1F* EffGlobm2s11;  TH1F* EffGlobm2s12; 
      
      //wheel-1
      TH1F* OGlobWm1;
      TH1F* PGlobWm1;
      TH1F* EffGlobWm1;
      TH1F* EffGlobm1s1;  TH1F* EffGlobm1s2;  TH1F* EffGlobm1s3;  TH1F* EffGlobm1s4;  TH1F* EffGlobm1s5;  TH1F* EffGlobm1s6; 
      TH1F* EffGlobm1s7;  TH1F* EffGlobm1s8;  TH1F* EffGlobm1s9;  TH1F* EffGlobm1s10;  TH1F* EffGlobm1s11;  TH1F* EffGlobm1s12; 
      
      //wheel0
      TH1F* OGlobW0;
      TH1F* PGlobW0;
      TH1F* EffGlobW0;
      TH1F* EffGlob1;  TH1F* EffGlob2;  TH1F* EffGlob3;  TH1F* EffGlob4;  TH1F* EffGlob5;  TH1F* EffGlob6; 
      TH1F* EffGlob7;  TH1F* EffGlob8;  TH1F* EffGlob9;  TH1F* EffGlob10;  TH1F* EffGlob11;  TH1F* EffGlob12; 

      //wheel1
      TH1F* OGlobW1;
      TH1F* PGlobW1;
      TH1F* EffGlobW1;
      TH1F* EffGlob1s1;  TH1F* EffGlob1s2;  TH1F* EffGlob1s3;  TH1F* EffGlob1s4;  TH1F* EffGlob1s5;  TH1F* EffGlob1s6; 
      TH1F* EffGlob1s7;  TH1F* EffGlob1s8;  TH1F* EffGlob1s9;  TH1F* EffGlob1s10;  TH1F* EffGlob1s11;  TH1F* EffGlob1s12; 

      //wheel2
      TH1F* OGlobW2;
      TH1F* PGlobW2;
      TH1F* EffGlobW2;
      TH1F* EffGlob2s1;  TH1F* EffGlob2s2;  TH1F* EffGlob2s3;  TH1F* EffGlob2s4;  TH1F* EffGlob2s5;  TH1F* EffGlob2s6; 
      TH1F* EffGlob2s7;  TH1F* EffGlob2s8;  TH1F* EffGlob2s9;  TH1F* EffGlob2s10;  TH1F* EffGlob2s11;  TH1F* EffGlob2s12; 
};
