/** \class RPCEfficiencySecond
 *
 * Class for RPC Monitoring using RPCDigi and DT and CSC Segments.
 *
 *  $Date: 2008/07/03 16:25:09 $
 *  $Revision: 1.3 $
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
class TGaxis;


class RPCEfficiencySecond : public edm::EDAnalyzer {
   public:
      explicit RPCEfficiencySecond(const edm::ParameterSet&);
      ~RPCEfficiencySecond();
  
  TFile * theFile;
  TFile * theFileout;

  MonitorElement * histoRPC;
  MonitorElement * histoDT;
  MonitorElement * histoeffIdRPC_DT;
  MonitorElement * BXDistribution;
  MonitorElement * histoRealRPC;

  MonitorElement * EffGlobWm2;
  MonitorElement * EffGlobWm1;
  MonitorElement * EffGlobW0;
  MonitorElement * EffGlobW1;
  MonitorElement * EffGlobW2;

  MonitorElement * EffGlobWm2far;
  MonitorElement * EffGlobWm1far;
  MonitorElement * EffGlobW0far;
  MonitorElement * EffGlobW1far;
  MonitorElement * EffGlobW2far;

  MonitorElement * BXGlobWm2;
  MonitorElement * BXGlobWm1;
  MonitorElement * BXGlobW0;
  MonitorElement * BXGlobW1;
  MonitorElement * BXGlobW2;
  
  MonitorElement * BXGlobWm2far;
  MonitorElement * BXGlobWm1far;
  MonitorElement * BXGlobW0far;
  MonitorElement * BXGlobW1far;
  MonitorElement * BXGlobW2far;

  MonitorElement * MaskedGlobWm2;
  MonitorElement * MaskedGlobWm1;
  MonitorElement * MaskedGlobW0;
  MonitorElement * MaskedGlobW1;
  MonitorElement * MaskedGlobW2;
  
  MonitorElement * MaskedGlobWm2far;
  MonitorElement * MaskedGlobWm1far;
  MonitorElement * MaskedGlobW0far;
  MonitorElement * MaskedGlobW1far;
  MonitorElement * MaskedGlobW2far;

  MonitorElement * AverageEffWm2;
  MonitorElement * AverageEffWm1;
  MonitorElement * AverageEffW0;
  MonitorElement * AverageEffW1;
  MonitorElement * AverageEffW2;

  MonitorElement * AverageEffWm2far;
  MonitorElement * AverageEffWm1far;
  MonitorElement * AverageEffW0far;
  MonitorElement * AverageEffW1far;
  MonitorElement * AverageEffW2far;

  MonitorElement * NoPredictionWm2;
  MonitorElement * NoPredictionWm1;
  MonitorElement * NoPredictionW0;
  MonitorElement * NoPredictionW1;
  MonitorElement * NoPredictionW2;

  MonitorElement * NoPredictionWm2far;
  MonitorElement * NoPredictionWm1far;
  MonitorElement * NoPredictionW0far;
  MonitorElement * NoPredictionW1far;
  MonitorElement * NoPredictionW2far;


   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      virtual void endRun(const edm::Run& , const edm::EventSetup& );

      bool SaveFile;
      std::string NameFile;

      DQMStore * dbe;

};

