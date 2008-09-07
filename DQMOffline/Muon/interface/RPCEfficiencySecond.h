/** \class RPCEfficiencySecond
 *
 * Class for RPC Monitoring using RPCDigi and DT and CSC Segments.
 *
 *  $Date: 2008/08/19 06:14:55 $
 *  $Revision: 1.5 $
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
  MonitorElement * histoCSC;
  MonitorElement * histoeffIdRPC_DT;
  MonitorElement * histoeffIdRPC_CSC;
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

  //EndCap

  MonitorElement * EffGlobDm3;
  MonitorElement * EffGlobDm2;
  MonitorElement * EffGlobDm1;
  MonitorElement * EffGlobD1;
  MonitorElement * EffGlobD2;
  MonitorElement * EffGlobD3;

  MonitorElement * EffGlobDm3far;
  MonitorElement * EffGlobDm2far;
  MonitorElement * EffGlobDm1far;
  MonitorElement * EffGlobD1far;
  MonitorElement * EffGlobD2far;
  MonitorElement * EffGlobD3far;

  MonitorElement * BXGlobDm3;
  MonitorElement * BXGlobDm2;
  MonitorElement * BXGlobDm1;
  MonitorElement * BXGlobD1;
  MonitorElement * BXGlobD2;
  MonitorElement * BXGlobD3;
  
  MonitorElement * BXGlobDm3far;
  MonitorElement * BXGlobDm2far;
  MonitorElement * BXGlobDm1far;
  MonitorElement * BXGlobD1far;
  MonitorElement * BXGlobD2far;
  MonitorElement * BXGlobD3far;

  MonitorElement * MaskedGlobDm3;
  MonitorElement * MaskedGlobDm2;
  MonitorElement * MaskedGlobDm1;
  MonitorElement * MaskedGlobD1;
  MonitorElement * MaskedGlobD2;
  MonitorElement * MaskedGlobD3;
  
  MonitorElement * MaskedGlobDm3far;
  MonitorElement * MaskedGlobDm2far;
  MonitorElement * MaskedGlobDm1far;
  MonitorElement * MaskedGlobD1far;
  MonitorElement * MaskedGlobD2far;
  MonitorElement * MaskedGlobD3far;

  MonitorElement * AverageEffDm3;
  MonitorElement * AverageEffDm2;
  MonitorElement * AverageEffDm1;
  MonitorElement * AverageEffD1;
  MonitorElement * AverageEffD2;
  MonitorElement * AverageEffD3;

  MonitorElement * AverageEffDm3far;
  MonitorElement * AverageEffDm2far;
  MonitorElement * AverageEffDm1far;
  MonitorElement * AverageEffD1far;
  MonitorElement * AverageEffD2far;
  MonitorElement * AverageEffD3far;

  MonitorElement * NoPredictionDm3;
  MonitorElement * NoPredictionDm2;
  MonitorElement * NoPredictionDm1;
  MonitorElement * NoPredictionD1;
  MonitorElement * NoPredictionD2;
  MonitorElement * NoPredictionD3;

  MonitorElement * NoPredictionDm3far;
  MonitorElement * NoPredictionDm2far;
  MonitorElement * NoPredictionDm1far;
  MonitorElement * NoPredictionD1far;
  MonitorElement * NoPredictionD2far;
  MonitorElement * NoPredictionD3far;

  
 private:
  virtual void beginJob(const edm::EventSetup&) ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;
  virtual void endRun(const edm::Run& , const edm::EventSetup& );
  
  bool debug;
  bool SaveFile;
  std::string NameFile;
  
  DQMStore * dbe;
  
};

