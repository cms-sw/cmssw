/** \class RPCEfficiencySecond
 *
 * Class for RPC Monitoring using RPCDigi and DT and CSC Segments.
 *
 *  $Date: 2008/06/16 12:25:25 $
 *  $Revision: 1.21 $
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

  TH1F * histoRPC;
  TH1F * histoDT;
  TH1F * histoRPC_2D;
  TH1F * histoDT_2D;
  TH1F * histoeffIdRPC_DT_2D;
  TH1F * histoeffIdRPC_DT;
  TH1F * BXDistribution;
  TH1F * histoRealRPC;

  TH1F * EffGlobWm2;
  TH1F * EffGlobWm1;
  TH1F * EffGlobW0;
  TH1F * EffGlobW1;
  TH1F * EffGlobW2;

  TH1F * EffGlobWm2far;
  TH1F * EffGlobWm1far;
  TH1F * EffGlobW0far;
  TH1F * EffGlobW1far;
  TH1F * EffGlobW2far;

  TH1F * BXGlobWm2;
  TH1F * BXGlobWm1;
  TH1F * BXGlobW0;
  TH1F * BXGlobW1;
  TH1F * BXGlobW2;
  
  TH1F * BXGlobWm2far;
  TH1F * BXGlobWm1far;
  TH1F * BXGlobW0far;
  TH1F * BXGlobW1far;
  TH1F * BXGlobW2far;

  TH1F * MaskedGlobWm2;
  TH1F * MaskedGlobWm1;
  TH1F * MaskedGlobW0;
  TH1F * MaskedGlobW1;
  TH1F * MaskedGlobW2;
  
  TH1F * MaskedGlobWm2far;
  TH1F * MaskedGlobWm1far;
  TH1F * MaskedGlobW0far;
  TH1F * MaskedGlobW1far;
  TH1F * MaskedGlobW2far;

  TGaxis * bxAxis;
  
   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      std::string file;
      std::string fileOut;
      std::ofstream rpcInfo;
      std::ofstream rpcNames;
      std::ofstream rollsWithData;
      std::ofstream rollsWithOutData;
      std::ofstream rollsBarrel;
      std::ofstream rollsEndCap;
      std::ofstream rollsPointedForASegment;
      std::ofstream rollsNotPointedForASegment;
      std::ofstream bxMeanList;
};

