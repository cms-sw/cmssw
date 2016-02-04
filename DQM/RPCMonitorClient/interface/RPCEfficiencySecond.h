/** \class RPCEfficiencySecond
 *
 * Class for RPC Monitoring: use RPCDigi and DT and CSC Segments.
 *
 *  $Date: 2011/07/06 09:09:58 $
 *  $Revision: 1.12 $
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
//#include<vector>
#include<map>
#include<fstream>

class RPCDetId;
/* class TFile; */
/* class TH1F; */
/* class TFile; */
/* class TCanvas; */
/* class TH2F; */
/* class TString; */
/* class TGaxis; */


class RPCEfficiencySecond : public edm::EDAnalyzer {
   public:
      explicit RPCEfficiencySecond(const edm::ParameterSet&);
      ~RPCEfficiencySecond();
      int rollY(std::string shortname,std::vector<std::string> rollNames);
  
 /*  TFile * theFile; */
/*   TFile * theFileout; */

 /*  MonitorElement * statistics; */
/*   MonitorElement * statistics2; */

/*   MonitorElement * hGlobalResClu1La1; */
/*   MonitorElement * hGlobalResClu1La2; */
/*   MonitorElement * hGlobalResClu1La3; */
/*   MonitorElement * hGlobalResClu1La4; */
/*   MonitorElement * hGlobalResClu1La5; */
/*   MonitorElement * hGlobalResClu1La6; */

/*   MonitorElement * hGlobalResClu2La1; */
/*   MonitorElement * hGlobalResClu2La2; */
/*   MonitorElement * hGlobalResClu2La3; */
/*   MonitorElement * hGlobalResClu2La4; */
/*   MonitorElement * hGlobalResClu2La5; */
/*   MonitorElement * hGlobalResClu2La6; */

/*   MonitorElement * hGlobalResClu3La1; */
/*   MonitorElement * hGlobalResClu3La2; */
/*   MonitorElement * hGlobalResClu3La3; */
/*   MonitorElement * hGlobalResClu3La4; */
/*   MonitorElement * hGlobalResClu3La5; */
/*   MonitorElement * hGlobalResClu3La6; */

/*   //Endcap   */
/*   MonitorElement * hGlobalResClu1R3C; */
/*   MonitorElement * hGlobalResClu1R3B; */
/*   MonitorElement * hGlobalResClu1R3A; */
/*   MonitorElement * hGlobalResClu1R2C; */
/*   MonitorElement * hGlobalResClu1R2B;  */
/*   MonitorElement * hGlobalResClu1R2A; */

/*   MonitorElement * hGlobalResClu2R3C; */
/*   MonitorElement * hGlobalResClu2R3B; */
/*   MonitorElement * hGlobalResClu2R3A; */
/*   MonitorElement * hGlobalResClu2R2C; */
/*   MonitorElement * hGlobalResClu2R2B; */
/*   MonitorElement * hGlobalResClu2R2A; */

/*   MonitorElement * hGlobalResClu3R3C; */
/*   MonitorElement * hGlobalResClu3R3B; */
/*   MonitorElement * hGlobalResClu3R3A; */
/*   MonitorElement * hGlobalResClu3R2C; */
/*   MonitorElement * hGlobalResClu3R2B; */
/*   MonitorElement * hGlobalResClu3R2A; */

/*   MonitorElement * hGlobal2ResClu1La1; */
/*   MonitorElement * hGlobal2ResClu1La2; */
/*   MonitorElement * hGlobal2ResClu1La3; */
/*   MonitorElement * hGlobal2ResClu1La4; */
/*   MonitorElement * hGlobal2ResClu1La5; */
/*   MonitorElement * hGlobal2ResClu1La6; */

/*   //SecondHistograms */

/*   MonitorElement * hGlobal2ResClu2La1; */
/*   MonitorElement * hGlobal2ResClu2La2; */
/*   MonitorElement * hGlobal2ResClu2La3; */
/*   MonitorElement * hGlobal2ResClu2La4; */
/*   MonitorElement * hGlobal2ResClu2La5; */
/*   MonitorElement * hGlobal2ResClu2La6; */

/*   MonitorElement * hGlobal2ResClu3La1; */
/*   MonitorElement * hGlobal2ResClu3La2; */
/*   MonitorElement * hGlobal2ResClu3La3; */
/*   MonitorElement * hGlobal2ResClu3La4; */
/*   MonitorElement * hGlobal2ResClu3La5; */
/*   MonitorElement * hGlobal2ResClu3La6; */

/*   //Endcap   */
/*   MonitorElement * hGlobal2ResClu1R3C; */
/*   MonitorElement * hGlobal2ResClu1R3B; */
/*   MonitorElement * hGlobal2ResClu1R3A; */
/*   MonitorElement * hGlobal2ResClu1R2C; */
/*   MonitorElement * hGlobal2ResClu1R2B;  */
/*   MonitorElement * hGlobal2ResClu1R2A; */

/*   MonitorElement * hGlobal2ResClu2R3C; */
/*   MonitorElement * hGlobal2ResClu2R3B; */
/*   MonitorElement * hGlobal2ResClu2R3A; */
/*   MonitorElement * hGlobal2ResClu2R2C; */
/*   MonitorElement * hGlobal2ResClu2R2B; */
/*   MonitorElement * hGlobal2ResClu2R2A; */

/*   MonitorElement * hGlobal2ResClu3R3C; */
/*   MonitorElement * hGlobal2ResClu3R3B; */
/*   MonitorElement * hGlobal2ResClu3R3A; */
/*   MonitorElement * hGlobal2ResClu3R2C; */
/*   MonitorElement * hGlobal2ResClu3R2B; */
/*   MonitorElement * hGlobal2ResClu3R2A; */

  //Histograms to use
  MonitorElement * histoRPC;
  MonitorElement * histoDT;
  MonitorElement * histoRealRPC;
  MonitorElement * histoCSC;
  //  MonitorElement * BXDistribution;

  MonitorElement * histoPRO;
  MonitorElement * histoeffIdRPC_DT;
  MonitorElement * histoeffIdRPC_CSC;

  //For Duplication
  MonitorElement * histoRPC2;
  MonitorElement * histoDT2;
  MonitorElement * histoRealRPC2;
  MonitorElement * histoCSC2;
  //  MonitorElement * BXDistribution2;


  //Eff Global Barrel
  MonitorElement * EffGlobWm2;
  MonitorElement * EffGlobWm1;
  MonitorElement * EffGlobW0;
  MonitorElement * EffGlobW1;
  MonitorElement * EffGlobW2;
 
  //MonitorElement * EffGlobWm2far;
  //MonitorElement * EffGlobWm1far;
  //MonitorElement * EffGlobW0far;
  //MonitorElement * EffGlobW1far;
  //MonitorElement * EffGlobW2far;

 /*  //BX Barrel  */
/*   MonitorElement * BXGlobWm2; */
/*   MonitorElement * BXGlobWm1; */
/*   MonitorElement * BXGlobW0; */
/*   MonitorElement * BXGlobW1; */
/*   MonitorElement * BXGlobW2; */
  
/*   MonitorElement * BXGlobWm2far; */
/*   MonitorElement * BXGlobWm1far; */
/*   MonitorElement * BXGlobW0far; */
/*   MonitorElement * BXGlobW1far; */
/*   MonitorElement * BXGlobW2far; */
  
  //Masked Barrel
  //MonitorElement * MaskedGlobWm2;
  //MonitorElement * MaskedGlobWm1;
  //MonitorElement * MaskedGlobW0;
  //MonitorElement * MaskedGlobW1;
  //MonitorElement * MaskedGlobW2;
  
  //MonitorElement * MaskedGlobWm2far;
  //MonitorElement * MaskedGlobWm1far;
  //MonitorElement * MaskedGlobW0far;
  //MonitorElement * MaskedGlobW1far;
  //MonitorElement * MaskedGlobW2far;

  //Average Eff Barrel 
  //MonitorElement * AverageEffWm2;
  //MonitorElement * AverageEffWm1;
  //MonitorElement * AverageEffW0;
  //MonitorElement * AverageEffW1;
  //MonitorElement * AverageEffW2;

  //MonitorElement * AverageEffWm2far;
  //MonitorElement * AverageEffWm1far;
  //MonitorElement * AverageEffW0far;
  //MonitorElement * AverageEffW1far;
  //MonitorElement * AverageEffW2far;

  //No Prediction Barrel 
  //MonitorElement * NoPredictionWm2;
  //MonitorElement * NoPredictionWm1;
  //MonitorElement * NoPredictionW0;
  //MonitorElement * NoPredictionW1;
  //MonitorElement * NoPredictionW2;

  //MonitorElement * NoPredictionWm2far;
  //MonitorElement * NoPredictionWm1far;
  //MonitorElement * NoPredictionW0far;
  //MonitorElement * NoPredictionW1far;
  //MonitorElement * NoPredictionW2far;

  //Eff Distro Barrel
  MonitorElement * EffDistroWm2;
  MonitorElement * EffDistroWm1;
  MonitorElement * EffDistroW0;
  MonitorElement * EffDistroW1;
  MonitorElement * EffDistroW2;

  //MonitorElement * EffDistroWm2far;
  //MonitorElement * EffDistroWm1far;
  //MonitorElement * EffDistroW0far;
  //MonitorElement * EffDistroW1far;
  //MonitorElement * EffDistroW2far;
 

  //Eff Global EndCap

  MonitorElement * EffGlobDm3;
  MonitorElement * EffGlobDm2;
  MonitorElement * EffGlobDm1;
  MonitorElement * EffGlobD1;
  MonitorElement * EffGlobD2;
  MonitorElement * EffGlobD3;

  //MonitorElement * EffGlobDm3far;
  //MonitorElement * EffGlobDm2far;
  //MonitorElement * EffGlobDm1far;
  //MonitorElement * EffGlobD1far;
  //MonitorElement * EffGlobD2far;
  //MonitorElement * EffGlobD3far;

  //BX EndCap
/*   MonitorElement * BXGlobDm3; */
/*   MonitorElement * BXGlobDm2; */
/*   MonitorElement * BXGlobDm1; */
/*   MonitorElement * BXGlobD1; */
/*   MonitorElement * BXGlobD2; */
/*   MonitorElement * BXGlobD3; */
  
/*   MonitorElement * BXGlobDm3far; */
/*   MonitorElement * BXGlobDm2far; */
/*   MonitorElement * BXGlobDm1far; */
/*   MonitorElement * BXGlobD1far; */
/*   MonitorElement * BXGlobD2far; */
/*   MonitorElement * BXGlobD3far; */

  //Masked EndCap
  //MonitorElement * MaskedGlobDm3;
  //MonitorElement * MaskedGlobDm2;
  //MonitorElement * MaskedGlobDm1;
  //MonitorElement * MaskedGlobD1;
  //MonitorElement * MaskedGlobD2;
  //MonitorElement * MaskedGlobD3;
  
  //MonitorElement * MaskedGlobDm3far;
  //MonitorElement * MaskedGlobDm2far;
  //MonitorElement * MaskedGlobDm1far;
  //MonitorElement * MaskedGlobD1far;
  //MonitorElement * MaskedGlobD2far;
  //MonitorElement * MaskedGlobD3far;

  //Average Eff EndCap
  //MonitorElement * AverageEffDm3;
  //MonitorElement * AverageEffDm2;
  //MonitorElement * AverageEffDm1;
  //MonitorElement * AverageEffD1;
  //MonitorElement * AverageEffD2;
  //MonitorElement * AverageEffD3;

  //MonitorElement * AverageEffDm3far;
  //MonitorElement * AverageEffDm2far;
  //MonitorElement * AverageEffDm1far;
  //MonitorElement * AverageEffD1far;
  //MonitorElement * AverageEffD2far;
  //MonitorElement * AverageEffD3far;

  //No Prediction EndCap
  //MonitorElement * NoPredictionDm3;
  //MonitorElement * NoPredictionDm2;
  //MonitorElement * NoPredictionDm1;
  //MonitorElement * NoPredictionD1;
  //MonitorElement * NoPredictionD2;
  //MonitorElement * NoPredictionD3;

  //MonitorElement * NoPredictionDm3far;
  //MonitorElement * NoPredictionDm2far;
  //MonitorElement * NoPredictionDm1far;
  //MonitorElement * NoPredictionD1far;
  //MonitorElement * NoPredictionD2far;
  //MonitorElement * NoPredictionD3far;

  //EffDistro EndCap
  MonitorElement * EffDistroDm3;
  MonitorElement * EffDistroDm2;
  MonitorElement * EffDistroDm1;
  MonitorElement * EffDistroD1;
  MonitorElement * EffDistroD2;
  MonitorElement * EffDistroD3;

  //MonitorElement * EffDistroDm3far;
  //MonitorElement * EffDistroDm2far;
  //MonitorElement * EffDistroDm1far;
  //MonitorElement * EffDistroD1far;
  //MonitorElement * EffDistroD2far;
  //MonitorElement * EffDistroD3far;

  //Summary Histograms.
  MonitorElement * Wheelm2Summary;
  MonitorElement * Wheelm1Summary; 
  MonitorElement * Wheel0Summary; 
  MonitorElement * Wheel1Summary; 
  MonitorElement * Wheel2Summary; 

  MonitorElement * Diskm3Summary;
  MonitorElement * Diskm2Summary;
  MonitorElement * Diskm1Summary;
  MonitorElement * Disk1Summary;
  MonitorElement * Disk2Summary;
  MonitorElement * Disk3Summary;

  //Azimultal Plots

  MonitorElement *  sectorEffWm2;  
  MonitorElement * sectorEffWm1;  
  MonitorElement * sectorEffW0;  
  MonitorElement * sectorEffW1;  
  MonitorElement * sectorEffW2;  

  MonitorElement *  OcsectorEffWm2;  
  MonitorElement * OcsectorEffWm1;  
  MonitorElement * OcsectorEffW0;  
  MonitorElement * OcsectorEffW1;  
  MonitorElement * OcsectorEffW2;  

  MonitorElement * ExsectorEffWm2;  
  MonitorElement * ExsectorEffWm1;  
  MonitorElement * ExsectorEffW0;  
  MonitorElement * ExsectorEffW1;  
  MonitorElement * ExsectorEffW2;  

  MonitorElement * GregD1R2;  
  MonitorElement * GregD1R3;  
  MonitorElement * GregD2R2;  
  MonitorElement * GregD2R3;  
  MonitorElement * GregD3R2;  
  MonitorElement * GregD3R3;  
  MonitorElement * GregDm1R2;  
  MonitorElement * GregDm1R3;  
  MonitorElement * GregDm2R2;  
  MonitorElement * GregDm2R3;  
  MonitorElement * GregDm3R2;  
  MonitorElement * GregDm3R3;  

  MonitorElement * OcGregD1R2;  
  MonitorElement * OcGregD1R3;  
  MonitorElement * OcGregD2R2;  
  MonitorElement * OcGregD2R3;  
  MonitorElement * OcGregD3R2;  
  MonitorElement * OcGregD3R3;  
  MonitorElement * OcGregDm1R2;  
  MonitorElement * OcGregDm1R3;  
  MonitorElement * OcGregDm2R2;  
  MonitorElement * OcGregDm2R3;  
  MonitorElement * OcGregDm3R2;  
  MonitorElement * OcGregDm3R3;  

  MonitorElement * ExGregD1R2;  
  MonitorElement * ExGregD1R3;  
  MonitorElement * ExGregD2R2;  
  MonitorElement * ExGregD2R3;  
  MonitorElement * ExGregD3R2;  
  MonitorElement * ExGregD3R3;  
  MonitorElement * ExGregDm1R2;  
  MonitorElement * ExGregDm1R3;  
  MonitorElement * ExGregDm2R2;  
  MonitorElement * ExGregDm2R3;  
  MonitorElement * ExGregDm3R2;  
  MonitorElement * ExGregDm3R3;  

  MonitorElement * ExpLayerWm2;
  MonitorElement * ExpLayerWm1;
  MonitorElement * ExpLayerW0;
  MonitorElement * ExpLayerW1;
  MonitorElement * ExpLayerW2;

  MonitorElement * ObsLayerWm2;
  MonitorElement * ObsLayerWm1;
  MonitorElement * ObsLayerW0;
  MonitorElement * ObsLayerW1;
  MonitorElement * ObsLayerW2;


  
 private:
  virtual void beginRun(const edm::Run&, const edm::EventSetup& iSetup) ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;
  virtual void endRun(const edm::Run& , const edm::EventSetup& );

  std::map<std::string, MonitorElement*> bookDetUnitSeg(RPCDetId & detId,int nstrips,std::string folder);
  std::map<int, std::map<std::string, MonitorElement*> >  meCollection;
  
  bool debug;
  bool endcap;
  bool barrel;
  bool SaveFile;
  std::string NameFile;
  std::string folderPath;
  
  DQMStore * dbe;
  
};

