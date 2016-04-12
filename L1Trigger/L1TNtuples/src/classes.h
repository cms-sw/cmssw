#include "L1Trigger/L1TNtuples/interface/L1AnalysisEventDataFormat.h"
#include "L1Trigger/L1TNtuples/interface/L1AnalysisCSCTFDataFormat.h"
#include "L1Trigger/L1TNtuples/interface/L1AnalysisDTTFDataFormat.h"
#include "L1Trigger/L1TNtuples/interface/L1AnalysisGCTDataFormat.h"
#include "L1Trigger/L1TNtuples/interface/L1AnalysisGMTDataFormat.h"
#include "L1Trigger/L1TNtuples/interface/L1AnalysisGTDataFormat.h"
#include "L1Trigger/L1TNtuples/interface/L1AnalysisRCTDataFormat.h"
#include "L1Trigger/L1TNtuples/interface/L1AnalysisCaloTPDataFormat.h"
#include "L1Trigger/L1TNtuples/interface/L1AnalysisGeneratorDataFormat.h"
#include "L1Trigger/L1TNtuples/interface/L1AnalysisSimulationDataFormat.h"
#include "L1Trigger/L1TNtuples/interface/L1AnalysisL1ExtraDataFormat.h"
#include "L1Trigger/L1TNtuples/interface/L1AnalysisRecoMuonDataFormat.h"
#include "L1Trigger/L1TNtuples/interface/L1AnalysisRecoRpcHitDataFormat.h"
#include "L1Trigger/L1TNtuples/interface/L1AnalysisRecoMetDataFormat.h"
#include "L1Trigger/L1TNtuples/interface/L1AnalysisRecoMetFilterDataFormat.h"
#include "L1Trigger/L1TNtuples/interface/L1AnalysisRecoJetDataFormat.h"
#include "L1Trigger/L1TNtuples/interface/L1AnalysisRecoClusterDataFormat.h"
#include "L1Trigger/L1TNtuples/interface/L1AnalysisRecoVertexDataFormat.h"
#include "L1Trigger/L1TNtuples/interface/L1AnalysisRecoTrackDataFormat.h"
#include "L1Trigger/L1TNtuples/interface/L1AnalysisL1MenuDataFormat.h"
#include "L1Trigger/L1TNtuples/interface/L1AnalysisL1UpgradeDataFormat.h"
#include "L1Trigger/L1TNtuples/interface/L1AnalysisL1UpgradeTfMuonDataFormat.h"
#include "L1Trigger/L1TNtuples/interface/L1AnalysisL1CaloTowerDataFormat.h"
#include "L1Trigger/L1TNtuples/interface/L1AnalysisL1CaloClusterDataFormat.h"

#include "L1Trigger/L1TNtuples/interface/L1AnalysisRecoTauDataFormat.h"
#include "L1Trigger/L1TNtuples/interface/L1AnalysisRecoMuon2DataFormat.h"
#include "L1Trigger/L1TNtuples/interface/L1AnalysisRecoElectronDataFormat.h"

namespace{
  namespace{
    L1Analysis::L1AnalysisEventDataFormat          pL1ev; 
    L1Analysis::L1AnalysisCSCTFDataFormat          pL1csctf;
    L1Analysis::L1AnalysisDTTFDataFormat    	   pL1dttf;
    L1Analysis::L1AnalysisGCTDataFormat   	   pL1gct;
    L1Analysis::L1AnalysisGMTDataFormat  	   pL1gmt;
    L1Analysis::L1AnalysisGTDataFormat   	   pL1gt;
    L1Analysis::L1AnalysisRCTDataFormat            pL1rct;
    L1Analysis::L1AnalysisCaloTPDataFormat         pL1calotp;
    L1Analysis::L1AnalysisGeneratorDataFormat      pL1generator;
    L1Analysis::L1AnalysisSimulationDataFormat     pL1simu;
    L1Analysis::L1AnalysisL1ExtraDataFormat        pLl1extra;
    L1Analysis::L1AnalysisRecoMuonDataFormat       pL1muon;
    L1Analysis::L1AnalysisRecoRpcHitDataFormat     pL1rpc;
    L1Analysis::L1AnalysisRecoJetDataFormat        pL1jet; 
    L1Analysis::L1AnalysisRecoMetDataFormat        pL1met; 
    L1Analysis::L1AnalysisRecoMetFilterDataFormat  pL1metfilter;
    L1Analysis::L1AnalysisRecoClusterDataFormat    pL1cluster;
    L1Analysis::L1AnalysisRecoVertexDataFormat     pL1vertex;   
    L1Analysis::L1AnalysisRecoTrackDataFormat      pL1track; 
    L1Analysis::L1AnalysisL1MenuDataFormat         pL1menu; 
    L1Analysis::L1AnalysisL1UpgradeDataFormat      pL1upgrade; 
    L1Analysis::L1AnalysisL1UpgradeTfMuonDataFormat pL1upgradeTfMuon; 
    L1Analysis::L1AnalysisL1CaloTowerDataFormat      pL1calotower; 
    L1Analysis::L1AnalysisL1CaloClusterDataFormat      pL1calocluster; 

    L1Analysis::L1AnalysisRecoTauDataFormat        pL1tau; 
    L1Analysis::L1AnalysisRecoMuon2DataFormat      pL1muon2;
    L1Analysis::L1AnalysisRecoElectronDataFormat   pL1electron;
   }
}
