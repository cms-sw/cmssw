#include "DataFormats/TauReco/interface/CaloTauTagInfo.h"
using namespace std;
using namespace edm;
using namespace reco;

CaloTauTagInfo* CaloTauTagInfo::clone()const{return new CaloTauTagInfo(*this);}
    
const CaloJetRef& CaloTauTagInfo::calojetRef()const{return CaloJetRef_;}
void CaloTauTagInfo::setcalojetRef(const CaloJetRef x){CaloJetRef_=x;}
 
const vector<pair<math::XYZPoint,float> > CaloTauTagInfo::positionAndEnergyECALRecHits()const{return positionAndEnergyECALRecHits_;}
void CaloTauTagInfo::setpositionAndEnergyECALRecHits(vector<pair<math::XYZPoint,float> > x){positionAndEnergyECALRecHits_=x;}
  
const vector<BasicClusterRef> CaloTauTagInfo::neutralECALBasicClusters()const{return neutralECALBasicClusters_;}
void CaloTauTagInfo::setneutralECALBasicClusters(vector<BasicClusterRef> x){neutralECALBasicClusters_=x;}
  
