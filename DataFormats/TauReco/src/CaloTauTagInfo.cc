#include "DataFormats/TauReco/interface/CaloTauTagInfo.h"
using namespace std;
using namespace edm;
using namespace reco;

CaloTauTagInfo* CaloTauTagInfo::clone()const{return new CaloTauTagInfo(*this);}
    
const CaloJetRef& CaloTauTagInfo::calojetRef()const{return CaloJetRef_;}
void CaloTauTagInfo::setcalojetRef(const CaloJetRef x){CaloJetRef_=x;}

const JetBaseRef CaloTauTagInfo::jetRef()const{
        if(JetRef_.isNonnull()) {
          return JetRef_;
        }else return JetBaseRef(CaloJetRef_);
}

#include "DataFormats/JetReco/interface/JPTJet.h"
void CaloTauTagInfo::setJetRef(const JetBaseRef x){
        JetRef_=x;
        const reco::Jet *base = x.get();

        if(dynamic_cast<const reco::CaloJet *>(base)) {
                CaloJetRef_ = x.castTo<reco::CaloJetRef>();
        }
        else if(dynamic_cast<const reco::JPTJet *>(base)) {
                reco::JPTJetRef const theJPTJetRef = JetRef_.castTo<reco::JPTJetRef>();
                reco::CaloJetRef const theCaloJetRef = (theJPTJetRef->getCaloJetRef()).castTo<reco::CaloJetRef>();
                CaloJetRef_ = theCaloJetRef;
        }
        else {
                throw cms::Exception("LogicError") << "CaloTauTagInfo supports reco::CaloJet and reco::JPTJet, got "
                                                   << typeid(base).name();
        }
}
 
const vector<pair<math::XYZPoint,float> > CaloTauTagInfo::positionAndEnergyECALRecHits()const{return positionAndEnergyECALRecHits_;}
void CaloTauTagInfo::setpositionAndEnergyECALRecHits(const std::vector<pair<math::XYZPoint,float> >& x){positionAndEnergyECALRecHits_=x;}
  
const vector<BasicClusterRef> CaloTauTagInfo::neutralECALBasicClusters()const{return neutralECALBasicClusters_;}
void CaloTauTagInfo::setneutralECALBasicClusters(const std::vector<BasicClusterRef>& x){neutralECALBasicClusters_=x;}
  
