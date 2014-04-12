#ifndef DataFormats_TauReco_CaloTauTagInfo_h
#define DataFormats_TauReco_CaloTauTagInfo_h

/* class CaloTauTagInfo
 * the object of this class is created by RecoTauTag/RecoTau CaloRecoTauTagInfoProducer EDProducer starting from JetTrackAssociations <a CaloJet,a list of Tracks> object,
 *                          is the initial object for building a CaloTau object;
 * created: Sep 4 2007,
 * revised: Feb 20 2008,
 * authors: Ludovic Houchu
 */

#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/TauReco/interface/BaseTauTagInfo.h"
#include "DataFormats/TauReco/interface/CaloTauTagInfoFwd.h"
#include "DataFormats/JetReco/interface/JetCollection.h"

namespace reco{ 
  class CaloTauTagInfo : public BaseTauTagInfo {
  public:
    CaloTauTagInfo(){}
    virtual ~CaloTauTagInfo(){};
    virtual CaloTauTagInfo* clone()const;
    
    //the reference to the CaloJet
    const CaloJetRef& calojetRef()const;
    void setcalojetRef(const CaloJetRef);

    const JetBaseRef jetRef()const;
    void setJetRef(const JetBaseRef);

    const std::vector<std::pair<math::XYZPoint,float> > positionAndEnergyECALRecHits()const;
    void setpositionAndEnergyECALRecHits(const std::vector<std::pair<math::XYZPoint,float> >&);

    const std::vector<BasicClusterRef> neutralECALBasicClusters()const;
    void setneutralECALBasicClusters(const std::vector<BasicClusterRef>&);
  private:
    CaloJetRef CaloJetRef_;
    std::vector<std::pair<math::XYZPoint,float> > positionAndEnergyECALRecHits_;
    std::vector<BasicClusterRef> neutralECALBasicClusters_;
    JetBaseRef JetRef_;
  };
}

#endif

