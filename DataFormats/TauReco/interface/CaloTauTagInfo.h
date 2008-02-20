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

using namespace std;
using namespace edm;
using namespace reco;

namespace reco{ 
  class CaloTauTagInfo : public BaseTauTagInfo {
  public:
    CaloTauTagInfo(){}
    virtual ~CaloTauTagInfo(){};
    virtual CaloTauTagInfo* clone()const;
    
    //the reference to the CaloJet
    const CaloJetRef& calojetRef()const;
    void setcalojetRef(const CaloJetRef);

    const vector<pair<math::XYZPoint,float> > positionAndEnergyECALRecHits()const;
    void setpositionAndEnergyECALRecHits(vector<pair<math::XYZPoint,float> >);

    const vector<BasicClusterRef> neutralECALBasicClusters()const;
    void setneutralECALBasicClusters(vector<BasicClusterRef>);
  private:
    CaloJetRef CaloJetRef_;
    vector<pair<math::XYZPoint,float> > positionAndEnergyECALRecHits_;
    vector<BasicClusterRef> neutralECALBasicClusters_;
  };
}

#endif

