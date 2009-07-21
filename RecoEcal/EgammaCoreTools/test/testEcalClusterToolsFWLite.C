#if !defined(__CINT__) && !defined(__MAKECINT__)
#include "DataFormats/FWLite/interface/Handle.h"
#include "DataFormats/FWLite/interface/Event.h"

//Headers for ROOT
#include "TFile.h"
#include "TH1F.h"
#include "TMath.h"

//Headers for the topology and cluster tools
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "Geometry/CaloTopology/interface/EcalBarrelHardcodedTopology.h"
#include "Geometry/CaloTopology/interface/EcalEndcapHardcodedTopology.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterTools.h"

//Headers for the data items
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#endif


void testEcalClusterToolsFWLite() {
  std::cout <<"opening file"<<std::endl;
  TFile *file=TFile::Open("rfio:///castor/cern.ch/cms/store/user/meridian/meridian/SingleGammaPt35_DSZS_V1/SingleGammaPt35_DSZS_V1/00b02d884670d693cb397a1e0af88088/SingleGammaPt35_cfi_py_GEN_SIM_DIGI_L1_DIGI2RAW_RAW2DIGI_RECO_1.root");
  
  TH1F iEtaiEtaEB("iEtaiEtaEB","iEtaiEtaEB",100,0.,0.03); 
  TH1F iEtaiPhiEB("iEtaiPhiEB","iEtaiPhiEB",100,0.,0.03); 
  TH1F iPhiiPhiEB("iPhiiPhiEB","iPhiiPhiEB",100,0.,0.03); 

  TH1F iEtaiEtaEE("iEtaiEtaEE","iEtaiEtaEE",100,0.,0.03); 
  TH1F iEtaiPhiEE("iEtaiPhiEE","iEtaiPhiEE",100,0.,0.03); 
  TH1F iPhiiPhiEE("iPhiiPhiEE","iPhiiPhiEE",100,0.,0.03); 

  fwlite::Event ev(file);

  CaloTopology *topology=new CaloTopology();
  EcalBarrelHardcodedTopology* ebTopology=new EcalBarrelHardcodedTopology();
  EcalEndcapHardcodedTopology* eeTopology=new EcalEndcapHardcodedTopology();
  topology->setSubdetTopology(DetId::Ecal,EcalBarrel,ebTopology);
  topology->setSubdetTopology(DetId::Ecal,EcalEndcap,eeTopology);

  for( ev.toBegin();
       ! ev.atEnd();
       ++ev) {

    fwlite::Handle<reco::BasicClusterCollection > pEBClusters;
    pEBClusters.getByLabel(ev,"hybridSuperClusters","hybridBarrelBasicClusters");
    const reco::BasicClusterCollection *ebClusters = pEBClusters.ptr();

    fwlite::Handle<reco::BasicClusterCollection > pEEClusters;
    pEEClusters.getByLabel(ev,"multi5x5BasicClusters","multi5x5EndcapBasicClusters");
    const reco::BasicClusterCollection *eeClusters = pEEClusters.ptr();

    fwlite::Handle< EcalRecHitCollection > pEBRecHits;
    pEBRecHits.getByLabel( ev, "reducedEcalRecHitsEB","");
    const EcalRecHitCollection *ebRecHits = pEBRecHits.ptr();

    fwlite::Handle< EcalRecHitCollection > pEERecHits;
    pEERecHits.getByLabel( ev, "reducedEcalRecHitsEE","");
    const EcalRecHitCollection *eeRecHits = pEERecHits.ptr();

    std::cout << "========== BARREL ==========" << std::endl;
    for (reco::BasicClusterCollection::const_iterator it = ebClusters->begin(); it != ebClusters->end(); ++it ) {
      std::cout << "----- new cluster -----" << std::endl;
      std::cout << "----------------- size: " << (*it).size() << " energy: " << (*it).energy() << std::endl;

      std::cout << "e1x3..................... " << EcalClusterTools::e1x3( *it, ebRecHits, topology ) << std::endl;
      std::cout << "e3x1..................... " << EcalClusterTools::e3x1( *it, ebRecHits, topology ) << std::endl;
      std::cout << "e1x5..................... " << EcalClusterTools::e1x5( *it, ebRecHits, topology ) << std::endl;
      //std::cout << "e5x1..................... " << EcalClusterTools::e5x1( *it, ebRecHits, topology ) << std::endl;
      std::cout << "e2x2..................... " << EcalClusterTools::e2x2( *it, ebRecHits, topology ) << std::endl;
      std::cout << "e3x3..................... " << EcalClusterTools::e3x3( *it, ebRecHits, topology ) << std::endl;
      std::cout << "e4x4..................... " << EcalClusterTools::e4x4( *it, ebRecHits, topology ) << std::endl;
      std::cout << "e5x5..................... " << EcalClusterTools::e5x5( *it, ebRecHits, topology ) << std::endl;
      std::cout << "e2x5Right................ " << EcalClusterTools::e2x5Right( *it, ebRecHits, topology ) << std::endl;
      std::cout << "e2x5Left................. " << EcalClusterTools::e2x5Left( *it, ebRecHits, topology ) << std::endl;
      std::cout << "e2x5Top.................. " << EcalClusterTools::e2x5Top( *it, ebRecHits, topology ) << std::endl;
      std::cout << "e2x5Bottom............... " << EcalClusterTools::e2x5Bottom( *it, ebRecHits, topology ) << std::endl;
      std::cout << "e2x5Max.................. " << EcalClusterTools::e2x5Max( *it, ebRecHits, topology ) << std::endl;
      std::cout << "eMax..................... " << EcalClusterTools::eMax( *it, ebRecHits ) << std::endl;
      std::cout << "e2nd..................... " << EcalClusterTools::e2nd( *it, ebRecHits ) << std::endl;
      std::vector<float> vEta = EcalClusterTools::energyBasketFractionEta( *it, ebRecHits );
      std::cout << "energyBasketFractionEta..";
      for (size_t i = 0; i < vEta.size(); ++i ) {
	std::cout << " " << vEta[i];
      }
      std::cout << std::endl;
      std::vector<float> vPhi = EcalClusterTools::energyBasketFractionPhi( *it, ebRecHits );
      std::cout << "energyBasketFractionPhi..";
      for (size_t i = 0; i < vPhi.size(); ++i ) {
	std::cout << " " << vPhi[i];
      }
      std::cout << std::endl;
      std::vector<float> vLocCov = EcalClusterTools::localCovariances( *it, ebRecHits, topology );
      std::cout << "local covariances........ " << vLocCov[0] << " " << vLocCov[1] << " " << vLocCov[2] << std::endl;
      if ((*it).energy() < 10) 
	continue;
      iEtaiEtaEB.Fill(TMath::Sqrt(vLocCov[0]));
      iEtaiPhiEB.Fill(TMath::Sqrt(vLocCov[1]));
      iPhiiPhiEB.Fill(TMath::Sqrt(vLocCov[2]));
    }

    std::cout << "========== ENDCAPS ==========" << std::endl;
    for (reco::BasicClusterCollection::const_iterator it = eeClusters->begin(); it != eeClusters->end(); ++it ) {
      std::cout << "----- new cluster -----" << std::endl;
      std::cout << "----------------- size: " << (*it).size() << " energy: " << (*it).energy() << std::endl;
                
      std::cout << "e1x3..................... " << EcalClusterTools::e1x3( *it, eeRecHits, topology ) << std::endl;
      std::cout << "e3x1..................... " << EcalClusterTools::e3x1( *it, eeRecHits, topology ) << std::endl;
      std::cout << "e1x5..................... " << EcalClusterTools::e1x5( *it, eeRecHits, topology ) << std::endl;
      //std::cout << "e5x1..................... " << EcalClusterTools::e5x1( *it, eeRecHits, topology ) << std::endl;
      std::cout << "e2x2..................... " << EcalClusterTools::e2x2( *it, eeRecHits, topology ) << std::endl;
      std::cout << "e3x3..................... " << EcalClusterTools::e3x3( *it, eeRecHits, topology ) << std::endl;
      std::cout << "e4x4..................... " << EcalClusterTools::e4x4( *it, eeRecHits, topology ) << std::endl;
      std::cout << "e5x5..................... " << EcalClusterTools::e5x5( *it, eeRecHits, topology ) << std::endl;
      std::cout << "e2x5Right................ " << EcalClusterTools::e2x5Right( *it, eeRecHits, topology ) << std::endl;
      std::cout << "e2x5Left................. " << EcalClusterTools::e2x5Left( *it, eeRecHits, topology ) << std::endl;
      std::cout << "e2x5Top.................. " << EcalClusterTools::e2x5Top( *it, eeRecHits, topology ) << std::endl;
      std::cout << "e2x5Bottom............... " << EcalClusterTools::e2x5Bottom( *it, eeRecHits, topology ) << std::endl;
      std::cout << "eMax..................... " << EcalClusterTools::eMax( *it, eeRecHits ) << std::endl;
      std::cout << "e2nd..................... " << EcalClusterTools::e2nd( *it, eeRecHits ) << std::endl;
      std::vector<float> vLocCov = EcalClusterTools::localCovariances( *it, eeRecHits, topology );
      std::cout << "local covariances........ " << vLocCov[0] << " " << vLocCov[1] << " " << vLocCov[2] << std::endl;
      if ((*it).energy() < 10) 
	continue;
      iEtaiEtaEE.Fill(TMath::Sqrt(vLocCov[0]));
      iEtaiPhiEE.Fill(TMath::Sqrt(vLocCov[1]));
      iPhiiPhiEE.Fill(TMath::Sqrt(vLocCov[2]));

    }


  }

  //Writing OutFile
  TFile outFile("locCov.root","RECREATE");
  outFile.cd();
  iEtaiEtaEB.Write();
  iEtaiPhiEB.Write();
  iPhiiPhiEB.Write();
  iEtaiEtaEE.Write();
  iEtaiPhiEE.Write();
  iPhiiPhiEE.Write();
  outFile.Write();
  outFile.Close();

  delete topology;
}
