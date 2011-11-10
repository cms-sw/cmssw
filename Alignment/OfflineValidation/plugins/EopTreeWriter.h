// system include files
#include <memory>

// framework include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/ServiceRegistry/interface/Service.h"

// user include files
#include <DataFormats/TrackReco/interface/Track.h>
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include <TMath.h>
#include <TH1.h>
#include "TTree.h"

#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "CommonTools/Utils/interface/TFileDirectory.h"

#include "TrackingTools/TrackAssociator/interface/TrackDetectorAssociator.h"
#include "TrackingTools/TrackAssociator/interface/TrackAssociatorParameters.h"
#include "TrackingTools/GeomPropagators/interface/PropagationExceptions.h"
#include "TrackPropagation/SteppingHelixPropagator/interface/SteppingHelixPropagator.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/CaloTowers/interface/CaloTowerFwd.h"
#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"

#include "RecoParticleFlow/PFRootEvent/interface/JetRecoTypes.h"
#include "RecoParticleFlow/PFRootEvent/interface/JetMaker.h"
#include "RecoParticleFlow/PFRootEvent/interface/ProtoJet.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"

#include "DataFormats/HcalIsolatedTrack/interface/IsolatedPixelTrackCandidateFwd.h"
#include "DataFormats/HcalIsolatedTrack/interface/IsolatedPixelTrackCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoCaloTowerCandidate.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/DTRecHit/interface/DTRecHitCollection.h"

#include "Alignment/OfflineValidation/interface/EopVariables.h"

using namespace reco;
using namespace std;

edm::Service<TFileService> fs;
TTree *tree;
EopVariables *treeMemPtr;
TrackDetectorAssociator trackAssociator_;
TrackAssociatorParameters parameters_;

double getDistInCM(double eta1, double phi1, double eta2, double phi2)
{
  double deltaPhi=phi1-phi2;
  while(deltaPhi >   TMath::Pi())deltaPhi-=2*TMath::Pi();
  while(deltaPhi <= -TMath::Pi())deltaPhi+=2*TMath::Pi();
  double dR, Rec;
  double theta1=2*atan(exp(-eta1));
  double theta2=2*atan(exp(-eta2));
  double cotantheta1;
  if(cos(theta1)==0)cotantheta1=0;
  else cotantheta1=1/tan(theta1);
  double cotantheta2;
  if(cos(theta2)==0)cotantheta2=0;
  else cotantheta2=1/tan(theta2);
  if (fabs(eta1)<1.479) Rec=129; //radius of ECAL barrel
  else Rec=317; //distance from IP to ECAL endcap
/*   //|vect| times tg of acos(scalar product) */
/*   dR=fabs((Rec/sin(theta1))*tan(acos(sin(theta1)*sin(theta2)*(sin(phi1)*sin(phi2)+cos(phi1)*cos(phi2))+cos(theta1)*cos(theta2)))); */
  if(fabs(eta1)<1.479)dR=129*sqrt((cotantheta1-cotantheta2)*(cotantheta1-cotantheta2)+deltaPhi*deltaPhi);
  else dR=317*sqrt(tan(theta1)*tan(theta1)+tan(theta2)*tan(theta2)-2*tan(theta1)*tan(theta2)*cos(deltaPhi));
  return dR;
}
