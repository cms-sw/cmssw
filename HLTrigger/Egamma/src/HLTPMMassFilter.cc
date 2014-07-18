/** \class HLTPMMassFilter
 *
 * Original Author: Jeremy Werner
 * Institution: Princeton University, USA
 * Contact: Jeremy.Werner@cern.ch
 * Date: February 21, 2007
 */

#include "FWCore/Framework/interface/ESHandle.h"

#include "HLTrigger/Egamma/interface/HLTPMMassFilter.h"

//
// constructors and destructor
//
HLTPMMassFilter::HLTPMMassFilter(const edm::ParameterSet& iConfig) : HLTFilter(iConfig)
{
  candTag_            = iConfig.getParameter< edm::InputTag > ("candTag");
  beamSpot_           = iConfig.getParameter< edm::InputTag > ("beamSpot");
  lowerMassCut_       = iConfig.getParameter<double> ("lowerMassCut");
  upperMassCut_       = iConfig.getParameter<double> ("upperMassCut");
  nZcandcut_          = iConfig.getParameter<int> ("nZcandcut");
  reqOppCharge_       = iConfig.getUntrackedParameter<bool> ("reqOppCharge",false);
  isElectron1_ = iConfig.getUntrackedParameter<bool> ("isElectron1",true) ;
  isElectron2_ = iConfig.getUntrackedParameter<bool> ("isElectron2",true) ;
  relaxed_ = iConfig.getUntrackedParameter<bool> ("relaxed",true) ;
  L1IsoCollTag_= iConfig.getParameter< edm::InputTag > ("L1IsoCand");
  L1NonIsoCollTag_= iConfig.getParameter< edm::InputTag > ("L1NonIsoCand");
  candToken_ = consumes<trigger::TriggerFilterObjectWithRefs>(candTag_);
  beamSpotToken_ = consumes<reco::BeamSpot>(beamSpot_);
}

HLTPMMassFilter::~HLTPMMassFilter(){}

void
HLTPMMassFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  makeHLTFilterDescription(desc);
  desc.add<edm::InputTag>("candTag",edm::InputTag("hltL1NonIsoDoublePhotonEt5UpsHcalIsolFilter"));
  desc.add<edm::InputTag>("beamSpot",edm::InputTag("hltOfflineBeamSpot"));
  desc.add<double>("lowerMassCut",8.0);
  desc.add<double>("upperMassCut",11.0);
  desc.add<int>("nZcandcut",1);
  desc.addUntracked<bool>("reqOppCharge",true);
  desc.addUntracked<bool>("isElectron1",false);
  desc.addUntracked<bool>("isElectron2",false);
  desc.addUntracked<bool>("relaxed",true);
  desc.add<edm::InputTag>("L1IsoCand",edm::InputTag("hltL1IsoRecoEcalCandidate"));
  desc.add<edm::InputTag>("L1NonIsoCand",edm::InputTag("hltL1IsoRecoEcalCandidate"));
  descriptions.add("hltPMMassFilter",desc);
}

// ------------ method called to produce the data  ------------
bool
HLTPMMassFilter::hltFilter(edm::Event& iEvent, const edm::EventSetup& iSetup, trigger::TriggerFilterObjectWithRefs & filterproduct) const
{
  using namespace std;
  using namespace edm;
  using namespace reco;
  using namespace trigger;

  // The filter object
  if (saveTags()) {
    filterproduct.addCollectionTag(L1IsoCollTag_);
    if (relaxed_) filterproduct.addCollectionTag(L1NonIsoCollTag_);
  }

  edm::ESHandle<MagneticField> theMagField;
  iSetup.get<IdealMagneticFieldRecord>().get(theMagField);

  edm::Handle<trigger::TriggerFilterObjectWithRefs> PrevFilterOutput;
  iEvent.getByToken (candToken_,PrevFilterOutput);

  // beam spot
  edm::Handle<reco::BeamSpot> recoBeamSpotHandle;
  iEvent.getByToken(beamSpotToken_,recoBeamSpotHandle);
  // gets its position
  const GlobalPoint vertexPos(recoBeamSpotHandle->position().x(),
			      recoBeamSpotHandle->position().y(),
			      recoBeamSpotHandle->position().z());

  int n = 0;

  // REMOVED USAGE OF STATIC ARRAYS
  // double px[66];
  // double py[66];
  // double pz[66];
  // double energy[66];
  std::vector<TLorentzVector> pEleCh1;
  std::vector<TLorentzVector> pEleCh2;
  std::vector<double> charge;
  std::vector<double> etaOrig;

  if (isElectron1_ && isElectron2_) {

    Ref< ElectronCollection > refele;

    vector< Ref< ElectronCollection > > electrons;
    PrevFilterOutput->getObjects(TriggerElectron, electrons);

    for (unsigned int i=0; i<electrons.size(); i++) {

      refele = electrons[i];

      TLorentzVector pThisEle(refele->px(), refele->py(),
			      refele->pz(), refele->energy() );
      pEleCh1.push_back( pThisEle );
      charge.push_back( refele->charge() );
    }

    for(unsigned int jj=0;jj<electrons.size();jj++){

      TLorentzVector p1Ele = pEleCh1.at(jj);
      for(unsigned int ii=jj+1;ii<electrons.size();ii++){
	
	TLorentzVector p2Ele = pEleCh1.at(ii);
	
	if(fabs(p1Ele.E() - p2Ele.E()) < 0.00001) continue;
	if(reqOppCharge_ && charge[jj]*charge[ii] > 0) continue;
	
	TLorentzVector pTot = p1Ele + p2Ele;
	double mass = pTot.M();
	
	if(mass>=lowerMassCut_ && mass<=upperMassCut_){
	  n++;
	  refele = electrons[ii];
	  filterproduct.addObject(TriggerElectron, refele);
	  refele = electrons[jj];
	  filterproduct.addObject(TriggerElectron, refele);
	}
      }
    }

  } else {

    Ref< RecoEcalCandidateCollection > refsc;

    vector< Ref< RecoEcalCandidateCollection > > scs;
    PrevFilterOutput->getObjects(TriggerCluster, scs);
    if(scs.empty()) PrevFilterOutput->getObjects(TriggerPhoton, scs);  //we dont know if its type trigger cluster or trigger photon

    for (unsigned int i=0; i<scs.size(); i++) {

      refsc = scs[i];
      const reco::SuperClusterRef sc = refsc->superCluster();
      TLorentzVector pscPos = approxMomAtVtx(theMagField.product(), vertexPos, sc, 1);
      pEleCh1.push_back( pscPos );

      TLorentzVector pscEle = approxMomAtVtx(theMagField.product(), vertexPos, sc, -1);
      pEleCh2.push_back( pscEle );
      etaOrig.push_back( sc->eta() );

    }

    for(unsigned int jj=0;jj<scs.size();jj++){

      TLorentzVector p1Ele = pEleCh1.at(jj);
      for(unsigned int ii=0;ii<scs.size();ii++){
	
	TLorentzVector p2Ele = pEleCh2.at(ii);
	
	if(fabs(p1Ele.E() - p2Ele.E()) < 0.00001) continue;
	
	TLorentzVector pTot = p1Ele + p2Ele;
	double mass = pTot.M();
	
	if(mass>= lowerMassCut_ && mass<=upperMassCut_){
	  n++;
	  refsc = scs[ii];
	  filterproduct.addObject(TriggerCluster, refsc);
	  refsc = scs[jj];
	  filterproduct.addObject(TriggerCluster, refsc);
	}
      }
    }
  }


  // filter decision
  bool accept(n>=nZcandcut_);
  // if (accept) std::cout << "n size = " << n << std::endl;

  return accept;
}

TLorentzVector HLTPMMassFilter::approxMomAtVtx( const MagneticField *magField, const GlobalPoint& xvert, const reco::SuperClusterRef sc, int charge) const
{
    GlobalPoint xsc(sc->position().x(),
		    sc->position().y(),
		    sc->position().z());
    float energy = sc->energy();
    FreeTrajectoryState theFTS = FTSFromVertexToPointFactory::get(*magField, xsc, xvert, energy, charge);
    float theApproxMomMod = theFTS.momentum().x()*theFTS.momentum().x() +
                            theFTS.momentum().y()*theFTS.momentum().y() +
                            theFTS.momentum().z()*theFTS.momentum().z();
    TLorentzVector theApproxMom(theFTS.momentum().x(),
				theFTS.momentum().y(),
				theFTS.momentum().z(),
                                sqrt(theApproxMomMod + 2.61121E-7));
    return theApproxMom ;
}
