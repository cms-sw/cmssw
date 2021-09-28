// -*- C++ -*-
//
// Package:    Electron_GNN_Regression/DRNTestNTuplizer
// Class:      DRNTestNTuplizer
//
/**\class DRNTestNTuplizer DRNTestNTuplizer.cc Electron_GNN_Regression/DRNTestNTuplizer/plugins/DRNTestNTuplizer.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Rajdeep Mohan Chatterjee
//         Created:  Fri, 21 Feb 2020 11:38:58 GMT
//
//


// system include files
#include <memory>
#include <iostream>
#include "TTree.h"
#include "Math/VectorUtil.h"
#include "TFile.h"
// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "SimDataFormats/PileupSummaryInfo/interface/PileupSummaryInfo.h"


#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/TruncatedPyramid.h"
#include "Geometry/EcalAlgo/interface/EcalPreshowerGeometry.h"
#include "Geometry/CaloTopology/interface/EcalPreshowerTopology.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
//#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"

#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/PatCandidates/interface/Photon.h"

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/PatCandidates/interface/VIDCutFlowResult.h"

#include "DataFormats/EgammaReco/interface/SuperCluster.h"

#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonSelectors.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

#include "RecoEcal/EgammaCoreTools/interface/EcalClusterTools.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterLazyTools.h"
#include "RecoEcal/EgammaCoreTools/interface/PositionCalc.h"

#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDetId/interface/ESDetId.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"



#define NELE 3
#define initSingleFloat     -999.
#define initSingleInt          0
#define initSingleIntCharge -100
#define initFloat     { initSingleFloat, initSingleFloat, initSingleFloat }
#define initInt       { initSingleInt, initSingleInt, initSingleInt }
#define initIntCharge { initSingleIntCharge, initSingleIntCharge, initSingleIntCharge }
#define PDGID 11

using reco::TrackCollection;

class DRNTestNTuplizer : public edm::one::EDAnalyzer<edm::one::SharedResources>  {
   public:
      explicit DRNTestNTuplizer(const edm::ParameterSet&);
      ~DRNTestNTuplizer();

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);


   private:
      virtual void beginJob() override;
      virtual void analyze(const edm::Event&, const edm::EventSetup&) override;
      virtual void endJob() override;



// Set event specific information
    void TreeSetEventSummaryVar(const edm::Event& iEvent);
    void TreeSetPileupVar(void);

//   clear the vectors 
     void InitNewTree(void);
     void ResetMainTreeVar();

// Helper functions to fill the trees
     void TreeSetDiElectronVar(const reco::GsfElectron& electron1, const reco::GsfElectron& electron2);
     void TreeSetSingleElectronVar(const reco::GsfElectron& electron1, int index);
     float GetMustEnergy(const reco::GsfElectron& electron, bool isEB);
// ----------member data ---------------------------
     TTree * _tree;                   //< output file for standard ntuple
     edm::Timestamp _eventTimeStamp;
     // Variables for Run info.
     UInt_t    _runNumber;     ///< run number
     UShort_t  _lumiBlock;     ///< lumi section
     Long64_t  _eventNumber;   ///< event number
     UInt_t    _eventTime;     ///< unix time of the event
     UShort_t  _nBX;           ///< bunch crossing
     Bool_t    _isTrain;


     // pileup
     Float_t  _rho;    ///< _rho fast jet
     UChar_t  _nPV;    ///< nVtx
     UChar_t  _nPU;    ///< number of PU (filled only for MC)


     // electron variables
      UInt_t _eleID[NELE] = initInt;      ///< bit mask for _eleID: 1=fiducial, 2=loose, 6=medium, 14=tight, 16=WP90PU, 48=WP80PU, 112=WP70PU, 128=loose25nsRun2, 384=medium25nsRun2, 896=tight25nsRun2, 1024=loose50nsRun2, 3072=medium50nsRun2, 7168=tight50nsRun2. Selection from https://twiki.cern.ch/twiki/bin/viewauth/CMS/EgammaCutBasedIdentification#Electron_ID_Working_Points
     Short_t  _chargeEle[NELE]    = initIntCharge; ///< -100: no electron, 0: SC or photon, -1 or +1:electron or muon //Char_t is interpreted as char and not as integer
     UChar_t  _recoFlagsEle[NELE] = initInt;       ///< 1=trackerDriven, 2=ecalDriven only, 3=tracker and ecal driven
     Float_t  _etaEle[NELE]       = initFloat;
     Float_t  _phiEle[NELE]       = initFloat;     ///< phi of the electron (electron object)
     Float_t  _R9Ele[NELE]        = initFloat;     ///< e3x3/_rawEnergySCEle


     // SC variables
     Float_t _etaSCEle[NELE]    = initFloat;
     Float_t _phiSCEle[NELE]    = initFloat; ///< phi of the SC
     Float_t _energy_ECAL_ele[NELE]            = initFloat;  ///< ele-tuned regression energy: mustache for rereco and correctedEcalEnergy for official reco	
     Float_t _mustEnergySCEle[NELE]    = initFloat;
     Float_t _rawEnergySCEle[NELE]             = initFloat;  ///< SC energy without cluster corrections
     Short_t _xSeedSC[NELE]                    = initInt;    ///< ieta(ix) of the SC seed in EB(EE)
     Short_t _ySeedSC[NELE]                    = initInt;    ///< iphi(iy) of the SC seed in EB(EE)

     // Gen part 4 vectors
     std::vector<float> Gen_Pt;
     std::vector<float> Gen_Eta;
     std::vector<float> Gen_Phi;
     std::vector<float> Gen_E;



      // -----------------Handles--------------------------
      edm::Handle<reco::VertexCollection>            _primaryVertexHandle;  	
      edm::Handle<double> rhoHandle;
      edm::Handle<std::vector< PileupSummaryInfo > > _PupInfo;
      edm::Handle<EcalRecHitCollection> EBRechitsHandle;
      edm::Handle<EcalRecHitCollection> EERechitsHandle;
      edm::Handle<EcalRecHitCollection> ESRechitsHandle;
      edm::Handle<std::vector<reco::SuperCluster>>   _EBSuperClustersHandle;
      edm::Handle<std::vector<reco::SuperCluster>>   _EESuperClustersHandle;
      edm::Handle<edm::View<reco::GsfElectron> > electrons; 
      edm::Handle<edm::View<reco::GenParticle> > genParticles;
      edm::Handle<edm::ValueMap<bool> > loose_id_decisions;
      edm::Handle<edm::ValueMap<bool> > medium_id_decisions;
      edm::Handle<edm::ValueMap<bool> > tight_id_decisions;
      //---------------- Input Tags-----------------------
      edm::EDGetTokenT<reco::VertexCollection>           _vtxCollectionToken;
      edm::EDGetTokenT<double> rhoToken_;
      edm::EDGetTokenT<std::vector<PileupSummaryInfo> > _pileupInfoToken;	
      edm::EDGetTokenT<EcalRecHitCollection> recHitCollectionEBToken_;
      edm::EDGetTokenT<EcalRecHitCollection> recHitCollectionEEToken_;
      edm::EDGetTokenT<EcalRecHitCollection> recHitCollectionESToken_;
      edm::EDGetTokenT<std::vector<reco::SuperCluster> > _EBSuperClustersToken;
      edm::EDGetTokenT<std::vector<reco::SuperCluster> > _EESuperClustersToken; 
      edm::EDGetToken electronsToken_;
      edm::EDGetTokenT<edm::View<reco::GenParticle> > genParticlesToken_;
      edm::EDGetTokenT<edm::ValueMap<bool> > eleLooseIdMapToken_;
      edm::EDGetTokenT<edm::ValueMap<bool> > eleMediumIdMapToken_;
      edm::EDGetTokenT<edm::ValueMap<bool> > eleTightIdMapToken_;






};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
DRNTestNTuplizer::DRNTestNTuplizer(const edm::ParameterSet& iConfig):
   _vtxCollectionToken(consumes<reco::VertexCollection>(iConfig.getParameter<edm::InputTag>("vertexCollection"))),
   rhoToken_(consumes<double>(iConfig.getParameter<edm::InputTag>("rhoFastJet"))),
   _pileupInfoToken(consumes<std::vector<PileupSummaryInfo>>(iConfig.getParameter<edm::InputTag>("pileupInfo"))),
   recHitCollectionEBToken_(consumes<EcalRecHitCollection>(edm::InputTag("reducedEcalRecHitsEB"))),
   recHitCollectionEEToken_(consumes<EcalRecHitCollection>(edm::InputTag("reducedEcalRecHitsEE"))),
   recHitCollectionESToken_(consumes<EcalRecHitCollection>(edm::InputTag("reducedEcalRecHitsES"))),
   _EBSuperClustersToken(consumes<reco::SuperClusterCollection>(edm::InputTag("DRNProducerEB"))),
   _EESuperClustersToken(consumes<reco::SuperClusterCollection>(edm::InputTag("DRNProducerEE"))),
   eleLooseIdMapToken_(consumes<edm::ValueMap<bool> >(iConfig.getParameter<edm::InputTag>("eleLooseIdMap"))),
   eleMediumIdMapToken_(consumes<edm::ValueMap<bool> >(iConfig.getParameter<edm::InputTag>("eleMediumIdMap"))),
   eleTightIdMapToken_(consumes<edm::ValueMap<bool> >(iConfig.getParameter<edm::InputTag>("eleTightIdMap")))
{
   //now do what ever initialization is needed
   electronsToken_ = mayConsume<edm::View<reco::GsfElectron> >(iConfig.getParameter<edm::InputTag>("electrons"));
   genParticlesToken_ = mayConsume<edm::View<reco::GenParticle> >(iConfig.getParameter<edm::InputTag>("genParticles"));
   usesResource("TFileService");
}


DRNTestNTuplizer::~DRNTestNTuplizer()
{

   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)




}


//
// member functions
//

// ------------ method called for each event  ------------
void
DRNTestNTuplizer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   using namespace std;
   using namespace reco;

   ResetMainTreeVar();

   _chargeEle[0] = initSingleIntCharge;
   _chargeEle[1] = initSingleIntCharge;
   _chargeEle[2] = initSingleIntCharge;


   iEvent.getByToken(_vtxCollectionToken, _primaryVertexHandle);
   iEvent.getByToken(rhoToken_, rhoHandle);
   iEvent.getByToken(_pileupInfoToken, _PupInfo);
   iEvent.getByToken(recHitCollectionEBToken_, EBRechitsHandle);
   iEvent.getByToken(recHitCollectionEEToken_, EERechitsHandle);
   iEvent.getByToken(recHitCollectionESToken_, ESRechitsHandle);

   iEvent.getByToken(_EBSuperClustersToken, _EBSuperClustersHandle);
   iEvent.getByToken(_EESuperClustersToken, _EESuperClustersHandle);

   iEvent.getByToken(electronsToken_, electrons);
   iEvent.getByToken(genParticlesToken_, genParticles);
   iEvent.getByToken(eleLooseIdMapToken_, loose_id_decisions);
   iEvent.getByToken(eleMediumIdMapToken_, medium_id_decisions);
   iEvent.getByToken(eleTightIdMapToken_ , tight_id_decisions);


  TreeSetEventSummaryVar(iEvent);
  TreeSetPileupVar();


///////////////////////////Fill Electron/Photon related stuff/////////////////////////////////////////////////////

   bool doFill = false;
   for (size_t i = 0; i < electrons->size(); i++){

	const auto ele1 = electrons->ptrAt(i);
	if(!(ele1->ecalDrivenSeed())) continue;
        if(ele1->parentSuperCluster().isNull()) continue;
	if( ele1->pt() < 15. ) continue;
	if(!(*loose_id_decisions)[ele1]) continue; 
	for (size_t j = i+1; j < electrons->size() && doFill == false; j++){

		const auto ele2 = electrons->ptrAt(j);
		if(!(ele2->ecalDrivenSeed())) continue;
        	if(ele2->parentSuperCluster().isNull()) continue;
        	if( ele2->pt() < 15. ) continue;
		double t1 = TMath::Exp(-ele1->eta());
                double t1q = t1 * t1;
                double t2 = TMath::Exp(-ele2->eta());
                double t2q = t2 * t2;
                double angle = 1 - ( (1 - t1q) * (1 - t2q) + 4 * t1 * t2 * cos(ele1->phi() - ele2->phi())) / ( (1 + t1q) * (1 + t2q) );
                double mass = sqrt(2 * ele1->energy() * ele2->energy() * angle);
		if(mass < 55 ) continue;
                doFill = true;
                TreeSetDiElectronVar(*ele1, *ele2);
	}
   }


//////////////////////// Gen Stuff hardcoded for status 1 photons /////////////////////////////////////

    for(edm::View<GenParticle>::const_iterator part = genParticles->begin(); part != genParticles->end(); ++part){
        if( part->status()==1 && abs(part->pdgId())==PDGID ){
                Gen_Pt.push_back(part->pt());
                Gen_Eta.push_back(part->eta());
                Gen_Phi.push_back(part->phi());
                Gen_E.push_back(part->energy());
        }
   }
 
////////////////////////////////////////////////////////////////////////////////////////////////////////

   if(doFill) _tree->Fill(); // Write out the events

#ifdef THIS_IS_AN_EVENTSETUP_EXAMPLE
   ESHandle<SetupData> pSetup;
   iSetup.get<SetupRecord>().get(pSetup);
#endif
}


// ------------ method called once each job just before starting event loop  ------------
void
DRNTestNTuplizer::beginJob()
{
        edm::Service<TFileService> fs;
        _tree=fs->make<TTree>("selected", "selected");
	InitNewTree();
}

// ------------ method called once each job just after ending the event loop  ------------
void
DRNTestNTuplizer::endJob()
{
}




//Clear tree vectors each time analyze method is called

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
DRNTestNTuplizer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);

  //Specify that only 'tracks' is allowed
  //To use, remove the default given above and uncomment below
  //ParameterSetDescription desc;
  //desc.addUntracked<edm::InputTag>("tracks","ctfWithMaterialTracks");
  //descriptions.addDefault(desc);
}

void DRNTestNTuplizer::TreeSetDiElectronVar(const reco::GsfElectron& electron1, const reco::GsfElectron& electron2){
	
	TreeSetSingleElectronVar(electron1, 0);
        TreeSetSingleElectronVar(electron2, 1);

}

void DRNTestNTuplizer::TreeSetSingleElectronVar(const reco::GsfElectron& electron, int index){

	if(index < 0) {
                _chargeEle[-index] = -100;
                _etaEle[-index]    = 0;
                _phiEle[-index]    = 0;
                _recoFlagsEle[-index] = -1;
                return;
        }

	assert(electron.ecalDrivenSeed());

	_chargeEle[index] = (Char_t)electron.charge();
	_etaEle[index]    = electron.eta();
	_phiEle[index]    = electron.phi();
	
	if(electron.ecalDrivenSeed()) {
                if(electron.trackerDrivenSeed()) _recoFlagsEle[index] = 3;
                else _recoFlagsEle[index] = 2;
        } else _recoFlagsEle[index] = 1;

	_R9Ele[index] = electron.full5x5_r9();
	

	const reco::SuperClusterRef& sc = electron.parentSuperCluster();
	DetId seedDetId = sc->seed()->seed();
        assert(!seedDetId.null());

	 _etaSCEle[index]    = sc->eta();
         _phiSCEle[index]    = sc->phi();
	 _rawEnergySCEle[index]     = sc->rawEnergy();
	 _energy_ECAL_ele[index] = electron.correctedEcalEnergy();

        if(seedDetId.subdetId() == EcalBarrel) {
                EBDetId seedDetIdEcal(seedDetId);
                _xSeedSC[index] = seedDetIdEcal.ieta();
                _ySeedSC[index] = seedDetIdEcal.iphi();
	} else {
		EEDetId seedDetIdEcal(seedDetId);
                _xSeedSC[index] = seedDetIdEcal.ix();
                _ySeedSC[index] = seedDetIdEcal.iy();
	}

	if( (*sc->seed()).hitsAndFractions().at(0).first.subdetId() == EcalBarrel)
                        _mustEnergySCEle[index] = GetMustEnergy(electron, 1);
        else if(  (*sc->seed()).hitsAndFractions().at(0).first.subdetId() == EcalEndcap)
                        _mustEnergySCEle[index] = GetMustEnergy(electron, 0);

}



void DRNTestNTuplizer::InitNewTree()
{
	std::cout << "[STATUS] InitNewTree" << std::endl;
	if(_tree == NULL) return;
        _tree->Branch("runNumber",     &_runNumber,   "runNumber/i");
        _tree->Branch("lumiBlock",     &_lumiBlock,   "lumiBlock/s");
        _tree->Branch("eventNumber",   &_eventNumber, "eventNumber/l");
        _tree->Branch("eventTime",       &_eventTime,     "eventTime/i");
        _tree->Branch("nBX",           &_nBX,         "nBX/s");
        _tree->Branch("isTrain",        &_isTrain,      "isTrain/B"); 

	_tree->Branch("rho", &_rho, "rho/F");
        _tree->Branch("nPV", &_nPV, "nPV/b");
        _tree->Branch("nPU", &_nPU, "nPU/b");


        _tree->Branch("eleID", _eleID, "eleID[3]/i");
        _tree->Branch("chargeEle",   _chargeEle,    "chargeEle[3]/S");
        _tree->Branch("recoFlagsEle", _recoFlagsEle, "recoFlagsEle[3]/b");
        _tree->Branch("etaEle",      _etaEle,       "etaEle[3]/F");
        _tree->Branch("phiEle",      _phiEle,       "phiEle[3]/F");
        _tree->Branch("R9Ele", _R9Ele, "R9Ele[3]/F");

        _tree->Branch("etaSCEle",      _etaSCEle,       "etaSCEle[3]/F");
        _tree->Branch("phiSCEle",      _phiSCEle,       "phiSCEle[3]/F");
        _tree->Branch("rawEnergySCEle", _rawEnergySCEle, "rawEnergySCEle[3]/F");
        _tree->Branch("mustEnergySCEle", _mustEnergySCEle, "mustEnergySCEle[3]/F");
        _tree->Branch("energy_ECAL_ele", _energy_ECAL_ele, "energy_ECAL_ele[3]/F"); ///< correctedEcalEnergy from MINIAOD or mustache regression if rereco
	_tree->Branch("xSeedSC",            _xSeedSC,            "xSeedSC[3]/S");
        _tree->Branch("ySeedSC",            _ySeedSC,            "ySeedSC[3]/S");
        _tree->Branch("Gen_Pt" , &Gen_Pt);
        _tree->Branch("Gen_Eta" , &Gen_Eta);
        _tree->Branch("Gen_Phi" , &Gen_Phi);
        _tree->Branch("Gen_E" , &Gen_E);


}


void DRNTestNTuplizer::ResetMainTreeVar()
{
	for (int i = 0; i < NELE; ++i) {
		_eleID[i] = initSingleInt;
		_chargeEle[i] = initSingleIntCharge;
		_recoFlagsEle[i] = 0;
                _etaEle[i] = initSingleFloat;
                _phiEle[i] = initSingleFloat;
                _R9Ele[i] = initSingleFloat;

                _etaSCEle[i] = initSingleFloat;
                _phiSCEle[i] = initSingleFloat;
                _xSeedSC[i] = -999;
                _ySeedSC[i] = -999;
                _rawEnergySCEle[i] = initSingleFloat;
                _mustEnergySCEle[i] = initSingleFloat;
		_energy_ECAL_ele[i] = initSingleFloat;

	}

       Gen_Pt.clear();
       Gen_Eta.clear();
       Gen_Phi.clear();
       Gen_E.clear();

}

void DRNTestNTuplizer::TreeSetEventSummaryVar(const edm::Event& iEvent)
{
	_eventTimeStamp   =  iEvent.eventAuxiliary().time();
        _eventTime = (UInt_t) _eventTimeStamp.unixTime();
        _runNumber = (UInt_t) iEvent.run();
        _eventNumber = (Long64_t) iEvent.id().event();
        if( (_eventNumber % 10) == 0)
                _isTrain = 0;
        else
                _isTrain = 1;
        _nBX = (UShort_t)  iEvent.bunchCrossing();
        _lumiBlock = (UShort_t) iEvent.luminosityBlock();

}


float DRNTestNTuplizer::GetMustEnergy(const reco::GsfElectron& electron, bool isEB){


	if(isEB){
        	 for( reco::SuperClusterCollection::const_iterator iter = _EBSuperClustersHandle->begin();
                                iter != _EBSuperClustersHandle->end();
                                iter++) {
			if( fabs(electron.parentSuperCluster()->rawEnergy() - iter->rawEnergy()) < 1E-6 )
				return iter->energy();

        	}
	} else{

              	for( reco::SuperClusterCollection::const_iterator iter1 = _EESuperClustersHandle->begin();
                                iter1 != _EESuperClustersHandle->end();
                                iter1++) {
			if( fabs(electron.parentSuperCluster()->rawEnergy() - iter1->rawEnergy()) < 1E-6 )
                                return iter1->energy();
        	}
	}

	return -999.;

}

void DRNTestNTuplizer::TreeSetPileupVar(void)
{
        _rho = *rhoHandle;
        _nPV = 255;
        _nPU = 255;

        if(_primaryVertexHandle->size() > 0) {
                for(reco::VertexCollection::const_iterator v = _primaryVertexHandle->begin(); v != _primaryVertexHandle->end(); ++v) {
			_nPV++;
		}
	}

	
	std::vector<PileupSummaryInfo>::const_iterator PVI;
        for(PVI = _PupInfo->begin(); PVI != _PupInfo->end(); ++PVI) {
                int BX = PVI->getBunchCrossing();
                if(BX == 0) { // in-time pu
                        _nPU = PVI->getTrueNumInteractions();
                }
        }


        return;
	
}

//define this as a plug-in
DEFINE_FWK_MODULE(DRNTestNTuplizer);
