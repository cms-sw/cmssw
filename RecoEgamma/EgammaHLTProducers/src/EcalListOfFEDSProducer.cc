
#include <FWCore/Framework/interface/Handle.h>
#include <FWCore/Framework/interface/Event.h>
#include <FWCore/Framework/interface/EventSetup.h>
#include <FWCore/Framework/interface/ESHandle.h>

#include "RecoEgamma/EgammaHLTProducers/interface/EcalListOfFEDSProducer.h"
#include "DataFormats/EcalRawData/interface/EcalListOfFEDS.h"

#include "FWCore/Utilities/interface/Exception.h"

// Ecal Mapping 
#include "Geometry/EcalMapping/interface/EcalElectronicsMapping.h"
#include "Geometry/EcalMapping/interface/EcalMappingRcd.h"
#include <DataFormats/FEDRawData/interface/FEDNumbering.h>


// Level 1 Trigger
#include "DataFormats/L1Trigger/interface/L1EmParticle.h"
#include "DataFormats/L1Trigger/interface/L1EmParticleFwd.h"
#include "L1TriggerConfig/L1Geometry/interface/L1CaloGeometry.h"
#include "L1TriggerConfig/L1Geometry/interface/L1CaloGeometryRecord.h"
                                                                                                                        
// EgammaCoreTools
#include "RecoEcal/EgammaCoreTools/interface/PositionCalc.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalEtaPhiRegion.h"

// Muon stuff
// #include "RecoMuon/MuonIsolation/interface/Direction.h"
// #include "RecoMuon/L2MuonIsolationProducer/src/L2MuonIsolationProducer.h"

// #include "DataFormats/Common/interface/AssociationMap.h"
// #include "DataFormats/TrackReco/interface/Track.h"
// #include "DataFormats/TrackReco/interface/TrackFwd.h"

#include <vector>

// using namespace reco;
// using namespace muonisolation;


EcalListOfFEDSProducer::EcalListOfFEDSProducer(const edm::ParameterSet& pset) {

 debug_ = pset.getUntrackedParameter<bool>("debug");


 EGamma_ = pset.getUntrackedParameter<bool>("EGamma");
 Muon_ = pset.getUntrackedParameter<bool>("Muon");

 if (EGamma_ && Muon_) {
  throw cms::Exception("EcalListOfFEDSProducer") << 
	" Wrong configuration : EGamma and Muon should not be true at the same time." ;
 }


 if (EGamma_) {
 	
 	EMl1TagIsolated_ = pset.getUntrackedParameter<edm::InputTag>("EM_l1TagIsolated");
 	EMl1TagNonIsolated_ = pset.getUntrackedParameter<edm::InputTag>("EM_l1TagNonIsolated");
 	EMdoIsolated_ = pset.getUntrackedParameter<bool>("EM_doIsolated",true);
 	EMdoNonIsolated_ = pset.getUntrackedParameter<bool>("EM_doNonIsolated",true);
 	EMregionEtaMargin_ = pset.getUntrackedParameter<double>("EM_regionEtaMargin",0.25);
 	EMregionPhiMargin_ = pset.getUntrackedParameter<double>("EM_regionPhiMargin",0.40);
 	Ptmin_iso_ = pset.getUntrackedParameter<double>("Ptmin_iso",0.);
 	Ptmin_noniso_ = pset.getUntrackedParameter<double>("Ptmin_noniso",0.);
 }

 if (Muon_) {
 	MUregionEtaMargin_ = pset.getUntrackedParameter<double>("MU_regionEtaMargin_",1.0);
 	MUregionPhiMargin_ = pset.getUntrackedParameter<double>("MU_regionPhiMargin",1.0);
 	Ptmin_muon_ = pset.getUntrackedParameter<double>("Ptmin_muon",0.);
	theSACollectionLabel_ = pset.getUntrackedParameter<edm::InputTag>("StandAloneCollectionLabel");
 }

 OutputLabel_ = pset.getUntrackedParameter<std::string>("OutputLabel");

 TheMapping = new EcalElectronicsMapping();
 first_ = true;

 produces<EcalListOfFEDS>(OutputLabel_);
}



EcalListOfFEDSProducer::~EcalListOfFEDSProducer() {
 delete TheMapping;
}


void EcalListOfFEDSProducer::beginJob(const edm::EventSetup& c){
}

void EcalListOfFEDSProducer::endJob(){
}

void EcalListOfFEDSProducer::produce(edm::Event & e, const edm::EventSetup& iSetup){

 std::pair<int,int> ecalfeds = FEDNumbering::getEcalFEDIds();
 int first_fed = ecalfeds.first;

 if (first_) {
   edm::ESHandle< EcalElectronicsMapping > ecalmapping;
   iSetup.get< EcalMappingRcd >().get(ecalmapping);
   const EcalElectronicsMapping* TheMapping_ = ecalmapping.product();
   *TheMapping = *TheMapping_;
   first_ = false;
 }                                                                                              

 std::auto_ptr<EcalListOfFEDS> productAddress(new EcalListOfFEDS);

 std::vector<int> feds;

 if (EGamma_) {
  feds = Egamma(e, iSetup);
 }

/*
 else if (Muon_) {
   feds = Muon(e, iSetup);
 }
*/

 else {
   for (int i=1; i <= 54; i++) {
      feds.push_back(i);
   }
 }

 int nf = (int)feds.size();
 for (int i=0; i <nf; i++) {
  feds[i] += first_fed;
  // std::cout << "Will unpack FED " << feds[i] << std::endl;
 }

 productAddress.get() -> SetList(feds);
 e.put(productAddress,OutputLabel_);
}


std::vector<int> EcalListOfFEDSProducer::Egamma(edm::Event& e, const edm::EventSetup& es) {

 std::vector<int> FEDs;

 if (debug_) std::cout << std::endl << std::endl << " enter in EcalListOfFEDSProducer::Egamma" << std::endl;

  //Get the L1 EM Particle Collection
  //Get the L1 EM Particle Collection
  edm::Handle< l1extra::L1EmParticleCollection > emIsolColl ;
  if(EMdoIsolated_)
    e.getByLabel(EMl1TagIsolated_, emIsolColl);
  //Get the L1 EM Particle Collection
  edm::Handle< l1extra::L1EmParticleCollection > emNonIsolColl ;
  if (EMdoNonIsolated_)
    e.getByLabel(EMl1TagNonIsolated_, emNonIsolColl);

  // Get the CaloGeometry
  edm::ESHandle<L1CaloGeometry> l1CaloGeom ;
  es.get<L1CaloGeometryRecord>().get(l1CaloGeom) ;

  if(EMdoIsolated_) {

    for( l1extra::L1EmParticleCollection::const_iterator emItr = emIsolColl->begin();
                        emItr != emIsolColl->end() ;++emItr ){

	float pt = emItr -> pt();
	if (pt < Ptmin_iso_ ) continue;
	if (debug_) std::cout << " Here is an L1 isoEM candidate of pt " << pt << std::endl;
        // Access the GCT hardware object corresponding to the L1Extra EM object.
        int etaIndex = emItr->gctEmCand()->etaIndex() ;
        int phiIndex = emItr->gctEmCand()->phiIndex() ;
        // Use the L1CaloGeometry to find the eta, phi bin boundaries.
        double etaLow  = l1CaloGeom->etaBinLowEdge( etaIndex ) ;
        double etaHigh = l1CaloGeom->etaBinHighEdge( etaIndex ) ;
        double phiLow  = l1CaloGeom->emJetPhiBinLowEdge( phiIndex ) ;
        double phiHigh = l1CaloGeom->emJetPhiBinHighEdge( phiIndex ) ;

	std::vector<int> feds = ListOfFEDS(etaLow, etaHigh, phiLow, phiHigh, EMregionEtaMargin_, EMregionPhiMargin_);
	for (int i=0; i < (int)feds.size(); i++) {
		if (std::find(FEDs.begin(), FEDs.end(), feds[i]) == FEDs.end()) FEDs.push_back(feds[i]);
	}

    } // end loop on L1EmParticleCollection

  }  // endif doIsolated_


  if (EMdoNonIsolated_) {

    for( l1extra::L1EmParticleCollection::const_iterator emItr = emNonIsolColl->begin();
                        emItr != emNonIsolColl->end() ;++emItr ){

	float pt = emItr -> pt();
	if (debug_) std::cout << " Here is an L1 nonisoEM candidate of pt " << pt << std::endl;
	if (pt < Ptmin_noniso_ ) continue;
        // Access the GCT hardware object corresponding to the L1Extra EM object.
        int etaIndex = emItr->gctEmCand()->etaIndex() ;
        int phiIndex = emItr->gctEmCand()->phiIndex() ;
        // std::cout << " etaIndex phiIndex " << etaIndex << " " << phiIndex << std::endl;
        // Use the L1CaloGeometry to find the eta, phi bin boundaries.
        double etaLow  = l1CaloGeom->etaBinLowEdge( etaIndex ) ;
        double etaHigh = l1CaloGeom->etaBinHighEdge( etaIndex ) ;
        double phiLow  = l1CaloGeom->emJetPhiBinLowEdge( phiIndex ) ;
        double phiHigh = l1CaloGeom->emJetPhiBinHighEdge( phiIndex ) ;

	std::vector<int> feds = ListOfFEDS(etaLow, etaHigh, phiLow, phiHigh, EMregionEtaMargin_, EMregionPhiMargin_);
        for (int i=0; i < (int)feds.size(); i++) {
                if (std::find(FEDs.begin(), FEDs.end(), feds[i]) == FEDs.end()) FEDs.push_back(feds[i]);
        }

    } // end loop on L1EmParticleCollection
  }

 // std::cout << "end of get list of feds " << std::endl;

 if (debug_) {
	std::cout << std::endl;
	for (int i=0; i < (int)FEDs.size(); i++) {
	  std::cout << "unpack FED " << FEDs[i] << std::endl;
	}
	std::cout << "Number of FEDS is " << FEDs.size() << std::endl;
 }

 return FEDs;

}


/*
std::vector<int> EcalListOfFEDSProducer::Muon(edm::Event& e, const edm::EventSetup& es) {

 std::vector<int> FEDs;

 if (debug_) std::cout << std::endl << std::endl << " enter in EcalListOfFEDSProducer::Muon" << std::endl;

 edm::Handle<TrackCollection> tracks;
 e.getByLabel(theSACollectionLabel_,tracks);

 // theCalExtractor.fillVetos(e,es,*tracks);

  double epsilon = 0.01;

  for (unsigned int i=0; i<tracks->size(); i++) {
      	TrackRef tk(tracks,i);

      // MuIsoDeposit calDeposit = theCalExtractor.deposit(event, eventSetup, *tk);
	double eta_track = tk -> eta();
	double phi_track = tk -> phi();
	double pt = tk -> pt();
	if (debug_) std::cout << " here is a muon Seed  with (eta,phi) = " << eta_track << " " << phi_track << " and pt " << pt << std::endl;
	if (pt < Ptmin_muon_ ) continue;

	std::vector<int> feds = ListOfFEDS(eta_track, eta_track, phi_track-epsilon, phi_track+epsilon, MUregionEtaMargin_, MUregionPhiMargin_);
        for (int i=0; i < (int)feds.size(); i++) {
                if (std::find(FEDs.begin(), FEDs.end(), feds[i]) == FEDs.end()) FEDs.push_back(feds[i]);
        }
  }

 if (debug_) {
        std::cout << std::endl;
        for (int i=0; i < (int)FEDs.size(); i++) {
          std::cout << "unpack FED " << FEDs[i] << std::endl;
        }
        std::cout << "Number of FEDS is " << FEDs.size() << std::endl;
 }

 return FEDs;

}
*/



std::vector<int> EcalListOfFEDSProducer::ListOfFEDS(double etaLow, double etaHigh, double phiLow, 
					 double phiHigh, double etamargin, double phimargin)
{

	std::vector<int> FEDs;

	if (phimargin > Geom::pi()) phimargin =  Geom::pi() ;


        if (debug_) std::cout << " etaLow etaHigh phiLow phiHigh " << etaLow << " " << 
			etaHigh << " " << phiLow << " " << phiHigh << std::endl;

        etaLow -= etamargin;
        etaHigh += etamargin;
        double phiMinus = phiLow - phimargin;
        double phiPlus = phiHigh + phimargin;

        bool all = false;
        double dd = fabs(phiPlus-phiMinus);
	if (debug_) std::cout << " dd = " << dd << std::endl;
        if (dd > 2.*Geom::pi() ) all = true;

        while (phiPlus > Geom::pi()) { phiPlus -= 2.*Geom::pi() ; }
        while (phiMinus < 0) { phiMinus += 2.*Geom::pi() ; }
        if ( phiMinus > Geom::pi()) phiMinus -= 2.*Geom::pi() ;

        double dphi = phiPlus - phiMinus;
        if (dphi < 0) dphi += 2.*Geom::pi() ;
	if (debug_) std::cout << "dphi = " << dphi << std::endl;
        if (dphi > Geom::pi()) {
                int fed_low1 = TheMapping -> GetFED(etaLow,phiMinus*180./Geom::pi());
                int fed_low2 = TheMapping -> GetFED(etaLow,phiPlus*180./Geom::pi());
		if (debug_) std::cout << "fed_low1 fed_low2 " << fed_low1 << " " << fed_low2 << std::endl;
                if (fed_low1 == fed_low2) all = true;
                int fed_hi1 = TheMapping -> GetFED(etaHigh,phiMinus*180./Geom::pi());
                int fed_hi2 = TheMapping -> GetFED(etaHigh,phiPlus*180./Geom::pi());
		if (debug_) std::cout << "fed_hi1 fed_hi2 " << fed_hi1 << " " << fed_hi2 << std::endl;
                if (fed_hi1 == fed_hi2) all = true;
        }


	if (all) {
		if (debug_) std::cout << " unpack everything in phi ! " << std::endl;
		phiMinus = -20 * Geom::pi() / 180.;  // -20 deg
		phiPlus = -40 * Geom::pi() / 180.;  // -20 deg
	}

        if (debug_) std::cout << " with margins : " << etaLow << " " << etaHigh << " " << 
			phiMinus << " " << phiPlus << std::endl;


        const EcalEtaPhiRegion ecalregion(etaLow,etaHigh,phiMinus,phiPlus);

        FEDs = TheMapping -> GetListofFEDs(ecalregion);

	if (debug_) {
           int nn = (int)FEDs.size();
           for (int ii=0; ii < nn; ii++) {
                   std::cout << "unpack fed " << FEDs[ii] << std::endl;
           }
   	   }

        return FEDs;

}



/*
bool EcalListOfFEDSProducer::PhiInbetween(double phiPlus,double phiMinus,double phiLow) {

 bool all = false;
 // first, everything back between 0 and 2pi

 if (phiPlus < 0) phiPlus += 2.*Geom::pi();
 if (phiMinus < 0) phiMinus += 2.*Geom::pi();
 if (phiLow < 0) phiLow += 2.*Geom::pi();

 if (phiMinus <= phiPlus && phiPlus <= phiLow) all = true;
 return all;

}
*/







