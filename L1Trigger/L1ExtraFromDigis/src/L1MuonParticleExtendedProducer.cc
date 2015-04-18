// -*- C++ -*-
//
// Class:      L1MuonParticleExtendedProducer
// 
//


// system include files
#include <memory>

// user include files
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"

#include "CondFormats/L1TObjects/interface/L1MuTriggerScales.h"
#include "CondFormats/DataRecord/interface/L1MuTriggerScalesRcd.h"
#include "CondFormats/L1TObjects/interface/L1MuTriggerPtScale.h"
#include "CondFormats/DataRecord/interface/L1MuTriggerPtScaleRcd.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTReadoutCollection.h"
#include "DataFormats/L1CSCTrackFinder/interface/L1CSCTrackCollection.h"

#include "DataFormats/L1Trigger/interface/L1MuonParticleExtended.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticleExtendedFwd.h"

#include "L1Trigger/CSCCommonTrigger/interface/CSCConstants.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/Common/interface/OrphanHandle.h"

#include "DataFormats/Math/interface/deltaPhi.h"

//
// class declaration
//
class L1MuonParticleExtendedProducer : public edm::EDProducer {
public:
  explicit L1MuonParticleExtendedProducer(const edm::ParameterSet&);
  ~L1MuonParticleExtendedProducer() {}

private:
  virtual void produce(edm::Event&, const edm::EventSetup&);

  //some helper functions
  float getSigmaEta(float eta, float etaP, float etaM) const;
  float getSigmaPhi(float phi, float phiP, float phiM) const;

  edm::InputTag gmtROSource_ ;
  edm::InputTag csctfSource_ ;

  bool writeAllCSCTFs_;

  static const double muonMassGeV_ ;

};


//
// static data member definitions
//

const double L1MuonParticleExtendedProducer::muonMassGeV_ = 0.105658369 ; // PDG06

//
// constructors and destructor
//
L1MuonParticleExtendedProducer::L1MuonParticleExtendedProducer(const edm::ParameterSet& iConfig)
   :   gmtROSource_( iConfig.getParameter< edm::InputTag >("gmtROSource" ) ),
       csctfSource_(iConfig.getParameter< edm::InputTag >("csctfSource")), 
       writeAllCSCTFs_(iConfig.getParameter< bool>("writeAllCSCTFs"))
{
   using namespace l1extra ;

   //register your products
   produces< L1MuonParticleExtendedCollection >() ;
   if (writeAllCSCTFs_) produces< L1MuonParticleExtendedCollection >("csc") ;
}


// ------------ method called to produce the data  ------------
void
L1MuonParticleExtendedProducer::produce( edm::Event& iEvent,
					 const edm::EventSetup& iSetup)
{
  using namespace edm ;
  using namespace l1extra ;
  using namespace std ;
  using namespace reco ;
  
  auto_ptr< L1MuonParticleExtendedCollection > muColl( new L1MuonParticleExtendedCollection );
  auto_ptr< L1MuonParticleExtendedCollection > cscColl( new L1MuonParticleExtendedCollection );
  
  ESHandle< L1MuTriggerScales > muScales ;
  iSetup.get< L1MuTriggerScalesRcd >().get( muScales ) ;
  
  ESHandle< L1MuTriggerPtScale > muPtScale ;
  iSetup.get< L1MuTriggerPtScaleRcd >().get( muPtScale ) ;
  
  Handle< L1MuGMTReadoutCollection > gmtRO_h ;
  iEvent.getByLabel( gmtROSource_, gmtRO_h ) ;
  const L1MuGMTReadoutCollection& gmtROs(*gmtRO_h.product());

  Handle<L1CSCTrackCollection> csctftH;
  iEvent.getByLabel(csctfSource_, csctftH);
  const L1CSCTrackCollection& csctfts(*csctftH.product());

  ESHandle<CSCGeometry> cscGH;
  iSetup.get<MuonGeometryRecord>().get(cscGH);
  const CSCGeometry* csc_geometry = &*cscGH;

  
  int curMuCollIdx = -1; //index of the GMT-based
  for (auto gmtRO : gmtROs.getRecords()){
    vector<int> cscGmtIdxs(4,-1); //index into muColl for a given CSC cand
    vector<int> dtGmtIdxs(4,-1); //index into muColl for a given DT cand

    auto const gmtCands = gmtRO.getGMTCands();
    auto const dtCands  = gmtRO.getDTBXCands();
    auto const cscCands = gmtRO.getCSCCands();
    auto const rpcBrlCands = gmtRO.getBrlRPCCands();
    auto const rpcFwdCands = gmtRO.getFwdRPCCands();

    for (auto const gmtCand : gmtCands){
      if (gmtCand.empty()) continue;
      
      float pt = muPtScale->getPtScale()->getLowEdge( gmtCand.ptIndex() ) + 1.e-6 ;
      float eta = muScales->getGMTEtaScale()->getCenter( gmtCand.etaIndex() ) ;
      float phi = muScales->getPhiScale()->getLowEdge( gmtCand.phiIndex() ) ;
      

      float etaP = muScales->getGMTEtaScale()->getCenter( gmtCand.etaIndex() +1 ) ;
      float phiP = muScales->getPhiScale()->getLowEdge( gmtCand.phiIndex() +1 ) ;
      
      float etaM = muScales->getGMTEtaScale()->getCenter( gmtCand.etaIndex() -1) ;
      float phiM = muScales->getPhiScale()->getLowEdge( gmtCand.phiIndex() -1) ;
      
      math::PtEtaPhiMLorentzVector p4( pt, eta, phi, muonMassGeV_);
      L1MuonParticle l1muP(gmtCand.charge(), p4, gmtCand, gmtCand.bx());
      int cscGmtIdx = -1;
      if ((!gmtCand.isRPC()) && gmtCand.isFwd()) cscGmtIdx = gmtCand.getDTCSCIndex();
      int dtGmtIdx = -1;
      if ((!gmtCand.isRPC()) && (!gmtCand.isFwd())) dtGmtIdx = gmtCand.getDTCSCIndex();
      int rpcGmtIdx = -1;
      if (gmtCand.isRPC()) rpcGmtIdx = gmtCand.getRPCIndex();

      muColl->push_back(L1MuonParticleExtended(l1muP));
      curMuCollIdx++;
      if (cscGmtIdx>=0) cscGmtIdxs[cscGmtIdx] = curMuCollIdx;
      if (dtGmtIdx>=0) dtGmtIdxs[dtGmtIdx] = curMuCollIdx;
      L1MuonParticleExtended* aGmtCand = &muColl->back();

      float sigmaEta = getSigmaEta(eta, etaP, etaM);
      float sigmaPhi = getSigmaPhi(phi, phiP, phiM);

      aGmtCand->setSigmaEta(sigmaEta);
      aGmtCand->setSigmaPhi(sigmaPhi);
      aGmtCand->setQuality(gmtCand.quality());

      //set regional candidates known to this GMT candidate here
      if (dtGmtIdx>=0)  aGmtCand->setDtCand(dtCands.at(dtGmtIdx)); //use .at, don't really trust the size
      if (cscGmtIdx>=0) aGmtCand->setCscCand(cscCands.at(cscGmtIdx));
      if (rpcGmtIdx>=0){
	if (gmtCand.isFwd()) aGmtCand->setRpcCand(rpcFwdCands.at(rpcGmtIdx));
	else aGmtCand->setRpcCand(rpcBrlCands.at(rpcGmtIdx));
      }
    }
    // I fill station level data derived from different kind of regional candidates below
    // this data is filled for both the GMT candidate and the regional candidates before GMT sorting
    // to avoid copy-paste, gmt-driven cands (started in the loop above) are modified below
    // only CSCTF details are filled in full

    // The intent is to create one L1MuExtended for each CSCTF track available
    if (1< 2){ //at the moment this is not really "ALL", just up to 4 from CSCTF showing up in the GMTReadoutRecord
      unsigned cscInd = -1;
      for (auto const csctfReg : gmtRO.getCSCCands() ){
	cscInd++;
	if (csctfReg.empty()) continue;
	
	float pt = muPtScale->getPtScale()->getLowEdge( csctfReg.pt_packed() ) + 1.e-6 ;
	float eta = muScales->getRegionalEtaScale(csctfReg.type_idx())->getCenter(csctfReg.eta_packed());
	float phi = muScales->getPhiScale()->getLowEdge( csctfReg.phi_packed() ) ;

	float etaP = muScales->getRegionalEtaScale(csctfReg.type_idx())->getCenter(csctfReg.eta_packed() + 1);
	float phiP = muScales->getPhiScale()->getLowEdge( csctfReg.phi_packed() +1 ) ;
	float etaM = muScales->getRegionalEtaScale(csctfReg.type_idx())->getCenter(csctfReg.eta_packed() -1);
	float phiM = muScales->getPhiScale()->getLowEdge( csctfReg.phi_packed() -1 ) ;

	int gmtIdx = cscGmtIdxs[cscInd];
	L1MuonParticleExtended* gmtParticle = 0;
	if (gmtIdx >=0) gmtParticle = &(*muColl)[gmtIdx];
	L1MuGMTExtendedCand gmtCand = gmtParticle != 0 ? gmtParticle->gmtMuonCand() : L1MuGMTExtendedCand();

	math::PtEtaPhiMLorentzVector p4( pt, eta, phi, muonMassGeV_);     
	int qCand = csctfReg.chargeValid() ? csctfReg.chargeValue() : 0;
	L1MuonParticle l1muP(qCand, p4, gmtCand, csctfReg.bx());

	L1MuonParticleExtended cscParticleTmp(l1muP);
	cscParticleTmp.setCscCand(csctfReg);
	cscColl->push_back(cscParticleTmp);
	L1MuonParticleExtended* cscParticle = &cscColl->back();

	float sigmaEta = getSigmaEta(eta, etaP, etaM);
	float sigmaPhi = getSigmaPhi(phi, phiP, phiM);

	cscParticle->setSigmaEta(sigmaEta);
	cscParticle->setSigmaPhi(sigmaPhi);

	unsigned int qual = 0;
	if (gmtParticle) qual = gmtParticle->quality();
	else {
	  unsigned int qualC = csctfReg.quality();
	  if (qualC ==2) qual = 3;
	  if (qualC ==3) qual = 6;
	}
	cscParticle->setQuality(qual);
	
	//now get the details, not the most optimal way .. not a big deal here
	unsigned int roWord = csctfReg.getDataWord();
	for (auto csctfFull : csctfts){
	  auto cscL1Tk = csctfFull.first;
	  unsigned int fWord = cscL1Tk.getDataWord();
	  //there's probably a more obvious way to match (just by local phi,sector,eta,etc)
	  if (cscL1Tk.endcap() == 2) fWord |= (1<< (L1MuRegionalCand::ETA_START+L1MuRegionalCand::ETA_LENGTH-1));
	  unsigned int fQuality = 0;
	  unsigned int fPt = 0;
	  cscL1Tk.decodeRank(cscL1Tk.rank(), fPt, fQuality);
	  fWord |= (fQuality<< L1MuRegionalCand::QUAL_START);
	  fWord |= (fPt << L1MuRegionalCand::PT_START);
	  //from L1Trigger/CSCTrackFinder/src/CSCTFMuonSorter.cc                                                                                                                                                                             
	  unsigned int fgPhi = cscL1Tk.localPhi() + (cscL1Tk.sector() -1)*24 + 6;
	  if(fgPhi > 143) fgPhi -= 143;
	  fWord |= (fgPhi <<  L1MuRegionalCand::PHI_START);
	  //      edm::LogWarning("MYDEBUG")<<"Checking or vs other "<<roWord<<" "<<fWord;                                                                                                                                                   
	  if (fWord == roWord){//matched
	    for (auto anLCT : csctfFull.second){
	      auto anID =  anLCT.first;
	      auto aDigi = anLCT.second.first;
	      int aStation = anID.station();
	      int aRing = anID.ring();

	      if (aStation>4 || aStation<1) { 
		edm::LogError("BADLCTID")<<"got station "<<aStation;
		continue;
	      }

	      auto layer_geometry = csc_geometry->chamber(anID)->layer(CSCConstants::KEY_CLCT_LAYER)->geometry();

	      typedef  L1MuonParticleExtended::StationData StationData;
	      StationData sData;
	      for (; aDigi != anLCT.second.second; ++aDigi){
		sData.valid = true; 
		sData.id = anID.rawId();
		sData.station = aStation;
		sData.ringOrWheel = aRing;
		sData.bx = aDigi->getBX()-6; //offset to be centered at zero as all higher level objects have it
		sData.quality = aDigi->getQuality();

		int aWG = aDigi->getKeyWG();
		int aHS = aDigi->getStrip();
		float fStrip = 0.5*(aHS+1) - 0.25 ; //mind the offset
		float fWire = layer_geometry->middleWireOfGroup(aWG+1);
		LocalPoint wireStripPoint = layer_geometry->intersectionOfStripAndWire(fStrip, fWire);
		CSCDetId key_id(anID.endcap(), anID.station(), anID.ring(), anID.chamber(), CSCConstants::KEY_CLCT_LAYER); 
		GlobalPoint gp = csc_geometry->idToDet(key_id)->surface().toGlobal(wireStripPoint); 
		sData.phi = gp.phi();
		sData.eta = gp.eta();

		int deltaWire = aWG +1 < layer_geometry->numberOfWireGroups() ? 1 : -1;
		int deltaStrip = fStrip +1 <= layer_geometry->numberOfStrips() ? 1 : -1;
     
		float fStripD1 = fStrip+0.5*deltaStrip;
		float fWireD1 = layer_geometry->middleWireOfGroup(aWG+1 + deltaWire);
		LocalPoint wireStripPlus1 = layer_geometry->intersectionOfStripAndWire(fStripD1, fWireD1);
		GlobalPoint gpD1 = csc_geometry->idToDet(key_id)->surface().toGlobal(wireStripPlus1);
		float dPhi = std::abs(deltaPhi(gp.phi(), gpD1.phi())); 

		float dEta = fabs(gp.eta() - gpD1.eta());

		if (dEta<1E-3){
		  edm::LogWarning("SmallDeltaEta")<<" st "<< aStation<<" r "<<aRing<<" hs "<<aHS<<" wg "<<aWG
						  <<" fs "<< fStrip <<" fw "<<fWire<<" fsp "<<fStripD1<<" fwp "<<fWireD1
						  <<" eta "<<gp.eta()<<" phi "<<gp.phi()<<" etaD1 "<<gpD1.eta()<<" phiD1 "<<gpD1.phi();
		}
		sData.sigmaPhi = dPhi/sqrt(12.); //just the roundoff uncertainty (could be worse)
		sData.sigmaEta = dEta/sqrt(12.);

		sData.bendPhi = aDigi->getGEMDPhi(); //FIXME: need something in normal global coordinates
		
		sData.bendPhiInt = aDigi->getCLCTPattern();
		if (aDigi->getBend()) sData.bendPhiInt*=-1;
	      } //LCTs for a given anID
	      
	      cscParticle->setCscData(sData, aStation);
	      if (gmtParticle) gmtParticle->setCscData(sData, aStation); 
	    } //LCT range for anID
	  }//matched full CSCTF track collection with the current L1MuRegionalCand
	}//loop over csctfts
      }
    }
  }
  
  iEvent.put( muColl );
  if (writeAllCSCTFs_) iEvent.put( cscColl, "csc" );
}

float L1MuonParticleExtendedProducer::getSigmaEta(float eta, float etaP, float etaM) const{
  float sigmaEta=99;

  float dEtaP = std::abs(etaP - eta);
  float dEtaM = std::abs(etaM - eta);
  if (dEtaP  < 1 && dEtaM < 1 && dEtaP  > 0 && dEtaM > 0){
    sigmaEta = sqrt(dEtaP*dEtaM); //take geom mean for no particular reason                                                                                                                                                            
  } else if (dEtaP < 1 || dEtaM < 1) {
    sigmaEta = dEtaP < dEtaM && dEtaP >0  ? dEtaP : dEtaM;
  }

  sigmaEta = sigmaEta/sqrt(12.);

  return sigmaEta;
}

float L1MuonParticleExtendedProducer::getSigmaPhi(float phi, float phiP, float phiM) const{
  float sigmaPhi=99;
  float dPhiP = std::abs(deltaPhi(phiP, phi));
  float dPhiM = std::abs(deltaPhi(phiM, phi));
  if (dPhiP  < 1 && dPhiM < 1 && dPhiP > 0 && dPhiM > 0){
    sigmaPhi = sqrt(dPhiP*dPhiM); //take geom mean for no particular reason                                                                                                                                                            
  } else if (dPhiP < 1 || dPhiM < 1) {
    sigmaPhi = dPhiP < dPhiM && dPhiP > 0 ? dPhiP : dPhiM;
  }
  sigmaPhi = sigmaPhi/sqrt(12.);

  return sigmaPhi;
}

//define this as a plug-in
DEFINE_FWK_MODULE(L1MuonParticleExtendedProducer);
