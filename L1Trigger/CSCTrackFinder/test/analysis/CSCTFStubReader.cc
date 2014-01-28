#include "L1Trigger/CSCTrackFinder/test/analysis/CSCTFStubReader.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include <FWCore/Framework/interface/MakerMacros.h>
#include "DataFormats/Common/interface/Handle.h"
#include <FWCore/Framework/interface/ESHandle.h>
#include <FWCore/MessageLogger/interface/MessageLogger.h>

#include <Geometry/Records/interface/MuonGeometryRecord.h>

// MC particles
#include <SimDataFormats/GeneratorProducts/interface/HepMCProduct.h>

// MC tests
#include <L1Trigger/CSCTriggerPrimitives/test/CSCAnodeLCTAnalyzer.h>
#include <L1Trigger/CSCTriggerPrimitives/test/CSCCathodeLCTAnalyzer.h>
#include <Geometry/CSCGeometry/interface/CSCLayer.h>
#include <Geometry/CSCGeometry/interface/CSCGeometry.h>


#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/MuonDetId/interface/CSCTriggerNumbering.h"
#include "L1Trigger/CSCTrackFinder/interface/CSCTFSectorProcessor.h"//
#include "DataFormats/L1CSCTrackFinder/interface/TrackStub.h"//
#include "L1Trigger/CSCTrackFinder/interface/CSCSectorReceiverLUT.h"//
#include "L1Trigger/CSCTrackFinder/interface/CSCTrackFinderDataTypes.h"//
#include "L1Trigger/CSCTrackFinder/interface/CSCTFPtLUT.h"//
#include "L1Trigger/CSCCommonTrigger/interface/CSCConstants.h"
#include "L1Trigger/CSCCommonTrigger/interface/CSCTriggerGeometry.h"


#include "TCanvas.h"
#include "TFile.h"
#include "TH1.h"
#include "TText.h"
#include "TPaveLabel.h"
#include "TPostScript.h"
#include "TStyle.h"

// Various useful constants
const double CSCTFStubReader::TWOPI = 2.*M_PI;
const std::string CSCTFStubReader::csc_type[CSC_TYPES] = {
  "ME1_1", "ME1_2", "ME1_3", "ME1_A", "ME2_1", "ME2_2", "ME3_1", "ME3_2",
  "ME4_1", "ME4_2"};
const Int_t CSCTFStubReader::MAX_WG[CSC_TYPES] = {//max. number of wiregroups
   48,  65,  43,  17, 120,  65,  96,  65,  96,  65};

const Int_t CSCTFStubReader::MAX_HS[CSC_TYPES] = {//max. number of halfstrips
  128, 160, 128,  96, 160, 160, 160, 160, 160, 160};
const int CSCTFStubReader::ptype[CSCConstants::NUM_CLCT_PATTERNS]= {
  -999,  3, -3,  2,  -2,  1, -1,  0};  // "signed" pattern (== phiBend)


std::map<std::string, CSCSectorReceiverLUT*> srLUTs_[2][MAX_SECTORS];
using namespace std;

bool CSCTFStubReader::bookedMuSimHitsVsMuDigis = false;

CSCTFStubReader::CSCTFStubReader(const edm::ParameterSet& conf){

  // Various input parameters.
  lctProducer_ = conf.getUntrackedParameter<std::string>("CSCTriggerPrimitivesProducer", "");
  wireDigiProducer_ = conf.getParameter<edm::InputTag>("CSCWireDigiProducer");
  compDigiProducer_ = conf.getParameter<edm::InputTag>("CSCComparatorDigiProducer");
  debug        = conf.getUntrackedParameter<bool>("debug", true);
  outFile = conf.getUntrackedParameter<std::string>("OutFile");
  edm::ParameterSet srLUTset = conf.getParameter<edm::ParameterSet>("SRLUT");
  for(int e = CSCDetId::minEndcapId(); e <= CSCDetId::maxEndcapId(); ++e)
    {
      for(int s = CSCTriggerNumbering::minTriggerSectorId();
          s <= CSCTriggerNumbering::maxTriggerSectorId(); ++s)
        {
          for(int i = 1; i <= 4; ++i)
            {
              if(i == 1)
                for(int j = 0; j < 2; j++)
                  {
                    srLUTs_[e-1][s-1][FPGAs[j]] = new CSCSectorReceiverLUT(e, s, j+1, i, srLUTset);
                  }
              else
                srLUTs_[e-1][s-1][FPGAs[i]] = new CSCSectorReceiverLUT(e, s, 0, i, srLUTset);
            }
        }
    }
  fAnalysis = new TFile(outFile.c_str(), "RECREATE");
  event=0;

}


CSCTFStubReader::~CSCTFStubReader(){
  //delete the file

  // delete srLUTs
  for(int e = CSCDetId::minEndcapId(); e <= CSCDetId::maxEndcapId(); ++e)
    {
      for(int s = CSCTriggerNumbering::minTriggerSectorId();
	  s <= CSCTriggerNumbering::maxTriggerSectorId(); ++s)
	{
	  for(int i = 0; i < 5; ++i)
	    {
	      delete srLUTs_[e-1][s-1][FPGAs[i]]; // delete the pointer
	      srLUTs_[e-1][s-1][FPGAs[i]] = NULL; // point it at a safe place
	    }
	}
    }

}

void CSCTFStubReader::analyze(const edm::Event& ev,
			      const edm::EventSetup& setup){

  event++;
  edm::ESHandle<CSCGeometry> cscGeom;
  setup.get<MuonGeometryRecord>().get(cscGeom);
  geom_ = &*cscGeom;

  CSCTriggerGeometry::setGeometry(cscGeom);

  edm::Handle<CSCALCTDigiCollection> alcts;
  edm::Handle<CSCCLCTDigiCollection> clcts;
  ev.getByLabel(lctProducer_,              alcts);
  ev.getByLabel(lctProducer_,              clcts);

  edm::Handle<CSCCorrelatedLCTDigiCollection> lcts_mpc;
  edm::Handle<CSCCorrelatedLCTDigiCollection> lcts_tmb;
  ev.getByLabel(lctProducer_, "MPCSORTED", lcts_mpc);
  ev.getByLabel(lctProducer_,  lcts_tmb);
  MCStudies(ev,lcts_mpc.product(),alcts.product(),clcts.product());
  //compareGetAClctMethods(lcts_mpc.product(),alcts.product(),clcts.product());
}

void CSCTFStubReader::MCStudies(const edm::Event& ev,
				const CSCCorrelatedLCTDigiCollection* lcts_mpc,
				const CSCALCTDigiCollection* alcts,
				const CSCCLCTDigiCollection* clcts){

  std::vector<edm::Handle<edm::HepMCProduct> > allhepmcp;
  // Use "getManyByType" to be able to check the existence of MC info.
  ev.getManyByType(allhepmcp);
  //cout << "HepMC info: " << allhepmcp.size() << endl;
  if (allhepmcp.size() > 0) {
    // If hepMC info is there, try to get wire and comparator digis,
    // and SimHits.
    edm::Handle<CSCWireDigiCollection>       wireDigis;
    edm::Handle<CSCComparatorDigiCollection> compDigis;
    edm::Handle<edm::PSimHitContainer>       simHits;
    ev.getByLabel(wireDigiProducer_.label(), wireDigiProducer_.instance(),
		  wireDigis);
    ev.getByLabel(compDigiProducer_.label(), compDigiProducer_.instance(),
		  compDigis);
    ev.getByLabel("g4SimHits", "MuonCSCHits", simHits);
    if (debug) LogDebug("CSCTFStubReader")
      << "   #CSC SimHits: " << simHits->size();

    fillMuSimHitsVsMuDigis(ev, lcts_mpc, alcts,clcts, wireDigis.product(), compDigis.product(),
			   simHits.product());
  }
}

void CSCTFStubReader::fillMuSimHitsVsMuDigis(const edm::Event& ev,
					     const CSCCorrelatedLCTDigiCollection* lcts,
					     const CSCALCTDigiCollection* alcts,
					     const CSCCLCTDigiCollection* clcts,
					     const CSCWireDigiCollection* wiredc,
					     const CSCComparatorDigiCollection* compdc,
					     const edm::PSimHitContainer* allSimHits)
{

  std::vector<edm::Handle<edm::HepMCProduct> > allhepmcp;
  ev.getManyByType(allhepmcp);
  //if (debug) cout << "HepMC info: " << allhepmcp.size() << endl;
  const HepMC::GenEvent& mc = allhepmcp[0]->getHepMCData();
  int igenp = 0;
  double genEta=0, genPhi=0;
  for (HepMC::GenEvent::particle_const_iterator p = mc.particles_begin();
       p != mc.particles_end(); ++p) {
    int id = (*p)->pdg_id();
    double phitmp = (*p)->momentum().phi();
    if (phitmp < 0) phitmp += 2.*M_PI;
    if (debug) LogDebug("CSCTFStubReader")
      << "MC part #" << ++igenp << ": id = "  << id
      << ", status = " << (*p)->status()
      << ", pT = " << (*p)->momentum().perp() << " GeV"
      << ", eta = " << (*p)->momentum().pseudoRapidity()
      << ", phi = " << phitmp*180./M_PI << " deg";
    if((id==13||id==-13)&& (*p)->momentum().perp()>0 )
      {
	genEta=(*p)->momentum().pseudoRapidity();
	genPhi=phitmp;
      }
  }

  const Double_t ETA_BIN = 0.0125;
  const Double_t PHI_BIN = 62.*M_PI/180./4096.; // 0.26 mrad
  const Double_t ETA_HALFBIN = ETA_BIN/2.;
  const Double_t PHI_HALFBIN = PHI_BIN/2.;
  //  Int_t numALCT = 0, numCLCT = 0;
  Double_t eta_sim = -999.0, eta_rec = -999.0;
  Double_t phi_sim = -999.0, phi_rec = -999.0;
  Double_t eta_diff = -999.0, eta_bdiff = -999.0;
  Double_t phi_diff = -999.0, phi_bdiff = -999.0;
  Int_t nValidALCTs = 0, nValidCLCTs = 0;


  if(!bookedMuSimHitsVsMuDigis) bookMuSimHitsVsMuDigis();
  CSCAnodeLCTAnalyzer alct_analyzer;
  alct_analyzer.setGeometry(geom_);
  CSCCathodeLCTAnalyzer clct_analyzer;
  clct_analyzer.setGeometry(geom_);
   //for stubs
  CSCTriggerContainer<csctf::TrackStub> stub_list;
  CSCCorrelatedLCTDigiCollection::DigiRangeIterator detUnitIt;
  for(detUnitIt = lcts->begin(); detUnitIt != lcts->end(); detUnitIt++) {
    const CSCDetId& id = (*detUnitIt).first;
    const CSCCorrelatedLCTDigiCollection::Range& range = (*detUnitIt).second;
    for (CSCCorrelatedLCTDigiCollection::const_iterator digiIt = range.first;
	 digiIt != range.second; digiIt++) {
      csctf::TrackStub theStub((*digiIt),id);
      stub_list.push_back(theStub);
    }
  }

  for(int e = CSCDetId::minEndcapId(); e <= CSCDetId::maxEndcapId(); ++e) {
    for(int s = CSCTriggerNumbering::minTriggerSectorId();
	s <= CSCTriggerNumbering::maxTriggerSectorId(); ++s) {
      CSCTriggerContainer<csctf::TrackStub> current_e_s = stub_list.get(e, s);
      std::vector<csctf::TrackStub> stub_vec = current_e_s.get();
      std::vector<csctf::TrackStub>::const_iterator end = stub_vec.end();
      // hStubNumSector[s-1]->Fill(stub_vec.size());
      for(std::vector<csctf::TrackStub>::iterator itr = stub_vec.begin();
	  itr != end; itr++) {
	CSCDetId id(itr->getDetId().rawId());
	if( ( id.triggerSector() >= 1 )            &&
	    ( id.triggerSector() <= MAX_SECTORS )  &&
	    ( id.station()       >= 1 )            &&
	    ( id.station()       <= MAX_STATIONS ) &&
	    ( id.triggerCscId()  >= 1 )            &&
	    ( id.triggerCscId()  <= MAX_CHAMBERS ) ) {
	  unsigned fpga = (id.station() == 1) ? CSCTriggerNumbering::triggerSubSectorFromLabels(id) - 1 : id.station();
	  // run through SR_LUTs
	  lclphidat lclPhi =
	    srLUTs_[e-1][s-1][FPGAs[fpga]]->localPhi(itr->getStrip(), itr->getPattern(),
						     itr->getQuality(),     itr->getBend());
	  gblphidat gblPhi = srLUTs_[e-1][s-1][FPGAs[fpga]]->globalPhiME(lclPhi.phi_local,
									 itr->getKeyWG(), itr->cscid());
	  gbletadat gblEta = srLUTs_[e-1][s-1][FPGAs[fpga]]->globalEtaME(lclPhi.phi_bend_local,
									 lclPhi.phi_local, itr->getKeyWG(), itr->cscid());
	  itr->setEtaPacked(gblEta.global_eta);
	  itr->setPhiPacked(gblPhi.global_phi);
	  //======change lphi->gphi
	  double cent = CSCTFConstants::SECTOR1_CENT_RAD;
	  double sec = CSCTFConstants::SECTOR_RAD;//SECTOR_DEG=62
	  double gphi = itr->phiValue() + cent - sec/2. + (s-1)*M_PI/3. ;// you should calc the sector offset carefully

	  if (gphi > TWOPI ) gphi -= TWOPI;

	  int station = id.station()-1; //according to station #
	  //==because in lable station num start from 1, but we need it start from 0
	  int csctype = getCSCType(id);//according to csctype #

	  bool alct_valid = false;
	  bool clct_valid = false;
	  const CSCCorrelatedLCTDigi* lct = itr->getDigi();
	  bool lct_valid = lct->isValid();

	  if (lct_valid) {

	    hLctMPCEndcap->Fill(id.endcap());
	    hLctMPCStation->Fill(id.station());
	    hLctMPCSector->Fill(id.triggerSector());
	    hLctMPCRing->Fill(id.ring());
	    hLctMPCChamber[id.station()-1]->Fill(id.triggerCscId());

	    int quality = lct->getQuality();
	    hLctMPCQuality->Fill(quality);
	    hLctMPCBXN->Fill(lct->getBX());

	    alct_valid = (quality != 4 && quality != 5);
	    if (alct_valid) {
	      hLctMPCKeyGroup->Fill(lct->getKeyWG());
	    }

	    clct_valid = (quality != 1 && quality != 3);
	    if (clct_valid) {
	      hLctMPCKeyStrip->Fill(lct->getStrip());
	      hLctMPCStripType->Fill(lct->getStripType());
	      hLctMPCPattern->Fill(lct->getCLCTPattern());
	      hLctMPCBend->Fill(lct->getBend());
	    }

	    CSCALCTDigi anodeLCT;
	    CSCCLCTDigi cathodeLCT;

	    if (quality != 4 && quality != 5) { //lct_alct_valid
	      int ialct=0;
	      CSCALCTDigiCollection::DigiRangeIterator AdetUnitIt;
	      for (AdetUnitIt = alcts->begin(); AdetUnitIt != alcts->end(); AdetUnitIt++) {
		const CSCDetId& aid = (*AdetUnitIt).first;
		const CSCALCTDigiCollection::Range& range = (*AdetUnitIt).second;
		for (CSCALCTDigiCollection::const_iterator digiIt = range.first;
		     digiIt != range.second; digiIt++) {
		  alct_valid = (*digiIt).isValid();
		  if (alct_valid) {
		    if(id == aid && //lct_id == alct_id
		       lct->getKeyWG() == (*digiIt).getKeyWG() &&
		       lct->getBX() == (*digiIt).getBX()) {
		      anodeLCT = *digiIt;
		      ialct++;
		    }//lctID == id
		  }//alct_valid
		}//range
	      }//alct loop
	    }//(quality!=4 && !=5 )
	    if (quality != 1 && quality != 3) { //lct_clct_valid
	      int iclct=0;
	      CSCCLCTDigiCollection::DigiRangeIterator CdetUnitIt;
	      for (CdetUnitIt = clcts->begin(); CdetUnitIt != clcts->end(); CdetUnitIt++) {
		const CSCDetId& cid = (*CdetUnitIt).first;
		const CSCCLCTDigiCollection::Range& range = (*CdetUnitIt).second;
		for (CSCCLCTDigiCollection::const_iterator digiIt = range.first;
		     digiIt != range.second; digiIt++) {

		  clct_valid = (*digiIt).isValid();
		  //if (clct_1.isValid() && clct_valid) {
		  if (clct_valid) {
		    if(id==cid && //lct_id == clct_id
		       lct->getStrip() == (*digiIt).getKeyStrip() &&
		       lct->getCLCTPattern() == (*digiIt).getPattern()) {
		      cathodeLCT = *digiIt;
		      iclct++;
		    }//lctID == id
		  }//clct_valid
		}//range
	      }//clct loop
	    }// quality!=1 && !=3

	    alct_valid = anodeLCT.isValid();
	    clct_valid = cathodeLCT.isValid();
	    int clctStripType=cathodeLCT.getStripType();
	    int clctKeyStrip = cathodeLCT.getKeyStrip();// halfstrip #
	    if (clctStripType == 0) clctKeyStrip /= 4;  // distrip # for distrip ptns

	    hAlctValid->Fill(alct_valid);
	    if (alct_valid) {
	      hAlctQuality->Fill(anodeLCT.getQuality());
	      hAlctAccel->Fill(anodeLCT.getAccelerator());
	      hAlctCollis->Fill(anodeLCT.getCollisionB());
	      hAlctKeyGroup->Fill(anodeLCT.getKeyWG());
	      hAlctBXN->Fill(anodeLCT.getBX());
	      hAlctPerCSC->Fill(csctype);
	      nValidALCTs++;
	    }

	    // CLCTs
	    hClctValid->Fill(clct_valid);
	    if (clct_valid) {
	      hClctQuality->Fill(cathodeLCT.getQuality());
	      hClctStripType->Fill(cathodeLCT.getStripType());
	      hClctSign->Fill(cathodeLCT.getBend());
	      hClctBXN->Fill(cathodeLCT.getBX());
	      hClctCFEB->Fill(cathodeLCT.getCFEB());
	      // ClctStrip->Fill(cathodeLCT.getStrip());
	      hClctKeyStrip[clctStripType]->Fill(clctKeyStrip);
	      hClctPattern[clctStripType]->Fill(cathodeLCT.getPattern());
	      hClctPerCSC->Fill(csctype);
	      hClctPatternCsc[csctype][clctStripType]->Fill(ptype[cathodeLCT.getPattern()]);
	      if (clctStripType == 0) // distrips
		hClctKeyStripCsc[csctype]->Fill(clctKeyStrip);
	      nValidCLCTs++;
	    }

	    // Eta resolutions
	    Bool_t alct_mc = true;
	    if (alct_valid) {
	      vector<CSCAnodeLayerInfo> alctInfo =
		alct_analyzer.getSimInfo(anodeLCT, id, wiredc, allSimHits);
	      double hitPhi = -999.0, hitEta = -999.0;
	      alct_analyzer.nearestWG(alctInfo, hitPhi, hitEta);

	      eta_sim = hitEta;
	      eta_rec = itr->etaValue() + ETA_HALFBIN;

	      if (eta_sim > -990.) { // eta_sim = -999 when MC info is not available
		eta_diff  = eta_rec - fabs(eta_sim);
		eta_bdiff = eta_diff/ETA_BIN;

		if (debug) LogDebug("CSCTFStubReader")
		  << "  Endcap = " << id.endcap()
		  << " station = " << id.station()
		  << " chamber = " << id.chamber()
		  << ": eta_sim = " << eta_sim << " eta_rec = " << eta_rec
		  << " diff = " << eta_diff;
	      }
	      else {
		if (debug) LogDebug("CSCTFStubReader")
		  << "+++ Event # " << event
		  << " some ALCT MC info is not available +++";
		alct_mc = false;
	      }
	    }

	    // Phi resolutions
	    Bool_t clct_mc = true;
	    if (clct_valid) {
	      vector<CSCCathodeLayerInfo> clctInfo =
		clct_analyzer.getSimInfo(cathodeLCT, id, compdc, allSimHits);
	      double hitPhi = -999.0, hitEta = -999.0;
	      clct_analyzer.nearestHS(clctInfo, hitPhi, hitEta);

	      phi_sim = hitPhi;
	      phi_rec = gphi + PHI_HALFBIN;
	      if (phi_sim > -990.) { // phi_sim = -999 when MC info is not available
		phi_diff = phi_rec - phi_sim;
		if (fabs(phi_diff) > M_PI) {
		  if (debug) {
		    LogDebug("CSCTFStubReader")
		      << "Correct phi_diff in event # " << event;
		    LogDebug("CSCTFStubReader")
		      << "\t Before correction: " << " phi_rec = "  << phi_rec
		      << " phi_sim = " << phi_sim << " phi_diff = " << phi_diff;
		  }
		  if      (phi_diff >  M_PI) phi_diff -= TWOPI;
		  else if (phi_diff < -M_PI) phi_diff += TWOPI;
		  if (debug) {
		    LogDebug("CSCTFStubReader")
		      << "\t After correction : "
		      << " phi_diff = " << phi_diff;
		  }
		}
		phi_bdiff = phi_diff/PHI_BIN;
		phi_diff *= 1000.; // convert to mrad

		if (debug) LogDebug("CSCTFStubReader")
		  << "  Endcap = " << id.endcap()
		  << " station = " << id.station()
		  << " chamber = " << id.chamber()
		  << ": phi_sim = " << phi_sim << " phi_rec = " << phi_rec
		  << " diff = " << phi_diff;

	      }
	      else {
		if (debug) LogDebug("CSCTFStubReader")
		  << "+++ Event # " << event
		  << " some CLCT MC info is not available +++";
		clct_mc = false;
	      }
	    }

	    // Eta histograms
	    if (alct_valid) {
	      LctVsEta[station][0]->Fill(eta_rec);
	      //LctVsEta[station][1]->Fill(lctEta[ilct]);

	      //LctVsEtaCsc[csctype]->Fill(lctEta[ilct]);

	      if (alct_mc) {
		EtaRecVsSim->Fill(fabs(eta_sim), eta_rec);
		EtaDiff[0]->Fill(eta_diff);
		EtaDiff[1]->Fill(eta_bdiff);

		EtaDiffCsc[csctype][0]->Fill(eta_diff);
		EtaDiffCsc[csctype][3]->Fill(eta_bdiff);
		EtaDiffCsc[csctype][e]->Fill(eta_diff);

		EtaDiffVsEta[station]->Fill(eta_rec, fabs(eta_diff));
		EtaDiffVsWireCsc[csctype]->Fill(anodeLCT.getKeyWG(), eta_bdiff);
	      }
	    }

	    // Phi histograms
	    if (clct_valid) {
	      LctVsPhi[station]->Fill(phi_rec);
	      if (clctStripType == 0) // distrips
		KeyStripCsc[csctype]->Fill(clctKeyStrip);
	      int phibend = ptype[cathodeLCT.getPattern()];
	      //PatternCsc[csctype][clctStripType]->Fill(cathodeLCT.getBend());
	      PatternCsc[csctype][clctStripType]->Fill(phibend);

	      if (clct_mc) {
		PhiRecVsSim->Fill(phi_sim, phi_rec);
		PhiDiff[0]->Fill(phi_diff);
		PhiDiff[1]->Fill(phi_bdiff);

		PhiDiffCsc[csctype][0]->Fill(phi_diff);
		PhiDiffCsc[csctype][3]->Fill(phi_bdiff);
		PhiDiffCsc[csctype][e]->Fill(phi_diff);
		PhiDiffCsc[csctype][clctStripType+4]->Fill(phi_diff);
		PhiDiffCsc[csctype][clctStripType+7]->Fill(phi_bdiff);
		if (clctStripType == 1 && phibend == 0)
		  PhiDiffCsc[csctype][6]->Fill(phi_diff); // halfstrips, straight pattern

		PhiDiffVsPhi[station]->Fill(phi_rec, fabs(phi_diff));
		PhiDiffVsStripCsc[csctype][clctStripType]->Fill(clctKeyStrip,
								phi_diff);

		// Histograms to check phi offsets for various pattern types
		if (clctStripType == 1) { // halfstrips
		  Double_t hsperrad = getHsPerRad(csctype); // halfstrips-per-radian
		  if((e == 1 && (station==1 || station==2)) ||
		     (e == 2 && (station==3 || station==4)))
		    //int phibend = ptype[cathodeLCT.getPattern()];
		    PhiDiffPattern[phibend+4]->Fill(phi_diff/1000*hsperrad);
		}
	      }
	    }

	    // Eta-vs-phi histograms
	    if (alct_valid && clct_valid) {
	      if (alct_mc) {
		EtaDiffCsc[csctype][clctStripType+4]->Fill(eta_diff);
		EtaDiffVsPhi[station]->Fill(phi_rec, fabs(eta_diff));
		EtaDiffVsStripCsc[csctype][clctStripType]->Fill(clctKeyStrip,
								eta_bdiff);
		if (clctStripType == 1) { // halfstrips
		  EtaDiffVsStripCsc[csctype][e+1]->Fill(clctKeyStrip,
							eta_bdiff);
		}
	      }
	      if (clct_mc) {
		PhiDiffVsEta[station]->Fill(eta_rec, fabs(phi_diff));
		PhiDiffVsWireCsc[csctype]->Fill(anodeLCT.getKeyWG(), phi_diff);
	      }
	    }//alct clct valid
	  }//lct_valid
	}
	else
	  {
	    edm::LogWarning("CSCTFStubReader") << "Det ID is out of range";
	    edm::LogWarning("CSCTFStubReader") << "Sector: " << id.triggerSector();
	    edm::LogWarning("CSCTFStubReader") << "Station: " <<id.station();
	    edm::LogWarning("CSCTFStubReader") << "CSCID: " << id.triggerCscId();
	  }
      }//stub_vec loop
    }//trigger sector loop
  }//endcap loop
  hAlctPerEvent->Fill(nValidALCTs);
  hClctPerEvent->Fill(nValidCLCTs);
}




// Returns chamber type (0-9) according to the station and ring number
int CSCTFStubReader::getCSCType(const CSCDetId& id) {
  int type = -999;

  if (id.station() == 1) {
    type = (id.triggerCscId()-1)/3;
  }
  else { // stations 2-4
    type = 3 + id.ring() + 2*(id.station()-2);
  }

  assert(type >= 0 && type < CSC_TYPES-1); // no ME4/2
  return type;
}

void CSCTFStubReader::bookMuSimHitsVsMuDigis()
{
  hLctMPCPerEvent  = new TH1F("LCTs_per_event", "LCTs per event",    11, -0.5, 10.5);
  hLctMPCPerCSC    = new TH1F("LCTs_per_CSC_type", "LCTs per CSC type", 10, -0.5,  9.5);
  hCorrLctMPCPerCSC= new TH1F("Corr_LCTs_per_CSC_type", "Corr. LCTs per CSC type", 10, -0.5,9.5);
  hLctMPCEndcap    = new TH1F("LCT_Endcap", "Endcap",             4, -0.5,  3.5);
  hLctMPCStation   = new TH1F("LCT_Station", "Station",            6, -0.5,  5.5);
  hLctMPCSector    = new TH1F("LCT_Sector", "Sector",             8, -0.5,  7.5);
  hLctMPCRing      = new TH1F("LCT_Ring", "Ring",               5, -0.5,  4.5);

  hLctMPCValid     = new TH1F("LCT_validity", "LCT validity",        3, -0.5,   2.5);
  hLctMPCQuality   = new TH1F("LCT_quality", "LCT quality",        17, -0.5,  16.5);
  hLctMPCKeyGroup  = new TH1F("LCT_key_wiregroup", "LCT key wiregroup", 120, -0.5, 119.5);
  hLctMPCKeyStrip  = new TH1F("LCT_key_strip", "LCT key strip",     160, -0.5, 159.5);
  hLctMPCStripType = new TH1F("LCT_strip_type", "LCT strip type",      3, -0.5,   2.5);
  hLctMPCPattern   = new TH1F("LCT_pattern", "LCT pattern",        10, -0.5,   9.5);
  hLctMPCBend      = new TH1F("LCT_LR_bend", "LCT L/R bend",        3, -0.5,   2.5);
  hLctMPCBXN       = new TH1F("LCT_bx", "LCT bx",             20, -0.5,  19.5);
  // LCT quantities per station
  char histname[60];
  for (int istat = 0; istat < MAX_STATIONS; istat++) {
    sprintf(histname, "LCT_CSCId_station%d", istat+1);
    hLctMPCChamber[istat] = new TH1F(histname, histname,  10, -0.5, 9.5);
  }

  //void CSCTriggerPrimitivesReader::bookALCTHistos() {
  hAlctPerEvent = new TH1F("o_ALCTsPerEvent", "ALCTs per event",     11, -0.5,  10.5);
  hAlctPerCSC   = new TH1F("o_ALCTsPerCSCType", "ALCTs per CSC type",  10, -0.5,   9.5);
  hAlctValid    = new TH1F("o_ALCTValidity", "ALCT validity",        3, -0.5,   2.5);
  hAlctQuality  = new TH1F("o_ALCTQuality", "ALCT quality",         5, -0.5,   4.5);
  hAlctAccel    = new TH1F("o_ALCTAccelFlag", "ALCT accel. flag",     3, -0.5,   2.5);
  hAlctCollis   = new TH1F("o_ALCTCollisionFlag", "ALCT collision. flag", 3, -0.5,   2.5);
  hAlctKeyGroup = new TH1F("o_ALCTKeyWireGroup", "ALCT key wiregroup", 120, -0.5, 119.5);
  hAlctBXN      = new TH1F("o_ALCTBx", "ALCT bx",             20, -0.5,  19.5);

  //void CSCTriggerPrimitivesReader::bookCLCTHistos() {
  hClctPerEvent  = new TH1F("o_CLCTsPerEvent", "CLCTs per event",    11, -0.5, 10.5);
  hClctPerCSC    = new TH1F("o_CLCTsPerCSCType", "CLCTs per CSC type", 10, -0.5,  9.5);
  hClctValid     = new TH1F("o_CLCTValidity", "CLCT validity",       3, -0.5,  2.5);
  hClctQuality   = new TH1F("o_CLCTLayersHit", "CLCT layers hit",     8, -0.5,  7.5);
  hClctStripType = new TH1F("o_CLCTStripType", "CLCT strip type",     3, -0.5,  2.5);
  hClctSign      = new TH1F("o_CLCTSignLR", "CLCT sign (L/R)",     3, -0.5,  2.5);
  hClctCFEB      = new TH1F("o_CLCTCFEBnum", "CLCT cfeb #",         6, -0.5,  5.5);
  hClctBXN       = new TH1F("o_CLCTBx", "CLCT bx",            20, -0.5, 19.5);

  hClctKeyStrip[0] = new TH1F("o_CLCTKeyStripDistrips","CLCT keystrip, distrips",   40, -0.5,  39.5);
  //hClctKeyStrip[0] = new TH1Fo_CLCT("","CLCT keystrip, distrips",  160, -0.5, 159.5);
  hClctKeyStrip[1] = new TH1F("o_CLCTKeyStripHalfStrips","CLCT keystrip, halfstrips",160, -0.5, 159.5);
  hClctPattern[0]  = new TH1F("o_CLCTPatternDistrips","CLCT pattern, distrips",    10, -0.5,   9.5);
  hClctPattern[1]  = new TH1F("o_CLCTPatternHalfStrips","CLCT pattern, halfstrips",  10, -0.5,   9.5);
  for (int i = 0; i < CSC_TYPES; i++) {
    string s1 = "Pattern number, " + csc_type[i];
    string n1 = "o_CLCTPatternNumber_"+csc_type[i];
    hClctPatternCsc[i][0] = new TH1F(n1.c_str(), s1.c_str(),  9, -4.5, 4.5);
    hClctPatternCsc[i][1] = new TH1F(n1.c_str(), s1.c_str(),  9, -4.5, 4.5);

    string s2 = "CLCT keystrip, " + csc_type[i];
    string n2 = "o_CLCTkeyStrip_" + csc_type[i];
    int max_ds = MAX_HS[i]/4;
    hClctKeyStripCsc[i]   = new TH1F(n2.c_str(), s2.c_str(), max_ds, 0., max_ds);
  }
  //---------


  EtaRecVsSim = new TH2F("o_eta_rec_vs_eta_sim", "#eta_rec vs #eta_sim",
			  64, 0.9,  2.5,  64, 0.9,  2.5);
  PhiRecVsSim = new TH2F("o_phi_rec_vs_phi_sim", "#phi_rec vs #phi_sim",
			 100, 0., TWOPI, 100, 0., TWOPI);

  // Limits for resolution histograms
  const Double_t EDMIN = -0.05; // eta min
  const Double_t EDMAX =  0.05; // eta max
  const Double_t PDMIN = -5.0;  // phi min (mrad)
  const Double_t PDMAX =  5.0;  // phi max (mrad)

  EtaDiff[0]  = new TH1F("o_eta_rec_minus_eta_sim", "#eta_rec-#eta_sim",        100, EDMIN, EDMAX);
  EtaDiff[1]  = new TH1F("o_eta_rec_minus_ets_sim_etabins", "#eta_rec-#eta_sim (#eta bins)", 100,  -4.,  4.);
  PhiDiff[0]  = new TH1F("o_phi_rec_minus_phi_sim_mrad", "#phi_rec-#phi_sim (mrad)", 100, PDMIN, PDMAX);
  PhiDiff[1]  = new TH1F("o_phi_rec_minus_phi_sim_phibins", "#phi_rec-#phi_sim (#phi bins)", 100, -30., 30.);

  // LCT quantities per station

  char histname1[60];
  char histtitle[60];

  for (Int_t i = 0; i < MAX_STATIONS; i++) {
//    sprintf(histname, "CSCId, station %d", i+1);
//    LctChamber[i]   = new TH1F("", histname,  10, -0.5,   9.5);

    sprintf(histtitle, "LCTs vs eta, station %d", i+1);
    sprintf(histname, "o_LCTs_vs_eta_station_%d", i+1);
    LctVsEta[i][0]  = new TH1F(histname, histtitle,  66,  0.875, 2.525);
    sprintf(histtitle, "LCTs vs eta bin, station %d", i+1);
    sprintf(histname, "o_LCTs_vs_etabin_station_%d", i+1);
    LctVsEta[i][1]  = new TH1F(histname, histtitle, 128,  0.,    128.);

    sprintf(histtitle, "LCTs vs phi, station %d", i+1);
    sprintf(histname, "o_LCTs_vs_phi_station_%d", i+1);
    LctVsPhi[i]     = new TH1F(histname, histtitle, 100,  0.,    TWOPI);

    sprintf(histtitle, "#LT#eta_rec-#eta_sim#GT, station %d", i+1);
    sprintf(histname, "o_mean_etadiff_vs_eta_station_%d", i+1);
    sprintf(histname1, "o_mean_etadiff_vs_phi_station_%d", i+1);
    EtaDiffVsEta[i] = new TH1F(histname, histtitle,  66,  0.875, 2.525);
    EtaDiffVsPhi[i] = new TH1F(histname1, histtitle, 100,  0.,    TWOPI);

    sprintf(histtitle, "#LT#phi_rec-#phi_sim#GT, station %d", i+1);
    sprintf(histname, "o_mean_phidiff_vs_eta_station_%d", i+1);
    sprintf(histname1, "o_mean_phidiff_vs_phi_station_%d", i+1);
    PhiDiffVsEta[i] = new TH1F(histname, histtitle,  66,  0.875, 2.525);
    PhiDiffVsPhi[i] = new TH1F(histname1, histtitle, 100,  0.,    TWOPI);
  }

  for (Int_t i = 0; i < CSC_TYPES; i++) {
    string t0 = "#eta_rec-#eta_sim, " + csc_type[i];
    string n0 = "o_etadiff_" + csc_type[i];
    EtaDiffCsc[i][0] = new TH1F(n0.c_str(), t0.c_str(), 100, EDMIN, EDMAX);
    string t1 = t0 + ", endcap1";
    string n1 = n0 + "_endcap1";
    EtaDiffCsc[i][1] = new TH1F(n1.c_str(), t1.c_str(), 100, EDMIN, EDMAX);
    string t2 = t0 + ", endcap2";
    string n2 = n0 + "_endcap2";
    EtaDiffCsc[i][2] = new TH1F(n2.c_str(), t2.c_str(), 100, EDMIN, EDMAX);
    string t3 = t0 + ", eta bins";
    string n3 = n0 + "_etabins";
    string n31 = n3 + "_endcap1";
    string n32 = n3 + "_endcap2";
    EtaDiffCsc[i][3] = new TH1F(n3.c_str(), t3.c_str(), 100,   -4.,    4.);
    EtaDiffCsc[i][4] = new TH1F(n31.c_str(), t0.c_str(), 100, EDMIN, EDMAX);
    EtaDiffCsc[i][5] = new TH1F(n32.c_str(), t0.c_str(), 100, EDMIN, EDMAX);

    string t4 = "#eta_rec-#eta_sim (bins) vs wiregroup, " + csc_type[i];
    string n4 = "o_etadiff_vs_wiregroup_" + csc_type[i];
    EtaDiffVsWireCsc[i] =
      new TH2F(n4.c_str(), t4.c_str(), MAX_WG[i], 0., MAX_WG[i], 100, -3., 3.);

    Int_t MAX_DS = MAX_HS[i]/4;
    string t5 = "#eta_rec-#eta_sim (bins) vs distrip, " + csc_type[i];
    string n5 = "o_etadiff_bins_vs_distrip_" + csc_type[i];
    EtaDiffVsStripCsc[i][0] =
      new TH2F(n5.c_str(), t5.c_str(), MAX_DS,    0., MAX_DS,    100, -3., 3.);
    string t6 = "#eta_rec-#eta_sim (bins) vs halfstrip, " + csc_type[i];
    string n6 = "o_etadiff_bins_vs_halfstrip_" + csc_type[i];
    EtaDiffVsStripCsc[i][1] =
      new TH2F(n6.c_str(), t6.c_str(), MAX_HS[i], 0., MAX_HS[i], 100, -3., 3.);
    string t7 = t6 + ", endcap1";
    string n7 = n6 + "_endcap1";
    EtaDiffVsStripCsc[i][2] =
      new TH2F(n7.c_str(), t7.c_str(), MAX_HS[i], 0., MAX_HS[i], 100, -3., 3.);
    string t8 = t6 + ", endcap2";
    string n8 = n6 + "_endcap2";
    EtaDiffVsStripCsc[i][3] =
      new TH2F(n8.c_str(), t8.c_str(), MAX_HS[i], 0., MAX_HS[i], 100, -3., 3.);

    string t9 = "LCTs vs eta bin, " + csc_type[i];
    string n9 = "o_LCTs_vs_eta_bin, " + csc_type[i];
    LctVsEtaCsc[i] = new TH1F(n9.c_str(), t9.c_str(), 128, 0., 128.);

    string u0 = "#phi_rec-#phi_sim, " + csc_type[i];
    string v0 = "o_phidiff_" + csc_type[i];
    PhiDiffCsc[i][0] = new TH1F(v0.c_str(), u0.c_str(), 100, PDMIN, PDMAX);
    string u1 = u0 + ", endcap1";
    string v1 = v0 + "_endcap1";
    PhiDiffCsc[i][1] = new TH1F(v1.c_str(), u1.c_str(), 100, PDMIN, PDMAX);
    string u2 = u0 + ", endcap2";
    string v2 = v0 + "_endcap2";
    PhiDiffCsc[i][2] = new TH1F(v2.c_str(), u2.c_str(), 100, PDMIN, PDMAX);
    string u3 = u0 + ", phi bins";
    string v33 = v0 + "_phibins3";
    string v34 = v0 + "_phibins4";
    string v35 = v0 + "_phibins5";
    string v36 = v0 + "_phibins6";
    string v37 = v0 + "_phibins7";
    string v38 = v0 + "_phibins8";
    PhiDiffCsc[i][3] = new TH1F(v33.c_str(), u3.c_str(), 100,  -10.,   10.);
    PhiDiffCsc[i][4] = new TH1F(v34.c_str(), u0.c_str(), 100, PDMIN, PDMAX);
    PhiDiffCsc[i][5] = new TH1F(v35.c_str(), u0.c_str(), 100, PDMIN, PDMAX);
    PhiDiffCsc[i][6] = new TH1F(v36.c_str(), u0.c_str(), 100, PDMIN, PDMAX);
    PhiDiffCsc[i][7] = new TH1F(v37.c_str(), u0.c_str(), 100,  -10.,   10.);
    PhiDiffCsc[i][8] = new TH1F(v38.c_str(), u0.c_str(), 100,  -10.,   10.);

    string u4 = "#phi_rec-#phi_sim (mrad) vs wiregroup, " + csc_type[i];
    string v4 = "o_phidiff_mrad_vs_wiregroup_" + csc_type[i];
    PhiDiffVsWireCsc[i] =
      new TH2F(v4.c_str(), u4.c_str(), MAX_WG[i], 0., MAX_WG[i], 100, PDMIN, PDMAX);

    string u5 = "#phi_rec-#phi_sim (mrad) vs distrip, " + csc_type[i];
    string v5 = "o_phidiff_mrad_vs_distrip_" + csc_type[i];
    PhiDiffVsStripCsc[i][0] =
      new TH2F(v5.c_str(), u5.c_str(), MAX_DS,    0., MAX_DS,    100, PDMIN, PDMAX);
    string u6 = "#phi_rec-#phi_sim (mrad) vs halfstrip, " + csc_type[i];
    string v6 = "o_phidiff_mrad_vs_halfstrip_" + csc_type[i];
    PhiDiffVsStripCsc[i][1] =
      new TH2F(v6.c_str(), u6.c_str(), MAX_HS[i], 0., MAX_HS[i], 100, PDMIN, PDMAX);

    string u7 = "CLCT keystrip, " + csc_type[i];
    string v7 = "o_CLCT_keystrip_" + csc_type[i];
    KeyStripCsc[i]   = new TH1F(v7.c_str(), u7.c_str(), MAX_DS, 0., MAX_DS);

    string u8 = "Pattern number, " + csc_type[i];
    string v81 = "o_Pattern_number_1_" + csc_type[i];
    string v82 = "o_Pattern_number_2_" + csc_type[i];
    PatternCsc[i][0] = new TH1F(v81.c_str(), u8.c_str(),  9, -4.5, 4.5);
    PatternCsc[i][1] = new TH1F(v82.c_str(), u8.c_str(),  9, -4.5, 4.5);
  }

  for (Int_t i = 0; i < 9; i++) {
    sprintf(histtitle, "#phi_rec-#phi_sim, bend = %d", i-4);
    sprintf(histname, "o_phidiff_bend_equal_%d", i-4);
    PhiDiffPattern[i] = new TH1F(histname,histtitle, 100, PDMIN, PDMAX);
  }




  bookedMuSimHitsVsMuDigis = true;
}
void CSCTFStubReader::deleteMuSimHitsVsMuDigis()
{

}
//---------------
// ROOT settings
//---------------
void CSCTFStubReader::setRootStyle() {
  gROOT->SetStyle("Plain");
  gStyle->SetFillColor(0);
  gStyle->SetOptDate();
  gStyle->SetOptStat(111110);
  gStyle->SetOptFit(1111);
  gStyle->SetPadTickX(1);
  gStyle->SetPadTickY(1);
  gStyle->SetMarkerSize(0.5);
  gStyle->SetMarkerStyle(8);
  gStyle->SetGridStyle(3);
  gStyle->SetPaperSize(TStyle::kA4);
  gStyle->SetStatW(0.25); // width of statistics box; default is 0.19
  gStyle->SetStatH(0.10); // height of statistics box; default is 0.1
  gStyle->SetStatFormat("6.4g");  // leave default format for now
  gStyle->SetTitleSize(0.055, "");   // size for pad title; default is 0.02
  // Really big; useful for talks.
  //gStyle->SetTitleSize(0.1, "");   // size for pad title; default is 0.02
  gStyle->SetLabelSize(0.05, "XYZ"); // size for axis labels; default is 0.04
  gStyle->SetStatFontSize(0.06);     // size for stat. box
  gStyle->SetTitleFont(32, "XYZ"); // times-bold-italic font (p. 153) for axes
  gStyle->SetTitleFont(32, "");    // same for pad title
  gStyle->SetLabelFont(32, "XYZ"); // same for axis labels
  gStyle->SetStatFont(32);         // same for stat. box
  gStyle->SetLabelOffset(0.006, "Y"); // default is 0.005
}

// Returns halfstrips-per-radian for different CSC types
Double_t CSCTFStubReader::getHsPerRad(const Int_t csctype) {

  Int_t nchambers;
  if (csctype == 4 || csctype == 6 || csctype == 8) // inner ring of stations 2, 3, and 4
    nchambers = 18;
  else
    nchambers = 36;

  return (nchambers*MAX_HS[csctype]/TWOPI);
}
void CSCTFStubReader::drawALCTHistos() {
  TCanvas *c1 = new TCanvas("c1", "", 0, 0, 500, 640);
  TPostScript *ps = new TPostScript("alcts.ps", 111);

  TPad *pad[MAXPAGES];
  for (int i_page = 0; i_page < MAXPAGES; i_page++) {
    pad[i_page] = new TPad("", "", .05, .05, .93, .93);
  }

  int page = 1;
  TText t;
  t.SetTextFont(32);
  t.SetTextSize(0.025);
  char pagenum[6];
  TPaveLabel *title;

  ps->NewPage();
  c1->Clear();  c1->cd(0);
  title = new TPaveLabel(0.1, 0.94, 0.9, 0.98, "Number of ALCTs");
  title->SetFillColor(10);  title->Draw();
  sprintf(pagenum, "- %d -", page);  t.DrawText(0.9, 0.02, pagenum);
  gStyle->SetOptStat(111110);
  pad[page]->Draw();
  pad[page]->Divide(1,2);
  pad[page]->cd(1);  hAlctPerEvent->Draw();
  for (int i = 0; i < CSC_TYPES; i++) {
    hAlctPerCSC->GetXaxis()->SetBinLabel(i+1, csc_type[i].c_str());
  }
  // Should be multiplied by 40/nevents to convert to MHz
  pad[page]->cd(2);  hAlctPerCSC->Draw();
  page++;  c1->Update();

  ps->NewPage();
  c1->Clear();  c1->cd(0);
  title = new TPaveLabel(0.1, 0.94, 0.9, 0.98, "ALCT quantities");
  title->SetFillColor(10);  title->Draw();
  sprintf(pagenum, "- %d -", page);  t.DrawText(0.9, 0.02, pagenum);
  gStyle->SetOptStat(111110);
  pad[page]->Draw();
  pad[page]->Divide(2,3);
  pad[page]->cd(1);  hAlctValid->Draw();
  pad[page]->cd(2);  hAlctQuality->Draw();
  pad[page]->cd(3);  hAlctAccel->Draw();
  pad[page]->cd(4);  hAlctCollis->Draw();
  pad[page]->cd(5);  hAlctKeyGroup->Draw();
  pad[page]->cd(6);  hAlctBXN->Draw();
  page++;  c1->Update();

  ps->Close();
}

void CSCTFStubReader::drawCLCTHistos() {
  TCanvas *c1 = new TCanvas("c1", "", 0, 0, 500, 640);
  TPostScript *ps = new TPostScript("clcts.ps", 111);

  TPad *pad[MAXPAGES];
  for (int i_page = 0; i_page < MAXPAGES; i_page++) {
    pad[i_page] = new TPad("", "", .05, .05, .93, .93);
  }

  int page = 1;
  TText t;
  t.SetTextFont(32);
  t.SetTextSize(0.025);
  char pagenum[6];
  TPaveLabel *title;

  ps->NewPage();
  c1->Clear();  c1->cd(0);
  title = new TPaveLabel(0.1, 0.94, 0.9, 0.98, "Number of CLCTs");
  title->SetFillColor(10);  title->Draw();
  sprintf(pagenum, "- %d -", page);  t.DrawText(0.9, 0.02, pagenum);
  gStyle->SetOptStat(111110);
  pad[page]->Draw();
  pad[page]->Divide(1,2);
  pad[page]->cd(1);  hClctPerEvent->Draw();
  for (int i = 0; i < CSC_TYPES; i++) {
    hClctPerCSC->GetXaxis()->SetBinLabel(i+1, csc_type[i].c_str());
  }
  // Should be multiplied by 40/nevents to convert to MHz
  pad[page]->cd(2);  hClctPerCSC->Draw();
  page++;  c1->Update();

  ps->NewPage();
  c1->Clear();  c1->cd(0);
  title = new TPaveLabel(0.1, 0.94, 0.9, 0.98, "CLCT quantities");
  title->SetFillColor(10);  title->Draw();
  sprintf(pagenum, "- %d -", page);  t.DrawText(0.9, 0.02, pagenum);
  pad[page]->Draw();
  pad[page]->Divide(2,3);
  pad[page]->cd(1);  hClctValid->Draw();
  pad[page]->cd(2);  hClctQuality->Draw();
  pad[page]->cd(3);  hClctSign->Draw();
  TH1F* hClctPatternTot = (TH1F*)hClctPattern[0]->Clone();
  hClctPatternTot->SetTitle("CLCT pattern #");
  hClctPatternTot->Add(hClctPattern[0], hClctPattern[1], 1., 1.);
  pad[page]->cd(4);  hClctPatternTot->Draw();
  hClctPattern[0]->SetLineStyle(2);  hClctPattern[0]->Draw("same");
  hClctPattern[1]->SetLineStyle(3);  hClctPattern[1]->Draw("same");
  pad[page]->cd(5);  hClctCFEB->Draw();
  pad[page]->cd(6);  hClctBXN->Draw();
  page++;  c1->Update();

  ps->NewPage();
  c1->Clear();  c1->cd(0);
  title = new TPaveLabel(0.1, 0.94, 0.9, 0.98, "CLCT quantities");
  title->SetFillColor(10);  title->Draw();
  sprintf(pagenum, "- %d -", page);  t.DrawText(0.9, 0.02, pagenum);
  pad[page]->Draw();
  pad[page]->Divide(1,3);
  pad[page]->cd(1);  hClctStripType->Draw();
  pad[page]->cd(2);  hClctKeyStrip[0]->Draw();
  pad[page]->cd(3);  hClctKeyStrip[1]->Draw();
  page++;  c1->Update();

  ps->NewPage();
  c1->Clear();  c1->cd(0);
  title = new TPaveLabel(0.1, 0.94, 0.9, 0.98, "CLCT halfstrip pattern types");
  title->SetFillColor(10);  title->Draw();
  sprintf(pagenum, "- %d -", page);  t.DrawText(0.9, 0.02, pagenum);
  gStyle->SetOptStat(110);
  pad[page]->Draw();
  pad[page]->Divide(2,5);
  for (int idh = 0; idh < CSC_TYPES-1; idh++) {
    pad[page]->cd(idh+1);
    hClctPatternCsc[idh][1]->GetXaxis()->SetTitle("Pattern number");
    hClctPatternCsc[idh][1]->GetYaxis()->SetTitle("Number of LCTs");
    hClctPatternCsc[idh][1]->Draw();
  }
  page++;  c1->Update();

  ps->NewPage();
  c1->Clear();  c1->cd(0);
  title = new TPaveLabel(0.1, 0.94, 0.9, 0.98, "CLCT distrip pattern types");
  title->SetFillColor(10);  title->Draw();
  sprintf(pagenum, "- %d -", page);  t.DrawText(0.9, 0.02, pagenum);
  pad[page]->Draw();
  pad[page]->Divide(2,5);
  for (int idh = 0; idh < CSC_TYPES-1; idh++) {
    pad[page]->cd(idh+1);
    hClctPatternCsc[idh][0]->GetXaxis()->SetTitle("Pattern number");
    hClctPatternCsc[idh][0]->GetYaxis()->SetTitle("Number of LCTs");
    hClctPatternCsc[idh][0]->Draw();
  }
  page++;  c1->Update();

  ps->NewPage();
  c1->Clear();  c1->cd(0);
  title = new TPaveLabel(0.1, 0.94, 0.9, 0.98,
			 "CLCT keystrip, distrip patterns only");
  title->SetFillColor(10);  title->Draw();
  sprintf(pagenum, "- %d -", page);  t.DrawText(0.9, 0.02, pagenum);
  pad[page]->Draw();
  pad[page]->Divide(2,5);
  for (int idh = 0; idh < CSC_TYPES-1; idh++) {
    pad[page]->cd(idh+1);
    hClctKeyStripCsc[idh]->GetXaxis()->SetTitle("Key distrip");
    hClctKeyStripCsc[idh]->GetXaxis()->SetTitleOffset(1.2);
    hClctKeyStripCsc[idh]->GetYaxis()->SetTitle("Number of LCTs");
    hClctKeyStripCsc[idh]->Draw();
  }
  page++;  c1->Update();

  ps->Close();
}

void CSCTFStubReader::drawMuSimHitsVsMuDigis() {

  TCanvas *c1 = new TCanvas("c1", "", 0, 0, 600, 768);
  //  TPostScript *ps = new TPostScript("csc_resolution.ps", 111);

  TPad *pad[MAXPAGES];
  for (Int_t i_page = 0; i_page < MAXPAGES; i_page++) {
    pad[i_page] = new TPad("", "", .05, .05, .93, .93);
  }

  Int_t page = 1;
  TText t;
  t.SetTextFont(32);
  t.SetTextSize(0.025);
  Char_t pagenum[6];
  TPaveLabel *title;


  c1->Clear();  c1->cd(0);
  title = new TPaveLabel(0.1, 0.94, 0.9, 0.98, "Numbers of LCTs");
  title->SetFillColor(10);  title->Draw();
  sprintf(pagenum, "- %d -", page);  t.DrawText(0.9, 0.02, pagenum);
  pad[page]->Draw();
  pad[page]->Divide(2,3);
  pad[page]->cd(1);  hAlctPerEvent->Draw();
  pad[page]->cd(2);  hClctPerEvent->Draw();
  for (Int_t i = 0; i < CSC_TYPES; i++) {
    //TotLctPerCSC->GetXaxis()->SetBinLabel(i+1, csc_type[i].c_str());
    //CorrLctPerCSC->GetXaxis()->SetBinLabel(i+1, csc_type[i].c_str());
    hAlctPerCSC->GetXaxis()->SetBinLabel(i+1, csc_type[i].c_str());
    hClctPerCSC->GetXaxis()->SetBinLabel(i+1, csc_type[i].c_str());
  }
  // Should be multiplied by 40/nevents to convert to MHz
  pad[page]->cd(3);  hAlctPerCSC->Draw();
  pad[page]->cd(4);  hClctPerCSC->Draw();
  //  pad[page]->cd(5);  TotLctPerCSC->Draw();
  //  pad[page]->cd(6);  CorrLctPerCSC->Draw();
  c1->Update();
  c1->Print("o_Numbers_of_LCTs.png");
  c1->Print("csc_resolution.ps(");
  page++;


  c1->Clear();  c1->cd(0);
  title = new TPaveLabel(0.1, 0.94, 0.9, 0.98, "LCT geometry");
  title->SetFillColor(10);  title->Draw();
  sprintf(pagenum, "- %d -", page);  t.DrawText(0.9, 0.02, pagenum);
  gStyle->SetOptStat(110110);
  pad[page]->Draw();
  pad[page]->Divide(2,4);
  pad[page]->cd(1);  hLctMPCEndcap->Draw();
  pad[page]->cd(2);  hLctMPCStation->Draw();
  pad[page]->cd(3);  hLctMPCSector->Draw();
  //  pad[page]->cd(4);  LctMPCSubsector->Draw();
  pad[page]->cd(4);  hLctMPCRing->Draw();
  for (Int_t istation = 0; istation < MAX_STATIONS; istation++) {
    pad[page]->cd(istation+5);  hLctMPCChamber[istation]->Draw();
  }
  c1->Update();
  c1->Print("o_LCT_geometry.png");
  c1->Print("csc_resolution.ps");
  page++;


  //  ps->NewPage();
  c1->Clear();  c1->cd(0);
  title = new TPaveLabel(0.1, 0.94, 0.9, 0.98,
			 "Number of LCTs as a function of eta");
  title->SetFillColor(10);  title->Draw();
  sprintf(pagenum, "- %d -", page);  t.DrawText(0.9, 0.02, pagenum);
  pad[page]->Draw();
  pad[page]->Divide(2,3);
  gStyle->SetOptStat(110110);
  TH1F* LctVsEtaTot[2];
  LctVsEtaTot[0] = (TH1F*)LctVsEta[0][0]->Clone();
  LctVsEtaTot[0]->SetTitle("LCTs vs eta, all stations");
  for (Int_t istation = 0; istation < MAX_STATIONS; istation++) {
    LctVsEta[istation][0]->GetXaxis()->SetTitleOffset(1.2);
    LctVsEta[istation][0]->GetYaxis()->SetTitleOffset(1.5);
    LctVsEta[istation][0]->GetXaxis()->SetTitle("#eta");
    LctVsEta[istation][0]->GetYaxis()->SetTitle("Number of LCTs");
    pad[page]->cd(istation+3);  LctVsEta[istation][0]->Draw();
    if (istation > 0) LctVsEtaTot[0]->Add(LctVsEta[istation][0], 1.);
  }
  LctVsEtaTot[0]->GetXaxis()->SetTitleOffset(1.2);
  LctVsEtaTot[0]->GetYaxis()->SetTitleOffset(1.5);
  LctVsEtaTot[0]->GetXaxis()->SetTitle("#eta");
  LctVsEtaTot[0]->GetYaxis()->SetTitle("Number of LCTs");
  pad[page]->cd(1);  LctVsEtaTot[0]->Draw();
  c1->Update();
  c1->Print("o_Number_of_LCTs_as_a_function_of_eta.png");
  c1->Print("csc_resolution.ps");
  page++;

//  ps->NewPage();
//  c1->Clear();  c1->cd(0);
//  title = new TPaveLabel(0.1, 0.94, 0.9, 0.98,
//			 "Number of LCTs as a function of eta bin");
//  title->SetFillColor(10);  title->Draw();
//  sprintf(pagenum, "- %d -", page);  t.DrawText(0.9, 0.02, pagenum);
//  pad[page]->Draw();
//  pad[page]->Divide(2,3);
//  gStyle->SetOptStat(110110);
//  LctVsEtaTot[1] = (TH1F*)LctVsEta[0][1]->Clone();
//  LctVsEtaTot[1]->SetTitle("LCTs vs eta bin, all stations");
//  for (Int_t istation = 0; istation < MAX_STATIONS; istation++) {
//    LctVsEta[istation][1]->GetXaxis()->SetTitleOffset(1.2);
//    LctVsEta[istation][1]->GetYaxis()->SetTitleOffset(1.5);
//    LctVsEta[istation][1]->GetXaxis()->SetTitle("#eta bin");
//    LctVsEta[istation][1]->GetYaxis()->SetTitle("Number of LCTs");
//    pad[page]->cd(istation+3);  LctVsEta[istation][1]->Draw();
//    if (istation > 0) LctVsEtaTot[1]->Add(LctVsEta[istation][1], 1.);
//  }
//  LctVsEtaTot[1]->GetXaxis()->SetTitleOffset(1.2);
//  LctVsEtaTot[1]->GetYaxis()->SetTitleOffset(1.5);
//  LctVsEtaTot[1]->GetXaxis()->SetTitle("#eta bin");
//  LctVsEtaTot[1]->GetYaxis()->SetTitle("Number of LCTs");
//  pad[page]->cd(1);  LctVsEtaTot[1]->Draw();
//    c1->Update();page++;c1->Print("o_Number_of_LCTs_as_a_function_of_eta_bin.png");
//
//  c1->Clear();  c1->cd(0);
//  title = new TPaveLabel(0.1, 0.94, 0.9, 0.98,
//			 "Number of LCTs as a function of #eta bin");
//  title->SetFillColor(10);  title->Draw();
//  sprintf(pagenum, "- %d -", page);  t.DrawText(0.9, 0.02, pagenum);
//  gStyle->SetOptStat(110);
//  pad[page]->Draw();
//  pad[page]->Divide(2,5);
//  for (Int_t idh = 0; idh < CSC_TYPES; idh++) {
//    pad[page]->cd(idh+1);  LctVsEtaCsc[idh]->Draw();
//  }
//    c1->Update();page++;c1->Print("o_Number_of_LCTs_as_a_function_of__eta_bin.png");

//  ps->NewPage();
  c1->Clear();  c1->cd(0);
  title = new TPaveLabel(0.1, 0.94, 0.9, 0.98,
			 "Number of LCTs as a function of phi");
  title->SetFillColor(10);  title->Draw();
  sprintf(pagenum, "- %d -", page);  t.DrawText(0.9, 0.02, pagenum);
  gStyle->SetOptStat(110);
  pad[page]->Draw();
  pad[page]->Divide(2,3);
  TH1F* LctVsPhiTot = (TH1F*)LctVsPhi[0]->Clone();
  LctVsPhiTot->SetTitle("LCTs vs phi, all stations");
  for (Int_t istation = 0; istation < MAX_STATIONS; istation++) {
    LctVsPhi[istation]->GetXaxis()->SetTitleOffset(1.2);
    LctVsPhi[istation]->GetYaxis()->SetTitleOffset(1.4);
    LctVsPhi[istation]->GetXaxis()->SetTitle("#phi");
    LctVsPhi[istation]->GetYaxis()->SetTitle("Number of LCTs");
    pad[page]->cd(istation+3);  LctVsPhi[istation]->Draw();
    if (istation > 0) LctVsPhiTot->Add(LctVsPhi[istation], 1.);
  }
  LctVsPhiTot->GetXaxis()->SetTitleOffset(1.2);
  LctVsPhiTot->GetYaxis()->SetTitleOffset(1.5);
  LctVsPhiTot->GetXaxis()->SetTitle("#phi");
  LctVsPhiTot->GetYaxis()->SetTitle("Number of LCTs");
  pad[page]->cd(1);  LctVsPhiTot->Draw();
  c1->Update();
  c1->Print("o_Number_of_LCTs_as_a_function_of_phi.png");
  c1->Print("csc_resolution.ps");
  page++;


  c1->Clear();  c1->cd(0);
  title = new TPaveLabel(0.1, 0.94, 0.9, 0.98, "ALCTs");
  title->SetFillColor(10);  title->Draw();
  sprintf(pagenum, "- %d -", page);  t.DrawText(0.9, 0.02, pagenum);
  gStyle->SetOptStat(111110);
  pad[page]->Draw();
  pad[page]->Divide(2,3);
  pad[page]->cd(1);  hAlctValid->Draw();
  pad[page]->cd(2);  hAlctQuality->Draw();
  pad[page]->cd(3);  hAlctAccel->Draw();
  pad[page]->cd(4);  hAlctKeyGroup->Draw();
  pad[page]->cd(5);  hAlctBXN->Draw();
  c1->Update();
  c1->Print("o_ALCTs.png");
  c1->Print("csc_resolution.ps");
  page++;


  c1->Clear();  c1->cd(0);
  title = new TPaveLabel(0.1, 0.94, 0.9, 0.98, "CLCTs");
  title->SetFillColor(10);  title->Draw();
  sprintf(pagenum, "- %d -", page);  t.DrawText(0.9, 0.02, pagenum);
  pad[page]->Draw();
  pad[page]->Divide(2,3);
  pad[page]->cd(1);  hClctValid->Draw();
  pad[page]->cd(2);  hClctQuality->Draw();
  pad[page]->cd(3);  hClctSign->Draw();
  TH1F* ClctPatternTot = (TH1F*)hClctPattern[0]->Clone();
  ClctPatternTot->SetTitle("CLCT pattern");
  ClctPatternTot->Add(hClctPattern[0], hClctPattern[1], 1., 1.);
  pad[page]->cd(4);  ClctPatternTot->Draw();
  hClctPattern[0]->SetLineStyle(2);  hClctPattern[0]->Draw("same");
  hClctPattern[1]->SetLineStyle(3);  hClctPattern[1]->Draw("same");
  pad[page]->cd(5);  hClctBXN->Draw();
  //  pad[page]->cd(6);  ClctBendAngle->Draw();
  c1->Update();
  c1->Print("o_CLCTs_1.png");
  c1->Print("csc_resolution.ps");
  page++;

  c1->Clear();  c1->cd(0);
  title = new TPaveLabel(0.1, 0.94, 0.9, 0.98, "CLCTs");
  title->SetFillColor(10);  title->Draw();
  sprintf(pagenum, "- %d -", page);  t.DrawText(0.9, 0.02, pagenum);
  pad[page]->Draw();
  pad[page]->Divide(1,3);
  pad[page]->cd(1);  hClctStripType->Draw();
  pad[page]->cd(2);  hClctKeyStrip[0]->Draw();
  pad[page]->cd(3);  hClctKeyStrip[1]->Draw();
  c1->Update();
  c1->Print("o_CLCTs_2.png");
  c1->Print("csc_resolution.ps");
  page++;


  c1->Clear();  c1->cd(0);
  title = new TPaveLabel(0.1, 0.94, 0.9, 0.98, "LCTs");
  title->SetFillColor(10);  title->Draw();
  sprintf(pagenum, "- %d -", page);  t.DrawText(0.9, 0.02, pagenum);
  pad[page]->Draw();
  pad[page]->Divide(2,2);
  pad[page]->cd(1);  hLctMPCQuality->SetLabelSize(0.045, "Y"); hLctMPCQuality->Draw();
  pad[page]->cd(2);  hLctMPCBend->SetLabelSize(0.045, "Y"); hLctMPCBend->Draw();
  pad[page]->cd(3);  hLctMPCStripType->SetLabelSize(0.045, "Y"); hLctMPCStripType->Draw();
  pad[page]->cd(4);  hLctMPCBXN->SetLabelSize(0.045, "Y");     hLctMPCBXN->Draw();
  c1->Update();
  c1->Print("o_LCTs.png");
  c1->Print("csc_resolution.ps");
  page++;

  c1->Clear();  c1->cd(0);
  title = new TPaveLabel(0.1, 0.94, 0.9, 0.98, "#eta resolution");
  title->SetFillColor(10);  title->Draw();
  sprintf(pagenum, "- %d -", page);  t.DrawText(0.9, 0.02, pagenum);
  pad[page]->Draw();
  pad[page]->Divide(2,2);
  gStyle->SetStatX(1.00);  gStyle->SetStatY(0.65);
  pad[page]->cd(1);  EtaRecVsSim->SetMarkerSize(0.2);  EtaRecVsSim->Draw();
  gPad->Update();  gStyle->SetStatX(1.00);  gStyle->SetStatY(0.995);
  EtaDiff[0]->GetXaxis()->SetNdivisions(505); // twice fewer divisions
  pad[page]->cd(3);  EtaDiff[0]->Draw();  EtaDiff[0]->Fit("gaus","Q");
  pad[page]->cd(4);  EtaDiff[1]->Draw();  EtaDiff[1]->Fit("gaus","Q");
  c1->Update();
  c1->Print("o_eta_resolution.png");
  c1->Print("csc_resolution.ps");
  page++;



  c1->Clear();  c1->cd(0);
  title = new TPaveLabel(0.1, 0.94, 0.9, 0.98, "#eta_rec-#eta_sim");
  title->SetFillColor(10);  title->Draw();
  sprintf(pagenum, "- %d -", page);  t.DrawText(0.9, 0.02, pagenum);
  pad[page]->Draw();
  pad[page]->Divide(2,5);
  for (Int_t idh = 0; idh < CSC_TYPES; idh++) {
    pad[page]->cd(idh+1);  EtaDiffCsc[idh][0]->Draw();
    if (EtaDiffCsc[idh][0]->GetEntries() > 1)
      EtaDiffCsc[idh][0]->Fit("gaus","Q");
  }
  c1->Update();
  c1->Print("o_eta_rec-eta_sim.png"); c1->Print("csc_resolution.ps");page++;

  //
  c1->Clear();  c1->cd(0);
  title = new TPaveLabel(0.1, 0.94, 0.9, 0.98,
			 "#eta_rec-#eta_sim, halfstrips only");
  title->SetFillColor(10);  title->Draw();
  sprintf(pagenum, "- %d -", page);  t.DrawText(0.9, 0.02, pagenum);
  pad[page]->Draw();
  pad[page]->Divide(2,5);
  for (Int_t idh = 0; idh < CSC_TYPES; idh++) {
    pad[page]->cd(idh+1);  EtaDiffCsc[idh][5]->Draw();
    if (EtaDiffCsc[idh][5]->GetEntries() > 1)
      EtaDiffCsc[idh][5]->Fit("gaus","Q");
  }
  c1->Update();
  c1->Print("o_eta_rec-eta_sim_halfstrips_only.png");
  c1->Print("csc_resolution.ps");
  page++;


  c1->Clear();  c1->cd(0);
  title = new TPaveLabel(0.1, 0.94, 0.9, 0.98,
			 "#eta_rec-#eta_sim, distrips only");
  title->SetFillColor(10);  title->Draw();
  sprintf(pagenum, "- %d -", page);  t.DrawText(0.9, 0.02, pagenum);
  pad[page]->Draw();
  pad[page]->Divide(2,5);
  for (Int_t idh = 0; idh < CSC_TYPES; idh++) {
    pad[page]->cd(idh+1);  EtaDiffCsc[idh][4]->Draw();
    if (EtaDiffCsc[idh][4]->GetEntries() > 1)
      EtaDiffCsc[idh][4]->Fit("gaus","Q");
  }
  c1->Update(); c1->Print("o_eta_rec-eta_sim_distrips_only.png");  c1->Print("csc_resolution.ps");page++;


  c1->Clear();  c1->cd(0);
  title = new TPaveLabel(0.1, 0.94, 0.9, 0.98, "#eta_rec-#eta_sim, endcap1");
  title->SetFillColor(10);  title->Draw();
  sprintf(pagenum, "- %d -", page);  t.DrawText(0.9, 0.02, pagenum);
  pad[page]->Draw();
  pad[page]->Divide(2,5);
  for (Int_t idh = 0; idh < CSC_TYPES; idh++) {
    pad[page]->cd(idh+1);  EtaDiffCsc[idh][1]->Draw();
    if (EtaDiffCsc[idh][1]->GetEntries() > 1)
      EtaDiffCsc[idh][1]->Fit("gaus","Q");
  }
  c1->Update(); c1->Print("o_eta_rec-eta_sim_endcap1.png");  c1->Print("csc_resolution.ps");page++;


  c1->Clear();  c1->cd(0);
  title = new TPaveLabel(0.1, 0.94, 0.9, 0.98, "#eta_rec-#eta_sim, endcap2");
  title->SetFillColor(10);  title->Draw();
  sprintf(pagenum, "- %d -", page);  t.DrawText(0.9, 0.02, pagenum);
  pad[page]->Draw();
  pad[page]->Divide(2,5);
  for (Int_t idh = 0; idh < CSC_TYPES; idh++) {
    pad[page]->cd(idh+1);  EtaDiffCsc[idh][2]->Draw();
    if (EtaDiffCsc[idh][2]->GetEntries() > 1)
      EtaDiffCsc[idh][2]->Fit("gaus","Q");
  }
  c1->Update(); c1->Print("o_eta_rec-eta_sim_endcap2.png");  c1->Print("csc_resolution.ps");page++;


  c1->Clear();  c1->cd(0);
  title = new TPaveLabel(0.1, 0.94, 0.9, 0.98,
			 "#eta_rec-#eta_sim, 0.0125 #eta bins");
  title->SetFillColor(10);  title->Draw();
  sprintf(pagenum, "- %d -", page);  t.DrawText(0.9, 0.02, pagenum);
  pad[page]->Draw();
  pad[page]->Divide(2,5);
  for (Int_t idh = 0; idh < CSC_TYPES; idh++) {
    pad[page]->cd(idh+1);  EtaDiffCsc[idh][3]->Draw();
    if (EtaDiffCsc[idh][3]->GetEntries() > 1)
      EtaDiffCsc[idh][3]->Fit("gaus","Q");
  }
  c1->Update(); c1->Print("o_eta_rec-eta_sim_0.0125etabins.png");  c1->Print("csc_resolution.ps");page++;


  c1->Clear();  c1->cd(0);
  title = new TPaveLabel(0.1, 0.94, 0.9, 0.98,
			 "#LT#eta_rec-#eta_sim#GT vs #eta_rec");
  title->SetFillColor(10);  title->Draw();
  sprintf(pagenum, "- %d -", page);  t.DrawText(0.9, 0.02, pagenum);
  gStyle->SetOptStat(0);
  pad[page]->Draw();
  pad[page]->Divide(2,2);
  TH1F *MeanEtaDiffVsEta[MAX_STATIONS];
  for (Int_t istation = 0; istation < MAX_STATIONS; istation++) {
    MeanEtaDiffVsEta[istation] = (TH1F*)EtaDiffVsEta[istation]->Clone();
    MeanEtaDiffVsEta[istation]->Divide(EtaDiffVsEta[istation],
				       LctVsEta[istation][0], 1., 1.);
    MeanEtaDiffVsEta[istation]->GetXaxis()->SetTitleOffset(1.2);
    MeanEtaDiffVsEta[istation]->GetXaxis()->SetTitle("#eta");
    MeanEtaDiffVsEta[istation]->SetMaximum(0.075);
    pad[page]->cd(istation+1);  MeanEtaDiffVsEta[istation]->Draw();
  }
  c1->Update(); c1->Print("o_mean_of_eta_rec-eta_sim_vs_eta_rec.png");  c1->Print("csc_resolution.ps");page++;

  c1->Clear();  c1->cd(0);
  title = new TPaveLabel(0.1, 0.94, 0.9, 0.98,
			 "#eta_rec-#eta_sim (0.0125 #eta bins) vs wiregroup");
  title->SetFillColor(10);  title->Draw();
  sprintf(pagenum, "- %d -", page);  t.DrawText(0.9, 0.02, pagenum);
  pad[page]->Draw();
  pad[page]->Divide(2,5);
  for (Int_t idh = 0; idh < CSC_TYPES; idh++) {
    pad[page]->cd(idh+1);  EtaDiffVsWireCsc[idh]->SetMarkerSize(0.2);
    EtaDiffVsWireCsc[idh]->GetXaxis()->SetTitle("Wiregroup");
    EtaDiffVsWireCsc[idh]->GetXaxis()->SetTitleOffset(1.2);
    EtaDiffVsWireCsc[idh]->GetYaxis()->SetTitle("#eta_rec-#eta_sim");
    EtaDiffVsWireCsc[idh]->Draw();
  }
  c1->Update(); c1->Print("o_eta_rec-eta_sim_0.0125etabins_vs_wiregroup.png");  c1->Print("csc_resolution.ps");page++;


  c1->Clear();  c1->cd(0);
  title = new TPaveLabel(0.1, 0.94, 0.9, 0.98,
			 "#LT#eta_rec-#eta_sim#GT vs #phi_rec");
  title->SetFillColor(10);  title->Draw();
  sprintf(pagenum, "- %d -", page);  t.DrawText(0.9, 0.02, pagenum);
  pad[page]->Draw();
  pad[page]->Divide(2,2);
  TH1F *MeanEtaDiffVsPhi[MAX_STATIONS];
  for (Int_t istation = 0; istation < MAX_STATIONS; istation++) {
    MeanEtaDiffVsPhi[istation] = (TH1F*)EtaDiffVsPhi[istation]->Clone();
    MeanEtaDiffVsPhi[istation]->Divide(EtaDiffVsPhi[istation],
				       LctVsPhi[istation], 1., 1.);
    MeanEtaDiffVsPhi[istation]->GetXaxis()->SetTitleOffset(1.2);
    MeanEtaDiffVsPhi[istation]->GetXaxis()->SetTitle("#phi (rad)");
    MeanEtaDiffVsPhi[istation]->SetMaximum(0.075);
    pad[page]->cd(istation+1);  MeanEtaDiffVsPhi[istation]->Draw();
  }
  c1->Update(); c1->Print("o_mean_of_eta_rec-eta_sim_vs_phi_rec.png");  c1->Print("csc_resolution.ps");page++;

  c1->Clear();  c1->cd(0);
  title = new TPaveLabel(0.1, 0.94, 0.9, 0.98,
			"#eta_rec-#eta_sim (0.0125 #eta bins) vs halfstrip #");
  title->SetFillColor(10);  title->Draw();
  sprintf(pagenum, "- %d -", page);  t.DrawText(0.9, 0.02, pagenum);
  pad[page]->Draw();
  pad[page]->Divide(2,5);
  for (Int_t idh = 0; idh < CSC_TYPES; idh++) {
    pad[page]->cd(idh+1);  EtaDiffVsStripCsc[idh][1]->SetMarkerSize(0.2);
    EtaDiffVsStripCsc[idh][1]->GetXaxis()->SetTitle("Halfstrip");
    EtaDiffVsStripCsc[idh][1]->GetXaxis()->SetTitleOffset(1.2);
    EtaDiffVsStripCsc[idh][1]->GetYaxis()->SetTitle("#eta_rec-#eta_sim");
    EtaDiffVsStripCsc[idh][1]->Draw();
  }
  c1->Update(); c1->Print("o_eta_rec-eta_sim_0.0125etabins_vs_halfstrip.png");  c1->Print("csc_resolution.ps");page++;

  c1->Clear();  c1->cd(0);
  title = new TPaveLabel(0.1, 0.94, 0.9, 0.98,
			 "#eta_rec-#eta_sim (0.0125 #eta bins) vs distrip #");
  title->SetFillColor(10);  title->Draw();
  sprintf(pagenum, "- %d -", page);  t.DrawText(0.9, 0.02, pagenum);
  pad[page]->Draw();
  pad[page]->Divide(2,5);
  for (Int_t idh = 0; idh < CSC_TYPES; idh++) {
    pad[page]->cd(idh+1);  EtaDiffVsStripCsc[idh][0]->SetMarkerSize(0.2);
    EtaDiffVsStripCsc[idh][0]->GetXaxis()->SetTitle("Distrip");
    EtaDiffVsStripCsc[idh][0]->GetXaxis()->SetTitleOffset(1.2);
    EtaDiffVsStripCsc[idh][0]->GetYaxis()->SetTitle("#eta_rec-#eta_sim");
    EtaDiffVsStripCsc[idh][0]->Draw();
  }
  c1->Update(); c1->Print("o_eta_rec-eta_sim_0.0125etabins_vs_distrip.png");  c1->Print("csc_resolution.ps");page++;

  c1->Clear();  c1->cd(0);
  title = new TPaveLabel(0.1, 0.94, 0.9, 0.98,
	      "#eta_rec-#eta_sim (0.0125 #eta bins) vs halfstrip #, endcap 1");
  title->SetFillColor(10);  title->Draw();
  sprintf(pagenum, "- %d -", page);  t.DrawText(0.9, 0.02, pagenum);
  pad[page]->Draw();
  pad[page]->Divide(2,5);
  for (Int_t idh = 0; idh < CSC_TYPES; idh++) {
    pad[page]->cd(idh+1);  EtaDiffVsStripCsc[idh][2]->SetMarkerSize(0.2);
    EtaDiffVsStripCsc[idh][2]->GetXaxis()->SetTitle("Halfstrip");
    EtaDiffVsStripCsc[idh][2]->GetXaxis()->SetTitleOffset(1.2);
    EtaDiffVsStripCsc[idh][2]->GetYaxis()->SetTitle("#eta_rec-#eta_sim");
    EtaDiffVsStripCsc[idh][2]->Draw();
  }
  c1->Update(); c1->Print("o_eta_rec-eta_sim_0.0125etabins_vs_halfstrip_endcap1.png");  c1->Print("csc_resolution.ps");page++;

  c1->Clear();  c1->cd(0);
  title = new TPaveLabel(0.1, 0.94, 0.9, 0.98,
	      "#eta_rec-#eta_sim (0.0125 #eta bins) vs halfstrip #, endcap 2");
  title->SetFillColor(10);  title->Draw();
  sprintf(pagenum, "- %d -", page);  t.DrawText(0.9, 0.02, pagenum);
  pad[page]->Draw();
  pad[page]->Divide(2,5);
  for (Int_t idh = 0; idh < CSC_TYPES; idh++) {
    pad[page]->cd(idh+1);  EtaDiffVsStripCsc[idh][3]->SetMarkerSize(0.2);
    EtaDiffVsStripCsc[idh][3]->GetXaxis()->SetTitle("Halfstrip");
    EtaDiffVsStripCsc[idh][3]->GetXaxis()->SetTitleOffset(1.2);
    EtaDiffVsStripCsc[idh][3]->GetYaxis()->SetTitle("#eta_rec-#eta_sim");
    EtaDiffVsStripCsc[idh][3]->Draw();
  }
  c1->Update(); c1->Print("o_eta_rec-eta_sim_0.0125etabins_vs_halfstrip_endcap2.png");  c1->Print("csc_resolution.ps");page++;


  c1->Clear();  c1->cd(0);
  title = new TPaveLabel(0.1, 0.94, 0.9, 0.98, "#phi resolution");
  title->SetFillColor(10);  title->Draw();
  sprintf(pagenum, "- %d -", page);  t.DrawText(0.9, 0.02, pagenum);
  gStyle->SetOptStat(111110);
  pad[page]->Draw();
  pad[page]->Divide(2,2);
  gStyle->SetStatX(1.00);  gStyle->SetStatY(0.65);
  pad[page]->cd(1);  PhiRecVsSim->SetMarkerSize(0.2);  PhiRecVsSim->Draw();
  gPad->Update();  gStyle->SetStatX(1.00);  gStyle->SetStatY(0.995);
  pad[page]->cd(3);  PhiDiff[0]->Draw();  PhiDiff[0]->Fit("gaus","Q");
  pad[page]->cd(4);  PhiDiff[1]->Draw();  PhiDiff[1]->Fit("gaus","Q");
  c1->Update(); c1->Print("o_phi_resolution.png");  c1->Print("csc_resolution.ps");page++;


  c1->Clear();  c1->cd(0);
  title = new TPaveLabel(0.1, 0.94, 0.9, 0.98, "#phi_rec-#phi_sim (mrad)");
  title->SetFillColor(10);  title->Draw();
  sprintf(pagenum, "- %d -", page);  t.DrawText(0.9, 0.02, pagenum);
  pad[page]->Draw();
  pad[page]->Divide(2,5);
  for (Int_t idh = 0; idh < CSC_TYPES; idh++) {
    pad[page]->cd(idh+1);  PhiDiffCsc[idh][0]->Draw();
    if (PhiDiffCsc[idh][0]->GetEntries() > 1)
      PhiDiffCsc[idh][0]->Fit("gaus","Q");
  }
  c1->Update(); c1->Print("o_phi_rec-phi_sim_mrad.png");  c1->Print("csc_resolution.ps");page++;


  c1->Clear();  c1->cd(0);
  title = new TPaveLabel(0.1, 0.94, 0.9, 0.98,
			 "#phi_rec-#phi_sim (mrad), halfstrips only");
  title->SetFillColor(10);  title->Draw();
  sprintf(pagenum, "- %d -", page);  t.DrawText(0.9, 0.02, pagenum);
  pad[page]->Draw();
  pad[page]->Divide(2,5);
  for (Int_t idh = 0; idh < CSC_TYPES; idh++) {
    pad[page]->cd(idh+1);  PhiDiffCsc[idh][5]->Draw();
    if (PhiDiffCsc[idh][5]->GetEntries() > 1)
      PhiDiffCsc[idh][5]->Fit("gaus","Q");
  }
  c1->Update(); c1->Print("o_phi_rec-phi_sim_mrad_halfstriponly.png");  c1->Print("csc_resolution.ps");page++;


  c1->Clear();  c1->cd(0);
  title = new TPaveLabel(0.1, 0.94, 0.9, 0.98,
			 "#phi_rec-#phi_sim (mrad), distrips only");
  title->SetFillColor(10);  title->Draw();
  sprintf(pagenum, "- %d -", page);  t.DrawText(0.9, 0.02, pagenum);
  pad[page]->Draw();
  pad[page]->Divide(2,5);
  for (Int_t idh = 0; idh < CSC_TYPES; idh++) {
    pad[page]->cd(idh+1);  PhiDiffCsc[idh][4]->Draw();
    if (PhiDiffCsc[idh][4]->GetEntries() > 1)
      PhiDiffCsc[idh][4]->Fit("gaus","Q");
  }
  c1->Update(); c1->Print("o_phi_rec-phi_sim_mrad_distriponly.png");  c1->Print("csc_resolution.ps");page++;


  c1->Clear();  c1->cd(0);
  title = new TPaveLabel(0.1, 0.94, 0.9, 0.98,
			 "#phi_rec-#phi_sim (mrad), endcap1");
  title->SetFillColor(10);  title->Draw();
  sprintf(pagenum, "- %d -", page);  t.DrawText(0.9, 0.02, pagenum);
  pad[page]->Draw();
  pad[page]->Divide(2,5);
  for (Int_t idh = 0; idh < CSC_TYPES; idh++) {
    pad[page]->cd(idh+1);  PhiDiffCsc[idh][1]->Draw();
    if (PhiDiffCsc[idh][1]->GetEntries() > 1)
      PhiDiffCsc[idh][1]->Fit("gaus","Q");
  }
  c1->Update(); c1->Print("o_phi_rec-phi_sim_mrad_endcap1.png");  c1->Print("csc_resolution.ps");page++;


  c1->Clear();  c1->cd(0);
  title = new TPaveLabel(0.1, 0.94, 0.9, 0.98,
			 "#phi_rec-#phi_sim (mrad), endcap2");
  title->SetFillColor(10);  title->Draw();
  sprintf(pagenum, "- %d -", page);  t.DrawText(0.9, 0.02, pagenum);
  pad[page]->Draw();
  pad[page]->Divide(2,5);
  for (Int_t idh = 0; idh < CSC_TYPES; idh++) {
    pad[page]->cd(idh+1);  PhiDiffCsc[idh][2]->Draw();
    if (PhiDiffCsc[idh][2]->GetEntries() > 1)
      PhiDiffCsc[idh][2]->Fit("gaus","Q");
  }
  c1->Update(); c1->Print("o_phi_rec-phi_sim_mrad_endcap2.png");  c1->Print("csc_resolution.ps");page++;


  c1->Clear();  c1->cd(0);
  title = new TPaveLabel(0.1, 0.94, 0.9, 0.98,
		       "#phi_rec-#phi_sim (mrad), halfstrips only, pattern 0");
  title->SetFillColor(10);  title->Draw();
  sprintf(pagenum, "- %d -", page);  t.DrawText(0.9, 0.02, pagenum);
  pad[page]->Draw();
  pad[page]->Divide(2,5);
  for (Int_t idh = 0; idh < CSC_TYPES; idh++) {
    pad[page]->cd(idh+1);  PhiDiffCsc[idh][6]->Draw();
    if (PhiDiffCsc[idh][6]->GetEntries() > 1)
      PhiDiffCsc[idh][6]->Fit("gaus","Q");
  }
  c1->Update(); c1->Print("o_phi_rec-phi_sim_mrad_halfstriponly_pattern0.png");  c1->Print("csc_resolution.ps");page++;


  c1->Clear();  c1->cd(0);
  title = new TPaveLabel(0.1, 0.94, 0.9, 0.98,
			 "#phi_rec-#phi_sim (0.26 mrad #phi bins)");
  title->SetFillColor(10);  title->Draw();
  sprintf(pagenum, "- %d -", page);  t.DrawText(0.9, 0.02, pagenum);
  pad[page]->Draw();
  pad[page]->Divide(2,5);
  for (Int_t idh = 0; idh < CSC_TYPES; idh++) {
    pad[page]->cd(idh+1);  PhiDiffCsc[idh][3]->Draw();
  }
  c1->Update(); c1->Print("o_phi_rec-phi_sim_0.26mrad_phibins.png");  c1->Print("csc_resolution.ps");page++;


  c1->Clear();  c1->cd(0);
  title = new TPaveLabel(0.1, 0.94, 0.9, 0.98,
		   "#phi_rec-#phi_sim (0.26 mrad #phi bins), halfstrips only");
  title->SetFillColor(10);  title->Draw();
  sprintf(pagenum, "- %d -", page);  t.DrawText(0.9, 0.02, pagenum);
  pad[page]->Draw();
  pad[page]->Divide(2,5);
  for (Int_t idh = 0; idh < CSC_TYPES; idh++) {
    pad[page]->cd(idh+1);  PhiDiffCsc[idh][8]->Draw();
  }
  c1->Update(); c1->Print("o_phi_rec-phi_sim_0.26mrad_phibins_halfstriponly.png");  c1->Print("csc_resolution.ps");page++;


  c1->Clear();  c1->cd(0);
  title = new TPaveLabel(0.1, 0.94, 0.9, 0.98,
		   "#phi_rec-#phi_sim (0.26 mrad #phi bins), distrips only");
  title->SetFillColor(10);  title->Draw();
  sprintf(pagenum, "- %d -", page);  t.DrawText(0.9, 0.02, pagenum);
  pad[page]->Draw();
  pad[page]->Divide(2,5);
  for (Int_t idh = 0; idh < CSC_TYPES; idh++) {
    pad[page]->cd(idh+1);  PhiDiffCsc[idh][7]->Draw();
  }
    c1->Update(); c1->Print("o_phi_rec-phi_sim_0.26mrad_phibins_distriponly.png");  c1->Print("csc_resolution.ps");page++;


  c1->Clear();  c1->cd(0);
  title = new TPaveLabel(0.1, 0.94, 0.9, 0.98,
			 "#LT#phi_rec-#phi_sim#GT (mrad) vs #eta_rec");
  title->SetFillColor(10);  title->Draw();
  sprintf(pagenum, "- %d -", page);  t.DrawText(0.9, 0.02, pagenum);
  gStyle->SetOptStat(0);
  pad[page]->Draw();
  pad[page]->Divide(2,2);
  TH1F *MeanPhiDiffVsEta[MAX_STATIONS];
  for (Int_t istation = 0; istation < MAX_STATIONS; istation++) {
    MeanPhiDiffVsEta[istation] = (TH1F*)PhiDiffVsEta[istation]->Clone();
    MeanPhiDiffVsEta[istation]->Divide(PhiDiffVsEta[istation],
				       LctVsEta[istation][0], 1., 1.);
    MeanPhiDiffVsEta[istation]->GetXaxis()->SetTitleOffset(1.2);
    MeanPhiDiffVsEta[istation]->GetYaxis()->SetTitleOffset(1.7);
    MeanPhiDiffVsEta[istation]->GetXaxis()->SetTitle("#eta");
    MeanPhiDiffVsEta[istation]->GetYaxis()->SetTitle("#LT#phi_rec-#phi_sim#GT (mrad)");
    MeanPhiDiffVsEta[istation]->SetMaximum(5.);
    pad[page]->cd(istation+1);  MeanPhiDiffVsEta[istation]->Draw();
  }
    c1->Update(); c1->Print("o_phi_rec-phi_sim_mrad_vs_eta_rec.png");  c1->Print("csc_resolution.ps");page++;

  c1->Clear();  c1->cd(0);
  title = new TPaveLabel(0.1, 0.94, 0.9, 0.98,
			 "#phi_rec-#phi_sim (mrad) vs wiregroup");
  title->SetFillColor(10);  title->Draw();
  sprintf(pagenum, "- %d -", page);  t.DrawText(0.9, 0.02, pagenum);
  gStyle->SetOptStat(0);
  pad[page]->Draw();
  pad[page]->Divide(2,5);
  for (Int_t idh = 0; idh < CSC_TYPES; idh++) {
    pad[page]->cd(idh+1);  PhiDiffVsWireCsc[idh]->SetMarkerSize(0.2);
    PhiDiffVsWireCsc[idh]->GetXaxis()->SetTitle("Wiregroup");
    PhiDiffVsWireCsc[idh]->GetXaxis()->SetTitleOffset(1.2);
    PhiDiffVsWireCsc[idh]->GetYaxis()->SetTitle("#phi_rec - #phi_sim");
    PhiDiffVsWireCsc[idh]->Draw();
  }
    c1->Update(); c1->Print("o_phi_rec-phi_sim_mrad_vs_wiregroup.png");  c1->Print("csc_resolution.ps");page++;


  c1->Clear();  c1->cd(0);
  title = new TPaveLabel(0.1, 0.94, 0.9, 0.98,
			 "#LT#phi_rec-#phi_sim#GT (mrad) vs #phi_rec");
  title->SetFillColor(10);  title->Draw();
  sprintf(pagenum, "- %d -", page);  t.DrawText(0.9, 0.02, pagenum);
  gStyle->SetOptStat(0);
  pad[page]->Draw();
  pad[page]->Divide(2,2);
  TH1F *MeanPhiDiffVsPhi[MAX_STATIONS];
  for (Int_t istation = 0; istation < MAX_STATIONS; istation++) {
    MeanPhiDiffVsPhi[istation] = (TH1F*)PhiDiffVsPhi[istation]->Clone();
    MeanPhiDiffVsPhi[istation]->Divide(PhiDiffVsPhi[istation],
				       LctVsPhi[istation], 1., 1.);
    MeanPhiDiffVsPhi[istation]->GetXaxis()->SetTitleOffset(1.2);
    MeanPhiDiffVsPhi[istation]->GetYaxis()->SetTitleOffset(1.7);
    MeanPhiDiffVsPhi[istation]->GetXaxis()->SetTitle("#phi");
    MeanPhiDiffVsPhi[istation]->GetYaxis()->SetTitle("#LT#phi_rec-#phi_sim#GT (mrad)");
    MeanPhiDiffVsPhi[istation]->SetMaximum(5.);
    pad[page]->cd(istation+1);  MeanPhiDiffVsPhi[istation]->Draw();
  }
    c1->Update(); c1->Print("o_phi_rec-phi_sim_mrad_vs_phi_rec.png");  c1->Print("csc_resolution.ps");page++;

  c1->Clear();  c1->cd(0);
  title = new TPaveLabel(0.1, 0.94, 0.9, 0.98,
			 "#phi_rec-#phi_sim (mrad) vs halfstrip #");
  title->SetFillColor(10);  title->Draw();
  sprintf(pagenum, "- %d -", page);  t.DrawText(0.9, 0.02, pagenum);
  gStyle->SetOptStat(0);
  pad[page]->Draw();
  pad[page]->Divide(2,5);
  for (Int_t idh = 0; idh < CSC_TYPES; idh++) {
    pad[page]->cd(idh+1);  PhiDiffVsStripCsc[idh][1]->SetMarkerSize(0.2);
    PhiDiffVsStripCsc[idh][1]->GetXaxis()->SetTitle("Halfstrip");
    PhiDiffVsStripCsc[idh][1]->GetXaxis()->SetTitleOffset(1.2);
    PhiDiffVsStripCsc[idh][1]->GetYaxis()->SetTitle("#phi_rec-#phi_sim (mrad)");
    PhiDiffVsStripCsc[idh][1]->Draw();
  }
    c1->Update(); c1->Print("o_phi_rec-phi_sim_mrad_vs_halfstrip.png");  c1->Print("csc_resolution.ps");page++;

  c1->Clear();  c1->cd(0);
  title = new TPaveLabel(0.1, 0.94, 0.9, 0.98,
			 "#phi_rec-#phi_sim (mrad) vs distrip #");
  title->SetFillColor(10);  title->Draw();
  sprintf(pagenum, "- %d -", page);  t.DrawText(0.9, 0.02, pagenum);
  pad[page]->Draw();
  pad[page]->Divide(2,5);
  for (Int_t idh = 0; idh < CSC_TYPES; idh++) {
    pad[page]->cd(idh+1);  PhiDiffVsStripCsc[idh][0]->SetMarkerSize(0.2);
    PhiDiffVsStripCsc[idh][0]->GetXaxis()->SetTitle("Distrip");
    PhiDiffVsStripCsc[idh][0]->GetXaxis()->SetTitleOffset(1.2);
    PhiDiffVsStripCsc[idh][0]->GetYaxis()->SetTitle("#phi_rec-#phi_sim (mrad)");
    PhiDiffVsStripCsc[idh][0]->Draw();
  }
    c1->Update(); c1->Print("o_phi_rec-phi_sim_mrad_vs_distrip.png");  c1->Print("csc_resolution.ps");page++;

  c1->Clear();  c1->cd(0);
  title = new TPaveLabel(0.1, 0.94, 0.9, 0.98, "CLCT halfstrip pattern types");
  title->SetFillColor(10);  title->Draw();
  sprintf(pagenum, "- %d -", page);  t.DrawText(0.9, 0.02, pagenum);
  gStyle->SetOptStat(110);
  pad[page]->Draw();
  pad[page]->Divide(2,5);
  for (Int_t idh = 0; idh < CSC_TYPES; idh++) {
    pad[page]->cd(idh+1);
    PatternCsc[idh][1]->GetXaxis()->SetTitle("Pattern number");
    PatternCsc[idh][1]->GetYaxis()->SetTitle("Number of LCTs");
    PatternCsc[idh][1]->Draw();
  }
    c1->Update(); c1->Print("o_clct_halfstrip_pattern_types.png");  c1->Print("csc_resolution.ps");page++;

  c1->Clear();  c1->cd(0);
  title = new TPaveLabel(0.1, 0.94, 0.9, 0.98, "CLCT distrip pattern types");
  title->SetFillColor(10);  title->Draw();
  sprintf(pagenum, "- %d -", page);  t.DrawText(0.9, 0.02, pagenum);
  pad[page]->Draw();
  pad[page]->Divide(2,5);
  for (Int_t idh = 0; idh < CSC_TYPES; idh++) {
    pad[page]->cd(idh+1);
    PatternCsc[idh][0]->GetXaxis()->SetTitle("Pattern number");
    PatternCsc[idh][0]->GetYaxis()->SetTitle("Number of LCTs");
    PatternCsc[idh][0]->Draw();
  }
    c1->Update(); c1->Print("o_clct_distrip_pattern_types.png");  c1->Print("csc_resolution.ps");page++;


  c1->Clear();  c1->cd(0);
  title = new TPaveLabel(0.1, 0.94, 0.9, 0.98, "ME1/A: pattern types");
  title->SetFillColor(10);  title->Draw();
  sprintf(pagenum, "- %d -", page);  t.DrawText(0.9, 0.02, pagenum);
  gStyle->SetOptStat(0);
  pad[page]->Draw();
  pad[page]->Divide(1,2);
  TH1F *PatternCscTot = (TH1F*)PatternCsc[3][0]->Clone();
  PatternCscTot->SetTitle("ME1/A pattern types");
  PatternCscTot->GetXaxis()->SetTitle("Pattern number");
  PatternCscTot->Add(PatternCsc[3][1], 1.);
  pad[page]->cd(1);  PatternCscTot->GetYaxis()->SetTitleOffset(1.2);
  PatternCscTot->SetStats(false);     PatternCscTot->Draw();
  PatternCsc[3][1]->SetLineStyle(2);
  PatternCsc[3][1]->SetStats(false);  PatternCsc[3][1]->Draw("same");
  PatternCsc[3][0]->SetLineStyle(3);
  PatternCsc[3][0]->SetStats(false);  PatternCsc[3][0]->Draw("same");

  TH1F *PatternCscRat = (TH1F*)PatternCsc[3][0]->Clone();
  PatternCscRat->SetTitle("Distrips / Total");
  PatternCscRat->Divide(PatternCsc[3][0], PatternCscTot);
  PatternCscRat->GetXaxis()->SetTitleOffset(1.2);
  PatternCscRat->GetYaxis()->SetTitleOffset(1.2);
  PatternCscRat->GetXaxis()->SetTitle("Pattern number");
  PatternCscRat->GetYaxis()->SetTitle("Distrips / Total");
  PatternCscRat->SetMaximum(1.05);
  pad[page]->cd(2);  PatternCscRat->SetLineStyle(1);
  PatternCscRat->SetStats(false);  PatternCscRat->Draw();
    c1->Update(); c1->Print("o_ME1A_pattern_types.png");  c1->Print("csc_resolution.ps");page++;

  c1->Clear();  c1->cd(0);
  title = new TPaveLabel(0.1, 0.94, 0.9, 0.98,
			 "CLCT keystrip, distrip patterns only");
  title->SetFillColor(10);  title->Draw();
  sprintf(pagenum, "- %d -", page);  t.DrawText(0.9, 0.02, pagenum);
  pad[page]->Draw();
  pad[page]->Divide(2,5);
  for (Int_t idh = 0; idh < CSC_TYPES; idh++) {
    pad[page]->cd(idh+1);
    KeyStripCsc[idh]->GetXaxis()->SetTitle("Key distrip");
    KeyStripCsc[idh]->GetXaxis()->SetTitleOffset(1.2);
    KeyStripCsc[idh]->GetYaxis()->SetTitle("Number of LCTs");
    KeyStripCsc[idh]->Draw();
  }
    c1->Update(); c1->Print("o_clct_keystrip_distrip_patterns_only.png");  c1->Print("csc_resolution.ps");page++;

  c1->Clear();  c1->cd(0);
  title = new TPaveLabel(0.1, 0.94, 0.9, 0.98,
		     "#phi_rec-#phi_sim, halfstrips only, different patterns");
  title->SetFillColor(10);  title->Draw();
  sprintf(pagenum, "- %d -", page);  t.DrawText(0.9, 0.02, pagenum);
  gStyle->SetOptStat(111110);
  pad[page]->Draw();
  pad[page]->Divide(3,3);
  for (Int_t idh = 0; idh < 9; idh++) {
    PhiDiffPattern[idh]->GetXaxis()->SetTitle("Halfstrip");
    PhiDiffPattern[idh]->GetXaxis()->SetTitleOffset(1.2);
    pad[page]->cd(idh+1);  PhiDiffPattern[idh]->Draw();
    if (PhiDiffPattern[idh]->GetEntries() > 1)
      PhiDiffPattern[idh]->Fit("gaus","Q");
  }
  c1->Update(); c1->Print("o_phi_rec-phi_sim_halfstripsonly_differentpatterns.png");  c1->Print("csc_resolution.ps)");page++;
  //  ps->Close();

}
void CSCTFStubReader::endJob()
{
  setRootStyle();

  if (bookedMuSimHitsVsMuDigis)
    {
      drawMuSimHitsVsMuDigis();
      drawALCTHistos();
      drawCLCTHistos();
    }
  fAnalysis->Write();
  if (bookedMuSimHitsVsMuDigis)
    {
      deleteMuSimHitsVsMuDigis();
    }

  delete fAnalysis;
}
