#include "Calibration/IsolatedParticles/plugins/ElectronStudy.h"
#include "Calibration/IsolatedParticles/interface/CaloPropagateTrack.h"
#include "Calibration/IsolatedParticles/interface/eECALMatrix.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"
#include "Geometry/CaloTopology/interface/CaloSubdetectorTopology.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include <algorithm>

ElectronStudy::ElectronStudy(const edm::ParameterSet& ps) {

  sourceLabel = ps.getUntrackedParameter<std::string>("SourceLabel","generatorSmeared");
  g4Label = ps.getUntrackedParameter<std::string>("ModuleLabel","g4SimHits");
  hitLabEB= ps.getUntrackedParameter<std::string>("EBCollection","EcalHitsEB");
  hitLabEE= ps.getUntrackedParameter<std::string>("EECollection","EcalHitsEE");


  tok_EBhit_  = consumes<edm::PCaloHitContainer>(edm::InputTag(g4Label,hitLabEB));
  tok_EEhit_  = consumes<edm::PCaloHitContainer>(edm::InputTag(g4Label,hitLabEE));
  tok_simTk_  = consumes<edm::SimTrackContainer>(edm::InputTag(g4Label));
  tok_simVtx_ = consumes<edm::SimVertexContainer>(edm::InputTag(g4Label));

  hotZone = ps.getUntrackedParameter<int>("HotZone",0);
  verbose = ps.getUntrackedParameter<int>("Verbosity",0);
  edm::LogInfo("ElectronStudy") << "Module Label: " << g4Label << "   Hits: "
				<< hitLabEB << ", " << hitLabEE;

  double tempPBins[NPBins+1] = {   0.0,   10.0,   20.0,  40.0,  60.0,  
				   100.0,  500.0, 1000.0, 10000.0};
  double tempEta[NEtaBins+1] = {0.0, 1.2, 1.6, 3.0};

  for (int i=0; i<NPBins+1; i++)  pBins[i]   = tempPBins[i];
  for(int i=0; i<NEtaBins+1; i++) etaBins[i] = tempEta[i];

  edm::Service<TFileService> tfile;
  if ( !tfile.isAvailable() ) {
    edm::LogInfo("ElectronStudy") << "TFileService unavailable: no histograms";
    histos = false;
  } else {
    char  name[20], title[200], cpbin[30], cebin[30];
    histos = true;
    for (unsigned int i=0; i<NPBins+1; ++i) {
      if (i == 0) sprintf (cpbin, " All p");
      else        sprintf (cpbin, " p (%6.0f:%6.0f)", pBins[i-1], pBins[i]);
      for (unsigned int j=0; j<NEtaBins+1; ++j) {
	if (j == 0) sprintf (cebin, " All #eta");
	else        sprintf (cebin, " #eta (%4.1f:%4.1f)", etaBins[j-1], etaBins[j]);
	sprintf (name, "R1%d%d", i, j);
	sprintf (title,"E1/E9 for %s%s", cpbin, cebin);
	histoR1[i][j] = tfile->make<TH1F>(name, title, 100, 0., 2.);
	histoR1[i][j]->GetXaxis()->SetTitle(title);
	histoR1[i][j]->GetYaxis()->SetTitle("Tracks");
	sprintf (name, "R2%d%d", i, j);
	sprintf (title,"E1/E25 for %s%s", cpbin, cebin);
	histoR2[i][j] = tfile->make<TH1F>(name, title, 100, 0., 2.);
	histoR2[i][j]->GetXaxis()->SetTitle(title);
	histoR2[i][j]->GetYaxis()->SetTitle("Tracks");
	sprintf (name, "R3%d%d", i, j);
	sprintf (title,"E9/E25 for %s%s", cpbin, cebin);
	histoR3[i][j] = tfile->make<TH1F>(name, title, 100, 0., 2.);
	histoR3[i][j]->GetXaxis()->SetTitle(title);
	histoR3[i][j]->GetYaxis()->SetTitle("Tracks");
	sprintf (name, "E1x1%d%d", i, j);
	sprintf (title,"E1/P for %s%s", cpbin, cebin);
	histoE1x1[i][j] = tfile->make<TH1F>(name, title, 100, 0., 2.);
	histoE1x1[i][j]->GetXaxis()->SetTitle(title);
	histoE1x1[i][j]->GetYaxis()->SetTitle("Tracks");
	sprintf (name, "E3x3%d%d", i, j);
	sprintf (title,"E9/P for %s%s", cpbin, cebin);
	histoE3x3[i][j] = tfile->make<TH1F>(name, title, 100, 0., 2.);
	histoE3x3[i][j]->GetXaxis()->SetTitle(title);
	histoE3x3[i][j]->GetYaxis()->SetTitle("Tracks");
	sprintf (name, "E5x5%d%d", i, j);
	sprintf (title,"E25/P for %s%s", cpbin, cebin);
	histoE5x5[i][j] = tfile->make<TH1F>(name, title, 100, 0., 2.);
	histoE5x5[i][j]->GetXaxis()->SetTitle(title);
	histoE5x5[i][j]->GetYaxis()->SetTitle("Tracks");
      }
    }
  } 
}

void ElectronStudy::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {

  if (verbose > 1) std::cout << "Run = " << iEvent.id().run() << " Event = " 
			     << iEvent.id().event() << std::endl;

  // get Geometry, B-field, Topology
  edm::ESHandle<MagneticField> bFieldH;
  iSetup.get<IdealMagneticFieldRecord>().get(bFieldH);
  const MagneticField* bField = bFieldH.product();

  edm::ESHandle<CaloGeometry> pG;
  iSetup.get<CaloGeometryRecord>().get(pG);
  const CaloGeometry* geo = pG.product();
  
  edm::ESHandle<CaloTopology> theCaloTopology;
  iSetup.get<CaloTopologyRecord>().get(theCaloTopology); 
  const CaloTopology* caloTopology = theCaloTopology.product();

  // get PCaloHits for ecal barrel
  edm::Handle<edm::PCaloHitContainer> caloHitEB;
  iEvent.getByToken(tok_EBhit_,caloHitEB); 

  // get PCaloHits for ecal endcap
  edm::Handle<edm::PCaloHitContainer> caloHitEE;
  iEvent.getByToken(tok_EEhit_,caloHitEE); 

  // get sim tracks
  edm::Handle<edm::SimTrackContainer>  SimTk;
  iEvent.getByToken(tok_simTk_, SimTk);
  
  // get sim vertices
  edm::Handle<edm::SimVertexContainer> SimVtx;
  iEvent.getByToken(tok_simVtx_, SimVtx);
  
  if (verbose>0) 
    std::cout << "ElectronStudy: hits valid[EB]: " << caloHitEB.isValid() 
	      << " valid[EE]: " << caloHitEE.isValid() << std::endl;
  
  if (caloHitEB.isValid() && caloHitEE.isValid()) {
    unsigned int indx;
    if (verbose>2) {
      edm::PCaloHitContainer::const_iterator ihit;
      for (ihit=caloHitEB->begin(),indx=0; ihit!=caloHitEB->end(); ihit++,indx++) {
	EBDetId id = ihit->id();
	std::cout << "Hit[" << indx << "] " << id << " E " << ihit->energy() 
		  << " T " << ihit->time() << std::endl;
      }
      for (ihit=caloHitEE->begin(),indx=0; ihit!=caloHitEE->end(); ihit++,indx++) {
	EEDetId id = ihit->id();
	std::cout << "Hit[" << indx << "] " << id << " E " << ihit->energy() 
		  << " T " << ihit->time() << std::endl;
      }
    }
    edm::SimTrackContainer::const_iterator simTrkItr=SimTk->begin();
    for (indx=0; simTrkItr!= SimTk->end(); simTrkItr++,indx++) {
      if (verbose>0) std::cout << "ElectronStudy: Track[" << indx << "] ID "
			       << simTrkItr->trackId() << " type " 
			       << simTrkItr->type()    << " charge " 
			       << simTrkItr->charge()  << " p "
			       << simTrkItr->momentum()<< " Generator Index "
			       << simTrkItr->genpartIndex() << " vertex "
			       << simTrkItr->vertIndex() << std::endl;
      if (std::abs(simTrkItr->type()) == 11 && simTrkItr->vertIndex() != -1) {
	int thisTrk = simTrkItr->trackId();
	spr::propagatedTrackDirection trkD = spr::propagateCALO(thisTrk, SimTk, SimVtx, geo, bField, (verbose>1));
	if (trkD.okECAL) {
	  const DetId isoCell = trkD.detIdECAL;
	  DetId hotCell = isoCell;
	  if (hotZone > 0) hotCell = spr::hotCrystal(isoCell, caloHitEB, caloHitEE, geo, caloTopology, hotZone, hotZone, -500.0, 500.0, (verbose>1));
	  double e1x1 = spr::eECALmatrix(hotCell, caloHitEB, caloHitEE, geo, caloTopology, 0, 0, -100.0, -100.0,-500.0, 500.0, (verbose>2));
	  double e3x3 = spr::eECALmatrix(hotCell, caloHitEB, caloHitEE, geo, caloTopology, 1, 1, -100.0, -100.0,-500.0, 500.0, (verbose>2));
	  double e5x5 = spr::eECALmatrix(hotCell, caloHitEB, caloHitEE, geo, caloTopology, 2, 2, -100.0, -100.0,-500.0, 500.0, (verbose>2));
	  double p    = simTrkItr->momentum().P();
	  double eta  = std::abs(simTrkItr->momentum().eta());
	  int etaBin=-1, momBin=-1;
	  for (int ieta=0; ieta<NEtaBins; ieta++)   {
	    if (eta>etaBins[ieta] && eta<etaBins[ieta+1] ) etaBin = ieta+1;
	  }
	  for (int ipt=0;  ipt<NPBins;   ipt++)  {
	    if (p>pBins[ipt]      &&  p<pBins[ipt+1] )     momBin = ipt+1;
	  }
	  double r1=-1, r2=-1, r3=-1;
	  if (e3x3 > 0) r1 = e1x1/e3x3;
	  if (e5x5 > 0) {r2 = e1x1/e5x5; r3 = e3x3/e5x5;}
	  if (verbose>0) {
	    std::cout << "ElectronStudy: p " << p << " [" << momBin << "] eta "
		      << eta << " [" << etaBin << "]";
	    if (isoCell.subdetId() == EcalBarrel) {
	      EBDetId id = isoCell;
	      std::cout << " Cell 0x" << std::hex << isoCell() << std::dec 
			<< " " << id;
	    } else if (isoCell.subdetId() == EcalEndcap) {
	      EEDetId id = isoCell;
	      std::cout << " Cell 0x" << std::hex << isoCell() << std::dec
			<< " " << id;
	    } else {
	      std::cout << " Cell 0x" << std::hex << isoCell() << std::dec
			<< " Unknown Type";
	    }
	    std::cout << " e1x1 " << e1x1 << "|" << r1 << "|" << r2 << " e3x3 "
		      << e3x3 << "|" << r3 << " e5x5 " << e5x5 << std::endl;
	  }
	  if (histos) {
	    histoR1[0][0]->Fill(r1);
	    histoR2[0][0]->Fill(r2);
	    histoR3[0][0]->Fill(r3);
	    histoE1x1[0][0]->Fill(e1x1/p);
	    histoE3x3[0][0]->Fill(e3x3/p);
	    histoE5x5[0][0]->Fill(e5x5/p);
	    if (momBin>0) {
	      histoR1[momBin][0]->Fill(r1);
	      histoR2[momBin][0]->Fill(r2);
	      histoR3[momBin][0]->Fill(r3);
	      histoE1x1[momBin][0]->Fill(e1x1/p);
	      histoE3x3[momBin][0]->Fill(e3x3/p);
	      histoE5x5[momBin][0]->Fill(e5x5/p);
	    }
	    if (etaBin>0) {
	      histoR1[0][etaBin]->Fill(r1);
	      histoR2[0][etaBin]->Fill(r2);
	      histoR3[0][etaBin]->Fill(r3);
	      histoE1x1[0][etaBin]->Fill(e1x1/p);
	      histoE3x3[0][etaBin]->Fill(e3x3/p);
	      histoE5x5[0][etaBin]->Fill(e5x5/p);
	      if (momBin>0) {
		histoR1[momBin][etaBin]->Fill(r1);
		histoR2[momBin][etaBin]->Fill(r2);
		histoR3[momBin][etaBin]->Fill(r3);
		histoE1x1[momBin][etaBin]->Fill(e1x1/p);
		histoE3x3[momBin][etaBin]->Fill(e3x3/p);
		histoE5x5[momBin][etaBin]->Fill(e5x5/p);
	      }
	    }
	  }
	}
      }
    }
  }

}

//define this as a plug-in
DEFINE_FWK_MODULE(ElectronStudy);
