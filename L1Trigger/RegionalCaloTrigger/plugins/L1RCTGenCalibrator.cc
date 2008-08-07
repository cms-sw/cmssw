#include "L1Trigger/RegionalCaloTrigger/interface/L1RCTGenCalibrator.h"

// Framework Stuff
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// Gen Collections
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

#include <fstream>
#include <map>

L1RCTGenCalibrator::L1RCTGenCalibrator(edm::ParameterSet const& ps):
  L1RCTCalibrator(ps)
{
  
}

L1RCTGenCalibrator::~L1RCTGenCalibrator()
{
}

// this gets called ONCE per event!!!!!
void L1RCTGenCalibrator::saveCalibrationInfo(const view_vector& calib_to,const edm::Handle<ecal_view>& e, 
					     const edm::Handle<hcal_view>& h, const edm::Handle<reg_view>& r)
{  
  if(debug() > 0) std::cout << "--------------- Begin L1RCTGenCalibration::saveCalibrationInfo() ---------------\n";

  event_data temp; // event information to save for reprocessing later
  std::vector<generator>* gtemp = &(temp.gen_particles);
  std::vector<region>* regtemp = &(temp.regions);
  std::vector<tpg>* tpgtemp = &(temp.tpgs);
 
  view_vector::const_iterator view = calib_to.begin();
  for(; view != calib_to.end(); ++view)
    {
      cand_iter c = (*view)->begin();
      for(; c != (*view)->end(); ++c)
	{
	  cand_view::pointer cand_ = &(*c); // get abstract candidate pointer

	  const reco::GenParticle* genp_ = dynamic_cast<const reco::GenParticle*>(cand_);

	  if(genp_) // just to be safe...
	    {
	      saveGenInfo(genp_,e,h,r,gtemp,regtemp,tpgtemp);
	    }
	}
    }

  temp.event = eventNumber();
  temp.run = runNumber();

  data_.push_back(temp);
  if(debug() > 0) std::cout << "--------------- End L1RCTGenCalibration::saveCalibrationInfo() ---------------\n";
}

void L1RCTGenCalibrator::postProcessing()
{  
  // first event data loop, calibrate ecal.
  for(std::vector<event_data>::const_iterator i = data_.begin(); i != data_.end(); ++i)
    {
      int ipt = 0;
      hEvent->Fill(i->event);
      hRun->Fill(i->run);

      std::vector<generator>::const_iterator gen = i->gen_particles.begin();

      for(; gen != i->gen_particles.end(); ++gen)
	{
	  if(gen->particle_type != 22) continue;

	  double regionsum = sumEt(gen->eta,gen->phi,i->regions);
	  
	  if(regionsum > 0.0)
	    {
	      std::vector<tpg> matchedTpgs = tpgsNear(gen->eta,gen->phi,i->tpgs);
	      std::pair<double,double> matchedCentroid = std::make_pair(avgEta(matchedTpgs),avgPhi(matchedTpgs));
	      std::pair<double,double> etAndDeltaR95 = showerSize(matchedTpgs);
	      std::pair<double,double> ecalEtandDeltaR95 = showerSize(matchedTpgs, .95, .5, true, false);
	      
	      int sumieta, sumiphi;
	      
	      etaBin(fabs(matchedCentroid.first), sumieta);
	      phiBin(matchedCentroid.second, sumiphi);

	      if(debug() > 0)
		{
		  int n_towers = tpgsNear(matchedCentroid.first, matchedCentroid.second, i->tpgs, ecalEtandDeltaR95.second).size();
		  LogDebug("PhotonTPGSumInfo") << "TPG sum near gen Photon with nearby non-zero RCT region found:\n"
					       << "\tNumber of Towers  : " << n_towers << std::endl
					       << "\tCentroid Eta      : " << matchedCentroid.first << std::endl
					       << "\tCentroid Phi      : " << matchedCentroid.second << std::endl
					       << "\tDelta R 95  (h+e) : " << etAndDeltaR95.second << std::endl
					       << "\tDelta R 95  (e)   : " << ecalEtandDeltaR95.second << std::endl
					       << "\tTotal Et in Cone  : " << etAndDeltaR95.first << std::endl
					       << "\tEcal Et in Cone   : " << ecalEtandDeltaR95.first << std::endl;
		}
		  
	      hPhotonDeltaR95[sumieta]->Fill(etAndDeltaR95.second);
	      gPhotonEtvsGenEt[sumieta]->SetPoint(ipt++, ecalEtandDeltaR95.first, gen->et);	      

	      hGenPhivsTpgSumPhi->Fill(gen->phi, matchedCentroid.second);
	      hGenEtavsTpgSumEta->Fill(gen->eta, matchedCentroid.first);
	      hTpgSumEt->Fill(etAndDeltaR95.first);
	      hTpgSumEta->Fill(matchedCentroid.first);
	      hTpgSumPhi->Fill(matchedCentroid.second);
	    }
	    
	}
    }
}

void L1RCTGenCalibrator::saveGenInfo(const reco::GenParticle* g_ , const edm::Handle<ecal_view>& e_, const edm::Handle<hcal_view>& h_,
				     const edm::Handle<reg_view>& r_, std::vector<generator>* g, std::vector<region>* r,
				     std::vector<tpg>* tp)
{  
  if(debug() > 0) std::cout << "--------------- Begin L1RCTGenCalibrator::saveGenInfo() ---------------------\n";
  ecal_iter ei;
  hcal_iter hi;
  region_iter ri;

  if((g_->pdgId() == 22 || abs(g_->pdgId()) == 211)  && g_->pt() > 5)
    {
      generator gen;

      gen.particle_type = g_->pdgId();
      gen.phi = uniPhi(g_->phi());
      gen.eta = g_->eta();
      gen.et = g_->et();
      gen.loc = makeRctLocation(gen.eta, gen.phi);
      
      if(debug() > 0)
	LogDebug("AddingParticle") << "Adding Gen Particle to Data Record:\n"
				   << "\tEta    :\t" << gen.eta << std::endl
				   << "\tPhi    :\t" << gen.phi << std::endl
				   << "\tEt     :\t" << gen.et << std::endl
				   << "\tRegion :\t" << gen.loc.region << std::endl
				   << "\tCard   :\t" << gen.loc.card << std::endl
				   << "\tCrate  :\t" << gen.loc.crate << std::endl;      

      hGenPhi->Fill(gen.phi);
      hGenEta->Fill(gen.eta);
      hGenEt->Fill(gen.et);


      g->push_back(gen);            
      
      for(ri = r_->begin(); ri != r_->end(); ++ri)
	{
	  region r_save;

	  r_save.loc.crate  = ri->rctCrate();
	  r_save.loc.card   = ri->rctCard();
	  r_save.loc.region = ri->rctRegionIndex();
	  r_save.linear_et = ri->et();

	  if( find<region>(r_save, *r) == -1 && isSelfOrNeighbor( r_save.loc, gen.loc ) &&
	      r_save.linear_et > 2 )
	    {
	      r_save.ieta = ri->gctEta();
	      r_save.iphi = ri->gctPhi();
	      
	      if(debug() > 0)
		LogDebug("AddingRegion") << "Adding RCT Region to Data Record:\n"
					 << "\tRegion:\t" << r_save.loc.region << std::endl
					 << "\tCard  :\t" << r_save.loc.card << std::endl
					 << "\tCrate :\t" << r_save.loc.crate << std::endl
					 << "\tEt    :\t" << r_save.linear_et << std::endl;

	      hRCTRegionEt->Fill(r_save.linear_et*.5);
	      hRCTRegionEta->Fill(l1Geometry()->globalEtaBinCenter(r_save.ieta));
	      hRCTRegionPhi->Fill(l1Geometry()->emJetPhiBinCenter(r_save.iphi));

	      hGenPhivsRegionPhi->Fill(gen.phi,l1Geometry()->emJetPhiBinCenter(r_save.iphi));
	      hGenEtavsRegionEta->Fill(gen.eta,l1Geometry()->globalEtaBinCenter(r_save.ieta));

	      r->push_back(r_save);
	    }
	}

      for(ei = e_->begin(); ei != e_->end(); ++ei)
	for(hi = h_->begin(); hi != h_->end(); ++hi)
	  {	    
	    if(ei->id().ieta() == hi->id().ieta() && ei->id().iphi() == hi->id().iphi() && ecalE(*ei) + hcalE(*hi) > 0.5)
	      {		
		tpg t_save;
	      		
		t_save.ieta = ei->id().ieta();
		t_save.iphi = ei->id().iphi();		
		t_save.loc = makeRctLocation(t_save.ieta, t_save.iphi);

		if( isSelfOrNeighbor(t_save.loc, gen.loc) )
		{
		  t_save.ecalEt = ecalEt(*ei);
		  t_save.hcalEt = hcalEt(*hi);
		  t_save.ecalE = ecalE(*ei);
		  t_save.hcalE = hcalE(*hi);
		
		  if( find<tpg>(t_save, *tp) == -1 )
		    {
		      if(debug() > 0)
			LogDebug("AddingTPG") << "Adding TPG to data record:\n"
					      << "\tieta  :\t" << t_save.ieta << std::endl
					      << "\tiphi  :\t" << t_save.iphi << std::endl
					      << "\tecalEt:\t" << t_save.ecalEt << std::endl
					      << "\thcalEt:\t" << t_save.hcalEt << std::endl
					      << "\tregion:\t" << t_save.loc.region << std::endl
					      << "\tcard  :\t" << t_save.loc.card << std::endl
					      << "\tcrate :\t" << t_save.loc.crate << std::endl;
		      
		      tp->push_back(t_save);
		    }
		}
	      }
	  }
    }
  if(debug() > 0) std::cout << "--------------- End L1RCTGenCalibrator::saveGenInfo() ---------------------\n";
}

void L1RCTGenCalibrator::bookHistograms()
{
  double deltaRbins[28];
  
  for(int i = 0; i < 28; ++i)
    {
      double delta_r, eta, phi;
      phiValue(0,phi);
      etaValue(i+1,eta);
      deltaR(0,0,eta,phi,delta_r);

      deltaRbins[i] = delta_r;
    }

  putHist(hEvent = new TH1F("hEvent","Event Number",10000,0,10000));
  putHist(hRun = new TH1F("hRun","Run Number",10000,0,10000));

  putHist(hGenPhi = new TH1F("hGenPhi","Generator #phi",72, 0, 2*M_PI));
  putHist(hGenEta = new TH1F("hGenEta","Generator #eta",100,-6,6));
  putHist(hGenEt = new TH1F("hGenEt","Generator E_{T}", 1000,0,500));
  putHist(hGenEtSel = new TH1F("hGenEtSel","Generator E_{T} Selected for Calibration",1000,0,500));
  
  putHist(hRCTRegionEt = new TH1F("hRCTRegionEt","RCT Region E_{T} used in Calibration", 1000, 0, 500));
  putHist(hRCTRegionPhi = new TH1F("hRCTRegionPhi","RCT Region #phi", 72, 0, 2*M_PI));
  putHist(hRCTRegionEta = new TH1F("hRCTRegionEta","RCT Region #eta", 23, -5, 5));
  
  putHist(hTpgSumEt = new TH1F("hTpgSumEt","TPG Sum E_{T}",1000,0,500));
  putHist(hTpgSumEta = new TH1F("hTpgSumEta","TPG Sum #eta Centroid", 57, -5,5));
  putHist(hTpgSumPhi = new TH1F("hTpgSumPhi","TPG Sum #phi Centroid", 72, 0, 2*M_PI));
  
  putHist(hGenPhivsTpgSumPhi = new TH2F("hGenPhivsTpgSumPhi","Generator Particle #phi vs. TPG Sum #phi Centroid",72,0,2*M_PI,72,0,2*M_PI));
  putHist(hGenEtavsTpgSumEta = new TH2F("hGenEtavsTpgSumEta","Generator Particle #eta vs. TPG Sum #eta Centroid",57,-5,5,57,-5,5));
  putHist(hGenPhivsRegionPhi = new TH2F("hGenPhivsRegionPhi","Generator Particle #phi vs. RCT Region #phi",72,0,2*M_PI,72,0,2*M_PI));
  putHist(hGenEtavsRegionEta = new TH2F("hGenEtavsRegionEta","Generator Particle #eta vs. RCT Region #eta",23,-5,5,23,-5,5));

  for(int i = 0; i < 28; ++i)
    {
      putHist(hPhotonDeltaR95[i] = new TH1F(TString("hPhotonDeltaR95") += i, TString("Photon #DeltaR Containing 95% of E_{T} in #eta Bin: ") +=i, 28, deltaRbins));
      putHist(hPionDeltaR95[i] = new TH1F(TString("hPhotonDeltaR95") += i, TString("Pion #DeltaR Containing 95% of E_{T} in #eta Bin: ") +=i, 28, deltaRbins));
      
      putHist(gPhotonEtvsGenEt[i] = new TGraph());
      gPhotonEtvsGenEt[i]->SetName(TString("hPhotonEtvsGenEt") += i); 
      gPhotonEtvsGenEt[i]->SetTitle(TString("Photon TPG Sum E_{T} vs. Generator E_{T}") += i);

      for(int i = 0; i < 12; ++i)
	{
	}	      
    }

  for(int i = 0; i < 12; ++i)
    {
      putHist(hDeltaEtPeakvsEtaBin_uc[i] = new TH1F(TString("hDeltaEtPeakvsEtaBin_uc") += i,
						    TString("Uncorrected #DeltaE_{T} Peak vs. #eta Bin in E_{T} Bin ") += i,
						    25,0.5,25.5)); 
      putHist(hDeltaEtPeakvsEtaBin_c[i] = new TH1F(TString("hDeltaEtPeakvsEtaBin_c") += i,
						   TString("Corrected #DeltaE_{T} Peak vs. #eta Bin in E_{T} Bin ") += i,
						   25,0.5,25.5));
      putHist(hDeltaEtPeakRatiovsEtaBin[i] = new TH1F(TString("hEtPeakRatiovsEtaBin") += i,
						      TString("frac{Corrected #E_{T}}{Uncorrected E_{T}} vs. #eta Bin in E_{T} Bin ") += i,
						      25,0.5,25.5));
    }
  
  putHist(hDeltaEtPeakvsEtaBinAllEt_uc = new TH1F("hDeltaEtPeakvsEtaBinAllEt_uc",
						  "Uncorrected #DeltaE_{T} vs. #eta Bin, All E_{T}",
						  25,0.5,25.5));
  putHist(hDeltaEtPeakvsEtaBinAllEt_c = new TH1F("hDeltaEtPeakvsEtaBinAllEt_c",
						 "Corrected #DeltaE_{T} vs. #eta Bin, All E_{T}",
						 25,0.5,25.5));
  putHist(hDeltaEtPeakRatiovsEtaBinAllEt = new TH1F("hDeltaEtPeakRatiovsEtaBinAllEt",
						    "frac{Corrected E_{T}}{Uncorrected E_{T} vs. #eta Bin, All E_{T}",
						    25,0.5,25.5));
}
