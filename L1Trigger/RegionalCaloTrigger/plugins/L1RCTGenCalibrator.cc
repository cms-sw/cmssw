#include "L1Trigger/RegionalCaloTrigger/interface/L1RCTGenCalibrator.h"

// Framework Stuff
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// Gen Collections
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

//RooFit
#include "RooRealVar.h"
#include "RooArgSet.h"

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
  if(debug() > 0) edm::LogVerbatim("saveCalibrationInfo()") << "--------------- Begin L1RCTGenCalibration::saveCalibrationInfo() ---------------\n";

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

  temp.event = totalEvents();
  temp.run = runNumber();

  if(!farmout()) 
    data_.push_back(temp);
  else
    {
      root_structs::Event evt;
      root_structs::Generator gen;
      root_structs::Region rgn;
      root_structs::TPG ttpg;

      evt.event = temp.event;
      evt.run = temp.run;

      gen.nGen = temp.gen_particles.size();
      rgn.nRegions = temp.regions.size();
      ttpg.nTPG = temp.tpgs.size();
      
      for(unsigned i = 0; i < gen.nGen && i < 100; ++i)
	{
	  gen.particle_type[i] = temp.gen_particles[i].particle_type;
	  gen.et[i] = temp.gen_particles[i].et;
	  gen.eta[i] = temp.gen_particles[i].eta;
	  gen.phi[i] = temp.gen_particles[i].phi;
	  gen.crate[i] = temp.gen_particles[i].loc.crate;
	  gen.card[i] = temp.gen_particles[i].loc.card;
	  gen.region[i] = temp.gen_particles[i].loc.region;
	}

      for(unsigned i = 0; i < rgn.nRegions; ++i)
	{
	  rgn.linear_et[i] = temp.regions[i].linear_et;
	  rgn.ieta[i] = temp.regions[i].ieta;
	  rgn.iphi[i] = temp.regions[i].iphi;
	  rgn.eta[i] = temp.regions[i].eta;
	  rgn.phi[i] = temp.regions[i].phi;
	  rgn.crate[i] = temp.regions[i].loc.crate;
	  rgn.card[i] = temp.regions[i].loc.card;
	  rgn.region[i] = temp.regions[i].loc.region;
	}

      for(unsigned i = 0; i< ttpg.nTPG; ++i)
	{
	  ttpg.ieta[i] = temp.tpgs[i].ieta;
	  ttpg.iphi[i] = temp.tpgs[i].iphi;
	  ttpg.eta[i] = temp.tpgs[i].eta;
	  ttpg.phi[i] = temp.tpgs[i].phi;
	  ttpg.ecalEt[i] = temp.tpgs[i].ecalEt;
	  ttpg.hcalEt[i] = temp.tpgs[i].hcalEt;
	  ttpg.ecalE[i] = temp.tpgs[i].ecalE;
	  ttpg.hcalE[i] = temp.tpgs[i].hcalE;
	  ttpg.crate[i] = temp.tpgs[i].loc.crate;
	  ttpg.card[i] = temp.tpgs[i].loc.card;
	  ttpg.region[i] = temp.tpgs[i].loc.region;
	}

      if(Tree()->GetBranch("Event"))
	Tree()->GetBranch("Event")->SetAddress(&evt);
      else
	Tree()->Branch("Event",&evt,"event/i:run");

      if(Tree()->GetBranch("Generator"))
	Tree()->GetBranch("Generator")->SetAddress(&gen);
      else
	Tree()->Branch("Generator",&gen,"nGen/i:particle_type[100]/I:et[100]/D:eta[100]:phi[100]:crate[100]/i:card[100]:region[100]");
      
      if(Tree()->GetBranch("Region"))
	Tree()->GetBranch("Region")->SetAddress(&rgn);
      else
	Tree()->Branch("Region",&rgn,"nRegions/i:linear_et[200]/I:ieta[200]:iphi[200]:eta[200]:phi[200]:crate[200]/i:card[200]:region[200]");

      if(Tree()->GetBranch("CaloTPG"))
	Tree()->GetBranch("CaloTPG")->SetAddress(&ttpg);
      else
	Tree()->Branch("CaloTPG",&ttpg,"nTPG/i:ieta[3100]/I:iphi[3100]:eta[3100]:phi[3100]:ecalEt[3100]/D:hcalEt[3100]:ecalE[3100]:hcalE[3100]:crate[3100]/i:card[3100]:region[3100]");
      
      Tree()->Fill();
      if(totalEvents() % 100 == 0) Tree()->AutoSave("SaveSelf");
    }
    
  if(debug() > 0) edm::LogVerbatim("saveCalibrationInfo()") << "--------------- End L1RCTGenCalibration::saveCalibrationInfo() ---------------\n";
}

void L1RCTGenCalibrator::postProcessing()
{  
  if(debug() > -1) edm::LogVerbatim("postProcessing()") << "------------------postProcessing()-------------------\n";
  int iph[28] = {0}, ipi[28] = {0}, ipi2[28] = {0},  nEvents = 0, nUseful = 0;
  
  // first event data loop, calibrate ecal, hcal with NI pions
  for(std::vector<event_data>::const_iterator i = data_.begin(); i != data_.end(); ++i)
    {
      /*
      for(int n = 0; n < 28; ++n)
	{
	  std::cout << "Trigger Tower " <<  n + 1 << ": " << iph[n] << ' ' << ipi[n] << ' ' << ipi2[n] << std::endl;
	}
      */

      nEvents++;
      hEvent->Fill(i->event);
      hRun->Fill(i->run);

      bool used_flag = false;

      std::vector<generator>::const_iterator gen = i->gen_particles.begin();

      std::vector<generator> ovls = overlaps(i->gen_particles);

      
      if(debug() > 0) LogTrace("StartOverlap") << "Finding overlapping selected particles in this event!\n";
      if(debug() > 0)
	{
	  edm::LogVerbatim("OverlapInfo") << "=======Overlapping accepted gen particles in event " << nEvents << "\n";
	  for(std::vector<generator>::const_iterator ov = ovls.begin(); ov != ovls.end(); ++ov)
	    edm::LogVerbatim("OverlapInfo") << '\t' <<  ov->particle_type << " " << ov->et << " " << ov->eta << " " << ov->phi << "\n";
	  edm::LogVerbatim("OverlapInfo") << "======================================================\n";
	}
      if(debug() > 0) LogTrace("EndOverlap") << "Done finding overlaps!\n";

      for(; gen != i->gen_particles.end(); ++gen)
	{	  
	  if(gen->particle_type != 22 && abs(gen->particle_type) != 211 || find<generator>(*gen, ovls) != -1) continue;
	  double regionsum = sumEt(gen->eta,gen->phi,i->regions);
	  if(regionsum > 0.0)
	    {	
	      std::vector<tpg> matchedTpgs = tpgsNear(gen->eta,gen->phi,i->tpgs);
	      std::pair<double,double> matchedCentroid = std::make_pair(avgEta(matchedTpgs),avgPhi(matchedTpgs));
	      std::pair<double,double> etAndDeltaR95 = showerSize(matchedTpgs);

	      if(gen->particle_type == 22)
		{		  
		  if(!used_flag) used_flag = true;

		  std::pair<double,double> ecalEtandDeltaR95 = showerSize(matchedTpgs, .95, .5, true, false);
		  
		  int sumieta;
		  
		  etaBin(fabs(matchedCentroid.first), sumieta);
		  
		  if(debug() > 0)
		    {
		      int n_towers = tpgsNear(matchedCentroid.first, matchedCentroid.second, matchedTpgs, ecalEtandDeltaR95.second).size();
		      LogTrace("PhotonTPGSumInfo") << "TPG sum near gen Photon with nearby non-zero RCT region found:\n"
						   << "Number of Towers  : " << n_towers << " "
						   << "\tCentroid Eta      : " << matchedCentroid.first << " "
						   << "\tCentroid Phi      : " << matchedCentroid.second << " "
						   << "\tDelta R 95  (h+e) : " << etAndDeltaR95.second << " "
						   << "\nDelta R 95  (e)   : " << ecalEtandDeltaR95.second << " "
						   << "\tTotal Et in Cone  : " << etAndDeltaR95.first << " "
						   << "\tEcal Et in Cone   : " << ecalEtandDeltaR95.first << " ";
		    }
		  
		  roorvPhotonTPGSumEt[sumieta - 1]->setVal(ecalEtandDeltaR95.first);
		  roorvPhotonGenEt[sumieta - 1]->setVal(gen->et);
		  roodsPhotonEtvsGenEt[sumieta - 1]->add(RooArgSet(*roorvPhotonTPGSumEt[sumieta-1],*roorvPhotonGenEt[sumieta-1]));
		      
		  hPhotonDeltaR95[sumieta - 1]->Fill(etAndDeltaR95.second);
		  gPhotonEtvsGenEt[sumieta - 1]->SetPoint(iph[sumieta - 1]++, ecalEtandDeltaR95.first, gen->et);	      
		}  

	      if(abs(gen->particle_type) == 211)
		{
		  if(!used_flag) used_flag = true;
		  
		  double ecal = sumEt(matchedCentroid.first, matchedCentroid.second, matchedTpgs, etAndDeltaR95.second, true, false);
		  double hcal = sumEt(matchedCentroid.first, matchedCentroid.second, matchedTpgs, etAndDeltaR95.second, false, true);

		  int sumieta;
		  etaBin(fabs(matchedCentroid.first), sumieta);
		  
		  if( ecal < 1.0  && etAndDeltaR95.first > 0)
		    {
		      if(debug() > 0)
			{
			  int n_towers = tpgsNear(matchedCentroid.first, matchedCentroid.second, matchedTpgs, etAndDeltaR95.second).size();
			  LogTrace("NIPionTPGSumInfo") << "TPG sum near gen Charged Pion with nearby non-zero RCT region and little ECAL energy found:\n"
						       << "Number of Towers  : " << n_towers << " "
						       << "\tCentroid Eta      : " << matchedCentroid.first << " "
						       << "\tCentroid Phi      : " << matchedCentroid.second << " "
						       << "\tDelta R 95  (h+e) : " << etAndDeltaR95.second << " "
						       << "\nTotal Et in Cone  : " << etAndDeltaR95.first << " "
						       << "\tEcal Et in Cone   : " << ecal << " "
						       << "\tHcal Et in Cone   : " << hcal << " ";
			}
		      
		      roorvNIPionTPGSumEt[sumieta - 1]->setVal(hcal);
		      roorvNIPionGenEt[sumieta - 1]->setVal(gen->et);
		      roodsNIPionEtvsGenEt[sumieta - 1]->add(RooArgSet(*roorvNIPionTPGSumEt[sumieta-1],*roorvNIPionGenEt[sumieta-1]));

		      hNIPionDeltaR95[sumieta - 1]->Fill(etAndDeltaR95.second);
		      gNIPionEtvsGenEt[sumieta - 1]->SetPoint(ipi[sumieta - 1]++, hcal, gen->et);
		    }
		  else if(etAndDeltaR95.first > 0)
		    {
		      if(debug() > 0) 
			{
			  int n_towers = tpgsNear(matchedCentroid.first, matchedCentroid.second, matchedTpgs, etAndDeltaR95.second).size();
			  LogTrace("PionTPGSumInfo") << "TPG sum near gen Charged Pion with nearby non-zero RCT region found:\n"
						     << "Number of Towers  : " << n_towers << " "
						     << "\tCentroid Eta      : " << matchedCentroid.first << " "
						     << "\tCentroid Phi      : " << matchedCentroid.second << " "
						     << "\tDelta R 95  (h+e) : " << etAndDeltaR95.second << " "
						     << "\nTotal Et in Cone  : " << etAndDeltaR95.first << " "
						     << "\tEcal Et in Cone   : " << ecal << " "
						     << "\tHcal Et in Cone   : " << hcal << " ";
			}
		      
		      hPionDeltaR95[sumieta - 1]->Fill(etAndDeltaR95.second);
		      gPionEcalEtvsHcalEtvsGenEt[sumieta - 1]->SetPoint(ipi2[sumieta - 1]++, ecal, hcal, gen->et);
		    }
		}

	      hGenPhivsTpgSumPhi->Fill(gen->phi, matchedCentroid.second);
	      hGenEtavsTpgSumEta->Fill(gen->eta, matchedCentroid.first);	      
	      hTpgSumEt->Fill(etAndDeltaR95.first);
	      hTpgSumEta->Fill(matchedCentroid.first);
	      hTpgSumPhi->Fill(matchedCentroid.second);		
	    }	    
	}

      if(used_flag) ++nUseful;
    }

  int pitot = 0, phtot = 0, pitot2 = 0;
  TF1 *photon[28] = {NULL};
  TF1 *NIpion_low[28] = {NULL};
  TF1 *NIpion_high[28] = {NULL};
        
  TF2* eh_surf[28] = {NULL};


  for(int i = 0; i < 28; ++i)
    {
      pitot += ipi[i];
      phtot += iph[i];
      pitot2 += ipi2[i];

      if(iph[i] > 100)
	{
	  photon[i] = new TF1((TString("ecal_fit")+=i).Data(),"x**3++x**2++x",0,100);
	  gPhotonEtvsGenEt[i]->Fit(photon[i],fitOpts().c_str());

	  ecal_[i][0] = photon[i]->GetParameter(0);
	  ecal_[i][1] = photon[i]->GetParameter(1);
	  ecal_[i][2] = photon[i]->GetParameter(2);
	}
      else
	{
	  ecal_[i][0] = 0;
	  ecal_[i][1] = 0;
	  ecal_[i][2] = 1;
	}
      if(ipi[i] > 100)
	{
	  NIpion_low[i] = new TF1((TString("hcal_fit_low")+=i).Data(),"x**3++x**2++x",0,23);
	  NIpion_high[i] = new TF1((TString("hcal_fit_high")+=i).Data(),"x**3++x**2++x",23,100);
	  gNIPionEtvsGenEt[i]->Fit(NIpion_low[i],fitOpts().c_str());
	  gNIPionEtvsGenEt[i]->Fit(NIpion_high[i],fitOpts().c_str());

	  hcal_[i][0] = NIpion_low[i]->GetParameter(0);
	  hcal_[i][1] = NIpion_low[i]->GetParameter(1);
	  hcal_[i][2] = NIpion_low[i]->GetParameter(2);
	  
	  hcal_high_[i][0] = NIpion_high[i]->GetParameter(0);
	  hcal_high_[i][1] = NIpion_high[i]->GetParameter(1);
	  hcal_high_[i][2] = NIpion_high[i]->GetParameter(2);
	}
      else
	{
	  hcal_[i][0] = hcal_[i][1] = hcal_high_[i][0] = hcal_high_[i][1] = 0;
	  hcal_[i][2] = hcal_high_[i][2] = 1;
	}
      if(ipi2[i] >100 && NIpion_low[i] && NIpion_high[i] && photon[i])
	{
	  eh_surf[i] = new TF2((TString("eh_surf")+=i).Data(),"x**3++x**2++x++y**3++y**2++y++x**2*y++y**2*x++x*y++x**3*y++y**3*x++x**2*y**2",0,23,0,23);
	  eh_surf[i]->FixParameter(0,photon[i]->GetParameter(0));
	  eh_surf[i]->FixParameter(1,photon[i]->GetParameter(1));
	  eh_surf[i]->FixParameter(2,photon[i]->GetParameter(2));
	  eh_surf[i]->FixParameter(3,NIpion_low[i]->GetParameter(0));
	  eh_surf[i]->FixParameter(4,NIpion_low[i]->GetParameter(1));
	  eh_surf[i]->FixParameter(5,NIpion_low[i]->GetParameter(2));
	  gPionEcalEtvsHcalEtvsGenEt[i]->Fit(eh_surf[i],fitOpts().c_str());

	  cross_[i][0] = eh_surf[i]->GetParameter(6);
	  cross_[i][1] = eh_surf[i]->GetParameter(7);
	  cross_[i][2] = eh_surf[i]->GetParameter(8);
	  cross_[i][3] = eh_surf[i]->GetParameter(9);
	  cross_[i][4] = eh_surf[i]->GetParameter(10);
	  cross_[i][5] = eh_surf[i]->GetParameter(11);
	}
      else
	{
	  cross_[i][0] = cross_[i][1] = cross_[i][2] = cross_[i][3] = cross_[i][4] = cross_[i][5] = 0;
	}
    }

  for(std::vector<event_data>::const_iterator i = data_.begin(); i != data_.end(); ++i)
    {
      std::vector<generator>::const_iterator gen = i->gen_particles.begin();

      std::vector<generator> ovls = overlaps(i->gen_particles);
      
      if(debug() > 0) LogTrace("StartOverlap") << "Finding overlapping selected particles in this event!\n";
      if(debug() > 0)
	{
	  edm::LogVerbatim("OverlapInfo") << "=======Overlapping accepted gen particles in event " << nEvents << "\n";
	  for(std::vector<generator>::const_iterator ov = ovls.begin(); ov != ovls.end(); ++ov)
	    edm::LogVerbatim("OverlapInfo") << '\t' <<  ov->particle_type << " " << ov->et << " " << ov->eta << " " << ov->phi << "\n";
	  edm::LogVerbatim("OverlapInfo") << "======================================================\n";
	}
      if(debug() > 0) LogTrace("EndOverlap") << "Done finding overlaps!\n";

      for(; gen != i->gen_particles.end(); ++gen)
	{	  
	  if(gen->particle_type != 22 && abs(gen->particle_type) != 211 || find<generator>(*gen, ovls) != -1) continue;
	  
	  double regionsum = sumEt(gen->eta,gen->phi,i->regions);
	  
	  if(regionsum > 0.0)
	    {	
	      std::vector<tpg> matchedTpgs = tpgsNear(gen->eta,gen->phi,i->tpgs);
	      std::pair<double,double> matchedCentroid = std::make_pair(avgEta(matchedTpgs),avgPhi(matchedTpgs));
	      std::pair<double,double> etAndDeltaR95 = showerSize(matchedTpgs);

	      if(gen->particle_type == 22)
		{		
		  std::pair<double,double> ecalEtandDeltaR95 = showerSize(matchedTpgs, .95, .5, true, false);
		  
		  int sumieta;
		  
		  etaBin(fabs(matchedCentroid.first), sumieta);
		  		  
		  if(etAndDeltaR95.first > 23)
		    {
		      double ecal_c = sumEt(matchedCentroid.first, matchedCentroid.second, matchedTpgs, etAndDeltaR95.second, true, false, true);
		      
		      double deltaeovere = (gen->et - ecal_c)/gen->et;
		      hPhotonDeltaEOverE[sumieta - 1]->Fill(deltaeovere);
		    }
		}  

	      if(abs(gen->particle_type) == 211)
		{		  
		  double ecal = sumEt(matchedCentroid.first, matchedCentroid.second, matchedTpgs, etAndDeltaR95.second, true, false);
		  double hcal = sumEt(matchedCentroid.first, matchedCentroid.second, matchedTpgs, etAndDeltaR95.second, false, true);

		  int sumieta;
		  etaBin(fabs(matchedCentroid.first), sumieta);
		 
		  if(ecal == 0.0) ecal += 0.00000000001;
		  if(hcal/ecal > 0.05 && etAndDeltaR95.first > 23)
		    {
		      double et_c = sumEt(matchedCentroid.first, matchedCentroid.second, matchedTpgs, etAndDeltaR95.second, true, true, true);

		      double deltaeovere = (gen->et - et_c)/gen->et;
		      hPionDeltaEOverE[sumieta - 1]->Fill(deltaeovere);
		    }
		}
	    }	    
	}
    }

  TF1* phGaus[28];
  TF1* piGaus[28];

  for(int i = 0; i < 28; ++i)
    {
      double ph_peak  = hPhotonDeltaEOverE[i]->GetBinCenter(hPhotonDeltaEOverE[i]->GetMaximumBin());
      double ph_upper = ph_peak + hPhotonDeltaEOverE[i]->GetRMS();
      double ph_lower = ph_peak - hPhotonDeltaEOverE[i]->GetRMS();

      double pi_peak  = hPionDeltaEOverE[i]->GetBinCenter(hPionDeltaEOverE[i]->GetMaximumBin());
      double pi_upper = pi_peak + hPionDeltaEOverE[i]->GetRMS();
      double pi_lower = pi_peak - hPionDeltaEOverE[i]->GetRMS();
      
      phGaus[i] = new TF1(TString("phGaus")+=i,"gaus",ph_lower,ph_upper);
      piGaus[i] = new TF1(TString("piGaus")+=i,"gaus",pi_lower,pi_upper);
      
      if(hPhotonDeltaEOverE[i]->GetEntries() > 100)
	{
	  hPhotonDeltaEOverE[i]->Fit(phGaus[i],fitOpts().c_str());
	  he_low_smear_[i] = 1/(1 - phGaus[i]->GetParameter(1));
	}
      else
	he_low_smear_[i] = 1;
	
      if(hPionDeltaEOverE[i]->GetEntries() > 100)
	{
	  hPionDeltaEOverE[i]->Fit(piGaus[i],fitOpts().c_str());
	  he_high_smear_[i] = 1/(1 - piGaus[i]->GetParameter(1));
	}
      else
	he_high_smear_[i] = 1;
    }

  for(int i = 0; i < 28; ++i)
    {
      if(phGaus[i]) delete phGaus[i];
      if(phGaus[i]) delete piGaus[i];
      if(photon[i]) delete photon[i];
      if(NIpion_low[i]) delete NIpion_low[i];
      if(NIpion_high[i]) delete NIpion_high[i];
      if(eh_surf[i]) delete eh_surf[i];
    }

  if(debug() > -1)
    edm::LogVerbatim("EndPostProcessing") << "Completed L1RCTGenCalibrator::postProcessing() with the following results:\n"
					  << "Total Events       : " << nEvents << "\n"
					  << "Useful Events      : " << nUseful << "\n"
					  << "Number of NI Pions : " << pitot << "\n"
					  << "Number of I Pions  : " << pitot2 << "\n"
					  << "Number of Photons  : " << phtot << "\n";
}

void L1RCTGenCalibrator::saveGenInfo(const reco::GenParticle* g_ , const edm::Handle<ecal_view>& e_, const edm::Handle<hcal_view>& h_,
				     const edm::Handle<reg_view>& r_, std::vector<generator>* g, std::vector<region>* r,
				     std::vector<tpg>* tp)
{  
  if(debug() > 0) LogTrace("saveGenInfo()") << "--------------- Begin L1RCTGenCalibrator::saveGenInfo() ---------------------\n";
  ecal_iter ei;
  hcal_iter hi;
  region_iter ri;
  unsigned reg_sum = 0;

  if((g_->pdgId() == 22 || abs(g_->pdgId()) == 211)  && g_->pt() > 9.5 && fabs(g_->eta()) < 2.5)
    {
      generator gen;

      gen.particle_type = g_->pdgId();
      gen.phi = uniPhi(g_->phi());
      gen.eta = g_->eta();
      gen.et = g_->et();
      gen.loc = makeRctLocation(gen.eta, gen.phi);
      
      if(debug() > 0)
	LogTrace("AddingParticle") << "Adding Gen Particle to Data Record:\n"
				   << "Eta    : " << gen.eta << " "
				   << "\tPhi    : " << gen.phi << " "
				   << "\tEt     : " << gen.et << " "
				   << "\nRegion : " << gen.loc.region << " "
				   << "\tCard   : " << gen.loc.card << " "
				   << "\tCrate  : " << gen.loc.crate << " ";      

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
	      reg_sum += r_save.linear_et;
	      r_save.ieta = ri->gctEta();
	      r_save.iphi = ri->gctPhi();
	      
	      if(debug() > 0)
		LogTrace("AddingRegion") << "Adding RCT Region to Data Record:\n"
					 << "Region: " << r_save.loc.region << " "
					 << "\tCard  : " << r_save.loc.card << " "
					 << "\tCrate : " << r_save.loc.crate << " "
					 << "\tEt    : " << r_save.linear_et << " ";

	      hRCTRegionEt->Fill(r_save.linear_et*.5);
	      hRCTRegionEta->Fill(l1Geometry()->globalEtaBinCenter(r_save.ieta));
	      hRCTRegionPhi->Fill(l1Geometry()->emJetPhiBinCenter(r_save.iphi));

	      r_save.eta = l1Geometry()->globalEtaBinCenter(r_save.ieta);
	      r_save.phi = l1Geometry()->emJetPhiBinCenter(r_save.iphi);

	      hGenPhivsRegionPhi->Fill(gen.phi,l1Geometry()->emJetPhiBinCenter(r_save.iphi));
	      hGenEtavsRegionEta->Fill(gen.eta,l1Geometry()->globalEtaBinCenter(r_save.ieta));

	      r->push_back(r_save);
	    }
	}

      if(reg_sum)
	for(ei = e_->begin(); ei != e_->end(); ++ei)
	  {	    
	    for(hi = h_->begin(); hi != h_->end(); ++hi)
	      if(hi->id().ieta() == ei->id().ieta() && hi->id().iphi() == ei->id().iphi())
		break;
	    
	    if(hi != h_->end() && ecalE(*ei) + hcalE(*hi) > 0.5)
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
		    
		    etaValue(t_save.ieta,t_save.eta);
		    phiValue(t_save.iphi,t_save.phi);

		    if( find<tpg>(t_save, *tp) == -1 )
		      {
			if(debug() > 0)
			  LogTrace("AddingTPG") << "Adding TPG to data record:\n"
						<< "ieta  : " << t_save.ieta << " "
						<< "\tiphi  : " << t_save.iphi << " "
						<< "\tecalEt: " << t_save.ecalEt << " "
						<< "\thcalEt: " << t_save.hcalEt << " "
						<< "\nregion: " << t_save.loc.region << " "
						<< "\tcard  : " << t_save.loc.card << " "
						<< "\tcrate : " << t_save.loc.crate << " ";
			
			tp->push_back(t_save);
		      }
		  }
		
	      }
	  }
    }
  if(debug() > 0) LogTrace("saveGenInfo()") << "--------------- End L1RCTGenCalibrator::saveGenInfo() ---------------------\n";
}

void L1RCTGenCalibrator::bookHistograms()
{
  double deltaRbins[29];
  
  for(int i = 0; i < 28; ++i)
    {
      double delta_r, eta, phi;
      phiValue(0,phi);
      etaValue(i+1,eta);
      deltaR(0,0,eta,phi,delta_r);

      deltaRbins[i+1] = delta_r;
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
      putHist(hPhotonDeltaR95[i] = new TH1F(TString("hPhotonDeltaR95") += i, TString("#gamma #DeltaR Containing 95% of E_{T} in #eta Bin: ") +=i, 28, deltaRbins));
      putHist(hNIPionDeltaR95[i] = new TH1F(TString("hNIPionDeltaR95") += i, TString("NI #pi^{#pm} #DeltaR Containing 95% of E_{T} in #eta Bin: ") +=i, 28, deltaRbins));
      putHist(hPionDeltaR95[i] = new TH1F(TString("hPionDeltaR95") += i, TString("#pi^{#pm} #DeltaR Containing 95% of E_{T} in #eta Bin: ") +=i, 28, deltaRbins));


      putHist(gPhotonEtvsGenEt[i] = new TGraphAsymmErrors());
      gPhotonEtvsGenEt[i]->SetName(TString("gPhotonEtvsGenEt") += i); 
      gPhotonEtvsGenEt[i]->SetTitle(TString("#gamma TPG Sum E_{T} vs. Generator E_{T}, #eta Bin: ") += i);

      putHist(gNIPionEtvsGenEt[i] = new TGraphAsymmErrors());
      gNIPionEtvsGenEt[i]->SetName(TString("gNIPionEtvsGenEt") += i);
      gNIPionEtvsGenEt[i]->SetTitle(TString("#pi^{#pm} with no/small ECAL deposit TPG Sum E_{T} vs. Generator E_{T}, #eta Bin: ") += i);

      putHist(gPionEcalEtvsHcalEtvsGenEt[i] = new TGraph2DErrors());
      gPionEcalEtvsHcalEtvsGenEt[i]->SetName(TString("gPionEcalEtvsHcalEtvsGenEt") += i);
      gPionEcalEtvsHcalEtvsGenEt[i]->SetTitle(TString("#pi^{#pm} Ecal E_{T} vs Hcal E_{T} vs Generator E_{T}, #eta Bin: ") += i);

      
      putHist(roorvPhotonGenEt[i] = new RooRealVar(TString("roorvPhotonGenEt")+=i,TString("#gamma Gen E_{T} in #eta bin: ")+=i,0,"GeV"));
      putHist(roorvPhotonTPGSumEt[i] = new RooRealVar(TString("roorvPhotonTPGSumEt")+=i,TString("#gamma TPG Sum E_{T} in #eta bin: ")+=i,0,"GeV"));
      
      putHist(roorvNIPionGenEt[i] = new RooRealVar(TString("roorvNIPionGenEt")+=i,TString("NI #pi^{#pm} Gen E_{T} in #eta bin: ")+=i,0,"GeV"));
      putHist(roorvNIPionTPGSumEt[i] = new RooRealVar(TString("roorvNIPionTPGSumEt")+=i,TString("NI #pi^{#pm} TPG Sum  E_{T} in #eta bin: ")+=i,0,"GeV"));

      putHist(roodsPhotonEtvsGenEt[i] = new RooDataSet(TString("roodsPhotonEtvsGenEt")+=i,TString("#gamma TPG Sum E_{T} vs. Gen E_{T} in #eta bin: ")+=i,
						       RooArgSet(*roorvPhotonTPGSumEt[i],*roorvPhotonGenEt[i],TString("#gammatpgvsgen")+=i)));
      putHist(roodsNIPionEtvsGenEt[i] = new RooDataSet(TString("roodsNIPionEtvsGenEt")+=i,TString("NI #pi^{#pm} TPG Sum E_{T} vs. Gen E_{T} in #eta bin: ")+=i,
						       RooArgSet(*roorvNIPionTPGSumEt[i],*roorvNIPionGenEt[i],TString("NI#piontpgvsgen")+=i)));


      
      putHist(hPhotonDeltaEOverE[i] = new TH1F(TString("gPhotonDeltaEOverE")+=i,TString("#gamma #frac{#DeltaE_{T}}{E_{T}} in #eta Bin: ")+=i,
					       199,-2,2));

      putHist(hPionDeltaEOverE[i] = new TH1F(TString("gPionDeltaEOverE")+=i,TString("#pi^{#pm} #frac{#DeltaE_{T}}{E_{T}} in #eta Bin: ")+=i,
					     199,-2,2));
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
						      TString("#frac{Corrected #E_{T}}{Uncorrected E_{T}} vs. #eta Bin in E_{T} Bin ") += i,
						      25,0.5,25.5));
    }
  
  putHist(hDeltaEtPeakvsEtaBinAllEt_uc = new TH1F("hDeltaEtPeakvsEtaBinAllEt_uc",
						  "Uncorrected #DeltaE_{T} vs. #eta Bin, All E_{T}",
						  25,0.5,25.5));
  putHist(hDeltaEtPeakvsEtaBinAllEt_c = new TH1F("hDeltaEtPeakvsEtaBinAllEt_c",
						 "Corrected #DeltaE_{T} vs. #eta Bin, All E_{T}",
						 25,0.5,25.5));
  putHist(hDeltaEtPeakRatiovsEtaBinAllEt = new TH1F("hDeltaEtPeakRatiovsEtaBinAllEt",
						    "#frac{Corrected E_{T}}{Uncorrected E_{T} vs. #eta Bin, All E_{T}",
						    25,0.5,25.5));
}

std::vector<L1RCTGenCalibrator::generator>
L1RCTGenCalibrator::overlaps(const std::vector<generator>& v) const
{
  std::vector<generator> result;

  if(v.begin() == v.end()) return result;

  for(std::vector<generator>::const_iterator i = v.begin(); i != v.end() - 1; ++i)
    {      
      for(std::vector<generator>::const_iterator j = i + 1; j != v.end(); ++j)
	{
	  double delta_r;

	  deltaR(i->eta, i->phi, j->eta, j->phi, delta_r);

	  if(delta_r < .5)
	    {
	      if(find<generator>(*i, result) == -1) 
		result.push_back(*i); 
	      if(find<generator>(*j, result) == -1)
		result.push_back(*j);
	    }	    
	}
    }

  return result;
}
