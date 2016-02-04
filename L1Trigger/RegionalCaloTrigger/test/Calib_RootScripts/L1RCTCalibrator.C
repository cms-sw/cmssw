#define L1RCTCalibrator_cxx
#include "L1RCTCalibrator.h"
#include <TH2.h>
#include <TStyle.h>
#include <TCanvas.h>
#include <iomanip>

#include <fstream>

void L1RCTCalibrator::Loop()
{
//   In a ROOT session, you can do:
//      Root > .L L1RCTCalibrator.C
//      Root > L1RCTCalibrator t
//      Root > t.GetEntry(12); // Fill t data members with entry number 12
//      Root > t.Show();       // Show values of entry 12
//      Root > t.Show(16);     // Read and show values of entry 16
//      Root > t.Loop();       // Loop on all entries
//

//     This is the loop skeleton where:
//    jentry is the global entry number in the chain
//    ientry is the entry number in the current Tree
//  Note that the argument to GetEntry must be:
//    jentry for TChain::GetEntry
//    ientry for TTree::GetEntry and TBranch::GetEntry
//
//       To read only selected branches, Insert statements like:
// METHOD1:
//    fChain->SetBranchStatus("*",0);  // disable all branches
//    fChain->SetBranchStatus("branchname",1);  // activate branchname
// METHOD2: replace line
//    fChain->GetEntry(jentry);       //read all branches
//by  b_branchname->GetEntry(ientry); //read only this branch
   if (fChain == 0) return;

   Long64_t nentries = fChain->GetEntriesFast();

   Long64_t nbytes = 0, nb = 0;
   for (Long64_t jentry=0; jentry<nentries;jentry++) {
      Long64_t ientry = LoadTree(jentry);
      if (ientry < 0) break;
      nb = fChain->GetEntry(jentry);   nbytes += nb;
   }
}

void L1RCTCalibrator::makeCalibration()
{
  if (fChain == 0) return;

   Long64_t nentries = fChain->GetEntries();
   unsigned prev = 0;

   std::cout << nentries << " to Process!" << std::endl;

   Long64_t nbytes = 0, nb = 0;
   for (Long64_t jentry=0; jentry<nentries;jentry++) {
      Long64_t ientry = LoadTree(jentry);
      if (ientry < 0) break;

      int check = (int)(100*(((double)jentry + 1)/ nentries));

      if(debug_ < 1 && ( jentry == 0 || check - prev == 1 ))
	{ 
	  prev = check;
	  std::cout << std::fixed << '\r' <<  std::setprecision(1) << 100*(((double)jentry + 1)/ nentries) << "% data caching finished!" << std::flush;
	}


      nb = fChain->GetEntry(jentry);   
      nbytes += nb;
      
      event_data temp;
            
      temp.event = Event_event;
      temp.run = Event_run;

      //std::cout << temp.event << ' ' << temp.run << ' ' << Generator_nGen << ' ' <<  Region_nRegions << ' ' << CaloTPG_nTPG << std::endl;

      for(unsigned i = 0; i < Generator_nGen; ++i)
	{
	  generator tgen;
	  tgen.particle_type = Generator_particle_type[i];
	  tgen.et = Generator_et[i];
	  tgen.phi = Generator_phi[i];
	  tgen.eta = Generator_eta[i];
	  temp.gen_particles.push_back(tgen);
	}
      for(unsigned i = 0; i < Region_nRegions; ++i)
	{
	  region treg;
	  treg.linear_et = Region_linear_et[i];
	  treg.ieta = Region_ieta[i];
	  treg.iphi = Region_iphi[i];
	  treg.eta = Region_eta[i];
	  treg.phi = Region_phi[i];	  
	  temp.regions.push_back(treg);
	}
      for(unsigned i = 0; i < CaloTPG_nTPG; ++i)
	{
	  tpg ttpg;
	  ttpg.ieta = CaloTPG_ieta[i];
	  ttpg.iphi = CaloTPG_iphi[i];
	  ttpg.eta = CaloTPG_eta[i];
	  ttpg.phi = CaloTPG_phi[i];
	  ttpg.ecalEt = CaloTPG_ecalEt[i];
	  ttpg.hcalEt = CaloTPG_hcalEt[i];
	  ttpg.ecalE = CaloTPG_ecalE[i];
	  ttpg.hcalE = CaloTPG_hcalE[i];
	  temp.tpgs.push_back(ttpg);
	}      
      data_.push_back(temp);
   }
   
   std::cout << "\nDONE CACHING DATA TO MEMORY" << std::endl;

   //now that we have all the data stored in memory, let's process it.. quickly...

   if(debug_ > -1) std::cout << "------------------postProcessing()-------------------\n";
   int iph[28] = {0}, ipi[28] = {0}, ipi2[28] = {0},  nEvents = 0, nUseful = 0;
   
   prev = 0;

   std::cout << "Phase One: Collect Underpants" << std::endl;
   // first event data loop, calibrate ecal, hcal with NI pions
   for(std::vector<event_data>::const_iterator i = data_.begin(); i != data_.end(); ++i)
     {
       /*
	 for(int n = 0; n < 28; ++n)
	 {
	 std::cout << "Trigger Tower " <<  n + 1 << ": " << iph[n] << ' ' << ipi[n] << ' ' << ipi2[n] << std::endl;
	 }
       */

       int check = (int)(100*(((double)nEvents + 1)/ nentries));

       if(debug_ < 1 && ( nEvents == 0 || check - prev == 1 ))
	 { 
	   prev = check;
	   std::cout << std::fixed << '\r' << std::setprecision(1) << 100*(((double)nEvents + 1)/ nentries) << "% finished with polynomial fit!" << std::flush;
	 }

       nEvents++;
       hEvent->Fill(i->event);
       hRun->Fill(i->run);
       
       bool used_flag = false;
       
       std::vector<generator>::const_iterator gen = i->gen_particles.begin();
       std::vector<generator> ovls = overlaps(i->gen_particles);
       
       
      if(debug_ > 0) std::cout << "Finding overlapping selected particles in this event!" << std::endl;
      if(debug_ > 0)
	{
	  std::cout << "=======Overlapping accepted gen particles in event " << nEvents << "" << std::endl;
	  for(std::vector<generator>::const_iterator ov = ovls.begin(); ov != ovls.end(); ++ov)
	    std::cout << '\t' <<  ov->particle_type << " " << ov->et << " " << ov->eta << " " << ov->phi << "" << std::endl;
	  std::cout << "======================================================" << std::endl;
	}
      if(debug_ > 0) std::cout << "Done finding overlaps!" << std::endl;
      
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
		  if(ecalEtandDeltaR95.first > 0)
		    {
		      etaBin(fabs(matchedCentroid.first), sumieta);
		      
		      hPhotonDeltaEOverE_uncor[sumieta - 1]->Fill((gen->et - ecalEtandDeltaR95.first)/gen->et);
		      
		      if(debug_ > 0)
			{
			  int n_towers = tpgsNear(matchedCentroid.first, matchedCentroid.second, matchedTpgs, ecalEtandDeltaR95.second).size();
			  std::cout << "TPG sum near gen Photon with nearby non-zero RCT region found:" << std::endl
				    << "Number of Towers  : " << n_towers << " "
				    << "\tCentroid Eta      : " << matchedCentroid.first << " "
				    << "\tCentroid Phi      : " << matchedCentroid.second << " "
				    << "\tDelta R 95  (h+e) : " << etAndDeltaR95.second << " "
				    << "\nDelta R 95  (e)   : " << ecalEtandDeltaR95.second << " "
				    << "\tTotal Et in Cone  : " << etAndDeltaR95.first << " "
				    << "\tEcal Et in Cone   : " << ecalEtandDeltaR95.first << " ";
			}
		      
		      hPhotonDeltaR95[sumieta - 1]->Fill(etAndDeltaR95.second);
		      gPhotonEtvsGenEt[sumieta - 1]->SetPoint(iph[sumieta - 1]++, ecalEtandDeltaR95.first, gen->et);	      
		    }
		}  

	      if(abs(gen->particle_type) == 211)
		{
		  if(!used_flag) used_flag = true;
		  
		  double ecal = sumEt(matchedCentroid.first, matchedCentroid.second, matchedTpgs, etAndDeltaR95.second, true, false);
		  double hcal = sumEt(matchedCentroid.first, matchedCentroid.second, matchedTpgs, etAndDeltaR95.second, false, true);

		  int sumieta;
		  etaBin(fabs(matchedCentroid.first), sumieta);

		  hPionDeltaEOverE_uncor[sumieta - 1]->Fill((gen->et - (ecal + hcal))/gen->et);
		  
		  if( ecal < 1.0  && etAndDeltaR95.first > 0)
		    {
		      if(debug_ > 0)
			{
			  int n_towers = tpgsNear(matchedCentroid.first, matchedCentroid.second, matchedTpgs, etAndDeltaR95.second).size();
			 std::cout << "TPG sum near gen Charged Pion with nearby non-zero RCT region and little ECAL energy found:" << std::endl
				   << "Number of Towers  : " << n_towers << " "
				   << "\tCentroid Eta      : " << matchedCentroid.first << " "
				   << "\tCentroid Phi      : " << matchedCentroid.second << " "
				   << "\tDelta R 95  (h+e) : " << etAndDeltaR95.second << " "
				   << "\nTotal Et in Cone  : " << etAndDeltaR95.first << " "
				   << "\tEcal Et in Cone   : " << ecal << " "
				   << "\tHcal Et in Cone   : " << hcal << " ";
			}
		      		      
		      hNIPionDeltaR95[sumieta - 1]->Fill(etAndDeltaR95.second);
		      gNIPionEtvsGenEt[sumieta - 1]->SetPoint(ipi[sumieta - 1]++, hcal, gen->et);
		    }
		  else if(etAndDeltaR95.first > 0)
		    {
		      if(debug_ > 0) 
			{
			  int n_towers = tpgsNear(matchedCentroid.first, matchedCentroid.second, matchedTpgs, etAndDeltaR95.second).size();
			  std::cout << "TPG sum near gen Charged Pion with nearby non-zero RCT region found:" << std::endl
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
  
  std::cout << "\nBegin Fit" << std::endl;
  for(int i = 0; i < 28; ++i)
    {
      std::cout << "Photon and NI/I Pion Counts: " << iph[i] << ' ' << ipi[i] << ' ' << ipi2[i] << std::endl;
      pitot += ipi[i];
      phtot += iph[i];
      pitot2 += ipi2[i];

      if(iph[i] > 100)
	{
	  std::cout << "Fitting Photons!" << std::endl;
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
	  std::cout << "Fitting NI Pions!" << std::endl;
	  NIpion_low[i] = new TF1((TString("hcal_fit_low")+=i).Data(),"x**3++x**2++x",0,100);
	  NIpion_high[i] = new TF1((TString("hcal_fit_high")+=i).Data(),"x**3++x**2++x",0,100);
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
      if(false && ipi2[i] >100 && NIpion_low[i] && NIpion_high[i] && photon[i])
	{
	  std::cout << "Fitting Pions" << std::endl;
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
  std::cout << "End Fit" << std::endl;
  
  unsigned e = 0;

  std::cout << "Phase Two: ????" << std::endl;
  for(std::vector<event_data>::const_iterator i = data_.begin(); i != data_.end(); ++i)
    {

      int check = (int)(100*(((double)e + 1)/ nentries));

       if(debug_ < 1 && ( e == 0 || check - prev == 1 ))
	 { 
	   prev = check;
	   std::cout << std::fixed << '\r' << std::setprecision(1) << 100*(((double)e + 1)/ nentries) << "% smearing factors!" << std::flush;
	 }
       ++e;

      std::vector<generator>::const_iterator gen = i->gen_particles.begin();

      std::vector<generator> ovls = overlaps(i->gen_particles);
      
      if(debug_ > 0) std::cout << "Finding overlapping selected particles in this event!" << std::endl;
      if(debug_ > 0)
	{
	  std::cout << "=======Overlapping accepted gen particles in event " << nEvents << "" << std::endl;
	  for(std::vector<generator>::const_iterator ov = ovls.begin(); ov != ovls.end(); ++ov)
	    std::cout << '\t' <<  ov->particle_type << " " << ov->et << " " << ov->eta << " " << ov->phi << "" << std::endl;
	  std::cout << "======================================================" << std::endl;
	}
      if(debug_ > 0) std::cout << "Done finding overlaps!" << std::endl;

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
		  		  
		  double ecal_c = sumEt(matchedCentroid.first, matchedCentroid.second, matchedTpgs, etAndDeltaR95.second, true, false, true);
		  
		  double deltaeovere = (gen->et - ecal_c)/gen->et;
		  hPhotonDeltaEOverE[sumieta - 1]->Fill(deltaeovere);
		  
		}  

	      if(abs(gen->particle_type) == 211)
		{
		  double et_c = sumEt(matchedCentroid.first, matchedCentroid.second, matchedTpgs, etAndDeltaR95.second, true, true, true);

		  int sumieta;
		  etaBin(fabs(matchedCentroid.first), sumieta);

		  double deltaeovere = (gen->et - et_c)/gen->et;
		  hPionDeltaEOverE[sumieta - 1]->Fill(deltaeovere);		  
		}
	    }	    
	}
    }

  std::cout << std::endl;

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

  e = 0;

  std::cout << "Phase 3: Profit!" << std::endl;
  for(std::vector<event_data>::const_iterator i = data_.begin(); i != data_.end(); ++i)
    {
      int check = (int)(100*(((double)e + 1)/ nentries));

       if(debug_ < 1 && ( e == 0 || check - prev == 1 ))
	 { 
	   prev = check;
	   std::cout << std::fixed << '\r' << std::setprecision(1) << 100*(((double)nEvents + 1)/ nentries) << "% finished with diagnostics!" << std::flush;
	 }
       ++e;

      std::vector<generator>::const_iterator gen = i->gen_particles.begin();

      std::vector<generator> ovls = overlaps(i->gen_particles);
      
      if(debug_ > 0) std::cout << "Finding overlapping selected particles in this event!" << std::endl;
      if(debug_ > 0)
	{
	  std::cout << "=======Overlapping accepted gen particles in event " << nEvents << "" << std::endl;
	  for(std::vector<generator>::const_iterator ov = ovls.begin(); ov != ovls.end(); ++ov)
	    std::cout << '\t' <<  ov->particle_type << " " << ov->et << " " << ov->eta << " " << ov->phi << "" << std::endl;
	  std::cout << "======================================================" << std::endl;
	}
      if(debug_ > 0) std::cout << "Done finding overlaps!" << std::endl;

      for(; gen != i->gen_particles.end(); ++gen)
	{	  
	  if(gen->particle_type != 22 && abs(gen->particle_type) != 211 || find<generator>(*gen, ovls) != -1) continue;
	  
	  double regionsum = sumEt(gen->eta,gen->phi,i->regions);
	  
	  if(regionsum > 0.0)
	    {	
	      std::vector<tpg> matchedTpgs = tpgsNear(gen->eta,gen->phi,i->tpgs);
	      std::pair<double,double> matchedCentroid = std::make_pair(avgEta(matchedTpgs),avgPhi(matchedTpgs));
	      std::pair<double,double> etAndDeltaR95 = showerSize(matchedTpgs);

	      double ecal = sumEt(matchedCentroid.first, matchedCentroid.second, matchedTpgs, etAndDeltaR95.second, true, false);
	      double hcal = sumEt(matchedCentroid.first, matchedCentroid.second, matchedTpgs, etAndDeltaR95.second, false, true);
	      double et_c = sumEt(matchedCentroid.first, matchedCentroid.second, matchedTpgs, etAndDeltaR95.second, true, true, true);

	      int sumieta;		  
	      etaBin(fabs(matchedCentroid.first), sumieta);

	      if(hcal/(ecal + hcal) > 0.05 && etAndDeltaR95.first > 0)
		et_c *= ( (he_high_smear_[sumieta - 1] != -999) ? he_high_smear_[sumieta - 1] : 1 );
	      else
		et_c *= ( (he_low_smear_[sumieta - 1] != -999) ? he_low_smear_[sumieta - 1] : 1 );

	      double deltaeovere = (gen->et - et_c)/gen->et;

	      if(gen->particle_type == 22)
		hPhotonDeltaEOverE_cor[sumieta - 1]->Fill(deltaeovere);
		
	      if(abs(gen->particle_type) == 211)
		{
		  ecal = sumEt(matchedCentroid.first, matchedCentroid.second, matchedTpgs, etAndDeltaR95.second, true, false);
		  hcal = sumEt(matchedCentroid.first, matchedCentroid.second, matchedTpgs, etAndDeltaR95.second, false, true, true);
		  hPionDeltaEOverE_cor[sumieta - 1]->Fill(deltaeovere);		
		}
	    }	    
	}
    }

  std::cout << std::endl;

  // Now fill quick diagnostic plots.
  for(int i = 0; i < 25; ++i)
    {
      Double_t upper = hPhotonDeltaEOverE_cor[i]->GetBinCenter(hPhotonDeltaEOverE_cor[i]->GetMaximumBin()) + hPhotonDeltaEOverE_cor[i]->GetRMS();
      Double_t lower = hPhotonDeltaEOverE_cor[i]->GetBinCenter(hPhotonDeltaEOverE_cor[i]->GetMaximumBin()) - hPhotonDeltaEOverE_cor[i]->GetRMS();
	  
      hPhotonDeltaEOverE_cor[i]->Fit("gaus",
				     "QELM",
				     "",
				     lower,
				     upper);
      
      hPhotonDeltaEtPeakvsEtaBinAllEt_c->SetBinContent(i+1,hPhotonDeltaEOverE_cor[i]->GetFunction("gaus")->GetParameter(1));
      hPhotonDeltaEtPeakvsEtaBinAllEt_c->SetBinError(i+1,hPhotonDeltaEOverE_cor[i]->GetFunction("gaus")->GetParError(1));

      upper = hPhotonDeltaEOverE_uncor[i]->GetBinCenter(hPhotonDeltaEOverE_uncor[i]->GetMaximumBin()) + hPhotonDeltaEOverE_uncor[i]->GetRMS();
      lower = hPhotonDeltaEOverE_uncor[i]->GetBinCenter(hPhotonDeltaEOverE_uncor[i]->GetMaximumBin()) - hPhotonDeltaEOverE_uncor[i]->GetRMS();
      
      hPhotonDeltaEOverE_uncor[i]->Fit("gaus",
				 "QELM",
				 "",
				 lower,
				 upper);
      
      hPhotonDeltaEtPeakvsEtaBinAllEt_uc->SetBinContent(i+1,hPhotonDeltaEOverE_uncor[i]->GetFunction("gaus")->GetParameter(1));
      hPhotonDeltaEtPeakvsEtaBinAllEt_uc->SetBinError(i+1,hPhotonDeltaEOverE_uncor[i]->GetFunction("gaus")->GetParError(1));

      upper = hPionDeltaEOverE_cor[i]->GetBinCenter(hPionDeltaEOverE_cor[i]->GetMaximumBin()) + hPionDeltaEOverE_cor[i]->GetRMS();
      lower = hPionDeltaEOverE_cor[i]->GetBinCenter(hPionDeltaEOverE_cor[i]->GetMaximumBin()) - hPionDeltaEOverE_cor[i]->GetRMS();
	  
      hPionDeltaEOverE_cor[i]->Fit("gaus",
				   "QELM",
				   "",
				   lower,
				   upper);
      
      hPionDeltaEtPeakvsEtaBinAllEt_c->SetBinContent(i+1,hPionDeltaEOverE_cor[i]->GetFunction("gaus")->GetParameter(1));
      hPionDeltaEtPeakvsEtaBinAllEt_c->SetBinError(i+1,hPionDeltaEOverE_cor[i]->GetFunction("gaus")->GetParError(1));

      upper = hPionDeltaEOverE_uncor[i]->GetBinCenter(hPionDeltaEOverE_uncor[i]->GetMaximumBin()) + hPionDeltaEOverE_uncor[i]->GetRMS();
      lower = hPionDeltaEOverE_uncor[i]->GetBinCenter(hPionDeltaEOverE_uncor[i]->GetMaximumBin()) - hPionDeltaEOverE_uncor[i]->GetRMS();
      
      hPionDeltaEOverE_uncor[i]->Fit("gaus",
				     "QELM",
				     "",
				     lower,
				     upper);
      
      hPionDeltaEtPeakvsEtaBinAllEt_uc->SetBinContent(i+1,hPionDeltaEOverE_uncor[i]->GetFunction("gaus")->GetParameter(1));
      hPionDeltaEtPeakvsEtaBinAllEt_uc->SetBinError(i+1,hPionDeltaEOverE_uncor[i]->GetFunction("gaus")->GetParError(1));
    }

  if(debug_ > -1)
    std::cout << "Completed L1RCTGenCalibrator::postProcessing() with the following results:" << std::endl
	      << "Total Events       : " << nEvents << "" << std::endl
	      << "Useful Events      : " << nUseful << "" << std::endl
	      << "Number of NI Pions : " << pitot << "" << std::endl
	      << "Number of I Pions  : " << pitot2 << "" << std::endl
	      << "Number of Photons  : " << phtot << "" << std::endl;

  ofstream out;
  out.open("RCTCalibration_cff.py");
  WriteHistos();
  printCfFragment(out);
  out.close();    
}

void L1RCTCalibrator::BookHistos()
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
      
      putHist(hPhotonDeltaEOverE[i] = new TH1F(TString("gPhotonDeltaEOverE")+=i,TString("Polynomial-fit-corrected #gamma #frac{#DeltaE_{T}}{E_{T}} in #eta Bin: ")+=i,
					       199,-2,2));

      putHist(hPionDeltaEOverE[i] = new TH1F(TString("gPionDeltaEOverE")+=i,TString("Polynomial-fit-corrected #pi^{#pm} #frac{#DeltaE_{T}}{E_{T}} in #eta Bin: ")+=i,
					     199,-2,2));

      putHist(hPhotonDeltaEOverE_cor[i] = new TH1F(TString("gPhotonDeltaEOverE_cor")+=i,TString("Corrected #gamma #frac{#DeltaE_{T}}{E_{T}} in #eta Bin: ")+=i,
						   199,-2,2));
      putHist(hPionDeltaEOverE_cor[i] = new TH1F(TString("gPionDeltaEOverE_cor")+=i,TString("Corrected #pi^{#pm} #frac{#DeltaE_{T}}{E_{T}} in #eta Bin: ")+=i,
						 199,-2,2));

      putHist(hPhotonDeltaEOverE_uncor[i] = new TH1F(TString("gPhotonDeltaEOverE_uncor")+=i,TString("Uncorrected #gamma #frac{#DeltaE_{T}}{E_{T}} in #eta Bin: ")+=i,
						     199,-2,2));
      putHist(hPionDeltaEOverE_uncor[i] = new TH1F(TString("gPionDeltaEOverE_uncor")+=i,TString("Uncorrected #pi^{#pm} #frac{#DeltaE_{T}}{E_{T}} in #eta Bin: ")+=i,
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
  
  putHist(hPhotonDeltaEtPeakvsEtaBinAllEt_uc = new TH1F("hPhotonDeltaEtPeakvsEtaBinAllEt_uc",
							"Photon Uncorrected #DeltaE_{T} vs. #eta Bin, All E_{T}",
							25,0.5,25.5));
  putHist(hPhotonDeltaEtPeakvsEtaBinAllEt_c = new TH1F("hPhotonDeltaEtPeakvsEtaBinAllEt_c",
						       "Photon Corrected #DeltaE_{T} vs. #eta Bin, All E_{T}",
						       25,0.5,25.5));
  putHist(hPhotonDeltaEtPeakRatiovsEtaBinAllEt = new TH1F("hPhotonDeltaEtPeakRatiovsEtaBinAllEt",
							  "Photon #frac{Corrected E_{T}}{Uncorrected E_{T} vs. #eta Bin, All E_{T}",
							  25,0.5,25.5));

  putHist(hPionDeltaEtPeakvsEtaBinAllEt_uc = new TH1F("hPionDeltaEtPeakvsEtaBinAllEt_uc",
							"Pion Uncorrected #DeltaE_{T} vs. #eta Bin, All E_{T}",
							25,0.5,25.5));
  putHist(hPionDeltaEtPeakvsEtaBinAllEt_c = new TH1F("hPionDeltaEtPeakvsEtaBinAllEt_c",
						       "Pion Corrected #DeltaE_{T} vs. #eta Bin, All E_{T}",
						       25,0.5,25.5));
  putHist(hPionDeltaEtPeakRatiovsEtaBinAllEt = new TH1F("hPionDeltaEtPeakRatiovsEtaBinAllEt",
							  "Pion #frac{Corrected E_{T}}{Uncorrected E_{T} vs. #eta Bin, All E_{T}",
							  25,0.5,25.5));
}

void L1RCTCalibrator::WriteHistos()
{
  output_->cd();

  for(std::vector<TObject*>::const_iterator i = hists_.begin(); i != hists_.end(); ++i)
    (*i)->Write();

  output_->Write();
  output_->Close();
}


double L1RCTCalibrator::uniPhi(const double& phi) const
{
  double result = ((phi < 0) ? phi + 2*M_PI : phi);
  while(result > 2*M_PI) result -= 2*M_PI;
  return result;
}

//calculate Delta R between two (eta,phi) coordinates
void L1RCTCalibrator::deltaR(const double& eta1, double phi1, 
			     const double& eta2, double phi2,double& dr) const
{
  double deta2 = std::pow(eta1-eta2,2.);  
  double dphi2 = std::pow(uniPhi(phi1)-uniPhi(phi2),2.);

  dr = std::sqrt(deta2 + dphi2);
}

void L1RCTCalibrator::etaBin(const double& veta, int& ieta) const
{
  double absEta = fabs(veta);

  if(absEta < maxEtaBarrel_)
    {
      ieta = static_cast<int>((absEta+0.000001)/deltaEtaBarrel_) + 1;
    }
  else
    {
      double temp = absEta - maxEtaBarrel_;
      int i = 0;
      while(temp > -0.0000001 && i < 8)
	{
	  temp -= endcapEta_[i++];
	}
      ieta = 20 + i;
    }
  ieta = ((veta < 0) ? -ieta : ieta);
}
 
void L1RCTCalibrator::etaValue(const int& ieta, double& veta) const
{
  int absEta = abs(ieta);

  if(absEta <= 20)
    {
      veta = (absEta-1)*0.0870 + 0.0870/2.;
    }
  else
    {
      int offset = abs(ieta) - 21;
      veta = maxEtaBarrel_;
      for(int i = 0; i < offset; ++i)
	veta += endcapEta_[i];
      veta += endcapEta_[offset]/2.;
    }
  veta = ((ieta < 0) ? -veta : veta);
}

void L1RCTCalibrator::phiBin(double vphi, int& iphi) const
{  
  iphi = static_cast<int>(uniPhi(vphi)/deltaPhi_);  
}

void L1RCTCalibrator::phiValue(const int& iphi, double& vphi) const
{
  vphi = iphi*deltaPhi_ + deltaPhi_/2.;
}

bool L1RCTCalibrator::sanityCheck() const
{
  for(int i = 1; i <= 28; ++i)
    {
      int j, l;
      double p, q;
      etaValue(i,p);
      etaBin(p,j);
      etaValue(-i,q);
      etaBin(q,l);
      if( i != j || -i != l)
	{
	  if(debug_ > -1)
	    std::cout << i <<  "\t" << p << "\t" << j << "\t" 
		      << -i << "\t" << q << "\t" << l << std::endl;
	  return false;
	}
    }
  for(int i = 0; i < 72; ++i)
    {
      int j;
      double p;
      phiValue(i,p);
      phiBin(p,j);
      if(i != j)
	{
	  if(debug_ > -1)
	    std::cout << i << "\t" << p << "\t" << j << std::endl;
	  return false;
	}
    }
  return true;
}

// energy, deltaR
std::pair<double,double> L1RCTCalibrator::showerSize(const std::vector<tpg>& tp, const double frac, const double& max_dr,
						     const bool& ecal, const bool& hcal) const
{
  double c_eta = avgEta(tp), c_phi = avgPhi(tp);
  double result = 0.0, 
    e_max = sumEt(c_eta, c_phi, tp, max_dr, ecal, hcal);
  
  double dr_iter = 0.0;
  
  do{
    result = sumEt(c_eta, c_phi, tp, dr_iter, ecal, hcal);
    dr_iter += 0.01;
  }while(result/e_max < frac);
  
  return std::make_pair(result,dr_iter);
}

double L1RCTCalibrator::sumEt(const double& eta, const double& phi, const std::vector<tpg>& tp, const double& dr, 
			      const bool& ecal, const bool& hcal, const bool& c, const double& crossover) const
{
  double delta_r, sum = 0.0;  

  for(std::vector<tpg>::const_iterator i = tp.begin(); i != tp.end(); ++i)
    {
      deltaR(eta,phi,i->eta,i->phi,delta_r);

      if(delta_r < dr)
	{
	  if(c)
	    {
	      int etabin = abs(i->ieta) - 1;
	      if(i->ecalE > .5 && ecal)
		{
		  if(ecal_[etabin][0] != -999 && ecal_[etabin][1] != -999 && ecal_[etabin][2] != -999) 		
		    sum += (ecal_[etabin][0]*std::pow(i->ecalEt,3.) +
			    ecal_[etabin][1]*std::pow(i->ecalEt,2.) +
			    ecal_[etabin][2]*i->ecalEt);
		  else
		    sum += i->ecalEt;
		}
	      if(i->hcalE > .5 && hcal)
		{
		  double crossterm = 0.0, hcal_c = 0.0;
		  if(i->ecalEt + i->hcalEt < crossover)
		    {
		      if(cross_[etabin][0] != -999 && cross_[etabin][1] != -999 && cross_[etabin][2] != -999 &&
			 cross_[etabin][3] != -999 && cross_[etabin][4] != -999 && cross_[etabin][5] != -999 &&
			 hcal_[etabin][0] != -999 && hcal_[etabin][1] != -999 && hcal_[etabin][2] != -999)
			{
			  crossterm = (cross_[etabin][0]*std::pow(i->ecalEt,2)*i->hcalEt +
				       cross_[etabin][1]*std::pow(i->hcalEt,2)*i->ecalEt +
				       cross_[etabin][2]*i->ecalEt*i->hcalEt +
				       cross_[etabin][3]*std::pow(i->ecalEt,3)*i->hcalEt +
				       cross_[etabin][4]*std::pow(i->hcalEt,3)*i->ecalEt +
				       cross_[etabin][5]*std::pow(i->ecalEt,2)*std::pow(i->hcalEt,2));
			  hcal_c = (hcal_[etabin][0]*std::pow(i->hcalEt,3.) +
				    hcal_[etabin][1]*std::pow(i->hcalEt,2.) +
				    hcal_[etabin][2]*i->hcalEt);
			}
		      else
			hcal_c = i->hcalEt;
		    }
		  else
		    {
		      if(hcal_high_[etabin][0] != -999 && hcal_high_[etabin][1] != -999 && hcal_high_[etabin][2] != -999)
			{
			  hcal_c = (hcal_high_[etabin][0]*std::pow(i->hcalEt,3.) +
				    hcal_high_[etabin][1]*std::pow(i->hcalEt,2.) +
				    hcal_high_[etabin][2]*i->hcalEt);
			}
		      else
			hcal_c = i->hcalEt;
		    }
		  sum += hcal_c + crossterm;
		}
	    }
	  else
	    {
	      if(i->ecalE > .5 && ecal) sum += i->ecalEt;
	      if(i->hcalE > .5 && hcal) sum += i->hcalEt;
	    }
	}
    }
  return sum;
}

double L1RCTCalibrator::sumEt(const double& eta, const double& phi, const std::vector<region>& regs, const double& dr) const
{
  double sum = 0.0, delta_r;

  for(std::vector<region>::const_iterator i = regs.begin(); i != regs.end(); ++i)
    {
      deltaR(eta,phi,i->eta,i->phi,delta_r);

      if(delta_r < dr) sum += i->linear_et*.5;
    }
  return sum;
}

double L1RCTCalibrator::avgPhi(const std::vector<tpg>& t) const
{
  double n = 0.0, d = 0.0;

  for(std::vector<tpg>::const_iterator i = t.begin(); i != t.end(); ++i)
    {
      n = (i->ecalEt + i->hcalEt)*i->phi;
      d = i->ecalEt + i->hcalEt;
    }

  return n/d;
}

double L1RCTCalibrator::avgEta(const std::vector<tpg>& t) const
{
  double n = 0.0, d = 0.0;

  for(std::vector<tpg>::const_iterator i = t.begin(); i != t.end(); ++i)
    {
      double temp_eta;
      etaValue(i->ieta, temp_eta);
      n = (i->ecalEt + i->hcalEt)*temp_eta;
      d = i->ecalEt + i->hcalEt;
    }
  return n/d;
}

std::vector<tpg> L1RCTCalibrator::tpgsNear(const double& eta, const double& phi, const std::vector<tpg>& tpgs, 
					   const double& dr) const
{
  std::vector<tpg> result;

  for(std::vector<tpg>::const_iterator i = tpgs.begin(); i != tpgs.end(); ++i)
    {
      double delta_r;
      deltaR(eta,phi,i->eta,i->phi,delta_r);
      
      if(delta_r < dr) result.push_back(*i);
    }
  return result;
}

//This prints out a nicely formatted .cfi file to be included in RCTConfigProducer.cfi
void L1RCTCalibrator::printCfFragment(std::ostream& out) const
{
  double* p = NULL;

  out.flush();
  
  out << ((python_) ? "import FWCore.ParameterSet.Config as cms\n\nrct_calibration = cms.PSet(" : "block rct_calibration = {") << std::endl;
  for(int i = 0; i < 6; ++i)
    {
      switch(i)
	{
	case 0:
	  p = const_cast<double*>(reinterpret_cast<const double*>(ecal_));
	  out << ((python_) ? "\tecal_calib_Lindsey = cms.vdouble(" : "\tvdouble ecal_calib_Lindsey = {") << std::endl;
	  break;
	case 1:
	  p = const_cast<double*>(reinterpret_cast<const double*>(hcal_));
	  out << ((python_) ? "\thcal_calib_Lindsey = cms.vdouble(" : "\tvdouble hcal_calib_Lindsey = {") << std::endl;
	  break;
	case 2:
	  p = const_cast<double*>(reinterpret_cast<const double*>(hcal_high_));
	  out << ((python_) ? "\thcal_high_calib_Lindsey = cms.vdouble(" : "\tvdouble hcal_high_calib_Lindsey = {") << std::endl;
	  break;
	case 3:
	  p = const_cast<double*>(reinterpret_cast<const double*>(cross_));
	  out << ((python_) ? "\tcross_terms_Lindsey = cms.vdouble(" : "\tvdouble cross_terms_Lindsey = {") << std::endl;
	  break;
	case 4:
	  p = const_cast<double*>(reinterpret_cast<const double*>(he_low_smear_));
	  out << ((python_) ? "\tHoverE_low_Lindsey = cms.vdouble(" : "\tvdouble HoverE_low_Lindsey = {") << std::endl;
	  break;
	case 5:
	  p = const_cast<double*>(reinterpret_cast<const double*>(he_high_smear_));
	  out << ((python_) ? "\tHoverE_high_Lindsey = cms.vdouble(" : "\tvdouble HoverE_high_Lindsey = {") << std::endl;
	};

      for(int j = 0; j < 28; ++j)
	{
	  if( p == reinterpret_cast<const double*>(ecal_) || p == reinterpret_cast<const double*>(hcal_) || 
	      p == reinterpret_cast<const double*>(hcal_high_) )
	    {	
	      double *q = p + 3*j;
	      if(q[0] != -999 && q[1] != -999 && q[2] != -999)
		{
		  out << "\t\t" << q[0] << ", " << q[1] << ", " << q[2];
		  out << ((j==27) ? "" : ",") << std::endl;
		}
	    }
	  else if( p == reinterpret_cast<const double*>(cross_) )
	    {
	      double *q = p + 6*j;
	      if(q[0] != -999 && q[1] != -999 && q[2] != -999 &&
		 q[3] != -999 && q[4] != -999 && q[5] != -999)
		{
		  out << "\t\t" << q[0] << ", " << q[1] << ", " << q[2] << ", "
		      << q[3] << ", " << q[4] << ", " << q[5];
		  out << ((j==27) ? "" : ",") << std::endl;
		}
	    }
	  else
	    {
	      double *q = p;
	      if(q[j] != -999)
		out << "\t\t" << q[j] << ((j==27) ? "" : ",") << std::endl;
	    }
	}
      if(python_)
	{
	  out << ((i != 5) ? "\t)," : "\t)") << std::endl;
	}
      else 
	out << "\t}" << std::endl;
    }
  out << ((python_) ? ")" : "}") << std::endl;
}

std::vector<generator>
L1RCTCalibrator::overlaps(const std::vector<generator>& v) const
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
