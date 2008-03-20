/*
 * \file L1THCALTPGXAna.cc
 *
 * $Date: 2008/03/14 20:35:46 $
 * $Revision: 1.12 $
 * \author J. Berryhill
 *
 * $Log: L1THCALTPGXAna.cc,v $
 * Revision 1.12  2008/03/14 20:35:46  berryhil
 *
 *
 * stripped out obsolete parameter settings
 *
 * rpc tpg restored with correct dn access and dbe handling
 *
 * Revision 1.11  2008/03/12 17:24:24  berryhil
 *
 *
 * eliminated log files, truncated HCALTPGXana histo output
 *
 * Revision 1.10  2008/03/01 00:40:00  lat
 * DQM core migration.
 *
 * Revision 1.9  2008/02/13 17:50:14  aurisano
 * bugfixes
 *
 * Revision 1.6  2008/01/02 11:54:15  elmer
 * Add missing math.h and TMath.h includes
 *
 * Revision 1.5  2007/12/21 17:41:21  berryhil
 *
 *
 * try/catch removal
 *
 * Revision 1.4  2007/12/05 14:03:19  berryhil
 *
 *
 * full functioning hcal tpg analyzer
 * reorganized tpg plots in subfolders
 *
 * Revision 1.3  2007/12/04 14:33:48  berryhil
 *
 *
 * hcal tpg xana activation
 *
 * Revision 1.2  2007/11/29 01:02:41  aurisano
 * changes to histo bounds
 *
 * Revision 1.1  2007/11/28 17:41:19  aurisano
 * New L1 Hcal monitor
 *
 * Revision 1.4  2007/06/12 19:32:53  berryhil
 *
 *
 * config files now include hcal tpg monitoring modules
 *
 * Revision 1.3  2007/02/23 22:00:16  wittich
 * add occ (weighted and unweighted) and rank histos
 *
 *
 */

#include "DQM/L1TMonitor/interface/L1THCALTPGXAna.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "DQMServices/Core/interface/DQMStore.h"
//#include "DQM/L1TMonitor/interface/hcal_root_prefs.h"
#include "TMath.h"

using namespace edm;

// Local definitions for the limits of the histograms
const unsigned int RTPBINS = 101;
const float RTPMIN = -0.5;
const float RTPMAX = 100.5;

const unsigned int TPPHIBINS = 72;
const float TPPHIMIN = 0.5;
const float TPPHIMAX = 72.5;

const unsigned int TPETABINS = 65;
const float TPETAMIN = -32.5;
const float TPETAMAX = 32.5;

const unsigned int effBins = 50;
const float effMinHBHE = -0.5;
const float effMaxHBHE = 5.5;
const float effMinHF = -0.5;
const float effMaxHF = 2.5;

const unsigned int ratiobins = 100;
const float ratiomin = 0.0;
const float ratiomax = 1.0;

const unsigned int tvsrecbins = 100;
const float tvsrecmin = 0.0;
const float tvsrecmax = 100.0;

const unsigned int effcombo = 6472;
const float effcombomin = -3272;
const float effcombomax = 3272;

const unsigned int fgbunchbins = 10;
const float fgbunchmin = 0;
const float fgbunchmax = 9;

const unsigned int fgbdiffbins = 20;
const float fgbdiffmin = -10;
const float fgbdiffmax = 10;

const unsigned int fgtdiffbins = 20;
const float fgtdiffmin = -250;
const float fgtdiffmax = 250;


L1THCALTPGXAna::L1THCALTPGXAna(const ParameterSet& ps)
  : hcaltpgSource_( ps.getParameter< InputTag >("hcaltpgSource") ),
    hbherecoSource_( ps.getParameter< InputTag >("hbherecoSource") ),
    hfrecoSource_( ps.getParameter< InputTag >("hfrecoSource") )
{
  // verbosity switch
  verbose_ = ps.getUntrackedParameter<bool>("verbose", false);
  if(verbose_) std::cout << "L1THCALTPGXAna: constructor...." << std::endl;

  //fake cut
  fakeCut_ = ps.getUntrackedParameter<double>("fakeCut",0.0);


  dbe = NULL;
  if ( ps.getUntrackedParameter<bool>("DQMStore", false) ) 
  {
    dbe = Service<DQMStore>().operator->();
    dbe->setVerbose(0);
  }

  outputFile_ = ps.getUntrackedParameter<std::string>("outputFile", "");
  if ( outputFile_.size() != 0 ) {
    std::cout << "L1T Monitoring histograms will be saved to " << outputFile_.c_str() << std::endl;
  }

  bool disable = ps.getUntrackedParameter<bool>("disableROOToutput", false);
  if(disable){
    outputFile_="";
  }


  if ( dbe !=NULL ) {
    dbe->setCurrentFolder("L1T/L1THCALTPGXAna");
  }

}

L1THCALTPGXAna::~L1THCALTPGXAna()
{
}

void L1THCALTPGXAna::beginJob(const EventSetup& iSetup)
{
  nev_ = 0;
  
  // get hold of back-end interface
  DQMStore* dbe = 0;
  dbe = Service<DQMStore>().operator->();
  if ( dbe ) 
    {
      dbe->setCurrentFolder("L1T/L1THCALTPGXAna");
      dbe->rmdir("L1T/L1THCALTPGXAna");
    }

  if ( dbe ) 
    {
      dbe->setCurrentFolder("L1T/L1THCALTPGXAna");
      //2-D plots
      hcalTpEtEtaPhi_ = 
	dbe->book2D("HcalTpEtEtaPhi", "HCAL TP E_{T}", TPETABINS, TPETAMIN,
		    TPETAMAX, TPPHIBINS, TPPHIMIN, TPPHIMAX);
      hcalTpOccEtaPhi_ =
	dbe->book2D("HcalTpOccEtaPhi", "HCAL TP OCCUPANCY", TPETABINS,
		    TPETAMIN, TPETAMAX, TPPHIBINS, TPPHIMIN, TPPHIMAX);
      hcalTpSat_ = 
	dbe->book2D("HcalSaturation", "HCAL Satuation", TPETABINS,
		    TPETAMIN, TPETAMAX, TPPHIBINS, TPPHIMIN, TPPHIMAX);
      hcalFakes_ =
	dbe->book2D("HcalFakes","Number of Fakes", TPETABINS, TPETAMIN,
		    TPETAMAX, TPPHIBINS, TPPHIMIN, TPPHIMAX);
      hcalNoFire_ =
	dbe->book2D("HcalNoFire","Highest Energy with TP = 0", TPETABINS, TPETAMIN,
		    TPETAMAX, TPPHIBINS, TPPHIMIN, TPPHIMAX);
      hcalTpgvsRec1_ = 
	dbe->book2D("TPGvsREC1", "TPG vs Rec Hit LUT 1", tvsrecbins, tvsrecmin, tvsrecmax, 
		    tvsrecbins, tvsrecmin, tvsrecmax);
      hcalTpgvsRec2_ = 
	dbe->book2D("TPGvsREC2", "TPG vs Rec Hit LUT 2", tvsrecbins, tvsrecmin, tvsrecmax, 
		    tvsrecbins, tvsrecmin, tvsrecmax);
      hcalTpgvsRec3_ = 
	dbe->book2D("TPGvsREC3", "TPG vs Rec Hit LUT 3", tvsrecbins, tvsrecmin, tvsrecmax, 
		    tvsrecbins, tvsrecmin, tvsrecmax);
      hcalTpgvsRec4_ = 
	dbe->book2D("TPGvsREC4", "TPG vs Rec Hit LUT 4", tvsrecbins, tvsrecmin, tvsrecmax, 
		    tvsrecbins, tvsrecmin, tvsrecmax);

      //1-D plots
      hcalTpRank_ =
	dbe->book1D("HcalTpRank", "HCAL TP RANK", RTPBINS, RTPMIN, RTPMAX);
      hcalEffDen_1_ = 
	dbe->book1D("HcalAll1","HCAL All Hits - 1",effBins,effMinHBHE,effMaxHBHE);
      hcalEffNum_1_ =
	dbe->book1D("HcalTP1","HCAL Hits with TP - 1",effBins,effMinHBHE,effMaxHBHE);
      hcalEffDen_2_ =
	dbe->book1D("HcalAll2","HCAL All Hits - 2",effBins,effMinHBHE,effMaxHBHE);
      hcalEffNum_2_ =
	dbe->book1D("HcalTP2","HCAL Hits with TP - 2",effBins,effMinHBHE,effMaxHBHE);
      hcalEffDen_3_ =
	dbe->book1D("HcalAll3","HCAL All Hits - 3",effBins,effMinHBHE,effMaxHBHE);
      hcalEffNum_3_ =
	dbe->book1D("HcalTP3","HCAL Hits with TP - 3",effBins,effMinHBHE,effMaxHBHE);
      hcalEffDen_4_ =
	dbe->book1D("HcalAll4","HCAL All Hits - 4",effBins,effMinHF,effMaxHF);
      hcalEffNum_4_ =
	dbe->book1D("HcalTP4","HCAL Hits with TP - 4",effBins,effMinHF,effMaxHF);
      hcalTpgRatiom1_ = 
	dbe->book1D("HcalTPGRatiom1", "Hcal Ration of E in bin SOI -1", ratiobins, ratiomin, ratiomax);
      hcalTpgRatioSOI_ = 
	dbe->book1D("HcalTPGRatiSOI", "Hcal Ration of E in bin SOI", ratiobins, ratiomin, ratiomax);
      hcalTpgRatiop1_ = 
	dbe->book1D("HcalTPGRatiop1", "Hcal Ration of E in bin SOI +1", ratiobins, ratiomin, ratiomax);
      hcalTpgRatiop2_ = 
	dbe->book1D("HcalTPGRatiop2", "Hcal Ration of E in bin SOI +21", ratiobins, ratiomin, ratiomax);
      hcalTpgfgperbunch_ =
	dbe->book1D("HcalFGperBunch", "Fine grain per bunch", fgbunchbins, fgbunchmin, fgbunchmax);
      hcalTpgfgbindiff_ =
	dbe->book1D("HcalFGbindiff", "Fine grain bunch difference", fgbdiffbins, fgbdiffmin, fgbdiffmax);
      hcalTpgfgtimediff_ =
	dbe->book1D("HcalFGtimediff", "Fine grain time diff", fgtdiffbins, fgtdiffmin, fgtdiffmax);


      if (0){
      dbe->setCurrentFolder("L1T/L1THCALTPGXAna/EffByChannel");
      //efficiency histos for HBHE
      for (int i=0; i < 56; i++)
	{      
	  char hname[20],htitle[20];
          char dirname[80];
 	  int ieta, iphi;
          if (i<28) ieta = i-28;
	  else ieta = i-27;
          sprintf(dirname,"L1T/L1THCALTPGXAna/EffByChannel/EtaTower%d",ieta);
          dbe->setCurrentFolder(dirname);
	  for (int j=0; j < 72; j++) 
	    {
	      iphi = j+1;
	      if (i<28) ieta = i-28;
	      else ieta = i-27;
              sprintf(hname,"eff_%d_%d_num",ieta,iphi);
              sprintf(htitle,"Eff Num <%d,%d>",ieta,iphi);
              hcalEffNum_HBHE[i][j] = dbe->book1D(hname, htitle, effBins,effMinHBHE,effMaxHBHE);
	      sprintf(hname,"eff_%d_%d_den",ieta,iphi);
	      sprintf(htitle,"Eff Den <%d,%d>",ieta,iphi); 
	      hcalEffDen_HBHE[i][j] = dbe->book1D(hname, htitle, effBins,effMinHBHE,effMaxHBHE);
	    }	     
	}
      //efficiency histos for HF
      for (int i=0; i < 8; i++)
	{
	  char hname[20],htitle[20];
          char dirname[80];
          int ieta, iphi;
          if (i<4) ieta = i-32;
	  else ieta = i+25;
          sprintf(dirname,"L1T/L1THCALTPGXAna/EffByChannel/EtaTower%d",ieta);
          dbe->setCurrentFolder(dirname);
	  for (int j=0; j < 18; j++)
	    {
	      iphi = j*4+1;
	      if (i<4) ieta = i-32;
	      else ieta = i+25;
	      sprintf(hname,"eff_%d_%d_num",ieta,iphi);
              sprintf(htitle,"Eff Num <%d,%d>",ieta,iphi);
	      hcalEffNum_HF[i][j] = dbe->book1D(hname, htitle, effBins,effMinHF,effMaxHF);
	      sprintf(hname,"eff_%d_%d_den",ieta,iphi);
              sprintf(htitle,"Eff Den <%d,%d>",ieta,iphi);
              hcalEffDen_HF[i][j] = dbe->book1D(hname, htitle, effBins,effMinHF,effMaxHF);
	    }
	}
      }

    }  
}


void L1THCALTPGXAna::endJob(void)
{
  if(verbose_) std::cout << "L1THCALTPGXAna: end job...." << std::endl;
  LogInfo("EndJob") << "analyzed " << nev_ << " events"; 

 if ( outputFile_.size() != 0  && dbe ) dbe->save(outputFile_);

 return;
}

void L1THCALTPGXAna::analyze(const Event& iEvent, const EventSetup& iSetup)
{
  nev_++; 
  if(verbose_) std::cout << "L1THCALTPGXAna: analyze...." << std::endl;

  edm::Handle<HcalTrigPrimDigiCollection> hcalTpgs;
  iEvent.getByLabel(hcaltpgSource_, hcalTpgs);
    
  if (!hcalTpgs.isValid())
    {
      edm::LogInfo("DataNotFound") << "can't find HCAL TPG's with label "
				     << hcaltpgSource_.label() ;
      return;
    }

  Handle<HBHERecHitCollection> hbhe_rec;
  
  
  iEvent.getByLabel(hbherecoSource_, hbhe_rec);
   
  if (!hbhe_rec.isValid())
    {
      edm::LogInfo("DataNotFound") << "can't find hbhe rec hits's with label "
                                     << hbherecoSource_.label() ;
      return;
    }

  Handle<HFRecHitCollection> hf_rec;
  iEvent.getByLabel(hfrecoSource_, hf_rec);
    
   if (!hf_rec.isValid())
    {
      edm::LogInfo("DataNotFound") << "can't find hf rec hits's with label "
                                     << hfrecoSource_.label() ;
      return;
    }

  std::vector<HcalTrigTowerDetId> towerids;
  Rec_towers.clear();
  double rec_e, eta1, eta2, et2e, eta_avg, rece_m, rece_p1, rece_p2;
  int rank, ieta, iphi, icombo;
 
  for(HBHERecHitCollection::const_iterator hbhe_iter = hbhe_rec->begin(); hbhe_iter != hbhe_rec->end(); ++hbhe_iter)
    {
      towerids = theTrigTowerGeometry.towerIds(hbhe_iter->id());
      assert(towerids.size() == 2 || towerids.size() == 1);
      for(unsigned int n = 0; n < towerids.size(); n++)
        {
          Rec_towers.insert(IdtoEnergy::value_type(towerids[n],hbhe_iter->energy()/towerids.size()));
        }
    }

  for(HFRecHitCollection::const_iterator hf_iter = hf_rec->begin(); hf_iter != hf_rec->end(); ++hf_iter)
    {
      towerids = theTrigTowerGeometry.towerIds(hf_iter->id());
      assert(towerids.size() == 2 || towerids.size() == 1);
      for(unsigned int n = 0; n < towerids.size(); n++)
        {
          Rec_towers.insert(IdtoEnergy::value_type(towerids[n],hf_iter->energy()/towerids.size()));
        }
    }

  numFG=0;
  for ( HcalTrigPrimDigiCollection::const_iterator tpg_iter = hcalTpgs->begin(); tpg_iter != hcalTpgs->end(); ++tpg_iter ) 
    {
       float ratioTotal = 0;
       float ratiom1 = 0;
       float ratiosoi= 0;
       float ratiop1 = 0;
       float ratiop2 = 0;;
      //get rec energy
      rec_e = 0.0;
      for(IdtoEnergy::iterator rec_iter = Rec_towers.lower_bound(tpg_iter->id()); rec_iter != Rec_towers.upper_bound(tpg_iter->id()); ++rec_iter)
        {
          rec_e += rec_iter->second;
        }

      //get ieta and iphi
      ieta = tpg_iter->id().ieta();
      iphi = tpg_iter->id().iphi();

      //get rank
      rank = tpg_iter->SOI_compressedEt();     

      //get eta bounds of tower (eta1 and eta2)
      theTrigTowerGeometry.towerEtaBounds(ieta,eta1,eta2);

      //get average eta of tower from eta1 and eta2
      eta_avg = find_eta(eta1,eta2);

      //conversion factor et -> e
      et2e = TMath::CosH(eta_avg);

      if (TMath::Abs(ieta) >= 29) rec_e = rec_e/et2e;

      if (0){
      //fill individual num. and den. of efficiency plots
      if (TMath::Abs(ieta) >= 29)
	{
	  if (ieta <0)
	    {
	      hcalEffDen_HF[(ieta+32)][iphi/4]->Fill(rec_e);
	      if (rank !=0)
		{
		  hcalEffNum_HF[(ieta+32)][iphi/4]->Fill(rec_e);
		}
	    }
	  else
	    {
              hcalEffDen_HF[(ieta-25)][iphi/4]->Fill(rec_e);
              if (rank !=0)
                {
                  hcalEffNum_HF[(ieta-25)][iphi/4]->Fill(rec_e);
                }
            }
	}
      else
	{
	  //account for there being no ieta=0
	  if (ieta < 0) 
	    {
	      hcalEffDen_HBHE[ieta+28][iphi-1]->Fill(rec_e);
	      if (rank !=0)
		{
		  hcalEffNum_HBHE[ieta+28][iphi-1]->Fill(rec_e);
		}
	    }
	  else
	    {
	      hcalEffDen_HBHE[ieta+27][iphi-1]->Fill(rec_e);
	      if (rank !=0)
		{
		  hcalEffNum_HBHE[ieta+27][iphi-1]->Fill(rec_e);
		}
	    }
	}
      }

      //fill num. and denom. of efficiency plots
      if (TMath::Abs(ieta) <= 20)
	{
	  hcalEffDen_1_->Fill(rec_e);
	  if (rank != 0)
	    {
	      hcalEffNum_1_->Fill(rec_e);
	      hcalTpgvsRec1_->Fill(rec_e,rank);
	    }
	}
      else if (TMath::Abs(ieta) <= 26)
	{
	  hcalEffDen_2_->Fill(rec_e);
	  if (rank != 0)
	    {
	      hcalEffNum_2_->Fill(rec_e);
	      hcalTpgvsRec2_->Fill(rec_e,rank);
	    }
	}
      else if (TMath::Abs(ieta) <= 28)
	{
          hcalEffDen_3_->Fill(rec_e);
	  if (rank != 0)
	    {
	      hcalEffNum_3_->Fill(rec_e);
	      hcalTpgvsRec3_->Fill(rec_e,rank);
	    }
	}
      else
	{
	  //fill HF with Et rather than E (triggering is done in Et)
	  if(et2e != 0) { hcalEffDen_4_->Fill(rec_e); }
	  if (rank != 0)
	    {
	      hcalEffNum_4_->Fill(rec_e);
	      hcalTpgvsRec4_->Fill(rec_e,rank);
	    }
	}
      
      if ( rank != 0 ) 
	{
	  // occupancy maps (weighted and unweighted
	  hcalTpOccEtaPhi_->Fill(ieta,iphi);
	  hcalTpEtEtaPhi_->Fill(ieta,iphi,rank);
	  if(rank == 1024)
	    {
	      hcalTpSat_->Fill(ieta,iphi);
	    }
	  // et
	  hcalTpRank_->Fill(rank);
	  if (rec_e < fakeCut_) { hcalFakes_->Fill(ieta,iphi); }
	} 
      else
	{
	  //add 33 to ieta to get proper bin number
	  double highest_energy  = hcalNoFire_->getBinContent(ieta+33,iphi);
	  if (highest_energy < rec_e) 
	   {
	     hcalNoFire_->setBinContent(ieta+33,iphi,rec_e); 
	   }
	} 

      for(int i = 0; i<10; i++)
        {
          if(tpg_iter->sample(i).fineGrain())
            {
              hcalTpgfgperbunch_->Fill(i);
              numFG++;
              if(numFG < 3)
                {
                  if(tpg_iter->id().iphi() < 37) binfg2 = i;
                  else binfg1 = i;
                }
	      
            }
        }

      if (verbose_)
	{
	  std::cout << "size  " <<  tpg_iter->size() << std::endl;
	  std::cout << "iphi  " <<  tpg_iter->id().iphi() << std::endl;
	  std::cout << "ieta  " <<  tpg_iter->id().ieta() << std::endl;
	  std::cout << "compressed Et  " <<  tpg_iter->SOI_compressedEt() << std::endl;
	  std::cout << "FG bit  " <<  tpg_iter->SOI_fineGrain() << std::endl;
	  std::cout << "raw  " <<  tpg_iter->t0().raw() << std::endl;
	  std::cout << "raw Et " <<  tpg_iter->t0().compressedEt() << std::endl;
	  std::cout << "raw FG " <<  tpg_iter->t0().fineGrain() << std::endl;
	  std::cout << "raw slb " <<  tpg_iter->t0().slb() << std::endl;
	  std::cout << "raw slbChan " <<  tpg_iter->t0().slbChan() << std::endl;
	  std::cout << "raw slbAndChan " <<  tpg_iter->t0().slbAndChan() << std::endl;
	  std::cout << "reco energy " << rec_e << std::endl;
	  std::cout << "tower eta " << eta_avg << std::endl;
	}
    }
}

double find_eta(double eta_start, double eta_end)
{
  double theta_avg = TMath::ATan(TMath::Exp(-eta_start)) + TMath::ATan(TMath::Exp(-eta_end));
  double average = -TMath::Log(TMath::Tan(theta_avg/2.0));
  return average;
}
