/* ******************************************
 * ZIterativeAlgorithmWithFit.cc
 *
 * 
 * Paolo Meridiani 06/07/2005
 * Rewritten for CMSSW 04/06/2007
 ********************************************/

#include "Calibration/Tools/interface/ZIterativeAlgorithmWithFit.h"
//#include "Calibration/Tools/interface/EcalCalibMap.h"

#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/TrackReco/interface/Track.h"

#include <TMath.h>
#include <TCanvas.h> 
#include "TH1.h"
#include "TH2.h"
#include "TF1.h"
#include "TH1F.h"
#include "TMinuit.h"
#include "TGraphErrors.h"
#include "THStack.h"
#include "TLegend.h"


#include <fstream>
#include <iostream>
#include <vector>

//#include "Tools.C"

//Scale and Bins for calibration factor histograms
#define MIN_RESCALE -0.5
#define MAX_RESCALE 0.5
#define NBINS_LOWETA 100
#define NBINS_HIGHETA 50

const double ZIterativeAlgorithmWithFit::M_Z_=91.187;

//  #if !defined(__CINT__)
//  ClassImp(Electron)
//  #endif

ZIterativeAlgorithmWithFit::ZIterativeAlgorithmWithFit()
{
  // cout<< "ZIterativeAlgorithmWithFit::Called Construnctor" << std::endl;
  numberOfIterations_=10;
  channels_=1;
  totalEvents_=0;
  currentEvent_=0;
  currentIteration_=0;
  optimizedCoefficients_.resize(channels_);
  calib_fac_.resize(channels_);
  weight_sum_.resize(channels_);
  electrons_.resize(1);
  massReco_.resize(1);
}

ZIterativeAlgorithmWithFit::ZIterativeAlgorithmWithFit(const edm::ParameterSet&  ps)
  //k, unsigned int events) 
{
  // std::cout<< "ZIterativeAlgorithmWithFit::Called Construnctor" << std::endl;
  numberOfIterations_=ps.getUntrackedParameter<unsigned int>("maxLoops",0);
  massMethod = ps.getUntrackedParameter<std::string>("ZCalib_InvMass","SCMass");
  calibType_= ps.getUntrackedParameter<std::string>("ZCalib_CalibType","RING"); 
  if (calibType_ == "RING")
    channels_=170;
  else if (calibType_ == "MODULE")
    channels_=144;  

  nCrystalCut_=ps.getUntrackedParameter<int>("ZCalib_nCrystalCut",-1);

  //Resetting currentEvent & iteration
  currentEvent_=0;
  currentIteration_=0;
  totalEvents_=0;

  //Reserving space for vectors (speedUp) 
//   electrons_.reserve(events);
//   massReco_.reserve(events);

  optimizedCoefficients_.resize(channels_);
  calib_fac_.resize(channels_);
  weight_sum_.resize(channels_);

  //Creating and booking histograms
  thePlots_ = new ZIterativeAlgorithmWithFitPlots;
  bookHistograms();

  //Setting up rescaling if needed
  UseStatWeights_=ps.getUntrackedParameter<bool>("ZCalib_UseStatWeights",false);
  if (UseStatWeights_) {
    WeightFileName_="weights.txt";
    StatWeights_.resize(channels_);
    getStatWeights(WeightFileName_);
    //    Event_Weight_.resize(events);
  }
}


void ZIterativeAlgorithmWithFit::bookHistograms()
{
  if (!thePlots_)
    return;

  for (unsigned int i2 = 0; i2 < numberOfIterations_; i2++) {
    for (unsigned int i1 = 0; i1 < channels_; i1++) {
      char histoName[200];
      char histoTitle[200];

      //WeightedRescaling factor
      sprintf(histoName, "WeightedRescaleFactor_channel_%d_Iteration_%d",i1, i2);
      sprintf(histoTitle, "WeightedRescaleFactor Channel_%d Iteration %d",i1, i2);
      if (i1>15 && i1<155)
	thePlots_->weightedRescaleFactor[i2][i1] = new TH1F(histoName, histoTitle, NBINS_LOWETA, MIN_RESCALE, MAX_RESCALE);
      else
	thePlots_->weightedRescaleFactor[i2][i1] = new TH1F(histoName, histoTitle, NBINS_HIGHETA, MIN_RESCALE, MAX_RESCALE);
      thePlots_->weightedRescaleFactor[i2][i1]->GetXaxis()->SetTitle("Rescale factor");
      thePlots_->weightedRescaleFactor[i2][i1]->GetYaxis()->SetTitle("a.u.");

      //UnweightedRescaling factor
      sprintf(histoName, "UnweightedRescaleFactor_channel_%d_Iteration_%d",i1, i2);
      sprintf(histoTitle, "UnweightedRescaleFactor Channel_%d Iteration %d",i1, i2);
      if (i1>15 && i1<155)
	thePlots_->unweightedRescaleFactor[i2][i1] = new TH1F(histoName, histoTitle, NBINS_LOWETA, MIN_RESCALE, MAX_RESCALE);
      else
	thePlots_->unweightedRescaleFactor[i2][i1] = new TH1F(histoName, histoTitle, NBINS_HIGHETA, MIN_RESCALE, MAX_RESCALE);
      thePlots_->unweightedRescaleFactor[i2][i1]->GetXaxis()->SetTitle("Rescale factor");
      thePlots_->unweightedRescaleFactor[i2][i1]->GetYaxis()->SetTitle("a.u.");

      //Weights
      sprintf(histoName, "Weight_channel_%d_Iteration_%d",i1, i2);
      sprintf(histoTitle, "Weight Channel_%d Iteration %d",i1, i2);
      thePlots_->weight[i2][i1] = new TH1F(histoName, histoTitle, 100, 0., 1.);
      thePlots_->weight[i2][i1]->GetXaxis()->SetTitle("Weight");
      thePlots_->weight[i2][i1]->GetYaxis()->SetTitle("a.u.");
    }
  }
}

void ZIterativeAlgorithmWithFit::getStatWeights(char* file) {
  ifstream statfile;
  statfile.open(file);
  if (!statfile) {
    std::cout << "ZIterativeAlgorithmWithFit::FATAL: stat weight  file " << file << " not found" << std::endl;
    exit(-1);
  }
  for(unsigned int i=0;i<channels_;i++) {
    int imod;
    statfile >> imod >> StatWeights_[i];
    std::cout << "Read Stat Weight for module " << imod << ": " <<  StatWeights_[i] << std::endl;
  }
}


bool ZIterativeAlgorithmWithFit::resetIteration()
{
  totalEvents_=0;
  currentEvent_=0;
  
  //Reset correction
  massReco_.clear();
  for (unsigned int i=0;i<channels_;i++) calib_fac_[i]=0.;
  for (unsigned int i=0;i<channels_;i++) weight_sum_[i]=0.;

  return kTRUE;
}    


bool ZIterativeAlgorithmWithFit::iterate()
{

  //Found optimized coefficients
  for (int i=0;i<(int)channels_;i++) 
    { 

      //RP      if (weight_sum_[i]!=0. && calib_fac_[i]!=0.) {
      if( (nCrystalCut_ == -1) || ((!(i <=  nCrystalCut_ -1 )) &&
				  !((i > (19-nCrystalCut_)) && (i <= (19+nCrystalCut_))) &&
				  !((i > (39-nCrystalCut_)) && (i <= (39+nCrystalCut_))) &&
				  !((i > (59-nCrystalCut_)) && (i <= (59+nCrystalCut_))) &&
				  !((i > (84-nCrystalCut_)) && (i <= (84+nCrystalCut_))) &&
				  !((i > (109-nCrystalCut_)) && (i <= (109+nCrystalCut_))) &&
				  !((i > (129-nCrystalCut_)) && (i <= (129+nCrystalCut_))) &&
				  !((i > (149-nCrystalCut_)) && (i <= (149+nCrystalCut_))) &&
				  !(i > (169-nCrystalCut_))))
	{
	  if (weight_sum_[i]!=0.) {
	    //optimizedCoefficients_[i] = calib_fac_[i]/weight_sum_[i];

	    float peak=findPeak((TH1F*)thePlots_->weightedRescaleFactor[currentIteration_][i]);

	    if (peak >=MIN_RESCALE && peak <= MAX_RESCALE)
	      optimizedCoefficients_[i] = 1 / (1 + peak);
	    else
	      optimizedCoefficients_[i] = 1 / (1 + calib_fac_[i]/weight_sum_[i]);

	  } else {
	    optimizedCoefficients_[i]=1.;
	  }
	  
// 	  //	  initialCoefficients_[i] *= optimizedCoefficients_[i];
// 	  EcalCalibMap::getMap()->setRingCalib(i, optimizedCoefficients_[i]);
	}

      else
	{
	  
	  optimizedCoefficients_[i]=1.;
	  
// 	  EcalCalibMap::getMap()->setRingCalib(i, optimizedCoefficients_[i]);
// 	  //	  initialCoefficients_[i] *= optimizedCoefficients_[i];
	}
      
      std::cout << "ZIterativeAlgorithmWithFit::run():Energy Rescaling Coefficient for region " 
		<< i << " is "  << optimizedCoefficients_[i] << " - number of events: " << weight_sum_[i] << std::endl;
    }
  
  currentIteration_++;
  return kTRUE;
}    


bool ZIterativeAlgorithmWithFit::addEvent(calib::CalibElectron* ele1, calib::CalibElectron* ele2, float invMassRescFactor)
{
  totalEvents_++;
  std::pair<calib::CalibElectron*, calib::CalibElectron*> Electrons(ele1, ele2);
#ifdef DEBUG
  std::cout  << "In addEvent " ;
  std::cout << ele1->getRecoElectron()->superCluster()->rawEnergy() << " " ;
  std::cout << ele1->getRecoElectron()->superCluster()->position().eta() << " " ;
  std::cout << ele2->getRecoElectron()->superCluster()->rawEnergy() << " " ;
  std::cout << ele2->getRecoElectron()->superCluster()->position().eta() << " " ;
  std::cout << std::endl;
#endif

  if (massMethod == "SCTRMass" )
    {
      massReco_.push_back(invMassCalc(ele1->getRecoElectron()->superCluster()->energy(), ele1->getRecoElectron()->eta(), ele1->getRecoElectron()->phi(), ele2->getRecoElectron()->superCluster()->energy(), ele2->getRecoElectron()->eta(), ele2->getRecoElectron()->phi()));
    }
  //   else if (massMethod == "S25TRMass" )
//     {
//       massReco_[currentEvent_] = invMassCalc(ele1->electronSCE25_, ele1->getRecoElectron()->eta(), ele1->getRecoElectron()->phi(), ele2->electronSCE25_, ele2->getRecoElectron()->eta(), ele2->getRecoElectron()->phi());
//     }
//   else if (massMethod == "S25CorrTRMass" )
//     {
//       massReco_[currentEvent_] = invMassCalc(ele1->electronSCE25Corr_, ele1->getRecoElectron()->eta(), ele1->getRecoElectron()->phi(), ele2->electronSCE25Corr_, ele2->getRecoElectron()->eta(), ele2->getRecoElectron()->phi());
//     }
  else if (massMethod == "SCMass" )
    {
      massReco_.push_back(invMassCalc(ele1->getRecoElectron()->superCluster()->energy(), ele1->getRecoElectron()->superCluster()->position().eta(), ele1->getRecoElectron()->superCluster()->position().phi(), ele2->getRecoElectron()->superCluster()->energy(), ele2->getRecoElectron()->superCluster()->position().eta(), ele2->getRecoElectron()->superCluster()->position().phi()));
    }  
  
  //  massReco_[currentEvent_]=invMassCalc(ele1->electronSCE_,ele1->getRecoElectron()->superCluster()->position().eta(),ele1->getRecoElectron()->superCluster()->position().phi(),ele2->electronSCE_,ele2->getRecoElectron()->superCluster()->position().eta(),ele2->getRecoElectron()->superCluster()->position().phi());
  
  //Calculate weights and correction factor
 //cout << "\nevent " << currentEvent_ << " - invMass " << massReco_[currentEvent_] << " - el energies: " << ele1->getRecoElectron()->superCluster()->energy() << " & " << ele2->getRecoElectron()->superCluster()->energy() << " - etas " << ele1->getRecoElectron()->superCluster()->position().eta() << " / " << ele2->getRecoElectron()->superCluster()->position().eta() << " - phis " << ele1->getRecoElectron()->superCluster()->position().phi() << " / " << ele2->getRecoElectron()->superCluster()->position().phi() << endl;
//     if ( massMethod == "SCMass")
//       {
#ifdef DEBUG
  std::cout << "Mass calculated " << massReco_[currentEvent_] << std::endl;
#endif
  if((ele2->getRecoElectron()->superCluster()->position().eta() > -10.) && (ele2->getRecoElectron()->superCluster()->position().eta() < 10.) && 
     (ele2->getRecoElectron()->superCluster()->position().phi() > -10.) && (ele2->getRecoElectron()->superCluster()->position().phi() < 10.)) {
    getWeight(currentEvent_, Electrons, invMassRescFactor);
  }
//       }
//     else if (massMethod == "SCTRMass" )
//       {
// 	if((ele2->getRecoElectron()->eta() > -10.) && (ele2->getRecoElectron()->eta() < 10.) && 
// 	   (ele2->getRecoElectron()->phi() > -10.) && (ele2->getRecoElectron()->phi() < 10.)) {
// 	  getWeight(currentEvent_, Electrons, invMassRescFactor);
// 	}
//       }

  currentEvent_++;
  return kTRUE;
}

    
void ZIterativeAlgorithmWithFit::getWeight(unsigned int event_id, std::pair<calib::CalibElectron*,calib::CalibElectron*> elepair, float invMassRescFactor) 
{
  // std::cout<< "Calculating weight for module " << module << " on electrons " << elestd::pair.first << " & " << elepair.second << std::endl; 
  //#ifdef DEBUG_VERBOSE
  //  std::cout << "####Event: " << event_id << std::endl;
  //#endif
  float event_weight;
  if (UseStatWeights_) {
    event_weight=getEventWeight(event_id);
  } else {
    event_weight=1/(elepair.first->getRecoElectron()->superCluster()->rawEnergy()+elepair.second->getRecoElectron()->superCluster()->rawEnergy());
  }
  //  getWeight(event_id, elepair.first, event_weight);
  //  getWeight(event_id, elepair.second, event_weight);
  // RP: introduco il fattore di rescaling della massa invariante
  getWeight(event_id, elepair.first, invMassRescFactor);
  getWeight(event_id, elepair.second, invMassRescFactor);
}

float ZIterativeAlgorithmWithFit::getEventWeight(unsigned int event_id) {

//   pair<calib::CalibElectron*,calib::CalibElectron*> elepair=electrons_[event_id];

//   float evweight=1.;
//   calib::CalibElectron* ele=elepair.first;
//   vector<int>* modules=(*ele).getModules();

//   for (unsigned int imod=0;imod<(*modules).size();imod++) {
//     unsigned int mod=(*modules)(imod);
//     if (mod<channels_) {
//       //      if((*weights)(imod)>=.8)
// #ifdef DEBUG_VERBOSE
//       std::cout<< "Found a stat weight for module "  << mod << " is " << (StatWeights_)(mod) << std::endl;
// #endif
//       //ADDED EVENT WEIGHT 17-04-2003
//       evweight*=(StatWeights_)(mod);
//     } else {
//       std::cout << "ZIterativeAlgorithmWithFit::FATAL:found a wrong module_id" << std::endl;
//     }
//   }

//   ele=elepair.second;
//   modules=(*ele).getModules();
  
//   for (unsigned int imod=0;imod<(*modules).size();imod++) {
//     unsigned int mod=(*modules)(imod);
//     if (mod<channels_) {
//       //      if((*weights)(imod)>=.8)
// #ifdef DEBUG_VERBOSE
//       std::cout<< "Found a stat weight for module "  << mod << " is " << (StatWeights_)(mod) << std::endl;
// #endif
//       //ADDED EVENT WEIGHT 17-04-2003
//       evweight*=(StatWeights_)(mod);
//     } else {
//       std::cout << "ZIterativeAlgorithmWithFit::FATAL:found a wrong module_id" << std::endl;
//     }
//   }

// #ifdef DEBUG_VERBOSE
//   std::cout<< "Returning a stat weight for event "  << event_id << " is " << evweight << std::endl;
// #endif

//   Event_Weight_(event_id)=evweight;

  
  return 1.;
}

void ZIterativeAlgorithmWithFit::getWeight(unsigned int event_id, calib::CalibElectron* ele, float evweight) {
  //  std::cout<< "getting weight for module " << module << " in electron " << ele << std::endl;

  std::vector< std::pair<int,float> > modules=(*ele).getCalibModulesWeights(calibType_); 

  for (int imod=0; imod< (int) modules.size(); imod++) {

    int mod = (int) modules[imod].first;
    
    if (mod< (int) channels_ && mod>=0) {

      if (modules[imod].second >= 0.12 && modules[imod].second < 10000.) 
	{
	  if( (nCrystalCut_ == -1) || ((!(mod <= nCrystalCut_ - 1 )) &&
				     !((mod > (19-nCrystalCut_)) && (mod <= (19+nCrystalCut_))) &&
				     !((mod > (39-nCrystalCut_)) && (mod <= (39+nCrystalCut_))) &&
				     !((mod > (59-nCrystalCut_)) && (mod <= (59+nCrystalCut_))) &&
				     !((mod > (84-nCrystalCut_)) && (mod <= (84+nCrystalCut_))) &&
				     !((mod > (109-nCrystalCut_)) && (mod <= (109+nCrystalCut_))) &&
				     !((mod > (129-nCrystalCut_)) && (mod <= (129+nCrystalCut_))) &&
				     !((mod > (149-nCrystalCut_)) && (mod <= (149+nCrystalCut_))) &&
				     !(mod > (169-nCrystalCut_))))
	    {
	      //	  float weight=modules[imod].second*evweight;
	      // RP: metto tutti i pesi ad 1 oppure divido solo per l'energia di un elettrone
	      //	      float weight = 1.;
	      float weight2 = modules[imod].second / ele->getRecoElectron()->superCluster()->rawEnergy();
#ifdef DEBUG
	      std::cout << "w2 " << weight2 << std::endl;
#endif
	      if (weight2>=0. && weight2<=1.)
		{
		// 	  calib_fac_[mod]+=TMath::Power(M_Z_/massReco_[event_id],AlphaExponent_)*weight;
	      // 	  weight_sum_[mod]+=weight;
	  
	      
	      //	  if(weight2 > 0.5) {
	      
	      
	      //	  calib_fac_[mod] += weight2 * (TMath::Power((massReco_[event_id]*evweight / M_Z_), 2.) - 1) / 2.;
	      //	  calib_fac_[mod]+= M_Z_ / massReco_[event_id]*weight2;
	      
	      // RP: nuova funzione
		  float rescale = (TMath::Power((massReco_[event_id] / evweight), 2.) - 1) / 2.;
#ifdef DEBUG
		  std::cout << "rescale " << rescale << std::endl;		  
#endif
		  if (rescale>= MIN_RESCALE && rescale<=MAX_RESCALE)
		    {
		      calib_fac_[mod] += weight2 * rescale;
		      weight_sum_[mod]+= weight2;
		      // PM 20050706: Filling Histograms for Fit Function
		      thePlots_->weightedRescaleFactor[currentIteration_][mod]->Fill(rescale,weight2);
		      thePlots_->unweightedRescaleFactor[currentIteration_][mod]->Fill(rescale,1.);
		      thePlots_->weight[currentIteration_][mod]->Fill(weight2,1.);
		    }
		  else
		    {
		      std::cout     << "[ZIterativeAlgorithmWithFit]::[getWeight]::rescale out " << rescale << std::endl;
		    }
		}
	      //	    std::cout     << "Ring "  <<  mod << " - Calib_fac is  " << calib_fac_[mod] << " & weight sum is " << weight_sum_[mod] << std::endl;
	      //	    cout << massReco_[event_id] << " - " << modules[imod].second << " - " << ele->electronSCE_ << endl;
	      
	      //}
	    }
	}
    } 
    else 
      {
	std::cout << "ZIterativeAlgorithmWithFit::FATAL:found a wrong module_id " << mod << " channels " << channels_ << std::endl;
      }
  }
}


ZIterativeAlgorithmWithFit::~ZIterativeAlgorithmWithFit()
{
  std::cout << "ZIterativeAlgorithmWithFit::Called Destructor" << std::endl;
  for (unsigned int i2 = 0; i2 < numberOfIterations_; i2++) 
    for (unsigned int i1 = 0; i1 < channels_; i1++)
      { 
	if (thePlots_->weightedRescaleFactor[i1][i2])  
	  delete thePlots_->weightedRescaleFactor[i1][i2];
	if (thePlots_->unweightedRescaleFactor[i1][i2])  
	  delete thePlots_->unweightedRescaleFactor[i1][i2];
	if (thePlots_->weight[i1][i2])  
	  delete thePlots_->weight[i1][i2];
      }
  
  //  if (eventData_) delete eventData_; //It seems that calling destructors for
  //if (massReco_) delete massReco_;  // HepMatrix(Vector) is not implemented and
  // gives segfault. PM 20030305 
}

float ZIterativeAlgorithmWithFit::findPeak(TH1F* histoou)
{
  double par[3];
  double errpar[3];
  TF1* fitFunc=gausfit(histoou,par,errpar,2.5,2.5);
  
  if (gMinuit->GetStatus() == 0)
    return par[1];
  else
    {
      std::cout << "MIGRAD has not converged"<< std::endl;
      return -9999.;
    }
}

TF1* ZIterativeAlgorithmWithFit::gausfit(TH1F * histoou,double* par,double* errpar,float nsigmalow, float nsigmaup) {
  TF1 *gausa = new TF1("gausa","gaus",histoou->GetMean()-3*histoou->GetRMS(),histoou->GetMean()+3*histoou->GetRMS());
  
  gausa->SetParameters(histoou->GetMaximum(),histoou->GetMean(),histoou->GetRMS());
  
  histoou->Fit("gausa","qR0N");
  
  double p1    = gausa->GetParameter(1);
  double sigma = gausa->GetParameter(2);
  double nor   = gausa->GetParameter(0);
    
  double xmi=p1-5*sigma;
  double xma=p1+5*sigma;
  double chi2=100;
  
  double xmin_fit=p1-nsigmalow*sigma;
  double xmax_fit=p1+nsigmaup*sigma;
  
  int iter=0;
  TF1* fitFunc;

  while ((chi2>1. && iter<10) || iter<2 )
    {
      xmin_fit=p1-nsigmalow*sigma;
      xmax_fit=p1+nsigmaup*sigma;
      xmi=p1-5*sigma;
      xma=p1+5*sigma;
      
      char suffix[20];
      sprintf (suffix,"_iter_%d",iter); 
      fitFunc = new TF1("FitFunc"+TString(suffix),"gaus",xmin_fit,xmax_fit);
      fitFunc->SetParameters(nor,p1,sigma);
      fitFunc->SetLineColor((int)(iter+1));
      fitFunc->SetLineWidth(1.);
      //histoou->Fit("FitFunc","lR+","");
      histoou->Fit("FitFunc"+TString(suffix),"qR0+","");
      
      //FIXME Controllo finale sul fit di fitfunc
      //       const int kNoDraw = 1<<9;
      //       histoou->GetFunction("FitFunc")->ResetBit(kNoDraw);
      
      histoou->GetXaxis()->SetRangeUser(xmi,xma);
      histoou->GetXaxis()->SetLabelSize(0.055);
      
      //      cout << fitFunc->GetParameters() << "," << par << endl;
      par[0]=(fitFunc->GetParameters())[0];
      par[1]=(fitFunc->GetParameters())[1];
      par[2]=(fitFunc->GetParameters())[2];
      errpar[0]=(fitFunc->GetParErrors())[0];
      errpar[1]=(fitFunc->GetParErrors())[1];
      errpar[2]=(fitFunc->GetParErrors())[2];
      if (fitFunc->GetNDF()!=0)
        {
          chi2=fitFunc->GetChisquare()/(fitFunc->GetNDF());
        }
      else
        {
	  chi2=100.;
//           par[0]=-99;
//           par[1]=-99;
//           par[2]=-99;
          std::cout << "WARNING: Not enough NDF" << std::endl;
//           return 0;
        }

      // Non visualizzare
      //      histoou->Draw();
      //      c1->Update();

      //      cout << "iter " << iter << " chi2 " << chi2 << endl;
      nor=par[0];
      p1=par[1];
      sigma=par[2];
      iter++;
    }
  return fitFunc;
}
