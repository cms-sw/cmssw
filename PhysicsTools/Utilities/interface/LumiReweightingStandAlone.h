#ifndef PhysicsTools_Utilities_interface_LumiReWeighting_h
#define PhysicsTools_Utilities_interface_LumiReWeighting_h


/**
  \class    LumiReWeighting LumiReWeighting.h "PhysicsTools/Utilities/interface/LumiReWeighting.h"
  \brief    Class to provide lumi weighting for analyzers to weight "flat-to-N" MC samples to data

  This class will trivially take two histograms:
  1. The generated "flat-to-N" distributions from a given processing
  2. A histogram generated from the "estimatePileup" macro here:

   -- This is the Stand-Alone version that doesn't use any CMS classes --

  https://twiki.cern.ch/twiki/bin/view/CMS/LumiCalc#How_to_use_script_estimatePileup

  \authors Salvatore Rappoccio, Mike Hildreth
*/

#include "TH1F.h"
#include "TH3.h"
#include "TFile.h"
#include "TRandom1.h"
#include "TRandom2.h"
#include "TRandom3.h"
#include "TStopwatch.h"
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>

namespace reweight {


  // add a class to shift the mean of a poisson-like luminosity distribution by an arbitrary amount. 
  // Only valid for small (<1.5) shifts about the 2011 lumi distribution for now, because shifts are non-linear
  // Each PoissonMeanShifter does one shift, so defining multiples can give you an arbitrary collection

  class PoissonMeanShifter {

  public:

    PoissonMeanShifter() { };

    PoissonMeanShifter( float Shift ){

      // these are the polynomial or exponential coefficients for each bin of a 25-bin sequence that
      // convert the Distribution of the 2011 luminosity to something with a lower or higher peak luminosity.
      // The distributions aren't quite poisson because they model luminosity decreasing during a fill. This implies that
      // they do get wider as the mean increases, so the weights are not linear with increasing mean.

      static const double p0_minus[20] = { 1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1. };
      static const double p1_minus[20] = {
	-0.677786,
	-0.619614,
	-0.49465,
	-0.357963,
	-0.238359,
	-0.110002,
	0.0348629,
	0.191263,
	0.347648,
	0.516615,
	0.679646,
	0.836673,
	0.97764,
	1.135,
	1.29922,
	1.42467,
	1.55901,
	1.61762,
	1.67275,
	1.96008
      };
      static const double p2_minus[20] = {
	0.526164,
	0.251816,
	0.11049,
	0.026917,
	-0.0464692,
	-0.087022,
	-0.0931581,
	-0.0714295,
	-0.0331772,
	0.0347473,
	0.108658,
	0.193048,
	0.272314,
	0.376357,
	0.4964,
	0.58854,
	0.684959,
	0.731063,
	0.760044,
	1.02386
      };

      static const double p1_expoM[5] = {
	1.63363e-03,
	6.79290e-04,
	3.69900e-04,
	2.24349e-04,
	9.87156e-06
      };

      static const double p2_expoM[5] = {
	2.64692,
	3.26585,
	3.53229,
	4.18035,
	5.64027
      };


      static const double p0_plus[20] = { 1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1. };
      static const double p1_plus[20] = {
	-0.739059,
	-0.594445,
	-0.477276,
	-0.359707,
	-0.233573,
	-0.103458,
	0.0373401,
	0.176571,
	0.337617,
	0.499074,
	0.675126,
	0.840522,
	1.00917,
	1.15847,
	1.23816,
	1.44271,
	1.52982,
	1.46385,
	1.5802,
	0.988689
      };
      static const double p2_plus[20] = {
	0.208068,
	0.130033,
	0.0850356,
	0.0448344,
	0.000749832,
	-0.0331347,
	-0.0653281,
	-0.0746009,
	-0.0800667,
	-0.0527636,
	-0.00402649,
	0.103338,
	0.261261,
	0.491084,
	0.857966,
	1.19495,
	1.75071,
	2.65559,
	3.35433,
	5.48835
      };

      static const double p1_expoP[5] = {
	1.42463e-01,
	4.18966e-02,
	1.12697e-01,
	1.66197e-01,
	1.50768e-01
      };

      static const double p2_expoP[5] = {
	1.98758,
	2.27217,
	2.26799,
	2.38455,
	2.52428
      };

      // initialize weights based on desired Shift



      for (int ibin=0;ibin<20;ibin++) {

	if( Shift < .0) {
	  Pweight_[ibin] = p0_minus[ibin] + p1_minus[ibin]*Shift + p2_minus[ibin]*Shift*Shift;
	}
	else {
	  Pweight_[ibin] = p0_plus[ibin] + p1_plus[ibin]*Shift + p2_plus[ibin]*Shift*Shift;
	}
      }

      // last few bins fit better to an exponential...

      for (int ibin=20;ibin<25;ibin++) {
	if( Shift < 0.) {
	  Pweight_[ibin] = p1_expoM[ibin-20]*exp(p2_expoM[ibin-20]*Shift);
	}
	else {
	  Pweight_[ibin] = p1_expoP[ibin-20]*exp(p2_expoP[ibin-20]*Shift);
	}
      } 

    };

    double ShiftWeight( int ibin ) {

      if(ibin<25 && ibin>=0) { return Pweight_[ibin]; }
      else { return 0;}

    };

    double ShiftWeight( float pvnum ) {

      int ibin = int(pvnum);

      if(ibin<25 && ibin>=0) { return Pweight_[ibin]; }
      else { return 0;}

    };

  private:

    double Pweight_[25];

  };


  class LumiReWeighting {
  public:

    LumiReWeighting ( ) { } ;

    LumiReWeighting( std::string generatedFile,
		     std::string dataFile,
		     std::string GenHistName,
		     std::string DataHistName) :
      generatedFileName_( generatedFile), 
      dataFileName_     ( dataFile ), 
      GenHistName_      ( GenHistName ), 
      DataHistName_     ( DataHistName )
	{
	  generatedFile_ = new TFile( generatedFileName_.c_str() ) ; //MC distribution
	  dataFile_      = new TFile( dataFileName_.c_str() );       //Data distribution

	  Data_distr_ = new TH1F(  *(static_cast<TH1F*>(dataFile_->Get( DataHistName_.c_str() )->Clone() )) );
	  MC_distr_ = new TH1F(  *(static_cast<TH1F*>(generatedFile_->Get( GenHistName_.c_str() )->Clone() )) );

	  // normalize both histograms first                                                                            

	  Data_distr_->Scale( 1.0/ Data_distr_->Integral() );
	  MC_distr_->Scale( 1.0/ MC_distr_->Integral() );

	  weights_ = new TH1F( *(Data_distr_)) ;

	  // MC * data/MC = data, so the weights are data/MC:

	  weights_->SetName("lumiWeights");

	  TH1F* den = new TH1F(*(MC_distr_));

	  weights_->Divide( den );  // so now the average weight should be 1.0

	  std::cout << " Lumi/Pileup Reweighting: Computed Weights per In-Time Nint " << std::endl;

	  int NBins = weights_->GetNbinsX();

	  for(int ibin = 1; ibin<NBins+1; ++ibin){
	    std::cout << "   " << ibin-1 << " " << weights_->GetBinContent(ibin) << std::endl;
	  }

	  weightOOT_init();

	  FirstWarning_ = true;

	}

    
      LumiReWeighting( const std::vector< float >& MC_distr, const std::vector< float >& Lumi_distr){
	// no histograms for input: use vectors
  
	// now, make histograms out of them:

	// first, check they are the same size...

	if( MC_distr.size() != Lumi_distr.size() ){   

	  std::cerr <<"ERROR: LumiReWeighting: input vectors have different sizes. Quitting... \n";
	  return;

	}

	Int_t NBins = MC_distr.size();

	MC_distr_ = new TH1F("MC_distr","MC dist",NBins,-0.5, float(NBins)-0.5);
	Data_distr_ = new TH1F("Data_distr","Data dist",NBins,-0.5, float(NBins)-0.5);

	weights_ = new TH1F("luminumer","luminumer",NBins,-0.5, float(NBins)-0.5);
	TH1F* den = new TH1F("lumidenom","lumidenom",NBins,-0.5, float(NBins)-0.5);

	for(int ibin = 1; ibin<NBins+1; ++ibin ) {
	  weights_->SetBinContent(ibin, Lumi_distr[ibin-1]);
	  Data_distr_->SetBinContent(ibin, Lumi_distr[ibin-1]);
	  den->SetBinContent(ibin,MC_distr[ibin-1]);
	  MC_distr_->SetBinContent(ibin,MC_distr[ibin-1]);
	}

	// check integrals, make sure things are normalized

	float deltaH = weights_->Integral();
	if(fabs(1.0 - deltaH) > 0.02 ) { //*OOPS*...
	  weights_->Scale( 1.0/ deltaH );
	  Data_distr_->Scale( 1.0/ deltaH );
	}
	float deltaMC = den->Integral();
	if(fabs(1.0 - deltaMC) > 0.02 ) {
	  den->Scale(1.0/ deltaMC );
	  MC_distr_->Scale(1.0/ deltaMC );
	}

	weights_->Divide( den );  // so now the average weight should be 1.0    

	std::cout << " Lumi/Pileup Reweighting: Computed Weights per In-Time Nint " << std::endl;

	for(int ibin = 1; ibin<NBins+1; ++ibin){
	  std::cout << "   " << ibin-1 << " " << weights_->GetBinContent(ibin) << std::endl;
	}

	weightOOT_init();

	FirstWarning_ = true;

      }

      void weight3D_init( float ScaleFactor, std::string WeightOutputFile="") { 

	//create histogram to write output weights, save pain of generating them again...

	TH3D* WHist = new TH3D("WHist","3D weights",50,0.,50.,50,0.,50.,50,0.,50. );
	TH3D* DHist = new TH3D("DHist","3D weights",50,0.,50.,50,0.,50.,50,0.,50. );
	TH3D* MHist = new TH3D("MHist","3D weights",50,0.,50.,50,0.,50.,50,0.,50. );


	using std::min;

	if( MC_distr_->GetEntries() == 0 ) {
	  std::cout << " MC and Data distributions are not initialized! You must call the LumiReWeighting constructor. " << std::endl;
	}

	// arrays for storing number of interactions

	double MC_ints[50][50][50];
	double Data_ints[50][50][50];

	for (int i=0; i<50; i++) {
	  for(int j=0; j<50; j++) {
	    for(int k=0; k<50; k++) {
	      MC_ints[i][j][k] = 0.;
	      Data_ints[i][j][k] = 0.;
	    }
	  }
	}

	double factorial[50];
	double PowerSer[50];
	double base = 1.;

	factorial[0] = 1.;
	PowerSer[0]=1.;

	for (int i = 1; i<50; ++i) {
	  base = base*float(i);
	  factorial[i] = base;
	}


	double x;
	double xweight;
	double probi, probj, probk;
	double Expval, mean;
	int xi;

	// Get entries for Data, MC, fill arrays:                                                                                                 
	int NMCbin = MC_distr_->GetNbinsX();

	for (int jbin=1;jbin<NMCbin+1;jbin++) {
	  x =  MC_distr_->GetBinCenter(jbin);
	  xweight = MC_distr_->GetBinContent(jbin); //use as weight for matrix         

	  //for Summer 11, we have this int feature:                      
	  xi = int(x);

	  // Generate Poisson distribution for each value of the mean     
	  mean = double(xi);

	  if(mean<0.) {
	    std::cout << "LumiReweighting:BadInputValue" << " Your histogram generates MC luminosity values less than zero!"
						  << " Please Check.  Terminating." << std::endl;
	  }


	  if(mean==0.){
	    Expval = 1.;
	  }
	  else {
	    Expval = exp(-1.*mean);
	  }

	  base = 1.;

	  for (int i = 1; i<50; ++i) {
	    base = base*mean;
	    PowerSer[i] = base; // PowerSer is mean^i                        
	  }

	  // compute poisson probability for each Nvtx in weight matrix      
	  for (int i=0; i<50; i++) {
	    probi = PowerSer[i]/factorial[i]*Expval;
	    for(int j=0; j<50; j++) {
	      probj = PowerSer[j]/factorial[j]*Expval;
	      for(int k=0; k<50; k++) {
		probk = PowerSer[k]/factorial[k]*Expval;
		// joint probability is product of event weights multiplied by weight of input distribution bin                                   
		MC_ints[i][j][k] = MC_ints[i][j][k]+probi*probj*probk*xweight;
	      }
	    }
	  }

	}

	int NDatabin = Data_distr_->GetNbinsX();

	for (int jbin=1;jbin<NDatabin+1;jbin++) {
	  mean =  (Data_distr_->GetBinCenter(jbin))*ScaleFactor;
	  xweight = Data_distr_->GetBinContent(jbin);

	  // Generate poisson distribution for each value of the mean
	  if(mean<0.) {
	    std::cout << "LumiReweighting:BadInputValue" << " Your histogram generates MC luminosity values less than zero!"
						  << " Please Check.  Terminating." << std::endl;
	  }

	  if(mean==0.){
	    Expval = 1.;
	  }
	  else {
	    Expval = exp(-1.*mean);
	  }

	  base = 1.;

	  for (int i = 1; i<50; ++i) {
	    base = base*mean;
	    PowerSer[i] = base;
	  }

	  // compute poisson probability for each Nvtx in weight matrix                                                                           

	  for (int i=0; i<50; i++) {
	    probi = PowerSer[i]/factorial[i]*Expval;
	    for(int j=0; j<50; j++) {
	      probj = PowerSer[j]/factorial[j]*Expval;
	      for(int k=0; k<50; k++) {
		probk = PowerSer[k]/factorial[k]*Expval;
		// joint probability is product of event weights multiplied by weight of input distribution bin                                   
		Data_ints[i][j][k] = Data_ints[i][j][k]+probi*probj*probk*xweight;
	      }
	    }
	  }

	}


	for (int i=0; i<50; i++) {
	  //if(i<5) std::cout << "i = " << i << std::endl;                       
	  for(int j=0; j<50; j++) {
	    for(int k=0; k<50; k++) {
	      if( (MC_ints[i][j][k])>0.) {
		Weight3D_[i][j][k]  =  Data_ints[i][j][k]/MC_ints[i][j][k];
	      }
	      else {
		Weight3D_[i][j][k]  = 0.;
	      }
	      WHist->SetBinContent( i+1,j+1,k+1,Weight3D_[i][j][k] );
	      DHist->SetBinContent( i+1,j+1,k+1,Data_ints[i][j][k] );
	      MHist->SetBinContent( i+1,j+1,k+1,MC_ints[i][j][k] );
	      //      if(i<5 && j<5 && k<5) std::cout << Weight3D_[i][j][k] << " " ;    
	    }
	    //      if(i<5 && j<5) std::cout << std::endl;        
	  }
	}

	if(! WeightOutputFile.empty() ) {
	  std::cout << " 3D Weight Matrix initialized! " << std::endl;
	  std::cout << " Writing weights to file " << WeightOutputFile << " for re-use...  " << std::endl;


	  TFile * outfile = new TFile(WeightOutputFile.c_str(),"RECREATE");
	  WHist->Write();
	  MHist->Write();
	  DHist->Write();
	  outfile->Write();
	  outfile->Close();
	  outfile->Delete();
	}
	
	return;
      }


      void weight3D_set( std::string WeightFileName ) { 

	TFile *infile = new TFile(WeightFileName.c_str());
	TH1F *WHist = (TH1F*)infile->Get("WHist");

	// Check if the histogram exists           
	if (!WHist) {
	  std::cout << " Could not find the histogram WHist in the file "
						    << "in the file " << WeightFileName << "." << std::endl;
	  return;
	}

	for (int i=0; i<50; i++) {  
	  for(int j=0; j<50; j++) {
	    for(int k=0; k<50; k++) {
	      Weight3D_[i][j][k] = WHist->GetBinContent(i,j,k);
	    }
	  }
	}

	std::cout << " 3D Weight Matrix initialized! " << std::endl;

	return;


      }



      void weightOOT_init() {

	// The following are poisson distributions with different means, where the maximum
	// of the function has been normalized to weight 1.0
	// These are used to reweight the out-of-time pileup to match the in-time distribution.
	// The total event weight is the product of the in-time weight, the out-of-time weight,
	// and a residual correction to fix the distortions caused by the fact that the out-of-time
	// distribution is not flat.

	static const double weight_24[25] = {
	  0,
	  0,
	  0,
	  0,
	  2.46277e-06,
	  2.95532e-05,
	  0.000104668,
	  0.000401431,
	  0.00130034,
	  0.00342202,
	  0.00818132,
	  0.0175534,
	  0.035784,
	  0.0650836,
	  0.112232,
	  0.178699,
	  0.268934,
	  0.380868,
	  0.507505,
	  0.640922,
	  0.768551,
	  0.877829,
	  0.958624,
	  0.99939,
	  1
	};

	static const double weight_23[25] = {
	  0,
	  1.20628e-06,
	  1.20628e-06,
	  2.41255e-06,
	  1.20628e-05,
	  6.39326e-05,
	  0.000252112,
	  0.000862487,
	  0.00244995,
	  0.00616527,
	  0.0140821,
	  0.0293342,
	  0.0564501,
	  0.100602,
	  0.164479,
	  0.252659,
	  0.36268,
	  0.491427,
	  0.627979,
	  0.75918,
	  0.873185,
	  0.957934,
	  0.999381,
	  1,
	  0.957738
	};

	static const double weight_22[25] = {
	  0,
	  0,
	  0,
	  5.88636e-06,
	  3.0609e-05,
	  0.000143627,
	  0.000561558,
	  0.00173059,
	  0.00460078,
	  0.0110616,
	  0.0238974,
	  0.0475406,
	  0.0875077,
	  0.148682,
	  0.235752,
	  0.343591,
	  0.473146,
	  0.611897,
	  0.748345,
	  0.865978,
	  0.953199,
	  0.997848,
	  1,
	  0.954245,
	  0.873688
	};

	static const double weight_21[25] = {
	  0,
	  0,
	  1.15381e-06,
	  8.07665e-06,
	  7.1536e-05,
	  0.000280375,
	  0.00107189,
	  0.00327104,
	  0.00809396,
	  0.0190978,
	  0.0401894,
	  0.0761028,
	  0.13472,
	  0.216315,
	  0.324649,
	  0.455125,
	  0.598241,
	  0.739215,
	  0.861866,
	  0.953911,
	  0.998918,
	  1,
	  0.956683,
	  0.872272,
	  0.76399
	};
 
 
	static const double weight_20[25] = {
	  0,
	  0,
	  1.12532e-06,
	  2.58822e-05,
	  0.000145166,
	  0.000633552,
	  0.00215048,
	  0.00592816,
	  0.0145605,
	  0.0328367,
	  0.0652649,
	  0.11893,
	  0.19803,
	  0.305525,
	  0.436588,
	  0.581566,
	  0.727048,
	  0.8534,
	  0.949419,
	  0.999785,
	  1,
	  0.953008,
	  0.865689,
	  0.753288,
	  0.62765
	}; 
	static const double weight_19[25] = {
	  0,
	  0,
	  1.20714e-05,
	  5.92596e-05,
	  0.000364337,
	  0.00124994,
	  0.00403953,
	  0.0108149,
	  0.025824,
	  0.0544969,
	  0.103567,
	  0.17936,
	  0.283532,
	  0.416091,
	  0.562078,
	  0.714714,
	  0.846523,
	  0.947875,
	  1,
	  0.999448,
	  0.951404,
	  0.859717,
	  0.742319,
	  0.613601,
	  0.48552
	};

	static const double weight_18[25] = {
	  0,
	  3.20101e-06,
	  2.88091e-05,
	  0.000164319,
	  0.000719161,
	  0.00250106,
	  0.00773685,
	  0.0197513,
	  0.0443693,
	  0.0885998,
	  0.159891,
	  0.262607,
	  0.392327,
	  0.543125,
	  0.69924,
	  0.837474,
	  0.943486,
	  0.998029,
	  1,
	  0.945937,
	  0.851807,
	  0.729309,
	  0.596332,
	  0.467818,
	  0.350434
	};

 
	static const double weight_17[25] = {
	  1.03634e-06,
	  7.25437e-06,
	  4.97443e-05,
	  0.000340956,
	  0.00148715,
	  0.00501485,
	  0.0143067,
	  0.034679,
	  0.0742009,
	  0.140287,
	  0.238288,
	  0.369416,
	  0.521637,
	  0.682368,
	  0.828634,
	  0.939655,
	  1,
	  0.996829,
	  0.94062,
	  0.841575,
	  0.716664,
	  0.582053,
	  0.449595,
	  0.331336,
	  0.234332
	};

 
	static const double weight_16[25] = {
	  4.03159e-06,
	  2.41895e-05,
	  0.000141106,
	  0.00081942,
	  0.00314565,
	  0.00990662,
	  0.026293,
	  0.0603881,
	  0.120973,
	  0.214532,
	  0.343708,
	  0.501141,
	  0.665978,
	  0.820107,
	  0.938149,
	  1,
	  0.99941,
	  0.940768,
	  0.837813,
	  0.703086,
	  0.564023,
	  0.42928,
	  0.312515,
	  0.216251,
	  0.14561
	};
 
 
	static const double weight_15[25] = {
	  9.76084e-07,
	  5.07564e-05,
	  0.000303562,
	  0.00174036,
	  0.00617959,
	  0.0188579,
	  0.047465,
	  0.101656,
	  0.189492,
	  0.315673,
	  0.474383,
	  0.646828,
	  0.809462,
	  0.934107,
	  0.998874,
	  1,
	  0.936163,
	  0.827473,
	  0.689675,
	  0.544384,
	  0.40907,
	  0.290648,
	  0.198861,
	  0.12951,
	  0.0808051
	};
 
 
	static const double weight_14[25] = {
	  1.13288e-05,
	  0.000124617,
	  0.000753365,
	  0.00345056,
	  0.0123909,
	  0.0352712,
	  0.0825463,
	  0.16413,
	  0.287213,
	  0.44615,
	  0.625826,
	  0.796365,
	  0.930624,
	  0.999958,
	  1,
	  0.934414,
	  0.816456,
	  0.672939,
	  0.523033,
	  0.386068,
	  0.269824,
	  0.180342,
	  0.114669,
	  0.0698288,
	  0.0406496
	};

 
	static const double weight_13[25] = {
	  2.54296e-05,
	  0.000261561,
	  0.00167018,
	  0.00748083,
	  0.0241308,
	  0.0636801,
	  0.138222,
	  0.255814,
	  0.414275,
	  0.600244,
	  0.779958,
	  0.92256,
	  0.999155,
	  1,
	  0.927126,
	  0.804504,
	  0.651803,
	  0.497534,
	  0.35976,
	  0.245834,
	  0.160904,
	  0.0991589,
	  0.0585434,
	  0.0332437,
	  0.0180159
	};

	static const double weight_12[25] = {
	  5.85742e-05,
	  0.000627706,
	  0.00386677,
	  0.0154068,
	  0.0465892,
	  0.111683,
	  0.222487,
	  0.381677,
	  0.5719,
	  0.765001,
	  0.915916,
	  1,
	  0.999717,
	  0.921443,
	  0.791958,
	  0.632344,
	  0.475195,
	  0.334982,
	  0.223666,
	  0.141781,
	  0.0851538,
	  0.048433,
	  0.0263287,
	  0.0133969,
	  0.00696683
	};

 
	static const double weight_11[25] = {
	  0.00015238,
	  0.00156064,
	  0.00846044,
	  0.0310939,
	  0.0856225,
	  0.187589,
	  0.343579,
	  0.541892,
	  0.74224,
	  0.909269,
	  0.998711,
	  1,
	  0.916889,
	  0.77485,
	  0.608819,
	  0.447016,
	  0.307375,
	  0.198444,
	  0.121208,
	  0.070222,
	  0.0386492,
	  0.0201108,
	  0.0100922,
	  0.00484937,
	  0.00222458
	};

	static const double weight_10[25] = {
	  0.000393044,
	  0.00367001,
	  0.0179474,
	  0.060389,
	  0.151477,
	  0.302077,
	  0.503113,
	  0.720373,
	  0.899568,
	  1,
	  0.997739,
	  0.909409,
	  0.75728,
	  0.582031,
	  0.415322,
	  0.277663,
	  0.174147,
	  0.102154,
	  0.0566719,
	  0.0298642,
	  0.0147751,
	  0.00710995,
	  0.00319628,
	  0.00140601,
	  0.000568796
	};

 
	static const double weight_9[25] = {
	  0.00093396,
	  0.00854448,
	  0.0380306,
	  0.113181,
	  0.256614,
	  0.460894,
	  0.690242,
	  0.888781,
	  1,
	  0.998756,
	  0.899872,
	  0.735642,
	  0.552532,
	  0.382726,
	  0.246114,
	  0.147497,
	  0.0825541,
	  0.0441199,
	  0.0218157,
	  0.0103578,
	  0.00462959,
	  0.0019142,
	  0.000771598,
	  0.000295893,
	  0.000111529
	};

 
	static const double weight_8[25] = {
	  0.00240233,
	  0.0192688,
	  0.0768653,
	  0.205008,
	  0.410958,
	  0.65758,
	  0.875657,
	  0.999886,
	  1,
	  0.889476,
	  0.711446,
	  0.517781,
	  0.345774,
	  0.212028,
	  0.121208,
	  0.0644629,
	  0.0324928,
	  0.0152492,
	  0.00673527,
	  0.0028547,
	  0.00117213,
	  0.000440177,
	  0.000168471,
	  5.80689e-05,
	  1.93563e-05
	};

	static const double weight_7[25] = {
	  0.00617233,
	  0.0428714,
	  0.150018,
	  0.350317,
	  0.612535,
	  0.856525,
	  0.999923,
	  1,
	  0.87544,
	  0.679383,
	  0.478345,
	  0.303378,
	  0.176923,
	  0.0950103,
	  0.0476253,
	  0.0222211,
	  0.00972738,
	  0.00392962,
	  0.0015258,
	  0.000559168,
	  0.000183928,
	  6.77983e-05,
	  1.67818e-05,
	  7.38398e-06,
	  6.71271e-07
	};
 
	static const double weight_6[25] = {
	  0.0154465,
	  0.0923472,
	  0.277322,
	  0.55552,
	  0.833099,
	  0.999035,
	  1,
	  0.855183,
	  0.641976,
	  0.428277,
	  0.256804,
	  0.139798,
	  0.0700072,
	  0.0321586,
	  0.0137971,
	  0.00544756,
	  0.00202316,
	  0.000766228,
	  0.000259348,
	  8.45836e-05,
	  1.80362e-05,
	  8.70713e-06,
	  3.73163e-06,
	  6.21938e-07,
	  0
	};
 
 
	static const double weight_5[25] = {
	  0.0382845,
	  0.191122,
	  0.478782,
	  0.797314,
	  1,
	  0.997148,
	  0.831144,
	  0.59461,
	  0.371293,
	  0.205903,
	  0.103102,
	  0.0471424,
	  0.0194997,
	  0.00749415,
	  0.00273709,
	  0.000879189,
	  0.000286049,
	  0.000102364,
	  1.70606e-05,
	  3.98081e-06,
	  2.27475e-06,
	  0,
	  0,
	  0,
	  0
	};
 
 
	static const double weight_4[25] = {
	  0.0941305,
	  0.373824,
	  0.750094,
	  1,
	  0.997698,
	  0.800956,
	  0.532306,
	  0.304597,
	  0.152207,
	  0.0676275,
	  0.0270646,
	  0.00975365,
	  0.00326077,
	  0.00101071,
	  0.000301781,
	  7.41664e-05,
	  1.58563e-05,
	  3.58045e-06,
	  1.02299e-06,
	  0,
	  5.11493e-07,
	  0,
	  0,
	  0,
	  0
	};
 
 
	static const double weight_3[25] = {
	  0.222714,
	  0.667015,
	  1,
	  0.999208,
	  0.750609,
	  0.449854,
	  0.224968,
	  0.0965185,
	  0.0361225,
	  0.012084,
	  0.00359618,
	  0.000977166,
	  0.000239269,
	  6.29422e-05,
	  1.16064e-05,
	  1.78559e-06,
	  0,
	  4.46398e-07,
	  0,
	  0,
	  0,
	  0,
	  0,
	  0,
	  0
	};
 
	static const double weight_2[25] = {
	  0.499541,
	  0.999607,
	  1,
	  0.666607,
	  0.333301,
	  0.13279,
	  0.0441871,
	  0.0127455,
	  0.00318434,
	  0.00071752,
	  0.000132204,
	  2.69578e-05,
	  5.16999e-06,
	  2.21571e-06,
	  0,
	  0,
	  0,
	  0,
	  0,
	  0,
	  0,
	  0,
	  0,
	  0,
	  0
	};
 
	static const double weight_1[25] = {
	  0.999165,
	  1,
	  0.499996,
	  0.166868,
	  0.0414266,
	  0.00831053,
	  0.00137472,
	  0.000198911,
	  2.66302e-05,
	  2.44563e-06,
	  2.71737e-07,
	  2.71737e-07,
	  0,
	  0,
	  0,
	  0,
	  0,
	  0,
	  0,
	  0,
	  0,
	  0,
	  0,
	  0,
	  0
	};
 
	static const double weight_0[25] = {
	  1,
	  0,
	  0,
	  0,
	  0,
	  0,
	  0,
	  0,
	  0,
	  0,
	  0,
	  0,
	  0,
	  0,
	  0,
	  0,
	  0,
	  0,
	  0,
	  0,
	  0,
	  0,
	  0,
	  0,
	  0
	};

	//WeightOOTPU_ = {0};

	const double* WeightPtr = 0;

	for(int iint = 0; iint<25; ++iint){
	  if(iint ==0) WeightPtr = weight_0;
	  if(iint ==1) WeightPtr = weight_1;
	  if(iint ==2) WeightPtr = weight_2;
	  if(iint ==3) WeightPtr = weight_3;
	  if(iint ==4) WeightPtr = weight_4;
	  if(iint ==5) WeightPtr = weight_5;
	  if(iint ==6) WeightPtr = weight_6;
	  if(iint ==7) WeightPtr = weight_7;
	  if(iint ==8) WeightPtr = weight_8;
	  if(iint ==9) WeightPtr = weight_9;
	  if(iint ==10) WeightPtr = weight_10;
	  if(iint ==11) WeightPtr = weight_11;
	  if(iint ==12) WeightPtr = weight_12;
	  if(iint ==13) WeightPtr = weight_13;
	  if(iint ==14) WeightPtr = weight_14;
	  if(iint ==15) WeightPtr = weight_15;
	  if(iint ==16) WeightPtr = weight_16;
	  if(iint ==17) WeightPtr = weight_17;
	  if(iint ==18) WeightPtr = weight_18;
	  if(iint ==19) WeightPtr = weight_19;
	  if(iint ==20) WeightPtr = weight_20;
	  if(iint ==21) WeightPtr = weight_21;
	  if(iint ==22) WeightPtr = weight_22;
	  if(iint ==23) WeightPtr = weight_23;
	  if(iint ==24) WeightPtr = weight_24;

	  for(int ibin = 0; ibin<25; ++ibin){
	    WeightOOTPU_[iint][ibin] = *(WeightPtr+ibin);
	  }
	}

      }


      double ITweight( int npv ){
	int bin = weights_->GetXaxis()->FindBin( npv );
	return weights_->GetBinContent( bin );
      }

      double ITweight3BX( float ave_int ){
	int bin = weights_->GetXaxis()->FindBin( ave_int );
	return weights_->GetBinContent( bin );
      }

      double weight( float n_int ){
	int bin = weights_->GetXaxis()->FindBin( n_int );
	return weights_->GetBinContent( bin );
      }


      double weight3D( int pv1, int pv2, int pv3 ) {

	using std::min;

	int npm1 = min(pv1,34);
	int np0 = min(pv2,34);
	int npp1 = min(pv3,34);

	return Weight3D_[npm1][np0][npp1];

      }



      double weightOOT( int npv_in_time, int npv_m50nsBX ){

	static const double Correct_Weights2011[25] = { // residual correction to match lumi spectrum
	  5.30031,
	  2.07903,
	  1.40729,
	  1.27687,
	  1.0702,
	  0.902094,
	  0.902345,
	  0.931449,
	  0.78202,
	  0.824686,
	  0.837735,
	  0.910261,
	  1.01394,
	  1.1599,
	  1.12778,
	  1.58423,
	  1.78868,
	  1.58296,
	  2.3291,
	  3.86641,
	  0,
	  0,
	  0,
	  0,
	  0
	};                        


	if(FirstWarning_) {

	  std::cout << " **** Warning: Out-of-time pileup reweighting appropriate only for PU_S3 **** " << std::endl;
	  std::cout << " ****                          will be applied                           **** " << std::endl;

	  FirstWarning_ = false;

	}


	// Note: for the "uncorrelated" out-of-time pileup, reweighting is only done on the 50ns
	// "late" bunch (BX=+1), since that is basically the only one that matters in terms of 
	// energy deposition.  

	if(npv_in_time < 0) {
	  std::cerr << " no in-time beam crossing found\n! " ;
	  std::cerr << " Returning event weight=0\n! ";
	  return 0.;
	}
	if(npv_m50nsBX < 0) {
	  std::cerr << " no out-of-time beam crossing found\n! " ;
	  std::cerr << " Returning event weight=0\n! ";
	  return 0.;
	}

	int bin = weights_->GetXaxis()->FindBin( npv_in_time );

	double inTimeWeight = weights_->GetBinContent( bin );

	double TotalWeight = 1.0;


	TotalWeight = inTimeWeight * WeightOOTPU_[bin-1][npv_m50nsBX] * Correct_Weights2011[bin-1];


	return TotalWeight;
 
      }

  protected:

      std::string generatedFileName_;
      std::string dataFileName_;
      std::string GenHistName_;
      std::string DataHistName_;
      TFile *generatedFile_;
      TFile *dataFile_;
      TH1F  *weights_;

      //keep copies of normalized distributions:                                                                                  
      TH1F*      MC_distr_;
      TH1F*      Data_distr_;

      double WeightOOTPU_[25][25];
      double Weight3D_[50][50][50];

      bool FirstWarning_;


  };
}



#endif
