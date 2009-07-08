#include "RooAddPdf.h"
#include "RooBifurGauss.h"
#include "RooCBShape.h"
#include "RooExponential.h"
#include "RooGaussian.h"
#include "RooLandau.h"
#include "RooPolynomial.h"
#include "RooBreitWigner.h"
#include "RooHistPdf.h"

#include "RooDataHist.h"
#include "RooArgList.h"

#if (defined (STANDALONE) or defined (__CINT__) )
   #include "RscBaseModel.h"
#else
   #include "PhysicsTools/RooStatsCms/interface/RscBaseModel.h"
#endif

#include "TH1F.h"
#include "TFile.h"
#include "TRandom.h"

/// Cint dictionaries

//For Cint
#if (defined (STANDALONE) or defined (__CINT__) )
ClassImp(RscBaseModel)
#endif
/*----------------------------------------------------------------------------*/

RscBaseModel::RscBaseModel(TString theName, RooRealVar& theVar, RooArgSet* discVars)
  : model("model","defines what model to use"),
    fileName(theName+"_fileName","Histogram file name",""),
    dataName(theName+"_dataName","Histogram name","")
{
  if (verbose())
    cout << "Instanciating a " << ClassName() << " object named " << theName << "\n";
  _name = theName;
  _discVars = discVars;
  x = &theVar;

  model.defineType("none");
  model.defineType("exponential");
  model.defineType("dblgauss");
  model.defineType("fourGaussians");
  model.defineType("gauss");
  model.defineType("poly7");
  model.defineType("BreitWigner");
  model.defineType("histo");
  model.defineType("yieldonly");
  model.defineType("flat");
  model.defineType("CBShapeGaussian");
  model.defineType("BifurGauss");
  model.defineType("Poly4_DblGauss");
  model.defineType("BifurGauss_BifurGauss");
  model.defineType("BifurGauss_BifurGauss_BifurGauss");
  model.defineType("BifurGauss_BifurGauss_Gauss");
  model.defineType("Landau_BifurGauss");
  model.defineType("CBShape_BreitWigner");
  model.setLabel("none");

  readDataCard();
}

/*----------------------------------------------------------------------------*/

RscBaseModel::~RscBaseModel() {
  // destructor
  if (verbose())
    cout << "Destructing the " << _name << " object\n"; 
}

/*----------------------------------------------------------------------------*/

void RscBaseModel::buildPdf() {
  // build the PDF

  if (verbose())
    cout << "Build PDF using " << model.getLabel() 
         << " model; named " << _name << endl;

  if (model=="yieldonly") {

    TH1F flatHist(_name+"_flatHist","",2,0,1);
    TRandom *randomGenerator = new TRandom();
    double random = randomGenerator->Rndm(); // ]0->1]
    delete randomGenerator;
    flatHist.SetBinContent(1,1+0.0001*random);
    flatHist.SetBinContent(2,1);
    RooDataHist* flatDataHist = new RooDataHist(_name+"_flatDataHist","",*x,&flatHist);
    _thePdf = new RooHistPdf(_name,"flat histogramme",*x,*flatDataHist);

//     TRandom *randomGenerator = new TRandom();
//     double random = randomGenerator->Rndm(); // ]0->1]
//     delete randomGenerator;
//     coef_0=new RooRealVar ("coef_0","coef_0",1);
//     coef_1=new RooRealVar ("coef_1","coef_1",0.0001*random);
//     _thePdf = new RooPolynomial((_name+"_flatHist").Data(),"Flat Histogram",*x,RooArgList(*coef_0,*coef_1));

  }
  else if (model=="CBShapeGaussian") {
      RooCBShape* cbshape = new RooCBShape(_name+"_cbshape","CBShape",*x,*m0,*sigma,*alpha,*n);
      RooGaussian* gaussian = new RooGaussian(_name+"_gaussian","Gaussian",*x,*gmean,*gsigma);
      _thePdf = new RooAddPdf(_name,"CBShapeGaussian",*cbshape,*gaussian,*frac);
  }
  else if (model=="BifurGauss") {
      _thePdf = new RooBifurGauss(_name,"BifurGauss",*x,*mean,*sigmaL,*sigmaR);
  }
  else if (model=="Landau_BifurGauss") {
      RooLandau* landau = new RooLandau(_name+"_landau","Landau",*x,*mean2,*sigma2);
      RooBifurGauss* bg = new RooBifurGauss(_name+"_bg","BifurGauss",*x,*mean,*sigmaL,*sigmaR);
      _thePdf = new RooAddPdf(_name,"Landau+BifurGauss",*landau,*bg,*frac);
  }
  else if (model=="BifurGauss_BifurGauss") {
      RooBifurGauss* bg1 = new RooBifurGauss(_name+"_bg1","BifurGauss",*x,*mean,*sigmaL,*sigmaR);
      RooBifurGauss* bg2 = new RooBifurGauss(_name+"_bg2","BifurGauss",*x,*mean2,*sigmaL2,*sigmaR2);
      _thePdf = new RooAddPdf(_name,"BifurGauss+BifurGauss",*bg1,*bg2,*frac);
  }
  else if (model=="BifurGauss_BifurGauss_Gauss") {
      RooBifurGauss* bg1 = new RooBifurGauss(_name+"_bg1","BifurGauss",*x,*mean,*sigmaL,*sigmaR);
      RooBifurGauss* bg2 = new RooBifurGauss(_name+"_bg2","BifurGauss",*x,*mean2,*sigmaL2,*sigmaR2);
      RooGaussian* gaussian = new RooGaussian(_name+"_gaussian","Gaussian",*x,*gmean,*gsigma);
      RooAddPdf* bgsum = new RooAddPdf(_name+"_bgsum","BifurGauss+BifurGauss",*bg1,*bg2,*frac);
      _thePdf = new RooAddPdf(_name,"BifurGauss+BifurGauss",*bgsum,*gaussian,*frac2);
  }
  else if (model=="BifurGauss_BifurGauss_BifurGauss") {
      RooBifurGauss* bg1 = new RooBifurGauss(_name+"_bg1","BifurGauss",*x,*mean,*sigmaL,*sigmaR);
      RooBifurGauss* bg2 = new RooBifurGauss(_name+"_bg2","BifurGauss",*x,*mean2,*sigmaL2,*sigmaR2);
      RooBifurGauss* bg3 = new RooBifurGauss(_name+"_bg3","Gaussian",*x,*mean3,*sigmaL3,*sigmaR3);
      RooAddPdf* bgsum = new RooAddPdf(_name+"_bgsum","BifurGauss+BifurGauss",*bg1,*bg2,*frac);
      _thePdf = new RooAddPdf(_name,"BifurGauss+BifurGauss",*bgsum,*bg3,*frac2);
  }
  else if (model=="gauss") {
    _thePdf = new RooGaussian(_name,"gaussian",*x,*mean,*sigma);
  }
  else if(model=="dblgauss"){
     RooGaussian* gaus1 = new RooGaussian(_name+"_gaus1","gaussian 1",*x,*mean1,*sigma1);
     RooGaussian* gaus2 = new RooGaussian(_name+"_gaus2","gaussian 2",*x,*mean2,*sigma2);
     _thePdf = new RooAddPdf(_name,"double gaussian",*gaus1,*gaus2,*frac);
  }
  else if (model=="Poly4_DblGauss") {
     RooGaussian* gaus1 = new RooGaussian(_name+"_gaus1","gaussian 1",*x,*mean1,*sigma1);
     RooGaussian* gaus2 = new RooGaussian(_name+"_gaus2","gaussian 2",*x,*mean2,*sigma2);
     RooAddPdf* dblGauss = new RooAddPdf(_name+"_dblGauss","double gaussian",*gaus1,*gaus2,*frac);
     RooArgList* coef_list = new RooArgList(*coef_0,
					    *coef_1,
					    *coef_2,
					    *coef_3,
					    *coef_4);
     RooPolynomial* poly4 = new RooPolynomial(_name+"_poly4","polynomial 4",*x,*coef_list);
    _thePdf = new RooAddPdf(_name,"Poly4+DblGauss PDF",*poly4,*dblGauss,*frac2);
  }
  else if (model=="fourGaussians") {
     RooGaussian* gaus1 = new RooGaussian(_name+"_gaus1","gaussian 1",*x,*mean1,*sigma1);
     RooGaussian* gaus2 = new RooGaussian(_name+"_gaus2","gaussian 2",*x,*mean2,*sigma2);
     RooGaussian* gaus3 = new RooGaussian(_name+"_gaus3","gaussian 3",*x,*mean3,*sigma3);
     RooGaussian* gaus4 = new RooGaussian(_name+"_gaus4","gaussian 4",*x,*mean4,*sigma4);
     RooAddPdf* gg = new RooAddPdf(_name+"_gg","",RooArgSet(*gaus1,*gaus2),*frac1);
     RooAddPdf* ggg = new RooAddPdf(_name+"_ggg","",RooArgSet(*gg,*gaus3),*frac2);
     _thePdf = new RooAddPdf(_name,"sum of four gaussians",*ggg,*gaus4,*frac3);
  }
  else if(model=="exponential")
    _thePdf = new RooExponential(_name,"exponential",*x,*slope);    
  else if(model=="CBShape_BreitWigner") {
    RooCBShape* cbshape = new RooCBShape(_name+"_cbshape","CBShape",*x,*m0,*sigma,*alpha,*n);
    RooBreitWigner* bwshape = new RooBreitWigner(_name+"_bwshape","BreitWigner",*x,*mean,*width);
     _thePdf = new RooAddPdf(_name,"CBShape_BreitWigner",*cbshape,*bwshape,*frac);
  }
  else if (model=="poly7"){
    RooArgList* coef_list = new RooArgList(*coef_0,
                                           *coef_1,
                                           *coef_2,
                                           *coef_3,
                                           *coef_4,
                                           *coef_5,
                                           *coef_6,
                                           *coef_7);

     _thePdf = new RooPolynomial(_name,"poly7",*x,*coef_list);
     }

  else if (model=="flat"){
      RooArgList coef_list(*coef_0);
     _thePdf = new RooPolynomial(_name,"flat",*x,coef_list);
     }


  else if (model=="BreitWigner"){
    _thePdf = new RooBreitWigner(_name,"BreitWigner",*x,*mean,*width);
    }

  else if (model=="histo") {

    if (verbose())
      cout << "opening ROOT file: " << fileName.getVal() << endl;

    TFile modelfile(fileName.getVal());

    if (modelfile.IsZombie()){
        std::cout << "ERROR: Problems in opening " << fileName.getVal() 
                  << ". Aborting...\n";
        abort();
        }

    TH1F* model_histo = (TH1F*) modelfile.Get(dataName.getVal());

    if(model_histo==0){
       std::cout << "ERROR: did not find histogram '" << dataName.getVal() 
                 << "' in the histogram file (" << fileName.getVal() 
                 << "). Aborting..." << std::endl;
       abort();
       }

    RooArgSet* var_set=new RooArgSet(*x);

    RooDataHist* model_histo_roofit=new RooDataHist(_name+"_data",
                                                    "Roohisto",
                                                    *x,
                                                    model_histo);

    if (verbose())
      cout << "building PDF from histogramme: " << model_histo->GetName() << endl;
    _thePdf = new RooHistPdf(_name, 
                             "pdf from histo", 
                             *var_set,
                             *model_histo_roofit,
                             0);
    }

  else {
    cerr << _name << ": Error: Model not specified: \'" << model.getLabel() << "\' unknown\n";
    abort();
    }
}

/*----------------------------------------------------------------------------*/

void RscBaseModel::readDataCard() {
  // read the data card
  if (getDataCard()) {
    RooArgSet(model).readFromFile(getDataCard(), 0, _name);

    if (model=="gauss") {
      readParameter(_name+"_mean",
		    "mean of Gaussian",
		    _name,
		    getDataCard(),
		    mean);

      readParameter(_name+"_sigma",
		    "sigma of Gaussian",
		    _name,
		    getDataCard(),
		    sigma);
    }

    if (model=="Poly4_DblGauss") {
      readParameter(_name+"_mean1",
		    "mean of gaussian 1",
		    _name,
		    getDataCard(),
		    mean1);
      readParameter(_name+"_mean2",
		    "mean of gaussian 2",
		    _name,
		    getDataCard(),
		    mean2);
      readParameter(_name+"_sigma1",
		    "sigma of gaussian 1",
		    _name,
		    getDataCard(),
		    sigma1);
      readParameter(_name+"_sigma2",
		    "sigma of gaussian 2",
		    _name,
		    getDataCard(),
		    sigma2);
      readParameter(_name+"_frac",
		    "fraction of gaussian 1",
		    _name,
		    getDataCard(),
		    frac,0,1,0.5);
      readParameter(_name+"_coef0",
		    "coefficient 0 of poly",
		    _name,
		    getDataCard(),
		    coef_0,0);
      readParameter(_name+"_coef1",
		    "coefficient 1 of poly",
		    _name,
		    getDataCard(),
		    coef_1,0);
      readParameter(_name+"_coef2",
		    "coefficient 2 of poly",
		    _name,
		    getDataCard(),
		    coef_2,0);
      readParameter(_name+"_coef3",
		    "coefficient 3 of poly",
		    _name,
		    getDataCard(),
		    coef_3,0);
      readParameter(_name+"_coef4",
		    "coefficient 4 of poly",
		    _name,
		    getDataCard(),
		    coef_4,0);
      readParameter(_name+"_frac2",
		    "fraction 2 of gaussians",
		    _name,
		    getDataCard(),
		    frac2,0,1,0.5);
    }

    if (model=="BreitWigner") {
      readParameter(_name+"_mean",
		    "mean of Gaussian",
		    _name,
		    getDataCard(),
		    mean);

      readParameter(_name+"_width",
		    "width of Gaussian",
		    _name,
		    getDataCard(),
		    width);
    }

    if (model=="dblgauss") {
      readParameter(_name+"_mean1",
		    "mean of gaussian 1",
		    _name,
		    getDataCard(),
		    mean1);
      readParameter(_name+"_mean2",
		    "mean of gaussian 2",
		    _name,
		    getDataCard(),
		    mean2);
      readParameter(_name+"_sigma1",
		    "sigma of gaussian 1",
		    _name,
		    getDataCard(),
		    sigma1);
      readParameter(_name+"_sigma2",
		    "sigma of gaussian 2",
		    _name,
		    getDataCard(),
		    sigma2);
      readParameter(_name+"_frac",
		    "fraction of gaussian 1",
		    _name,
		    getDataCard(),
		    frac,0,1,0.5);
    }

    if (model=="fourGaussians") {
      readParameter(_name+"_mean1",
		    "mean of gaussian 1",
		    _name,
		    getDataCard(),
		    mean1);
      readParameter(_name+"_mean2",
		    "mean of gaussian 2",
		    _name,
		    getDataCard(),
		    mean2);
      readParameter(_name+"_sigma1",
		    "sigma of gaussian 1",
		    _name,
		    getDataCard(),
		    sigma1);
      readParameter(_name+"_sigma2",
		    "sigma of gaussian 2",
		    _name,
		    getDataCard(),
		    sigma2);
      readParameter(_name+"_mean3",
		    "mean of gaussian 3",
		    _name,
		    getDataCard(),
		    mean3);
      readParameter(_name+"_mean4",
		    "mean of gaussian 4",
		    _name,
		    getDataCard(),
		    mean4);
      readParameter(_name+"_sigma3",
		    "sigma of gaussian 3",
		    _name,
		    getDataCard(),
		    sigma3);
      readParameter(_name+"_sigma4",
		    "sigma of gaussian 4",
		    _name,
		    getDataCard(),
		    sigma4);
      readParameter(_name+"_frac1",
		    "fraction 1 of gaussians",
		    _name,
		    getDataCard(),
		    frac1,0,1,0.5);
      readParameter(_name+"_frac2",
		    "fraction 2 of gaussians",
		    _name,
		    getDataCard(),
		    frac2,0,1,0.5);
      readParameter(_name+"_frac3",
		    "fraction 3 of gaussians",
		    _name,
		    getDataCard(),
		    frac3,0,1,0.5);
    }

    if (model=="exponential") {
      readParameter(_name+"_slope",
		    "exponential slope",
		    _name,
		    getDataCard(),
		    slope,-1,1);
    }

    if (model=="flat") {
      readParameter(_name+"_flat distr",
            "Constant distribution",
            _name,
            getDataCard(),
            coef_0,0);
    }

    if (model=="poly7") {
      readParameter(_name+"_coef0",
		    "coefficient 0 of poly",
		    _name,
		    getDataCard(),
		    coef_0,0);
      readParameter(_name+"_coef1",
		    "coefficient 1 of poly",
		    _name,
		    getDataCard(),
		    coef_1,0);
      readParameter(_name+"_coef2",
		    "coefficient 2 of poly",
		    _name,
		    getDataCard(),
		    coef_2,0);
      readParameter(_name+"_coef3",
		    "coefficient 3 of poly",
		    _name,
		    getDataCard(),
		    coef_3,0);
      readParameter(_name+"_coef4",
		    "coefficient 4 of poly",
		    _name,
		    getDataCard(),
		    coef_4,0);
      readParameter(_name+"_coef5",
		    "coefficient 5 of poly",
		    _name,
		    getDataCard(),
		    coef_5,0);
      readParameter(_name+"_coef6",
		    "coefficient 6 of poly",
		    _name,
		    getDataCard(),
		    coef_6,0);
      readParameter(_name+"_coef7",
		    "coefficient 7 of poly",
		    _name,
		    getDataCard(),
		    coef_7,0);
    }
    if (model=="histo") {
        RooArgSet(fileName,dataName).readFromFile(getDataCard(), 0, _name);
    }
    if (model=="BifurGauss") {
	readParameter(_name+"_mean",
		      "mean",
		      _name,
		      getDataCard(),
		      mean,0);
    	readParameter(_name+"_sigmaL",
		      "sigmaL",
		      _name,
		      getDataCard(),
		      sigmaL,0);
    	readParameter(_name+"_sigmaR",
		      "sigmaR",
		      _name,
		      getDataCard(),
		      sigmaR,0);
    }
    if (model=="Landau_BifurGauss") {
	readParameter(_name+"_mean",
		      "mean",
		      _name,
		      getDataCard(),
		      mean,0);
    	readParameter(_name+"_sigmaL",
		      "sigmaL",
		      _name,
		      getDataCard(),
		      sigmaL,0);
    	readParameter(_name+"_sigmaR",
		      "sigmaR",
		      _name,
		      getDataCard(),
		      sigmaR,0);
	readParameter(_name+"_mean2",
		      "mean2",
		      _name,
		      getDataCard(),
		      mean2,0);
      readParameter(_name+"_sigma2",
		    "sigma of gaussian 2",
		    _name,
		    getDataCard(),
		    sigma2);
	readParameter(_name+"_frac",
		      "frac",
		      _name,
		      getDataCard(),
		      frac,0);
    }
    if (model=="BifurGauss_BifurGauss") {
	readParameter(_name+"_mean",
		      "mean",
		      _name,
		      getDataCard(),
		      mean,0);
    	readParameter(_name+"_sigmaL",
		      "sigmaL",
		      _name,
		      getDataCard(),
		      sigmaL,0);
    	readParameter(_name+"_sigmaR",
		      "sigmaR",
		      _name,
		      getDataCard(),
		      sigmaR,0);
	readParameter(_name+"_mean2",
		      "mean2",
		      _name,
		      getDataCard(),
		      mean2,0);
    	readParameter(_name+"_sigmaL2",
		      "sigmaL2",
		      _name,
		      getDataCard(),
		      sigmaL2,0);
    	readParameter(_name+"_sigmaR2",
		      "sigmaR2",
		      _name,
		      getDataCard(),
		      sigmaR2,0);
	readParameter(_name+"_frac",
		      "frac",
		      _name,
		      getDataCard(),
		      frac,0);
    }
    if (model=="BifurGauss_BifurGauss_BifurGauss") {
	readParameter(_name+"_mean",
		      "mean",
		      _name,
		      getDataCard(),
		      mean,0);
    	readParameter(_name+"_sigmaL",
		      "sigmaL",
		      _name,
		      getDataCard(),
		      sigmaL,0);
    	readParameter(_name+"_sigmaR",
		      "sigmaR",
		      _name,
		      getDataCard(),
		      sigmaR,0);
	readParameter(_name+"_mean2",
		      "mean2",
		      _name,
		      getDataCard(),
		      mean2,0);
    	readParameter(_name+"_sigmaL2",
		      "sigmaL2",
		      _name,
		      getDataCard(),
		      sigmaL2,0);
    	readParameter(_name+"_sigmaR2",
		      "sigmaR2",
		      _name,
		      getDataCard(),
		      sigmaR2,0);
	readParameter(_name+"_frac",
		      "frac",
		      _name,
		      getDataCard(),
		      frac,0);
	readParameter(_name+"_mean3",
		      "mean3",
		      _name,
		      getDataCard(),
		      mean3,0);
    	readParameter(_name+"_sigmaL3",
		      "sigmaL3",
		      _name,
		      getDataCard(),
		      sigmaL3,0);
    	readParameter(_name+"_sigmaR3",
		      "sigmaR3",
		      _name,
		      getDataCard(),
		      sigmaR3,0);
	readParameter(_name+"_frac2",
		      "frac2",
		      _name,
		      getDataCard(),
		      frac2,0);
    }
    if (model=="BifurGauss_BifurGauss_Gauss") {
	readParameter(_name+"_mean",
		      "mean",
		      _name,
		      getDataCard(),
		      mean,0);
    	readParameter(_name+"_sigmaL",
		      "sigmaL",
		      _name,
		      getDataCard(),
		      sigmaL,0);
    	readParameter(_name+"_sigmaR",
		      "sigmaR",
		      _name,
		      getDataCard(),
		      sigmaR,0);
	readParameter(_name+"_mean2",
		      "mean2",
		      _name,
		      getDataCard(),
		      mean2,0);
    	readParameter(_name+"_sigmaL2",
		      "sigmaL2",
		      _name,
		      getDataCard(),
		      sigmaL2,0);
    	readParameter(_name+"_sigmaR2",
		      "sigmaR2",
		      _name,
		      getDataCard(),
		      sigmaR2,0);
	readParameter(_name+"_frac",
		      "frac",
		      _name,
		      getDataCard(),
		      frac,0);
	readParameter(_name+"_gmean",
		      "gmean",
		      _name,
		      getDataCard(),
		      gmean,0);
	readParameter(_name+"_gsigma",
		      "gsigma",
		      _name,
		      getDataCard(),
		      gsigma,0);
	readParameter(_name+"_frac2",
		      "frac2",
		      _name,
		      getDataCard(),
		      frac2,0);
    }
    if (model=="CBShapeGaussian") {
	readParameter(_name+"_m0",
		      "m0",
		      _name,
		      getDataCard(),
		      m0,0);
	readParameter(_name+"_sigma",
		      "sigma",
		      _name,
		      getDataCard(),
		      sigma,0);
	readParameter(_name+"_alpha",
		      "alpha",
		      _name,
		      getDataCard(),
		      alpha,0);
	readParameter(_name+"_n",
		      "n",
		      _name,
		      getDataCard(),
		      n,0);
	readParameter(_name+"_gmean",
		      "gmean",
		      _name,
		      getDataCard(),
		      gmean,0);
	readParameter(_name+"_gsigma",
		      "gsigma",
		      _name,
		      getDataCard(),
		      gsigma,0);
	readParameter(_name+"_frac",
		      "frac",
		      _name,
		      getDataCard(),
		      frac,0);
    if (model=="CBShape_BreitWigner") {
	readParameter(_name+"_m0",
		      "m0",
		      _name,
		      getDataCard(),
		      m0,0);
	readParameter(_name+"_sigma",
		      "sigma",
		      _name,
		      getDataCard(),
		      sigma,0);
	readParameter(_name+"_alpha",
		      "alpha",
		      _name,
		      getDataCard(),
		      alpha,0);
	readParameter(_name+"_n",
		      "n",
		      _name,
		      getDataCard(),
		      n,0);
	readParameter(_name+"_mean",
		      "mean",
		      _name,
		      getDataCard(),
		      mean,0);
	readParameter(_name+"_width",
		      "width",
		      _name,
		      getDataCard(),
		      width,0);
	readParameter(_name+"_frac",
		      "frac",
		      _name,
		      getDataCard(),
		      frac,0);

    }
  }
}

/*----------------------------------------------------------------------------*/

void RscBaseModel::writeDataCard(ostream& out) {
  // write the data card (but not the constraints lines)
  out << "[" << _name << "]" << endl;
    RooArgSet(model).writeToStream(out,false);

    if (model=="gauss")
      RooArgSet(*mean,*sigma)
    .writeToStream(out,false);

    if (model=="fourGaussians") {
      RooArgSet(*mean1,*sigma1,*mean2,*sigma2,*mean3,*sigma3,*mean4,*sigma4)
	.writeToStream(out,false);
      RooArgSet(*frac1,*frac2,*frac3)
	.writeToStream(out,false);
    }

    if (model=="Poly4_DblGauss") {
      RooArgSet(*coef_0,
                *coef_1,
                *coef_2,
                *coef_3,
                *coef_4, *frac2)
                .writeToStream(out,false);
      RooArgSet(*mean1,*sigma1,*mean2,*sigma2,*frac)
	.writeToStream(out,false);
    }

    if (model=="dblgauss") RooArgSet(*mean1,*sigma1,*mean2,*sigma2,*frac)
			     .writeToStream(out,false);

    if (model=="CBShapeGaussian") RooArgSet(*m0,*sigma,*alpha,*n,*gmean,*gsigma,*frac)
			     .writeToStream(out,false);

    if (model=="BifurGauss") RooArgSet(*mean,*sigmaL,*sigmaR)
			     .writeToStream(out,false);

    if (model=="Landau_BifurGauss") RooArgSet(*mean,*sigmaL,*sigmaR,*mean2,*sigma2,*frac)
			     .writeToStream(out,false);

    if (model=="BifurGauss_BifurGauss") RooArgSet(*mean,*sigmaL,*sigmaR,*mean2,*sigmaL2,*sigmaR2,*frac)
			     .writeToStream(out,false);

 
    if (model=="BifurGauss_BifurGauss_BifurGauss") {
      RooArgSet(*mean,*sigmaL,*sigmaR,*mean2,*sigmaL2,*sigmaR2,*frac)
	.writeToStream(out,false);
      RooArgSet(*mean3,*sigmaL3,*sigmaR3,*frac2)
	.writeToStream(out,false);
    }

    if (model=="BifurGauss_BifurGauss_Gauss") {
      RooArgSet(*mean,*sigmaL,*sigmaR,*mean2,*sigmaL2,*sigmaR2,*frac)
	.writeToStream(out,false);
      RooArgSet(*gmean,*gsigma,*frac2)
	.writeToStream(out,false);
    }

    if (model=="exponential") RooArgSet(*slope)
				.writeToStream(out,false);

    if (model=="CBShape_BreitWigner")  RooArgSet(*m0,*sigma,*alpha,*n,*mean,*width,*frac).writeToStream(out,false);

    if (model=="poly7") 
      RooArgSet(*coef_0,
                *coef_1,
                *coef_2,
                *coef_3,
                *coef_4,
                *coef_5,
                *coef_6,
                *coef_7)
                .writeToStream(out,false);

    if (model=="BreitWigner")
        RooArgSet(*mean,*width)
                 .writeToStream(out,false);

    if (model=="histo")
        RooArgSet(fileName, dataName)
                 .writeToStream(out,false);
}

// Automatically converted from the standalone version Wed Apr 15 11:36:34 2009
