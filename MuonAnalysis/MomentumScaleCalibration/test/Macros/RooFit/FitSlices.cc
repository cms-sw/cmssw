#ifndef FitSlices_cc
#define FitSlices_cc

#include "FitXslices.cc"
#include "TFile.h"
#include "TH1F.h"
#include "TROOT.h"

/**
 * This class can be used to fit the X slices of a TH1 histogram using RooFit.
 * It uses the FitXslices class to do the fitting.
 */
class FitSlices
{
public:
  FitSlices() :
    rebinX(2), rebinY(2), rebinZ(2), sigma2(0.1), sigma2Min(0.), sigma2Max(10.), useChi2(false)
  {}

  // virtual void fit(const TString & inputFileName = "0_MuScleFit.root", const TString & outputFileName = "BiasCheck_0.root",
  // 		   const TString & signalType = "gaussian", const TString & backgroundType = "exponential",
  // 		   const double & xMean = 3.1, const double & xMin = 3., const double & xMax = 3.2,
  // 		   const double & sigma = 0.03, const double & sigmaMin = 0., const double & sigmaMax = 0.1,
  // 		   const TString & histoBaseName = "hRecBestResVSMu", const TString & histoBaseTitle = "MassVs") = 0;

  void fitSlice(const TString & histoName, const TString & dirName,
		const double & xMean, const double & xMin, const double & xMax,
		const double & sigma, const double & sigmaMin, const double & sigmaMax,
		const TString & signalType, const TString & backgroundType,
		TFile * inputFile, TDirectory * outputFile)
  {
    FitXslices fitXslices;
    fitXslices.fitter()->useChi2_ = useChi2;
    fitXslices.fitter()->initMean( xMean, xMin, xMax );
    fitXslices.fitter()->initSigma( sigma, sigmaMin, sigmaMax );
    fitXslices.fitter()->initSigma2( sigma2, sigma2Min, sigma2Max );

    fitXslices.fitter()->initAlpha(1.5, 0.05, 10.);
    fitXslices.fitter()->initN(1, 0.01, 100.);
    fitXslices.fitter()->initFGCB(0.4, 0., 1.);

    std::cout << "Fit slices: initialization complete" << std::endl;
    
    //r.c. patch --------------
    if (histoName=="hRecBestResVSMu_MassVSEtaPhiPlus" || histoName=="hRecBestResVSMu_MassVSEtaPhiMinus" ||
	histoName=="hRecBestResVSMu_MassVSPhiPlusPhiMinus" || histoName=="hRecBestResVSMu_MassVSEtaPlusEtaMinus"
	)
      {
	TH3* histoPt3 = (TH3*)inputFile->FindObjectAny(histoName);
	outputFile->mkdir(dirName);
	outputFile->cd(dirName);

	//	histoPt3 = rebin3D(histoPt3);
	fitXslices(histoPt3, xMin, xMax, signalType, backgroundType, rebinZ); 

	//	histoPt3->RebinX(rebinX);
	//	histoPt3->RebinY(rebinX);
	//	histoPt3->RebinY(rebinY);
	//	(histoPt3->DoProject2D())->RebinX(rebinX);
	//	(histoPt3->DoProject2D())->RebinY(rebinY);
      }
    else  {
      TH2* histoPt2 = (TH2*)inputFile->FindObjectAny(histoName);
      histoPt2->RebinX(rebinX);
      histoPt2->RebinY(rebinY);
      // TH2 * histoPt = 0;
      // inputFile->GetObject(histoName, histoPt);
      outputFile->mkdir(dirName);
      outputFile->cd(dirName);
      fitXslices(histoPt2, xMin, xMax, signalType, backgroundType, rebinZ);
    } 
   
 }
  unsigned int rebinX;
  unsigned int rebinY;
  unsigned int rebinZ;
  double sigma2, sigma2Min, sigma2Max;
  bool useChi2;

  //~ TH3* rebin3D(TH3* histo3D)
  //~ {
    //~ unsigned int zbins=histo3D->GetNbinsZ();
    //~ // std::cout<< "number of bins in z (and tempHisto) --> "<<zbins<<std::endl;
    //~ std::map<unsigned int, TH2*> twoDprojection;
    //~ for(unsigned int z=1;z<zbins;++z) {
      //~ histo3D->GetZaxis()->SetRange(z,z);
      //~ TH2*tempHisto= (TH2*)histo3D->Project3D("xy");
      //~ std::stringstream ss;
      //~ ss << z;
      //~ tempHisto->SetName(TString(tempHisto->GetName())+ss.str());
      //~ tempHisto->RebinX(rebinX);
      //~ tempHisto->RebinY(rebinY);
      //~ twoDprojection.insert(std::make_pair(z,tempHisto));
    //~ }
    //~ unsigned int xbins, ybins;
    //~ TH3* rebinned3D= new TH3(TString(histo3D->GetName())+"_rebinned",histo3D->GetTitle(),
			    //~ xbins,histo3D->GetXaxis()->GetXmin(),histo3D->GetXaxis()->GetXmax(),
			    //~ ybins,histo3D->GetYaxis()->GetXmin(),histo3D->GetYaxis()->GetXmax(),
			    //~ zbins,histo3D->GetZaxis()->GetXmin(),histo3D->GetZaxis()->GetXmax() );
    //~ if(twoDprojection.size()!=0)
      //~ {
	//~ xbins=twoDprojection[1]->GetNbinsX();	
	//~ ybins=twoDprojection[1]->GetNbinsY();	
	//~ //std::cout<< "number of bins in x --> "<<xbins<<std::endl;
	//~ //std::cout<< "number of bins in y --> "<<ybins<<std::endl;
	//~ for(unsigned int z=1;z<zbins;++z)
	  //~ {
	    //~ for(unsigned int y=1;y<ybins;++y)
	      //~ {
		//~ for(unsigned int x=1;x<xbins;++x)
		  //~ {
		    //~ std::cout<< "x/y/z= "<<x<<"/"<<y<<"/"<<z <<std::endl;
		    //~ std::cout<< "number of bins in x --> "<<xbins<<std::endl;
		    //~ std::cout<< "number of bins in y --> "<<ybins<<std::endl;
		    //~ rebinned3D->Fill(x,y,twoDprojection[z]->GetBinContent(x,y));
		  //~ }
		//~ 
	      //~ }
//~ 
	  //~ }
      //~ }
    //~ return rebinned3D;
  //~ }

};

#endif
