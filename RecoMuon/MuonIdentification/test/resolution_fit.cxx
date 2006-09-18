RooPlot* resolution_fit(TH1* histo, TString title = "")
{
   RooRealVar x("x","x",0);
   RooRealVar mean1("mean1","mean1",-100,100);
   RooRealVar mean2("mean2","mean2",-100,100);
   RooRealVar sigma1("sigma1","sigma1",0.001,100);
   RooRealVar sigma2("sigma2","sigma2",1,100);
   RooGaussian pdf1("gaus1","gaus1",x,mean1,sigma1);
   RooGaussian pdf2("gaus2","gaus2",x,mean2,sigma2);
   RooRealVar frac("frac","frac",0,1);
   RooAddPdf pdf("pdf","pdf",pdf1,pdf2,frac);
   RooDataHist data("data","data",x,histo);
   pdf.fitTo(data,RooFit::Minos(kFALSE));
   frame=x.frame();
   data.plotOn(frame);
   data.statOn(frame,What("N"));
   pdf.paramOn(frame,Format("NEA",AutoPrecision(2)));
   pdf.plotOn(frame);
   frame->SetTitle(title);
   frame->Draw();
   return frame;
}

