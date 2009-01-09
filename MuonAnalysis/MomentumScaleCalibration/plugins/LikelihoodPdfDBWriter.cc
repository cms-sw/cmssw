// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

// Conditions database
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

#include "LikelihoodPdfDBWriter.h"
// #include "MuonAnalysis/MomentumScaleCalibrationObjects/interface/MuScleFitLikelihoodPdf.h"
#include "CondFormats/MomentumScaleCalibrationObjects/interface/MuScleFitLikelihoodPdf.h"
#include <TH2D.h>

using namespace std;
using namespace edm;
using namespace PhysicsTools;

LikelihoodPdfDBWriter::LikelihoodPdfDBWriter(const edm::ParameterSet& ps)

{
  // now do what ever initialization is needed
  inputFile_ = ps.getParameter<string>( "inputFileName" );
}

LikelihoodPdfDBWriter::~LikelihoodPdfDBWriter()
{
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}

// ------------ method called to for each event  ------------
void
LikelihoodPdfDBWriter::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  vector<TFile*> files;
  files.push_back( TFile::Open( inputFile_.c_str() ) );

  // names in the ROOT files could not match names in the Likelihood
  vector<string> variablesROOT;
  variablesROOT.push_back("GL0");
  variablesROOT.push_back("GL1");
  variablesROOT.push_back("GL2");
  variablesROOT.push_back("GL3");
  variablesROOT.push_back("GL4");
  variablesROOT.push_back("GL5");

  variablesROOT.push_back("GLZ0");
  variablesROOT.push_back("GLZ1");
  variablesROOT.push_back("GLZ2");
  variablesROOT.push_back("GLZ3");
  variablesROOT.push_back("GLZ4");
  variablesROOT.push_back("GLZ5");
  variablesROOT.push_back("GLZ6");
  variablesROOT.push_back("GLZ7");
  variablesROOT.push_back("GLZ8");
  variablesROOT.push_back("GLZ9");
  variablesROOT.push_back("GLZ10");
  variablesROOT.push_back("GLZ11");
  variablesROOT.push_back("GLZ12");
  variablesROOT.push_back("GLZ13");
  variablesROOT.push_back("GLZ14");
  variablesROOT.push_back("GLZ15");
  variablesROOT.push_back("GLZ16");
  variablesROOT.push_back("GLZ17");
  variablesROOT.push_back("GLZ18");
  variablesROOT.push_back("GLZ19");
  variablesROOT.push_back("GLZ20");
  variablesROOT.push_back("GLZ21");
  variablesROOT.push_back("GLZ22");
  variablesROOT.push_back("GLZ23");
  variablesROOT.push_back("GLZ24");
  variablesROOT.push_back("GLZ25");
  variablesROOT.push_back("GLZ26");
  variablesROOT.push_back("GLZ27");
  variablesROOT.push_back("GLZ28");
  variablesROOT.push_back("GLZ29");
  variablesROOT.push_back("GLZ30");
  variablesROOT.push_back("GLZ31");
  variablesROOT.push_back("GLZ32");
  variablesROOT.push_back("GLZ33");
  variablesROOT.push_back("GLZ34");
  variablesROOT.push_back("GLZ35");
  variablesROOT.push_back("GLZ36");
  variablesROOT.push_back("GLZ37");
  variablesROOT.push_back("GLZ38");
  variablesROOT.push_back("GLZ39");

  MuScleFitLikelihoodPdf * likelihoodPdf = new MuScleFitLikelihoodPdf;

  files[0]->cd();
  vector<string>::const_iterator histoName = variablesROOT.begin();
  int histoNum = 0;
  for( ; histoName != variablesROOT.end(); ++histoName, ++histoNum) {
    TH2D *histo = (TH2D*)files[0]->Get( &((*histoName)[0]) );
    cout << "histo name = " << histo->GetName() << endl;
    // fill a FW histogram to be put in the DB
    int nBinsX = histo->GetNbinsX();
    TAxis * xAxis = histo->GetXaxis();
    float xMin = xAxis->GetBinLowEdge(1);
    float xMax = xAxis->GetBinLowEdge(nBinsX+1);
    int nBinsY = histo->GetNbinsY();
    TAxis * yAxis = histo->GetYaxis();
    float yMin = yAxis->GetBinLowEdge(1);
    float yMax = yAxis->GetBinLowEdge(nBinsY+1);

    Calibration::HistogramD2D pdfHisto(nBinsX,xMin,xMax,nBinsY,yMin,yMax);

    for(int xBin=0; xBin<=nBinsX+1; ++xBin) {
      for(int yBin=0; yBin<=nBinsY+1; ++yBin) {
        // cout << "for calibHisto = " << calibHisto << " xBin = " << xBin << ", yBin = " << yBin << ", value = " << histo->GetBinContent(xBin,yBin) << endl;
        // calibHisto->setBinContent(xBin, yBin, histo->GetBinContent(xBin,yBin));
        pdfHisto.setBinContent(xBin, yBin, histo->GetBinContent(xBin,yBin));
        cout << *histoName << "("<<xBin<<", "<<yBin<<" ) = " << histo->GetBinContent(xBin, yBin ) << endl;
      }
    }

    // Save the histogram in the object that will be written to db.
    likelihoodPdf->histograms.push_back(pdfHisto);
    likelihoodPdf->names.push_back(*histoName);
    likelihoodPdf->xBins.push_back(nBinsX);
    likelihoodPdf->yBins.push_back(nBinsY);
  }

  // Save the histograms in the db.
  edm::Service<cond::service::PoolDBOutputService> mydbservice;
  if( mydbservice.isAvailable() ){
    if( mydbservice->isNewTagRequest("MuScleFitLikelihoodPdfRcd") ){
      mydbservice->createNewIOV<MuScleFitLikelihoodPdf>(likelihoodPdf,mydbservice->beginOfTime(),mydbservice->endOfTime(),"MuScleFitLikelihoodPdfRcd");      
    } else {
      mydbservice->appendSinceTime<MuScleFitLikelihoodPdf>(likelihoodPdf,mydbservice->currentTime(),"MuScleFitLikelihoodPdfRcd");      
    }
  } else {
    edm::LogError("LikelihoodPdfDBWriter")<<"Service is unavailable"<<std::endl;
  }

}

// ------------ method called once each job just before starting event loop  ------------
void 
LikelihoodPdfDBWriter::beginJob(const edm::EventSetup&)
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
LikelihoodPdfDBWriter::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(LikelihoodPdfDBWriter);
