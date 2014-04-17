// Author: Benedikt Hegner
// Email:  benedikt.hegner@cern.ch

#include "TFile.h"
#include "TList.h"
#include "TString.h"
#include "TKey.h"
#include "TH1.h"
#include <sstream>
#include <vector>
#include <iterator>
#include <algorithm>
#include <sstream>
#include <memory>
#include <string>
#include <fstream>
#include <iostream>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "CondFormats/JetMETObjects/interface/QGLikelihoodObject.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

class  QGLikelihoodDBWriter : public edm::EDAnalyzer
{
 public:
  QGLikelihoodDBWriter(const edm::ParameterSet&);
  virtual void beginJob() override;
  virtual void analyze(const edm::Event&, const edm::EventSetup&) override {}
  virtual void endJob() override {}
  ~QGLikelihoodDBWriter() {}

 private:
  std::string inputRootFile;
  std::string payloadTag;
};

// Constructor
QGLikelihoodDBWriter::QGLikelihoodDBWriter(const edm::ParameterSet& pSet)
{
  inputRootFile    = pSet.getParameter<std::string>("src");
  payloadTag       = pSet.getParameter<std::string>("payload");
}

// Begin Job
void QGLikelihoodDBWriter::beginJob()
{

  QGLikelihoodObject *payload = new QGLikelihoodObject();
  payload->data.clear();

  TFile * f = TFile::Open(edm::FileInPath(inputRootFile.c_str()).fullPath().c_str());
  
  // For easy keeping, here are the various strings in the histogram name. 
  std::string varName0 = "nPFCand_QC_ptCutJet0";
  std::string varName1 = "ptD_QCJet0";
  std::string varName2 = "axis1_QCJet0";
  std::string varName3 = "axis2_QCJet0";
  std::string etaName0 = "_F";
  std::string etaName1 = "";
  std::string quarkName= "_quark_";
  std::string gluonName= "_gluon_";
  std::string rhoName  = "_rho";

  // The structure of the root file is that it is in a bunch of 
  // subdirectories. Need to traverse through them and add the histograms. 
  TList * keys = f->GetListOfKeys();
  if ( !keys ) {
    edm::LogError  ("NoKeys") << "There are no keys in the input file." << std::endl;
    return;
  }


  TIter nextdir(keys);
  TKey *keydir;
  while ((keydir = (TKey*)nextdir())) {
    TDirectory * dir = (TDirectory*)keydir->ReadObj() ;
    TIter nexthist(dir->GetListOfKeys());
    TKey * keyhist;
    while ( (keyhist = (TKey*)nexthist())) {

      double ptMin=0.0, ptMax=0.0, rhoVal=-99.99;
      int etaBin=-1;
      int varIndex=-1;
      int qgBin=-1;

      std::string histname = keyhist->GetName();

      // Histogram names encode the binning. Examples: 
      //     nPFCand_QC_ptCutJet0_gluon_pt40_51_rho0
      //     ptD_QCJet0_gluon_pt40_51_rho0
      //     axis1_QCJet0_gluon_pt40_51_rho0
      //     axis2_QCJet0_gluon_pt40_51_rho0
      size_t varName0Pos = histname.find( varName0 );
      size_t varName1Pos = histname.find( varName1 );
      size_t varName2Pos = histname.find( varName2 );
      size_t varName3Pos = histname.find( varName3 );
      size_t eta_qg_0Pos = histname.find( etaName0 + quarkName );
      size_t eta_qg_1Pos = histname.find( etaName1 + quarkName );
      size_t eta_qg_2Pos = histname.find( etaName0 + gluonName );
      size_t eta_qg_3Pos = histname.find( etaName1 + gluonName );
      size_t rhoPos      = histname.find( rhoName );
      size_t ptFirstPos  = 0;


      // First check the variable name, and offset the first position of the "pt" substring
      if ( varName0Pos != std::string::npos ) {
	varIndex = 0;
	ptFirstPos += varName0.size();
      }
      else if ( varName1Pos != std::string::npos ) {
	varIndex = 1;
	ptFirstPos += varName1.size();
      }
      else if ( varName2Pos != std::string::npos ) {
	varIndex = 2;
	ptFirstPos += varName2.size();
      }
      else if ( varName3Pos != std::string::npos ) {
	varIndex = 3;
	ptFirstPos += varName3.size();
      }
      else {
	edm::LogError  ("NoName") << "Cannot find the variable name " << std::endl;
	return;
      }

      // Next check central vs. forward eta, q vs g, and get the offest of the position where the pt is stored
      if ( eta_qg_0Pos != std::string::npos ) {
	etaBin = 0;
	qgBin = 0;
	ptFirstPos += etaName0.size() + quarkName.size();
      }
      else if ( eta_qg_1Pos != std::string::npos ) {
	etaBin = 1;
	qgBin = 0;
	ptFirstPos += etaName1.size() + quarkName.size();
      }
      else if ( eta_qg_2Pos != std::string::npos ) {
	etaBin = 0;
	qgBin = 1;
	ptFirstPos += etaName0.size() + gluonName.size();
      }
      else if ( eta_qg_3Pos != std::string::npos ) {
	etaBin = 1;
	qgBin = 1;
	ptFirstPos += etaName1.size() + gluonName.size();
      } else {
	edm::LogError  ("NoBins") << "Cannot find eta, qg and pt bins." << std::endl;
	return;
      }

      // Access the pt information
      ptFirstPos += 2;
      char junk('_');
      std::stringstream sptVal ( histname.substr( ptFirstPos, rhoPos ) );
      sptVal >> ptMin >> junk >> ptMax;

      // Access the rho information
      size_t rhoEndPos = rhoPos + rhoName.size();
      std::stringstream srhoVal ( histname.substr( rhoEndPos, histname.size() ) );
      srhoVal >> rhoVal;
       

      // Print out for debugging      
      char buff[1000];
      sprintf( buff, "%50s : var=%1d, eta=%1d, qg=%1d, ptMin=%8.2f, ptMax=%8.2f, rhoVal=%6.2f", histname.c_str(), varIndex, etaBin, qgBin, ptMin, ptMax, rhoVal );
      edm::LogVerbatim   ("HistName") << buff << std::endl;

      // Create the new QGLikelihoodCategory and add to the list. 
      TObject * objhist = keyhist->ReadObj();
      TH1* th1hist = (TH1*)objhist;
      


      QGLikelihoodCategory category;
      category.RhoVal = rhoVal;
      category.PtMin = ptMin;
      category.PtMax = ptMax;
      category.EtaBin = etaBin;
      category.QGIndex = qgBin;
      category.VarIndex = varIndex;
      QGLikelihoodObject::Histogram histogram (th1hist->GetNbinsX(), 
					       th1hist->GetXaxis()->GetBinLowEdge(1), 
					       th1hist->GetXaxis()->GetBinUpEdge( th1hist->GetNbinsX() )
					       );
      for ( int ibin = 0; ibin < th1hist->GetNbinsX(); ++ibin ) {
	histogram.setBinContent( ibin, th1hist->GetBinContent( ibin+1 ) ); // ROOT TH1 indexing off-by-one
      }

      QGLikelihoodObject::Entry entry;
      entry.category = category;
      entry.histogram = histogram; 
      payload->data.push_back( entry );

    }

  }

   
  

  edm::LogInfo   ("UserOutput") << "Opening PoolDBOutputService" << std::endl;

  // now write it into the DB
  edm::Service<cond::service::PoolDBOutputService> s;
  if (s.isAvailable()) 
    {
      edm::LogInfo   ("UserOutput") <<  "Setting up payload with " << payload->data.size() <<  " entries and tag " << payloadTag << std::endl;
      if (s->isNewTagRequest(payloadTag)) 
	s->createNewIOV<QGLikelihoodObject>(payload, s->beginOfTime(), s->endOfTime(), payloadTag);
      else 
	s->appendSinceTime<QGLikelihoodObject>(payload, 111, payloadTag);
    }
  edm::LogInfo   ("UserOutput") <<  "Wrote in CondDB QGLikelihood payload label: " << payloadTag << std::endl;
}


DEFINE_FWK_MODULE(QGLikelihoodDBWriter);

