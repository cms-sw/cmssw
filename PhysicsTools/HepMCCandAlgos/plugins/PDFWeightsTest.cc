#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/LHEEventProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"
#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"
#include "PhysicsTools/HepMCCandAlgos/interface/PDFWeightsHelper.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "TTree.h"
#include <iostream>
using namespace std;
using namespace edm;
using namespace HepMC;

class PDFWeightsTest : public edm::one::EDAnalyzer<> {
private:
  PDFWeightsHelper pdfweightshelper_;
  EDGetTokenT<LHEEventProduct> srcToken_;
  EDGetTokenT<LHEEventProduct> srcTokenAlt_;
  EDGetTokenT<GenEventInfoProduct> srcTokenGen_;
  
  unsigned int pdfWeightOffset_;
  unsigned int nPdfWeights_;
  unsigned int nPdfEigWeights_;
  
  TTree *tree_;
  std::vector<float> pdfweights_;
  std::vector<float> pdfeigweights_;
  float weight_;
  
public:
  explicit PDFWeightsTest( const ParameterSet & cfg ) :
    srcToken_( consumes<LHEEventProduct>(edm::InputTag("externalLHEProducer"))),
    srcTokenAlt_( consumes<LHEEventProduct>(edm::InputTag("source"))),
    srcTokenGen_( consumes<GenEventInfoProduct>(edm::InputTag("generator"))),
    pdfWeightOffset_(cfg.getParameter<unsigned int>("pdfWeightOffset")),
    nPdfWeights_(cfg.getParameter<unsigned int>("nPdfWeights")),
    nPdfEigWeights_(cfg.getParameter<unsigned int>("nPdfEigWeights")),
    pdfweights_(nPdfWeights_),
    pdfeigweights_(nPdfEigWeights_)
  {
    
    edm::Service<TFileService> fs;
    tree_ = fs->make<TTree>("tree", "");
    
    tree_->Branch("pdfrep",&pdfweights_);
    tree_->Branch("pdfeig",&pdfeigweights_);
    tree_->Branch("weight",&weight_);
    
    edm::FileInPath mc2hessianCSV = cfg.getParameter<edm::FileInPath>("mc2hessianCSV");
    pdfweightshelper_.Init(nPdfWeights_,nPdfEigWeights_,mc2hessianCSV);
    
  }
  
private:
  void analyze( const Event & evt, const EventSetup& es ) override {
    Handle<LHEEventProduct> lheInfo;
    evt.getByToken( srcToken_, lheInfo );
    
    if (!lheInfo.isValid()) {
      evt.getByToken( srcTokenAlt_, lheInfo );
    }

    double nomlheweight = lheInfo->weights()[0].wgt;

    Handle<GenEventInfoProduct> genInfo;
    evt.getByToken( srcTokenGen_, genInfo );
    
    weight_ = genInfo->weight();
    
    //get the original mc replica weights
    std::vector<double> inpdfweights(nPdfWeights_);
    for (unsigned int ipdf=0; ipdf<nPdfWeights_; ++ipdf) {
      unsigned int iwgt = ipdf + pdfWeightOffset_;
      
      //this is the weight to be used for evaluating uncertainties with mc replica weights
      pdfweights_[ipdf] = lheInfo->weights()[iwgt].wgt*weight_/nomlheweight;
      
      //this is the raw weight to be fed to the mc2hessian convertor
      inpdfweights[ipdf] = lheInfo->weights()[iwgt].wgt;
      
    }
    
    std::vector<double> outpdfweights(nPdfEigWeights_);
    //do the actual conversion, where the nominal lhe weight is needed as the reference point for the linearization
    pdfweightshelper_.DoMC2Hessian(nomlheweight,inpdfweights.data(),outpdfweights.data());
    
    for (unsigned int iwgt=0; iwgt<nPdfEigWeights_; ++iwgt) {
      double wgtval = outpdfweights[iwgt];
      
      //the is the weight to be used for evaluating uncertainties with hessian weights
      pdfeigweights_[iwgt] = wgtval*weight_/nomlheweight;
    }    

    tree_->Fill();
    
  }


};

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE( PDFWeightsTest );



