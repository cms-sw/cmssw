#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/Candidate/interface/CompositeCandidate.h"
#include "CommonTools/CandUtils/interface/AddFourMomenta.h"
#include "CommonTools/CandUtils/interface/Booster.h"
#include <Math/VectorUtil.h>

//
// class declaration
//
class PdfWeightProducer : public edm::EDProducer {
   public:
      explicit PdfWeightProducer(const edm::ParameterSet&);
      ~PdfWeightProducer();

   private:
      virtual void beginJob() ;
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

      bool fixPOWHEG_;
      edm::InputTag pdfInfoTag_;
      std::vector<std::string> pdfSetNames_;
      std::vector<std::string> pdfShortNames_;
      int powhegPdfIndex_;
};

#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"
namespace LHAPDF {
      void initPDFSet(int nset, const std::string& filename, int member=0);
      int numberPDF(int nset);
      void usePDFMember(int nset, int member);
      double xfx(int nset, double x, double Q, int fl);
}

/////////////////////////////////////////////////////////////////////////////////////
PdfWeightProducer::PdfWeightProducer(const edm::ParameterSet& pset) :
 fixPOWHEG_(pset.getUntrackedParameter<bool> ("FixPOWHEG", false)),
 pdfInfoTag_(pset.getUntrackedParameter<edm::InputTag> ("PdfInfoTag", edm::InputTag("generator"))),
 pdfSetNames_(pset.getUntrackedParameter<std::vector<std::string> > ("PdfSetNames"))
{
      if (pdfSetNames_.size()>3) {
            edm::LogWarning("") << pdfSetNames_.size() << " PDF sets requested on input. Using only the first 3 sets and ignoring the rest!!";
            pdfSetNames_.erase(pdfSetNames_.begin()+3,pdfSetNames_.end());
      }

      powhegPdfIndex_ = -1;
      for (unsigned int k=0; k<pdfSetNames_.size(); k++) {
            size_t dot = pdfSetNames_[k].find_first_of('.');
            pdfShortNames_.push_back(pdfSetNames_[k].substr(0,dot));
            produces<std::vector<double> >(pdfShortNames_[k].data());
            if (fixPOWHEG_ && pdfShortNames_[k]=="cteq6mE") {
                  powhegPdfIndex_ = k+1;
            } else if (fixPOWHEG_ && powhegPdfIndex_<0 && pdfShortNames_[k]=="cteq6m"){
                   powhegPdfIndex_ = k+1;
            }
      }
      if (fixPOWHEG_ && powhegPdfIndex_<0) {
            edm::LogWarning("") << " You have requested to fix POWHEG PDFs, but a cteq6m set has not be found in the list";
            edm::LogWarning("") << " We will use the first PDF set found in the list as reference !!";
            powhegPdfIndex_ = 1;
      }
} 

/////////////////////////////////////////////////////////////////////////////////////
PdfWeightProducer::~PdfWeightProducer(){}

/////////////////////////////////////////////////////////////////////////////////////
void PdfWeightProducer::beginJob() {
      for (unsigned int k=1; k<=pdfSetNames_.size(); k++) {
            LHAPDF::initPDFSet(k,pdfSetNames_[k-1]);
            //LHAPDF::getDescription(k);
      }
}

/////////////////////////////////////////////////////////////////////////////////////
void PdfWeightProducer::endJob(){}

/////////////////////////////////////////////////////////////////////////////////////
void PdfWeightProducer::produce(edm::Event& iEvent, const edm::EventSetup&) {

      if (iEvent.isRealData()) return;

      edm::Handle<GenEventInfoProduct> pdfstuff;
      if (!iEvent.getByLabel(pdfInfoTag_, pdfstuff)) return;

      float Q = pdfstuff->pdf()->scalePDF;

      int id1 = pdfstuff->pdf()->id.first;
      double x1 = pdfstuff->pdf()->x.first;
      double pdf1 = pdfstuff->pdf()->xPDF.first;

      int id2 = pdfstuff->pdf()->id.second;
      double x2 = pdfstuff->pdf()->x.second;
      double pdf2 = pdfstuff->pdf()->xPDF.second; 

      // Ad-hoc fix for POWHEG
      if (fixPOWHEG_) {
            edm::Handle<reco::GenParticleCollection> genParticles;
            if (!iEvent.getByLabel(edm::InputTag("genParticles"), genParticles)) return;
            unsigned int gensize = genParticles->size();
            double mboson = 0.;
            for(unsigned int i = 0; i<gensize; ++i) {
                  const reco::GenParticle& part = (*genParticles)[i];
                  int status = part.status();
                  if (status!=3) continue;
                  int id = part.pdgId();
                  if (id!=23 && abs(id)!=24) continue;
                  mboson = part.mass();
                  break;
            }
            Q = sqrt(mboson*mboson+Q*Q);
            LHAPDF::usePDFMember(powhegPdfIndex_,0);
            pdf1 = LHAPDF::xfx(powhegPdfIndex_, x1, Q, id1)/x1;
            pdf2 = LHAPDF::xfx(powhegPdfIndex_, x2, Q, id2)/x2;
      }

      // Put PDF weights in the event
      for (unsigned int k=1; k<=pdfSetNames_.size(); ++k) {
            std::auto_ptr<std::vector<double> > weights (new std::vector<double>);
            unsigned int nweights = 1;
            if (LHAPDF::numberPDF(k)>1) nweights += LHAPDF::numberPDF(k);
            weights->reserve(nweights);
        
            for (unsigned int i=0; i<nweights; ++i) {
                  LHAPDF::usePDFMember(k,i);
                  double newpdf1 = LHAPDF::xfx(k, x1, Q, id1)/x1;
                  double newpdf2 = LHAPDF::xfx(k, x2, Q, id2)/x2;
                  weights->push_back(newpdf1/pdf1*newpdf2/pdf2);
            }
            iEvent.put(weights,pdfShortNames_[k-1]);
      }
}

DEFINE_FWK_MODULE(PdfWeightProducer);
