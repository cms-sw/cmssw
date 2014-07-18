#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/Candidate/interface/CompositeCandidate.h"
#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"
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
      virtual void beginJob() override ;
      virtual void produce(edm::Event&, const edm::EventSetup&) override;
      virtual void endJob() override ;

      std::string fixPOWHEG_;
      bool useFirstAsDefault_;
      edm::InputTag genTag_;
      edm::EDGetTokenT<reco::GenParticleCollection> genToken_;
      edm::InputTag pdfInfoTag_;
      edm::EDGetTokenT<GenEventInfoProduct> pdfInfoToken_;
      std::vector<std::string> pdfSetNames_;
      std::vector<std::string> pdfShortNames_;
};

namespace LHAPDF {
      void initPDFSet(int nset, const std::string& filename, int member=0);
      int numberPDF(int nset);
      void usePDFMember(int nset, int member);
      double xfx(int nset, double x, double Q, int fl);
      double getXmin(int nset, int member);
      double getXmax(int nset, int member);
      double getQ2min(int nset, int member);
      double getQ2max(int nset, int member);
      void extrapolate(bool extrapolate=true);
}

/////////////////////////////////////////////////////////////////////////////////////
PdfWeightProducer::PdfWeightProducer(const edm::ParameterSet& pset) :
 fixPOWHEG_(pset.getUntrackedParameter<std::string> ("FixPOWHEG", "")),
 useFirstAsDefault_(pset.getUntrackedParameter<bool>("useFirstAsDefault",false)),
 genTag_(pset.getUntrackedParameter<edm::InputTag> ("GenTag", edm::InputTag("genParticles"))),
 genToken_(mayConsume<reco::GenParticleCollection>(genTag_)),
 pdfInfoTag_(pset.getUntrackedParameter<edm::InputTag> ("PdfInfoTag", edm::InputTag("generator"))),
 pdfInfoToken_(consumes<GenEventInfoProduct>(pdfInfoTag_)),
 pdfSetNames_(pset.getUntrackedParameter<std::vector<std::string> > ("PdfSetNames"))
{
      if (fixPOWHEG_ != "") pdfSetNames_.insert(pdfSetNames_.begin(),fixPOWHEG_);

      if (pdfSetNames_.size()>3) {
            edm::LogWarning("") << pdfSetNames_.size() << " PDF sets requested on input. Using only the first 3 sets and ignoring the rest!!";
            pdfSetNames_.erase(pdfSetNames_.begin()+3,pdfSetNames_.end());
      }

      for (unsigned int k=0; k<pdfSetNames_.size(); k++) {
            size_t dot = pdfSetNames_[k].find_first_of('.');
            size_t underscore = pdfSetNames_[k].find_first_of('_');
            if (underscore<dot) {
                  pdfShortNames_.push_back(pdfSetNames_[k].substr(0,underscore));
            } else {
                  pdfShortNames_.push_back(pdfSetNames_[k].substr(0,dot));
            }
            produces<std::vector<double> >(pdfShortNames_[k].data());
      }
}

/////////////////////////////////////////////////////////////////////////////////////
PdfWeightProducer::~PdfWeightProducer(){}

/////////////////////////////////////////////////////////////////////////////////////
void PdfWeightProducer::beginJob() {
      for (unsigned int k=1; k<=pdfSetNames_.size(); k++) {
            LHAPDF::initPDFSet(k,pdfSetNames_[k-1]);
      }
}

/////////////////////////////////////////////////////////////////////////////////////
void PdfWeightProducer::endJob(){}

/////////////////////////////////////////////////////////////////////////////////////
void PdfWeightProducer::produce(edm::Event& iEvent, const edm::EventSetup&) {

      if (iEvent.isRealData()) return;

      edm::Handle<GenEventInfoProduct> pdfstuff;
      if (!iEvent.getByToken(pdfInfoToken_, pdfstuff)) {
            edm::LogError("PDFWeightProducer") << ">>> PdfInfo not found: " << pdfInfoTag_.encode() << " !!!";
            return;
      }

      float Q = pdfstuff->pdf()->scalePDF;

      int id1 = pdfstuff->pdf()->id.first;
      double x1 = pdfstuff->pdf()->x.first;
      double pdf1 = pdfstuff->pdf()->xPDF.first;

      int id2 = pdfstuff->pdf()->id.second;
      double x2 = pdfstuff->pdf()->x.second;
      double pdf2 = pdfstuff->pdf()->xPDF.second;
      if (useFirstAsDefault_ && pdf1 == -1. && pdf2 == -1. ) {
         LHAPDF::usePDFMember(1,0);
         pdf1 = LHAPDF::xfx(1, x1, Q, id1)/x1;
         pdf2 = LHAPDF::xfx(1, x2, Q, id2)/x2;
      }

      // Ad-hoc fix for POWHEG
      if (fixPOWHEG_!="") {
            edm::Handle<reco::GenParticleCollection> genParticles;
            if (!iEvent.getByToken(genToken_, genParticles)) {
                  edm::LogError("PDFWeightProducer") << ">>> genParticles  not found: " << genTag_.encode() << " !!!";
                  return;
            }
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
            LHAPDF::usePDFMember(1,0);
            pdf1 = LHAPDF::xfx(1, x1, Q, id1)/x1;
            pdf2 = LHAPDF::xfx(1, x2, Q, id2)/x2;
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
