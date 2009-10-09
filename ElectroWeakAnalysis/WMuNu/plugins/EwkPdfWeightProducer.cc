#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/Candidate/interface/CompositeCandidate.h"
#include "PhysicsTools/CandUtils/interface/AddFourMomenta.h"
#include "PhysicsTools/CandUtils/interface/Booster.h"
#include <Math/VectorUtil.h>

//
// class declaration
//
class EwkPdfWeightProducer : public edm::EDProducer {
   public:
      explicit EwkPdfWeightProducer(const edm::ParameterSet&);
      ~EwkPdfWeightProducer();

   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

      edm::InputTag pdfInfoTag_;
      const std::string pdfSetName_;
};


#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"
#include "TSystem.h"
#include "LHAPDF/LHAPDF.h"

/////////////////////////////////////////////////////////////////////////////////////
EwkPdfWeightProducer::EwkPdfWeightProducer(const edm::ParameterSet& pset) :
 pdfInfoTag_(pset.getUntrackedParameter<edm::InputTag> ("PdfInfoTag", edm::InputTag("generator"))),
 pdfSetName_(pset.getUntrackedParameter<std::string> ("PdfSetName", "cteq65.LHgrid"))
{
     produces<std::vector<double> >();
} 

/////////////////////////////////////////////////////////////////////////////////////
EwkPdfWeightProducer::~EwkPdfWeightProducer(){}

/////////////////////////////////////////////////////////////////////////////////////
void EwkPdfWeightProducer::beginJob(const edm::EventSetup&) {

      // Force unsetting of the LHAPATH variable to avoid potential problems
      gSystem->Setenv("LHAPATH","");

      /* Examples, see $LHAPATH directory for available sets
            LHAPDF::initPDFByName("cteq65.LHgrid"); // NLO interpolated
            LHAPDF::initPDFByName("MRST2007lomod.LHgrid"); // LO modified, interpolated
            LHAPDF::initPDFByName("cteq6mLHpdf"); // NLO evolved
            LHAPDF::initPDFByName("MRST2004nlo.LHpdf"); // NLO, evolved
      */
      LHAPDF::setVerbosity(LHAPDF::SILENT);
      LHAPDF::initPDFSet(pdfSetName_);
      LHAPDF::getDescription();
}

/////////////////////////////////////////////////////////////////////////////////////
void EwkPdfWeightProducer::endJob(){}

/////////////////////////////////////////////////////////////////////////////////////
void EwkPdfWeightProducer::produce(edm::Event& iEvent, const edm::EventSetup&) {

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

      // Put PDF weights in the event
      std::auto_ptr<std::vector<double> > weights (new std::vector<double>);
      unsigned int nmembers = LHAPDF::numberPDF() + 1;
      weights->reserve(nmembers);

      for (unsigned int i=0; i<nmembers; ++i) {
            LHAPDF::usePDFMember(i);
            double newpdf1 = LHAPDF::xfx(x1, Q, id1)/x1;
            double newpdf2 = LHAPDF::xfx(x2, Q, id2)/x2;
            weights->push_back(newpdf1/pdf1*newpdf2/pdf2);
      }
      iEvent.put(weights);
}

DEFINE_FWK_MODULE(EwkPdfWeightProducer);
