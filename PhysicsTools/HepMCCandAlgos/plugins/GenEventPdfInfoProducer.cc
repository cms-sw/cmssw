/** \class GenEventPdfInfoProducer
 *
 * \author C.Hof, H.Pieta
 *
 * Save the PDFInfo stored in HepMC so that they are available in AODs
 *
 */
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "DataFormats/HepMCCandidate/interface/PdfInfo.h"

namespace edm { class ParameterSet; }
namespace HepMC { class GenEvent; class PDFInfo;}

class GenEventPdfInfoProducer : public edm::EDProducer {
 public:
  /// constructor
  GenEventPdfInfoProducer(const edm::ParameterSet &);

 private:
  void produce(edm::Event& evt, const edm::EventSetup& es);
  edm::InputTag src_;
};

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
using namespace edm;
using namespace std;
using namespace HepMC;

GenEventPdfInfoProducer::GenEventPdfInfoProducer(const ParameterSet & p) :
  src_(p.getParameter<InputTag>("src")) {
  produces<reco::PdfInfo>();
}

void GenEventPdfInfoProducer::produce(Event& evt, const EventSetup& es) {
  // get the HepMC
  Handle<HepMCProduct> mcp;
  evt.getByLabel(src_, mcp);
  const GenEvent * mc = mcp->GetEvent();
  if(mc == 0) {
    throw edm::Exception(edm::errors::InvalidReference) 
      << "HepMC has null pointer to GenEvent\n";
  }
  PdfInfo* pdf_info = mc->pdf_info();    
  auto_ptr<reco::PdfInfo> info(new reco::PdfInfo);
  if (pdf_info != 0) {
     info->id1 = pdf_info->id1();
     info->id2 = pdf_info->id2();
     info->x1  = pdf_info->x1();
     info->x2  = pdf_info->x2();
     info->scalePDF = pdf_info->scalePDF();
     info->pdf1 = pdf_info->pdf1();
     info->pdf2 = pdf_info->pdf2();
  } else {
     info->id1 = -99;
     info->id2 = -99;  
     info->x1  = -1.;
     info->x2  = -1.;
     info->scalePDF = -1.;
     info->pdf1 = -1.;
     info->pdf2 = -1.;         
  }
  evt.put(info);
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(GenEventPdfInfoProducer);
