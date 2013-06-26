#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimDataFormats/GeneratorProducts/interface/LHEEventProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/LHERunInfoProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"

#include <sstream>
//class that reweights a pure parton level event from the originale COM energy to 
//an energy that is < than original COM energy.

//
// class declaration
//
class LHECOMWeightProducer : public edm::EDProducer {
   public:
      explicit LHECOMWeightProducer(const edm::ParameterSet&);
      ~LHECOMWeightProducer();

   private:
      virtual void beginJob() override;
      virtual void beginRun(edm::Run const& run, const edm::EventSetup &es) override;
      virtual void produce(edm::Event&, const edm::EventSetup&) override;

      edm::InputTag lheTag_;
      int _pdfset;
      int _pdfmember;
      double _origECMS;
      double _newECMS;
      std::string _label;
};

#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"
namespace LHAPDF {
      void initPDFSet(int nset, int setid, int member=0);
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
LHECOMWeightProducer::LHECOMWeightProducer(const edm::ParameterSet& pset) :
 lheTag_(pset.getParameter<edm::InputTag> ("lheSrc")),
 _newECMS(pset.getParameter< double > ("NewECMS"))
{
  std::stringstream labelStr;
  labelStr << "com" << "To" << _newECMS;
  _label = labelStr.str();
  produces<GenEventInfoProduct>(_label);
} 

/////////////////////////////////////////////////////////////////////////////////////
LHECOMWeightProducer::~LHECOMWeightProducer(){}

/////////////////////////////////////////////////////////////////////////////////////
void LHECOMWeightProducer::beginRun(edm::Run const& run, const edm::EventSetup &es){
  using namespace edm;
  Handle<LHERunInfoProduct> lheRun;
  run.getByLabel(lheTag_, lheRun);
  //assumes the same pdf is used for both beams
  _pdfset    = lheRun->heprup().PDFSUP.first;
  _pdfmember = lheRun->heprup().PDFGUP.first;
  _origECMS  = lheRun->heprup().EBMUP.first + lheRun->heprup().EBMUP.second;
  edm::LogInfo("LHECOMWeightProducer") << "PDFSET: " << _pdfset << "; member: " << _pdfmember << "; COM energy: " << _origECMS;
  if ( _newECMS > _origECMS )
      throw cms::Exception("LHECOMWeightProducer") << "You cannot reweight COM energy to a higher than original energy ";
  LHAPDF::initPDFSet(1,_pdfset, _pdfmember);
}


/////////////////////////////////////////////////////////////////////////////////////
void LHECOMWeightProducer::beginJob() {
  //LHAPDF::initPDFSet(1,pdfSetName_);
}

/////////////////////////////////////////////////////////////////////////////////////
void LHECOMWeightProducer::produce(edm::Event& iEvent, const edm::EventSetup&) {

      using namespace std;
      bool verbose = false;

      if (iEvent.isRealData()) return;

      edm::Handle<LHEEventProduct> lheevent;
      iEvent.getByLabel(lheTag_, lheevent);

      float Q = lheevent->hepeup().SCALUP;

      int id1        = lheevent->hepeup().IDUP[0];
      double x1      = fabs(lheevent->hepeup().PUP[0][2]/(_origECMS/2));
      double x1prime = fabs(lheevent->hepeup().PUP[0][2]/(_newECMS/2));

      int id2        = lheevent->hepeup().IDUP[1];
      double x2      = fabs(lheevent->hepeup().PUP[1][2]/(_origECMS/2));
      double x2prime = fabs(lheevent->hepeup().PUP[1][2]/(_newECMS/2));

      LogTrace("LHECOMWeightProducer") << "*******LHECOMWeightProducer*******\n" << 
                                          " Q  : " << Q << "\n" <<
                                          " id1: " << id1 << "\n" <<
                                          " x1 : " << x1  << "\n" <<
                                          " x1': " << x1prime << "\n" <<
                                          " id2: " << id2 << "\n" <<
                                          " x2 : " << x2  << "\n" <<
                                          " x2': " << x2prime ;
      //gluon is 0 in the LHAPDF numbering
      if (id1 == 21)
        id1 = 0;
      if (id2 == 21)
        id2 = 0;

      // Put PDF weights in the event
      if (verbose)
        cout << " Set : " << _pdfset << "  member : " << _pdfmember << endl;
     
      LHAPDF::usePDFMember(1,_pdfmember);
      double oldpdf1 = LHAPDF::xfx(1, x1, Q, id1)/x1;
      double oldpdf2 = LHAPDF::xfx(1, x2, Q, id2)/x2;
      double newpdf1 = LHAPDF::xfx(1, x1prime, Q, id1)/x1prime;
      double newpdf2 = LHAPDF::xfx(1, x2prime, Q, id2)/x2prime;
      LogTrace("LHECOMWeightProducer") <<
          "     xfx1 : " << oldpdf1 << "\n" <<
          "     xfx2 : " << oldpdf2 << "\n" <<
          "     xfx1': " << newpdf1 << "\n" <<
          "     xfx2': " << newpdf2 << "\n" <<
          "     weight:" << (newpdf1/oldpdf1)*(newpdf2/oldpdf2);
      double weight = (newpdf1/oldpdf1)*(newpdf2/oldpdf2);
      std::vector<double> weights;
      weights.push_back(weight);
      std::auto_ptr<GenEventInfoProduct> info(new GenEventInfoProduct());
      info->setWeights(weights);
      iEvent.put(info, _label);
}

DEFINE_FWK_MODULE(LHECOMWeightProducer);
