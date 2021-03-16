#include <vector>
#include <ap_int.h>
#include <ap_fixed.h>
#include <TVector2.h>

#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/L1TParticleFlow/interface/PFCandidate.h"
#include "DataFormats/L1Trigger/interface/EtSum.h"
#include "DataFormats/Math/interface/LorentzVector.h"

using namespace l1t;

class L1MetPfProducer : public edm::global::EDProducer<> {
public:
  explicit L1MetPfProducer(const edm::ParameterSet&);
  ~L1MetPfProducer() override;

private:
  void produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const override;
  edm::EDGetTokenT<vector<l1t::PFCandidate>> _l1PFToken;

  long unsigned int maxCands_=128;
    const float maxPt_ = (1<<14);
    const float inverseMax_ = (1<<12); // pt bits - drop bits
    const float pt_lsb = 0.25; // GeV
    const float phi_lsb = M_PI/720; // rad

    // LUT sizes for trig functions
    static const int PROJ_TAB_SIZE = (1<<8);
    static const int DROP_BITS = 2;
    static const int ATAN_TAB_SIZE = (1<<7);

    inline double quantize(double value, double to_integer) const {
        return round(value * to_integer) / to_integer;
    };
    inline double bound(double value, double min, double max) const {
        return value > max ? max : (value < min ? min : value);
    };    
};


L1MetPfProducer::L1MetPfProducer(const edm::ParameterSet& cfg)
    : _l1PFToken(consumes<std::vector<l1t::PFCandidate>>(cfg.getParameter<edm::InputTag>("L1PFObjects"))) {
    produces<std::vector<l1t::EtSum> >();
    
}

void L1MetPfProducer::produce(edm::StreamID, 
                              edm::Event& iEvent, 
                              const edm::EventSetup& iSetup) const {

  edm::Handle<l1t::PFCandidateCollection> l1PFCandidates;
  iEvent.getByToken(_l1PFToken, l1PFCandidates);

  // to cleanup

  float sumx = 0;
  float sumy = 0;

  for(long unsigned int i=0; i<l1PFCandidates->size() && i<maxCands_; i++){
      const auto& l1PFCand = l1PFCandidates->at(i);

      // quantize inputs
      
      float pt = quantize(l1PFCand.pt(), 1./pt_lsb);
      if (pt > maxPt_) pt = maxPt_;

      float phi = quantize( TVector2::Phi_mpi_pi(l1PFCand.phi()), 1./phi_lsb);

      // LUTs correspond to Q1 only to allow a smaller number of entries
      float table_phi = quantize(phi, PROJ_TAB_SIZE/(M_PI/2));

      float px = quantize( pt * cos(table_phi), 1./pt_lsb);
      float py = quantize( pt * sin(table_phi), 1./pt_lsb);

      sumx = quantize(sumx + px, 1./pt_lsb);
      sumy = quantize(sumy + py, 1./pt_lsb);
      if( fabs(sumx) > maxPt_ ) sumx = maxPt_ * (sumx>0 ? 1 : -1);
      if( fabs(sumy) > maxPt_ ) sumy = maxPt_ * (sumy>0 ? 1 : -1);
  }

  float met = quantize(pow(sumx,2) + pow(sumy,2), 1./(pt_lsb*pt_lsb));
  met = quantize(sqrt(met), 1./pt_lsb); // stand-in for HLS::sqrt function
  if ( met > maxPt_ ) met = maxPt_;

  // to calculate arctan for evaluation in [0,pi/4), need two more LUTs for atan(py/px)
  float numerator = min(fabs(sumx), fabs(sumy));
  float denominator = max(fabs(sumx), fabs(sumy));
  if (denominator > inverseMax_) denominator = inverseMax_;
  // no add'l quantization needed, since we currently store inverse for all pt values below max
  // ratio is stored in units of 1/max pt val 
  float met_phi = atan( quantize(numerator / denominator, ATAN_TAB_SIZE) );
  
  // recover the full angle from [0,pi/4)
  if( fabs(sumx) < fabs(sumy) ) met_phi = M_PI/2 - met_phi; // to [0,pi/2)
  if ( sumx < 0 ) met_phi = M_PI - met_phi;  // to [0,pi)
  if ( sumy < 0 ) met_phi = - met_phi;  // to [-pi,pi)
  met_phi = quantize(met_phi, 1./phi_lsb);


  reco::Candidate::PolarLorentzVector metVector;
  metVector.SetPt( met );
  metVector.SetPhi( met_phi );
  metVector.SetEta(0);
  l1t::EtSum theMET(metVector, l1t::EtSum::EtSumType::kTotalHt, 0, 0, 0, 0);

  std::cout << "metVector: " << metVector.pt() << std::endl;

  std::unique_ptr<std::vector<l1t::EtSum> > metCollection(new std::vector<l1t::EtSum>(0));
  metCollection->push_back(theMET);
  iEvent.put(std::move(metCollection)); //, "myMetCollection");

}

L1MetPfProducer::~L1MetPfProducer() {}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(L1MetPfProducer);
