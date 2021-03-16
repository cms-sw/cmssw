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

// sizes in bits
typedef ap_ufixed<14, 12, AP_RND, AP_SAT> pt_t;
typedef ap_int<11> phi_t;
constexpr float LSB_PHI = M_PI/720;

// helpers
typedef ap_fixed<pt_t::width+1, pt_t::iwidth+1, AP_RND, AP_SAT> pxy_t;
typedef ap_ufixed<2*pt_t::width, 2*pt_t::iwidth, AP_RND, AP_SAT> pt2_t;
typedef ap_ufixed<pt_t::width, pt_t::iwidth, pt_t::qmode, AP_WRAP> pt_wrap_t;

// LUT sizes for trig functions
constexpr uint PROJ_TAB_BITS = 8; // configurable
constexpr uint PROJ_TAB_SIZE = (1<<PROJ_TAB_BITS);
constexpr uint INV_TAB_BITS = 10;
constexpr uint INV_TAB_SIZE = (1<<INV_TAB_BITS);

// constexpr uint PT_SIZE = 16;
// constexpr uint PHI_SIZE = 10;
// constexpr uint PT2_SIZE = 2*PT_SIZE;
//constexpr uint PT_DEC_BITS = 2;
// typedef ap_uint<PT_SIZE> pt_t;
// typedef ap_int<PT_SIZE+1> pxy_t;
// typedef ap_uint<PT2_SIZE> pt2_t;
// typedef ap_int<PHI_SIZE> phi_t;


// typedef ap_ufixed<16,2> pt_t;
// typedef ap_fixed<16,2> pxy_t;
// typedef ap_int<10> phi_t;
// ...

class L1MetPfProducer : public edm::global::EDProducer<> {
public:
  explicit L1MetPfProducer(const edm::ParameterSet&);
  ~L1MetPfProducer() override;

private:
  void produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const override;
  edm::EDGetTokenT<vector<l1t::PFCandidate>> _l1PFToken;

  // const int pi_hw = (1<<(phi_t::width-1));

  // static float px_table[PROJ_TAB_SIZE];
  // static float py_table[PROJ_TAB_SIZE];

    //void initProjXTable() const;

  pxy_t Project(const pt_t pt, const phi_t phi, const bool doX) const;
  
  // pxy_t ProjY(const pt_t pt, const phi_t phi) const;
    // phi_t PhiFromXY(const pxy_t x, const pxy_t y, const pt_t pt) const;
  phi_t PhiFromXY(const pxy_t x, const pxy_t y) const;

};

// void L1MetPfProducer::initProjXTable() const {
//     // Return table of cos(phi) where phi is in (0,pi/2)
//     // multiply result by 2^(PT_SIZE) (equal to 1 in our units)
//     for (uint i = 0; i < PROJ_TAB_SIZE; i++) {
//         //store result, guarding overflow near costheta=1
//         pt2_t x = round((1<<PT_SIZE) * cos(float(i)/PROJ_TAB_SIZE * M_PI/2));
//         // (using extra precision here (pt2_t, not pt_t) to check the out of bounds condition)
//         if(x >= (1<<PT_SIZE)) px_table[i] = (1<<PT_SIZE)-1;
//         else px_table[i] = x;
//     }
// }

L1MetPfProducer::L1MetPfProducer(const edm::ParameterSet& cfg)
    : _l1PFToken(consumes<std::vector<l1t::PFCandidate>>(cfg.getParameter<edm::InputTag>("L1PFObjects"))) {
    produces<std::vector<l1t::EtSum> >();
    
    //initProjXTable();
}


pxy_t L1MetPfProducer::Project(const pt_t pt, const phi_t phi, const bool doX) const {
    
    //map phi to the equivalent first quadrant value: range [0, 2^(phi_t::width-2)) i.e. [0,pi/2)
    ap_uint<phi_t::width-2> phiQ1 = phi;
    if(phi>=(1<<(phi_t::width-2))) phiQ1 = (1<<(phi_t::width-2)) -1 - phiQ1; // map e.g. 64-128 (0-63) to 63-0
    if(phi<0 && phi>=-(1<<(phi_t::width-2))) phiQ1 = (1<<(phi_t::width-2)) -1 - phiQ1; // map e.g. -64-1 (0-63) to 63-0

    // Lut stores cos(x) for x in [0,pi/2), steps of pi/2 * 1/PROJ_TAB_SIZE
    int index = phiQ1.to_int();
    if(phi_t::width-2 > PROJ_TAB_BITS)
        index = index >> (phi_t::width-2 - PROJ_TAB_BITS);
    if(phi_t::width-2 < PROJ_TAB_BITS){
        // silly, this table has more entries than possible values of phi!!
        index = index << (PROJ_TAB_BITS - phi_t::width-2);
    }
    float fPhi = index/PROJ_TAB_SIZE * M_PI/2;
    float fCosSin = doX ? cos(fPhi) : sin(fPhi);
    pxy_t pxy = pt_t(pt_t::width * fCosSin);
    if(doX){
        // -px for theta>pi/2, theta<-pi/2
        if( phi>=(1<<(phi_t::width-2)) || phi<-(1<<(phi_t::width-2))) pxy = -pxy;
    } else {
        // -py for theta<0
        if( phi<0 ) pxy = -pxy;
    }

    return pxy;
}

// pxy_t L1MetPfProducer::ProjY(const pt_t pt, const phi_t phi) const {
//     return 0;
// }


// phi_t L1MetPfProducer::PhiFromXY(const pxy_t px, const pxy_t py) const {
//     if(px==0 && py==0) return 0;

//     // get Q1 coordinates
//     pt_t x =  px; //px>=0 ? px : -px;
//     pt_t y =  py; //py>=0 ? py : -py;
//     if(px<0) x = -px;
//     if(py<0) y = -py;
//     // transform so a<b
//     pt_t a = x; //x<y ? x : y;
//     pt_t b = y; //x<y ? y : x;
//     if(a>b){ a = y; b = x; }

//     pt_t inv_b;
//     const uint DROP_BITS = 2;
//     if(b>= (1<<(pt_t::iwidth-DROP_BITS))) inv_b = 1; 
//     // don't bother to store these large numbers in the LUT...
//     // instead approximate their inverses as 1
//     else inv_b = inv_table[b];

//     pt_t a_over_b = a * inv_b; // x 2^(PT_SIZE-DROP_BITS)
//     ap_uint<ATAN_SIZE> atan_index = a_over_b >> (PT_SIZE-ATAN_SIZE); // keep only most significant bits
//     phi = atan_table[atan_index];

//     // rotate from (0,pi/4) to full quad1
//     if(y>x) phi = (1<<(PHI_SIZE-2)) - phi; //phi = pi/2 - phi
//     // other quadrants
//     if( px < 0 && py > 0 ) phi = (1<<(PHI_SIZE-1)) - phi;    // Q2 phi = pi - phi
//     if( px > 0 && py < 0 ) phi = -phi;                       // Q4 phi = -phi
//     if( px < 0 && py < 0 ) phi = -((1<<(PHI_SIZE-1)) - phi); // Q3 composition of both

// }

phi_t L1MetPfProducer::PhiFromXY(const pxy_t px, const pxy_t py) const {
    // // First, get px/pt
    // pt_wrap_t wPx = px>0 ? px : -px;
    // pt_wrap_t wPt = pt;
    
    // pt_wrap_t wInvPt = (1<<pt_t::iwidth) / pt.to_float(); // from LUT
    // pt_wrap_t wPxOverPt = pt_wrap_t(px) * wInvPt;

    // inversion table covers [0,2^12 TeV] with indices [0,INV_TAB_SIZE)
    // intPt = pt.

    //typedef pt_wrap_t ap_ufixed<pt_t::width, pt_t::iwidth, pt_t::qmode, AP_WRAP>;

    if(px==0 && py==0) return 0;

    // Lookup atan in relatively narrow range [0,pi/4) to avoid having a big table for no reason
    pxy_t maxXY = max( fabs(px.to_float()), fabs(py.to_float()) );
    pxy_t minXY = min( fabs(px.to_float()), fabs(py.to_float()) );

    // set large number's inverses to 1 directly instead of storing in LUT
    const uint DROP_BITS = 2;
    const int fixed_to_int = 1<< (pt_t::width - pt_t::iwidth);
    pt_wrap_t wInvMax = 1./fixed_to_int; // just set LSB
    if(maxXY < (1<<(pt_t::iwidth-DROP_BITS))){
        // emulate LUT
        int int_maxXY = round(maxXY.to_float() * fixed_to_int);
        int index_inv = int_maxXY;
        if(pt_t::width-DROP_BITS > INV_TAB_BITS) index_inv = index_inv >> ((pt_t::width-DROP_BITS) - INV_TAB_BITS);
        if(pt_t::width-DROP_BITS < INV_TAB_BITS) index_inv = index_inv << (INV_TAB_BITS - (pt_t::width-DROP_BITS));
        wInvMax = (INV_TAB_SIZE / index_inv) * pt_t::width;
    }

    pt_wrap_t wMinOverMax = minXY * wInvMax;

    // convert py/px (or vice versa) to an angle in [0,pi/4)
    phi_t phi = atan(wMinOverMax.to_float() / (1<<pt_t::iwidth)) / LSB_PHI;

    if(py>px) phi = (1<<(phi_t::width-2)) - phi; // rotate to [pi/4,pi/2)
    if( px < 0 && py > 0 ) phi = (1<<(phi_t::width-1)) - phi;    // Q2 phi = pi - phi
    if( px > 0 && py < 0 ) phi = -phi;                           // Q4 phi = -phi
    if( px < 0 && py < 0 ) phi = -((1<<(phi_t::width-1)) - phi); // Q3 composition of both

    return phi;

    /*
    // get integer representation of pt to mimic the index lookup used in the inversion table
    int to_int = 1<< (pt_t::width - pt_t::iwidth);
    int int_pt = round(pt.to_float() * to_int);
    int index_inv = int_pt;
    if(pt_t::width > INV_TAB_BITS) index_inv = index_inv >> (pt_t::width - INV_TAB_BITS);
    if(pt_t::width < INV_TAB_BITS) index_inv = index_inv << (INV_TAB_BITS - pt_t::width);
    pt_wrap_t wInvPt = round(((1<<pt_t::iwidth)-1) * float(index)/INV_TAB_SIZE);
    // get px/pt to convert to phi w/ acos
    pt_wrap_t wPxOverPt = pt_wrap_t(px) * wInvPt;
    phi_t phi = acos(wPxOverPt.to_float() / (1<<pt_t::iwidth)) / LSB_PHI;
    */

    // phi is in [0,pi/4)

    // int max_int_pt = (1<<pt_t::width);
    // int int_inv_pt = max_int_pt/int_pt;

    // uint index = x.round((1<<PT_SIZE) / float(i));

    //return 0;
}

void L1MetPfProducer::produce(edm::StreamID, 
                              edm::Event& iEvent, 
                              const edm::EventSetup& iSetup) const {

  edm::Handle<l1t::PFCandidateCollection> l1PFCandidates;
  iEvent.getByToken(_l1PFToken, l1PFCandidates);

  // to cleanup
  long unsigned int maxCands_=128;

  float x=0;
  float y=0;
  pxy_t hw_px = 0;
  pxy_t hw_py = 0;

  for(long unsigned int i=0; i<l1PFCandidates->size() && i<maxCands_; i++){
      const auto& l1PFCand = l1PFCandidates->at(i);
      
      pt_t hw_pt = l1PFCand.pt();
      phi_t hw_phi = TVector2::Phi_mpi_pi( l1PFCand.phi()) / LSB_PHI;

      hw_px += Project(hw_pt, hw_phi, /*doX*/ true);
      hw_py += Project(hw_pt, hw_phi, /*doX*/ false);
  }

  pt2_t hw_met = pt2_t(hw_px)*pt2_t(hw_px) + pt2_t(hw_py)*pt2_t(hw_py);
  hw_met = sqrt( hw_met.to_float() ); // FIXME

  phi_t hw_met_phi = 0;//PhiFromXY(hw_px, hw_py, hw_met);

  reco::Candidate::PolarLorentzVector metVector;
  metVector.SetPt( hw_met.to_float() );
  metVector.SetPhi( hw_met_phi * LSB_PHI );
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
