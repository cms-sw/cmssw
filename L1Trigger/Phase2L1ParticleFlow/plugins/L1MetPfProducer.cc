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
    
    long unsigned int maxCands=128;

    // quantization controllers
    static constexpr uint PT_SIZE = 16;
    static constexpr uint PT2_SIZE = 2*PT_SIZE;
    static constexpr uint PT_DEC_BITS = 2;
    static constexpr uint PHI_SIZE = 10;
    typedef ap_uint<PT_SIZE> pt_t;
    typedef ap_int<PT_SIZE+1> pxy_t;
    typedef ap_uint<PT2_SIZE> pt2_t;
    typedef ap_int<PHI_SIZE> phi_t;
    const float pt_lsb = 1./(1<<PT_DEC_BITS); // GeV
    const float phi_lsb = 2*M_PI/(1<<PHI_SIZE); // rad

    // for LUTs
    static constexpr int PROJ_TAB_SIZE = (1<<(PHI_SIZE-2));
    static constexpr int DROP_BITS = 2;
    static constexpr int INV_TAB_SIZE = (1<<(PT_SIZE-DROP_BITS));
    static constexpr int ATAN_SIZE = (PHI_SIZE-3);
    static constexpr int ATAN_TAB_SIZE = (1<<ATAN_SIZE);

    //static pt_t cos_table[PROJ_TAB_SIZE];
    pt_t cos_table[PROJ_TAB_SIZE];
    pt_t sin_table[PROJ_TAB_SIZE];
    pt_t inv_table[INV_TAB_SIZE];
    pt_t atan_table[ATAN_TAB_SIZE];
    void init_projx_table(pt_t table_out[PROJ_TAB_SIZE]);
    void init_projy_table(pt_t table_out[PROJ_TAB_SIZE]);
    void init_inv_table(pt_t table_out[INV_TAB_SIZE]);
    void init_atan_table(pt_t table_out[ATAN_TAB_SIZE]);

    void ProjX(pt_t pt, phi_t phi, pxy_t &x, bool debug=false) const;
    void ProjY(pt_t pt, phi_t phi, pxy_t &y, bool debug=false) const;
    void PhiFromXY(pxy_t px, pxy_t py, phi_t &phi) const;

    void CalcMetHLS(std::vector<float> pt, std::vector<float> phi,
                    reco::Candidate::PolarLorentzVector &metVector) const;
    
    void CalcMetFast(std::vector<float> pt, std::vector<float> phi,
                    reco::Candidate::PolarLorentzVector &metVector) const;
    
    bool useHLS=true;

    // for alternate floating-pt implementation
    const float maxPt_ = (1<<(PT_SIZE-PT_DEC_BITS));
    const float inverseMax_ = (1<<(PT_SIZE-DROP_BITS));
    
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
    
    init_projx_table( cos_table );
    init_projy_table( sin_table );
    init_inv_table( inv_table );
    init_atan_table( atan_table );
}

void L1MetPfProducer::produce(edm::StreamID, 
                              edm::Event& iEvent, 
                              const edm::EventSetup& iSetup) const {

  edm::Handle<l1t::PFCandidateCollection> l1PFCandidates;
  iEvent.getByToken(_l1PFToken, l1PFCandidates);

  std::vector<float> pt;
  std::vector<float> phi;

  for(long unsigned int i=0; i<l1PFCandidates->size() && i<maxCands; i++){
      const auto& l1PFCand = l1PFCandidates->at(i);
      pt.push_back( l1PFCand.pt() );
      phi.push_back( l1PFCand.phi() );
  }

  reco::Candidate::PolarLorentzVector metVector;

  if(useHLS) CalcMetHLS(pt, phi, metVector);
  else CalcMetFast(pt, phi, metVector);
      
  l1t::EtSum theMET(metVector, l1t::EtSum::EtSumType::kTotalHt, 0, 0, 0, 0);

  std::unique_ptr<std::vector<l1t::EtSum> > metCollection(new std::vector<l1t::EtSum>(0));
  metCollection->push_back(theMET);
  iEvent.put(std::move(metCollection));
}

void L1MetPfProducer::CalcMetHLS(std::vector<float> pt, std::vector<float> phi,
                                    reco::Candidate::PolarLorentzVector &metVector) const{
    pxy_t hw_px = 0;
    pxy_t hw_py = 0;
    pxy_t hw_sumx = 0;
    pxy_t hw_sumy = 0;

    for(uint i=0; i< pt.size(); i++){
        pt_t hw_pt = pt[i] / pt_lsb;
        phi_t hw_phi = TVector2::Phi_mpi_pi( phi[i] ) / phi_lsb;
      
        ProjX(hw_pt, hw_phi, hw_px);
        ProjY(hw_pt, hw_phi, hw_py);

        hw_sumx = hw_sumx + hw_px;
        hw_sumy = hw_sumy + hw_py;
    }

    pt2_t hw_met = pt2_t(hw_sumx)*pt2_t(hw_sumx) + pt2_t(hw_sumy)*pt2_t(hw_sumy);
    hw_met = sqrt( int(hw_met) ); // stand-in for HLS::sqrt

    phi_t hw_met_phi = 0;
    PhiFromXY(hw_sumx,hw_sumy,hw_met_phi);

    metVector.SetPt( hw_met * pt_lsb );
    metVector.SetPhi( hw_met_phi * phi_lsb );
    metVector.SetEta(0);
}


void L1MetPfProducer::CalcMetFast(std::vector<float> pt_vec, std::vector<float> phi_vec,
                                    reco::Candidate::PolarLorentzVector &metVector) const{
    float sumx = 0;
    float sumy = 0;
    
    for(uint i=0; i< pt_vec.size(); i++){      
        float pt = quantize( pt_vec[i], 1./pt_lsb);
        if (pt > maxPt_) pt = maxPt_;

        float phi = quantize( TVector2::Phi_mpi_pi(phi_vec[i]), 1./phi_lsb);

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

    metVector.SetPt( met );
    metVector.SetPhi( met_phi );
    metVector.SetEta(0);
}

void L1MetPfProducer::init_projx_table(pt_t table_out[PROJ_TAB_SIZE]) {
    // Return table of cos(phi) where phi is in (0,pi/2)
    // multiply result by 2^(PT_SIZE) (equal to 1 in our units)
    for (int i = 0; i < PROJ_TAB_SIZE; i++) {
        //store result, guarding overflow near costheta=1      
        pt2_t x = round((1<<PT_SIZE) * cos(float(i)/PROJ_TAB_SIZE * M_PI/2));
        // (using extra precision here (pt2_t, not pt_t) to check the out of bounds condition)
        if(x >= (1<<PT_SIZE)) table_out[i] = (1<<PT_SIZE)-1;
        else table_out[i] = x;
    }
    return;
}

void L1MetPfProducer::ProjX(pt_t pt, phi_t phi, pxy_t &x, bool debug) const{
    //map phi to first quadrant value: range [0, 2^(PHI_SIZE-2))
    ap_uint<PHI_SIZE-2> phiQ1 = phi;
    if(phi>=(1<<(PHI_SIZE-2))) phiQ1 = (1<<(PHI_SIZE-2)) -1 - phiQ1; // map 64-128 (0-63) to 63-0
    if(phi<0 && phi>=-(1<<(PHI_SIZE-2))) phiQ1 = (1<<(PHI_SIZE-2)) -1 - phiQ1; // map -64-1 (0-63) to 63-0

    // get x component and flip sign if necessary
    x = (pt * cos_table[phiQ1]) >> PT_SIZE;
    if(debug) std::cout << pt << "  cos_table[" << phiQ1 << "] = " << cos_table[phiQ1] << "  " << x << std::endl;
    if( phi>=(1<<(PHI_SIZE-2))
        || phi<-(1<<(PHI_SIZE-2)))
        x = -x;

    return;
}

void L1MetPfProducer::init_projy_table(pt_t table_out[PROJ_TAB_SIZE]) {
    for (int i = 0; i < PROJ_TAB_SIZE; i++) {
        pt2_t x = round((1<<PT_SIZE) * sin(float(i)/PROJ_TAB_SIZE * M_PI/2));
        if(x >= (1<<PT_SIZE)) table_out[i] = (1<<PT_SIZE)-1;
        else table_out[i] = x;
    }
    return;
}

void L1MetPfProducer::ProjY(pt_t pt, phi_t phi, pxy_t &y, bool debug) const{

    //map phi to first quadrant value: range [0, 2^(PHI_SIZE-2))
    ap_uint<PHI_SIZE-2> phiQ1 = phi;
    if(phi>=(1<<(PHI_SIZE-2))) phiQ1 = (1<<(PHI_SIZE-2)) -1 - phiQ1; // map 64-128 (0-63) to 63-0
    if(phi<0 && phi>=-(1<<(PHI_SIZE-2))) phiQ1 = (1<<(PHI_SIZE-2)) -1 - phiQ1; // map -64-1 (0-63) to 63-0

    // get y component and flip sign if necessary
    y = (pt * sin_table[phiQ1]) >> PT_SIZE;
    if(debug) std::cout << pt << "  sin_table[" << phiQ1 << "] = " << sin_table[phiQ1] << "  " << y << std::endl;
    if( phi<0 ) y = -y;

    return;
}

void L1MetPfProducer::init_inv_table(pt_t table_out[INV_TAB_SIZE]) {
    // multiply result by 1=2^(PT-SIZE)
    table_out[0]=(1<<PT_SIZE)-1;
    for (int i = 1; i < INV_TAB_SIZE; i++) {
        table_out[i] = round((1<<PT_SIZE) / float(i));
    }
    return;
}
void L1MetPfProducer::init_atan_table(pt_t table_out[ATAN_TAB_SIZE]) {
    // multiply result by 1=2^(PT-SIZE)
    table_out[0]=int(0);
    for (int i = 1; i < ATAN_TAB_SIZE; i++) {
        table_out[i] = int(round(atan(float(i)/ATAN_TAB_SIZE) * (1<<(PHI_SIZE-3)) / (M_PI/4)));
    }
    return;
}

void L1MetPfProducer::PhiFromXY(pxy_t px, pxy_t py, phi_t &phi) const{
    if(px==0 && py==0){ phi = 0; return; }

    // get q1 coordinates                                                                              
    pt_t x =  px; //px>=0 ? px : -px;                                                                  
    pt_t y =  py; //py>=0 ? py : -py;                                                                  
    if(px<0) x = -px;
    if(py<0) y = -py;
    // transform so a<b                                                                                
    pt_t a = x; //x<y ? x : y;                                                                         
    pt_t b = y; //x<y ? y : x;                                                                         
    if(a>b){ a = y; b = x; }

    pt_t inv_b;
    if(b>= (1<<(PT_SIZE-DROP_BITS))) inv_b = 1;
    // don't bother to store these large numbers in the LUT...                                         
    // instead approximate their inverses as 1                                                         
    else inv_b = inv_table[b];

    pt_t a_over_b = a * inv_b; // x 2^(PT_SIZE-DROP_BITS)                                              
    ap_uint<ATAN_SIZE> atan_index = a_over_b >> (PT_SIZE-ATAN_SIZE); // keep only most significant bits
    phi = atan_table[atan_index];

    // rotate from (0,pi/4) to full quad1                                                              
    if(y>x) phi = (1<<(PHI_SIZE-2)) - phi; //phi = pi/2 - phi                                          
    // other quadrants                                                                                 
    if( px < 0 && py > 0 ) phi = (1<<(PHI_SIZE-1)) - phi;    // Q2 phi = pi - phi                      
    if( px > 0 && py < 0 ) phi = -phi;                       // Q4 phi = -phi                          
    if( px < 0 && py < 0 ) phi = -((1<<(PHI_SIZE-1)) - phi); // Q3 composition of both                 

    return;
}

L1MetPfProducer::~L1MetPfProducer() {}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(L1MetPfProducer);
