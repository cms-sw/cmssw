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

class L1MetPfProducerNewTypes : public edm::global::EDProducer<> {
public:
  explicit L1MetPfProducerNewTypes(const edm::ParameterSet&);
  ~L1MetPfProducerNewTypes() override;

private:
  void produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const override;
  edm::EDGetTokenT<vector<l1t::PFCandidate>> _l1PFToken;

    const long unsigned int maxCands_=128;

    // quantization controllers
    typedef ap_ufixed<14,12, AP_RND, AP_WRAP> pt_t; // LSB is 0.25 and max is 4 TeV
    typedef ap_int<12> phi_t; // LSB is pi/720 ~ 0.0044 and max is +/-8.9
    const float ptLSB_ = 0.25; // GeV
    const float phiLSB_ = M_PI/720; // rad

    // derived, helper types
    typedef ap_fixed<pt_t::width+1,pt_t::iwidth+1, AP_RND, AP_SAT> pxy_t;
    typedef ap_fixed<2*pt_t::width,2*pt_t::iwidth, AP_RND, AP_SAT> pt2_t;
    // derived, helper constants    
    const float maxPt_ = ((1<<pt_t::width)-1)*ptLSB_;
    const phi_t hwPi_ = round(M_PI/phiLSB_);
    const phi_t hwPiOverTwo_ = round(M_PI/(2*phiLSB_));

    typedef ap_ufixed<pt_t::width,0> inv_t; // can't easily use the MAXPT/pt trick with ap_fixed

    // to make configurable...
    const int dropBits_=2;
    const int dropFactor_=(1<<dropBits_);
    const int invTableBits_=10;
    const int invTableSize_=(1<<invTableBits_);


    // static constexpr uint PT_SIZE = 16;
    // static constexpr uint PT2_SIZE = 2*PT_SIZE;
    // static constexpr uint PT_DEC_BITS = 2;
    // static constexpr uint PHI_SIZE = 10;
    // typedef ap_uint<PT_SIZE> pt_t;
    // typedef ap_int<PT_SIZE+1> pxy_t;
    // typedef ap_uint<PT2_SIZE> pt2_t;
    // typedef ap_int<PHI_SIZE> phi_t;
    // const float pt_lsb = 1./(1<<PT_DEC_BITS); // GeV
    // const float phi_lsb = 2*M_PI/(1<<PHI_SIZE); // rad

    // for LUTs
    // static constexpr int PROJ_TAB_SIZE = (1<<(PHI_SIZE-2));
    // static constexpr int DROP_BITS = 2;
    // static constexpr int INV_TAB_SIZE = (1<<(PT_SIZE-DROP_BITS));
    // static constexpr int ATAN_SIZE = (PHI_SIZE-3);
    // static constexpr int ATAN_TAB_SIZE = (1<<ATAN_SIZE);

    // std::vector<pt_t> cos_table;
    // std::vector<pt_t> sin_table;
    // std::vector<pt_t> inv_table;
    // std::vector<pt_t> atan_table;
    // uint sinCosTableBits_;
    // uint sinCosTableSize_;
    // uint inverseDropBits_;
    // uint inversionTableSize_;
    // uint arcTanTableBits_;
    // uint arcTanTableSize_;
    
    // void init_projx_table(std::vector<pt_t> table_out);
    // void init_projy_table(std::vector<pt_t> table_out);
    // void init_inv_table(std::vector<pt_t> table_out);
    // void init_atan_table(std::vector<pt_t> table_out);

    void Project(pt_t pt, phi_t phi, pxy_t &pxy, bool isX, bool debug=false) const;
    void PhiFromXY(pxy_t px, pxy_t py, phi_t &phi) const;

    void CalcMetHLS(std::vector<float> pt, std::vector<float> phi,
                    reco::Candidate::PolarLorentzVector &metVector) const;
    
    // void CalcMetFast(std::vector<float> pt, std::vector<float> phi,
    //                 reco::Candidate::PolarLorentzVector &metVector) const;
    
    // bool useHLS=true;

    // // for alternate floating-pt implementation
    // const float maxPt_ = (1<<(PT_SIZE-PT_DEC_BITS));
    // const float inverseMax_ = (1<<(PT_SIZE-inverseDropBits_));
    
    // inline double quantize(double value, double to_integer) const {
    //     return round(value * to_integer) / to_integer;
    // };
    // inline double bound(double value, double min, double max) const {
    //     return value > max ? max : (value < min ? min : value);
    // };    

};

L1MetPfProducerNewTypes::L1MetPfProducerNewTypes(const edm::ParameterSet& cfg)
    : _l1PFToken(consumes<std::vector<l1t::PFCandidate>>(cfg.getParameter<edm::InputTag>("L1PFObjects"))){
      // sinCosTableBits_(cfg.getParameter<uint>("sinCosTableBits")),
      // inverseDropBits_(cfg.getParameter<uint>("inverseDropBits")),
      // arcTanTableBits_(cfg.getParameter<uint>("arcTanTableBits")){

    produces<std::vector<l1t::EtSum> >();

    // sinCosTableSize_ = 1<<sinCosTableBits_;
    // cos_table.resize(sinCosTableSize_);
    // sin_table.resize(sinCosTableSize_);

    // inversionTableSize_ = (1<<(PT_SIZE-inverseDropBits_));
    // inv_table.resize(inversionTableSize_);

    // arcTanTableSize_ = 1<<arcTanTableBits_;
    // atan_table.resize(arcTanTableSize_);

    // init_projx_table( cos_table );
    // init_projy_table( sin_table );
    // init_inv_table( inv_table );
    // init_atan_table( atan_table );
}

void L1MetPfProducerNewTypes::produce(edm::StreamID, 
                              edm::Event& iEvent, 
                              const edm::EventSetup& iSetup) const {

  edm::Handle<l1t::PFCandidateCollection> l1PFCandidates;
  iEvent.getByToken(_l1PFToken, l1PFCandidates);

  std::vector<float> pt;
  std::vector<float> phi;

  for(long unsigned int i=0; i<l1PFCandidates->size() && i<maxCands_; i++){
      const auto& l1PFCand = l1PFCandidates->at(i);
      pt.push_back( l1PFCand.pt() );
      phi.push_back( l1PFCand.phi() );
  }

  reco::Candidate::PolarLorentzVector metVector;

  CalcMetHLS(pt, phi, metVector);
      
  l1t::EtSum theMET(metVector, l1t::EtSum::EtSumType::kTotalHt, 0, 0, 0, 0);

  std::unique_ptr<std::vector<l1t::EtSum> > metCollection(new std::vector<l1t::EtSum>(0));
  metCollection->push_back(theMET);
  iEvent.put(std::move(metCollection));
}

void L1MetPfProducerNewTypes::CalcMetHLS(std::vector<float> pt, std::vector<float> phi,
                                    reco::Candidate::PolarLorentzVector &metVector) const{
    pxy_t hw_px = 0;
    pxy_t hw_py = 0;
    pxy_t hw_sumx = 0;
    pxy_t hw_sumy = 0;

    for(uint i=0; i< pt.size(); i++){
        pt_t hw_pt = min(pt[i], maxPt_);
        phi_t hw_phi = TVector2::Phi_mpi_pi( phi[i] ) / phiLSB_;
      
        Project(hw_pt, hw_phi, hw_px, true);
        Project(hw_pt, hw_phi, hw_py, false);

        hw_sumx = hw_sumx - hw_px;
        hw_sumy = hw_sumy - hw_py;
        
        // printf("part %d  (pt,phi) = float(%f, %f) hw(%f, %f = %f); (x,y) = float(%f, %f) hw(%f, %f); sumHW(%f, %f)\n",
        //        i, pt[i], TVector2::Phi_mpi_pi( phi[i] ), hw_pt.to_double(), hw_phi.to_double(), hw_phi.to_double()*phiLSB_,
        //        pt[i]*cos(phi[i]), pt[i]*sin(phi[i]), hw_px.to_double(), hw_py.to_double(),
        //        hw_sumx.to_double(), hw_sumy.to_double()
        //        );
    }
    // std::cout << std::endl;

    pt2_t hw_met = pt2_t(hw_sumx)*pt2_t(hw_sumx) + pt2_t(hw_sumy)*pt2_t(hw_sumy);
    hw_met = sqrt( int(hw_met) ); // stand-in for HLS::sqrt

    phi_t hw_met_phi = 0;
    PhiFromXY(hw_sumx,hw_sumy,hw_met_phi);

    metVector.SetPt( hw_met.to_double() );
    metVector.SetPhi( hw_met_phi.to_double() * phiLSB_ );
    //(M_PI/hwPi_.to_double())
    // printf("    phi hw, float, real = %f, %f    (%f rad from x,y = %f, %f) \n", 
    //        phi.to_double(), phi.to_double() * (M_PI/hwPi_.to_double()), atan2(py.to_double(),px.to_double()), px.to_double(), py.to_double());


    metVector.SetEta(0);
}



// Convert pt and phi to px (py)
// 1) Map phi to the first quadrant to reduce LUT size
// 2) Lookup sin(phiQ1), where the result is in [0,maxPt]
//   which is used to encode [0,1].
// 3) Multiply pt by sin(phiQ1) to get px. Result will be px*maxPt, but
// wrapping multiplication is 'mod maxPt' so the correct value is returned.
// 4) Check px=-|px|.


void L1MetPfProducerNewTypes::Project(pt_t pt, phi_t phi, pxy_t &pxy, bool isX, bool debug) const{
    // set phi to first quadrant
    phi_t phiQ1 = (phi>0) ? phi : phi_t(-phi); // Q1/Q4
    if(phiQ1 >= hwPiOverTwo_) phiQ1 = hwPi_ - phiQ1;

    if (phiQ1 > hwPiOverTwo_){
        std::cout << "unexpected phi (high)" << std::endl;
        phiQ1 = hwPiOverTwo_;
    } else if (phiQ1<0){
        std::cout << "unexpected phi (low)" << std::endl;
        phiQ1 = 0;
    }
    if(isX){
        typedef ap_ufixed<14,12, AP_RND, AP_WRAP> pt_t; // LSB is 0.25 and max is 4 TeV
        ap_ufixed<pt_t::width,0> cosPhi = cos(phiQ1.to_double() / hwPiOverTwo_.to_double() * M_PI/2);
        // pt_t cosPhi = cos(phiQ1.to_double() / hwPiOverTwo_.to_double()) * maxPt_; // assume fixed tab size (pi/2=360)
        pxy = pt * cosPhi;
        //printf("  phi = %f, phiQ1 = %f, cosPhi = %f, pxy = %f \n", phi.to_double(), phiQ1.to_double(), cosPhi.to_double(), pxy.to_double() );
        if (phi > hwPiOverTwo_ || phi < -hwPiOverTwo_) pxy = -pxy;
    } else {
        ap_ufixed<pt_t::width,0> sinPhi = sin(phiQ1.to_double() / hwPiOverTwo_.to_double() * M_PI/2);
        // pt_t sinPhi = sin(phiQ1.to_double() / hwPiOverTwo_.to_double()) * maxPt_; // assume fixed tab size (pi/2=360)
        pxy = pt * sinPhi;
        if (phi < 0) pxy = -pxy;
    }
    
    return;
}

// void L1MetPfProducerNewTypes::init_inv_table(std::vector<pt_t> table_out) {
//     // multiply result by 1=2^(PT-SIZE)
//     table_out[0]=(1<<PT_SIZE)-1;
//     for (uint i = 1; i < inversionTableSize_; i++) {
//         table_out[i] = round((1<<PT_SIZE) / float(i));
//     }
//     return;
// }
// void L1MetPfProducerNewTypes::init_atan_table(std::vector<pt_t> table_out) {
//     // multiply result by 1=2^(PT-SIZE)
//     table_out[0]=int(0);
//     for (uint i = 1; i < arcTanTableSize_; i++) {
//         table_out[i] = int(round(atan(float(i)/arcTanTableSize_) * (1<<(PHI_SIZE-3)) / (M_PI/4)));
//     }
//     return;
// }

void L1MetPfProducerNewTypes::PhiFromXY(pxy_t px, pxy_t py, phi_t &phi) const{
    if(px==0 && py==0){ phi = 0; return; }
    if(px==0){ phi = py>0 ? hwPiOverTwo_ : phi_t(-hwPiOverTwo_); return; }
    if(py==0){ phi = px>0 ? phi_t(0) : phi_t(-hwPi_); return; }

    // get q1 coordinates                                                                              
    pt_t x =  px>0 ? pt_t(px) : pt_t(-px); //px>=0 ? px : -px;                                                                  
    pt_t y =  py>0 ? pt_t(py) : pt_t(-py); //px>=0 ? px : -px;                                                                  
    // transform so a<b                                                                                
    pt_t a = x<y ? x : y;
    pt_t b = x<y ? y : x;


    if (b.to_double() > maxPt_ / dropFactor_) b = maxPt_ / dropFactor_;
    // map [0,max/4) to inv table size
    int index = round((b.to_double()/(maxPt_/dropFactor_)) * invTableSize_);
    float bcheck = (float(index) / invTableSize_) * (maxPt_/dropFactor_);
    inv_t inv_b = 1./((float(index) / invTableSize_) * (maxPt_/dropFactor_));

    inv_t a_over_b = a * inv_b;
    
    printf("  a, b = %f, %f;   index, inv = %d, %f; ratio = %f \n", a.to_double(), b.to_double(),index, inv_b.to_double(), a_over_b.to_double() );
    printf("    bcheck, 1/bc = %f, %f  -- %d  %f  %d  \n", bcheck, 1./bcheck, invTableSize_ , maxPt_, dropFactor_ );

    int atanTableBits_=7;
    int atanTableSize_=(1<<atanTableBits_);
    index = round(a_over_b.to_double() * atanTableSize_);
    phi = atan(float(index) / atanTableSize_) / phiLSB_;

    printf("    atan index, phi = %d, %f (%f rad)  real atan(a/b)= %f  \n", index, phi.to_double(), 
           phi.to_double() * (M_PI/hwPi_.to_double()),  atan(a.to_double()/b.to_double()));

    // rotate from (0,pi/4) to full quad1                                                              
    if(y>x) phi = hwPiOverTwo_ - phi; //phi = pi/2 - phi                                          
    // other quadrants                                                                                 
    if( px < 0 && py > 0 ) phi = hwPi_ - phi;    // Q2 phi = pi - phi                      
    if( px > 0 && py < 0 ) phi = -phi;                       // Q4 phi = -phi                          
    if( px < 0 && py < 0 ) phi = -(hwPi_ - phi); // Q3 composition of both                 

    // if(y>x)  phi = hwPiOverTwo_ - phi;
    // if(px<0) phi = hwPi_ - phi;
    // if(py<0) phi = - phi;
    printf("    phi hw, float, real = %f, %f    (%f rad from x,y = %f, %f) \n", 
           phi.to_double(), phi.to_double() * (M_PI/hwPi_.to_double()), atan2(py.to_double(),px.to_double()), px.to_double(), py.to_double());

    return;
}

L1MetPfProducerNewTypes::~L1MetPfProducerNewTypes() {}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(L1MetPfProducerNewTypes);
