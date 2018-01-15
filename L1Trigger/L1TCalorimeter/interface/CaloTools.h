///
/// \class l1t::CaloTools
///
/// Description: A collection of useful functions for the Calorimeter that are of generic interest
///
/// Implementation:
///   currently implimented as a static class rather than a namespace, open to re-writing it as namespace  
///
/// \author: Sam Harper - RAL
///

//

#ifndef L1Trigger_L1TCommon_CaloTools_h
#define L1Trigger_L1TCommon_CaloTools_h

#include "DataFormats/L1TCalorimeter/interface/CaloTower.h"
#include "DataFormats/L1TCalorimeter/interface/CaloCluster.h"

#include "DataFormats/L1Trigger/interface/EGamma.h"
#include "DataFormats/L1Trigger/interface/Tau.h"
#include "DataFormats/L1Trigger/interface/Jet.h"
#include "DataFormats/L1Trigger/interface/EtSum.h"

#include "DataFormats/Math/interface/LorentzVector.h"

namespace l1t {

  class CaloTools{
  
    //class is not designed to be instanced
  private:
    CaloTools(){}
    ~CaloTools(){}
  
  public:
    //temporary location of these key parameters, probably should be read in from a database
    //they are private to stop people using them as they will change (naming is invalid for a start)
    static const int kHBHEEnd=28;
    static const int kHFBegin=29;
    static const int kHFEnd=41;
    static const int kHFPhiSeg=1;  // to be deprecated!
    static const int kHFNrPhi=72/kHFPhiSeg;  // to be deprecated!
    static const int kHBHENrPhi=72;  // to be deprecated!
    static const int kNPhi=72;
    static const int kNrTowers = ((kHFEnd-kHFBegin+1)*kHFNrPhi + kHBHEEnd*kHBHENrPhi )*2;
    static const int kNrHBHETowers = kHBHEEnd*kHBHENrPhi*2;

    // These are the saturation codes sent from Layer 1 as the tower pT to Layer 2
    // 509 = Layer 1 received saturated HCAL TP
    // 510 = Layer 1 received saturated ECAL TP
    // 511 = Layer 1 received both saturated ECAL & HCAL TPs
    static const int kSatHcal = 509;
    static const int kSatEcal = 510;
    static const int kSatTower = 511;

    // Jet saturation value
    static const int kSatJet = 65535;

  public:
    enum SubDet{ECAL=0x1,HCAL=0x2,CALO=0x3}; //CALO is a short cut for ECAL|HCAL

    static bool insertTower( std::vector<l1t::CaloTower>& towers, const l1t::CaloTower& tower);

    static const l1t::CaloTower&   getTower(const std::vector<l1t::CaloTower>& towers,int iEta,int iPhi);
    static const l1t::CaloCluster& getCluster(const std::vector<l1t::CaloCluster>& clusters,int iEta,int iPhi);

    //returns a hash suitable for indexing a vector, returns caloTowerHashMax if invalid iEta,iPhi
    static size_t caloTowerHash(int iEta,int iPhi);

    //returns maximum size of hash, for vector allocation
    static size_t caloTowerHashMax();

    //checks if the iEta, iPhi is valid (ie -28->28, 1->72; |29|->|32|,1-72, %4=1)
    static bool isValidIEtaIPhi(int iEta,int iPhi);
    
    //returns the hw Et sum of of a rectangle bounded by iEta-localEtaMin,iEta+localEtaMax,iPhi-localPhiMin,iPhi-localPhiMax (inclusive)
    //sum is either ECAL, HCAL or CALO (ECAL+HCAL) Et
    static int calHwEtSum(int iEta,int iPhi,const std::vector<l1t::CaloTower>& towers,
			  int localEtaMin,int localEtaMax,int localPhiMin,int localPhiMax,SubDet etMode=CALO);
    static int calHwEtSum(int iEta,int iPhi,const std::vector<l1t::CaloTower>& towers,
			  int localEtaMin,int localEtaMax,int localPhiMin,int localPhiMax,
			  int iEtaAbsMax,SubDet etMode=CALO);


    //returns the number of towers with minHwEt<=hwEt<=maxHwEt and iEtaMin<=iEta<=iEtaMax and iPhiMin<=iPhi<=iPhiMax
    //hwEt is either ECAL, HCAL or CALO (ECAL+HCAL) Et
    static size_t calNrTowers(int iEtaMin,int iEtaMax,int iPhiMin,int iPhiMax,const std::vector<l1t::CaloTower>& towers,int minHwEt,int maxHwEt,SubDet etMode=CALO);

    // physical eta/phi position and sizes of trigger towers
    static float towerEta(int ieta);
    static float towerPhi(int ieta, int iphi);
    static float towerEtaSize(int ieta);
    static float towerPhiSize(int ieta);

    // conversion to other index systems
    static int mpEta(int ieta);      // convert to internal MP numbering
    static int caloEta(int ietaMP);  // convert from internal MP to Calo ieta
    static int regionEta(int ieta);  // RCT region
    static int bin16Eta(int ieta);     // gives the eta bin label 
    static int gtEta(int ieta);      // GT eta scale
    static int gtPhi(int ieta, int iphi);      // GT phi scale

    // conversion methods
    static math::PtEtaPhiMLorentzVector p4Demux(l1t::L1Candidate*);
    static l1t::EGamma egP4Demux(l1t::EGamma&);
    static l1t::Tau    tauP4Demux(l1t::Tau&);
    static l1t::Jet    jetP4Demux(l1t::Jet&);
    static l1t::EtSum  etSumP4Demux(l1t::EtSum&);

    static math::PtEtaPhiMLorentzVector p4MP(l1t::L1Candidate*);
    static l1t::EGamma egP4MP(l1t::EGamma&);
    static l1t::Tau    tauP4MP(l1t::Tau&);
    static l1t::Jet    jetP4MP(l1t::Jet&);
    static l1t::EtSum  etSumP4MP(l1t::EtSum&);

    static const int64_t cos_coeff[72];
    static const int64_t sin_coeff[72];

    // mapping between sums in emulator and data
    static const int emul_to_data_sum_index_map[31];

  private:
    // trigger tower eta boundaries
    static std::pair<float,float> towerEtaBounds(int ieta);

    static const l1t::CaloTower nullTower_; //to return when we need to return a tower which was not found/invalid rather than throwing an exception
    static const l1t::CaloCluster nullCluster_; //to return when we need to return a cluster which was not found/invalid rather than throwing an exception

    static const float kGTEtaLSB;
    static const float kGTPhiLSB;
    static const float kGTEtLSB;

  };

}


#endif
