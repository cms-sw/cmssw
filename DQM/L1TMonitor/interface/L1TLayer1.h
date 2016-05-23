#ifndef L1TLayer1_H
#define L1TLayer1_H

// system include files
#include <memory>
#include <string>
#include <array>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"

#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"

class L1TLayer1 : public DQMEDAnalyzer {
  public:
    L1TLayer1(const edm::ParameterSet& ps);
    virtual ~L1TLayer1();
  
  protected:
    void analyze(const edm::Event& e, const edm::EventSetup& c);
    virtual void bookHistograms(DQMStore::IBooker &ibooker, const edm::Run&, const edm::EventSetup&) override;
    virtual void dqmBeginRun(const edm::Run&, const edm::EventSetup&);
    void beginLuminosityBlock(const edm::LuminosityBlock&, const edm::EventSetup&);
  
  private:
    void updateMismatch(const edm::Event& e, int mismatchType);
    // Input and config info 
    edm::EDGetTokenT<EcalTrigPrimDigiCollection> ecalTPSourceRecd_;
    std::string ecalTPSourceRecdLabel_;
    edm::EDGetTokenT<HcalTrigPrimDigiCollection> hcalTPSourceRecd_;
    std::string hcalTPSourceRecdLabel_;
    edm::EDGetTokenT<EcalTrigPrimDigiCollection> ecalTPSourceSent_;
    std::string ecalTPSourceSentLabel_;
    edm::EDGetTokenT<HcalTrigPrimDigiCollection> hcalTPSourceSent_;
    std::string hcalTPSourceSentLabel_;
    std::string histFolder_;
    int tpFillThreshold_;

    // Some types to ease the pain of compring the sets of towers
    class SimpleTower {
      public:
        int8_t ieta_;
        int8_t iphi_;
        // Sometimes pt, sometimes pt+feature bits
        int32_t data_;
        // Only available from received towers
        int16_t linkFlags_;
        SimpleTower(int8_t ieta, int8_t iphi, int32_t data, int16_t linkFlags=0):ieta_(ieta),iphi_(iphi), data_(data), linkFlags_(linkFlags) {};

        // Lexicographical strict weak ordering
        // allows use of std::set_intersection, etc.
        bool operator<(const SimpleTower& b) const {
          if ( ieta_ < b.ieta_ ) return true;
          if ( ieta_ == b.ieta_ && iphi_ < b.iphi_ ) return true;
          if ( ieta_ == b.ieta_ && iphi_ == b.iphi_ && data_ < b.data_ ) return true;
          return false;
        };
    };
    typedef std::set<SimpleTower> SimpleTowerSet;

    MonitorElement *ecalOccSent_;
    MonitorElement *ecalOccSent_isFineGrainVB_;

    MonitorElement *ecalOccRecd_;
    MonitorElement *ecalOccRecd_isFineGrainVB_;
    MonitorElement *ecalOccRecd_isECALTowerMasked_;
    MonitorElement *ecalOccRecd_isECALLinkMasked_;
    MonitorElement *ecalOccRecd_isECALLinkInError_;

    MonitorElement *ecalOccSentAndRecd_;
    MonitorElement *ecalOccSentNotRecd_;
    MonitorElement *ecalOccRecdNotSent_;

    MonitorElement *ecalOccMaskedSentAndRecd_;
    MonitorElement *ecalOccMaskedSentNotRecd_;
    MonitorElement *ecalOccMaskedRecdNotSent_;

    MonitorElement *ecalTPRawEtRecd_;
    MonitorElement *ecalTPRawEtSent_;
    MonitorElement *ecalTPRawEtRecdNotSent_;
    MonitorElement *ecalTPRawEtSentNotRecd_;
    MonitorElement *ecalTPRawEtSentAndRecd_;

    MonitorElement *hcalOccSent_;
    MonitorElement *hcalOccSent_hasFeatureBits_;

    MonitorElement *hcalOccRecd_;
    MonitorElement *hcalOccRecd_hasFeatureBits_;
    MonitorElement *hcalOccRecd_isHCALTowerMasked_;
    MonitorElement *hcalOccRecd_isHCALLinkMasked_;
    MonitorElement *hcalOccRecd_isHCALLinkInError_;

    MonitorElement *hcalOccSentAndRecd_;
    MonitorElement *hcalOccSentNotRecd_;
    MonitorElement *hcalOccRecdNotSent_;

    MonitorElement *hcalOccMaskedSentAndRecd_;
    MonitorElement *hcalOccMaskedSentNotRecd_;
    MonitorElement *hcalOccMaskedRecdNotSent_;

    MonitorElement *hcalTPRawEtRecd_;
    MonitorElement *hcalTPRawEtSent_;
    MonitorElement *hcalTPRawEtRecdNotSent_;
    MonitorElement *hcalTPRawEtSentNotRecd_;
    MonitorElement *hcalTPRawEtSentAndRecd_;

    MonitorElement *last20Mismatches_;
    std::array<std::pair<std::string, int>, 20> last20MismatchArray_;
    size_t lastMismatchIndex_{0};
};

#endif
