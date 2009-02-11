#ifndef HiggsToZZ4LeptonsHLTAnalysis_h
#define HiggsToZZ4LeptonsHLTAnalysis_h

/** \class HiggsToZZ4LeptonsHLTAnalysis
 *
 *  
 *  This class is an HLTProducer creating a flag for each HLT pattern satisfied
 *
 *
 *  \author Nicola De Filippis
 *
 */
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/TriggerNames.h"
#include<vector>
#include<string>

//
// class declaration
//

class HiggsToZZ4LeptonsHLTAnalysis : public edm::EDProducer {

  public:

    explicit HiggsToZZ4LeptonsHLTAnalysis(const edm::ParameterSet&);
    ~HiggsToZZ4LeptonsHLTAnalysis();

 private:
    virtual void produce(edm::Event& event, const edm::EventSetup& eventSetup);
    virtual void endJob();


    /// HLT TriggerResults EDProduct
    edm::InputTag inputTag_,muonlabel_,electronlabel_;
    /// HLT trigger names
    edm::TriggerNames triggerNames_;

    /// false=and-mode (all requested triggers), true=or-mode (at least one)
    bool andOr_;
    bool firstevent_;
    
    /*
    // user provides: true: HLT Names (vstring), or false: HLT Index (vuint32)
    // bool byName_;
    // disabled: user must always provide names, never indices
    */

    /// number of HLT trigger paths requested in configuration
    unsigned int n_;

    /// list of required HLT triggers by HLT name
    std::vector<std::string > HLTPathsByName_;
    /// list of required HLT triggers by HLT index
    std::vector<unsigned int> HLTPathsByIndex_;

    unsigned int nEvents,npassed;
    std::vector<unsigned int> ntrig;
    std::vector<bool> boolflag;
    std::vector<std::string> valias;
    std::string aliasinput,aliasaccept;

};

#endif //HiggsToZZ4LeptonsHLTAnalysis_h
