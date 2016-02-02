#ifndef HiJetBackground_HiFJRhoProducer_h
#define HiJetBackground_HiFJRhoProducer_h

// system include files
#include <memory>
#include <sstream>
#include <string>
#include <vector>

//root
#include "TTree.h"

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/JetReco/interface/Jet.h"

//#include <boost/icl/interval_map.hpp>

//
// class declaration
//

class HiFJRhoProducer : public edm::EDProducer {
   public:
      explicit HiFJRhoProducer(const edm::ParameterSet&);
      ~HiFJRhoProducer();

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

   private:
      virtual void beginJob() override;
      virtual void produce(edm::Event&, const edm::EventSetup&) override;
      virtual void endJob() override;
      
      double calcMd(const reco::Jet *jet);
      bool   isPackedCandidate(const reco::Candidate* candidate);
      
      // ----------member data ---------------------------
      //input
      edm::EDGetTokenT<edm::View<reco::Jet>>    jetsToken_;
      
      //members
      unsigned int   nExcl_;              ///Number of leading jets to exclude
      bool           checkJetCand, usingPackedCand;
      //std::map<int,double> mapEtaRanges_;         //eta ranges
      //boost::icl::interval_map<double, int> mapToIndex_; //eta intervals to be stored in event
      /* std::map<int,double>           mapToRho_;   //rho to be stored in event */
      /* std::map<int,double>           mapToRhoM_;  //rhoM to be stored in event */
      
};

#endif
