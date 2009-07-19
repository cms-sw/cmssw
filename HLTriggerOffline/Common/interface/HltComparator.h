#ifndef HLTCOMPARATOR_H
#define HLTCOMPARATOR_H
// Original Author: James Jackson
// $Id: HltComparator.h,v 1.1 2009/06/22 20:11:40 wittich Exp $

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/TriggerNames.h"
#include "DataFormats/Common/interface/TriggerResults.h"

class TH1F;

class HltComparator : public edm::EDFilter
{
    public:
        explicit HltComparator(const edm::ParameterSet&);
        ~HltComparator();

    private:
        edm::InputTag hltOnlineResults_;
        edm::InputTag hltOfflineResults_;
        edm::TriggerNames onlineTriggerNames_;
        edm::TriggerNames offlineTriggerNames_;

        std::vector<std::string> onlineActualNames_;
        std::vector<std::string> offlineActualNames_;
        std::vector<unsigned int> onlineToOfflineBitMappings_;

        std::vector<TH1F*> comparisonHists_;
        std::map<unsigned int, std::map<std::string, unsigned int> > triggerComparisonErrors_;

        bool init_;
	bool verbose_;
	bool verbose() const { return verbose_; }

	std::vector<std::string> skipPathList_;

        unsigned int numTriggers_;

        virtual void beginJob(const edm::EventSetup&) ;
        virtual bool filter(edm::Event&, const edm::EventSetup&);
        virtual void endJob() ;
        void initialise(const edm::TriggerResults&, 
			const edm::TriggerResults&);
        std::string formatResult(const unsigned int);
};

#endif // HLTCOMPARATOR_HH

