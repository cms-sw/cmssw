#ifndef HLTCOMPARATOR_H
#define HLTCOMPARATOR_H
// Original Author: James Jackson
// $Id: HltComparator.h,v 1.7 2010/02/25 19:14:36 wdd Exp $

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

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

        std::vector<std::string> onlineActualNames_;
        std::vector<std::string> offlineActualNames_;
        std::vector<unsigned int> onlineToOfflineBitMappings_;

        std::vector<TH1F*> comparisonHists_;
        std::map<unsigned int, std::map<std::string, unsigned int> > triggerComparisonErrors_;

        bool init_;
	bool verbose_;
	bool verbose() const { return verbose_; }

	std::vector<std::string> skipPathList_;
	std::vector<std::string> usePathList_;

        unsigned int numTriggers_;

        virtual void beginJob() ;
        virtual bool filter(edm::Event&, const edm::EventSetup&);
        virtual void endJob() ;
        void initialise(const edm::TriggerResults&, 
			const edm::TriggerResults&,
                        edm::Event& e);
        std::string formatResult(const unsigned int);
};

#endif // HLTCOMPARATOR_HH

