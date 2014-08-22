#ifndef HLTBTagHarvestingAnalyzer_H
#define HLTBTagHarvestingAnalyzer_H

// system include files
#include <memory>
#include <string>
#include <vector>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

//DQM services
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/MonitorElement.h"

/** \class HLTBTagHarvestingAnalyzer
 *
 *  Top level steering routine for HLT b tag performance analysis.
 *
 */

#include "TH1F.h"
#include "TH2F.h"
#include "TH3F.h"
#include "TProfile.h"
#include "RVersion.h"

#if ROOT_VERSION_CODE >= ROOT_VERSION(5,27,0)
#include "TEfficiency.h"
#else
#include "TGraphAsymmErrors.h"
#endif

 
using namespace edm;

using namespace std;

#if ROOT_VERSION_CODE >= ROOT_VERSION(5,27,0)
class HLTBTagHarvestingAnalyzer : public edm::EDAnalyzer { 
#else
class HLTBTagHarvestingAnalyzer : public edm::EDAnalyzer , public TGraphAsymmErrors{
#endif
 

   public:
      explicit HLTBTagHarvestingAnalyzer(const edm::ParameterSet&);
      ~HLTBTagHarvestingAnalyzer();

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);


   private:
      virtual void beginJob() ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

     TProfile * calculateEfficiency1D( TH1* num, TH1* den, string name );
     bool GetNumDenumerators(string num,string den,TH1 * & ptrnum,TH1* & ptrden,int type, double minTag, double maxTag);
     void mistagrate( TProfile* num, TProfile* den, string effName );
      virtual void beginRun(edm::Run const&, edm::EventSetup const&);
      virtual void endRun(edm::Run const&, edm::EventSetup const&);
      virtual void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&);
      virtual void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&);
        
      // ----------member data ---------------------------
      std::vector<std::string>  hltPathNames_;

typedef unsigned int            flavour_t;
typedef std::vector<flavour_t>  flavours_t;

      std::vector<std::string>  m_mcLabels;         // MC truth match - labels
      std::vector<flavours_t>   m_mcFlavours;       // MC truth match - flavours selection
      bool                      m_mcMatching;       // MC truth matching anabled/disabled
		std::vector<double> minTags;
		double maxTag;
      DQMStore * dqm;
      // Histogram handler
      std::map<std::string, MonitorElement *> H1_;

};


#endif
