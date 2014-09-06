#ifndef HLTBTagHarvestingAnalyzer_H
#define HLTBTagHarvestingAnalyzer_H

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

#include "TCutG.h"

#if ROOT_VERSION_CODE >= ROOT_VERSION(5,27,0)
#include "TEfficiency.h"
#else
#include "TGraphAsymmErrors.h"
#endif


/** \class HLTBTagHarvestingAnalyzer
 *
 *  Code used to produce DQM validation plots for b-tag at HLT.
 *  This class read the plots producted by HLTBTagPerformanceAnalyzer and make plots of: b-tag efficiency vs discr, b-tag efficiency vs jet pt, b-tag efficiency vs mistag rate 
 *
 */

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

			TH1F * calculateEfficiency1D( TH1* num, TH1* den, string name );
			bool GetNumDenumerators(string num,string den,TH1 * & ptrnum,TH1* & ptrden,int type);
			void mistagrate( TH1F* num, TH1F* den, string effName );
			virtual void beginRun(edm::Run const&, edm::EventSetup const&);
			virtual void endRun(edm::Run const&, edm::EventSetup const&);
			virtual void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&);
			virtual void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&);

			// ----------member data ---------------------------
			std::vector<std::string>			hltPathNames_;
			typedef unsigned int				flavour_t;
			typedef std::vector<flavour_t>		flavours_t;
			double 								m_minTag;
			std::vector<std::string>			m_mcLabels;
			std::vector<flavours_t>				m_mcFlavours;
			bool								m_mcMatching;
			std::vector< std::string>			m_histoName;
			DQMStore *							dqm;

			// Histogram handler
			std::map<std::string, MonitorElement *> H1_;

	};


#endif

