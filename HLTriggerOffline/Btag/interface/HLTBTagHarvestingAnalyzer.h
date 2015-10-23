#ifndef HLTBTagHarvestingAnalyzer_H
#define HLTBTagHarvestingAnalyzer_H

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DQMServices/Core/interface/DQMEDHarvester.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

//DQM services
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "TCutG.h"
#include "TEfficiency.h"


/** \class HLTBTagHarvestingAnalyzer
 *
 *  Code used to produce DQM validation plots for b-tag at HLT.
 *  This class read the plots producted by HLTBTagPerformanceAnalyzer and make plots of: b-tag efficiency vs discr, b-tag efficiency vs jet pt, b-tag efficiency vs mistag rate 
 *
 */


class HLTBTagHarvestingAnalyzer : public DQMEDHarvester { 
		public:
			explicit HLTBTagHarvestingAnalyzer(const edm::ParameterSet&);
			~HLTBTagHarvestingAnalyzer();

			virtual void dqmEndJob(DQMStore::IBooker & ibooker, DQMStore::IGetter & igetter);
			TH1F  calculateEfficiency1D( DQMStore::IBooker& ibooker, DQMStore::IGetter& igetter, TH1 & num, TH1 & den, std::string name );
			bool GetNumDenumerators(DQMStore::IBooker& ibooker, DQMStore::IGetter& igetter, std::string num, std::string den,TH1 * & ptrnum,TH1* & ptrden,int type);
			void mistagrate( DQMStore::IBooker& ibooker, DQMStore::IGetter& igetter, TH1F* num, TH1F* den, std::string effName );

		private:
			// ----------member data ---------------------------
			std::vector<std::string>			hltPathNames_;
			typedef unsigned int				flavour_t;
			typedef std::vector<flavour_t>		flavours_t;
			double 								m_minTag;
			std::vector<std::string>			m_mcLabels;
			std::vector<flavours_t>				m_mcFlavours;
			bool								m_mcMatching;
			std::vector< std::string>			m_histoName;

			// Histogram handler
			std::map<std::string, MonitorElement *> H1_;

	};


#endif

