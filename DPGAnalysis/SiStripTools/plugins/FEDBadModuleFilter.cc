// -*- C++ -*-
//
// Package:    SiStripTools
// Class:      FEDBadModuleFilter
// 
/**\class FEDBadModuleFilter FEDBadModuleFilter.cc DPGAnalysis/SiStripTools/plugins/FEDBadModuleFilter.cc

 Description: template EDFilter to select events with selected modules with SiStripDigis or SiStripClusters

 Implementation:
     
*/
//
// Original Author:  Andrea Venturi
//         Created:  Wed Oct 22 17:54:30 CEST 2008
//
//


// system include files
#include <memory>
#include "TProfile.h"

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <vector>
#include <set>

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/DetId/interface/DetIdCollection.h"

#include "DPGAnalysis/SiStripTools/interface/RunHistogramManager.h"
#include "CommonTools/UtilAlgos/interface/DetIdSelector.h"

//
// class declaration
//

class FEDBadModuleFilter : public edm::EDFilter {
   public:
      explicit FEDBadModuleFilter(const edm::ParameterSet&);
      ~FEDBadModuleFilter();

private:
  virtual void beginJob() ;
  virtual bool filter(edm::Event&, const edm::EventSetup&);
  virtual bool beginRun(edm::Run&, const edm::EventSetup&);
  virtual bool endRun(edm::Run&, const edm::EventSetup&);
  virtual void endJob() ;
  
      // ----------member data ---------------------------

  edm::InputTag m_digibadmodulecollection;
  unsigned int m_modulethr;
  std::set<unsigned int> m_modules;
  DetIdSelector m_modsel;
  bool m_wantedhist;
  bool m_printlist;
  const unsigned int m_maxLS;
  const unsigned int m_LSfrac;
  RunHistogramManager m_rhm;
  TH1F** m_nbadrun;
  TProfile** m_nbadvsorbrun;

};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
FEDBadModuleFilter::FEDBadModuleFilter(const edm::ParameterSet& iConfig):
  m_digibadmodulecollection(iConfig.getParameter<edm::InputTag>("collectionName")),
  m_modulethr(iConfig.getParameter<unsigned int>("badModThr")),
  m_modsel(),
  m_wantedhist(iConfig.getUntrackedParameter<bool>("wantedHisto",false)),
  m_printlist(iConfig.getUntrackedParameter<bool>("printList",false)),
  m_maxLS(iConfig.getUntrackedParameter<unsigned int>("maxLSBeforeRebin",100)),
  m_LSfrac(iConfig.getUntrackedParameter<unsigned int>("startingLSFraction",4)),
  m_rhm()

{
   //now do what ever initialization is needed

  if(m_wantedhist) {
    m_nbadrun = m_rhm.makeTH1F("nbadrun","Number of bad channels",500,-0.5,499.5);
    m_nbadvsorbrun = m_rhm.makeTProfile("nbadvsorbrun","Number of bad channels vs time",m_LSfrac*m_maxLS,0,m_maxLS*262144);
  }

  std::vector<unsigned int> modules = iConfig.getUntrackedParameter<std::vector<unsigned int> >("moduleList",std::vector<unsigned int>());
  m_modules = std::set<unsigned int>(modules.begin(),modules.end());

  if(iConfig.exists("moduleSelection")) {
    m_modsel = DetIdSelector(iConfig.getUntrackedParameter<edm::ParameterSet>("moduleSelection"));
  }
}


FEDBadModuleFilter::~FEDBadModuleFilter()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called on each new Event  ------------
bool
FEDBadModuleFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;

   Handle<DetIdCollection > badmodules;
   iEvent.getByLabel(m_digibadmodulecollection,badmodules);

   unsigned int nbad = 0;
   if(m_printlist || m_modules.size()!=0 || m_modsel.isValid() ) {
     for(DetIdCollection::const_iterator mod = badmodules->begin(); mod!=badmodules->end(); ++mod) {
       if((m_modules.size() == 0 || m_modules.find(*mod) != m_modules.end() ) && (!m_modsel.isValid() || m_modsel.isSelected(*mod))) {
	 ++nbad; 
	 if(m_printlist) edm::LogInfo("FEDBadModule") << *mod;
       }
     }
   }
   else {
     nbad = badmodules->size();
   }

   if(m_wantedhist) {
     if(m_nbadvsorbrun && *m_nbadvsorbrun) (*m_nbadvsorbrun)->Fill(iEvent.orbitNumber(),nbad);
     if(m_nbadrun && *m_nbadrun) (*m_nbadrun)->Fill(nbad);
   }
   
   return (nbad >= m_modulethr);

}

// ------------ method called once each job just before starting event loop  ------------
void 
FEDBadModuleFilter::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
FEDBadModuleFilter::endJob() {
}

bool
FEDBadModuleFilter::beginRun(edm::Run& iRun, const edm::EventSetup& iSetup) 
{

  if(m_wantedhist) {

    m_rhm.beginRun(iRun);
    if(*m_nbadvsorbrun) {
      (*m_nbadvsorbrun)->SetBit(TH1::kCanRebin);
      (*m_nbadvsorbrun)->GetXaxis()->SetTitle("time [Orb#]"); 
      (*m_nbadvsorbrun)->GetYaxis()->SetTitle("Bad Channels"); 
    }

  }

  return false;

}

bool
FEDBadModuleFilter::endRun(edm::Run& iRun, const edm::EventSetup& iSetup) 
{
  return false;
}
//define this as a plug-in
DEFINE_FWK_MODULE(FEDBadModuleFilter);
