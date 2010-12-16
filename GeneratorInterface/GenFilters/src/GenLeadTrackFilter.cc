/***********************************************************
*                 GenLeadTrackFilter                       *
*                 ------------------                       *
*                                                          *
* Original Author: Souvik Das, Cornell University          *
* Created        : 7 August 2009                           *
*                                                          *
*  Allows events which have at least one generator level   *
*  charged particle with pT greater than X GeV within      *
*  |eta| less than Y, where X and Y are specified in the   *
*  cfi configuration file.                                 *
***********************************************************/

#include "GeneratorInterface/GenFilters/interface/GenLeadTrackFilter.h"
#include "FWCore/Framework/interface/EventSetup.h"

using namespace edm;
using namespace reco;
using namespace math;

GenLeadTrackFilter::GenLeadTrackFilter(const edm::ParameterSet& iConfig)
{
  hepMCProduct_label_   = iConfig.getParameter<InputTag>("HepMCProduct");
  genLeadTrackPt_       = iConfig.getParameter<double>("GenLeadTrackPt");
  genEta_               = iConfig.getParameter<double>("GenEta");
}

GenLeadTrackFilter::~GenLeadTrackFilter()
{
}

bool GenLeadTrackFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  bool allow=false;
  
  ESHandle<HepPDT::ParticleDataTable> pdt;
  iSetup.getData(pdt);

  Handle<HepMCProduct> genEvent;
  iEvent.getByLabel(hepMCProduct_label_, genEvent);
  if (genEvent.isValid())
  {
    float genLeadTrackPt=-100;
    for (HepMC::GenEvent::particle_const_iterator iter=(*(genEvent->GetEvent())).particles_begin();
         iter!=(*(genEvent->GetEvent())).particles_end(); 
         ++iter)
    {
      HepMC::GenParticle* theParticle=*iter;
      double pt=pow(pow(theParticle->momentum().px(),2)+pow(theParticle->momentum().py(),2), 0.5);
      double charge=pdt->particle(theParticle->pdg_id())->charge();
      if (theParticle->status()==1 &&
          charge!=0 &&
          fabs(theParticle->momentum().eta())<genEta_ &&
          pt>genLeadTrackPt)
      {
        genLeadTrackPt=pt;
      }
    }
    if (genLeadTrackPt>genLeadTrackPt_) allow=true; else allow=false;
  }
  else 
  {
    std::cout<<"genEvent in not valid!"<<std::endl;
  }
  return allow;
}

// ------------ method called once each job just before starting event loop  ------------

// ------------ method called once each job just after ending the event loop  ------------
void GenLeadTrackFilter::endJob() 
{
}

//define this as a plug-in
DEFINE_FWK_MODULE(GenLeadTrackFilter);
