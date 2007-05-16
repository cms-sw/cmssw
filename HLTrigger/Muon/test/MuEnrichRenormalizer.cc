#include <iostream>

#include "HLTrigger/Muon/test/MuEnrichRenormalizer.h"
 
#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"
 
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
 
#include "FWCore/Framework/interface/MakerMacros.h"
 
using namespace edm;
using namespace std;

 
MuEnrichRenormalizer::MuEnrichRenormalizer( const ParameterSet& pset ):
type(pset.getParameter<int>("type"))
{
    anaIntlumi=0;
    anaLight=0;
    anaBC=0;
    rwbc=1.;
    rwlight=1.;
    if ( type == 1) {
      genIntlumi=0.02465 ; // nb-1 generated events
      genLight=426000; // total generated light quark events
      genBC=192085; // total generated bc quark events
    } else if ( type == 2) {
      genIntlumi=0.2465 ; // nb-1 generated events
      genLight=168300; // total generated light quark events
      genBC=257561; // total generated bc quark events
    } else if ( type == 3) {
      genIntlumi=0.242;
      genLight=10800;
      genBC=141596;
    }
}

void MuEnrichRenormalizer::beginJob( const EventSetup& )
{}
 
void MuEnrichRenormalizer::analyze( const Event& e, const EventSetup& )
{
      
  Handle< HepMCProduct > EvtHandle ;   
  e.getByLabel( "VtxSmeared", EvtHandle ) ;
  const HepMC::GenEvent* Gevt = EvtHandle->GetEvent() ;
  bool mybc=false;
  int npart=0;
  if (Gevt != 0 ) {   
    for ( HepMC::GenEvent::particle_const_iterator
	    particle=Gevt->particles_begin(); particle!=Gevt->particles_end()&& npart<10; ++particle )
      {
	++npart;
	int id=abs((*particle)->pdg_id());
	int status=(*particle)->status();
	if (id < 6 && status>1){
	  if (id>3){
	    mybc=true;
	    break;
	  }
	}
      }
    if (mybc)++anaBC;
    else ++anaLight;
  }
  return ;
}

void MuEnrichRenormalizer::endJob()
{
  double rLight=double(anaLight)/genLight ;
  double rBC=double(anaBC)/genBC ;
 edm::LogVerbatim ("MuEnrichRenormalizer")  << "Generated uds events="<<genLight;
 edm::LogVerbatim ("MuEnrichRenormalizer")  << "Generated bc events="<<genBC;
 edm::LogVerbatim ("MuEnrichRenormalizer")  << "Analyzed uds events="<<anaLight;
 edm::LogVerbatim ("MuEnrichRenormalizer")  << "Analyzed bc events="<<anaBC;
 edm::LogVerbatim ("MuEnrichRenormalizer")  << "Ratio of analyzed to simulated uds="<<rLight;
 edm::LogVerbatim ("MuEnrichRenormalizer")  << "Ratio of analyzed to simulated bc ="<<rBC;
  if (rBC<rLight) {
    rwlight=rBC/rLight;
    anaIntlumi=genIntlumi*rBC;
      }else{
        rwbc=rLight/rBC;
        anaIntlumi=genIntlumi*rLight;
      }
 edm::LogVerbatim ("MuEnrichRenormalizer")  << "rwbc="<<rwbc<<" rwlight="<<rwlight;
 edm::LogVerbatim ("MuEnrichRenormalizer")  << "Corresponding Lumi ="<<anaIntlumi<<" nb-1";
  return ;
}
 
DEFINE_FWK_MODULE(MuEnrichRenormalizer);
