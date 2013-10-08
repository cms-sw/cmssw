#include <iostream>

#include "HLTrigger/Muon/test/MuEnrichRenormalizer.h"
 
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
 
#include "FWCore/Framework/interface/MakerMacros.h"
 
using namespace edm;
using namespace std;

 
MuEnrichRenormalizer::MuEnrichRenormalizer( const ParameterSet& pset ):
type(pset.getParameter<int>("type"))
{
    theGenToken = consumes<edm::HepMCProduct>(edm::InputTag("VtxSmeared"));
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

void MuEnrichRenormalizer::beginJob()
{}
 
void MuEnrichRenormalizer::analyze( const Event& e, const EventSetup& )
{
      
  Handle< HepMCProduct > EvtHandle ;   
  e.getByToken(theGenToken, EvtHandle) ;
  const HepMC::GenEvent* Gevt = EvtHandle->GetEvent() ;
  bool mybc=false;
  int npart=0;
  int nb=0;
  int nc=0;
  if (Gevt != 0 ) {   
    for ( HepMC::GenEvent::particle_const_iterator
	    particle=Gevt->particles_begin(); particle!=Gevt->particles_end(); ++particle )
      {
	++npart;
	int id=abs((*particle)->pdg_id());
	//int status=(*particle)->status();
	if (id==5 || id== 4){
	  if (npart==6 || npart ==7){
	    mybc=true;
	    break;
	  } else {
	    HepMC::GenVertex* parent=(*particle)->production_vertex();
	    for (HepMC::GenVertex::particles_in_const_iterator ic=parent->particles_in_const_begin() ; ic!=parent->particles_in_const_end() ; ic++ )
	      {
		int pid=(*ic)->pdg_id(); 
		if (pid == 21 && id==5 ) nb++;
		else if (pid == 21 && id==4 ) nc++;
	      }
	  }
	}
      }
  }
  if (nb>1 || nc>1) mybc=true;
  if (mybc)++anaBC;
  else ++anaLight;
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
