#include "GeneratorInterface/GenFilters/interface/PythiaFilterHT.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include <iostream>

using namespace edm;
using namespace std;


PythiaFilterHT::PythiaFilterHT(const edm::ParameterSet& iConfig) :
	label_(iConfig.getUntrackedParameter("moduleLabel",std::string("generator"))),
	/*minpcut(iConfig.getUntrackedParameter("MinP", 0.)),
	maxpcut(iConfig.getUntrackedParameter("MaxP", 10000.)),
	minptcut(iConfig.getUntrackedParameter("MinPt", 0.)),
	maxptcut(iConfig.getUntrackedParameter("MaxPt", 10000.)),
	minetacut(iConfig.getUntrackedParameter("MinEta", -10.)),
	maxetacut(iConfig.getUntrackedParameter("MaxEta", 10.)),
	minrapcut(iConfig.getUntrackedParameter("MinRapidity", -20.)),
	maxrapcut(iConfig.getUntrackedParameter("MaxRapidity", 20.)),
	minphicut(iConfig.getUntrackedParameter("MinPhi", -3.5)),
	maxphicut(iConfig.getUntrackedParameter("MaxPhi", 3.5)),*/
	minhtcut(iConfig.getUntrackedParameter("MinHT", 0.)),
	motherID(iConfig.getUntrackedParameter("MotherID", 0)) {

		theNumberOfTestedEvt = 0;
		theNumberOfSelected = 0;

}


PythiaFilterHT::~PythiaFilterHT() {
}


//
// member functions
//

// ------------ method called to produce the data  ------------
bool PythiaFilterHT::filter(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  
	theNumberOfTestedEvt++;
	if(theNumberOfTestedEvt%1000 == 0) cout << "Number of tested events = " << theNumberOfTestedEvt <<  endl;

	bool accepted = false;
	Handle<HepMCProduct> evt;
	iEvent.getByLabel(label_, evt);

	const HepMC::GenEvent * myGenEvent = evt->GetEvent();

	double HT = 0;

	for ( HepMC::GenEvent::particle_const_iterator p = myGenEvent->particles_begin(); p != myGenEvent->particles_end(); ++p ) {

		if ( ( (*p)->status() == 23 ) && ( ( abs( (*p)->pdg_id() ) < 6 ) || ( (*p)->pdg_id() == 21 ) ) ) {
			  
			/*rapidity = 0.5*log( ((*p)->momentum().e()+(*p)->momentum().pz()) / ((*p)->momentum().e()-(*p)->momentum().pz()) );

			if ( (*p)->momentum().rho() > minpcut 
					&& (*p)->momentum().rho() < maxpcut
					&& (*p)->momentum().perp() > minptcut 
					&& (*p)->momentum().perp() < maxptcut
					&& (*p)->momentum().eta() > minetacut
					&& (*p)->momentum().eta() < maxetacut 
					&& rapidity > minrapcut
					&& rapidity < maxrapcut 
					&& (*p)->momentum().phi() > minphicut
					&& (*p)->momentum().phi() < maxphicut ) {*/

				if ( motherID == 0 ) {
				   HT += (*p)->momentum().perp();
				} else {
					HepMC::GenParticle* mother = (*((*p)->production_vertex()->particles_in_const_begin()));
					if (abs(mother->pdg_id()) == abs(motherID)) {
						HT += (*p)->momentum().perp();
					}
				}
			//}			
		}
	}
	if ( HT > minhtcut ) accepted = true;

	if (accepted){
		theNumberOfSelected++;
		cout << "========>  Event preselected " << theNumberOfSelected << " HT = " << HT << endl;
		return true; 
	} else {
		//cout << "========>  Event rejected HT = " << HT << endl;
		return false;
	}

}

