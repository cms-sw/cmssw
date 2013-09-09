// -*- C++ -*-
//
// Package:    GenFilters
// Class:      ComphepSingletopFilter
// 
/**\class ComphepSingletopFilter ComphepSingletopFilter.cc GeneratorInterface/GenFilters/src/ComphepSingletopFilter.cc

 Description: a filter to match LO/NLO in comphep-generated singletop

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Vladimir Molchanov
//         Created:  Wed Mar 25 19:43:12 CET 2009
//
//


// system include files
#include <vector>
#include <boost/format.hpp>

// user include files
#include "GeneratorInterface/GenFilters/interface/ComphepSingletopFilter.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"


//
// class declaration
//

ComphepSingletopFilter::ComphepSingletopFilter(const edm::ParameterSet& iConfig) :
    hepMCProductTag_(iConfig.getParameter<edm::InputTag>("hepMCProductTag")) {
    ptsep = iConfig.getParameter<double>("pTSep");
}


ComphepSingletopFilter::~ComphepSingletopFilter() {}


void ComphepSingletopFilter::beginJob() {
    read22 = read23 = 0;
    pass22 = pass23 = 0;
}


void ComphepSingletopFilter::endJob() {
    using namespace std;
    cout << "Proc:     2-->2     2-->3     Total" << endl;
    cout << boost::format("Read: %9d %9d %9d") % read22 % read23 % (read22+read23)
         << endl;
    cout << boost::format("Pass: %9d %9d %9d") % pass22 % pass23 % (pass22+pass23)
         << endl;
}


bool ComphepSingletopFilter::filter(
    edm::Event& iEvent,
    const edm::EventSetup& iSetup) {

    using namespace std;
    using namespace HepMC;
  
    edm::Handle<edm::HepMCProduct> evt;
    iEvent.getByLabel(hepMCProductTag_, evt);
    const HepMC::GenEvent * myEvt = evt->GetEvent();
//  myEvt->print();  // to print the record

    const GenParticle * gp_clep = NULL;

    for (GenEvent::particle_const_iterator it = myEvt->particles_begin();
         it != myEvt->particles_end(); ++it) {
        if ((*it)->status() == 3) {
            int abs_id = abs((*it)->pdg_id());
            if (abs_id==11 || abs_id==13 || abs_id==15) {
                gp_clep = *it;
                break;
            }
        }
    }

    if (! gp_clep) {
        cerr << "ERROR: ComphepSingletopFilter: no charged lepton" << endl;
        return false;
    }

    int id_bdec = 0;
    vector<const GenParticle *> vgp_bsec;

    GenVertex * gv_hard = gp_clep->production_vertex();

    for (GenVertex::particle_iterator it = gv_hard->particles_begin(children);
         it != gv_hard->particles_end(children); ++it) {
        int pdg_id = (*it)->pdg_id();
        if (abs(pdg_id) == 5) {
            if (pdg_id * (gp_clep->pdg_id()) < 0) {
                id_bdec = pdg_id;
            } else {
                vgp_bsec.push_back(*it);
            }
        }
    }

    bool process22 = (vgp_bsec.size() == 0);

    GenVertex * gv = NULL;
    if (process22) {
        for (GenVertex::particle_iterator it = gv_hard->particles_begin(parents);
             it != gv_hard->particles_end(parents); ++it) {
            if ((*it)->pdg_id() == id_bdec) {
                gv = (*it)->production_vertex();
                break;
            }
        }
        if (! gv) {
            cerr << "ERROR: ComphepSingletopFilter: HepMC inconsistency" << endl;
            myEvt->print();
            return false;
        }
    } else {
        gv = vgp_bsec.back()->end_vertex();
    }
    const GenParticle * gp;
    while (gv) {
        gp = NULL;
        for (GenVertex::particle_iterator it = gv->particles_begin(children);
             it != gv->particles_end(children); ++it) {
            if ((*it)->pdg_id() == -id_bdec) {
                if (!gp || (*it)->momentum().perp2() > gp->momentum().perp2()) {
                    gp = *it;
                }
            }
        }
        if (gp) {
            gv = gp->end_vertex();
            vgp_bsec.push_back(gp);
        } else {
            gv = NULL;
        }
    }

    if (vgp_bsec.size() == 0) {
        cerr << "ERROR: ComphepSingletopFilter: HepMC inconsistency" << endl;
        return false;
    }

    double pt = vgp_bsec.back()->momentum().perp();
    bool pass;
    if (process22) {
        read22 += 1;
        pass = pt < ptsep;
        if (pass) pass22 += 1;
    } else {
        read23 += 1;
        pass = ptsep <= pt;
        if (pass) pass23 += 1;
    }

    return pass;
}


