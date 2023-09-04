#include "GeneratorInterface/GenFilters/plugins/PhotonGenFilter.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "GeneratorInterface/GenFilters/plugins/MCFilterZboostHelper.h"
#include <iostream>

using namespace edm;
using namespace std;

PhotonGenFilter::PhotonGenFilter(const edm::ParameterSet &iConfig)
    : token_(consumes<edm::HepMCProduct>(
          edm::InputTag(iConfig.getUntrackedParameter("moduleLabel", std::string("generator")), "unsmeared"))),
      betaBoost(iConfig.getUntrackedParameter("BetaBoost", 0.))
{
    // Constructor implementation
    vector<int> defpid;
    defpid.push_back(0);
    particleID = iConfig.getUntrackedParameter<vector<int>>("ParticleID", defpid);
    vector<double> defptmin;
    defptmin.push_back(0.);
    ptMin = iConfig.getUntrackedParameter<vector<double>>("MinPt", defptmin);

    vector<double> defetamin;
    defetamin.push_back(-10.);
    etaMin = iConfig.getUntrackedParameter<vector<double>>("MinEta", defetamin);
    vector<double> defetamax;
    defetamax.push_back(10.);
    etaMax = iConfig.getUntrackedParameter<vector<double>>("MaxEta", defetamax);
    vector<int> defstat;
    defstat.push_back(0);
    status = iConfig.getUntrackedParameter<vector<int>>("Status", defstat);
    vector<double> defdrmin;
    defdrmin.push_back(0.);
    drMin = iConfig.getUntrackedParameter<vector<double>>("drMin", defdrmin);

    // check if beta is smaller than 1
    if (std::abs(betaBoost) >= 1)
    {
        edm::LogError("PhotonGenFilter") << "Input beta boost is >= 1 !";
    }
}

PhotonGenFilter::~PhotonGenFilter()
{
    // Destructor implementation
}

bool PhotonGenFilter::filter(edm::StreamID, edm::Event &iEvent, const edm::EventSetup &iSetup) const
{
    using namespace edm;
    Handle<HepMCProduct> evt;
    iEvent.getByToken(token_, evt);
    const HepMC::GenEvent * myGenEvent = evt->GetEvent();

    for (HepMC::GenEvent::particle_const_iterator p = myGenEvent->particles_begin(); p != myGenEvent->particles_end();
         ++p)
    {
        for (unsigned int i = 0; i < particleID.size(); i++)
        {
            if (particleID[i] == (*p)->pdg_id() || particleID[i] == 0)
            {
                if ((*p)->momentum().perp() > ptMin[i] && ((*p)->status() == status[i] || status[i] == 0))
                {
                    HepMC::FourVector mom = MCFilterZboostHelper::zboost((*p)->momentum(), betaBoost);
                    if (mom.eta() > etaMin[i] && mom.eta() < etaMax[i])
                    {
                        bool accepted_photon = true;
                        double phi = (*p)->momentum().phi();
                        double eta = mom.eta();
                        for (HepMC::GenEvent::particle_const_iterator q = myGenEvent->particles_begin(); q != myGenEvent->particles_end();
                             ++q)
                        {
                            if (&p != &q)
                            {
                                if ((*q)->momentum().perp() > 2 && (*q)->pdg_id() != 22 && (*q)->status() == 1)// && abs((*q)->charge()) > 0)
                                {
                                    double phi2 = (*p)->momentum().phi();
                                    double deltaphi = fabs(phi - phi2);
                                    if (deltaphi > M_PI)
                                        deltaphi = 2. * M_PI - deltaphi;
                                    HepMC::FourVector mom2 = MCFilterZboostHelper::zboost((*q)->momentum(), betaBoost);
                                    double eta2 = mom2.eta();
                                    double deltaeta = fabs(eta - eta2);
                                    double deltaR = sqrt(deltaeta * deltaeta + deltaphi * deltaphi);
                                    if (deltaR < 0.1)
                                        accepted_photon = false;
                                }
                            }
                        }
                        if (accepted_photon) return true;
                    }
                }
            }
        }
    }

    // Implementation for event filtering
    return false; // Return true if event passes the filter, false otherwise
}

DEFINE_FWK_MODULE(PhotonGenFilter);