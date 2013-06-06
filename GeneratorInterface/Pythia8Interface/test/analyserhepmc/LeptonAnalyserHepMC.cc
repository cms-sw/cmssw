#include "GeneratorInterface/Pythia8Interface/test/analyserhepmc/LeptonAnalyserHepMC.h"
#include "HepMC/GenVertex.h"
#include "HepMC/GenParticle.h"
#include <iterator>
#include <algorithm>


LeptonAnalyserHepMC::LeptonAnalyserHepMC(double aMaxEta, double aThresholdEt)
: MaxEta(aMaxEta), ThresholdEt(aThresholdEt),
  RConeIsol(0.3), MaxPtIsol(2.),
  RIdJet(0.3), EpsIdJet(0.6)
{;}


std::vector<HepMC::GenParticle>
          LeptonAnalyserHepMC::isolatedLeptons(const HepMC::GenEvent* pEv)
{
  HepMC::GenEvent::particle_const_iterator part;
  HepMC::GenEvent::particle_const_iterator part1;

  std::vector<HepMC::GenParticle> isoleptons;
  bool lepton = false;
  for(part = pEv->particles_begin(); part != pEv->particles_end(); ++part ) {
    lepton = false;
    if( abs((*part)->pdg_id()) == 11 ) lepton = true;
    if( abs((*part)->pdg_id()) == 13 ) lepton = true;
    if( !(*part)->end_vertex() && (*part)->status() == 1 && lepton &&
        fabs((*part)->momentum().eta()) < MaxEta &&
        (*part)->momentum().perp() > ThresholdEt ) {

      double eta0 = (*part)->momentum().eta();
      double phi0 = (*part)->momentum().phi();
      double pti, dist, etai, phii;
      bool isol = true;
      for(part1 = pEv->particles_begin();
          part1 != part && part1 != pEv->particles_end();
                                                 part1++ ) {
        if( !(*part1)->end_vertex() && (*part1)->status() == 1 ) {
          pti = (*part1)->momentum().perp();
          etai = (*part1)->momentum().eta();
          phii = (*part1)->momentum().phi();
          dist = sqrt( (eta0-etai)*(eta0-etai) + (phi0-phii)*(phi0-phii) );
          if(dist < RConeIsol && pti > MaxPtIsol ) { isol = false; break;}
        }
      }
      if(isol) isoleptons.push_back(HepMC::GenParticle(**part));
    }
  }
  return isoleptons;
}


int LeptonAnalyserHepMC::nIsolatedLeptons(const HepMC::GenEvent* pEv)
{
  std::vector<HepMC::GenParticle> isoleptons = isolatedLeptons(pEv);
  return isoleptons.size();
}


double LeptonAnalyserHepMC::MinMass(const HepMC::GenEvent* pEv)
{
  std::vector<HepMC::GenParticle> isoleptons = isolatedLeptons(pEv);
  if(isoleptons.size() < 2) return 0.;
  double MinM=100000.;
  std::vector<HepMC::GenParticle>::iterator ipart, ipart1;
  for ( ipart=isoleptons.begin(); ipart != isoleptons.end(); ipart++) {
    for ( ipart1=isoleptons.begin(); ipart1 != isoleptons.end(); ipart1++) {
      if(ipart1 == ipart) continue;
      double px = ipart->momentum().px() + ipart1->momentum().px();
      double py = ipart->momentum().py() + ipart1->momentum().py();
      double pz = ipart->momentum().pz() + ipart1->momentum().pz();
      double e = ipart->momentum().e() + ipart1->momentum().e();
      double mass = sqrt(e*e - px*px - py*py -pz*pz);
      if(mass < MinM) MinM = mass;
    }
  }
  return MinM;
}


std::vector <fastjet::PseudoJet>
  LeptonAnalyserHepMC::removeLeptonsFromJets(std::vector<fastjet::PseudoJet>& jets,
                                             const HepMC::GenEvent* pEv)
{
  std::vector<HepMC::GenParticle> isoleptons = isolatedLeptons(pEv);
  if(isoleptons.empty()) return jets;
  std::vector<fastjet::PseudoJet>::iterator ijet;
  std::vector<HepMC::GenParticle>::iterator ipart;
  std::vector<fastjet::PseudoJet> newjets;
  for ( ijet = jets.begin(); ijet != jets.end(); ijet++) {
    if (fabs(ijet->rap()) > 5.0) continue;
    bool isLepton = false;
    for ( ipart=isoleptons.begin(); ipart != isoleptons.end(); ipart++) {
      fastjet::PseudoJet fjLept(ipart->momentum().px(),
                                ipart->momentum().py(),
                                ipart->momentum().pz(),
                                ipart->momentum().e() );
      //cout << "lepton eta = " << fjLept.rap() << endl;

      if ( fjLept.squared_distance(*ijet) < RIdJet*RIdJet &&
           fabs(ijet->e() - ipart->momentum().e()) / ijet->e() < EpsIdJet )
                                                             isLepton = true;

    }

    if(!isLepton) newjets.push_back(*ijet);

  }   
  return newjets;
}


#if 0
vector<Jet> LeptonAnalyserHepMC::removeLeptonsFromJets(vector<Jet>& jets,
                                                       HepMC::GenEvent* pEv)
{
  vector<HepMC::GenParticle> isoleptons = isolatedLeptons(pEv);
  if(isoleptons.empty()) return jets;
  vector<Jet>::iterator ijet;
  vector<HepMC::GenParticle>::iterator ipart;
#if 00
  // this code can be used if first argument is vector<Jet>, not vector<Jet>&
  for ( ijet = jets.begin(); ijet < jets.end(); ijet++) {
    bool bad = false;
    for ( ipart=isoleptons.begin(); ipart != isoleptons.end(); ipart++) {
      JetableObject jpart(*ipart);
      if(ijet->dist(jpart) < RIdJet &&
         fabs(ijet->e()-ipart->momentum().e())/ijet->e() < EpsIdJet )
                                                             bad = true;
    }
    if(bad) {ijet = jets.erase(ijet); ijet--;}
  }
  return jets;
#endif
  vector<Jet> newjets;
  for ( ijet = jets.begin(); ijet != jets.end(); ijet++) {
    bool islepton = false;
    for ( ipart=isoleptons.begin(); ipart != isoleptons.end(); ipart++) {
      JetableObject jpart(*ipart);
      if(ijet->dist(jpart) < RIdJet &&
         fabs(ijet->e()-ipart->momentum().e())/ijet->e() < EpsIdJet )
                                                       islepton = true;
    }
    if(!islepton) newjets.push_back(*ijet);
  }
  return newjets;
}
#endif
