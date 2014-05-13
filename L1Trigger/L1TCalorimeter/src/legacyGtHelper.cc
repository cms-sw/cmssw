// legacyGtHelper.cc
// Authors: Alex Barbieri
//
// This is a collection of helper methods to make sure that
// the objects passed to the legacy GT are using the proper
// Et scales and eta coordinates.

#include "L1Trigger/L1TCalorimeter/interface/legacyGtHelper.h"

const unsigned int gtEta(const unsigned int iEta);

namespace l1t {

  void JetToGtScales(CaloParams *params,
		     const std::vector<l1t::Jet> * input,
		     std::vector<l1t::Jet> *output){

    for(std::vector<l1t::Jet>::const_iterator itJet = input->begin();
	itJet != input->end(); ++itJet){
      const unsigned newEta = gtEta(itJet->hwEta());
      const uint16_t rankPt = params->jetScale().rank((uint16_t)itJet->hwPt());

      ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> > *jetLorentz =
	  new ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> >();

      l1t::Jet gtJet(*jetLorentz, rankPt, newEta, itJet->hwPhi(), itJet->hwQual());
      output->push_back(gtJet);
    }
  }

  void EGammaToGtScales(CaloParams *params,
			const std::vector<l1t::EGamma> * input,
			std::vector<l1t::EGamma> *output){
  }

  void TauToGtScales(CaloParams *params,
		     const std::vector<l1t::Tau> * input,
		     std::vector<l1t::Tau> *output){
    for(std::vector<l1t::Tau>::const_iterator itTau = input->begin();
	itTau != input->end(); ++itTau){
      const unsigned newEta = gtEta(itTau->hwEta());
      const uint16_t rankPt = params->jetScale().rank((uint16_t)itTau->hwPt());

      ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> > *tauLorentz =
	  new ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> >();

      l1t::Tau gtTau(*tauLorentz, rankPt, newEta, itTau->hwPhi(), itTau->hwQual());
      output->push_back(gtTau);
    }

  }

  void EtSumToGtScales(CaloParams *params,
		       const std::vector<l1t::EtSum> * input,
		       std::vector<l1t::EtSum> *output){
  }


}

const unsigned int gtEta(const unsigned int iEta)
{
  unsigned rctEta = (iEta<11 ? 10-iEta : iEta-11);
  return (((rctEta % 7) & 0x7) | (iEta<11 ? 0x8 : 0));
}
