// legacyGtHelper.cc
// Authors: Alex Barbieri
//
// This is a collection of helper methods to make sure that
// the objects passed to the legacy GT are using the proper
// Et scales and eta coordinates.

#include "L1Trigger/L1TCalorimeter/interface/legacyGtHelper.h"

const unsigned int gtEta(const unsigned int iEta);

namespace l1t {

  void JetToGtScales(CaloParamsStage1 *params,
		     const std::vector<l1t::Jet> * input,
		     std::vector<l1t::Jet> *output){

    for(std::vector<l1t::Jet>::const_iterator itJet = input->begin();
	itJet != input->end(); ++itJet){
      const unsigned newEta = gtEta(itJet->hwEta());
      const uint16_t rankPt = params->jetScale().rank((uint16_t)itJet->hwPt());

      ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> > ldummy(0,0,0,0);

      l1t::Jet gtJet(*&ldummy, rankPt, newEta, itJet->hwPhi(), itJet->hwQual());
      output->push_back(gtJet);
    }
  }

  void EGammaToGtScales(CaloParamsStage1 *params,
			const std::vector<l1t::EGamma> * input,
			std::vector<l1t::EGamma> *output){

    for(std::vector<l1t::EGamma>::const_iterator itEGamma = input->begin();
	itEGamma != input->end(); ++itEGamma){
      const unsigned newEta = gtEta(itEGamma->hwEta());
      // const uint16_t rankPt = params->emScale().rank((uint16_t)itEGamma->hwPt());

      // LA  Hack till we get the right em scale from conditions DB
      const uint16_t rankPt = (uint16_t)itEGamma->hwPt();

      ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> > ldummy(0,0,0,0);

      l1t::EGamma gtEGamma(*&ldummy, rankPt, newEta, itEGamma->hwPhi(),
			   itEGamma->hwQual(), itEGamma->hwIso());
      output->push_back(gtEGamma);
    }
  }

  void TauToGtScales(CaloParamsStage1 *params,
		     const std::vector<l1t::Tau> * input,
		     std::vector<l1t::Tau> *output){
    for(std::vector<l1t::Tau>::const_iterator itTau = input->begin();
	itTau != input->end(); ++itTau){
      const unsigned newEta = gtEta(itTau->hwEta());
      const uint16_t rankPt = params->jetScale().rank((uint16_t)itTau->hwPt());

      ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> > ldummy(0,0,0,0);

      l1t::Tau gtTau(*&ldummy, rankPt, newEta, itTau->hwPhi(), itTau->hwQual());
      output->push_back(gtTau);
    }
  }

  void EtSumToGtScales(CaloParamsStage1 *params,
		       const std::vector<l1t::EtSum> * input,
		       std::vector<l1t::EtSum> *output){
    for(std::vector<l1t::EtSum>::const_iterator itEtSum = input->begin();
	itEtSum != input->end(); ++itEtSum){

      uint16_t rankPt;
      // Hack for now to make sure they come out with the right scale
      //rankPt = params->jetScale().rank((uint16_t)itEtSum->hwPt());
      rankPt = (uint16_t)itEtSum->hwPt();
	//if (EtSum::EtSumType::kMissingHt == itEtSum->getType())
	//rankPt = params->HtMissScale().rank((uint16_t)itEtSum->hwPt());

      ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> > ldummy(0,0,0,0);

      l1t::EtSum gtEtSum(*&ldummy, itEtSum->getType(), rankPt, 0,
			 itEtSum->hwPhi(), itEtSum->hwQual());

      output->push_back(gtEtSum);
    }
  }
}

const unsigned int gtEta(const unsigned int iEta)
{
  unsigned rctEta = (iEta<11 ? 10-iEta : iEta-11);
  return (((rctEta % 7) & 0x7) | (iEta<11 ? 0x8 : 0));
}
