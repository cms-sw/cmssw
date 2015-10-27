// legacyGtHelper.cc
// Authors: Alex Barbieri
//
// This is a collection of helper methods to make sure that
// the objects passed to the legacy GT are using the proper
// Et scales and eta coordinates.

#include "L1Trigger/L1TCalorimeter/interface/legacyGtHelper.h"

namespace l1t {

  void calibrateAndRankJets(CaloParamsStage1 *params,
			    const std::vector<l1t::Jet> * input,
			    std::vector<l1t::Jet> *output){

    for(std::vector<l1t::Jet>::const_iterator itJet = input->begin();
	itJet != input->end(); ++itJet){
      unsigned int pt = itJet->hwPt();
      if(pt > ((1<<10) -1) )
	pt = ((1<<10) -1);
      unsigned int eta = itJet->hwEta();
      if (eta>10){ // LUT is symmetric in eta. For eta>10 map to corresponding eta<10 bin
	int offset=2*(eta-10)-1;
	eta=eta-offset;
      }
      unsigned int lutAddress = (eta<<10)+pt;

      unsigned int rank = params->jetCalibrationLUT()->data(lutAddress);

      ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> > ldummy(0,0,0,0);
      l1t::Jet outJet(*&ldummy, rank, itJet->hwEta(), itJet->hwPhi(), itJet->hwQual());
      output->push_back(outJet);
    }
  }

  void calibrateAndRankTaus(CaloParamsStage1 *params,
			    const std::vector<l1t::Tau> * input,
			    std::vector<l1t::Tau> *output){

    for(std::vector<l1t::Tau>::const_iterator itTau = input->begin();
	itTau != input->end(); ++itTau){
      unsigned int pt = itTau->hwPt();
      if(pt > ((1<<10) -1) )
	pt = ((1<<10) -1);
      unsigned int eta = itTau->hwEta();
      unsigned int lutAddress = (eta<<10)+pt;

      unsigned int rank = params->tauCalibrationLUT()->data(lutAddress);

      ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> > ldummy(0,0,0,0);
      l1t::Tau outTau(*&ldummy, rank, itTau->hwEta(), itTau->hwPhi(), itTau->hwQual());
      output->push_back(outTau);
    }
  }


  void JetToGtEtaScales(CaloParamsStage1 *params,
			const std::vector<l1t::Jet> * input,
			std::vector<l1t::Jet> *output){

    for(std::vector<l1t::Jet>::const_iterator itJet = input->begin();
	itJet != input->end(); ++itJet){
      unsigned newPhi = itJet->hwPhi();
      unsigned newEta = gtEta(itJet->hwEta());

      // jets with hwQual & 10 ==10 are "padding" jets from a sort, set their eta and phi
      // to the max value
      if((itJet->hwQual() & 0x10) == 0x10)
      {
	newEta = 0x0;
	newPhi = 0x0;
      }

      ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> > ldummy(0,0,0,0);

      l1t::Jet gtJet(*&ldummy, itJet->hwPt(), newEta, newPhi, itJet->hwQual());
      output->push_back(gtJet);
    }
  }

  void JetToGtPtScales(CaloParamsStage1 *params,
			const std::vector<l1t::Jet> * input,
			std::vector<l1t::Jet> *output){

    for(std::vector<l1t::Jet>::const_iterator itJet = input->begin();
	itJet != input->end(); ++itJet){
      uint16_t linPt = (uint16_t)itJet->hwPt();
      if(linPt > params->jetScale().linScaleMax() ) linPt = params->jetScale().linScaleMax();
      const uint16_t rankPt = params->jetScale().rank(linPt);

      ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> > ldummy(0,0,0,0);

      l1t::Jet gtJet(*&ldummy, rankPt, itJet->hwEta(), itJet->hwPhi(), itJet->hwQual());
      output->push_back(gtJet);
    }
  }


  void EGammaToGtScales(CaloParamsStage1 *params,
			const std::vector<l1t::EGamma> * input,
			std::vector<l1t::EGamma> *output){

    for(std::vector<l1t::EGamma>::const_iterator itEGamma = input->begin();
	itEGamma != input->end(); ++itEGamma){
      unsigned newEta = gtEta(itEGamma->hwEta());
      unsigned newPhi = itEGamma->hwPhi();
      const uint16_t rankPt = (uint16_t)itEGamma->hwPt(); //max value?

      //hwQual &10 == 10 means that the object came from a sort and is padding
      if((itEGamma->hwQual() & 0x10) == 0x10)
      {
	newEta = 0x0;
	newPhi = 0x0;
      }

      ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> > ldummy(0,0,0,0);

      l1t::EGamma gtEGamma(*&ldummy, rankPt, newEta, newPhi,
			   itEGamma->hwQual(), itEGamma->hwIso());
      output->push_back(gtEGamma);
    }
  }

  void TauToGtEtaScales(CaloParamsStage1 *params,
			const std::vector<l1t::Tau> * input,
			std::vector<l1t::Tau> *output){
    for(std::vector<l1t::Tau>::const_iterator itTau = input->begin();
	itTau != input->end(); ++itTau){
      unsigned newPhi = itTau->hwPhi();
      unsigned newEta = gtEta(itTau->hwEta());

      // taus with hwQual & 10 ==10 are "padding" jets from a sort, set their eta and phi
      // to the max value
      if((itTau->hwQual() & 0x10) == 0x10)
      {
	newEta = 0x0;
	newPhi = 0x0;
      }


      ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> > ldummy(0,0,0,0);

      l1t::Tau gtTau(*&ldummy, itTau->hwPt(), newEta, newPhi, itTau->hwQual(), itTau->hwIso());
      output->push_back(gtTau);
    }
  }

  void TauToGtPtScales(CaloParamsStage1 *params,
		       const std::vector<l1t::Tau> * input,
		       std::vector<l1t::Tau> *output){
    for(std::vector<l1t::Tau>::const_iterator itTau = input->begin();
	itTau != input->end(); ++itTau){
      uint16_t linPt = (uint16_t)itTau->hwPt();
      if(linPt > params->jetScale().linScaleMax() ) linPt = params->jetScale().linScaleMax();
      const uint16_t rankPt = params->jetScale().rank(linPt);

      ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> > ldummy(0,0,0,0);

      l1t::Tau gtTau(*&ldummy, rankPt, itTau->hwEta(), itTau->hwPhi(), itTau->hwQual(), itTau->hwIso());
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
      if (EtSum::EtSumType::kMissingHt == itEtSum->getType())
      {
	// if(rankPt > params->HtMissScale().linScaleMax()) rankPt = params->HtMissScale().linScaleMax();
	// params->HtMissScale().linScaleMax() always returns zero.  Hardcode 512 for now

	// comment out for mht/ht (already in GT scale)
	//if(rankPt > 512) rankPt = 512;
	//rankPt = params->HtMissScale().rank(rankPt*params->emScale().linearLsb());
      }

      ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> > ldummy(0,0,0,0);

      l1t::EtSum gtEtSum(*&ldummy, itEtSum->getType(), rankPt, 0,
			 itEtSum->hwPhi(), itEtSum->hwQual());

      output->push_back(gtEtSum);
    }
  }

  const unsigned int gtEta(const unsigned int iEta)
  {
    unsigned rctEta = (iEta<11 ? 10-iEta : iEta-11);
    return (((rctEta % 7) & 0x7) | (iEta<11 ? 0x8 : 0));
  }
}
