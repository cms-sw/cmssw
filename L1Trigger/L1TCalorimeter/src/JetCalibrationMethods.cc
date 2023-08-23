// JetCalibrationMethods.cc
// Authors: Inga Bucinskaite, UIC
//
// This file should contain the different algorithms used for Jet Calibration
//

#include "L1Trigger/L1TCalorimeter/interface/JetCalibrationMethods.h"
#include <vector>

namespace l1t {

  void JetCalibration(std::vector<l1t::Jet>* uncalibjets,
                      std::vector<double> jetCalibrationParams,
                      std::vector<l1t::Jet>* jets,
                      std::string jetCalibrationType,
                      double jetLSB) {
    for (std::vector<l1t::Jet>::const_iterator uncalibjet = uncalibjets->begin(); uncalibjet != uncalibjets->end();
         ++uncalibjet) {
      if (jetCalibrationType == "None") {
        const l1t::Jet& corrjets = *uncalibjet;
        jets->push_back(corrjets);
        continue;
      }

      if (jetCalibrationType == "Stage1JEC") {
        int jetPt = (uncalibjet->hwPt()) * jetLSB;  // correction factors are parameterized as functions of physical pt
        int jetPhi = uncalibjet->hwPhi();
        int jetEta = uncalibjet->hwEta();
        int jetQual = uncalibjet->hwQual();
        double jpt = 0.0;

        double alpha = jetCalibrationParams[2 * jetEta + 0];      //Scale factor (See jetSF_cfi.py)
        double gamma = ((jetCalibrationParams[2 * jetEta + 1]));  //Offset

        jpt = jetPt * alpha + gamma;
        unsigned int corjetET = (int)(jpt / jetLSB);

        ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> > jetLorentz(0, 0, 0, 0);
        l1t::Jet corrjets(*&jetLorentz, corjetET, jetEta, jetPhi, jetQual);

        jets->push_back(corrjets);
      }
    }
  }
}  // namespace l1t
