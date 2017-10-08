///
/// Description: Firmware headers
///
/// Implementation:
/// Collects concrete algorithm implmentations.
///
/// \author: R. Alex Barbieri MIT
///          Kalanand Mishra, Fermilab
///

//
// This header file contains the class definitions for all of the concrete
// implementations of the firmware interface. The Stage1Layer2FirmwareFactory
// selects the appropriate implementation based on the firmware version in the
// configuration.
//

#ifndef L1TCALOSTAGE1EGAMMAALGORITHMIMP_H
#define L1TCALOSTAGE1EGAMMAALGORITHMIMP_H

#include "L1Trigger/L1TCalorimeter/interface/Stage1Layer2EGammaAlgorithm.h"
#include "L1Trigger/L1TCalorimeter/interface/CaloParamsHelper.h"

namespace l1t {

  class Stage1Layer2EGammaAlgorithmImpPP : public Stage1Layer2EGammaAlgorithm {
  public:
    Stage1Layer2EGammaAlgorithmImpPP(CaloParamsHelper* params);
    ~Stage1Layer2EGammaAlgorithmImpPP() override;
    void processEvent(const std::vector<l1t::CaloEmCand> & EMCands,
			      const std::vector<l1t::CaloRegion> & regions,
			      const std::vector<l1t::Jet> * jets,
			      std::vector<l1t::EGamma>* egammas) override;
  private:
    CaloParamsHelper* const params_;
    double Isolation(int ieta, int iphi,
    		     const std::vector<l1t::CaloRegion> & regions)  const;
    double HoverE(int et, int ieta, int iphi,
    		  const std::vector<l1t::CaloRegion> & regions)  const;
    int AssociatedJetPt(int ieta, int iphi,
		           const std::vector<l1t::Jet> * jets) const;

    unsigned isoLutIndex(unsigned int etaPt,unsigned int jetPt) const;
  };

  class Stage1Layer2EGammaAlgorithmImpHI : public Stage1Layer2EGammaAlgorithm {
  public:
    Stage1Layer2EGammaAlgorithmImpHI(CaloParamsHelper* params);
    ~Stage1Layer2EGammaAlgorithmImpHI() override;
    void processEvent(const std::vector<l1t::CaloEmCand> & EMCands,
			      const std::vector<l1t::CaloRegion> & regions,
			      const std::vector<l1t::Jet> * jets,
			      std::vector<l1t::EGamma>* egammas) override;
  private:
    CaloParamsHelper* const params_;
    double Isolation(int ieta, int iphi,
    		     const std::vector<l1t::CaloRegion> & regions)  const;
    double HoverE(int et, int ieta, int iphi,
    		  const std::vector<l1t::CaloRegion> & regions)  const;
    int AssociatedJetPt(int ieta, int iphi,
		           const std::vector<l1t::Jet> * jets) const;

    unsigned isoLutIndex(unsigned int etaPt,unsigned int jetPt) const;
  };

  class Stage1Layer2EGammaAlgorithmImpHW : public Stage1Layer2EGammaAlgorithm {
  public:
    Stage1Layer2EGammaAlgorithmImpHW(CaloParamsHelper* params);
    ~Stage1Layer2EGammaAlgorithmImpHW() override;
    void processEvent(const std::vector<l1t::CaloEmCand> & EMCands,
			      const std::vector<l1t::CaloRegion> & regions,
			      const std::vector<l1t::Jet> * jets,
			      std::vector<l1t::EGamma>* egammas) override;
  private:
    CaloParamsHelper* const params_;
    int AssociatedJetPt(int ieta, int iphi,
		           const std::vector<l1t::Jet> * jets) const;

    unsigned isoLutIndex(unsigned int etaPt,unsigned int jetPt) const;
  };

}

#endif
