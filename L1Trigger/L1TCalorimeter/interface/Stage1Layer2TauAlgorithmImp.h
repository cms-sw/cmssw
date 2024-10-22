#ifndef Stage1Layer2TauAlgorithmImp_h
#define Stage1Layer2TauAlgorithmImp_h

#include "L1Trigger/L1TCalorimeter/interface/Stage1Layer2TauAlgorithm.h"
#include "L1Trigger/L1TCalorimeter/interface/Stage1Layer2JetAlgorithmImp.h"
#include "L1Trigger/L1TCalorimeter/interface/CaloParamsHelper.h"

#include "L1Trigger/L1TCalorimeter/interface/Stage1TauIsolationLUT.h"

#include <memory>
//	This is the implementation of the Stage1Layer2TauAlgorithm abstract base class.
//	This class will be used to find sngle high pt tracks in heavy ion collisions.

namespace l1t {

  class Stage1Layer2SingleTrackHI : public Stage1Layer2TauAlgorithm {
  public:
    Stage1Layer2SingleTrackHI(CaloParamsHelper const* params);
    ~Stage1Layer2SingleTrackHI() override = default;
    void processEvent(const std::vector<l1t::CaloEmCand>& clusters,
                      const std::vector<l1t::CaloRegion>& regions,
                      std::vector<l1t::Tau>* isoTaus,
                      std::vector<l1t::Tau>* taus) override;

  private:
    CaloParamsHelper const* const params_;
  };

  class Stage1Layer2TauAlgorithmImpPP : public Stage1Layer2TauAlgorithm {
  public:
    Stage1Layer2TauAlgorithmImpPP(CaloParamsHelper const* params);
    ~Stage1Layer2TauAlgorithmImpPP() override = default;
    void processEvent(const std::vector<l1t::CaloEmCand>& EMCands,
                      const std::vector<l1t::CaloRegion>& regions,
                      std::vector<l1t::Tau>* isoTaus,
                      std::vector<l1t::Tau>* taus) override;

  private:
    CaloParamsHelper const* const params_;

    string findNESW(int ieta, int iphi, int neta, int nphi) const;

    double JetIsolation(int et, int ieta, int iphi, const std::vector<l1t::Jet>& jets) const;

    unsigned isoLutIndex(unsigned int tauPt, unsigned int jetPt) const;

    int AssociatedJetPt(int ieta, int iphi, const std::vector<l1t::Jet>* jets) const;
  };

  class Stage1Layer2TauAlgorithmImpHW : public Stage1Layer2TauAlgorithm {
  public:
    Stage1Layer2TauAlgorithmImpHW(CaloParamsHelper const* params);
    ~Stage1Layer2TauAlgorithmImpHW() override = default;
    void processEvent(const std::vector<l1t::CaloEmCand>& EMCands,
                      const std::vector<l1t::CaloRegion>& regions,
                      std::vector<l1t::Tau>* isoTaus,
                      std::vector<l1t::Tau>* taus) override;

  private:
    CaloParamsHelper const* const params_;
    std::unique_ptr<Stage1TauIsolationLUT> isoTauLut;

    string findNESW(int ieta, int iphi, int neta, int nphi) const;

    double JetIsolation(int et, int ieta, int iphi, const std::vector<l1t::Jet>& jets) const;

    unsigned isoLutIndex(unsigned int tauPt, unsigned int jetPt) const;

    int AssociatedJetPt(int ieta, int iphi, const std::vector<l1t::Jet>* jets) const;
  };
}  // namespace l1t
#endif
