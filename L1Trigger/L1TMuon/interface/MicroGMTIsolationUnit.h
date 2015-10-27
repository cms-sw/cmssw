#ifndef __l1microgmtisolationunit_h
#define __l1microgmtisolationunit_h

#include "MicroGMTConfiguration.h"
#include "MicroGMTExtrapolationLUT.h"
#include "MicroGMTRelativeIsolationCheckLUT.h"
#include "MicroGMTAbsoluteIsolationCheckLUT.h"
#include "MicroGMTCaloIndexSelectionLUT.h"


namespace l1t {
  class MicroGMTIsolationUnit {
    public:
      explicit MicroGMTIsolationUnit (const edm::ParameterSet&);
      virtual ~MicroGMTIsolationUnit ();

      // returns the index corresponding to the calo tower sum using the LUT
      int getCaloIndex(MicroGMTConfiguration::InterMuon&) const;
      // copies the energy values to the m_towerEnergies map for consistent access
      void setTowerSums(const MicroGMTConfiguration::CaloInputCollection& inputs, int bx);
      // First step done for calo input preparation, calculates strip sums
      void calculate5by1Sums(const MicroGMTConfiguration::CaloInputCollection&, int bx);
      // Second step, only done for the sums needed for final iso requirement
      int calculate5by5Sum(unsigned index) const;

      // Checks with LUT isolation for all muons in list, assuming input calo is non-summed
      void isolate(MicroGMTConfiguration::InterMuonList&) const;
      // Checks with LUT isolation for all muons in list, assuming input calo is pre-summed
      void isolatePreSummed(MicroGMTConfiguration::InterMuonList& muons) const;
      // Uses *Extrapolation LUTs to project trajectory to the vertex and adds info to muon
      void extrapolateMuons(MicroGMTConfiguration::InterMuonList&) const;

    private:
      MicroGMTExtrapolationLUT m_BEtaExtrapolation;
      MicroGMTExtrapolationLUT m_BPhiExtrapolation;
      MicroGMTExtrapolationLUT m_OEtaExtrapolation;
      MicroGMTExtrapolationLUT m_OPhiExtrapolation;
      MicroGMTExtrapolationLUT m_FEtaExtrapolation;
      MicroGMTExtrapolationLUT m_FPhiExtrapolation;

      std::map<tftype, MicroGMTExtrapolationLUT*> m_phiExtrapolationLUTs;
      std::map<tftype, MicroGMTExtrapolationLUT*> m_etaExtrapolationLUTs;

      MicroGMTCaloIndexSelectionLUT m_IdxSelMemEta;
      MicroGMTCaloIndexSelectionLUT m_IdxSelMemPhi;

      MicroGMTRelativeIsolationCheckLUT m_RelIsoCheckMem;
      MicroGMTAbsoluteIsolationCheckLUT m_AbsIsoCheckMem;

      std::vector<int> m_5by1TowerSums;
      std::map<int, int> m_towerEnergies;
      bool m_initialSums;
  };
}

#endif /* defined(__l1microgmtisolationunit_h) */
