#include "L1Trigger/Phase2L1ParticleFlow/interface/egamma/L1EGPuppiIsoAlgo.h"

using namespace l1ct;

L1EGPuppiIsoAlgo::L1EGPuppiIsoAlgo(const edm::ParameterSet& pSet)
    : config_(pSet.getParameter<std::string>("pfIsoType"),
              pSet.getParameter<double>("pfPtMin"),
              pSet.getParameter<double>("dZ"),
              pSet.getParameter<double>("dRMin"),
              pSet.getParameter<double>("dRMax"),
              pSet.getParameter<bool>("pfCandReuse")) {}

void L1EGPuppiIsoAlgo::run(const EGIsoObjsEmu& l1EGs,
                           const PuppiObjs& l1PFCands,
                           EGIsoObjsEmu& outL1EGs,
                           z0_t z0) const {
  outL1EGs.reserve(l1EGs.size());

  // make a list of pointers to PF candidates
  // the pointer will be removed from the list once the candidate has been used and the the module is configured to to so
  std::list<const PuppiObj*> workPFCands;
  std::list<const PuppiObj*> workPFCandsPV;
  for (const auto& l1PFCand : l1PFCands) {
    workPFCands.emplace_back(&l1PFCand);
    workPFCandsPV.emplace_back(&l1PFCand);
  }

  for (const auto& l1EG : l1EGs) {
    auto outL1EG(l1EG);
    iso_t iso = 0;
    iso_t isoPV = 0;
    if (!workPFCands.empty()) {
      iso = calcIso(l1EG, workPFCands);
      isoPV = calcIso(l1EG, workPFCandsPV, z0);
    }

    if (config_.pfIsoType_ == L1EGPuppiIsoAlgoConfig::kPFIso) {
      outL1EG.setHwIso(EGIsoObjEmu::IsoType::PfIso, iso);
      outL1EG.setHwIso(EGIsoObjEmu::IsoType::PfIsoPV, isoPV);
    } else {
      outL1EG.setHwIso(EGIsoObjEmu::IsoType::PuppiIso, iso);
      outL1EG.setHwIso(EGIsoObjEmu::IsoType::PuppiIsoPV, isoPV);
    }
    outL1EGs.emplace_back(outL1EG);
  }
}

void L1EGPuppiIsoAlgo::run(EGIsoObjsEmu& l1EGs, const PuppiObjs& l1PFCands, z0_t z0) const {
  // make a list of pointers to PF candidates
  // the pointer will be removed from the list once the candidate has been used and the the module is configured to to so
  std::list<const PuppiObj*> workPFCands;
  std::list<const PuppiObj*> workPFCandsPV;
  for (const auto& l1PFCand : l1PFCands) {
    workPFCands.emplace_back(&l1PFCand);
    workPFCandsPV.emplace_back(&l1PFCand);
  }

  for (auto& l1EG : l1EGs) {
    iso_t iso = 0;
    iso_t isoPV = 0;
    if (!workPFCands.empty()) {
      iso = calcIso(l1EG, workPFCands);
      isoPV = calcIso(l1EG, workPFCandsPV, z0);
    }

    if (config_.pfIsoType_ == L1EGPuppiIsoAlgoConfig::kPFIso) {
      l1EG.setHwIso(EGIsoObjEmu::IsoType::PfIso, iso);
      l1EG.setHwIso(EGIsoObjEmu::IsoType::PfIsoPV, isoPV);
    } else {
      l1EG.setHwIso(EGIsoObjEmu::IsoType::PuppiIso, iso);
      l1EG.setHwIso(EGIsoObjEmu::IsoType::PuppiIsoPV, isoPV);
    }
  }
}

void L1EGPuppiIsoAlgo::run(EGIsoEleObjsEmu& l1Eles, const PuppiObjs& l1PFCands) const {
  // make a list of pointers to PF candidates
  // the pointer will be removed from the list once the candidate has been used and the the module is configured to to so
  std::list<const PuppiObj*> workPFCands;
  for (const auto& l1PFCand : l1PFCands) {
    workPFCands.emplace_back(&l1PFCand);
  }

  for (auto& l1Ele : l1Eles) {
    iso_t iso = 0;
    if (!workPFCands.empty()) {
      iso = calcIso(l1Ele, workPFCands);
    }

    if (config_.pfIsoType_ == L1EGPuppiIsoAlgoConfig::kPFIso) {
      l1Ele.setHwIso(EGIsoEleObjEmu::IsoType::PfIso, iso);
    } else {
      l1Ele.setHwIso(EGIsoEleObjEmu::IsoType::PuppiIso, iso);
    }
  }
}

iso_t L1EGPuppiIsoAlgo::calcIso(const EGIsoObj& l1EG, std::list<const PuppiObj*>& workPFCands, z0_t z0) const {
  iso_t sumPt = 0;

  auto pfIt = workPFCands.cbegin();
  while (pfIt != workPFCands.cend()) {
    // use the PF candidate pT if it is within the cone and optional dz cut for charged PF candidates
    const auto workPFCand = *pfIt;
    z0_t pfCandZ0 = 0;
    if (workPFCand->hwId.charged()) {
      pfCandZ0 = workPFCand->hwZ0();
    }

    // calculate dz
    ap_int<z0_t::width + 1> dz = z0 - pfCandZ0;
    if (dz < 0) {
      dz = -dz;
    }

    if (workPFCand->intCharge() == 0 || (workPFCand->intCharge() != 0 && dz < config_.dZMax_)) {
      const auto dR2 = dr2_int(l1EG.hwEta, l1EG.hwPhi, workPFCand->hwEta, workPFCand->hwPhi);
      if (dR2 >= config_.dRMin2_ && dR2 < config_.dRMax2_ && workPFCand->hwPt >= config_.ptMin_) {
        sumPt += workPFCand->hwPt;
        // remove the candidate from the collection if the module is configured to not reuse them
        if (!config_.pfCandReuse_) {
          // this returns an iterator to the next element already so no need to increase here
          pfIt = workPFCands.erase(pfIt);
          continue;
        }
      }
    }
    ++pfIt;
  }

  return sumPt;
}
