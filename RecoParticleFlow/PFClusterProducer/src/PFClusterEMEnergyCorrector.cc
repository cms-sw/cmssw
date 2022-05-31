#include "RecoParticleFlow/PFClusterProducer/interface/PFClusterEMEnergyCorrector.h"

#include "vdt/vdtMath.h"
#include <array>

namespace {
  typedef reco::PFCluster::EEtoPSAssociation::value_type EEPSPair;
  bool sortByKey(const EEPSPair &a, const EEPSPair &b) { return a.first < b.first; }

  double getOffset(const double lo, const double hi) { return lo + 0.5 * (hi - lo); }
  double getScale(const double lo, const double hi) { return 0.5 * (hi - lo); }
}  // namespace

PFClusterEMEnergyCorrector::PFClusterEMEnergyCorrector(const edm::ParameterSet &conf, edm::ConsumesCollector &&cc)
    : ecalClusterToolsESGetTokens_{std::move(cc)},
      ecalReadoutToolsESGetTokens_{conf, std::move(cc)},
      calibrator_(new PFEnergyCalibration) {
  applyCrackCorrections_ = conf.getParameter<bool>("applyCrackCorrections");
  applyMVACorrections_ = conf.getParameter<bool>("applyMVACorrections");
  srfAwareCorrection_ = conf.getParameter<bool>("srfAwareCorrection");
  setEnergyUncertainty_ = conf.getParameter<bool>("setEnergyUncertainty");
  maxPtForMVAEvaluation_ = conf.getParameter<double>("maxPtForMVAEvaluation");

  if (applyMVACorrections_) {
    meanlimlowEB_ = -0.336;
    meanlimhighEB_ = 0.916;
    meanoffsetEB_ = getOffset(meanlimlowEB_, meanlimhighEB_);
    meanscaleEB_ = getScale(meanlimlowEB_, meanlimhighEB_);

    meanlimlowEE_ = -0.336;
    meanlimhighEE_ = 0.916;
    meanoffsetEE_ = getOffset(meanlimlowEE_, meanlimhighEE_);
    meanscaleEE_ = getScale(meanlimlowEE_, meanlimhighEE_);

    sigmalimlowEB_ = 0.001;
    sigmalimhighEB_ = 0.4;
    sigmaoffsetEB_ = getOffset(sigmalimlowEB_, sigmalimhighEB_);
    sigmascaleEB_ = getScale(sigmalimlowEB_, sigmalimhighEB_);

    sigmalimlowEE_ = 0.001;
    sigmalimhighEE_ = 0.4;
    sigmaoffsetEE_ = getOffset(sigmalimlowEE_, sigmalimhighEE_);
    sigmascaleEE_ = getScale(sigmalimlowEE_, sigmalimhighEE_);

    recHitsEB_ = cc.consumes<EcalRecHitCollection>(conf.getParameter<edm::InputTag>("recHitsEBLabel"));
    recHitsEE_ = cc.consumes<EcalRecHitCollection>(conf.getParameter<edm::InputTag>("recHitsEELabel"));
    autoDetectBunchSpacing_ = conf.getParameter<bool>("autoDetectBunchSpacing");

    if (autoDetectBunchSpacing_) {
      bunchSpacing_ = cc.consumes<unsigned int>(edm::InputTag("bunchSpacingProducer"));
      bunchSpacingManual_ = 0;
    } else {
      bunchSpacingManual_ = conf.getParameter<int>("bunchSpacing");
    }

    condnames_mean_25ns_.insert(condnames_mean_25ns_.end(),
                                {"ecalPFClusterCorV2_EB_pfSize1_mean_25ns",
                                 "ecalPFClusterCorV2_EB_pfSize2_mean_25ns",
                                 "ecalPFClusterCorV2_EB_pfSize3_ptbin1_mean_25ns",
                                 "ecalPFClusterCorV2_EB_pfSize3_ptbin2_mean_25ns",
                                 "ecalPFClusterCorV2_EB_pfSize3_ptbin3_mean_25ns",
                                 "ecalPFClusterCorV2_EE_pfSize1_mean_25ns",
                                 "ecalPFClusterCorV2_EE_pfSize2_mean_25ns",
                                 "ecalPFClusterCorV2_EE_pfSize3_ptbin1_mean_25ns",
                                 "ecalPFClusterCorV2_EE_pfSize3_ptbin2_mean_25ns",
                                 "ecalPFClusterCorV2_EE_pfSize3_ptbin3_mean_25ns"});
    condnames_sigma_25ns_.insert(condnames_sigma_25ns_.end(),
                                 {"ecalPFClusterCorV2_EB_pfSize1_sigma_25ns",
                                  "ecalPFClusterCorV2_EB_pfSize2_sigma_25ns",
                                  "ecalPFClusterCorV2_EB_pfSize3_ptbin1_sigma_25ns",
                                  "ecalPFClusterCorV2_EB_pfSize3_ptbin2_sigma_25ns",
                                  "ecalPFClusterCorV2_EB_pfSize3_ptbin3_sigma_25ns",
                                  "ecalPFClusterCorV2_EE_pfSize1_sigma_25ns",
                                  "ecalPFClusterCorV2_EE_pfSize2_sigma_25ns",
                                  "ecalPFClusterCorV2_EE_pfSize3_ptbin1_sigma_25ns",
                                  "ecalPFClusterCorV2_EE_pfSize3_ptbin2_sigma_25ns",
                                  "ecalPFClusterCorV2_EE_pfSize3_ptbin3_sigma_25ns"});
    condnames_mean_50ns_.insert(condnames_mean_50ns_.end(),
                                {"ecalPFClusterCorV2_EB_pfSize1_mean_50ns",
                                 "ecalPFClusterCorV2_EB_pfSize2_mean_50ns",
                                 "ecalPFClusterCorV2_EB_pfSize3_ptbin1_mean_50ns",
                                 "ecalPFClusterCorV2_EB_pfSize3_ptbin2_mean_50ns",
                                 "ecalPFClusterCorV2_EB_pfSize3_ptbin3_mean_50ns",
                                 "ecalPFClusterCorV2_EE_pfSize1_mean_50ns",
                                 "ecalPFClusterCorV2_EE_pfSize2_mean_50ns",
                                 "ecalPFClusterCorV2_EE_pfSize3_ptbin1_mean_50ns",
                                 "ecalPFClusterCorV2_EE_pfSize3_ptbin2_mean_50ns",
                                 "ecalPFClusterCorV2_EE_pfSize3_ptbin3_mean_50ns"});
    condnames_sigma_50ns_.insert(condnames_sigma_50ns_.end(),
                                 {"ecalPFClusterCorV2_EB_pfSize1_sigma_50ns",
                                  "ecalPFClusterCorV2_EB_pfSize2_sigma_50ns",
                                  "ecalPFClusterCorV2_EB_pfSize3_ptbin1_sigma_50ns",
                                  "ecalPFClusterCorV2_EB_pfSize3_ptbin2_sigma_50ns",
                                  "ecalPFClusterCorV2_EB_pfSize3_ptbin3_sigma_50ns",
                                  "ecalPFClusterCorV2_EE_pfSize1_sigma_50ns",
                                  "ecalPFClusterCorV2_EE_pfSize2_sigma_50ns",
                                  "ecalPFClusterCorV2_EE_pfSize3_ptbin1_sigma_50ns",
                                  "ecalPFClusterCorV2_EE_pfSize3_ptbin2_sigma_50ns",
                                  "ecalPFClusterCorV2_EE_pfSize3_ptbin3_sigma_50ns"});

    if (srfAwareCorrection_) {
      sigmalimlowEE_ = 0.001;
      sigmalimhighEE_ = 0.1;
      sigmaoffsetEE_ = getOffset(sigmalimlowEE_, sigmalimhighEE_);
      sigmascaleEE_ = getScale(sigmalimlowEE_, sigmalimhighEE_);

      ebSrFlagToken_ = cc.consumes<EBSrFlagCollection>(conf.getParameter<edm::InputTag>("ebSrFlagLabel"));
      eeSrFlagToken_ = cc.consumes<EESrFlagCollection>(conf.getParameter<edm::InputTag>("eeSrFlagLabel"));

      condnames_mean_.insert(condnames_mean_.end(),
                             {"ecalPFClusterCor2017V2_EB_ZS_mean_25ns",
                              "ecalPFClusterCor2017V2_EB_Full_ptbin1_mean_25ns",
                              "ecalPFClusterCor2017V2_EB_Full_ptbin2_mean_25ns",
                              "ecalPFClusterCor2017V2_EB_Full_ptbin3_mean_25ns",
                              "ecalPFClusterCor2017V2_EE_ZS_mean_25ns",
                              "ecalPFClusterCor2017V2_EE_Full_ptbin1_mean_25ns",
                              "ecalPFClusterCor2017V2_EE_Full_ptbin2_mean_25ns",
                              "ecalPFClusterCor2017V2_EE_Full_ptbin3_mean_25ns"});

      condnames_sigma_.insert(condnames_sigma_.end(),
                              {"ecalPFClusterCor2017V2_EB_ZS_sigma_25ns",
                               "ecalPFClusterCor2017V2_EB_Full_ptbin1_sigma_25ns",
                               "ecalPFClusterCor2017V2_EB_Full_ptbin2_sigma_25ns",
                               "ecalPFClusterCor2017V2_EB_Full_ptbin3_sigma_25ns",
                               "ecalPFClusterCor2017V2_EE_ZS_sigma_25ns",
                               "ecalPFClusterCor2017V2_EE_Full_ptbin1_sigma_25ns",
                               "ecalPFClusterCor2017V2_EE_Full_ptbin2_sigma_25ns",
                               "ecalPFClusterCor2017V2_EE_Full_ptbin3_sigma_25ns"});

      for (short i = 0; i < (short)condnames_mean_.size(); i++) {
        forestMeanTokens_25ns_.emplace_back(cc.esConsumes(edm::ESInputTag("", condnames_mean_[i])));
        forestSigmaTokens_25ns_.emplace_back(cc.esConsumes(edm::ESInputTag("", condnames_sigma_[i])));
      }
    } else {
      for (short i = 0; i < (short)condnames_mean_25ns_.size(); i++) {
        forestMeanTokens_25ns_.emplace_back(cc.esConsumes(edm::ESInputTag("", condnames_mean_25ns_[i])));
        forestSigmaTokens_25ns_.emplace_back(cc.esConsumes(edm::ESInputTag("", condnames_sigma_25ns_[i])));
      }
      for (short i = 0; i < (short)condnames_mean_50ns_.size(); i++) {
        forestMeanTokens_50ns_.emplace_back(cc.esConsumes(edm::ESInputTag("", condnames_mean_50ns_[i])));
        forestSigmaTokens_50ns_.emplace_back(cc.esConsumes(edm::ESInputTag("", condnames_sigma_50ns_[i])));
      }
    }
  }
}

void PFClusterEMEnergyCorrector::correctEnergies(const edm::Event &evt,
                                                 const edm::EventSetup &es,
                                                 const reco::PFCluster::EEtoPSAssociation &assoc,
                                                 reco::PFClusterCollection &cs) {
  // First deal with pre-MVA corrections
  // Kept for backward compatibility (and for HLT)
  if (!applyMVACorrections_) {
    for (unsigned int idx = 0; idx < cs.size(); ++idx) {
      reco::PFCluster &cluster = cs[idx];
      bool iseb = cluster.layer() == PFLayer::ECAL_BARREL;
      float ePS1 = 0., ePS2 = 0.;
      if (!iseb)
        getAssociatedPSEnergy(idx, assoc, ePS1, ePS2);
      double correctedEnergy = calibrator_->energyEm(cluster, ePS1, ePS2, applyCrackCorrections_);
      cluster.setCorrectedEnergy(correctedEnergy);
    }
    return;
  }

  // Common objects for SRF-aware and old style corrections
  EcalClusterLazyTools lazyTool(evt, ecalClusterToolsESGetTokens_.get(es), recHitsEB_, recHitsEE_);
  EcalReadoutTools readoutTool(evt, es, ecalReadoutToolsESGetTokens_);

  if (!srfAwareCorrection_) {
    int bunchspacing = 450;
    if (autoDetectBunchSpacing_) {
      edm::Handle<unsigned int> bunchSpacingH;
      evt.getByToken(bunchSpacing_, bunchSpacingH);
      bunchspacing = *bunchSpacingH;
    } else {
      bunchspacing = bunchSpacingManual_;
    }

    const unsigned int ncor = (bunchspacing == 25) ? condnames_mean_25ns_.size() : condnames_mean_50ns_.size();

    std::vector<edm::ESHandle<GBRForestD> > forestH_mean(ncor);
    std::vector<edm::ESHandle<GBRForestD> > forestH_sigma(ncor);

    if (bunchspacing == 25) {
      for (unsigned int icor = 0; icor < ncor; ++icor) {
        forestH_mean[icor] = es.getHandle(forestMeanTokens_25ns_[icor]);
        forestH_sigma[icor] = es.getHandle(forestSigmaTokens_25ns_[icor]);
      }
    } else {
      for (unsigned int icor = 0; icor < ncor; ++icor) {
        forestH_mean[icor] = es.getHandle(forestMeanTokens_50ns_[icor]);
        forestH_sigma[icor] = es.getHandle(forestSigmaTokens_50ns_[icor]);
      }
    }

    std::array<float, 5> eval;
    for (unsigned int idx = 0; idx < cs.size(); ++idx) {
      reco::PFCluster &cluster = cs[idx];
      bool iseb = cluster.layer() == PFLayer::ECAL_BARREL;
      float ePS1 = 0., ePS2 = 0.;
      if (!iseb)
        getAssociatedPSEnergy(idx, assoc, ePS1, ePS2);

      double e = cluster.energy();
      double pt = cluster.pt();
      double evale = e;
      if (maxPtForMVAEvaluation_ > 0. && pt > maxPtForMVAEvaluation_) {
        evale *= maxPtForMVAEvaluation_ / pt;
      }
      double invE = (e == 0.) ? 0. : 1. / e;  //guard against dividing by 0.
      int size = lazyTool.n5x5(cluster);

      int ietaix = 0;
      int iphiiy = 0;
      if (iseb) {
        EBDetId ebseed(cluster.seed());
        ietaix = ebseed.ieta();
        iphiiy = ebseed.iphi();
      } else {
        EEDetId eeseed(cluster.seed());
        ietaix = eeseed.ix();
        iphiiy = eeseed.iy();
      }

      //find index of corrections (0-4 for EB, 5-9 for EE, depending on cluster size and raw pt)
      int coridx = std::min(size, 3) - 1;
      if (coridx == 2) {
        if (pt > 4.5) {
          coridx += 1;
        }
        if (pt > 18.) {
          coridx += 1;
        }
      }
      if (!iseb) {
        coridx += 5;
      }

      const GBRForestD &meanforest = *forestH_mean[coridx].product();
      const GBRForestD &sigmaforest = *forestH_sigma[coridx].product();

      //fill array for forest evaluation
      eval[0] = evale;
      eval[1] = ietaix;
      eval[2] = iphiiy;
      if (!iseb) {
        eval[3] = ePS1 * invE;
        eval[4] = ePS2 * invE;
      }

      //these are the actual BDT responses
      double rawmean = meanforest.GetResponse(eval.data());
      double rawsigma = sigmaforest.GetResponse(eval.data());

      //apply transformation to limited output range (matching the training)
      double mean = iseb ? meanoffsetEB_ + meanscaleEB_ * vdt::fast_sin(rawmean)
                         : meanoffsetEE_ + meanscaleEE_ * vdt::fast_sin(rawmean);
      double sigma = iseb ? sigmaoffsetEB_ + sigmascaleEB_ * vdt::fast_sin(rawsigma)
                          : sigmaoffsetEE_ + sigmascaleEE_ * vdt::fast_sin(rawsigma);

      //regression target is ln(Etrue/Eraw)
      //so corrected energy is ecor=exp(mean)*e, uncertainty is exp(mean)*eraw*sigma=ecor*sigma
      double ecor = vdt::fast_exp(mean) * e;
      double sigmacor = sigma * ecor;

      cluster.setCorrectedEnergy(ecor);
      if (setEnergyUncertainty_)
        cluster.setCorrectedEnergyUncertainty(sigmacor);
      else
        cluster.setCorrectedEnergyUncertainty(0.);
    }
    return;
  }

  // Selective Readout Flags
  edm::Handle<EBSrFlagCollection> ebSrFlags;
  edm::Handle<EESrFlagCollection> eeSrFlags;
  evt.getByToken(ebSrFlagToken_, ebSrFlags);
  evt.getByToken(eeSrFlagToken_, eeSrFlags);
  if (not ebSrFlags.isValid() or not eeSrFlags.isValid())
    edm::LogInfo("PFClusterEMEnergyCorrector") << "SrFlagCollection information is not available. The ECAL PFCluster "
                                                  "corrections will assume \"full readout\" for all hits.";

  const unsigned int ncor = forestMeanTokens_25ns_.size();
  std::vector<edm::ESHandle<GBRForestD> > forestH_mean(ncor);
  std::vector<edm::ESHandle<GBRForestD> > forestH_sigma(ncor);

  for (unsigned int icor = 0; icor < ncor; ++icor) {
    forestH_mean[icor] = es.getHandle(forestMeanTokens_25ns_[icor]);
    forestH_sigma[icor] = es.getHandle(forestSigmaTokens_25ns_[icor]);
  }

  std::array<float, 6> evalEB;
  std::array<float, 5> evalEE;

  for (unsigned int idx = 0; idx < cs.size(); ++idx) {
    reco::PFCluster &cluster = cs[idx];
    bool iseb = cluster.layer() == PFLayer::ECAL_BARREL;
    float ePS1 = 0., ePS2 = 0.;
    if (!iseb)
      getAssociatedPSEnergy(idx, assoc, ePS1, ePS2);

    double e = cluster.energy();
    double pt = cluster.pt();
    double evale = e;
    if (maxPtForMVAEvaluation_ > 0. && pt > maxPtForMVAEvaluation_) {
      evale *= maxPtForMVAEvaluation_ / pt;
    }
    double invE = (e == 0.) ? 0. : 1. / e;  //guard against dividing by 0.
    int size = lazyTool.n5x5(cluster);
    int reducedHits = size;
    if (size >= 3)
      reducedHits = 3;

    int ietaix = 0;
    int iphiiy = 0;
    if (iseb) {
      EBDetId ebseed(cluster.seed());
      ietaix = ebseed.ieta();
      iphiiy = ebseed.iphi();
    } else {
      EEDetId eeseed(cluster.seed());
      ietaix = eeseed.ix();
      iphiiy = eeseed.iy();
    }

    // Hardcoded number are positions of modules boundaries of ECAL
    int signeta = (ietaix > 0) ? 1 : -1;
    int ietamod20 = (std::abs(ietaix) < 26) ? ietaix - signeta : (ietaix - 26 * signeta) % 20;
    int iphimod20 = (iphiiy - 1) % 20;

    // Assume that hits for which no information is avaiable have a Full Readout (binary 0011)
    int clusFlag = 3;
    if (iseb) {
      if (ebSrFlags.isValid()) {
        auto ecalUnit = readoutTool.readOutUnitOf(static_cast<EBDetId>(cluster.seed()));
        EBSrFlagCollection::const_iterator srf = ebSrFlags->find(ecalUnit);
        if (srf != ebSrFlags->end())
          clusFlag = srf->value();
      }
    } else {
      if (eeSrFlags.isValid()) {
        auto ecalUnit = readoutTool.readOutUnitOf(static_cast<EEDetId>(cluster.seed()));
        EESrFlagCollection::const_iterator srf = eeSrFlags->find(ecalUnit);
        if (srf != eeSrFlags->end())
          clusFlag = srf->value();
      }
    }

    // Find index of corrections (0-3 for EB, 4-7 for EE, depending on cluster size and raw pt)
    int coridx = 0;
    int regind = 0;
    if (!iseb)
      regind = 4;

    ///////////////////////////////////////////////////////////////////////////////////
    ///a hit can be ZS or forced ZS. A hit can be in Full readout or Forced to be FULL readout
    ///if it is ZS, then clusFlag (in binary) = 0001
    ///if it is forced ZS, then clusFlag (in binary) = 0101
    ///if it is FR, then clusFlag (in binary) = 0011
    ///if it is forced FR, then clusFlag (in binary) = 0111
    ///i.e 3rd bit is set.
    ///Even if it is forced, we should mark it is as ZS or FR. To take care of it, just check the LSB and second LSB(SLSB)
    ///////////////////////////////////////////////////////////////////////////////////
    int ZS_bit = clusFlag >> 0 & 1;
    int FR_bit = clusFlag >> 1 & 1;

    if (ZS_bit != 0 && FR_bit == 0)  ///it is clusFlag==1, 5
      coridx = 0 + regind;
    else {
      if (pt < 2.5)
        coridx = 1 + regind;
      else if (pt >= 2.5 && pt < 6.)
        coridx = 2 + regind;
      else if (pt >= 6.)
        coridx = 3 + regind;
    }
    if (ZS_bit == 0 || clusFlag > 7) {
      edm::LogWarning("PFClusterEMEnergyCorrector")
          << "We can only correct regions readout in ZS (flag 1,5) or FULL readout (flag 3,7). Flag " << clusFlag
          << " is not recognized."
          << "\n"
          << "Assuming FULL readout and continuing";
    }

    const GBRForestD &meanforest = *forestH_mean[coridx].product();
    const GBRForestD &sigmaforest = *forestH_sigma[coridx].product();

    //fill array for forest evaluation
    if (iseb) {
      evalEB[0] = evale;
      evalEB[1] = ietaix;
      evalEB[2] = iphiiy;
      evalEB[3] = ietamod20;
      evalEB[4] = iphimod20;
      evalEB[5] = reducedHits;
    } else {
      evalEE[0] = evale;
      evalEE[1] = ietaix;
      evalEE[2] = iphiiy;
      evalEE[3] = (ePS1 + ePS2) * invE;
      evalEE[4] = reducedHits;
    }

    //these are the actual BDT responses
    double rawmean = 1;
    double rawsigma = 0;

    if (iseb) {
      rawmean = meanforest.GetResponse(evalEB.data());
      rawsigma = sigmaforest.GetResponse(evalEB.data());
    } else {
      rawmean = meanforest.GetResponse(evalEE.data());
      rawsigma = sigmaforest.GetResponse(evalEE.data());
    }

    //apply transformation to limited output range (matching the training)
    //the training was done with different transformations for EB and EE (width only)
    //makes a the code a bit more cumbersome, but it is not a problem per se
    double mean = iseb ? meanoffsetEB_ + meanscaleEB_ * vdt::fast_sin(rawmean)
                       : meanoffsetEE_ + meanscaleEE_ * vdt::fast_sin(rawmean);
    double sigma = iseb ? sigmaoffsetEB_ + sigmascaleEB_ * vdt::fast_sin(rawsigma)
                        : sigmaoffsetEE_ + sigmascaleEE_ * vdt::fast_sin(rawsigma);

    //regression target is ln(Etrue/Eraw)
    //so corrected energy is ecor=exp(mean)*e, uncertainty is exp(mean)*eraw*sigma=ecor*sigma
    double ecor = iseb ? vdt::fast_exp(mean) * e : vdt::fast_exp(mean) * (e + ePS1 + ePS2);
    double sigmacor = sigma * ecor;

    LogDebug("PFClusterEMEnergyCorrector")
        << "ieta : iphi : ietamod20 : iphimod20 : size : reducedHits = " << ietaix << " " << iphiiy << " " << ietamod20
        << " " << iphimod20 << " " << size << " " << reducedHits << "\n"
        << "isEB : eraw : ePS1 : ePS2 : (eps1+eps2)/raw : Flag = " << iseb << " " << evale << " " << ePS1 << " " << ePS2
        << " " << (ePS1 + ePS2) / evale << " " << clusFlag << "\n"
        << "response : correction = " << exp(mean) << " " << ecor;

    cluster.setCorrectedEnergy(ecor);
    if (setEnergyUncertainty_)
      cluster.setCorrectedEnergyUncertainty(sigmacor);
    else
      cluster.setCorrectedEnergyUncertainty(0.);
  }
}

void PFClusterEMEnergyCorrector::getAssociatedPSEnergy(const size_t clusIdx,
                                                       const reco::PFCluster::EEtoPSAssociation &assoc,
                                                       float &e1,
                                                       float &e2) {
  e1 = 0;
  e2 = 0;
  auto ee_key_val = std::make_pair(clusIdx, edm::Ptr<reco::PFCluster>());
  const auto clustops = std::equal_range(assoc.begin(), assoc.end(), ee_key_val, sortByKey);
  for (auto i_ps = clustops.first; i_ps != clustops.second; ++i_ps) {
    edm::Ptr<reco::PFCluster> psclus(i_ps->second);
    switch (psclus->layer()) {
      case PFLayer::PS1:
        e1 += psclus->energy();
        break;
      case PFLayer::PS2:
        e2 += psclus->energy();
        break;
      default:
        break;
    }
  }
}
