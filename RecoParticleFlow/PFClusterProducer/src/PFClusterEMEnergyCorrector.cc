#include "RecoParticleFlow/PFClusterProducer/interface/PFClusterEMEnergyCorrector.h"

#include "vdt/vdtMath.h"
#include <array>

namespace {
  typedef reco::PFCluster::EEtoPSAssociation::value_type EEPSPair;
  bool sortByKey(const EEPSPair& a, const EEPSPair& b) {
    return a.first < b.first;
  } 
}

PFClusterEMEnergyCorrector::PFClusterEMEnergyCorrector(const edm::ParameterSet& conf, edm::ConsumesCollector &&cc) :
  calibrator_(new PFEnergyCalibration) {


   applyCrackCorrections_ = conf.getParameter<bool>("applyCrackCorrections");
   applyMVACorrections_ = conf.getParameter<bool>("applyMVACorrections");

   maxPtForMVAEvaluation_ = conf.getParameter<double>("maxPtForMVAEvaluation");
  
   ebSrFlagToken_ = cc.consumes<EBSrFlagCollection>(conf.getParameter<edm::InputTag>("ebSrFlagLabel"));
   eeSrFlagToken_ = cc.consumes<EESrFlagCollection>(conf.getParameter<edm::InputTag>("eeSrFlagLabel"));
   
   recHitsEB_ = cc.consumes<EcalRecHitCollection>(conf.getParameter<edm::InputTag>("recHitsEBLabel"));
   recHitsEE_ = cc.consumes<EcalRecHitCollection>(conf.getParameter<edm::InputTag>("recHitsEELabel"));

   srfAwareCorrection_ = conf.getParameter<bool>("srfAwareCorrection");
   
   if (srfAwareCorrection_) {
     condnames_mean_.push_back("ecalPFClusterCor2017V2_EB_ZS_mean_25ns");
     condnames_mean_.push_back("ecalPFClusterCor2017V2_EB_Full_ptbin1_mean_25ns");
     condnames_mean_.push_back("ecalPFClusterCor2017V2_EB_Full_ptbin2_mean_25ns");
     condnames_mean_.push_back("ecalPFClusterCor2017V2_EB_Full_ptbin3_mean_25ns");
     condnames_mean_.push_back("ecalPFClusterCor2017V2_EE_ZS_mean_25ns");
     condnames_mean_.push_back("ecalPFClusterCor2017V2_EE_Full_ptbin1_mean_25ns");
     condnames_mean_.push_back("ecalPFClusterCor2017V2_EE_Full_ptbin2_mean_25ns");
     condnames_mean_.push_back("ecalPFClusterCor2017V2_EE_Full_ptbin3_mean_25ns");
     
     condnames_sigma_.push_back("ecalPFClusterCor2017V2_EB_ZS_sigma_25ns");
     condnames_sigma_.push_back("ecalPFClusterCor2017V2_EB_Full_ptbin1_sigma_25ns");
     condnames_sigma_.push_back("ecalPFClusterCor2017V2_EB_Full_ptbin2_sigma_25ns");
     condnames_sigma_.push_back("ecalPFClusterCor2017V2_EB_Full_ptbin3_sigma_25ns");
     condnames_sigma_.push_back("ecalPFClusterCor2017V2_EE_ZS_sigma_25ns");
     condnames_sigma_.push_back("ecalPFClusterCor2017V2_EE_Full_ptbin1_sigma_25ns");
     condnames_sigma_.push_back("ecalPFClusterCor2017V2_EE_Full_ptbin2_sigma_25ns");
     condnames_sigma_.push_back("ecalPFClusterCor2017V2_EE_Full_ptbin3_sigma_25ns");
   } else {
     if (applyMVACorrections_) {
       autoDetectBunchSpacing_ = conf.getParameter<bool>("autoDetectBunchSpacing");
       
       if (autoDetectBunchSpacing_) {
	 bunchSpacing_ = cc.consumes<unsigned int>(edm::InputTag("bunchSpacingProducer"));
	 bunchSpacingManual_ = 0;
       }
       else {
	 bunchSpacingManual_ = conf.getParameter<int>("bunchSpacing");
       }

       condnames_mean_25ns_.push_back("ecalPFClusterCorV2_EB_pfSize1_mean_25ns");
       condnames_mean_25ns_.push_back("ecalPFClusterCorV2_EB_pfSize2_mean_25ns");
       condnames_mean_25ns_.push_back("ecalPFClusterCorV2_EB_pfSize3_ptbin1_mean_25ns");
       condnames_mean_25ns_.push_back("ecalPFClusterCorV2_EB_pfSize3_ptbin2_mean_25ns");
       condnames_mean_25ns_.push_back("ecalPFClusterCorV2_EB_pfSize3_ptbin3_mean_25ns");
       condnames_mean_25ns_.push_back("ecalPFClusterCorV2_EE_pfSize1_mean_25ns");
       condnames_mean_25ns_.push_back("ecalPFClusterCorV2_EE_pfSize2_mean_25ns");
       condnames_mean_25ns_.push_back("ecalPFClusterCorV2_EE_pfSize3_ptbin1_mean_25ns");
       condnames_mean_25ns_.push_back("ecalPFClusterCorV2_EE_pfSize3_ptbin2_mean_25ns");
       condnames_mean_25ns_.push_back("ecalPFClusterCorV2_EE_pfSize3_ptbin3_mean_25ns");
       
       condnames_sigma_25ns_.push_back("ecalPFClusterCorV2_EB_pfSize1_sigma_25ns");
       condnames_sigma_25ns_.push_back("ecalPFClusterCorV2_EB_pfSize2_sigma_25ns");
       condnames_sigma_25ns_.push_back("ecalPFClusterCorV2_EB_pfSize3_ptbin1_sigma_25ns");
       condnames_sigma_25ns_.push_back("ecalPFClusterCorV2_EB_pfSize3_ptbin2_sigma_25ns");
       condnames_sigma_25ns_.push_back("ecalPFClusterCorV2_EB_pfSize3_ptbin3_sigma_25ns");
       condnames_sigma_25ns_.push_back("ecalPFClusterCorV2_EE_pfSize1_sigma_25ns");
       condnames_sigma_25ns_.push_back("ecalPFClusterCorV2_EE_pfSize2_sigma_25ns");
       condnames_sigma_25ns_.push_back("ecalPFClusterCorV2_EE_pfSize3_ptbin1_sigma_25ns");
       condnames_sigma_25ns_.push_back("ecalPFClusterCorV2_EE_pfSize3_ptbin2_sigma_25ns");
       condnames_sigma_25ns_.push_back("ecalPFClusterCorV2_EE_pfSize3_ptbin3_sigma_25ns");

       condnames_mean_50ns_.push_back("ecalPFClusterCorV2_EB_pfSize1_mean_50ns");
       condnames_mean_50ns_.push_back("ecalPFClusterCorV2_EB_pfSize2_mean_50ns");
       condnames_mean_50ns_.push_back("ecalPFClusterCorV2_EB_pfSize3_ptbin1_mean_50ns");
       condnames_mean_50ns_.push_back("ecalPFClusterCorV2_EB_pfSize3_ptbin2_mean_50ns");
       condnames_mean_50ns_.push_back("ecalPFClusterCorV2_EB_pfSize3_ptbin3_mean_50ns");
       condnames_mean_50ns_.push_back("ecalPFClusterCorV2_EE_pfSize1_mean_50ns");
       condnames_mean_50ns_.push_back("ecalPFClusterCorV2_EE_pfSize2_mean_50ns");
       condnames_mean_50ns_.push_back("ecalPFClusterCorV2_EE_pfSize3_ptbin1_mean_50ns");
       condnames_mean_50ns_.push_back("ecalPFClusterCorV2_EE_pfSize3_ptbin2_mean_50ns");
       condnames_mean_50ns_.push_back("ecalPFClusterCorV2_EE_pfSize3_ptbin3_mean_50ns");
       
       condnames_sigma_50ns_.push_back("ecalPFClusterCorV2_EB_pfSize1_sigma_50ns");
       condnames_sigma_50ns_.push_back("ecalPFClusterCorV2_EB_pfSize2_sigma_50ns");
       condnames_sigma_50ns_.push_back("ecalPFClusterCorV2_EB_pfSize3_ptbin1_sigma_50ns");
       condnames_sigma_50ns_.push_back("ecalPFClusterCorV2_EB_pfSize3_ptbin2_sigma_50ns");
       condnames_sigma_50ns_.push_back("ecalPFClusterCorV2_EB_pfSize3_ptbin3_sigma_50ns");
       condnames_sigma_50ns_.push_back("ecalPFClusterCorV2_EE_pfSize1_sigma_50ns");
       condnames_sigma_50ns_.push_back("ecalPFClusterCorV2_EE_pfSize2_sigma_50ns");
       condnames_sigma_50ns_.push_back("ecalPFClusterCorV2_EE_pfSize3_ptbin1_sigma_50ns");
       condnames_sigma_50ns_.push_back("ecalPFClusterCorV2_EE_pfSize3_ptbin2_sigma_50ns");
       condnames_sigma_50ns_.push_back("ecalPFClusterCorV2_EE_pfSize3_ptbin3_sigma_50ns");
     }
   }
    
}

void PFClusterEMEnergyCorrector::correctEnergies(const edm::Event &evt, 
						 const edm::EventSetup &es, 
						 const reco::PFCluster::EEtoPSAssociation &assoc, 
						 reco::PFClusterCollection& cs) {

  // First deal with pre-MVA corrections
  // Kept for backward compatibility
  if (!applyMVACorrections_) {
    for (unsigned int idx = 0; idx<cs.size(); ++idx) {
      reco::PFCluster &cluster = cs[idx];
      bool iseb = cluster.layer() == PFLayer::ECAL_BARREL;
      
      //compute preshower energies for endcap clusters
      double ePS1=0, ePS2=0;
      if(!iseb) {
        auto ee_key_val = std::make_pair(idx,edm::Ptr<reco::PFCluster>());
        const auto clustops = std::equal_range(assoc.begin(),
					       assoc.end(),
					       ee_key_val,
					       sortByKey);
        for( auto i_ps = clustops.first; i_ps != clustops.second; ++i_ps) {
          edm::Ptr<reco::PFCluster> psclus(i_ps->second);
          switch( psclus->layer() ) {
          case PFLayer::PS1:
            ePS1 += psclus->energy();
            break;
          case PFLayer::PS2:
            ePS2 += psclus->energy();
            break;
          default:
            break;
          }
        }
      }
      
      double correctedEnergy = calibrator_->energyEm(cluster,ePS1,ePS2,applyCrackCorrections_);
      cluster.setCorrectedEnergy(correctedEnergy);
      
    }
    return;
  }

  // Common defintion for SRF-aware and old style corrections

  //magic numbers for MINUIT-like transformation of BDT output onto limited range
  //(These should be stored inside the conditions object in the future as well)
  const double meanlimlow = -0.336;
  const double meanlimhigh = 0.916;
  const double meanoffset = meanlimlow + 0.5*(meanlimhigh-meanlimlow);
  const double meanscale = 0.5*(meanlimhigh-meanlimlow);
  
  const double sigmalimlow = 0.001;
  const double sigmalimhigh = 0.4;
  const double sigmaoffset = sigmalimlow + 0.5*(sigmalimhigh-sigmalimlow);
  const double sigmascale = 0.5*(sigmalimhigh-sigmalimlow);  

  EcalClusterLazyTools lazyTool(evt, es, recHitsEB_, recHitsEE_);

  if (!srfAwareCorrection_) {
    int bunchspacing = 450;  
    if (autoDetectBunchSpacing_) {
      edm::Handle<unsigned int> bunchSpacingH;
      evt.getByToken(bunchSpacing_,bunchSpacingH);
      bunchspacing = *bunchSpacingH;
    }
    else {
      bunchspacing = bunchSpacingManual_;
    }
    
    const std::vector<std::string> condnames_mean = (bunchspacing == 25) ? condnames_mean_25ns_ : condnames_mean_50ns_;
    const std::vector<std::string> condnames_sigma = (bunchspacing == 25) ? condnames_sigma_25ns_ : condnames_sigma_50ns_;
  
    const unsigned int ncor = condnames_mean.size();
    
    std::vector<edm::ESHandle<GBRForestD> > forestH_mean(ncor);
    std::vector<edm::ESHandle<GBRForestD> > forestH_sigma(ncor);  
    
    for (unsigned int icor=0; icor<ncor; ++icor) {
      es.get<GBRDWrapperRcd>().get(condnames_mean[icor],forestH_mean[icor]);
      es.get<GBRDWrapperRcd>().get(condnames_sigma[icor],forestH_sigma[icor]);
    }
    
    std::array<float,5> eval;    
    
    
    for (unsigned int idx = 0; idx<cs.size(); ++idx) {
      
      reco::PFCluster &cluster = cs[idx];
      
      double e = cluster.energy();
      double pt = cluster.pt(); 
      
      //limit raw energy value used to evaluate corrections
      //to avoid bad extrapolation
      double evale = e;
      if (maxPtForMVAEvaluation_>0. && pt>maxPtForMVAEvaluation_) {
	evale *= maxPtForMVAEvaluation_/pt; 
      }
      
      double invE = (e == 0.) ? 0. : 1./e; //guard against dividing by 0.      
      int size = lazyTool.n5x5(cluster);      
      bool iseb = cluster.layer() == PFLayer::ECAL_BARREL;
      
      //find index of corrections (0-4 for EB, 5-9 for EE, depending on cluster size and raw pt)
      int coridx = std::min(size,3)-1;
      if (coridx==2) {
	if (pt>4.5) {
	  coridx += 1;
	}
	if (pt>18.) {
	  coridx += 1;
	}
      }
      if (!iseb) {
	coridx += 5;
      }
      
      const GBRForestD &meanforest = *forestH_mean[coridx].product();
      const GBRForestD &sigmaforest = *forestH_sigma[coridx].product();
      
      //find seed crystal indices
      int ietaix=0;
      int iphiiy=0;
      if (iseb) {
	EBDetId ebseed(cluster.seed());
	ietaix = ebseed.ieta();
	iphiiy = ebseed.iphi();
      }
      else {
	EEDetId eeseed(cluster.seed());
	ietaix = eeseed.ix();
	iphiiy = eeseed.iy();      
      }
      
      
      //compute preshower energies for endcap clusters
      double ePS1=0, ePS2=0;
      if(!iseb) {
	auto ee_key_val = std::make_pair(idx,edm::Ptr<reco::PFCluster>());
	const auto clustops = std::equal_range(assoc.begin(),
					       assoc.end(),
					       ee_key_val,
					       sortByKey);
	for( auto i_ps = clustops.first; i_ps != clustops.second; ++i_ps) {
	  edm::Ptr<reco::PFCluster> psclus(i_ps->second);
	  switch( psclus->layer() ) {
	  case PFLayer::PS1:
	    ePS1 += psclus->energy();
	    break;
	  case PFLayer::PS2:
	    ePS2 += psclus->energy();
	    break;
	  default:
	    break;
	  }
	}
      }
      
      //fill array for forest evaluation
      eval[0] = evale;
      eval[1] = ietaix;
      eval[2] = iphiiy;
      if (!iseb) {
	eval[3] = ePS1*invE;
	eval[4] = ePS2*invE;
      }
      
      //these are the actual BDT responses
      double rawmean = meanforest.GetResponse(eval.data());
      double rawsigma = sigmaforest.GetResponse(eval.data());
      
      //apply transformation to limited output range (matching the training)
      double mean = meanoffset + meanscale*vdt::fast_sin(rawmean);
      double sigma = sigmaoffset + sigmascale*vdt::fast_sin(rawsigma);
      
      //regression target is ln(Etrue/Eraw)
      //so corrected energy is ecor=exp(mean)*e, uncertainty is exp(mean)*eraw*sigma=ecor*sigma
      double ecor = vdt::fast_exp(mean)*e;
      double sigmacor = sigma*ecor;
      
      cluster.setCorrectedEnergy(ecor);
      cluster.setCorrectedEnergyUncertainty(sigmacor);
      
    }
    return;
  }

  // Selective Readout Flags
  edm::Handle<EBSrFlagCollection> ebSrFlags;   
  evt.getByToken(ebSrFlagToken_, ebSrFlags);  
  if (!ebSrFlags.isValid()){    
    edm::LogWarning("PFClusterEMEnergyCorrector") << "This version of PFCluster corrections requires the EBSrFlagCollection  information to proceed!!! I will not correct. Will stop here ";
    return;
  }

  
  edm::Handle<EESrFlagCollection> eeSrFlags;  
  evt.getByToken(eeSrFlagToken_, eeSrFlags );
  if (!eeSrFlags.isValid()){    
    edm::LogWarning("PFClusterEMEnergyCorrector") << "This version of PFCluster corrections requires the EESrFlagCollection  information to proceed!!! I will not correct. Will stop here ";
    return;
  }
  
  ///needed for reading the SR flag
  edm::ESHandle<EcalTrigTowerConstituentsMap> hTriggerTowerMap;
  es.get<IdealGeometryRecord>().get(hTriggerTowerMap);
  triggerTowerMap_ = hTriggerTowerMap.product(); 

  //electronics map
  edm::ESHandle< EcalElectronicsMapping > ecalmapping;
  es.get< EcalMappingRcd >().get(ecalmapping);
  elecMap_ = ecalmapping.product();
  
  const unsigned int ncor = condnames_mean_.size();
  
  std::vector<edm::ESHandle<GBRForestD> > forestH_mean(ncor);
  std::vector<edm::ESHandle<GBRForestD> > forestH_sigma(ncor);  
  
  for (unsigned int icor=0; icor<ncor; ++icor) {
    es.get<GBRDWrapperRcd>().get(condnames_mean_[icor],forestH_mean[icor]);
    es.get<GBRDWrapperRcd>().get(condnames_sigma_[icor],forestH_sigma[icor]);
  }
  
  std::array<float,6> evalEB;
  std::array<float,5> evalEE;    
    
  for (unsigned int idx = 0; idx<cs.size(); ++idx) {
    
    reco::PFCluster &cluster = cs[idx];
    
    double e = cluster.energy();
    double pt = cluster.pt();
   
    //limit raw energy value used to evaluate corrections
    //to avoid bad extrapolation
    double evale = e;
    if (maxPtForMVAEvaluation_ > 0. && pt > maxPtForMVAEvaluation_) {
      evale *= maxPtForMVAEvaluation_/pt; 
    }
    
    double invE = (e == 0.) ? 0. : 1./e; //guard against dividing by 0.
    
    bool iseb = (cluster.layer() == PFLayer::ECAL_BARREL);
    int clusFlag = 0;
    
    if (iseb){
      EBSrFlagCollection::const_iterator srf = ebSrFlags->find(readOutUnitOf(static_cast<EBDetId>(cluster.seed())));      

      if (srf != ebSrFlags->end())
	clusFlag = srf->value();
      else
	clusFlag = 3; ///if the flag of that seed xtal is not found - just assume full readout
      
    } else {
      EESrFlagCollection::const_iterator srf = eeSrFlags->find(readOutUnitOf(static_cast<EEDetId>(cluster.seed())));      

      if (srf != eeSrFlags->end())
	clusFlag = srf->value();
      else
	clusFlag = 3;
      
    }
  


    //find index of corrections (0-3 for EB, 4-7 for EE, depending on cluster size and raw pt)
    int coridx = 0;
    int regind = 0;

    if (!iseb) regind = 4;

    if (clusFlag==1)
      coridx = 0 + regind;
    else {
      if (pt<2.5) coridx = 1 + regind;
      else if (pt>=2.5 && pt<6.) coridx = 2 + regind;
      else if (pt>=6.) coridx = 3 + regind;
    }
    if (clusFlag!=1 && clusFlag!=3) {
      edm::LogWarning("PFClusterEMEnergyCorrector") << "We can only correct regions readout in ZS (flag 1) or FULL readout (flag 3). Flag " << clusFlag << " is not recognized."
						    << "\n" << "Assuming FULL readout and continuing";
    }
           
    const GBRForestD &meanforest = *forestH_mean[coridx].product();
    const GBRForestD &sigmaforest = *forestH_sigma[coridx].product();
    
    //find seed crystal indices
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

    // Form ietamod20 and iphimod20 as well
    // hardcoded number are positions of modules boundaries
    int signeta = (ietaix > 0) ? 1 : -1;
    int ietamod20 = (std::abs(ietaix) < 26) ? ietaix - signeta : (ietaix - 26*signeta) % 20;
    int iphimod20 = (iphiiy-1) % 20;

    int size = lazyTool.n5x5(cluster);    
    int reducedHits = size;    
    if(size>=3)
      reducedHits = 3;

    //compute preshower energies for endcap clusters
    double ePS1=0, ePS2=0;
    if (!iseb) {
      auto ee_key_val = std::make_pair(idx,edm::Ptr<reco::PFCluster>());
      const auto clustops = std::equal_range(assoc.begin(),
                                            assoc.end(),
                                            ee_key_val,
                                            sortByKey);
      for (auto i_ps = clustops.first; i_ps != clustops.second; ++i_ps) {
        edm::Ptr<reco::PFCluster> psclus(i_ps->second);
	       
        switch( psclus->layer() ) {
        case PFLayer::PS1:
          ePS1 += psclus->energy();
          break;
        case PFLayer::PS2:
          ePS2 += psclus->energy();
	  break;
        default:
	  break;
        }
      }
    }

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
      evalEE[3] = (ePS1+ePS2)*invE;
      evalEE[4] = reducedHits;
    }
     
    //these are the actual BDT responses
    double rawmean = 1;
    double rawsigma = 0;

    if(iseb){
      rawmean = meanforest.GetResponse(evalEB.data());
      rawsigma = sigmaforest.GetResponse(evalEB.data());
    } else {
      rawmean = meanforest.GetResponse(evalEE.data());
      rawsigma = sigmaforest.GetResponse(evalEE.data());
    }

    //apply transformation to limited output range (matching the training)
    double mean = meanoffset + meanscale*vdt::fast_sin(rawmean);
    double sigma = sigmaoffset + sigmascale*vdt::fast_sin(rawsigma);
    
    //regression target is ln(Etrue/Eraw)
    //so corrected energy is ecor=exp(mean)*e, uncertainty is exp(mean)*eraw*sigma=ecor*sigma
    double ecor = iseb ? vdt::fast_exp(mean)*e : vdt::fast_exp(mean)*(e+ePS1+ePS2);
    double sigmacor = sigma*ecor;

    LogDebug("PFClusterEMEnergyCorrector") << "ieta : iphi : ietamod20 : iphimod20 : size : reducedHits = "
					   << ietaix << " " << iphiiy << " " 
					   << ietamod20 << " " << iphimod20 << " " 
					   << size << " " << reducedHits
					   << "\n" << "isEB : eraw : ePS1 : ePS2 : (eps1+eps2)/raw : Flag = "
					   << iseb << " " << evale << " " << ePS1 << " " << ePS2 << " " << (ePS1+ePS2)/evale << " " << clusFlag
					   << "\n" << "response : correction = " 
					   << exp(mean) << " " << ecor;
    
    cluster.setCorrectedEnergy(ecor);
    cluster.setCorrectedEnergyUncertainty(sigmacor);
  }
  
}

EcalTrigTowerDetId PFClusterEMEnergyCorrector::readOutUnitOf(const EBDetId& xtalId) const{
  return triggerTowerMap_->towerOf(xtalId);
}

EcalScDetId PFClusterEMEnergyCorrector::readOutUnitOf(const EEDetId& xtalId) const{
  const EcalElectronicsId& EcalElecId = elecMap_->getElectronicsId(xtalId);
  int iDCC= EcalElecId.dccId();
  int iDccChan = EcalElecId.towerId();
  const bool ignoreSingle = true;
  const std::vector<EcalScDetId> id = elecMap_->getEcalScDetId(iDCC, iDccChan, ignoreSingle);
  return id.size()>0?id[0]:EcalScDetId();
}
