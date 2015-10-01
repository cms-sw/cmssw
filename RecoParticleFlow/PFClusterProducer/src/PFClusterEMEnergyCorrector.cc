#include "RecoParticleFlow/PFClusterProducer/interface/PFClusterEMEnergyCorrector.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterLazyTools.h"
#include "CondFormats/DataRecord/interface/GBRDWrapperRcd.h"
#include "CondFormats/EgammaObjects/interface/GBRForestD.h"
#include "vdt/vdtMath.h"
#include <array>

namespace {
  typedef reco::PFCluster::EEtoPSAssociation::value_type EEPSPair;
  bool sortByKey(const EEPSPair& a, const EEPSPair& b) {
    return a.first < b.first;
  } 
}

PFClusterEMEnergyCorrector::PFClusterEMEnergyCorrector(const edm::ParameterSet& conf, edm::ConsumesCollector &&cc) :
  _calibrator(new PFEnergyCalibration) {

   _applyCrackCorrections = conf.getParameter<bool>("applyCrackCorrections");
   _applyMVACorrections = conf.getParameter<bool>("applyMVACorrections");
  
   
  if (_applyMVACorrections) {
    _recHitsEB = cc.consumes<EcalRecHitCollection>(conf.getParameter<edm::InputTag>("recHitsEBLabel"));
    _recHitsEE = cc.consumes<EcalRecHitCollection>(conf.getParameter<edm::InputTag>("recHitsEELabel"));
    
    autoDetectBunchSpacing_ = conf.getParameter<bool>("autoDetectBunchSpacing");

    if (autoDetectBunchSpacing_) {
      bunchSpacing_ = cc.consumes<unsigned int>(edm::InputTag("bunchSpacingProducer"));
      bunchSpacingManual_ = 0;
    }
    else {
      bunchSpacingManual_ = conf.getParameter<int>("bunchSpacing");
    }
    
    _condnames_mean_50ns.push_back("ecalPFClusterCorV2_EB_pfSize1_mean_50ns");
    _condnames_mean_50ns.push_back("ecalPFClusterCorV2_EB_pfSize2_mean_50ns");
    _condnames_mean_50ns.push_back("ecalPFClusterCorV2_EB_pfSize3_ptbin1_mean_50ns");
    _condnames_mean_50ns.push_back("ecalPFClusterCorV2_EB_pfSize3_ptbin2_mean_50ns");
    _condnames_mean_50ns.push_back("ecalPFClusterCorV2_EB_pfSize3_ptbin3_mean_50ns");
    _condnames_mean_50ns.push_back("ecalPFClusterCorV2_EE_pfSize1_mean_50ns");
    _condnames_mean_50ns.push_back("ecalPFClusterCorV2_EE_pfSize2_mean_50ns");
    _condnames_mean_50ns.push_back("ecalPFClusterCorV2_EE_pfSize3_ptbin1_mean_50ns");
    _condnames_mean_50ns.push_back("ecalPFClusterCorV2_EE_pfSize3_ptbin2_mean_50ns");
    _condnames_mean_50ns.push_back("ecalPFClusterCorV2_EE_pfSize3_ptbin3_mean_50ns");
    
    _condnames_sigma_50ns.push_back("ecalPFClusterCorV2_EB_pfSize1_sigma_50ns");
    _condnames_sigma_50ns.push_back("ecalPFClusterCorV2_EB_pfSize2_sigma_50ns");
    _condnames_sigma_50ns.push_back("ecalPFClusterCorV2_EB_pfSize3_ptbin1_sigma_50ns");
    _condnames_sigma_50ns.push_back("ecalPFClusterCorV2_EB_pfSize3_ptbin2_sigma_50ns");
    _condnames_sigma_50ns.push_back("ecalPFClusterCorV2_EB_pfSize3_ptbin3_sigma_50ns");
    _condnames_sigma_50ns.push_back("ecalPFClusterCorV2_EE_pfSize1_sigma_50ns");
    _condnames_sigma_50ns.push_back("ecalPFClusterCorV2_EE_pfSize2_sigma_50ns");
    _condnames_sigma_50ns.push_back("ecalPFClusterCorV2_EE_pfSize3_ptbin1_sigma_50ns");
    _condnames_sigma_50ns.push_back("ecalPFClusterCorV2_EE_pfSize3_ptbin2_sigma_50ns");
    _condnames_sigma_50ns.push_back("ecalPFClusterCorV2_EE_pfSize3_ptbin3_sigma_50ns");
    
    _condnames_mean_25ns.push_back("ecalPFClusterCorV2_EB_pfSize1_mean_25ns");
    _condnames_mean_25ns.push_back("ecalPFClusterCorV2_EB_pfSize2_mean_25ns");
    _condnames_mean_25ns.push_back("ecalPFClusterCorV2_EB_pfSize3_ptbin1_mean_25ns");
    _condnames_mean_25ns.push_back("ecalPFClusterCorV2_EB_pfSize3_ptbin2_mean_25ns");
    _condnames_mean_25ns.push_back("ecalPFClusterCorV2_EB_pfSize3_ptbin3_mean_25ns");
    _condnames_mean_25ns.push_back("ecalPFClusterCorV2_EE_pfSize1_mean_25ns");
    _condnames_mean_25ns.push_back("ecalPFClusterCorV2_EE_pfSize2_mean_25ns");
    _condnames_mean_25ns.push_back("ecalPFClusterCorV2_EE_pfSize3_ptbin1_mean_25ns");
    _condnames_mean_25ns.push_back("ecalPFClusterCorV2_EE_pfSize3_ptbin2_mean_25ns");
    _condnames_mean_25ns.push_back("ecalPFClusterCorV2_EE_pfSize3_ptbin3_mean_25ns");
    
    _condnames_sigma_25ns.push_back("ecalPFClusterCorV2_EB_pfSize1_sigma_25ns");
    _condnames_sigma_25ns.push_back("ecalPFClusterCorV2_EB_pfSize2_sigma_25ns");
    _condnames_sigma_25ns.push_back("ecalPFClusterCorV2_EB_pfSize3_ptbin1_sigma_25ns");
    _condnames_sigma_25ns.push_back("ecalPFClusterCorV2_EB_pfSize3_ptbin2_sigma_25ns");
    _condnames_sigma_25ns.push_back("ecalPFClusterCorV2_EB_pfSize3_ptbin3_sigma_25ns");
    _condnames_sigma_25ns.push_back("ecalPFClusterCorV2_EE_pfSize1_sigma_25ns");
    _condnames_sigma_25ns.push_back("ecalPFClusterCorV2_EE_pfSize2_sigma_25ns");
    _condnames_sigma_25ns.push_back("ecalPFClusterCorV2_EE_pfSize3_ptbin1_sigma_25ns");
    _condnames_sigma_25ns.push_back("ecalPFClusterCorV2_EE_pfSize3_ptbin2_sigma_25ns");
    _condnames_sigma_25ns.push_back("ecalPFClusterCorV2_EE_pfSize3_ptbin3_sigma_25ns");
  }
  

  
}


void PFClusterEMEnergyCorrector::correctEnergies(const edm::Event &evt, const edm::EventSetup &es, const reco::PFCluster::EEtoPSAssociation &assoc, reco::PFClusterCollection& cs) {

  //legacy corrections
  if (!_applyMVACorrections) {
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
      
      double correctedEnergy = _calibrator->energyEm(cluster,ePS1,ePS2,_applyCrackCorrections);
      cluster.setCorrectedEnergy(correctedEnergy);
      
    }
    return;
  }
  
  int bunchspacing = 450;  
  
  if (autoDetectBunchSpacing_) {
      edm::Handle<unsigned int> bunchSpacingH;
      evt.getByToken(bunchSpacing_,bunchSpacingH);
      bunchspacing = *bunchSpacingH;
  }
  else {
    bunchspacing = bunchSpacingManual_;
  }
  
  const std::vector<std::string> condnames_mean = (bunchspacing == 25) ? _condnames_mean_25ns : _condnames_mean_50ns;
  const std::vector<std::string> condnames_sigma = (bunchspacing == 25) ? _condnames_sigma_25ns : _condnames_sigma_50ns;
  
  const unsigned int ncor = condnames_mean.size();
  
  std::vector<edm::ESHandle<GBRForestD> > forestH_mean(ncor);
  std::vector<edm::ESHandle<GBRForestD> > forestH_sigma(ncor);  
  
  for (unsigned int icor=0; icor<ncor; ++icor) {
    es.get<GBRDWrapperRcd>().get(condnames_mean[icor],forestH_mean[icor]);
    es.get<GBRDWrapperRcd>().get(condnames_sigma[icor],forestH_sigma[icor]);
  }
  
  std::array<float,5> eval;
    
  EcalClusterLazyTools lazyTool(evt, es, _recHitsEB, _recHitsEE);
  
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
  
  for (unsigned int idx = 0; idx<cs.size(); ++idx) {
    
    reco::PFCluster &cluster = cs[idx];
    
    double e = cluster.energy();
    double pt = cluster.pt(); 
    
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
    eval[0] = e;
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
    double ecor = exp(mean)*e;
    double sigmacor = sigma*ecor;
    
    cluster.setCorrectedEnergy(ecor);
    cluster.setCorrectedEnergyUncertainty(sigmacor);
    
  }
  
}



