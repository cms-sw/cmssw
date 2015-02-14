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
    _vertices  = cc.consumes<reco::VertexCollection>(conf.getParameter<edm::InputTag>("verticesLabel"));
    
    autoDetectBunchSpacing_ = conf.getParameter<bool>("autoDetectBunchSpacing");

    if (autoDetectBunchSpacing_) {
      bunchSpacing_ = cc.consumes<int>(edm::InputTag("addPileupInfo","bunchSpacing"));
      bunchSpacingManual_ = 0;
    }
    else {
      bunchSpacingManual_ = conf.getParameter<int>("bunchSpacing");
    }
    
    _condnames_mean_50ns.push_back("ecalPFClusterCor_EB_pfSize1_mean_50ns");
    _condnames_mean_50ns.push_back("ecalPFClusterCor_EB_pfSize2_mean_50ns");
    _condnames_mean_50ns.push_back("ecalPFClusterCor_EB_pfSize3_mean_50ns");
    _condnames_mean_50ns.push_back("ecalPFClusterCor_EE_pfSize1_mean_50ns");
    _condnames_mean_50ns.push_back("ecalPFClusterCor_EE_pfSize2_mean_50ns");
    _condnames_mean_50ns.push_back("ecalPFClusterCor_EE_pfSize3_mean_50ns");

    _condnames_sigma_50ns.push_back("ecalPFClusterCor_EB_pfSize1_sigma_50ns");
    _condnames_sigma_50ns.push_back("ecalPFClusterCor_EB_pfSize2_sigma_50ns");
    _condnames_sigma_50ns.push_back("ecalPFClusterCor_EB_pfSize3_sigma_50ns");
    _condnames_sigma_50ns.push_back("ecalPFClusterCor_EE_pfSize1_sigma_50ns");
    _condnames_sigma_50ns.push_back("ecalPFClusterCor_EE_pfSize2_sigma_50ns");
    _condnames_sigma_50ns.push_back("ecalPFClusterCor_EE_pfSize3_sigma_50ns");
    
    _condnames_mean_25ns.push_back("ecalPFClusterCor_EB_pfSize1_mean_25ns");
    _condnames_mean_25ns.push_back("ecalPFClusterCor_EB_pfSize2_mean_25ns");
    _condnames_mean_25ns.push_back("ecalPFClusterCor_EB_pfSize3_mean_25ns");
    _condnames_mean_25ns.push_back("ecalPFClusterCor_EE_pfSize1_mean_25ns");
    _condnames_mean_25ns.push_back("ecalPFClusterCor_EE_pfSize2_mean_25ns");
    _condnames_mean_25ns.push_back("ecalPFClusterCor_EE_pfSize3_mean_25ns");

    _condnames_sigma_25ns.push_back("ecalPFClusterCor_EB_pfSize1_sigma_25ns");
    _condnames_sigma_25ns.push_back("ecalPFClusterCor_EB_pfSize2_sigma_25ns");
    _condnames_sigma_25ns.push_back("ecalPFClusterCor_EB_pfSize3_sigma_25ns");
    _condnames_sigma_25ns.push_back("ecalPFClusterCor_EE_pfSize1_sigma_25ns");
    _condnames_sigma_25ns.push_back("ecalPFClusterCor_EE_pfSize2_sigma_25ns");
    _condnames_sigma_25ns.push_back("ecalPFClusterCor_EE_pfSize3_sigma_25ns");      
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
    if (evt.isRealData()) {
      edm::RunNumber_t run = evt.run();
      if (run == 178003 ||
          run == 178004 ||
          run == 209089 ||
          run == 209106 ||
          run == 209109 ||
          run == 209146 ||
          run == 209148 ||
          run == 209151) {
        bunchspacing = 25;
      }
      else {
        bunchspacing = 50;
      }
    }
    else {
      edm::Handle<int> bunchSpacingH;
      evt.getByToken(bunchSpacing_,bunchSpacingH);
      bunchspacing = *bunchSpacingH;
    }
  }
  
  const unsigned int ncor = 6;
  
  std::vector<edm::ESHandle<GBRForestD> > forestH_mean(ncor);
  std::vector<edm::ESHandle<GBRForestD> > forestH_sigma(ncor);
  
  const std::vector<std::string> condnames_mean = (bunchspacing == 25) ? _condnames_mean_25ns : _condnames_mean_50ns;
  const std::vector<std::string> condnames_sigma = (bunchspacing == 25) ? _condnames_sigma_25ns : _condnames_sigma_50ns;
  
  for (unsigned int icor=0; icor<ncor; ++icor) {
    es.get<GBRDWrapperRcd>().get(condnames_mean[icor],forestH_mean[icor]);
    es.get<GBRDWrapperRcd>().get(condnames_sigma[icor],forestH_sigma[icor]);
  }
  
  std::array<float,11> eval;
    
  EcalClusterLazyTools lazyTool(evt, es, _recHitsEB, _recHitsEE);

  //count number of primary vertices for pileup correction
  edm::Handle<reco::VertexCollection> vtxH;
  evt.getByToken(_vertices,vtxH);
  int nvtx = 0;
  for (const reco::Vertex &vtx : *vtxH) {
    if (!vtx.isFake()) {
      ++nvtx;  
    }
  }
  
  //magic numbers for MINUIT-like transformation of BDT output onto limited range
  //(These should be stored inside the conditions object in the future as well)
  const double meanlimlow = 1./1.4;
  const double meanlimhigh = 1./0.4;
  const double meanoffset = meanlimlow + 0.5*(meanlimhigh-meanlimlow);
  const double meanscale = 0.5*(meanlimhigh-meanlimlow);
  
  const double sigmalimlow = 0.003;
  const double sigmalimhigh = 0.5;
  const double sigmaoffset = sigmalimlow + 0.5*(sigmalimhigh-sigmalimlow);
  const double sigmascale = 0.5*(sigmalimhigh-sigmalimlow);  
  
  for (unsigned int idx = 0; idx<cs.size(); ++idx) {
    
    reco::PFCluster &cluster = cs[idx];
    
    double e = cluster.energy();
    double eta = cluster.eta();
    double phi = cluster.phi();    
    
    double invE = 1./e;
    
    int size = lazyTool.n5x5(cluster);
    
    bool iseb = cluster.layer() == PFLayer::ECAL_BARREL;

    //find index of corrections (0-2 for EB, 3-5 for EE, depending on cluster size)
    int coridx = std::min(size,3)-1;
    if (!iseb) {
      coridx += 3;
    }
    
    const GBRForestD &meanforest = *forestH_mean[coridx].product();
    const GBRForestD &sigmaforest = *forestH_sigma[coridx].product();
    
    double e1x3    = lazyTool.e1x3(cluster);
    double e2x2    = lazyTool.e2x2(cluster);
    double e2x5max = lazyTool.e2x5Max(cluster);    
    double e3x3    = lazyTool.e3x3(cluster);
    double e5x5    = lazyTool.e5x5(cluster);    
    
    
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
    eval[1] = eta;
    eval[2] = phi;
    
    if (size==1) {
      eval[3] = nvtx;
      if (!iseb) {
        eval[4] = ePS1*invE;
        eval[5] = ePS2*invE;
      }
    }
    else if (size==2) {
      eval[3] = e1x3*invE;
      eval[4] = nvtx;
      if (!iseb) {
        eval[5] = ePS1*invE;
        eval[6] = ePS2*invE;
      }
    }
    else if (size>2) {
      eval[3] = e1x3*invE;
      eval[4] = e2x2*invE;
      eval[5] = e2x5max*invE;
      eval[6] = e3x3*invE;
      eval[7] = e5x5*invE;
      eval[8] = nvtx;
      if (!iseb) {
        eval[9] = ePS1*invE;
        eval[10] = ePS2*invE;
      }
    }
    
    //these are the actual BDT responses
    double rawmean = meanforest.GetResponse(eval.data());
    double rawsigma = sigmaforest.GetResponse(eval.data());
    
    //apply transformation to limited output range (matching the training)
    double mean = meanoffset + meanscale*vdt::fast_sin(rawmean);
    double sigma = sigmaoffset + sigmascale*vdt::fast_sin(rawsigma);
    
    cluster.setCorrectedEnergy(mean*e);
    cluster.setCorrectedEnergyUncertainty(sigma*e);
    
  }
  
}



