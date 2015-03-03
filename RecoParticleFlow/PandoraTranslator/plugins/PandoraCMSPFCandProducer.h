// system include files
#include <memory>
#include <vector>
#include <map>
#include <unordered_map>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Pandora/Pandora.h"
#include "Pandora/StatusCodes.h"
#include "Api/PandoraApi.h"
#include "TLorentzVector.h"
#include "Objects/ParticleFlowObject.h"
#include "Pandora/PandoraInputTypes.h"
#include "Pandora/PandoraInternal.h"
#include "Objects/Cluster.h"
#include "Objects/Track.h"

#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "DataFormats/GeometrySurface/interface/BoundDisk.h"

#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitFwd.h"

#include "DataFormats/RecoCandidate/interface/TrackAssociation.h"
#include "DataFormats/HGCRecHit/interface/HGCRecHitCollections.h"
#include "DataFormats/ForwardDetId/interface/ForwardSubdetector.h"

#include "DataFormats/ParticleFlowReco/interface/PFRecTrack.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecTrackFwd.h"

#include "DataFormats/VertexReco/interface/Vertex.h"

#include "DataFormats/Math/interface/Vector3D.h"
//DQM services for histogram
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "RecoParticleFlow/PandoraTranslator/interface/steerManager.h"

#include <TH1.h>
#include <TFile.h>
#include <TTree.h>
#include "TH1F.h"

//
// class declaration
//
class CaloGeometryRecord;
class IdealGeometryRecord;
class CaloGeometry;
class HGCalGeometry;
class MagneticField;
class TrackerGeometry;
class PropagatorWithMaterial;

// namespace pandora {class Pandora;}

//helper class to store calibration info for HGC
class CalibHGC {
  public:
    //constructor
    CalibHGC(ForwardSubdetector id, std::string name, bool debug_) : m_id(id), m_name(name), debug(debug_), initialized(false)
    {
      m_name = name;
      m_stm_nameLayer = "layerSet_" + m_name;
      m_stm_nameEM = "energyWeight_EM_" + m_name;
      m_stm_nameHAD = "energyWeight_HAD_" + m_name;
      //skip layer 0 in all vectors
      nCellInteractionLengths.push_back(0.);
      nCellRadiationLengths.push_back(0.);
      m_absorberCorrectionEM.push_back(1.);
      m_absorberCorrectionHAD.push_back(1.);
      m_energyWeightEM.push_back(1.);
      m_energyWeightHAD.push_back(1.);
    }
    //destructor
    virtual ~CalibHGC() {}
    
    //accessors
    virtual double GetADC2GeV() { return m_Calibr_ADC2GeV; }
    virtual double GetEMCalib(unsigned int layer, double eta) { 
      if(layer>m_TotalLayers) return 1.;
      
      if(m_energyCorrMethod == "ABSCORR"){
        return m_CalToEMGeV * GetAbsCorrEM(layer,eta) * m_EM_addCalibr;
      }
      else if(m_energyCorrMethod == "WEIGHTING"){
        return m_CalToEMGeV * m_energyWeightEM[layer] * m_CalToMip;
      }
      else return 1.;
    }
    virtual double GetHADCalib(unsigned int layer, double eta) {
      if(layer>m_TotalLayers) return 1.;
      
      if(m_energyCorrMethod == "ABSCORR"){
        return m_CalToHADGeV * GetAbsCorrHAD(layer,eta) * m_HAD_addCalibr;
      }
      else if(m_energyCorrMethod == "WEIGHTING"){
        return m_CalToHADGeV * m_energyWeightHAD[layer] * m_CalToMip;
      }
      else return 1.;
    }
    // change this to return simply the number of x0s
    virtual double GetAbsCorrEM(unsigned int layer, double eta){
      if(layer==1 && m_id==ForwardSubdetector::HGCEE && useOverburdenCorrection){
        //lower bound: first element in map with key >= name
        typename std::map<double,double>::iterator lb = nOverburdenRadiationLengths.lower_bound(eta);
        if(lb != nOverburdenRadiationLengths.begin()) lb--;
        if(lb != nOverburdenRadiationLengths.end()){
          return nCellRadiationLengths[layer] + lb->second;
        }
        else return 1.;
      }
      else return nCellRadiationLengths[layer];
    }
    // change this to return simply the number of lambdas
    virtual double GetAbsCorrHAD(unsigned int layer, double eta){
      if(layer==1 && m_id==ForwardSubdetector::HGCEE && useOverburdenCorrection){
        //lower bound: first element in map with key >= name
        typename std::map<double,double>::iterator lb = m_absorberCorrectionHADeta.lower_bound(eta);
        if(lb != m_absorberCorrectionHADeta.begin()) lb--;
        if(lb != m_absorberCorrectionHADeta.end()){
          return nCellInteractionLengths[layer] + lb->second;
        }
        else return 1.;
      }
      else return nCellInteractionLengths[layer];
    }
    
    //helper functions
    virtual void initialize() {
      if(initialized) return;
      getLayerProperties();
      initialized = true;
    }
    virtual void getLayerProperties() {
      for(unsigned layer = 1; layer <= m_TotalLayers; layer++){     
	m_absorberCorrectionEM.push_back(nCellRadiationLengths[layer]/calibrationRadiationLength);
        m_absorberCorrectionHAD.push_back(nCellInteractionLengths[layer]/calibrationInteractionLength);
        if(debug) {
	  std::cout << m_name << ": nCellRadiationLengths = " << nCellRadiationLengths[layer] 
		    << " calibrationRadiationLength = " << calibrationRadiationLength << std::endl;
	  std::cout << m_name << ": nCellInteractionLengths = " << nCellInteractionLengths[layer] 
		    << " calibrationInteractionLength = " << calibrationInteractionLength << std::endl;
	  std::cout << m_name << ": absCorrEM = " << m_absorberCorrectionEM.back() << ", absCorrHAD = " << m_absorberCorrectionHAD.back() << std::endl;
	}
        
        m_energyWeightEM.push_back(m_stm->getCorrectionAtPoint(layer+1,m_stm_nameLayer,m_stm_nameEM));
        m_energyWeightHAD.push_back(m_stm->getCorrectionAtPoint(layer+1,m_stm_nameLayer,m_stm_nameHAD));
      }
      //eta-dependent correction for EE layer 1
      if(m_id==ForwardSubdetector::HGCEE && useOverburdenCorrection){
        typename std::map<double,double>::iterator EMit = nOverburdenRadiationLengths.begin();
        typename std::map<double,double>::iterator HADit = nOverburdenInteractionLengths.begin();
        for(; EMit != nOverburdenRadiationLengths.end(); EMit++){
          m_absorberCorrectionEMeta[EMit->first] = (nCellRadiationLengths[1] + EMit->second)/calibrationRadiationLength;
          m_absorberCorrectionHADeta[HADit->first] = (nCellInteractionLengths[1] + HADit->second)/calibrationInteractionLength;
          if(debug) std::cout << m_name << ": eta = " << EMit->first << ", absCorrEM = " << m_absorberCorrectionEMeta[EMit->first] << ", absCorrHAD = " << m_absorberCorrectionHADeta[HADit->first] << std::endl;
          HADit++;
        }
      }
    }
    
    //member variables
    ForwardSubdetector m_id;
    std::string m_name;
    bool debug;
    bool initialized;
    std::string m_energyCorrMethod;
    steerManager * m_stm;
    std::string m_stm_nameLayer, m_stm_nameEM, m_stm_nameHAD;
    double m_CalThresh;
    double m_CalMipThresh;
    double m_CalToMip;
    double m_CalToEMGeV;
    double m_CalToHADGeV;
    double m_Calibr_ADC2GeV;
    double m_EM_addCalibr;
    double m_HAD_addCalibr;
    unsigned int m_TotalLayers;
    std::vector<double> nCellInteractionLengths;
    std::vector<double> nCellRadiationLengths;
    double calibrationInteractionLength;
    double calibrationRadiationLength;
    std::map<double,double> nOverburdenInteractionLengths;
    std::map<double,double> nOverburdenRadiationLengths;
    bool useOverburdenCorrection;
    std::vector<double> m_absorberCorrectionEM;
    std::vector<double> m_absorberCorrectionHAD;
    std::map<double,double> m_absorberCorrectionEMeta;
    std::map<double,double> m_absorberCorrectionHADeta;
    std::vector<double> m_energyWeightEM;
    std::vector<double> m_energyWeightHAD;
};


class PandoraCMSPFCandProducer : public edm::EDProducer {
public:
  explicit PandoraCMSPFCandProducer(const edm::ParameterSet&);
  ~PandoraCMSPFCandProducer();

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  std::unique_ptr<pandora::Pandora>        m_pPandora;

  void prepareTrack(edm::Event& iEvent);
  void prepareHits(edm::Event& iEvent);
  void preparemcParticle(edm::Event& iEvent);
  void ProcessRecHits(const reco::PFRecHit* rh, unsigned int index, const HGCalGeometry* geom, CalibHGC& calib, int& nCaloHits, int& nNotFound, reco::Vertex& pv,
                     const pandora::HitType hitType, const pandora::HitRegion hitRegion, PandoraApi::RectangularCaloHitParameters& caloHitParameters);

  
  void preparePFO(edm::Event& iEvent);
  void prepareGeometry();
  void SetDefaultSubDetectorParameters(const std::string &subDetectorName, const pandora::SubDetectorType subDetectorType, PandoraApi::Geometry::SubDetector::Parameters &parameters) const;
  void CalculateCornerSubDetectorParameters(const CaloSubdetectorGeometry* geom,  const std::vector<DetId>& cells, const pandora::SubDetectorType subDetectorType, 
                                            double& min_innerRadius, double& max_outerRadius, double& min_innerZ, double& max_outerZ,
                                            bool doLayers, std::vector<double>& min_innerR_depth, std::vector<double>& min_innerZ_depth) const;
  void SetCornerSubDetectorParameters(PandoraApi::Geometry::SubDetector::Parameters &parameters, 
                                      const double& min_innerRadius, const double& max_outerRadius, const double& min_innerZ, const double& max_outerZ) const;
  void SetSingleLayerParameters(PandoraApi::Geometry::SubDetector::Parameters &parameters, PandoraApi::Geometry::LayerParameters &layerParameters) const;
  void SetMultiLayerParameters(PandoraApi::Geometry::SubDetector::Parameters &parameters, std::vector<PandoraApi::Geometry::LayerParameters*> &layerParameters,
                               std::vector<double>& min_innerR_depth, std::vector<double>& min_innerZ_depth, const unsigned int& nTotalLayers, CalibHGC& calib) const;
                                         
  TrackingParticleRefVector getTpSiblings(TrackingParticleRef tp);
  TrackingParticleRefVector getTpDaughters(TrackingParticleRef tp);

  void convertPandoraToCMSSW(const edm::Handle<reco::PFRecTrackCollection>&,
                 const edm::Handle<reco::PFRecHitCollection>&, 
                 edm::Event& iEvent);
  
  std::string _outputFileName;
  std::string electronOutputCol_;
  edm::FileInPath m_pandoraSettingsXmlFile;

  edm::FileInPath m_calibrationParameterFile;
  edm::FileInPath m_energyWeightingFilename;
  edm::FileInPath m_layerDepthFilename;
  edm::FileInPath m_overburdenDepthFilename;
  bool m_useOverburdenCorrection;

  std::string m_energyCorrMethod; //energy correction method

  void initPandoraCalibrParameters();
  void readCalibrParameterFile();
  void readEnergyWeight(); //FIXME part of calibration, to be merged to readCalibrParameterFile??

private:
  virtual void beginJob() override;
  virtual void produce(edm::Event&, const edm::EventSetup&) override;
  virtual void endJob() override;
  virtual void beginLuminosityBlock(const edm::LuminosityBlock&, const edm::EventSetup&) override;

  virtual void resetVariables() {
     for (int il=0; il<100; il++) {
        m_hitEperLayer_EM[ForwardSubdetector::HGCEE][il] = 0.;
        m_hitEperLayer_EM[ForwardSubdetector::HGCHEF][il] = 0.;
        m_hitEperLayer_EM[ForwardSubdetector::HGCHEB][il] = 0.;
        m_hitEperLayer_HAD[ForwardSubdetector::HGCEE][il] = 0.;
        m_hitEperLayer_HAD[ForwardSubdetector::HGCHEF][il] = 0.;
        m_hitEperLayer_HAD[ForwardSubdetector::HGCHEB][il] = 0.;    
     }
  };

  steerManager * stm;

  //virtual void beginRun(edm::Run const&, edm::EventSetup const&) override;
  //virtual void endRun(edm::Run const&, edm::EventSetup const&) override;
  //virtual void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;
  //virtual void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;

  // ----------member data ---------------------------

  //flags
  bool debugPrint, debugHisto;
  bool useRecoTrackAsssociation;
  
  // ----------access to event data
  edm::InputTag    inputTagHGCrechit_;
  edm::InputTag    inputTagtPRecoTrackAsssociation_;
  edm::InputTag    inputTagGenParticles_;
  edm::InputTag    inputTagGeneralTracks_;
  //std::vector<std::string> mFileNames;
  
  // hash tables to translate back to CMSSW collection index from Pandora
  std::unordered_map<const void*,unsigned int> recHitMap;
  std::unordered_map<const void*,unsigned int> recTrackMap;
  
  //geometry handles
  edm::ESHandle<CaloGeometry> geoHandle;
  edm::ESHandle<HGCalGeometry> hgceeGeoHandle ; 
  edm::ESHandle<HGCalGeometry> hgchefGeoHandle ; 
  edm::ESHandle<HGCalGeometry> hgchebGeoHandle ; 
  edm::ESHandle<MagneticField> magneticField;
  edm::ESHandle<TrackerGeometry> tkGeom;
  
  //for track propagation to calorimeter
  std::vector<ReferenceCountingPointer<BoundDisk> > _plusSurface,_minusSurface;
  std::unique_ptr<PropagatorWithMaterial> _mat_prop;  

  TFile * file;
  TTree *mytree;
  double ene_track, ene_match_track,ene_match_em,ene_match_had, ene_had,ene_em,ene_match,mass_match,pid_match,pT_match,charge_match;
  double ene_true,mass_true,pid_true,pT_true,charge_true;
  double first_layer,last_layer,first_layer_match,last_layer_match;
  int runno, eventno, lumi , nbPFOs;

  int isDecBefCal;
  double RminVtxDaughter[2];
  double ZminVtxDaughter[2];
  int isDecayedBeforeCalo[2];

  TH1F * Epfos;
  TH1F * Egenpart;
  TH1F * Energy_res;

  TH1F * h_sumPfoE;
  TH1F * h_sumPfoEEM;
  TH1F * h_sumPfoEHad;
  TH1F * h_nbPFOs;

  TH1F * h_sumCaloE;
  TH1F * h_sumEcalEEM;
  TH1F * h_sumHcalEEM;
  TH1F * h_sumEcalEHad;
  TH1F * h_sumEcalEHad_unc;
  TH1F * h_sumHcalEHad;
  TH1F * h_sumEcalEHadc;
  TH1F * h_sumHcalEHadc;
  TH1F * h_sumEHad;

  TH1F * h_sumCaloEM;
  TH1F * h_sumCaloHad;

  TH1F * h_simDir_sumCaloEM ;//take only hits in sim part. direction
  TH1F * h_simDir_sumCaloHad;//take only hits in sim part. direction

  std::map<ForwardSubdetector,TH1F*> h_MIP ;
  std::map<ForwardSubdetector,TH1F*> h_MIP_Corr ;
  TH1F * h_MCp_Eta;
  TH1F * h_MCp_Phi;
  TH1F * h_hit_Eta;
  TH1F * h_hit_Phi;

  std::map<ForwardSubdetector,TH1F*> h_hitEperLayer_EM;
  std::map<ForwardSubdetector,double*> m_hitEperLayer_EM;
  std::map<ForwardSubdetector,TH1F*> h_hitEperLayer_HAD;
  std::map<ForwardSubdetector,double*> m_hitEperLayer_HAD;

  TH2F * h2_Calo_EM_hcalEecalE;
  TH2F * h2_Calo_Had_hcalEecalE;

  TH2F * h2_EM_hcalEecalE;
  TH2F * h2_Had_hcalEecalE;

  double sumCaloEnergy;
  double sumCaloEnergyEM;
  double sumCaloEnergyHAD;

  double simDir_sumCaloEnergyEM;
  double simDir_sumCaloEnergyHAD;

  double sumCaloECALEnergyEM;
  double sumCaloHCALEnergyEM;
  double sumCaloECALEnergyHAD;
  double sumCaloECALEnergyHAD_unc;
  double sumCaloHCALEnergyHAD;
  
  //-------------- energy weighting ------------------
  unsigned int nHGCeeLayers, nHGChefLayers, nHGChebLayers; 
  CalibHGC m_calibEE, m_calibHEF, m_calibHEB;
  
  double  offSet_EM;
  double  offSet_Had;

  double m_firstMCpartEta;
  double m_firstMCpartPhi;
  double m_secondMCpartEta;
  double m_secondMCpartPhi;

  double m_muonToMip;

  bool calibInitialized;
  short _debugLevel;
  double speedoflight;

  double _deltaPtOverPtForPfo;
  double _deltaPtOverPtForClusterlessPfo;

};
