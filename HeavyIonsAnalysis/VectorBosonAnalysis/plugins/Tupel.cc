/* -*- c-basic-offset: 2; -*-

   Code by: Bugra Bilin, Kittikul Kovitanggoon, Tomislav Seva, Efe Yazgan,
   Philippe Gras...

*/

#include <map>
#include <string>
#include <fstream>
#include <vector>
#include <memory>
#include <TLorentzVector.h>
#include <stdlib.h>
#include "TH1.h"
#include "TH2.h"
#include "TTree.h"
#include "FWCore/Utilities/interface/TimeOfDay.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Version/interface/GetReleaseVersion.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "FWCore/Common/interface/TriggerNames.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/PatCandidates/interface/TriggerEvent.h"
#include "PhysicsTools/PatUtils/interface/TriggerHelper.h"
#include "DataFormats/PatCandidates/interface/TriggerObjectStandAlone.h"
#include "CondFormats/JetMETObjects/interface/JetCorrectionUncertainty.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/LHEEventProduct.h"
#include "EgammaAnalysis/ElectronTools/interface/EGammaCutBasedEleId.h"
#include "EgammaAnalysis/ElectronTools/interface/ElectronEffectiveArea.h"
#include "RecoEgamma/EgammaTools/interface/ConversionTools.h"
#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/PatCandidates/interface/Photon.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/Tau.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/MET.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "FWCore/Framework/interface/GenericHandle.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "SimDataFormats/PileupSummaryInfo/interface/PileupSummaryInfo.h"
#include "PhysicsTools/Utilities/interface/LumiReWeighting.h"
#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"
//#include "CondFormats/JetMETObjects/interface/JetCorrectorParameters.h"
#include "JetMETCorrections/Objects/interface/JetCorrectionsRecord.h"
#include "JetMETCorrections/Objects/interface/JetCorrector.h"
//#include "CMGTools/External/interface/PileupJetIdentifier.h"
#include "EgammaAnalysis/ElectronTools/interface/PFIsolationEstimator.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/BTauReco/interface/JetTag.h"
#include "SimDataFormats/GeneratorProducts/interface/LHERunInfoProduct.h"

#include "TreeHelper.h"

const double pi = 4*atan(1.);

#define QUOTE2(a) #a
#define QUOTE(a) QUOTE2(a)
const static char* checksum = QUOTE(MYFILE_CHECKSUM);

#define QUICK_FIX_RUND

/**
 * Define a bit of a bit field.
 * 1. Document the bit in the ROOT tree
 * 2. Set the variable with the bit mask (integer with the relevant bit set).
 * The second version DEF_BIT2 omits this setting. It can be useful when we
 * want to avoid defining the variable.
 * 3. Set a map to match bit name to bit mask
 * A field named after label prefixed with a k and with an undercore
 * appended to it must be defined in the class members. This macro will set it.
 * @param bitField: name (verbatim string without quotes) of the event tree branch the bit belongs to
 * @param bitNum: bit position, LSB is 0, MSB is 31
 * @param label: used to build the class field names (k<label>_ for the mask
 * and <label>Map_ for the name->mask map). The label is stored as
 * description in the ROOT file. Use a verbatim string without quotes.
 *
 *  For long description including spaces please use the DEF_BIT_L version
 */
#define DEF_BIT(bitField, bitNum, label)		\
  k ## label ## _ = 1 <<bitNum;				\
  bitField ## Map ## _[#label] = 1LL <<bitNum;		\
  treeHelper_->defineBit(#bitField, bitNum, #label);

#define DEF_BIT2(bitField, bitNum, label)		\
  bitField ## Map ## _[#label] = 1LL <<bitNum;		\
  treeHelper_->defineBit(#bitField, bitNum, #label);

/**
 * See DEF_BIT. Version for long description which can not be used
 * as variable name. The argument desc must be a c string including
 * the quotes or a c-string (char*) type variable.
 */
#define DEF_BIT_L(bitField, bitNum, label, desc)	\
  k ## label ## _ = 1 <<bitNum;				\
  bitField ## Map ## _[#label] = 1 <<bitNum;		\
  treeHelper_->defineBit(#bitField, bitNum, desc);

#define DEF_BIT2_L(bitField, bitNum, label, desc)	\
  bitField ## Map ## _[#label] = 1 <<bitNum;		\
  treeHelper_->defineBit(#bitField, bitNum, desc);


//#name -> "name", name ## _ -> name_
#define ADD_BRANCH_D(name, desc) treeHelper_->addBranch(#name, name ## _, desc)
#define ADD_BRANCH(name) treeHelper_->addBranch(#name, name ## _)
#define ADD_MOMENTUM_BRANCH_D(name, desc) treeHelper_->addMomentumBranch(#name, name ## Pt_, name ## Eta_, name ## Phi_, name ## E_, desc)
#define ADD_MOMENTUM_BRANCH(name) treeHelper_->addMomentumBranch(#name, name ## Pt_, name ## Eta_, name ## Phi_, name ## E_)


class TTree;
class Tupel : public edm::EDAnalyzer {

public:
  /// default constructor
  explicit Tupel(const edm::ParameterSet&);
  /// default destructor
  ~Tupel();

private:
  /// everything that needs to be done before the event loop
  virtual void beginJob() ;
  /// everything that needs to be done during the event loop
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  /// everything that needs to be done after the event loop

  virtual void endJob();

  virtual void endRun(edm::Run const& iRun, edm::EventSetup const&);

  void allocateTrigMap(int ntrigMax);

  void defineBitFields();

  //help function to compute pseudo rapidity of a TLorentz without
  //ROOT exception in case of infinite pseudo rapidity
  double eta(TLorentzVector v){
    if(fabs(v.CosTheta()) < 1) return -0.5 * log((1.0-v.CosTheta())/(1.0+v.CosTheta()));
    else if(v.Pz() > 0) return 10e10;
    else return -10e10;
  }

  //Write Job information tree in the output. The tree
  //contains one event, with general information
  void writeHeader();

  //Retrieves event objets
  void readEvent(const edm::Event& iEvent);

  //Processes MET
  void processMET(const edm::Event& iEvent);

  //Processes vertices
  void processVtx();

  //Processes PU
  void processPu(const edm::Event& iEvent);

  //Processes GenParticles
  void processGenParticles(const edm::Event& iEvent);

  void processGenJets(const edm::Event& iEvent);

  void processPdfInfo(const edm::Event& iEvent);

  void processTrigger(const edm::Event& iEvent);

  void processMuons();

  void processElectrons();

  void processJets();

  void processPhotons();

  /** Check if the argument is one of the
   * HLT trigger considered to be stored and if it is set
   * the corresponding bit.
   */
  void fillTrig(const std::string& trigname);

  void writeTriggerStat();

  /** Checks trigger name, ignoring the version number.
   * @param triggerBasename trigger base name without the version number
   * @param triggerName with possibly the _vNNN version suffix
   * @return true iff triggerName is identical to triggerBasename once the
   * version suffix discarded
   */
  bool compTrigger(const char* triggerBasename, const char* triggerName) const;
  
  // input tags
  //edm::InputTag trigger_;
  //edm::InputTag triggerEvent_;
  //edm::InputTag triggerSummaryLabel_;
  std::string elecMatch_;
  std::string muonMatch_;
  std::string muonMatch2_;
  edm::InputTag photonSrc_;
  edm::InputTag elecSrc_;
  edm::InputTag muonSrc_;
  edm::InputTag tauSrc_;
  edm::InputTag jetSrc_;
  edm::InputTag gjetSrc_;
  edm::InputTag metSrc_;
  // edm::InputTag mSrcRho_;
  // edm::InputTag CaloJet_;
  edm::InputTag vtxSrc_;
  edm::InputTag lheSource_;
  edm::InputTag genParticleSrc_;
  std::vector<edm::InputTag> metSources;
  bool triggerStat_;

  /** Total number of events analyzed so far
   */
  Long64_t analyzedEventCnt_;

  bool photonIdsListed_;
  bool elecIdsListed_;
  bool hltListed_;

  //edm::EDGetTokenT<edm::ValueMap<float> > full5x5SigmaIEtaIEtaMapToken_;

  // ----------member data ---------------------------
  TTree *myTree;
  std::auto_ptr<TreeHelper> treeHelper_;

  //Event
  std::auto_ptr<int>      EvtIsRealData_;
  std::auto_ptr<unsigned> EvtNum_;
  std::auto_ptr<unsigned> EvtRunNum_;
  std::auto_ptr<int> 	  EvtLumiNum_;
  std::auto_ptr<int> 	  EvtBxNum_;
  std::auto_ptr<int> 	  EvtVtxCnt_;
  std::auto_ptr<int> 	  EvtPuCnt_;
  std::auto_ptr<int> 	  EvtPuCntTruth_;
  std::auto_ptr<std::vector<double> > EvtWeights_;
  //std::auto_ptr<float>    EvtFastJetRho_;

  //Trigger
  std::auto_ptr<unsigned>          TrigHlt_;
  std::map<std::string, unsigned>  TrigHltMap_; //bit assignment
  std::auto_ptr<ULong64_t>         TrigHltMu_;
  std::map<std::string, ULong64_t> TrigHltMuMap_; //bit assignment
  std::auto_ptr<ULong64_t>         TrigHltDiMu_;
  std::map<std::string, ULong64_t> TrigHltDiMuMap_; //bit assignment
  std::auto_ptr<ULong64_t>         TrigHltEl_;
  std::map<std::string, ULong64_t> TrigHltElMap_; //bit assignment
  std::auto_ptr<ULong64_t>         TrigHltDiEl_;
  std::map<std::string, ULong64_t> TrigHltDiElMap_; //bit assignment
  std::auto_ptr<ULong64_t>         TrigHltElMu_;
  std::map<std::string, ULong64_t> TrigHltElMuMap_; //bit assignment
  struct  TrigHltMapRcd {
    TrigHltMapRcd(): pMap(0), pTrig(0) {}
    TrigHltMapRcd(std::map<std::string, ULong64_t>* pMap_, ULong64_t* pTrig_): pMap(pMap_), pTrig(pTrig_) {
      assert(pTrig_);
    }
    //    TrigHltMapRcd(const TrigHltMapRcd& a){ this->pMap = a.pMap; this->pTrig = a.pTrig; }
    std::map<std::string, ULong64_t>* pMap;
    ULong64_t* pTrig;
  };
  std::vector<TrigHltMapRcd> trigHltMapList_; //list of trigger maps.

  //Missing energy
  std::auto_ptr<std::vector<float> > METPt_;
  std::auto_ptr<std::vector<float> > METPx_;
  std::auto_ptr<std::vector<float> > METPy_;
  std::auto_ptr<std::vector<float> > METPz_;
  std::auto_ptr<std::vector<float> > METE_;
  std::auto_ptr<std::vector<float> > METsigx2_;
  std::auto_ptr<std::vector<float> > METsigxy_;
  std::auto_ptr<std::vector<float> > METsigy2_;
  std::auto_ptr<std::vector<float> > METsig_;

  //Generator level leptons, dressed
  std::auto_ptr<std::vector<float> > 	GLepDr01Pt_;
  std::auto_ptr<std::vector<float> > 	GLepDr01Eta_;
  std::auto_ptr<std::vector<float> > 	GLepDr01Phi_;
  std::auto_ptr<std::vector<float> > 	GLepDr01E_;
  std::auto_ptr<std::vector<int> >      GLepDr01Id_;
  std::auto_ptr<std::vector<int> >      GLepDr01St_;
  std::auto_ptr<std::vector<int> >      GLepDr01MomId_;

  //Generator level leptons, not-dressed
  std::auto_ptr<std::vector<float> > 	GLepBarePt_;
  std::auto_ptr<std::vector<float> > 	GLepBareEta_;
  std::auto_ptr<std::vector<float> > 	GLepBarePhi_;
  std::auto_ptr<std::vector<float> > 	GLepBareE_;
  std::auto_ptr<std::vector<int> >      GLepBareId_;
  std::auto_ptr<std::vector<int> > 	GLepBareSt_;
  std::auto_ptr<std::vector<int> > 	GLepBareMomId_;

  //Generator level leptons, status 3
  std::auto_ptr<std::vector<float> > GLepSt3Pt_;
  std::auto_ptr<std::vector<float> > GLepSt3Eta_;
  std::auto_ptr<std::vector<float> > GLepSt3Phi_;
  std::auto_ptr<std::vector<float> > GLepSt3E_;
  std::auto_ptr<std::vector<float> > GLepSt3M_;
  std::auto_ptr<std::vector<int> >   GLepSt3Id_;
  std::auto_ptr<std::vector<int> >   GLepSt3St_;
  std::auto_ptr<std::vector<int> >   GLepSt3Mother0Id_;
  std::auto_ptr<std::vector<int> >   GLepSt3MotherCnt_;

  //Generator level photons
  std::auto_ptr<std::vector<float> > GPhotPt_;
  std::auto_ptr<std::vector<float> > GPhotEta_;
  std::auto_ptr<std::vector<float> > GPhotPhi_;
  std::auto_ptr<std::vector<float> > GPhotE_;
  std::auto_ptr<std::vector<int> >   GPhotMotherId_;
  std::auto_ptr<std::vector<int> >   GPhotSt_;
  std::auto_ptr<std::vector<float> > GPhotIsoEDR03_;
  std::auto_ptr<std::vector<float> > GPhotIsoEDR04_;
  std::auto_ptr<std::vector<float> > GPhotIsoEDR05_;
  std::auto_ptr<std::vector<float> > GPhotIsoSumPtDR03_;
  std::auto_ptr<std::vector<float> > GPhotIsoSumPtDR04_;
  std::auto_ptr<std::vector<float> > GPhotIsoSumPtDR05_;

  //Photons in the vicinity of the leptons
  std::auto_ptr<std::vector<float> > GLepClosePhotPt_;
  std::auto_ptr<std::vector<float> > GLepClosePhotEta_;
  std::auto_ptr<std::vector<float> > GLepClosePhotPhi_;
  std::auto_ptr<std::vector<float> > GLepClosePhotE_;
  std::auto_ptr<std::vector<float> > GLepClosePhotM_;
  std::auto_ptr<std::vector<int> >   GLepClosePhotId_;
  std::auto_ptr<std::vector<int> >   GLepClosePhotMother0Id_;
  std::auto_ptr<std::vector<int> >   GLepClosePhotMotherCnt_;
  std::auto_ptr<std::vector<int> >   GLepClosePhotSt_;

  //Gen Jets
  std::auto_ptr<std::vector<float> > GJetAk04Pt_;
  std::auto_ptr<std::vector<float> > GJetAk04Eta_;
  std::auto_ptr<std::vector<float> > GJetAk04Phi_;
  std::auto_ptr<std::vector<float> > GJetAk04E_;
  std::auto_ptr<std::vector<float> > GJetAk04Id_;
  std::auto_ptr<std::vector<float> > GJetAk04PuId_;
  std::auto_ptr<std::vector<float> > GJetAk04ChFrac_;
  std::auto_ptr<std::vector<int> >   GJetAk04ConstCnt_;
  std::auto_ptr<std::vector<int> >   GJetAk04ConstId_;
  std::auto_ptr<std::vector<float> > GJetAk04ConstPt_;
  std::auto_ptr<std::vector<float> > GJetAk04ConstEta_;
  std::auto_ptr<std::vector<float> > GJetAk04ConstPhi_;
  std::auto_ptr<std::vector<float> > GJetAk04ConstE_;

  //Exta generator information
  std::auto_ptr<std::vector<int> >   GPdfId1_;
  std::auto_ptr<std::vector<int> >   GPdfId2_;
  std::auto_ptr<std::vector<float> > GPdfx1_;
  std::auto_ptr<std::vector<float> > GPdfx2_;
  std::auto_ptr<std::vector<float> > GPdfScale_;
  std::auto_ptr<float>               GBinningValue_;
  std::auto_ptr<int>                 GNup_;


  ///Muons
  std::auto_ptr<std::vector<float> > 	MuPt_;
  std::auto_ptr<std::vector<float> > 	MuEta_;
  std::auto_ptr<std::vector<float> > 	MuPhi_;
  std::auto_ptr<std::vector<float> > 	MuE_;
  std::auto_ptr<std::vector<unsigned> > MuId_;
  std::map<std::string, unsigned>       MuIdMap_; //bit assignment
  std::auto_ptr<std::vector<unsigned> > MuIdTight_;
  std::map<std::string, unsigned>    	MuIdTightMap_; //bit assignment
  std::auto_ptr<std::vector<float> > 	MuCh_;
  std::auto_ptr<std::vector<float> > 	MuVtxZ_;
  std::auto_ptr<std::vector<float> > 	MuDxy_;
  //std::auto_ptr<std::vector<float> > 	MuIsoRho_;
  std::auto_ptr<std::vector<float> > 	MuPfIso_;
  std::auto_ptr<std::vector<float> > 	MuType_;
  std::map<std::string, unsigned>    	MuTypeMap_; //bit assignment
  std::auto_ptr<std::vector<float> > 	MuIsoTkIsoAbs_;
  std::auto_ptr<std::vector<float> > 	MuIsoTkIsoRel_;
  std::auto_ptr<std::vector<float> > 	MuIsoCalAbs_;
  std::auto_ptr<std::vector<float> > 	MuIsoCombRel_;
  std::auto_ptr<std::vector<float> > 	MuTkNormChi2_;
  std::auto_ptr<std::vector<int> > 	MuTkHitCnt_;
  std::auto_ptr<std::vector<int> > 	MuMatchedStationCnt_;
  std::auto_ptr<std::vector<float> > 	MuDz_;
  std::auto_ptr<std::vector<int> > 	MuPixelHitCnt_;
  std::auto_ptr<std::vector<int> > 	MuTkLayerCnt_;
  std::auto_ptr<std::vector<float> > 	MuPfIsoChHad_;
  std::auto_ptr<std::vector<float> > 	MuPfIsoNeutralHad_;
  std::auto_ptr<std::vector<float> > 	MuPfIsoRawRel_;
  std::auto_ptr<std::vector<unsigned> > MuHltMatch_;

  //Electrons
  std::auto_ptr<std::vector<float> > 	ElPt_;
  std::auto_ptr<std::vector<float> > 	ElEta_;
  std::auto_ptr<std::vector<float> > 	ElEtaSc_;
  std::auto_ptr<std::vector<float> > 	ElPhi_;
  std::auto_ptr<std::vector<float> > 	ElE_;
  std::auto_ptr<std::vector<float> > 	ElCh_;
  std::auto_ptr<std::vector<unsigned> > ElId_;
  std::map<std::string, unsigned>    	ElIdMap_; //bit assignment
  std::auto_ptr<std::vector<float> > 	ElMvaTrig_;
  std::auto_ptr<std::vector<float> > 	ElMvaNonTrig_;
  std::auto_ptr<std::vector<float> > 	ElMvaPresel_;
  std::auto_ptr<std::vector<float> > 	ElDEtaTkScAtVtx_;
  std::auto_ptr<std::vector<float> > 	ElDPhiTkScAtVtx_;
  std::auto_ptr<std::vector<float> > 	ElHoE_;
  std::auto_ptr<std::vector<float> > 	ElSigmaIetaIeta_;
  std::auto_ptr<std::vector<float> > 	ElSigmaIetaIetaFull5x5_;
  std::auto_ptr<std::vector<float> > 	ElEinvMinusPinv_;
  std::auto_ptr<std::vector<float> > 	ElD0_;
  std::auto_ptr<std::vector<float> > 	ElDz_;
  std::auto_ptr<std::vector<int> >   	ElExpectedMissingInnerHitCnt_;
  std::auto_ptr<std::vector<int> >   	ElPassConvVeto_;
  std::auto_ptr<std::vector<unsigned> > ElHltMatch_;
  std::auto_ptr<std::vector<float> > 	ElPfIsoChHad_;
  std::auto_ptr<std::vector<float> > 	ElPfIsoNeutralHad_;
  std::auto_ptr<std::vector<float> > 	ElPfIsoIso_;
  std::auto_ptr<std::vector<float> > 	ElPfIsoPuChHad_;
  std::auto_ptr<std::vector<float> > 	ElPfIsoRaw_;
  std::auto_ptr<std::vector<float> > 	ElPfIsoDbeta_;
  //std::auto_ptr<std::vector<float> > 	ElPfIsoRho_;
  std::auto_ptr<std::vector<float> > 	ElAEff_;

  std::auto_ptr<std::vector<float> > 	 charged_;
  std::auto_ptr<std::vector<float> > 	 photon_;
  std::auto_ptr<std::vector<float> > 	 neutral_;
  std::auto_ptr<std::vector<float> > 	 charged_Tom_;
  std::auto_ptr<std::vector<float> > 	 photon_Tom_;
  std::auto_ptr<std::vector<float> > 	 neutral_Tom_;

  //Photons
  //photon momenta
  std::auto_ptr<std::vector<float> > PhotPt_;
  std::auto_ptr<std::vector<float> > PhotEta_;
  std::auto_ptr<std::vector<float> > PhotPhi_;
  std::auto_ptr<std::vector<float> > PhotScRawE_;
  std::auto_ptr<std::vector<float> > PhotScEta_;
  std::auto_ptr<std::vector<float> > PhotScPhi_;


  //photon isolations
  std::auto_ptr<std::vector<float> > PhotIsoEcal_;
  std::auto_ptr<std::vector<float> > PhotIsoHcal_;
  std::auto_ptr<std::vector<float> > PhotIsoTk_;
  std::auto_ptr<std::vector<float> > PhotPfIsoChHad_;
  std::auto_ptr<std::vector<float> > PhotPfIsoNeutralHad_;
  std::auto_ptr<std::vector<float> > PhotPfIsoPhot_;
  std::auto_ptr<std::vector<float> > PhotPfIsoPuChHad_;
  std::auto_ptr<std::vector<float> > PhotPfIsoEcalClus_;
  std::auto_ptr<std::vector<float> > PhotPfIsoHcalClus_;

  //photon cluster shapes
  std::auto_ptr<std::vector<float> > PhotE3x3_;
  std::auto_ptr<std::vector<float> > PhotE1x5_;
  std::auto_ptr<std::vector<float> > PhotE1x3_;
  std::auto_ptr<std::vector<float> > PhotE2x2_;
  std::auto_ptr<std::vector<float> > PhotE2x5_;
  std::auto_ptr<std::vector<float> > PhotE5x5_;
  std::auto_ptr<std::vector<float> > PhotSigmaIetaIeta_;
  std::auto_ptr<std::vector<float> > PhotSigmaIetaIphi_;
  std::auto_ptr<std::vector<float> > PhotSigmaIphiIphi_;
  std::auto_ptr<std::vector<float> > PhotEtaWidth_;
  std::auto_ptr<std::vector<float> > PhotPhiWidth_;

  //photon preshower
  std::auto_ptr<std::vector<float> > PhotEsE_;
  std::auto_ptr<std::vector<float> > PhotEsSigmaIxIx_;
  std::auto_ptr<std::vector<float> > PhotEsSigmaIyIy_;
  std::auto_ptr<std::vector<float> > PhotEsSigmaIrIr_;

  //photon id (bit field)
  std::auto_ptr<std::vector<unsigned> > PhotId_;
  std::auto_ptr<std::vector<float> >    PhotHoE_;
  std::auto_ptr<std::vector<bool> >     PhotHasPixelSeed_;

  //photon timing
  std::auto_ptr<std::vector<float> > PhotTime_;

  ///PF Jets
  std::auto_ptr<std::vector<float> > JetAk04Pt_;
  std::auto_ptr<std::vector<float> > JetAk04Eta_;
  std::auto_ptr<std::vector<float> > JetAk04Phi_;
  std::auto_ptr<std::vector<float> > JetAk04E_;
  std::auto_ptr<std::vector<float> > JetAk04Id_;
  std::auto_ptr<std::vector<bool> >  JetAk04PuId_;
  std::auto_ptr<std::vector<float> > JetAk04PuMva_;
  std::auto_ptr<std::vector<float> > JetAk04RawPt_;
  std::auto_ptr<std::vector<float> > JetAk04RawE_;
  std::auto_ptr<std::vector<float> > JetAk04HfHadE_;
  std::auto_ptr<std::vector<float> > JetAk04HfEmE_;
  std::auto_ptr<std::vector<float> > JetAk04JetBetaClassic_;
  std::auto_ptr<std::vector<float> > JetAk04JetBeta_;
  std::auto_ptr<std::vector<float> > JetAk04JetBetaStar_;
  std::auto_ptr<std::vector<float> > JetAk04JetBetaStarClassic_;
  std::auto_ptr<std::vector<float> > JetAk04ChHadFrac_;
  std::auto_ptr<std::vector<float> > JetAk04NeutralHadAndHfFrac_;
  std::auto_ptr<std::vector<float> > JetAk04ChEmFrac_;
  std::auto_ptr<std::vector<float> > JetAk04NeutralEmFrac_;
  std::auto_ptr<std::vector<float> > JetAk04ChMult_;
  std::auto_ptr<std::vector<float> > JetAk04ConstCnt_;
  std::auto_ptr<std::vector<float> > JetAk04BTagCsv_;
  std::auto_ptr<std::vector<float> > JetAk04BTagCsvV1_;
  std::auto_ptr<std::vector<float> > JetAk04BTagCsvSLV1_;
  std::auto_ptr<std::vector<float> > JetAk04BDiscCisvV2_;
  std::auto_ptr<std::vector<float> > JetAk04BDiscJp_;
  std::auto_ptr<std::vector<float> > JetAk04BDiscBjp_;
  std::auto_ptr<std::vector<float> > JetAk04BDiscTche_;
  std::auto_ptr<std::vector<float> > JetAk04BDiscTchp_;
  std::auto_ptr<std::vector<float> > JetAk04BDiscSsvhe_;
  std::auto_ptr<std::vector<float> > JetAk04BDiscSsvhp_;
  std::auto_ptr<std::vector<float> > JetAk04PartFlav_;
  std::auto_ptr<std::vector<float> > JetAk04JecUncUp_;
  std::auto_ptr<std::vector<float> > JetAk04JecUncDwn_;
  std::auto_ptr<std::vector<int> >   JetAk04ConstId_;
  std::auto_ptr<std::vector<float> > JetAk04ConstPt_;
  std::auto_ptr<std::vector<float> > JetAk04ConstEta_;
  std::auto_ptr<std::vector<float> > JetAk04ConstPhi_;
  std::auto_ptr<std::vector<float> > JetAk04ConstE_;
  std::auto_ptr<std::vector<int> >   JetAk04GenJet_;

  //bits
  unsigned kMuIdLoose_;
  unsigned kMuIdCustom_;
  unsigned kGlobMu_;
  unsigned kTkMu_;
  unsigned kPfMu_;

  unsigned kHLT_HIDoublePhoton15_Eta1p5_Mass50_1000_; //FIXME
  unsigned kHLT_HIL2Mu15_;

  ///Event objects
  edm::Handle<GenParticleCollection> genParticles_h;
  const GenParticleCollection* genParticles;
  edm::Handle<edm::View<pat::Muon> > muons;
  const edm::View<pat::Muon> * muon;
  edm::Handle<vector<pat::Electron> > electrons;
  const vector<pat::Electron>  *electron;
  edm::Handle<reco::ConversionCollection> conversions_h;
  edm::Handle<edm::View<pat::Tau> > taus;
  edm::Handle<edm::View<pat::Jet> > jets;
  const edm::View<pat::Jet> * jettt;
  edm::Handle<edm::View<pat::MET> > mets;
  const edm::View<pat::Photon>  *photons;
  edm::Handle<edm::View<reco::Vertex> >  pvHandle;
  edm ::Handle<reco::VertexCollection> vtx_h;
  const edm::View<reco::Vertex> * vtxx;
  //double rhoIso;
  edm::Handle<reco::BeamSpot> beamSpotHandle;
  reco::BeamSpot beamSpot;
  //edm::Handle<double> rho;
  edm::Handle<reco::GenJetCollection> genjetColl_;
  ///

  std::vector<std::vector<int> > trigAccept_;
  std::vector<std::string> trigNames_;


  std::vector<bool> elIdEnabled_;
  
  bool trigStatValid_;

  //  int singleMuOnly_;

  struct TrigSorter{
    TrigSorter(Tupel* t): tupel_(t){}
    bool operator()(int i, int j) const{
      return tupel_->trigAccept_[1+i][0] > tupel_->trigAccept_[1+j][0];
    }
    Tupel* tupel_;
  };

};

using namespace std;
using namespace reco;
//int ccnevent=0;
Tupel::Tupel(const edm::ParameterSet& iConfig):
  //trigger_( iConfig.getParameter< edm::InputTag >( "trigger" ) ),
  //triggerEvent_( iConfig.getParameter< edm::InputTag >( "triggerEvent" ) ),
  //triggerSummaryLabel_( iConfig.getParameter< edm::InputTag >( "triggerSummaryLabel" ) ), //added by jyhan
  elecMatch_( iConfig.getParameter< std::string >( "elecMatch" ) ),
  muonMatch_( iConfig.getParameter< std::string >( "muonMatch" ) ),
  muonMatch2_( iConfig.getParameter< std::string >( "muonMatch2" ) ),
  photonSrc_(iConfig.getUntrackedParameter<edm::InputTag>("photonSrc")),
  elecSrc_(iConfig.getUntrackedParameter<edm::InputTag>("electronSrc")),
  muonSrc_(iConfig.getUntrackedParameter<edm::InputTag>("muonSrc")),
  //tauSrc_(iConfig.getUntrackedParameter<edm::InputTag>("tauSrc" )),
  jetSrc_(iConfig.getUntrackedParameter<edm::InputTag>("jetSrc" )),
  gjetSrc_(iConfig.getUntrackedParameter<edm::InputTag>("gjetSrc" )),
  metSrc_(iConfig.getUntrackedParameter<edm::InputTag>("metSrc" )),
  //mSrcRho_(iConfig.getUntrackedParameter<edm::InputTag>("mSrcRho" )),
  //CaloJet_(iConfig.getUntrackedParameter<edm::InputTag>("CalojetLabel")),
  vtxSrc_(iConfig.getUntrackedParameter<edm::InputTag>("vtxSrc" )),
  lheSource_(iConfig.getUntrackedParameter<edm::InputTag>("lheSource")),
  genParticleSrc_(iConfig.getUntrackedParameter<edm::InputTag >("genSrc")),
  metSources(iConfig.getParameter<std::vector<edm::InputTag> >("metSource")),
  triggerStat_(iConfig.getUntrackedParameter<bool>("triggerStat", false)),
  analyzedEventCnt_(0),
  photonIdsListed_(false),
  elecIdsListed_(false),
  hltListed_(false),
  trigStatValid_(true)
  //singleMuOnly_(0)
  //full5x5SigmaIEtaIEtaMapToken_(consumes<edm::ValueMap<float> >(iConfig.getParameter<edm::InputTag>("full5x5SigmaIEtaIEtaMap")))
{
}

Tupel::~Tupel()
{
}

void Tupel::defineBitFields(){

  trigHltMapList_.push_back(TrigHltMapRcd(&TrigHltMuMap_,   TrigHltMu_.get()));
  trigHltMapList_.push_back(TrigHltMapRcd(&TrigHltDiMuMap_, TrigHltDiMu_.get()));
  trigHltMapList_.push_back(TrigHltMapRcd(&TrigHltElMap_,   TrigHltEl_.get()));
  trigHltMapList_.push_back(TrigHltMapRcd(&TrigHltDiElMap_, TrigHltDiEl_.get()));
  trigHltMapList_.push_back(TrigHltMapRcd(&TrigHltElMuMap_, TrigHltElMu_.get()));

  DEF_BIT(TrigHlt, 0, HLT_HIDoublePhoton15_Eta1p5_Mass50_1000); //FIXME
  DEF_BIT(TrigHlt, 1, HLT_HIL2Mu15);

  DEF_BIT2(TrigHltDiMu, 1 ,HLT_HIL1DoubleMu0);
  DEF_BIT2(TrigHltDiMu, 2 ,HLT_HIL1DoubleMu10);
  DEF_BIT2(TrigHltDiMu, 3 ,HLT_HIL2DoubleMu0_NHitQ);
  DEF_BIT2(TrigHltDiMu, 4 ,HLT_HIL3DoubleMu0_OS_m2p5to4p5);
  DEF_BIT2(TrigHltDiMu, 5 ,HLT_HIL3DoubleMu0_OS_m7to14);
  DEF_BIT2(TrigHltDiMu, 6 ,HLT_HIL1DoubleMu0_2HF);
  DEF_BIT2(TrigHltDiMu, 7 ,HLT_HIL1DoubleMu0_2HF0);
  DEF_BIT2(TrigHltDiMu, 8 ,HLT_HIL2DoubleMu0_NHitQ_2HF);
  DEF_BIT2(TrigHltDiMu, 9 ,HLT_HIL2DoubleMu0_NHitQ_2HF0);
  DEF_BIT2(TrigHltDiMu, 10,HLT_HIL1DoubleMu0_2HF_Cent30100);
  DEF_BIT2(TrigHltDiMu, 11,HLT_HIL1DoubleMu0_2HF0_Cent30100);
  DEF_BIT2(TrigHltDiMu, 12,HLT_HIL2DoubleMu0_2HF_Cent30100_NHitQ);
  DEF_BIT2(TrigHltDiMu, 13,HLT_HIL1DoubleMu0_Cent30);
  DEF_BIT2(TrigHltDiMu, 14,HLT_HIL2DoubleMu0_2HF0_Cent30100_NHitQ);
  DEF_BIT2(TrigHltDiMu, 15,HLT_HIL2DoubleMu0_Cent30_OS_NHitQ);
  DEF_BIT2(TrigHltDiMu, 16,HLT_HIL2DoubleMu0_Cent30_NHitQ);
  DEF_BIT2(TrigHltDiMu, 17,HLT_HIL3DoubleMu0_Cent30);
  DEF_BIT2(TrigHltDiMu, 18,HLT_HIL3DoubleMu0_Cent30_OS_m2p5to4p5);
  DEF_BIT2(TrigHltDiMu, 19,HLT_HIL3DoubleMu0_Cent30_OS_m7to14);

  DEF_BIT2(TrigHltMu, 1 , HLT_HIL2Mu3_NHitQ10);
  DEF_BIT2(TrigHltMu, 2 , HLT_HIL3Mu3_NHitQ15);
  DEF_BIT2(TrigHltMu, 3 , HLT_HIL2Mu5_NHitQ10);
  DEF_BIT2(TrigHltMu, 4 , HLT_HIL3Mu5_NHitQ15);
  DEF_BIT2(TrigHltMu, 5 , HLT_HIL2Mu7_NHitQ10);
  DEF_BIT2(TrigHltMu, 6 , HLT_HIL3Mu7_NHitQ15);
  DEF_BIT2(TrigHltMu, 7 , HLT_HIL2Mu15);
  DEF_BIT2(TrigHltMu, 8 , HLT_HIL3Mu15);
  DEF_BIT2(TrigHltMu, 9 , HLT_HIL2Mu20);
  DEF_BIT2(TrigHltMu, 10, HLT_HIL3Mu20);
  DEF_BIT2(TrigHltMu, 11, HLT_HIL2Mu3_NHitQ10_2HF);
  DEF_BIT2(TrigHltMu, 12, HLT_HIL2Mu3_NHitQ10_2HF0);
  DEF_BIT2(TrigHltMu, 13, HLT_HIL3Mu3_NHitQ15_2HF);
  DEF_BIT2(TrigHltMu, 14, HLT_HIL3Mu3_NHitQ15_2HF0);
  DEF_BIT2(TrigHltMu, 15, HLT_HIL2Mu5_NHitQ10_2HF);
  DEF_BIT2(TrigHltMu, 16, HLT_HIL2Mu5_NHitQ10_2HF0);
  DEF_BIT2(TrigHltMu, 17, HLT_HIL3Mu5_NHitQ15_2HF);
  DEF_BIT2(TrigHltMu, 18, HLT_HIL3Mu5_NHitQ15_2HF0);
  DEF_BIT2(TrigHltMu, 19, HLT_HIL2Mu7_NHitQ10_2HF);
  DEF_BIT2(TrigHltMu, 20, HLT_HIL2Mu7_NHitQ10_2HF0);
  DEF_BIT2(TrigHltMu, 21, HLT_HIL3Mu7_NHitQ15_2HF);
  DEF_BIT2(TrigHltMu, 22, HLT_HIL3Mu7_NHitQ15_2HF0);
  DEF_BIT2(TrigHltMu, 23, HLT_HIL2Mu15_2HF);
  DEF_BIT2(TrigHltMu, 24, HLT_HIL2Mu15_2HF0);
  DEF_BIT2(TrigHltMu, 25, HLT_HIL3Mu15_2HF);
  DEF_BIT2(TrigHltMu, 26, HLT_HIL3Mu15_2HF0);
  DEF_BIT2(TrigHltMu, 27, HLT_HIL2Mu20_2HF);
  DEF_BIT2(TrigHltMu, 28, HLT_HIL2Mu20_2HF0);
  DEF_BIT2(TrigHltMu, 29, HLT_HIL3Mu20_2HF);
  DEF_BIT2(TrigHltMu, 30, HLT_HIL3Mu20_2HF0);

  DEF_BIT2(TrigHltEl, 1,  HLT_HISinglePhoton10_Eta1p5);
  DEF_BIT2(TrigHltEl, 2,  HLT_HISinglePhoton15_Eta1p5);
  DEF_BIT2(TrigHltEl, 3,  HLT_HISinglePhoton20_Eta1p5);
  DEF_BIT2(TrigHltEl, 4,  HLT_HISinglePhoton30_Eta1p5);
  DEF_BIT2(TrigHltEl, 5,  HLT_HISinglePhoton40_Eta1p5);
  DEF_BIT2(TrigHltEl, 6,  HLT_HISinglePhoton50_Eta1p5);
  DEF_BIT2(TrigHltEl, 7,  HLT_HISinglePhoton60_Eta1p5);
  DEF_BIT2(TrigHltEl, 8,  HLT_HISinglePhoton10_Eta3p1);
  DEF_BIT2(TrigHltEl, 9,  HLT_HISinglePhoton15_Eta3p1);
  DEF_BIT2(TrigHltEl, 10, HLT_HISinglePhoton20_Eta3p1);
  DEF_BIT2(TrigHltEl, 11, HLT_HISinglePhoton30_Eta3p1);
  DEF_BIT2(TrigHltEl, 12, HLT_HISinglePhoton40_Eta3p1);
  DEF_BIT2(TrigHltEl, 13, HLT_HISinglePhoton50_Eta3p1);
  DEF_BIT2(TrigHltEl, 14, HLT_HISinglePhoton60_Eta3p1);

  DEF_BIT2(TrigHltEl, 15, HLT_HISinglePhoton10_Eta1p5_Cent50_100);
  DEF_BIT2(TrigHltEl, 16, HLT_HISinglePhoton15_Eta1p5_Cent50_100);
  DEF_BIT2(TrigHltEl, 17, HLT_HISinglePhoton20_Eta1p5_Cent50_100);
  DEF_BIT2(TrigHltEl, 18, HLT_HISinglePhoton30_Eta1p5_Cent50_100);
  DEF_BIT2(TrigHltEl, 19, HLT_HISinglePhoton40_Eta1p5_Cent50_100);
  DEF_BIT2(TrigHltEl, 20, HLT_HISinglePhoton10_Eta1p5_Cent30_100);
  DEF_BIT2(TrigHltEl, 21, HLT_HISinglePhoton15_Eta1p5_Cent30_100);
  DEF_BIT2(TrigHltEl, 22, HLT_HISinglePhoton20_Eta1p5_Cent30_100);
  DEF_BIT2(TrigHltEl, 23, HLT_HISinglePhoton30_Eta1p5_Cent30_100);
  DEF_BIT2(TrigHltEl, 24, HLT_HISinglePhoton40_Eta1p5_Cent30_100);
  DEF_BIT2(TrigHltEl, 25, HLT_HISinglePhoton40_Eta2p1);
  DEF_BIT2(TrigHltEl, 26, HLT_HISinglePhoton10_Eta3p1_Cent50_100);
  DEF_BIT2(TrigHltEl, 27, HLT_HISinglePhoton15_Eta3p1_Cent50_100);
  DEF_BIT2(TrigHltEl, 28, HLT_HISinglePhoton20_Eta3p1_Cent50_100);
  DEF_BIT2(TrigHltEl, 29, HLT_HISinglePhoton30_Eta3p1_Cent50_100);
  DEF_BIT2(TrigHltEl, 30, HLT_HISinglePhoton40_Eta3p1_Cent50_100);
  DEF_BIT2(TrigHltEl, 31, HLT_HISinglePhoton10_Eta3p1_Cent30_100);
  DEF_BIT2(TrigHltEl, 32, HLT_HISinglePhoton15_Eta3p1_Cent30_100);
  DEF_BIT2(TrigHltEl, 33, HLT_HISinglePhoton20_Eta3p1_Cent30_100);
  DEF_BIT2(TrigHltEl, 34, HLT_HISinglePhoton30_Eta3p1_Cent30_100);
  DEF_BIT2(TrigHltEl, 35, HLT_HISinglePhoton40_Eta3p1_Cent30_100);

  DEF_BIT2(TrigHltDiEl, 1, HLT_HIDoublePhoton15_Eta1p5_Mass50_1000);
  DEF_BIT2(TrigHltDiEl, 2, HLT_HIDoublePhoton15_Eta1p5_Mass50_1000_R9HECut);
  DEF_BIT2(TrigHltDiEl, 3, HLT_HIDoublePhoton15_Eta2p1_Mass50_1000_R9Cut);
  DEF_BIT2(TrigHltDiEl, 4, HLT_HIDoublePhoton15_Eta2p5_Mass50_1000_R9SigmaHECut);

  DEF_BIT2(TrigHltElMu, 1, HLT_HIL2Mu3Eta2p5_HIPhoton10Eta1p5);
  DEF_BIT2(TrigHltElMu, 2, HLT_HIL2Mu3Eta2p5_HIPhoton15Eta1p5);
  DEF_BIT2(TrigHltElMu, 3, HLT_HIL2Mu3Eta2p5_HIPhoton20Eta1p5);
  DEF_BIT2(TrigHltElMu, 4, HLT_HIL2Mu3Eta2p5_HIPhoton30Eta1p5);
  DEF_BIT2(TrigHltElMu, 5, HLT_HIL2Mu3Eta2p5_HIPhoton40Eta1p5);

  DEF_BIT(MuId, 0, MuIdLoose);
  DEF_BIT_L(MuId, 3, MuIdCustom, "Mu Id: isGlobalMuon\n"
	    "&& isPFMuon\n"
	    "&& normChi2 < 10\n"
	    "&& muonHits > 0 && nMatches > \n"
	    "&& dB < 0.2 && dZ < 0.5\n"
	    "&& pixelHits > 0 && trkLayers > 5");
  DEF_BIT(MuType, 0, GlobMu);
  DEF_BIT(MuType, 1, TkMu);
  DEF_BIT(MuType, 2, PfMu);

  int nElId = 0;
  DEF_BIT2(ElId, nElId++, cutBasedElectronID-CSA14-50ns-V1-standalone-veto); 
  DEF_BIT2(ElId, nElId++, cutBasedElectronID-CSA14-50ns-V1-standalone-loose);
  DEF_BIT2(ElId, nElId++, cutBasedElectronID-CSA14-50ns-V1-standalone-medium);
  DEF_BIT2(ElId, nElId++, cutBasedElectronID-CSA14-50ns-V1-standalone-tight);

  DEF_BIT2(ElId, nElId++, cutBasedElectronID-CSA14-PU20bx25-V0-standalone-loose);
  DEF_BIT2(ElId, nElId++, cutBasedElectronID-CSA14-PU20bx25-V0-standalone-medium);
  DEF_BIT2(ElId, nElId++, cutBasedElectronID-CSA14-PU20bx25-V0-standalone-tight);
  DEF_BIT2(ElId, nElId++, cutBasedElectronID-CSA14-PU20bx25-V0-standalone-veto);

  DEF_BIT2(ElId, nElId++,  cutBasedElectronID-Spring15-25ns-V1-standalone-loose);
  DEF_BIT2(ElId, nElId++,  cutBasedElectronID-Spring15-25ns-V1-standalone-medium);
  DEF_BIT2(ElId, nElId++, cutBasedElectronID-Spring15-25ns-V1-standalone-tight);
  DEF_BIT2(ElId, nElId++, cutBasedElectronID-Spring15-25ns-V1-standalone-veto);

  DEF_BIT2(ElId, nElId++, cutBasedElectronID-Spring15-50ns-V1-standalone-loose); 
  DEF_BIT2(ElId, nElId++, cutBasedElectronID-Spring15-50ns-V1-standalone-medium);
  DEF_BIT2(ElId, nElId++, cutBasedElectronID-Spring15-50ns-V1-standalone-tight);
  DEF_BIT2(ElId, nElId++, cutBasedElectronID-Spring15-50ns-V1-standalone-veto);

  elIdEnabled_ = std::vector<bool>(nElId, true);
}

void Tupel::readEvent(const edm::Event& iEvent){
  *EvtNum_        = iEvent.id().event();
  *EvtRunNum_     = iEvent.id().run();
  *EvtLumiNum_    = iEvent.luminosityBlock();
  *EvtBxNum_      = iEvent.bunchCrossing();
  *EvtIsRealData_ = iEvent.isRealData();

  const pat::helper::TriggerMatchHelper matchHelper;
  iEvent.getByLabel(genParticleSrc_, genParticles_h);
  genParticles  = genParticles_h.failedToGet () ? 0 : &*genParticles_h;

  // get muon collection
  iEvent.getByLabel(muonSrc_,muons);
  muon = muons.failedToGet () ? 0 : &*muons;


  // get electron collection
  iEvent.getByLabel(elecSrc_,electrons);
  electron = electrons.failedToGet () ? 0 :  &*electrons;

  // edm::Handle<reco::GsfElectronCollection> els_h;
  // iEvent.getByLabel("gsfElectrons", els_h);
  iEvent.getByLabel(InputTag("reducedEgamma","reducedConversions"), conversions_h);

  // get tau collection
  iEvent.getByLabel(tauSrc_,taus);

  // get jet collection
  iEvent.getByLabel(jetSrc_,jets);
  //  jettt = jets.failedToGet () ? 0 : &*jets ;
  jettt = &*jets;

  // get met collection
  iEvent.getByLabel(metSrc_,mets);

  // get photon collection
  edm::Handle<edm::View<pat::Photon> > hPhotons;
  iEvent.getByLabel(photonSrc_, hPhotons);
  //  photons = hPhotons.failedToGet () ? 0 :  &*hPhotons;
  photons = &*hPhotons;

  //get Gen jets
  iEvent.getByLabel(gjetSrc_, genjetColl_);

  iEvent.getByLabel(vtxSrc_, pvHandle);
  //vtxx = pvHandle.failedToGet () ? 0 : &*pvHandle ;
  vtxx = &*pvHandle;

  iEvent.getByLabel(vtxSrc_, vtx_h);
  //  if(vtxx){
  //    int nvtx=vtx_h->size();
  //    if(nvtx==0) return;
  //    reco::VertexRef primVtx(vtx_h,0);
  //  }

  //iEvent.getByLabel(mSrcRho_, rho);
  //rhoIso=99;
  //if(!rho.failedToGet()) rhoIso = *rho;

  iEvent.getByLabel("offlineBeamSpot", beamSpotHandle);

  if(!beamSpotHandle.failedToGet()) beamSpot = *beamSpotHandle;
}

void Tupel::processMET(const edm::Event& iEvent){
  for(unsigned int imet=0;imet<metSources.size();imet++){
    Handle<View<pat::MET> > metH;
    iEvent.getByLabel(metSources[imet], metH);
    if(!metH.isValid())continue;
    //cout<<"MET"<<imet<<"  "<<metSources[imet]<<"  "<<metH->ptrAt(0)->pt()<<endl;

    METPt_->push_back(metH->ptrAt(0)->pt());
    METPx_->push_back(metH->ptrAt(0)->px());
    METPy_->push_back(metH->ptrAt(0)->py());
    METPz_->push_back(metH->ptrAt(0)->pz());
    METE_->push_back(metH->ptrAt(0)->energy());
    METsigx2_->push_back(metH->ptrAt(0)->getSignificanceMatrix()(0,0));
    METsigxy_->push_back(metH->ptrAt(0)->getSignificanceMatrix()(0,1));
    METsigy2_->push_back(metH->ptrAt(0)->getSignificanceMatrix()(1,1));
    METsig_->push_back(metH->ptrAt(0)->significance());
    //Output object in EDM format
    //std::auto_ptr<llvvMet> metOut(new llvvMet());
    //llvvMet& met = *metOut;

    //////////////////////////////////

    // met.SetPxPyPzE(metH->ptrAt(0)->px(), metH->ptrAt(0)->py(), metH->ptrAt(0)->pz(), metH->ptrAt(0)->energy());
    //met.sigx2 = metH->ptrAt(0)->getSignificanceMatrix()(0,0);
    //met.sigxy = metH->ptrAt(0)->getSignificanceMatrix()(0,1);
    //met.sigy2 = metH->ptrAt(0)->getSignificanceMatrix()(1,1);
    //met.sig   = metH->ptrAt(0)->significance();

    //iEvent.put(metOut, metSources[imet].label()); //save the object to the EvtNum here, to keep it in the loop
  }
}

void Tupel::processVtx(){
  if(vtxx){
    for (edm::View<reco::Vertex>::const_iterator vtx = pvHandle->begin(); vtx != pvHandle->end(); ++vtx){
      if (vtx->isValid() && !vtx->isFake()) ++(*EvtVtxCnt_);
    }
  }
}

void Tupel::processPu(const edm::Event& iEvent){
  Handle<std::vector< PileupSummaryInfo > >  PupInfo;
  iEvent.getByLabel("addPileupInfo", PupInfo);
  if(!PupInfo.failedToGet()){
    std::vector<PileupSummaryInfo>::const_iterator PVI;
    float npT=-1.;
    float npIT=-1.;

    for(PVI = PupInfo->begin(); PVI != PupInfo->end(); ++PVI) {

      int BX = PVI->getBunchCrossing();

      if(BX == 0) {
	npT = PVI->getTrueNumInteractions();
	npIT = PVI->getPU_NumInteractions();
      }
    }
    //////TO CHECK
    *EvtPuCntTruth_ = npT;
    *EvtPuCnt_      = npIT;
  }
  else {
    *EvtPuCntTruth_ = -2.;
    *EvtPuCnt_ = -2.;
  }
}

void Tupel::processGenParticles(const edm::Event& iEvent){
  const std::vector<reco::GenParticle> & gen = *genParticles_h;
  for (size_t i=0; i<genParticles->size(); ++i){
    TLorentzVector genR1DressLep1(0,0,0,0);
    //      TLorentzVector genPho(0,0,0,0);
    int st = gen[i].status();
    int id = gen[i].pdgId();


    // ---- consider only gen  photons -----------------
    if (TMath::Abs(id) == 22) {
      TLorentzVector genPho(0, 0, 0, 0);
      genPho.SetPtEtaPhiE(gen[i].pt(),gen[i].eta(),gen[i].phi(),gen[i].energy());

      GPhotPt_->push_back(gen[i].pt());
      GPhotEta_->push_back(gen[i].eta());
      GPhotPhi_->push_back(gen[i].phi());
      GPhotE_->push_back(gen[i].energy());
      GPhotMotherId_->push_back(gen[i].mother()->pdgId());
      GPhotSt_->push_back(gen[i].status());

      //--- search for stable particles around photon in DR=0.3,0.4 and 0.5 cone around the photon
      //--- for particle isolation, don't check for invisible particles
      float isoEDR03 = 0;
      float isoEDR04 = 0;
      float isoEDR05 = 0;
      float isoSumPtDR03 = 0;
      float isoSumPtDR04 = 0;
      float isoSumPtDR05 = 0;
      for (size_t j=0; j<genParticles->size(); ++j) {
	//remove the genPho photon from the list of particles
	if (j != i) {
	  //don't use neutrinos
	  if (gen[j].status() == 1 && abs(gen[j].pdgId()) !=12 && abs(gen[j].pdgId()) !=14 && abs(gen[j].pdgId()) !=16){
	    TLorentzVector partP4GEN(0, 0, 0, 0);
	    partP4GEN.SetPtEtaPhiE(gen[j].pt(),gen[j].eta(),gen[j].phi(),gen[j].energy());
	    float DR = partP4GEN.DeltaR(genPho);
	    if (DR<0.5) {
	      isoEDR05 += gen[j].energy();
	      isoSumPtDR05 += gen[j].pt();
	    }
	    if (DR<0.4){
	      isoEDR04 += gen[j].energy();
	      isoSumPtDR04 += gen[j].pt();
	    }
	    if (DR<0.3) {
	      isoEDR03 += gen[j].energy();
	      isoSumPtDR03 += gen[j].pt();
	    }
	  }
	}
      } //gen photon isolation
      GPhotIsoEDR03_->push_back(isoEDR03);
      GPhotIsoEDR04_->push_back(isoEDR04);
      GPhotIsoEDR05_->push_back(isoEDR05);
      GPhotIsoSumPtDR03_->push_back(isoSumPtDR03);
      GPhotIsoSumPtDR04_->push_back(isoSumPtDR04);
      GPhotIsoSumPtDR05_->push_back(isoSumPtDR05);
    } // if |id| == 22


    if(gen[i].numberOfMothers()){
      //        if (st!=3 && fabs(id)!=13&& fabs(id)!=11 && fabs(id)!=22 && fabs(id)!=23) continue;
      // if(abs(st)==13 ||abs(st)==12||abs(st)==11||abs(st)==23 ||abs(st)==22||abs(st)==21||abs(st)==61 )cout<<"AAA "<<gen[i].numberOfMothers() <<"  "<< gen[i].mother()->pdgId()<<"  "<< gen[i].pdgId()<<"  "<<st<<"  "<<gen[i].px()<<"  "<<gen[i].py()<<"  "<<gen[i].pz()<<"  "<<gen[i].energy()<<endl;
      if (abs(st)==23 ||abs(st)==22||abs(st)==21||abs(st)==61 ||abs(st)==3 ){
	TLorentzVector genLep3(0,0,0,0);
	if(abs(st)!=21)genLep3.SetPtEtaPhiE(gen[i].pt(),gen[i].eta(),gen[i].phi(),gen[i].energy());
	if(abs(st)==21)genLep3.SetPxPyPzE(0.001,0.001,gen[i].pz(),gen[i].energy());
	GLepSt3Pt_->push_back(genLep3.Pt());
	GLepSt3Eta_->push_back(eta(genLep3));
	GLepSt3Phi_->push_back(genLep3.Phi());
	GLepSt3E_->push_back(genLep3.Energy());
	GLepSt3Mother0Id_->push_back(gen[i].mother()->pdgId());
	GLepSt3Id_->push_back(id);
	GLepSt3St_->push_back(st);
      }
      GLepSt3MotherCnt_->push_back(gen[i].numberOfMothers());
      /* if(abs(id)==15){
      //cout<<gen[i].numberOfMothers() <<"  "<< gen[i].mother()->pdgId()<<"  "<< gen[i].pdgId()<<"  "<<st<<endl;
      n_tau++;
      }*/

      int momId = gen[i].mother()->pdgId();
      if(gen[i].numberOfMothers() ==1 &&  momId != id){
	//if(abs(id)==15)cout<<"DEAD"<<endl;
	continue;
      }
      if (st==1 && (abs(id)==13||abs(id)==11 || abs(id)==15 ||abs(id)==12||abs(id)==14||abs(id)==16) /*&& gen[i].pt() > 0.1 && fabs(gen[i].eta())<3.0*/){

	TLorentzVector genLep1(0,0,0,0);
	genLep1.SetPtEtaPhiE(gen[i].pt(),gen[i].eta(),gen[i].phi(),gen[i].energy());
	TLorentzVector genR1Pho1(0,0,0,0);

	edm::Handle<std::vector<reco::GenParticle> > genpart2;//DONT know why we Need to handle another collection
	iEvent.getByLabel(genParticleSrc_, genpart2);
	const std::vector<reco::GenParticle> & gen2 = *genpart2;
	//LOOP over photons//
	if (st==1 && (abs(id)==13||abs(id)==11)){
	  for(unsigned int j=0; j<genpart2->size(); ++j){
	    if(gen2[j].numberOfMothers()){
	      if( gen2[j].status()!=1|| gen2[j].pdgId()!=22 || gen2[j].energy()<0.000001 /*|| fabs(MomId2)!=fabs(id)*/) continue;
	      TLorentzVector thisPho1(0,0,0,0);
	      thisPho1.SetPtEtaPhiE(gen2[j].pt(),gen2[j].eta(),gen2[j].phi(),gen2[j].energy());
	      double dR = genLep1.DeltaR(thisPho1);
	      if(dR<0.1){
		genR1Pho1+=thisPho1;
	      }

	      if(dR<0.2){
		GLepClosePhotPt_->push_back(thisPho1.Pt());
		GLepClosePhotEta_->push_back(thisPho1.Eta());
		GLepClosePhotPhi_->push_back(thisPho1.Phi());
		GLepClosePhotE_->push_back(thisPho1.Energy());
		GLepClosePhotId_->push_back(gen2[j].pdgId());
		GLepClosePhotMother0Id_->push_back(fabs(gen2[j].mother()->pdgId()));
		GLepClosePhotMotherCnt_->push_back(gen2[j].numberOfMothers());
		GLepClosePhotSt_->push_back(gen2[j].status());
	      }
	    }
	  }
	}

	genR1DressLep1 = genLep1+genR1Pho1;
	GLepDr01Pt_->push_back(genR1DressLep1.Pt());
	GLepDr01Eta_->push_back(genR1DressLep1.Eta());
	GLepDr01Phi_->push_back(genR1DressLep1.Phi());
	GLepDr01E_->push_back(genR1DressLep1.Energy());
	GLepDr01Id_->push_back(id);
	GLepDr01MomId_->push_back(momId);
	GLepDr01St_->push_back(st);

	GLepBarePt_->push_back(genLep1.Pt());
	GLepBareEta_->push_back(genLep1.Eta());
	GLepBarePhi_->push_back(genLep1.Phi());
	GLepBareE_->push_back(genLep1.Energy());
	GLepBareId_->push_back(id);
	GLepBareMomId_->push_back(momId);
	GLepBareSt_->push_back(st);
      }
    }
  }
}

void Tupel::processGenJets(const edm::Event& iEvent){
  //matrix element info
  Handle<LHEEventProduct> lheH;
  iEvent.getByLabel(lheSource_,lheH);//to be modularized!!!
  if(lheH.isValid()) *GNup_ = lheH->hepeup().NUP;
  if(!genjetColl_.failedToGet()){
    const reco::GenJetCollection & genjet = *genjetColl_;
    for(unsigned int k=0; k < genjet.size(); ++k){
      GJetAk04Pt_->push_back(genjet[k].pt());
      GJetAk04Eta_->push_back(genjet[k].eta());
      GJetAk04Phi_->push_back(genjet[k].phi());
      GJetAk04E_->push_back(genjet[k].energy());
      //double isChargedJet=false;
      //double chargedFraction = 0.;
      //std::vector<const GenParticle*> mcparticles = genjet[k].getGenConstituents();
      //for(std::vector <const GenParticle*>::const_iterator thepart =mcparticles.begin();thepart != mcparticles.end(); ++ thepart ) {
      //  if ( (**thepart).charge()!=0 ){
      //isChargedJet=true;
      //    chargedFraction += (**thepart).pt();
      //  }
      //}
      //if ( chargedFraction == 0 ) cout << " is chargeid: " << isChargedJet << "   " << chargedFraction/genjet[k].pt()<< endl;
      //GJetAk04ChargedFraction_->push_back(chargedFraction/genjet[k].pt());
      GJetAk04ConstCnt_->push_back(genjet[k].numberOfDaughters());
      /*if(genjet[k].numberOfDaughters()>0){
	for(unsigned int idx =0; idx<genjet[k].numberOfDaughters();idx++){
	//cout<<genjet[k].numberOfDaughters()<<" GEN AHMEEEEET "<<idx<<"  "<<genjet[k].daughter(idx)->pdgId()<<"  "<<endl;
	GJetAk04ConstId->push_back(genjet[k].daughter(idx)->pdgId());
	GJetAk04ConstPt->push_back(genjet[k].daughter(idx)->pt());
	GJetAk04ConstEta->push_back(genjet[k].daughter(idx)->eta());
	GJetAk04ConstPhi->push_back(genjet[k].daughter(idx)->phi());
	GJetAk04ConstE->push_back(genjet[k].daughter(idx)->energy();)
	}
	}*/
    }
  }
}

void Tupel::processPdfInfo(const edm::Event& iEvent){
  edm::Handle<GenEventInfoProduct> genEventInfoProd;
  if (iEvent.getByLabel("generator", genEventInfoProd)) {
    if (genEventInfoProd->hasBinningValues()){
      *GBinningValue_ = genEventInfoProd->binningValues()[0];
    }
    *EvtWeights_ = genEventInfoProd->weights();
  }
  /// now get the PDF information
  edm::Handle<GenEventInfoProduct> pdfInfoHandle;
  if (iEvent.getByLabel("generator", pdfInfoHandle)) {
    if (pdfInfoHandle->pdf()) {
      GPdfId1_->push_back(pdfInfoHandle->pdf()->id.first);
      GPdfId2_->push_back(pdfInfoHandle->pdf()->id.second);
      GPdfx1_->push_back(pdfInfoHandle->pdf()->x.first);
      GPdfx2_->push_back(pdfInfoHandle->pdf()->x.second);
      //pdfInfo_->push_back(pdfInfoHandle->pdf()->xPDF.first);
      //dfInfo_->push_back(pdfInfoHandle->pdf()->xPDF.second);
      GPdfScale_->push_back(pdfInfoHandle->pdf()->scalePDF);
    }
  }
}


void Tupel::processTrigger(const edm::Event& iEvent){
  bool trigNameFilled = trigNames_.size();
  int ntrigs;
  edm::Handle< edm::TriggerResults > HLTResHandle;
  edm::InputTag HLTTag = edm::InputTag( "TriggerResults", "", "HLT");
  std::vector<int> trigIndexList;
  if(triggerStat_) trigIndexList.reserve(30);
  iEvent.getByLabel(HLTTag, HLTResHandle);
  std::ofstream f;
  /*if(analyzedEventCnt_==1){
    f.open("trigger_list.txt");
    f << "List of triggers extracted from event " << iEvent.id().event()
      << " of run " << iEvent.id().run() << "\n\n";
  }*/
  if ( HLTResHandle.isValid() && !HLTResHandle.failedToGet() ) {
    edm::RefProd<edm::TriggerNames> trigNames( &(iEvent.triggerNames( *HLTResHandle )) );
    ntrigs = (int)trigNames->size();
    if(triggerStat_) allocateTrigMap(ntrigs);
    for (int i = 0; i < ntrigs; i++) {
      //if(analyzedEventCnt_==1) f << trigNames->triggerName(i) << "\n";
      if(triggerStat_){
	if(!trigNameFilled) trigNames_[i] = trigNames->triggerName(i);
	else if(trigNames_[i] != trigNames->triggerName(i)) trigStatValid_ = false;
      }
      //insert trigger name in the acceptance map if not yet in:
      if (HLTResHandle->accept(i)){
	if(triggerStat_) trigIndexList.push_back(i);
	if(trigNames->triggerName(i).find("HLT_HIDoublePhoton15_Eta1p5_Mass50_1000") != std::string::npos) *TrigHlt_ |= kHLT_HIDoublePhoton15_Eta1p5_Mass50_1000_; //FIXME
	if(trigNames->triggerName(i).find("HLT_HIL2Mu15") != std::string::npos) *TrigHlt_ |= kHLT_HIL2Mu15_;
	std::string thisTrigger = trigNames->triggerName(i);
	fillTrig(std::string(trigNames->triggerName(i)));
      }
    }

    if(triggerStat_){
      if(trigIndexList.size()==0) trigAccept_[0][0] += 1;
      for(std::vector<int>::iterator it1 = trigIndexList.begin();
	  it1 != trigIndexList.end();
	  ++it1){
	trigAccept_[1 + *it1][0] += 1;
	if(trigIndexList.size()==1) trigAccept_[1+*it1][1] += 1;
	for(std::vector<int>::iterator it2 = trigIndexList.begin();
	    it2 != trigIndexList.end();
	    ++it2){
	  trigAccept_[1 + *it1][2 + *it2] += 1;
	}
      }
    }
  }
}

void Tupel::fillTrig(const std::string& trigname){
  for(std::vector<TrigHltMapRcd>::iterator itTrigHltMap = trigHltMapList_.begin();
      itTrigHltMap != trigHltMapList_.end(); ++itTrigHltMap){
    const std::map<std::string, ULong64_t>& trigHltMap = *(itTrigHltMap->pMap);
    ULong64_t* pTrig = itTrigHltMap->pTrig;
    for(std::map<std::string, ULong64_t>::const_iterator it = trigHltMap.begin();
	it != trigHltMap.end(); ++it){
      if(compTrigger(it->first.c_str(), trigname.c_str()))  *pTrig |= it->second;
      //      std::cout << it->first.c_str() << ", " <<  trigname.c_str() << " -> "
      //		<< (compTrigger(it->first.c_str(), trigname.c_str()) ? "identical" : "different") << "\n";
    }
  }
}

void Tupel::processMuons(){
  double MuFill = 0;

  for (unsigned int j = 0; j < muons->size(); ++j){
    const edm::View<pat::Muon> & mu = *muons;
    if(mu[j].isGlobalMuon()){
      //const pat::TriggerObjectRef trigRef( matchHelper.triggerMatchObject( muons,j,muonMatch_, iEvent, *triggerEvent ) );
      //if ( trigRef.isAvailable() && trigRef.isNonnull() ) {
      //  Mu17_Mu8_Matched=1;
      //	}
      //	const pat::TriggerObjectRef trigRef2( matchHelper.triggerMatchObject( muons,j,muonMatch2_, iEvent, *triggerEvent ) );
      //	if ( trigRef2.isAvailable() && trigRef2.isNonnull() ) {
      //	  Mu17_TkMu8_Matched=1;
      //	}
      //TODO: filled MuHltMatch
      //patMuonIdMedium_->push_back(mu[j].isMediumMuon()); Requires CMSSW >= 4_7_2
      if(vtxx){
	unsigned bit = 0;
	unsigned muonTightIds = 0;
	for (edm::View<reco::Vertex>::const_iterator vtx = pvHandle->begin(); vtx != pvHandle->end(); ++vtx){
	  if(vtx->isValid() && !vtx->isFake() && mu[j].isTightMuon(*vtx)){
	    muonTightIds |= (1 <<bit);
	  }
	  ++bit;
	  if(bit > 31) break;
	}
	MuIdTight_->push_back(muonTightIds);
      }


      //MuFill=1;
      MuPt_->push_back(mu[j].pt());
      MuEta_->push_back(mu[j].eta());
      MuPhi_->push_back(mu[j].phi());
      MuE_->push_back(mu[j].energy());
      MuCh_->push_back(mu[j].charge());
      MuVtxZ_->push_back(mu[j].vz());
      MuDxy_->push_back(mu[j].dB());

      double trkLayers = -99999;
      double pixelHits = -99999;
      double muonHits  = -99999;
      double nMatches  = -99999;
      double normChi2  = +99999;
      double dZ = -99999;
      bool isTrackMuon =mu[j].isTrackerMuon();
      bool isGlobalMuon =mu[j].isGlobalMuon();
      if(isTrackMuon && isGlobalMuon){
	trkLayers     = mu[j].innerTrack()->hitPattern().trackerLayersWithMeasurement();
	pixelHits     = mu[j].innerTrack()->hitPattern().numberOfValidPixelHits();
	muonHits      = mu[j].globalTrack()->hitPattern().numberOfValidMuonHits();
	nMatches      = mu[j].numberOfMatchedStations();
	normChi2      = mu[j].globalTrack()->normalizedChi2();
	if( !pvHandle->empty() && !pvHandle->front().isFake() ) {
	  const reco::Vertex &vtx = pvHandle->front();
	  dZ= mu[j].muonBestTrack()->dz(vtx.position());
	}
      }
      MuTkNormChi2_->push_back(normChi2);
      MuTkHitCnt_->push_back(muonHits);
      MuMatchedStationCnt_->push_back(nMatches);
      MuDz_->push_back(dZ);
      MuPixelHitCnt_->push_back(pixelHits);
      MuTkLayerCnt_->push_back(trkLayers);

      bool customMuId =	( mu[j].isGlobalMuon()
			  && mu[j].isPFMuon()
			  && normChi2<10
			  && muonHits>0 && nMatches>1
			  && mu[j].dB()<0.2 && dZ<0.5
			  && pixelHits>0 && trkLayers>5 );
      unsigned muId = 0;
      if(mu[j].isLooseMuon()) muId |= kMuIdLoose_;
      //if(mu[j].isMediumMuon()) muId |= kMuIdMedium_;
      if(customMuId) muId |= kMuIdCustom_;
      MuId_->push_back(muId);

      //      unsigned muHltMath = 0;
      //float muEta = mu[j].eta(); // essentially track direction at Vtx (recommended prescription)
      //float Aecal=0.041; // initiallize with EE value
      //float Ahcal=0.032; // initiallize with HE value
      //if (fabs(muEta)<1.48) {
	//Aecal = 0.074;   // substitute EB value
	//Ahcal = 0.023;   // substitute EE value
      //}
      //double theRho = *rho;
      //float muonIsoRho = mu[j].isolationR03().sumPt + std::max(0.,(mu[j].isolationR03().emEt -Aecal*(theRho))) + std::max(0.,(mu[j].isolationR03().hadEt-Ahcal*(theRho)));
      //double dbeta = muonIsoRho/mu[j].pt();
      //MuIsoRho_->push_back(dbeta);

      // pf Isolation variables
      double chargedHadronIso = mu[j].pfIsolationR04().sumChargedHadronPt;
      double chargedHadronIsoPU = mu[j].pfIsolationR04().sumPUPt;
      double neutralHadronIso  = mu[j].pfIsolationR04().sumNeutralHadronEt;
      double photonIso  = mu[j].pfIsolationR04().sumPhotonEt;
      double a=0.5;
      // OPTION 1: DeltaBeta corrections for iosolation
      float RelativeIsolationDBetaCorr = (chargedHadronIso + std::max(photonIso+neutralHadronIso - 0.5*chargedHadronIsoPU,0.))/std::max(a, mu[j].pt());
      MuPfIso_->push_back(RelativeIsolationDBetaCorr);
      int muType = 0;
      if(mu[j].isGlobalMuon()) muType |= kGlobMu_;
      if(mu[j].isTrackerMuon()) muType |= kTkMu_;
      if(mu[j].isPFMuon()) muType |= kPfMu_;
      MuType_->push_back(muType);
      MuIsoTkIsoAbs_->push_back(mu[j].isolationR03().sumPt);
      MuIsoTkIsoRel_->push_back(mu[j].isolationR03().sumPt/mu[j].pt());
      MuIsoCalAbs_->push_back(mu[j].isolationR03().emEt + mu[j].isolationR03().hadEt);
      MuIsoCombRel_->push_back((mu[j].isolationR03().sumPt+mu[j].isolationR03().hadEt)/mu[j].pt());
      MuPfIsoChHad_->push_back(mu[j].pfIsolationR03().sumChargedHadronPt);
      MuPfIsoNeutralHad_->push_back(mu[j].pfIsolationR03().sumNeutralHadronEt);
      MuPfIsoRawRel_->push_back((mu[j].pfIsolationR03().sumChargedHadronPt+mu[j].pfIsolationR03().sumNeutralHadronEt)/mu[j].pt());
      MuFill++;
    }
  }
}

void Tupel::processElectrons(){
  int ElecFill=0;
  auto_ptr<vector<pat::Electron> > electronColl( new vector<pat::Electron> (*electrons) );
  for (unsigned int j=0; j < electronColl->size();++j){
    pat::Electron & el = (*electronColl)[j];

    double dEtaIn_;
    double dPhiIn_;
    double hOverE_;
    double sigmaIetaIeta_;
    double full5x5_sigmaIetaIeta_;
    //double relIsoWithDBeta_;
    double ooEmooP_;
    double d0_ = -99.;
    double dz_ = -99.;
    int   expectedMissingInnerHits_;
    int   passConversionVeto_;

    std::vector<std::pair<std::string,float> > idlist = el.electronIDs();
    /*if(!elecIdsListed_) {
      std::ofstream f("electron_id_list.txt");
      f << "Autogenerated file\n\n"
	"Supported electron ids:\n";
      for (unsigned k  = 0 ; k < idlist.size(); ++k){
	f << idlist[k].first << ": " << idlist[k].second  << "\n";
      }
      f.close();
      elecIdsListed_ = true;
    }*/
    unsigned elecid = 0;
    
    for(unsigned i = 0; i < idlist.size(); ++i){
      //const int idAndConvRejectMask = 0x3;
      //if(int(idlist[i].second) & idAndConvRejectMask){
      if(int(idlist[i].second)){
	std::map<std::string, unsigned>::const_iterator it = ElIdMap_.find(idlist[i].first);
	if(it != ElIdMap_.end()) elecid  |= it->second;	
      }
    }
      
    ElId_->push_back(elecid);

    dEtaIn_ = el.deltaEtaSuperClusterTrackAtVtx();
    dPhiIn_ = el.deltaPhiSuperClusterTrackAtVtx();
    hOverE_ = el.hcalOverEcal();
    sigmaIetaIeta_ = el.sigmaIetaIeta();
    full5x5_sigmaIetaIeta_ =  el.full5x5_sigmaIetaIeta();
    if( el.ecalEnergy() == 0 ){
      // printf("Electron energy is zero!\n");
      ooEmooP_ = 1e30;
    }else if( !std::isfinite(el.ecalEnergy())){
      // printf("Electron energy is not finite!\n");
      ooEmooP_ = 1e30;
    }else{
      ooEmooP_ = fabs(1.0/el.ecalEnergy() - el.eSuperClusterOverP()/el.ecalEnergy() );
    }

    if(vtx_h->size() > 0){
      d0_ = (-1) * el.gsfTrack()->dxy((*vtx_h)[0].position() );
      dz_ = el.gsfTrack()->dz( (*vtx_h)[0].position() );
    }



    //     expectedMissingInnerHits_ = el.gsfTrack()->trackerExpectedHitsInner().numberOfLostHits();//MISSING!!
    passConversionVeto_ = false;
    if( beamSpotHandle.isValid() && conversions_h.isValid()) {
      passConversionVeto_ = !ConversionTools::hasMatchedConversion(el,conversions_h,
								   beamSpotHandle->position());
    }else{
      printf("\n\nERROR!!! conversions not found!!!\n");
    }


    //cout<<dEtaIn_<<"  "<<dPhiIn_<<"  "<<hOverE_<<"  "<<sigmaIetaIeta_<<"  "<<full5x5_sigmaIetaIeta_<<"  "<<ooEmooP_<<"  "<< d0_<<"  "<< dz_<<"  "<<expectedMissingInnerHits_<<"  "<<passConversionVeto_<<endl;

    ElDEtaTkScAtVtx_->push_back(dEtaIn_);
    ElDPhiTkScAtVtx_->push_back(dPhiIn_);
    ElHoE_->push_back(hOverE_);
    ElSigmaIetaIeta_->push_back(sigmaIetaIeta_);
    ElSigmaIetaIetaFull5x5_->push_back(full5x5_sigmaIetaIeta_);
    ElEinvMinusPinv_->push_back(ooEmooP_);
    ElD0_->push_back(d0_);
    ElDz_->push_back(dz_);
    ElExpectedMissingInnerHitCnt_->push_back(expectedMissingInnerHits_);
    ElPassConvVeto_->push_back(passConversionVeto_);

    int hltMatch = 0;

    ElHltMatch_->push_back(hltMatch);//no matching yet...BB
    const string mvaTrigV0 = "mvaTrigV0";
    const string mvaNonTrigV0 = "mvaNonTrigV0";

    ElPt_->push_back(el.pt());
    ElEta_->push_back(el.eta());

    ElEtaSc_->push_back(el.superCluster()->eta());
    ElPhi_->push_back(el.phi());
    ElE_->push_back(el.energy());
    ElCh_->push_back(el.charge());

    double aeff = ElectronEffectiveArea::GetElectronEffectiveArea(ElectronEffectiveArea::kEleGammaAndNeutralHadronIso03, el.superCluster()->eta(), ElectronEffectiveArea::kEleEAData2012);
    ElAEff_->push_back(aeff);

    const double chIso03_ = el.chargedHadronIso();
    const double nhIso03_ = el.neutralHadronIso();
    const double phIso03_ = el.photonIso();
    const double puChIso03_= el.puChargedHadronIso();
    ElPfIsoChHad_->push_back(chIso03_);
    ElPfIsoNeutralHad_->push_back(nhIso03_);
    ElPfIsoIso_->push_back(phIso03_);
    ElPfIsoPuChHad_->push_back(puChIso03_);
    ElPfIsoRaw_->push_back(( chIso03_ + nhIso03_ + phIso03_ ) / el.pt());
    ElPfIsoDbeta_->push_back(( chIso03_ + max(0.0, nhIso03_ + phIso03_ - 0.5*puChIso03_) )/ el.pt());

    //*EvtFastJetRho_ =  rhoIso;
    //double rhoPrime = std::max(0., rhoIso);

    //ElPfIsoRho_->push_back(( chIso03_ + max(0.0, nhIso03_ + phIso03_ - rhoPrime*(aeff)) )/ el.pt());

    bool myTrigPresel = false;
    if(fabs(el.superCluster()->eta()) < 1.479){
      if(el.sigmaIetaIeta() < 0.014 &&
	 el.hadronicOverEm() < 0.15 &&
	 el.dr03TkSumPt()/el.pt() < 0.2 &&
	 el.dr03EcalRecHitSumEt()/el.pt() < 0.2 &&
	 el.dr03HcalTowerSumEt()/el.pt() < 0.2 /*&&
						 el.gsfTrack()->trackerExpectedHitsInner().numberOfLostHits() == 0*/)
	myTrigPresel = true;
    }
    else {
      if(el.sigmaIetaIeta() < 0.035 &&
	 el.hadronicOverEm() < 0.10 &&
	 el.dr03TkSumPt()/el.pt() < 0.2 &&
	 el.dr03EcalRecHitSumEt()/el.pt() < 0.2 &&
	 el.dr03HcalTowerSumEt()/el.pt() < 0.2 /*&&
						 el.gsfTrack()->trackerExpectedHitsInner().numberOfLostHits() == 0*/)
	myTrigPresel = true;
    }
    ElMvaPresel_->push_back(myTrigPresel);
    ElecFill++;
  }
}

void Tupel::processJets(){
  //double PFjetFill=0;

  for ( unsigned int i=0; i<jets->size(); ++i ) {
    const pat::Jet & jet = jets->at(i);

    double hfhe = 0;
    double hfeme = 0;
    double chf = 0;
    double nhf = 0;
    double cemf = 0;
    double nemf = 0;
    double cmult = 0;
    double nconst = 0;

    //JetAk04PuMva_->push_back(jet.userFloat("pileupJetId:fullDiscriminant"));

    nconst = jet.numberOfDaughters();

    if(jet.isPFJet() || jet.isJPTJet()) {
      hfhe = jet.HFHadronEnergy();
      hfeme = jet.HFEMEnergy();
      chf = jet.chargedHadronEnergyFraction();
      nhf = (jet.neutralHadronEnergy()+hfhe)/jet.correctedJet(0).energy();
      cemf = jet.chargedEmEnergyFraction();
      nemf = jet.neutralEmEnergyFraction();
      cmult = jet.chargedMultiplicity();

    }

    // cout<<"jet.bDiscriminator(combinedSecondaryVertexBJetTags)=  "<<jet.bDiscriminator("combinedSecondaryVertexBJetTags")<<endl;
    //  cout<<"jet.bDiscriminator(combinedSecondaryVertexV1BJetTags)=  "<<jet.bDiscriminator("combinedSecondaryVertexV1BJetTags")<<endl;
    //  cout<<"jet.bDiscriminator(combinedSecondaryVertexSoftPFLeptonV1BJetTags)=  "<<jet.bDiscriminator("combinedSecondaryVertexSoftPFLeptonV1BJetTags")<<endl;
    JetAk04BTagCsv_->push_back(jet.bDiscriminator("combinedSecondaryVertexBJetTags"));
    JetAk04BTagCsvV1_->push_back(jet.bDiscriminator("combinedSecondaryVertexV1BJetTags"));
    JetAk04BTagCsvSLV1_->push_back(jet.bDiscriminator("combinedSecondaryVertexSoftPFLeptonV1BJetTags"));
    JetAk04BDiscCisvV2_->push_back(jet.bDiscriminator("pfCombinedInclusiveSecondaryVertexV2BJetTags"));
    JetAk04BDiscJp_->push_back(jet.bDiscriminator("pfJetProbabilityBJetTags"));
    JetAk04BDiscBjp_->push_back(jet.bDiscriminator("pfJetBProbabilityBJetTags"));
    JetAk04BDiscTche_->push_back(jet.bDiscriminator("pfTrackCountingHighEffBJetTags"));
    JetAk04BDiscTchp_->push_back(jet.bDiscriminator("pfTrackCountingHighPurBJetTags"));
    JetAk04BDiscSsvhe_->push_back(jet.bDiscriminator("pfSimpleSecondaryVertexHighEffBJetTags"));
    JetAk04BDiscSsvhp_->push_back(jet.bDiscriminator("pfSimpleSecondaryVertexHighPurBJetTags"));
    JetAk04PartFlav_->push_back(jet.partonFlavour());

    JetAk04E_->push_back(jet.energy());
    JetAk04Pt_->push_back(jet.pt());
    JetAk04Eta_->push_back(jet.eta());
    JetAk04Phi_->push_back(jet.phi());
    JetAk04RawPt_->push_back(jet.correctedJet(0).pt());
    JetAk04RawE_->push_back(jet.correctedJet(0).energy());
    JetAk04HfHadE_->push_back(hfhe);
    JetAk04HfEmE_->push_back(hfeme);
    JetAk04ChHadFrac_->push_back(chf);
    JetAk04NeutralHadAndHfFrac_->push_back(nhf);
    JetAk04ChEmFrac_->push_back(cemf);
    JetAk04NeutralEmFrac_->push_back(nemf);
    JetAk04ChMult_->push_back(cmult);
    JetAk04ConstCnt_->push_back(nconst);

    for(unsigned int idx =0; idx<jet.numberOfDaughters();idx++){
      //cout<<jet.numberOfDaughters()<<" RECO AHMEEEEET "<<idx<<"  "<<jet.daughter(idx)->pdgId()<<"  "<<endl;
      JetAk04ConstId_->push_back(jet.daughter(idx)->pdgId());
      JetAk04ConstPt_->push_back(jet.daughter(idx)->pt());
      JetAk04ConstEta_->push_back(jet.daughter(idx)->eta());
      JetAk04ConstPhi_->push_back(jet.daughter(idx)->phi());
      JetAk04ConstE_->push_back(jet.daughter(idx)->energy());


    }
    //TODO: insert correct value
    double unc = 1.;
    JetAk04JecUncUp_->push_back(unc);
    JetAk04JecUncDwn_->push_back(unc);
    double tempJetID=0;
    if( abs(jet.eta())<2.4){
      if(chf>0 && nhf<0.99 && cmult>0.0 && cemf<0.99 && nemf<0.99 && nconst>1) tempJetID=1;
      if((chf>0)&&(nhf<0.95)&&(cmult>0.0)&&(cemf<0.99)&&(nemf<0.95)&&(nconst>1)) tempJetID=2;
      if((chf>0)&&(nhf<0.9)&&(cmult>0.0)&&(cemf<0.99)&&(nemf<0.9)&&(nconst>1)) tempJetID=3;
    }
    if( abs(jet.eta())>=2.4){
      if ((nhf<0.99)&&(nemf<0.99)&&(nconst>1))tempJetID=1;
      if ((nhf<0.95)&&(nemf<0.95)&&(nconst>1))tempJetID=2;
      if ((nhf<0.9)&&(nemf<0.9)&&(nconst>1))tempJetID=3;
    }
    JetAk04Id_->push_back(tempJetID);//ala
    //PFjetFill++;

    if(!*EvtIsRealData_){
      int genJetIdx = -1; //code for not matched jets
      if (jet.genJet()){
	//	genJetIdx = jet.genJetFwdRef().key();
	for(genJetIdx = 0; (unsigned) genJetIdx < genjetColl_->size();++genJetIdx){
	  if(&(*genjetColl_)[genJetIdx] == jet.genJet()) break;
	}
	//genJetIdx = jet.genJetFwdRef().backRef().key();
	//assert(jet.genJet() == &(*genjetColl_)[genJetIdx]);
	//	if((unsigned)genJetIdx == genjetColl_->size()){
	//	  genJetIdx = -2;
	//	  std::cerr << "Matched gen jet not found!\n";
	//	} else{
	//	  double dphi = fabs(jet.phi()-(*genjetColl_)[genJetIdx].phi());
	//	  if(dphi > pi ) dphi = 2*pi - dphi;
	//	  std::cout << "Jet matching check: R = "
	//		    << sqrt(std::pow(jet.eta()-(*genjetColl_)[genJetIdx].eta(),2)
	//			    + std::pow(dphi,2))
	//		    << "\tPt ratio = "
	//		    << jet.pt() / (*genjetColl_)[genJetIdx].pt()
	//		    << "\t" << jet.pt() / jet.genJet()->pt()
	//		    << "\t key: " << genJetIdx
	//		    << "\t coll size: " <<  genjetColl_->size()
	//		    << "\t" << hex << jet.genJet()
	//		    << "\t" << &((*genjetColl_)[genJetIdx]) << dec
	//		    << "\tIndices: " << genJetIdx << ", " << jet.genJetFwdRef().backRef().key() << ", " << jet.genJetFwdRef().ref().key() << ", " << jet.genJetFwdRef().key()
	//		    << "\n";
	//}
      }
      JetAk04GenJet_->push_back(genJetIdx);
    }
  }
}

void Tupel::processPhotons(){
  for (unsigned j = 0; j < photons->size(); ++j){
    const pat::Photon& photon = (*photons)[j];
    //photon momentum
    PhotPt_->push_back(photon.pt());
    PhotEta_->push_back(photon.eta());
    PhotPhi_->push_back(photon.phi());
    PhotScRawE_->push_back(photon.superCluster()->rawEnergy());
    PhotScEta_->push_back(photon.superCluster()->eta());
    PhotScPhi_->push_back(photon.superCluster()->phi());

    //photon isolation
    PhotIsoEcal_->push_back(photon.ecalIso());
    PhotIsoHcal_->push_back(photon.hcalIso());
    PhotIsoTk_->push_back(photon.trackIso());
    PhotPfIsoChHad_->push_back(photon.chargedHadronIso());
    PhotPfIsoNeutralHad_->push_back(photon.neutralHadronIso());
    PhotPfIsoPhot_->push_back(photon.photonIso());
    PhotPfIsoPuChHad_->push_back(photon.puChargedHadronIso());
    PhotPfIsoEcalClus_->push_back(photon.ecalPFClusterIso());
    PhotPfIsoHcalClus_->push_back(photon.hcalPFClusterIso());

    //photon cluster shape
    PhotE3x3_->push_back(photon.e3x3());
    PhotE1x5_->push_back(photon.e1x5());
    //PhotE1x3_->push_back(...);
    //PhotE2x2_->push_back(...);
    PhotE2x5_->push_back(photon.e2x5());
    PhotE5x5_->push_back(photon.e5x5());
    PhotSigmaIetaIeta_->push_back(photon.sigmaIetaIeta());
    //PhotSigmaIetaIphi_->push_back(...);
    //PhotSigmaIphiIphi_->push_back(...);
    PhotEtaWidth_->push_back(photon.sigmaEtaEta());
    //PhotPhiWidth_->push_back(...);

    //preshower
    //PhotEsE_->push_back(...);
    //PhotEsSigmaIxIx_->push_back(...);
    //PhotEsSigmaIyIy_->push_back(...);
    //PhotEsSigmaIrIr_->push_back(sqrt(std::pow(PhotEsSigmaIxIx_, 2)
    //                                  + std::pow(PhotEsSigmaIyIy_, 2));

    //photon time
    //PhotTime_->push_back(...);

    //photon ids:
    std::vector<std::pair<std::string,Bool_t> > idlist = photon.photonIDs();
    /*if(!photonIdsListed_) {
      std::ofstream f("electron_id_list.txt");
      f << "Autogenerated file\n\n"
	"Supported photon ids:\n";
      for (unsigned k  = 0 ; k < idlist.size(); ++k){
	f << idlist[k].first << ": " << (idlist[k].second ? "yes" : "no") << "\n";
      }
      f.close();
      photonIdsListed_ = true;
    }*/

    int photonId = 0;
    if(photon.isPhotonIDAvailable(std::string("PhotonCutBasedIDLoose"))) {
      if(photon.photonID(std::string("PhotonCutBasedIDLoose"))) photonId |= 1;
    }
    if(photon.isPhotonIDAvailable(std::string("PhotonCutBasedIDTight"))) {
      if(photon.photonID(std::string("PhotonCutBasedIDTight"))) photonId |= 4;
    }
    PhotId_->push_back(photonId);
    PhotHoE_->push_back(photon.hadronicOverEm());
    PhotHasPixelSeed_->push_back(photon.hasPixelSeed());
  }
}

void Tupel::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup){
  using namespace edm; //ADD
  ++analyzedEventCnt_;

  readEvent(iEvent);

  // PAT trigger EvtNum
  //edm::Handle<pat::TriggerEvent>  triggerEvent;
  //iEvent.getByLabel( triggerEvent_, triggerEvent );


  // edm::Handle<edm::ValueMap<float> > full5x5sieie;
  // iEvent.getByToken(full5x5SigmaIEtaIEtaMapToken_,full5x5sieie);

  processMET(iEvent);

  processVtx();

  if (!*EvtIsRealData_){
    processPu(iEvent);
    if (genParticles) processGenParticles(iEvent);
    processGenJets(iEvent);
    processPdfInfo(iEvent);
  }

  processTrigger(iEvent);

  if(muon) processMuons();

  //electrons B.B.
  if(electron) processElectrons();

  //jets
  if(jettt) processJets();

  //photons. Ph. G.
  if(photons) processPhotons();

  //Stores the EvtNum in the output tree
  treeHelper_->fill();

}

void
Tupel::writeHeader(){
  TTree* t = new TTree("Header", "Header");
  TString checksum_(checksum);
  t->Branch("Tupel_cc_githash", &checksum_);

  TString cmssw_release(edm::getReleaseVersion().c_str());
  t->Branch("CMSSW_RELEASE", &cmssw_release);

  char hostname[256];
  gethostname(hostname, sizeof(hostname));
  hostname[sizeof(hostname)-1] = 0;
  TString hostname_(hostname);
  t->Branch("Hostname", &hostname_);

  edm::TimeOfDay ts;
  stringstream s;
  s << setprecision(0) << ts;
  TString ts_(s.str().c_str());
  t->Branch("CreationTime", &ts_);

  t->Fill();
}

void
Tupel::beginJob()
{
  //  jecUnc = new JetCorrectionUncertainty("Fall12_V7_DATA_Uncertainty_AK5PFchs.txt");
  // register to the TFileService
  edm::Service<TFileService> fs;
  TFileDirectory TestDir = fs->mkdir("test");

  writeHeader();

  myTree = new TTree("EventTree"," EventTree");
  treeHelper_ = std::auto_ptr<TreeHelper>
    (new TreeHelper(myTree, new TTree("Description", "Description"),
		    new TTree("BitFields", "BitFields")));

  //Event
  ADD_BRANCH_D(EvtIsRealData, "True if real data, false if MC");
  ADD_BRANCH_D(EvtNum, "Event number");
  ADD_BRANCH_D(EvtRunNum, "Run number");
  ADD_BRANCH_D(EvtLumiNum, "Luminosity block number");
  ADD_BRANCH_D(EvtBxNum, "Bunch crossing number");
  ADD_BRANCH_D(EvtVtxCnt, "Number of reconstructed primary vertices");
  ADD_BRANCH_D(EvtPuCnt, "Number of measured pile-up events");
  ADD_BRANCH_D(EvtPuCntTruth, "True number of pile-up events");
  ADD_BRANCH(EvtWeights); //description filled in endRun()
  //ADD_BRANCH_D(EvtFastJetRho, "Fastjet pile-up variable \\rho");

  //Trigger
  ADD_BRANCH_D(TrigHlt, "HLT triggger bits. See BitField.TrigHlt for bit description.");
  ADD_BRANCH_D(TrigHltMu, "HLT Muon triggger bits. See BitField.TrigHltMu for bit description.");
  ADD_BRANCH_D(TrigHltDiMu, "HLT Dimuon triggger bits. See BitField.TrigHltDiMu for bit description.");
  ADD_BRANCH_D(TrigHltEl, "HLT Electron triggger bits. See BitField.TrigHltEl for bit description.");
  ADD_BRANCH_D(TrigHltDiEl, "HLT Dielecton triggger bits. See BitField.TrigHltDiEl for bit description.");
  ADD_BRANCH_D(TrigHltElMu, "HLT Muon + Electron triggger bits. See BitField.TrigHltElMu for bit description.");

  //Missing Energy
  treeHelper_->addDescription("MET", "PF MET");
  ADD_BRANCH(METPt);
  ADD_BRANCH(METPx);
  ADD_BRANCH(METPy);
  ADD_BRANCH(METPz);
  ADD_BRANCH(METE);
  ADD_BRANCH(METsigx2);
  ADD_BRANCH(METsigxy);
  ADD_BRANCH(METsigy2);
  ADD_BRANCH(METsig);

  //Generator level leptons.
  treeHelper_->addDescription("GLepDr01", "Generator-level leptons. Muons and electrons and their antiparticles are dressed using a cone of radius R = 0.1");
  ADD_BRANCH(GLepDr01Pt);
  ADD_BRANCH(GLepDr01Eta);
  ADD_BRANCH(GLepDr01Phi);
  ADD_BRANCH(GLepDr01E);
  ADD_BRANCH(GLepDr01Id);
  ADD_BRANCH(GLepDr01St);
  ADD_BRANCH(GLepDr01MomId);
  treeHelper_->addDescription("GLepBare", "Generator-level leptons, status 1 without dressing.");
  ADD_BRANCH(GLepBarePt);
  ADD_BRANCH(GLepBareEta);
  ADD_BRANCH(GLepBarePhi);
  ADD_BRANCH(GLepBareE);
  ADD_BRANCH(GLepBareId);
  ADD_BRANCH(GLepBareSt);
  ADD_BRANCH(GLepBareMomId);
  treeHelper_->addDescription("GLepDr01", "Status 3 generator-level leptons.");
  ADD_BRANCH(GLepSt3Pt);
  ADD_BRANCH(GLepSt3Eta);
  ADD_BRANCH(GLepSt3Phi);
  ADD_BRANCH(GLepSt3E);
  ADD_BRANCH(GLepSt3Id);
  ADD_BRANCH(GLepSt3St);
  ADD_BRANCH_D(GLepSt3Mother0Id, "Lepton mother PDG Id. Filled only for first mother. Number of mothers can be checked in GLepoSt3MotherCnt.");
  ADD_BRANCH(GLepSt3MotherCnt);
  //Generator level photons.
  treeHelper_->addDescription("GPhot", "Generator-level photons.");
  ADD_BRANCH(GPhotPt);
  ADD_BRANCH(GPhotEta);
  ADD_BRANCH(GPhotPhi);
  ADD_BRANCH(GPhotE);
  ADD_BRANCH_D(GPhotMotherId, "Photon mother PDG Id. Filled only for first mother.");
  ADD_BRANCH(GPhotSt);
  ADD_BRANCH(GPhotIsoEDR03);
  ADD_BRANCH(GPhotIsoEDR04);
  ADD_BRANCH(GPhotIsoEDR05);
  ADD_BRANCH(GPhotIsoSumPtDR03);
  ADD_BRANCH(GPhotIsoSumPtDR04);
  ADD_BRANCH(GPhotIsoSumPtDR05);

  //Photons in the vicinity of the leptons
  treeHelper_->addDescription("GLepClosePhot", "Photons aroud leptons. Selection cone size: R = 0.2");
  ADD_BRANCH(GLepClosePhotPt);
  ADD_BRANCH(GLepClosePhotEta);
  ADD_BRANCH(GLepClosePhotPhi);
  ADD_BRANCH(GLepClosePhotE);
  ADD_BRANCH(GLepClosePhotId);
  ADD_BRANCH_D(GLepClosePhotMother0Id, "Photon mother PDG Id. Filled only for first mother. Number of mothers can be checked in GLepoSt3MotherCnt.");
  ADD_BRANCH(GLepClosePhotMotherCnt);
  ADD_BRANCH(GLepClosePhotSt);

  //Gen Jets
  treeHelper_->addDescription("GJetAk04", "Generator-level reconstructed with the anti-kt algorithm with distance parameter R = 0.4");
  ADD_BRANCH(GJetAk04Pt);
  ADD_BRANCH(GJetAk04Eta);
  ADD_BRANCH(GJetAk04Phi);
  ADD_BRANCH(GJetAk04E);
  ADD_BRANCH(GJetAk04ChFrac);
  ADD_BRANCH(GJetAk04ConstCnt);
  ADD_BRANCH(GJetAk04ConstId);
  ADD_BRANCH(GJetAk04ConstPt);
  ADD_BRANCH(GJetAk04ConstEta);
  ADD_BRANCH(GJetAk04ConstPhi);
  ADD_BRANCH(GJetAk04ConstE);

  //Exta generator information
  ADD_BRANCH_D(GPdfId1, "PDF Id for beam 1");
  ADD_BRANCH_D(GPdfId2, "PDF Id for beam 2");
  ADD_BRANCH_D(GPdfx1, "PDF x for beam 1");
  ADD_BRANCH_D(GPdfx2, "PDF x for beam 1");
  ADD_BRANCH_D(GPdfScale, "PDF energy scale");
  ADD_BRANCH_D(GBinningValue, "Value of the observable used to split the MC sample generation (e.g. pt_hat for a pt_hat binned MC sample).");
  ADD_BRANCH_D(GNup, "Number of particles/partons included in the matrix element.");

  //Muons
  treeHelper_->addDescription("Mu", "PF reconstruced muons.");
  ADD_BRANCH(MuPt);
  ADD_BRANCH(MuEta);
  ADD_BRANCH(MuPhi);
  ADD_BRANCH(MuE);
  ADD_BRANCH(MuId);
  ADD_BRANCH_D(MuIdTight, "Muon tight id. Bit field, one bit per primary vertex hypothesis. Bit position corresponds to index in EvtVtx");
  ADD_BRANCH(MuCh);
  ADD_BRANCH(MuVtxZ);
  ADD_BRANCH(MuDxy);
  //ADD_BRANCH(MuIsoRho);
  ADD_BRANCH(MuPfIso);
  ADD_BRANCH(MuType);
  ADD_BRANCH(MuIsoTkIsoAbs);
  ADD_BRANCH(MuIsoTkIsoRel);
  ADD_BRANCH(MuIsoCalAbs);
  ADD_BRANCH(MuIsoCombRel);
  ADD_BRANCH(MuTkNormChi2);
  ADD_BRANCH(MuTkHitCnt);
  ADD_BRANCH(MuMatchedStationCnt);
  ADD_BRANCH(MuDz);
  ADD_BRANCH(MuPixelHitCnt);
  ADD_BRANCH(MuTkLayerCnt);
  ADD_BRANCH(MuPfIsoChHad);
  ADD_BRANCH(MuPfIsoNeutralHad);
  ADD_BRANCH(MuPfIsoRawRel);
  ADD_BRANCH(MuHltMatch);

  //Electrons
  treeHelper_->addDescription("El", "PF reconstructed electrons");
  ADD_BRANCH(ElPt);
  ADD_BRANCH(ElEta);
  ADD_BRANCH(ElEtaSc);
  ADD_BRANCH(ElPhi);
  ADD_BRANCH(ElE);
  ADD_BRANCH(ElId);
  ADD_BRANCH(ElCh);
  ADD_BRANCH(ElMvaTrig);
  ADD_BRANCH(ElMvaNonTrig);
  ADD_BRANCH(ElMvaPresel);
  ADD_BRANCH(ElDEtaTkScAtVtx);
  ADD_BRANCH(ElDPhiTkScAtVtx);
  ADD_BRANCH(ElHoE);
  ADD_BRANCH(ElSigmaIetaIeta);
  ADD_BRANCH(ElSigmaIetaIetaFull5x5);
  ADD_BRANCH(ElEinvMinusPinv);
  ADD_BRANCH(ElD0);
  ADD_BRANCH(ElDz);
  ADD_BRANCH(ElExpectedMissingInnerHitCnt);
  ADD_BRANCH(ElPassConvVeto);
  ADD_BRANCH(ElHltMatch);
  ADD_BRANCH(ElPfIsoChHad);
  ADD_BRANCH(ElPfIsoNeutralHad);
  ADD_BRANCH(ElPfIsoIso);
  ADD_BRANCH(ElPfIsoPuChHad);
  ADD_BRANCH(ElPfIsoRaw);
  ADD_BRANCH(ElPfIsoDbeta);
  //ADD_BRANCH(ElPfIsoRho);
  ADD_BRANCH_D(ElAEff, "Electron effecive area");

  ADD_BRANCH(charged);
  ADD_BRANCH(photon);
  ADD_BRANCH(neutral);
  ADD_BRANCH(charged_Tom);
  ADD_BRANCH(photon_Tom);
  ADD_BRANCH(neutral_Tom);


  //photon momenta
  treeHelper_->addDescription("Phot", "Particle flow photons");
  ADD_BRANCH(PhotPt);;
  ADD_BRANCH(PhotEta);;
  ADD_BRANCH(PhotPhi);;
  ADD_BRANCH_D(PhotScRawE, "Photon Supercluster uncorrected energy");
  ADD_BRANCH_D(PhotScEta, "Photon Supercluster eta");
  ADD_BRANCH_D(PhotScPhi, "Photon Supercluster phi");

  //photon isolations
  ADD_BRANCH(PhotIsoEcal);
  ADD_BRANCH(PhotIsoHcal);
  ADD_BRANCH(PhotIsoTk);
  ADD_BRANCH(PhotPfIsoChHad);
  ADD_BRANCH(PhotPfIsoNeutralHad);
  ADD_BRANCH(PhotPfIsoPhot);
  ADD_BRANCH(PhotPfIsoPuChHad);
  ADD_BRANCH(PhotPfIsoEcalClus);
  ADD_BRANCH(PhotPfIsoHcalClus);

  //photon cluster shapes
  ADD_BRANCH_D(PhotE3x3, "Photon energy deposited in 3x3 ECAL crystal array. Divide this quantity by PhotScRawE to obtain the R9 variable.");
  ADD_BRANCH(PhotE1x5);
  //ADD_BRANCH(PhotE1x3);
  //ADD_BRANCH(PhotE2x2);
  ADD_BRANCH(PhotE2x5);
  ADD_BRANCH(PhotE5x5);
  ADD_BRANCH(PhotSigmaIetaIeta);
  //ADD_BRANCH(PhotSigmaIetaIphi);
  //ADD_BRANCH(PhotSigmaIphiIphi);
  ADD_BRANCH(PhotEtaWidth);
  ADD_BRANCH(PhotPhiWidth);
  ADD_BRANCH(PhotHoE);

  //photon ES
  //ADD_BRANCH_D(PhotEsE, "Photon. Energy deposited in the preshower");
  //ADD_BRANCH_D(PhotEsSigmaIxIx, "Photon. Preshower cluster extend along x-axis");
  //ADD_BRANCH_D(PhotEsSigmaIyIy, "Photon. Preshower cluster extend along y-axis");
  //ADD_BRANCH_D(PhotEsSigmaIrIr, "Photon. \\sqrt(PhotEsSigmaIxIx**2+PhotEsSigmaIyIy**2)");

  //photon ID
  ADD_BRANCH_D(PhotId, "Photon Id. Field of bits described in BitFields.PhotId");
  ADD_BRANCH_D(PhotHasPixelSeed, "Pixel and tracker based variable to discreminate photons from electrons");

  //photon timing
  //ADD_BRANCH_D(PhotTime, "Photon. Timing from ECAL");

  //PF Jets
  treeHelper_->addDescription("JetAk04", "Reconstricuted jets clustered with the anti-ket algorithm with distance parameter R = 0.4");
  ADD_BRANCH(JetAk04Pt);
  ADD_BRANCH(JetAk04Eta);
  ADD_BRANCH(JetAk04Phi);
  ADD_BRANCH(JetAk04E);
  ADD_BRANCH_D(JetAk04Id, "Id to reject fake jets from electronic noice");
  ADD_BRANCH_D(JetAk04PuId, "Id to reject jets from pile-up events");
  ADD_BRANCH_D(JetAk04PuMva, "MVA based descriminant for PU jets");
  ADD_BRANCH_D(JetAk04RawPt, "Jet Pt before corrections");
  ADD_BRANCH_D(JetAk04RawE, "Jet energy before corrections");
  ADD_BRANCH_D(JetAk04HfHadE, "Jet hadronic energy deposited in HF");
  ADD_BRANCH_D(JetAk04HfEmE, "Jet electromagnetic energy deposited in HF");
  ADD_BRANCH(JetAk04ChHadFrac);
  ADD_BRANCH(JetAk04NeutralHadAndHfFrac);
  ADD_BRANCH(JetAk04ChEmFrac);
  ADD_BRANCH(JetAk04NeutralEmFrac);
  ADD_BRANCH(JetAk04ChMult);
  ADD_BRANCH(JetAk04ConstCnt);
  ADD_BRANCH(JetAk04JetBeta);
  ADD_BRANCH(JetAk04JetBetaClassic);
  ADD_BRANCH(JetAk04JetBetaStar);
  ADD_BRANCH(JetAk04JetBetaStarClassic);
  treeHelper_->addDescription("JetAk04BTag", "B tagging with different algorithms");
  ADD_BRANCH_D(JetAk04BTagCsv, "combinedSecondaryVertexBJetTags");
  ADD_BRANCH_D(JetAk04BTagCsvV1, "combinedSecondaryVertexV1BJetTags");
  ADD_BRANCH_D(JetAk04BTagCsvSLV1, "combinedSecondaryVertexSoftPFLeptonV1BJetTags");
  ADD_BRANCH_D(JetAk04BDiscCisvV2, "pfCombinedInclusiveSecondaryVertexV2BJetTags");
  ADD_BRANCH_D(JetAk04BDiscJp, "pfJetProbabilityBJetTags");
  ADD_BRANCH_D(JetAk04BDiscBjp, "pfJetBProbabilityBJetTags");
  ADD_BRANCH_D(JetAk04BDiscTche, "pfTrackCountingHighEffBJetTags");
  ADD_BRANCH_D(JetAk04BDiscTchp, "pfTrackCountingHighPurBJetTags");
  ADD_BRANCH_D(JetAk04BDiscSsvhe, "pfSimpleSecondaryVertexHighEffBJetTags");
  ADD_BRANCH_D(JetAk04BDiscSsvhp, "pfSimpleSecondaryVertexHighPurBJetTags");
  ADD_BRANCH_D(JetAk04PartFlav, "Quark-based jet.");
  ADD_BRANCH(JetAk04JecUncUp);
  ADD_BRANCH(JetAk04JecUncDwn);
  ADD_BRANCH(JetAk04ConstId);
  ADD_BRANCH(JetAk04ConstPt);
  ADD_BRANCH(JetAk04ConstEta);
  ADD_BRANCH(JetAk04ConstPhi);
  ADD_BRANCH(JetAk04ConstE);
  ADD_BRANCH(JetAk04GenJet);

  defineBitFields();

}

void Tupel::allocateTrigMap(int nTrigMax){
  std::vector<int> row(nTrigMax+2, 0);
  std::vector<std::vector<int> > newMap(nTrigMax+1, row);
  std::vector<std::string> newTrigNames(nTrigMax, "");
  for(unsigned i = 0; i < trigAccept_.size(); ++i){
    if(i < trigNames_.size()) newTrigNames[i] = trigNames_[i];
    for(unsigned j = 0; j < trigAccept_[i].size(); ++j){
      newMap[i][j] = trigAccept_[i][j];
    }
  }
  trigNames_.swap(newTrigNames);
  trigAccept_.swap(newMap);
}

void
Tupel::endJob(){
  //  delete jecUnc;
  //  myTree->Print();
  treeHelper_->fillDescriptionTree();

  if(triggerStat_) writeTriggerStat();
}

void
Tupel::writeTriggerStat(){
  std::ofstream f("trigger_stat.txt");
  if(!trigStatValid_){
    f << "Trigger statistics is not available. \n\n"
      "Events with different trigger index maps were processed. Trigger statistics"
      "is not supported for such case.\n\n";
    f << "Column S contains the number of events the trigger indicated in second column of is the only one fired.\n"
      << "The columns on the right of the column S contain for each combination of two triggers the number of events "
      "for which both triggers have been fired. The number in the column header refers to the trigger indices "
      "contained in the first column of the table.\n\n";
  } else {

    f << "Trigger statistics\n"
      << "------------------\n\n"
      << "Total number of processed events: " << analyzedEventCnt_ << "\n\n";

    f  << "#\tTrigger line\tAccept count\tAccept rate\tS";
    //sort the map per value by inverting key and value:
    std::vector<int> trigSortedIndex(trigNames_.size());
    for(unsigned i = 0; i < trigSortedIndex.size(); ++i){
      trigSortedIndex[i] = i;
      if(i<10) f << "\t" << (1+i);
    }
    f << "\n";
    sort(trigSortedIndex.begin(), trigSortedIndex.end(), TrigSorter(this));
    trigSortedIndex.resize(10);
    f << "0\tNone\t" << trigAccept_[0][0] << "\n";
    for(unsigned i = 0; i < trigSortedIndex.size(); ++i){
      unsigned ii = trigSortedIndex[i];
      f << (i+1) << "\t" << trigNames_[ii] << " (" << ii << ")"
	<< "\t" << trigAccept_[1+ii][0]
	<< "\t" << double(trigAccept_[1+ii][0]) / analyzedEventCnt_
	<< "\t" << trigAccept_[1+ii][1];
      for(unsigned j = 0; j < trigSortedIndex.size(); ++j){
	unsigned jj = trigSortedIndex[j];
	f << "\t" << trigAccept_[1+ii][2+jj];
      }
      f << "\n";
    }
  }
}

void
Tupel::endRun(edm::Run const& iRun, edm::EventSetup const&){

  std::string desc = "List of MC event weights. The first weight is the default weight to use when filling histograms.";
  edm::Handle<LHERunInfoProduct> lheRun;
  iRun.getByLabel(lheSource_, lheRun );

  if(!lheRun.failedToGet ()){
    const LHERunInfoProduct& myLHERunInfoProduct = *(lheRun.product());
    for (std::vector<LHERunInfoProduct::Header>::const_iterator iter = myLHERunInfoProduct.headers_begin();
	 iter != myLHERunInfoProduct.headers_end();
	 iter++){
      if(iter->tag() == "initrwgt" && iter->size() > 0){
	desc += "\n";
	for(std::vector<std::string>::const_iterator it = iter->begin();
	    it != iter->end(); ++it){
	  desc += *it;
	}
	break;
      }
    }
  }
  //Suppresses spurious last line with "<" produced with CMSSW_7_4_1:
  if(desc.size() > 2 && desc[desc.size()-1] == '<'){
    std::string::size_type p = desc.find_last_not_of(" \n", desc.size()-2);
    if(p != std::string::npos) desc.erase(p + 1);
  }
  treeHelper_->addDescription("EvtWeights", desc.c_str());
}

bool Tupel::compTrigger(const char* a, const char* bv) const{
  int i = 0;
  for(;a[i]!=0 && bv[i]!=0; ++i){
    if(a[i]!= bv[i]) return false;
  }
  if(a[i]) return false;
  if(bv[i]==0) return true;
  if(bv[i] != '_') return false;
  if(bv[++i]!='v') return false;
  for(;;){
    if(bv[++i]==0) return true;
    if(!isdigit(bv[i])) return false;
  }
  return true;
};


DEFINE_FWK_MODULE(Tupel);
