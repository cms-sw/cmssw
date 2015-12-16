#ifndef MNguyen_InclusiveJetAnalyzer_HiPFJetAnalyzer_
#define MNguyen_InclusiveJetAnalyzer_HiPFJetAnalyzer_

// system include files
#include <memory>
#include <string>
#include <iostream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"



#include "TFile.h"
#include "TTree.h"
#include "TH1.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"

//

/**\class HiPFJetAnalyzer

   \author Matt Nguyen
   \date   November 2010
*/




class HiPFJetAnalyzer : public edm::EDAnalyzer {
public:

  explicit HiPFJetAnalyzer(const edm::ParameterSet&);

  ~HiPFJetAnalyzer();

  virtual void analyze(const edm::Event&, const edm::EventSetup&);

  virtual void beginRun(const edm::Run & r, const edm::EventSetup & c);

  virtual void beginJob();


  void getPartons( const edm::Event & iEvent, const edm::EventSetup & iEs );

  void fillL1Bits(const edm::Event &iEvent);

  void fillHLTBits(const edm::Event &iEvent);

  template <typename TYPE>
    void getProduct(const std::string name, edm::Handle<TYPE> &prod,
		    const edm::Event &event) const;
  template <typename TYPE>
    bool getProductSafe(const std::string name, edm::Handle<TYPE> &prod,
			const edm::Event &event) const;

private:



  edm::InputTag  pfCandidatesTag_, trackTag_, vertexTag_, simTracksTag_, genParticleTag_, eventInfoTag_;
  edm::InputTag jetTag1_, jetTag2_, jetTag3_, jetTag4_;
  edm::InputTag recoJetTag1_, recoJetTag2_, recoJetTag3_, recoJetTag4_;
  edm::InputTag genJetTag1_, genJetTag2_, genJetTag3_, genJetTag4_;
  edm::InputTag L1gtReadout_;
  edm::InputTag associatorMap_;

  /// verbose ?
  bool   verbose_;

  bool useCentrality_;
  bool isMC_;
  double genParticleThresh_;

  bool hasSimInfo_;


  std::string                   hltResName_;         //HLT trigger results name
  std::vector<std::string>      hltProcNames_;       //HLT process name(s)
  std::vector<std::string>      hltTrgNames_;        //HLT trigger name(s)

  std::vector<int>              hltTrgBits_;         //HLT trigger bit(s)
  std::vector<bool>             hltTrgDeci_;         //HLT trigger descision(s)
  std::vector<std::string>      hltTrgUsedNames_;    //HLT used trigger name(s)
  std::string                   hltUsedResName_;     //used HLT trigger results name


  TTree *t;
  edm::Service<TFileService> fs2;

  static const int MAXJETS = 25000;
  static const int MAXUNMATCH = 25000;
  static const int MAXPFCANDS = 25000;
  static const int MAXTRACKS = 25000;
  static const int MAXGENPS = 25000;
  static const int MAXHLTBITS = 500000;

  struct JRA{

    int nj1;
    int nj2;
    int nj3;
    int nj4;
    int run;
    int evt;
    int lumi;
    int bin;
    float b;
    float vx, vy, vz;
    float bsx, bsy, bsz;
    float bswx, bswy, bswz;

    float vxErr, vyErr, vzErr;
    bool hasVtx;
    float hf;

    float rawpt_j1[MAXJETS];
    float jtpt_j1[MAXJETS];
    float refpt_j1[MAXJETS];
    float jteta_j1[MAXJETS];
    float refeta_j1[MAXJETS];
    float jtphi_j1[MAXJETS];
    float refphi_j1[MAXJETS];
    float jty_j1[MAXJETS];
    float refy_j1[MAXJETS];
    float refdrjt_j1[MAXJETS];
    float preL1et_j1[MAXJETS];
    float L2_j1[MAXJETS];
    float L3_j1[MAXJETS];
    float area_j1[MAXJETS];
    float refpartonpt_j1[MAXJETS];
    float refpartonflavor_j1[MAXJETS];

    float rawpt_j2[MAXJETS];
    float jtpt_j2[MAXJETS];
    float refpt_j2[MAXJETS];
    float jteta_j2[MAXJETS];
    float refeta_j2[MAXJETS];
    float jtphi_j2[MAXJETS];
    float refphi_j2[MAXJETS];
    float jty_j2[MAXJETS];
    float refy_j2[MAXJETS];
    float refdrjt_j2[MAXJETS];
    float preL1et_j2[MAXJETS];
    float L2_j2[MAXJETS];
    float L3_j2[MAXJETS];
    float area_j2[MAXJETS];
    float refpartonpt_j2[MAXJETS];
    float refpartonflavor_j2[MAXJETS];

    float rawpt_j3[MAXJETS];
    float jtpt_j3[MAXJETS];
    float refpt_j3[MAXJETS];
    float jteta_j3[MAXJETS];
    float refeta_j3[MAXJETS];
    float jtphi_j3[MAXJETS];
    float refphi_j3[MAXJETS];
    float jty_j3[MAXJETS];
    float refy_j3[MAXJETS];
    float refdrjt_j3[MAXJETS];
    float preL1et_j3[MAXJETS];
    float L2_j3[MAXJETS];
    float L3_j3[MAXJETS];
    float area_j3[MAXJETS];
    float refpartonpt_j3[MAXJETS];
    float refpartonflavor_j3[MAXJETS];

    float rawpt_j4[MAXJETS];
    float jtpt_j4[MAXJETS];
    float refpt_j4[MAXJETS];
    float jteta_j4[MAXJETS];
    float refeta_j4[MAXJETS];
    float jtphi_j4[MAXJETS];
    float refphi_j4[MAXJETS];
    float jty_j4[MAXJETS];
    float refy_j4[MAXJETS];
    float refdrjt_j4[MAXJETS];
    float preL1et_j4[MAXJETS];
    float L2_j4[MAXJETS];
    float L3_j4[MAXJETS];
    float area_j4[MAXJETS];
    float refpartonpt_j4[MAXJETS];
    float refpartonflavor_j4[MAXJETS];

    int nunmatch_j1;
    float unmatchpt_j1[MAXUNMATCH];
    float unmatcheta_j1[MAXUNMATCH];
    float unmatchy_j1[MAXUNMATCH];
    float unmatchphi_j1[MAXUNMATCH];

    int nunmatch_j2;
    float unmatchpt_j2[MAXUNMATCH];
    float unmatcheta_j2[MAXUNMATCH];
    float unmatchy_j2[MAXUNMATCH];
    float unmatchphi_j2[MAXUNMATCH];

    int nunmatch_j3;
    float unmatchpt_j3[MAXUNMATCH];
    float unmatcheta_j3[MAXUNMATCH];
    float unmatchy_j3[MAXUNMATCH];
    float unmatchphi_j3[MAXUNMATCH];

    int nunmatch_j4;
    float unmatchpt_j4[MAXUNMATCH];
    float unmatcheta_j4[MAXUNMATCH];
    float unmatchy_j4[MAXUNMATCH];
    float unmatchphi_j4[MAXUNMATCH];

    int nPFcand;
    int candId[MAXPFCANDS];
    float candpt[MAXPFCANDS];
    float candeta[MAXPFCANDS];
    float candphi[MAXPFCANDS];
    float candy[MAXPFCANDS];

    int ntrack;
    int tracknhits[MAXTRACKS];
    float trackpt[MAXTRACKS];
    float tracketa[MAXTRACKS];
    float trackphi[MAXTRACKS];
    float tracksumecal[MAXTRACKS];
    float tracksumhcal[MAXTRACKS];
    int trackfake[MAXTRACKS];
    int trackqual[MAXTRACKS];

    int trackNlayer[MAXTRACKS];
    int trackNlayer3D[MAXTRACKS];

    float trackd0Err[MAXTRACKS];
    float trackdzErr[MAXTRACKS];
    float trackd0[MAXTRACKS];
    float trackdz[MAXTRACKS];

    float trackptErr[MAXTRACKS];
    float trackchi2[MAXTRACKS];
    float trackchi2hit1D[MAXTRACKS];

    float trackd0ErrBS[MAXTRACKS];
    float trackdzErrBS[MAXTRACKS];
    float trackd0BS[MAXTRACKS];
    float trackdzBS[MAXTRACKS];

    float trackd0ErrTrk[MAXTRACKS];
    float trackdzErrTrk[MAXTRACKS];

    int ngenp;
    int genppdgId[MAXGENPS];
    float genppt[MAXGENPS];
    float genpeta[MAXGENPS];
    float genpphi[MAXGENPS];
    float genpy[MAXGENPS];

    float pthat;

    int parton1_flavor, parton2_flavor;
    float parton1_pt, parton2_pt;
    float parton1_phi, parton2_phi;
    float parton1_eta, parton2_eta;
    float parton1_y, parton2_y;

    // hlt
    int nHLTBit;
    bool hltBit[MAXHLTBITS];

    // l1
    int nL1TBit;
    bool l1TBit[MAXHLTBITS];
    int nL1ABit;
    bool l1ABit[MAXHLTBITS];
  };

  JRA jets_;
};

#endif
