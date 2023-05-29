/**
   \class Hydjet2Analyzer
   \brief HepMC events analyzer
   \version 2.2
   \authors Yetkin Yilmaz, Andrey Belyaev
*/

// system include files
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>

// user include files
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "FWCore/ServiceRegistry/interface/Service.h"

#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimDataFormats/HiGenData/interface/GenHIEvent.h"
#include "SimDataFormats/Vertex/interface/SimVertex.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"

#include "HepMC/GenEvent.h"
#include "HepMC/HeavyIon.h"

#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"

// root include file
#include "TFile.h"
#include "TNtuple.h"
#include "TH1.h"
#include "TH2.h"

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

using namespace edm;
using namespace std;
namespace {
  static const int MAXPARTICLES = 5000000;
  static const int ETABINS = 3;  // Fix also in branch string
}  // namespace
struct Hydjet2Event {
  int event;
  float b;
  float npart;
  float ncoll;
  float nhard;
  float phi0;
  float scale;
  int n[ETABINS];
  float ptav[ETABINS];
  int mult;
  float px[MAXPARTICLES];
  float py[MAXPARTICLES];
  float pz[MAXPARTICLES];
  float e[MAXPARTICLES];
  float pseudoRapidity[MAXPARTICLES];
  float pt[MAXPARTICLES];
  float eta[MAXPARTICLES];
  float phi[MAXPARTICLES];
  int pdg[MAXPARTICLES];
  int chg[MAXPARTICLES];

  float vx;
  float vy;
  float vz;
  float vr;
};
class Hydjet2Analyzer : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit Hydjet2Analyzer(const edm::ParameterSet &);
  ~Hydjet2Analyzer();

private:
  void beginJob() final;
  void analyze(const edm::Event &, const edm::EventSetup &) final;
  void endJob() final;
  // ----------member data ---------------------------
  std::ofstream out_b;
  std::string fBFileName;
  std::ofstream out_n;
  std::string fNFileName;
  std::ofstream out_m;
  std::string fMFileName;
  TTree *hydjetTree_;
  Hydjet2Event hev_;
  TNtuple *nt;
  std::string output;  // Output filename
  bool doTestEvent_;
  bool doAnalysis_;
  bool printLists_;
  bool doCF_;
  bool doVertex_;
  bool useHepMCProduct_;
  bool doHI_;
  bool doParticles_;
  double etaMax_;
  double ptMin_;
  bool doHistos_, userHistos_;

  float *ptBins;
  float *etaBins;
  float *phiBins;
  float *v2ptBins;
  float *v2etaBins;

  vector<double> uPtBins_;
  vector<double> uEtaBins_;
  vector<double> uPhiBins_;
  vector<double> uV2ptBins_;
  vector<double> uV2etaBins_;

  int uPDG_1;
  int uPDG_2;
  int uPDG_3;
  int uStatus_;
  int nintPt = 0;
  int nintEta = 0;
  int nintPhi = 0;
  int nintV2pt = 0;
  int nintV2eta = 0;

  double minPt = 0.;
  double minEta = 0.;
  double minPhi = 0.;
  double minV2pt = 0.;
  double minV2eta = 0.;

  double maxPt = 0.;
  double maxEta = 0.;
  double maxPhi = 0.;
  double maxV2pt = 0.;
  double maxV2eta = 0.;

  double upTetaCut_ = 0.;
  double downTetaCut_ = -1.;
  const double pi = 3.14159265358979;

  edm::EDGetTokenT<edm::HepMCProduct> srcT_;
  edm::EDGetTokenT<CrossingFrame<edm::HepMCProduct>> srcTmix_;

  edm::InputTag genParticleSrc_;
  edm::InputTag genHIsrc_;
  edm::InputTag simVerticesTag_;
  edm::ESGetToken<HepPDT::ParticleDataTable, edm::DefaultRecord> pdtToken_;

  //common

  TH1D *dhphi;

  TH1D *dhpdg;

  TH1D *dhet_sum;
  TH1D *dhet_barrel_sum;
  TH1D *dhe_sum;
  TH1D *dhe_barrel_sum;
  TH1D *dheta;
  TH1D *dhpt;

  TH1D *dhv2pt_cha;
  TH1D *dhv0pt_cha;
  TH1D *dhv2eta_cha;
  TH1D *dhv0eta_cha;
  TH1D *dhphi_cha;
  TH1D *dhet_cha_sum;
  TH1D *dhet_cha_barrel_sum;
  TH1D *dheta_cha;
  TH1D *dhpt_cha;

  TH1D *dhv2pt_ch;
  TH1D *dhv0pt_ch;
  TH1D *dhv2eta_ch;
  TH1D *dhv0eta_ch;
  TH1D *dhphi_ch;
  TH1D *dhet_ch_sum;
  TH1D *dhet_ch_barrel_sum;
  TH1D *dhe_ch_sum;
  TH1D *dhe_ch_barrel_sum;
  TH1D *dheta_ch;
  TH1D *dhpt_ch;

  TH1D *dhet_ph_sum;
  TH1D *dhet_ph_barrel_sum;
  TH1D *dhe_ph_sum;
  TH1D *dhe_ph_barrel_sum;
  TH1D *dheta_ph;
  TH1D *dhpt_ph;

  TH1D *dhet_n_sum;
  TH1D *dhet_n_barrel_sum;
  TH1D *dhe_n_sum;
  TH1D *dhe_n_barrel_sum;
  TH1D *dheta_n;
  TH1D *dhpt_n;

  TH1D *dhet_p_sum;
  TH1D *dhet_p_barrel_sum;
  TH1D *dheta_p;
  TH1D *dhpt_p;

  TH1D *dhet_pi_sum;
  TH1D *dhet_pi_barrel_sum;
  TH1D *dheta_pi;
  TH1D *dhpt_pi;

  TH1D *dhet_K_sum;
  TH1D *dhet_K_barrel_sum;
  TH1D *dheta_K;
  TH1D *dhpt_K;

  TH2D *dhpdg_st;

  TH1D *hNev;

  //Users
  TH1D *uhpt;
  TH1D *uhpth;
  TH1D *uhptj;

  TH1D *uhpt_db;
  TH1D *uhpth_db;
  TH1D *uhptj_db;

  TH1D *uhNpart;
  TH1D *uhNparth;
  TH1D *uhNpartj;

  TH1D *uhNpart_db;
  TH1D *uhNparth_db;
  TH1D *uhNpartj_db;

  TH1D *uhPtNpart;
  TH1D *uhPtNparth;
  TH1D *uhPtNpartj;

  TH1D *uhPtNpart_db;
  TH1D *uhPtNparth_db;
  TH1D *uhPtNpartj_db;

  TH1D *uhv2Npart;
  TH1D *uhv2Nparth;
  TH1D *uhv2Npartj;

  TH1D *uhv2Npart_db;
  TH1D *uhv2Nparth_db;
  TH1D *uhv2Npartj_db;

  TH1D *uheta;
  TH1D *uhetah;
  TH1D *uhetaj;

  TH1D *uhphi;
  TH1D *uhphih;
  TH1D *uhphij;

  TH1D *uhv2pt;
  TH1D *uhv2pth;
  TH1D *uhv2ptj;

  TH1D *uhv3pt;
  TH1D *uhv4pt;
  TH1D *uhv5pt;
  TH1D *uhv6pt;

  TH1D *uhv0pt;
  TH1D *uhv0pth;
  TH1D *uhv0ptj;

  TH1D *uhv2pt_db;
  TH1D *uhv2pth_db;
  TH1D *uhv2ptj_db;

  TH1D *uhv3pt_db;
  TH1D *uhv4pt_db;
  TH1D *uhv5pt_db;
  TH1D *uhv6pt_db;

  TH1D *uhv0pt_db;
  TH1D *uhv0pth_db;
  TH1D *uhv0ptj_db;

  TH1D *uhv2eta;
  TH1D *uhv2etah;
  TH1D *uhv2etaj;

  TH1D *uhv3eta;
  TH1D *uhv4eta;
  TH1D *uhv5eta;
  TH1D *uhv6eta;

  TH1D *uhv0eta;
  TH1D *uhv0etah;
  TH1D *uhv0etaj;

  TH1D *uhv2eta_db;
  TH1D *uhv2etah_db;
  TH1D *uhv2etaj_db;

  TH1D *uhv3eta_db;
  TH1D *uhv4eta_db;
  TH1D *uhv5eta_db;
  TH1D *uhv6eta_db;

  TH1D *uhv0eta_db;
  TH1D *uhv0etah_db;
  TH1D *uhv0etaj_db;
};

Hydjet2Analyzer::Hydjet2Analyzer(const edm::ParameterSet &iConfig) {
  fBFileName = iConfig.getUntrackedParameter<std::string>("output_b", "b_values.txt");
  fNFileName = iConfig.getUntrackedParameter<std::string>("output_n", "n_values.txt");
  fMFileName = iConfig.getUntrackedParameter<std::string>("output_m", "m_values.txt");
  doAnalysis_ = iConfig.getUntrackedParameter<bool>("doAnalysis", false);
  useHepMCProduct_ = iConfig.getUntrackedParameter<bool>("useHepMCProduct", true);
  printLists_ = iConfig.getUntrackedParameter<bool>("printLists", false);
  doCF_ = iConfig.getUntrackedParameter<bool>("doMixed", false);
  doVertex_ = iConfig.getUntrackedParameter<bool>("doVertex", false);
  if (doVertex_) {
    simVerticesTag_ = iConfig.getParameter<edm::InputTag>("simVerticesTag");
  }
  etaMax_ = iConfig.getUntrackedParameter<double>("etaMax", 2.);
  ptMin_ = iConfig.getUntrackedParameter<double>("ptMin", 0);
  srcT_ = mayConsume<HepMCProduct>(
      iConfig.getUntrackedParameter<edm::InputTag>("src", edm::InputTag("generator", "unsmeared")));
  srcTmix_ = consumes<CrossingFrame<edm::HepMCProduct>>(
      iConfig.getUntrackedParameter<edm::InputTag>("srcMix", edm::InputTag("mix", "generatorSmeared")));

  genParticleSrc_ = iConfig.getUntrackedParameter<edm::InputTag>("src", edm::InputTag("hiGenParticles"));
  genHIsrc_ = iConfig.getUntrackedParameter<edm::InputTag>("src", edm::InputTag("heavyIon"));

  if (useHepMCProduct_)
    pdtToken_ = esConsumes();
  doTestEvent_ = iConfig.getUntrackedParameter<bool>("doTestEvent", false);
  doParticles_ = iConfig.getUntrackedParameter<bool>("doParticles", false);
  doHistos_ = iConfig.getUntrackedParameter<bool>("doHistos", false);
  if (doHistos_) {
    userHistos_ = iConfig.getUntrackedParameter<bool>("userHistos", false);
    if (userHistos_) {
      uStatus_ = iConfig.getUntrackedParameter<int>("uStatus");
      uPDG_1 = iConfig.getUntrackedParameter<int>("uPDG_1");
      uPDG_2 = iConfig.getUntrackedParameter<int>("uPDG_2", uPDG_1);
      uPDG_3 = iConfig.getUntrackedParameter<int>("uPDG_3", uPDG_1);
      upTetaCut_ = iConfig.getUntrackedParameter<double>("uPTetaCut", 0.8);
      downTetaCut_ = iConfig.getUntrackedParameter<double>("dPTetaCut", -1.);
      uPtBins_ = iConfig.getUntrackedParameter<vector<double>>("PtBins");
      uEtaBins_ = iConfig.getUntrackedParameter<vector<double>>("EtaBins");
      uPhiBins_ = iConfig.getUntrackedParameter<vector<double>>("PhiBins");
      uV2ptBins_ = iConfig.getUntrackedParameter<vector<double>>("v2PtBins");
      uV2etaBins_ = iConfig.getUntrackedParameter<vector<double>>("v2EtaBins");

      //Pt
      int PtSize = uPtBins_.size();
      if (PtSize > 1) {
        ptBins = new float[PtSize];
        nintPt = PtSize - 1;
        for (int k = 0; k < PtSize; k++) {
          ptBins[k] = uPtBins_[k];
        }
      } else {
        nintPt = iConfig.getUntrackedParameter<int>("nintPt");
        maxPt = iConfig.getUntrackedParameter<double>("maxPt");
        minPt = iConfig.getUntrackedParameter<double>("minPt");
        ptBins = new float[nintPt + 1];
        for (int k = 0; k < nintPt + 1; k++) {
          ptBins[k] = minPt + k * ((maxPt - minPt) / nintPt);
        }
      }

      //Eta
      int EtaSize = uEtaBins_.size();
      if (EtaSize > 1) {
        etaBins = new float[EtaSize];
        nintEta = EtaSize - 1;
        for (int k = 0; k < EtaSize; k++) {
          etaBins[k] = uEtaBins_[k];
        }
      } else {
        nintEta = iConfig.getUntrackedParameter<int>("nintEta");
        maxEta = iConfig.getUntrackedParameter<double>("maxEta");
        minEta = iConfig.getUntrackedParameter<double>("minEta");
        etaBins = new float[nintEta + 1];
        for (int k = 0; k < nintEta + 1; k++) {
          etaBins[k] = minEta + k * ((maxEta - minEta) / nintEta);
        }
      }

      //Phi
      int PhiSize = uPhiBins_.size();
      if (PhiSize > 1) {
        phiBins = new float[PhiSize];
        nintPhi = PhiSize - 1;
        for (int k = 0; k < PhiSize; k++) {
          phiBins[k] = uPhiBins_[k];
        }
      } else {
        nintPhi = iConfig.getUntrackedParameter<int>("nintPhi");
        maxPhi = iConfig.getUntrackedParameter<double>("maxPhi");
        minPhi = iConfig.getUntrackedParameter<double>("minPhi");
        phiBins = new float[nintPhi + 1];
        for (int k = 0; k < nintPhi + 1; k++) {
          phiBins[k] = minPhi + k * ((maxPhi - minPhi) / nintPhi);
        }
      }

      //v2Pt
      int v2PtSize = uV2ptBins_.size();
      if (v2PtSize > 1) {
        v2ptBins = new float[v2PtSize];
        nintV2pt = v2PtSize - 1;
        for (int k = 0; k < v2PtSize; k++) {
          v2ptBins[k] = uV2ptBins_[k];
        }
      } else {
        nintV2pt = iConfig.getUntrackedParameter<int>("nintV2pt");
        maxV2pt = iConfig.getUntrackedParameter<double>("maxV2pt");
        minV2pt = iConfig.getUntrackedParameter<double>("minV2pt");
        v2ptBins = new float[nintV2pt + 1];
        for (int k = 0; k < nintV2pt + 1; k++) {
          v2ptBins[k] = minV2pt + k * ((maxV2pt - minV2pt) / nintV2pt);
        }
      }

      //v2Eta
      int v2EtaSize = uV2etaBins_.size();
      if (v2EtaSize > 1) {
        v2etaBins = new float[v2EtaSize];
        nintV2eta = v2EtaSize - 1;
        for (int k = 0; k < v2EtaSize; k++) {
          v2etaBins[k] = uV2etaBins_[k];
        }
      } else {
        nintV2eta = iConfig.getUntrackedParameter<int>("nintV2eta");
        maxV2eta = iConfig.getUntrackedParameter<double>("maxV2eta");
        minV2eta = iConfig.getUntrackedParameter<double>("minV2eta");
        v2etaBins = new float[nintV2eta + 1];
        for (int k = 0; k < nintV2eta + 1; k++) {
          v2etaBins[k] = minV2eta + k * ((maxV2eta - minV2eta) / nintV2eta);
        }
      }
    }  //user histo
  }    //do histo
}
Hydjet2Analyzer::~Hydjet2Analyzer() {}

// ------------ method called to for each event  ------------
void Hydjet2Analyzer::analyze(const edm::Event &iEvent, const edm::EventSetup &iSetup) {
  using namespace edm;
  using namespace HepMC;
  hev_.event = iEvent.id().event();
  for (int ieta = 0; ieta < ETABINS; ++ieta) {
    hev_.n[ieta] = 0;
    hev_.ptav[ieta] = 0;
  }
  hev_.mult = 0;
  double phi0 = 0.;
  double phi3 = 0.;
  double b = -1.;
  double v2, v3, v4, v5, v6;
  double scale = -1.;
  int npart = -1;
  int ncoll = -1;
  int nhard = -1;
  double vx = -99.;
  double vy = -99.;
  double vz = -99.;
  double vr = -99.;
  const GenEvent *evt;
  int nmix = -1;
  int np = 0;
  int sig = -1;
  int src = -1;
  if (useHepMCProduct_) {
    edm::ESHandle<ParticleDataTable> pdt = iSetup.getHandle(pdtToken_);

    if (doCF_) {
      Handle<CrossingFrame<HepMCProduct>> cf;
      iEvent.getByToken(srcTmix_, cf);
      MixCollection<HepMCProduct> mix(cf.product());
      nmix = mix.size();
      //cout << "Mix Collection Size: " << mix.size() <<", pileup size: " <<mix.sizePileup() << ", signal: "<<mix.sizeSignal()<< endl;
      MixCollection<HepMCProduct>::iterator mbegin = mix.begin();
      MixCollection<HepMCProduct>::iterator mend = mix.end();
      for (MixCollection<HepMCProduct>::iterator mixit = mbegin; mixit != mend; ++mixit) {
        const GenEvent *subevt = (*mixit).GetEvent();
        int all = subevt->particles_size();
        //cout << "Subevent size: " << all << " Subevent type (1-signal): "<< (mixit).getTrigger()<<" Source type (pileup=0, cosmics=1, beam halo+ =2, beam halo- =3): "<< (mixit).getSourceType()<<" Bunchcrossing number: "<< mixit.bunch()<<" Impact: " <<subevt->heavy_ion()->impact_parameter()<<endl;
        np += all;
        HepMC::GenEvent::particle_const_iterator begin = subevt->particles_begin();
        HepMC::GenEvent::particle_const_iterator end = subevt->particles_end();
        for (HepMC::GenEvent::particle_const_iterator it = begin; it != end; ++it) {
          if ((*it)->status() == 1) {
            int pdg_id = (*it)->pdg_id();
            float eta = (*it)->momentum().eta();
            float phi = (*it)->momentum().phi();
            float pt = (*it)->momentum().perp();
            const ParticleData *part = pdt->particle(pdg_id);
            int charge = static_cast<int>(part->charge());
            hev_.pt[hev_.mult] = pt;
            hev_.eta[hev_.mult] = eta;
            hev_.phi[hev_.mult] = phi;
            hev_.pdg[hev_.mult] = pdg_id;
            hev_.chg[hev_.mult] = charge;

            //cout << "Mix Particles: pt= " << pt<<" eta="<<eta<<" phi="<<phi<<" pdg="<< pdg_id<<" charge="<<charge << endl;
            eta = fabs(eta);
            int etabin = 0;
            if (eta > 0.5)
              etabin = 1;
            if (eta > 1.)
              etabin = 2;
            if (eta < 2.) {
              hev_.ptav[etabin] += pt;
              ++(hev_.n[etabin]);
            }
            ++(hev_.mult);
          }
        }
      }
    } else {  //not mixing
      Handle<HepMCProduct> mc;
      iEvent.getByToken(srcT_, mc);
      evt = mc->GetEvent();
      scale = evt->event_scale();
      const HeavyIon *hi = evt->heavy_ion();

      if (hi) {
        b = hi->impact_parameter();
        npart = hi->Npart_proj() + hi->Npart_targ();
        ncoll = hi->Ncoll();
        nhard = hi->Ncoll_hard();
        phi0 = hi->event_plane_angle();
        phi3 =
            hi->eccentricity();  // 0.;  // No HepMC entry for Psi3 exist, but in private code it's possible to use hi->eccentricity();
        if (printLists_) {
          out_b << b << endl;
          out_n << npart << endl;
        }
      }

      src = evt->particles_size();
      if (doTestEvent_) {
        std::cout << "Event size: " << src << std::endl;
        mc->GetEvent()->print();
      }

      float et_sum = 0., et_barrel_sum = 0., e_sum = 0., e_barrel_sum = 0., et_cha_sum = 0., et_cha_barrel_sum = 0.,
            et_ch_sum = 0., et_ch_barrel_sum = 0., e_ch_sum = 0., e_ch_barrel_sum = 0., et_ph_sum = 0.,
            et_ph_barrel_sum = 0., e_ph_sum = 0., e_ph_barrel_sum = 0., et_n_sum = 0., et_n_barrel_sum = 0.,
            et_p_sum = 0., et_p_barrel_sum = 0., et_pi_sum = 0., et_pi_barrel_sum = 0., et_K_sum = 0.,
            et_K_barrel_sum = 0., e_n_sum = 0., e_n_barrel_sum = 0.;

      HepMC::GenEvent::particle_const_iterator begin = evt->particles_begin();
      HepMC::GenEvent::particle_const_iterator end = evt->particles_end();
      for (HepMC::GenEvent::particle_const_iterator it = begin; it != end; ++it) {
        if (((*it)->status() >= -1) && ((*it)->status() < 31)) {
          const ParticleData *part;
          int st = (*it)->status();
          int pdg_id = (*it)->pdg_id();
          float eta = (*it)->momentum().eta();
          float phi = (*it)->momentum().phi();
          float pt = (*it)->momentum().perp();

          float px = (*it)->momentum().px();
          float py = (*it)->momentum().py();
          float pz = (*it)->momentum().pz();
          float e = (*it)->momentum().e();
          float pseudoRapidity = (*it)->momentum().pseudoRapidity();
          int charge = -1;
          float mass = -1.;
          if ((pdt->particle(pdg_id))) {
            part = pdt->particle(pdg_id);
            charge = static_cast<int>(part->charge());
            mass = (part->mass());
          }
          if (mass < 0) {
            if ((abs(pdg_id) == 130) || (abs(pdg_id) == 310)) {
              mass = 0.497611;
            } else {
              //cout<<"Error! Mass for "<< pdg_id <<" not found in PDT!!!"<<endl;
              //return;
            }
          }

          float et = sqrt((pt * pt) + (mass * mass));

          //if(std::abs(eta)>5) continue;

          dhpdg_st->Fill(pdg_id, st);

          hev_.px[hev_.mult] = px;
          hev_.py[hev_.mult] = py;
          hev_.pz[hev_.mult] = pz;
          hev_.e[hev_.mult] = e;
          hev_.pseudoRapidity[hev_.mult] = pseudoRapidity;
          hev_.pt[hev_.mult] = pt;
          hev_.eta[hev_.mult] = eta;
          hev_.phi[hev_.mult] = phi;
          hev_.pdg[hev_.mult] = pdg_id;
          hev_.chg[hev_.mult] = charge;

          phi = phi - phi0;
          ///
          double phiTrue;
          if (phi > pi) {
            phiTrue = phi - (2 * pi);
          } else if (phi < (-1 * pi)) {
            phiTrue = phi + (2 * pi);
          } else {
            phiTrue = phi;
          }
          ///
          v2 = std::cos(2 * (phiTrue));
          v3 = std::cos(3 * (phiTrue - phi3));
          v4 = std::cos(4 * (phiTrue));
          v5 = std::cos(5 * (phiTrue - phi3));
          v6 = std::cos(6 * (phiTrue));

          if (doHistos_) {
            //common histos
            if ((*it)->status() == 1) {
              dhpdg->Fill(pdg_id);
              dheta->Fill(eta);

              e_sum += e;
              et_sum += et;
              if (std::abs(eta) < 1.5) {
                e_barrel_sum += e;
                et_barrel_sum += et;
              }

              if (std::abs(pdg_id) == 22) {  //ph
                e_ph_sum += e;
                et_ph_sum += et;
                if (std::abs(eta) < 1.5) {
                  e_ph_barrel_sum += e;
                  et_ph_barrel_sum += et;
                }
                if (std::abs(eta) < 0.8) {  //ALICE
                  dhpt_ph->Fill(pt);
                }
                dheta_ph->Fill(eta);
              }

              if (std::abs(pdg_id) == 2112) {  //n
                e_n_sum += e;
                et_n_sum += et;
                if (std::abs(eta) < 1.5) {
                  e_n_barrel_sum += e;
                  et_n_barrel_sum += et;
                }
                if (std::abs(eta) < 0.8) {
                  dhpt_n->Fill(pt);
                }
                dheta_n->Fill(eta);
              }

              if (std::abs(pdg_id) == 2212) {  //p
                et_p_sum += et;
                if (std::abs(eta) < 1.5) {
                  et_p_barrel_sum += et;
                }
                if (std::abs(eta) < 0.8) {
                  dhpt_p->Fill(pt);
                }
                dheta_p->Fill(eta);
              }

              if (std::abs(pdg_id) == 211) {  //pi
                et_pi_sum += et;
                if (std::abs(eta) < 1.5) {
                  et_pi_barrel_sum += et;
                }
                if (std::abs(eta) < 0.8) {
                  dhpt_pi->Fill(pt);
                }
                dheta_pi->Fill(eta);
              }

              if (std::abs(pdg_id) == 321) {  //K
                et_K_sum += et;
                if (std::abs(eta) < 1.5) {
                  et_K_barrel_sum += et;
                }
                if (std::abs(eta) < 0.8) {
                  dhpt_K->Fill(pt);
                }
                dheta_K->Fill(eta);
              }

              if (std::abs(eta) < 0.8) {
                dhpt->Fill(pt);
                dhphi->Fill(phiTrue);
              }

              if (charge == 1) {
                et_cha_sum += et;
                if (std::abs(eta) < 1.5) {
                  et_cha_barrel_sum += et;
                }

                if (std::abs(eta) < 1.) {  //CMS
                  dhv0pt_cha->Fill(pt, 1.);
                  dhv2pt_cha->Fill(pt, v2);
                }

                if (std::abs(eta) < 0.8) {  //ALICE
                  dhpt_cha->Fill(pt);
                  dhphi_cha->Fill(phiTrue);
                }
                dhv0eta_cha->Fill(eta, 1.);
                dhv2eta_cha->Fill(eta, v2);
                dheta_cha->Fill(eta);
              }

              if (std::abs(pdg_id) == 211 || std::abs(pdg_id) == 321 || std::abs(pdg_id) == 2212) {  //ch
                et_ch_sum += et;
                e_ch_sum += e;

                if (std::abs(eta) < 1.5) {
                  e_ch_barrel_sum += e;
                  et_ch_barrel_sum += et;
                }

                if (std::abs(eta) < 0.8) {
                  dhv0pt_ch->Fill(pt, 1.);
                  dhv2pt_ch->Fill(pt, v2);
                  dhpt_ch->Fill(pt);
                  dhphi_ch->Fill(phiTrue);
                }

                dhv0eta_ch->Fill(eta, 1.);
                dhv2eta_ch->Fill(eta, v2);
                dheta_ch->Fill(eta);
              }  //ch
            }    //status 1

            //user histos
            if (userHistos_ && ((uStatus_ == 3) || (((*it)->status() < 10) && (uStatus_ == 1)) ||
                                (((*it)->status() > 10) && (uStatus_ == 2)))) {  //user status

              //set1
              if (std::abs(pdg_id) == uPDG_1 || std::abs(pdg_id) == uPDG_2 || std::abs(pdg_id) == uPDG_3) {  //uPDG
                if ((uStatus_ == 3) && ((*it)->status() < 10))
                  cout << "ustatus=3, but stab. part. found!!!" << endl;

                if (std::abs(eta) > downTetaCut_ && std::abs(eta) < upTetaCut_) {  //eta cut

                  uhv0pt->Fill(pt, 1.);
                  uhv2pt->Fill(pt, v2);
                  uhv3pt->Fill(pt, v3);
                  uhv4pt->Fill(pt, v4);
                  uhv5pt->Fill(pt, v5);
                  uhv6pt->Fill(pt, v6);

                  uhv0pt_db->Fill(pt, 1.);
                  uhv2pt_db->Fill(pt, v2);
                  uhv3pt_db->Fill(pt, v3);
                  uhv4pt_db->Fill(pt, v4);
                  uhv5pt_db->Fill(pt, v5);
                  uhv6pt_db->Fill(pt, v6);

                  if (pt >= 1.5 && pt < 10.) {
                    uhv2Npart->Fill(npart, v2);
                    uhv2Npart_db->Fill(npart, v2);

                    uhPtNpart->Fill(npart, pt);
                    uhPtNpart_db->Fill(npart, pt);

                    uhNpart->Fill(npart, 1.);
                    uhNpart_db->Fill(npart, 1.);
                  }

                  uhpt->Fill(pt);
                  uhpt_db->Fill(pt);
                  uhphi->Fill(phiTrue);

                  if (((*it)->status() == 16) || ((*it)->status() == 6)) {  //hydro
                    uhv0pth->Fill(pt, 1.);
                    uhv2pth->Fill(pt, v2);

                    uhv0pth_db->Fill(pt, 1.);
                    uhv2pth_db->Fill(pt, v2);

                    if (pt >= 1.5 && pt < 10.) {
                      uhv2Nparth->Fill(npart, v2);
                      uhv2Nparth_db->Fill(npart, v2);
                    }

                    uhPtNparth->Fill(npart, pt);
                    uhPtNparth_db->Fill(npart, pt);

                    uhpth->Fill(pt);
                    uhpth_db->Fill(pt);
                    uhphih->Fill(phiTrue);
                  }

                  if (((*it)->status() == 17) || ((*it)->status() == 7)) {  //jet
                    uhv0ptj->Fill(pt, 1.);
                    uhv2ptj->Fill(pt, v2);

                    uhv0ptj_db->Fill(pt, 1.);
                    uhv2ptj_db->Fill(pt, v2);

                    if (pt >= 1.5 && pt < 10.) {
                      uhv2Npartj->Fill(npart, v2);
                      uhv2Npartj_db->Fill(npart, v2);
                    }

                    uhPtNpartj->Fill(npart, pt);
                    uhPtNpartj_db->Fill(npart, pt);

                    uhptj->Fill(pt);
                    uhptj_db->Fill(pt);
                    uhphij->Fill(phiTrue);
                  }
                }  //eta cut

                uheta->Fill(eta);

                uhv0eta->Fill(eta, 1.);
                uhv2eta->Fill(eta, v2);
                uhv3eta->Fill(eta, v3);
                uhv4eta->Fill(eta, v4);
                uhv5eta->Fill(eta, v5);
                uhv6eta->Fill(eta, v6);

                uhv0eta_db->Fill(eta, 1.);
                uhv2eta_db->Fill(eta, v2);
                uhv3eta_db->Fill(eta, v3);
                uhv4eta_db->Fill(eta, v4);
                uhv5eta_db->Fill(eta, v5);
                uhv6eta_db->Fill(eta, v6);

                if (((*it)->status() == 16) || ((*it)->status() == 6)) {  //hydro
                  uhv2etah->Fill(eta, v2);
                  uhv0etah->Fill(eta, 1.);

                  uhv2etah_db->Fill(eta, v2);
                  uhv0etah_db->Fill(eta, 1.);

                  uhetah->Fill(eta);
                }
                if (((*it)->status() == 17) || ((*it)->status() == 7)) {  //jet
                  uhv2etaj->Fill(eta, v2);
                  uhv0etaj->Fill(eta, 1.);

                  uhv2etaj_db->Fill(eta, v2);
                  uhv0etaj_db->Fill(eta, 1.);

                  uhetaj->Fill(eta);
                }
              }  //uPDG

            }  //user status

          }  //doHistos_

          eta = fabs(eta);
          int etabin = 0;
          if (eta > 0.5)
            etabin = 1;
          if (eta > 1.)
            etabin = 2;
          if (eta < 2.) {
            hev_.ptav[etabin] += pt;
            ++(hev_.n[etabin]);
          }
          ++(hev_.mult);
        }
      }  //particle iterator
      dhet_sum->Fill(et_sum);
      dhet_barrel_sum->Fill(et_barrel_sum);
      dhe_sum->Fill(e_sum);
      dhe_barrel_sum->Fill(e_barrel_sum);

      dhet_cha_sum->Fill(et_cha_sum);
      dhet_cha_barrel_sum->Fill(et_cha_barrel_sum);

      dhet_ch_sum->Fill(et_ch_sum);
      dhet_ch_barrel_sum->Fill(et_ch_barrel_sum);
      dhe_ch_sum->Fill(e_ch_sum);
      dhe_ch_barrel_sum->Fill(e_ch_barrel_sum);

      dhet_ph_sum->Fill(et_ph_sum);
      dhet_ph_barrel_sum->Fill(et_ph_barrel_sum);
      dhe_ph_sum->Fill(e_ph_sum);
      dhe_ph_barrel_sum->Fill(e_ph_barrel_sum);

      dhet_n_sum->Fill(et_n_sum);
      dhet_n_barrel_sum->Fill(et_n_barrel_sum);
      dhe_n_sum->Fill(e_n_sum);
      dhe_n_barrel_sum->Fill(e_n_barrel_sum);

      dhet_p_sum->Fill(et_p_sum);
      dhet_p_barrel_sum->Fill(et_p_barrel_sum);
      dhet_pi_sum->Fill(et_pi_sum);
      dhet_pi_barrel_sum->Fill(et_pi_barrel_sum);
      dhet_K_sum->Fill(et_K_sum);
      dhet_K_barrel_sum->Fill(et_K_barrel_sum);

    }       //not mixing
  } else {  // not HepMC
    edm::Handle<reco::GenParticleCollection> parts;
    iEvent.getByLabel(genParticleSrc_, parts);
    for (unsigned int i = 0; i < parts->size(); ++i) {
      const reco::GenParticle &p = (*parts)[i];
      hev_.pt[hev_.mult] = p.pt();
      hev_.eta[hev_.mult] = p.eta();
      hev_.phi[hev_.mult] = p.phi();
      hev_.pdg[hev_.mult] = p.pdgId();
      hev_.chg[hev_.mult] = p.charge();
      double eta = fabs(p.eta());

      int etabin = 0;
      if (eta > 0.5)
        etabin = 1;
      if (eta > 1.)
        etabin = 2;
      if (eta < 2.) {
        hev_.ptav[etabin] += p.pt();
        ++(hev_.n[etabin]);
      }
      ++(hev_.mult);
    }
    if (doHI_) {
      edm::Handle<GenHIEvent> higen;
      iEvent.getByLabel(genHIsrc_, higen);
    }
  }

  if (doVertex_) {
    edm::Handle<edm::SimVertexContainer> simVertices;
    iEvent.getByLabel<edm::SimVertexContainer>(simVerticesTag_, simVertices);

    if (!simVertices.isValid())
      throw cms::Exception("FatalError") << "No vertices found\n";
    int inum = 0;

    edm::SimVertexContainer::const_iterator it = simVertices->begin();
    SimVertex vertex = (*it);
    cout << " Vertex position " << inum << " " << vertex.position().rho() << " " << vertex.position().z() << endl;
    vx = vertex.position().x();
    vy = vertex.position().y();
    vz = vertex.position().z();
    vr = vertex.position().rho();
  }

  for (int i = 0; i < 3; ++i) {
    hev_.ptav[i] = hev_.ptav[i] / hev_.n[i];
  }

  hev_.b = b;
  hev_.scale = scale;
  hev_.npart = npart;
  hev_.ncoll = ncoll;
  hev_.nhard = nhard;
  hev_.phi0 = phi0;
  hev_.vx = vx;
  hev_.vy = vy;
  hev_.vz = vz;
  hev_.vr = vr;

  if (doAnalysis_) {
    nt->Fill(nmix, np, src, sig);
    hydjetTree_->Fill();
  }

  //event counter
  if (doHistos_) {
    hNev->Fill(1., 1);
  }
}
// ------------ method called once each job just before starting event loop  ------------
void Hydjet2Analyzer::beginJob() {
  if (printLists_) {
    out_b.open(fBFileName.c_str());
    if (out_b.good() == false)
      throw cms::Exception("BadFile") << "Can\'t open file " << fBFileName;
    out_n.open(fNFileName.c_str());
    if (out_n.good() == false)
      throw cms::Exception("BadFile") << "Can\'t open file " << fNFileName;
    out_m.open(fMFileName.c_str());
    if (out_m.good() == false)
      throw cms::Exception("BadFile") << "Can\'t open file " << fMFileName;
  }

  if (doHistos_) {
    if (userHistos_) {
      //pt
      uhpt = new TH1D("uhpt", "uhpt", nintPt, ptBins);
      uhptj = new TH1D("uhptj", "uhptj", nintPt, ptBins);
      uhpth = new TH1D("uhpth", "uhpth", nintPt, ptBins);

      //pt_db
      uhpt_db = new TH1D("uhpt_db", "uhpt_db", 1000, 0.0000000000001, 100.);
      uhptj_db = new TH1D("uhptj_db", "uhptj_db", 1000, 0.0000000000001, 100.);
      uhpth_db = new TH1D("uhpth_db", "uhpth_db", 1000, 0.0000000000001, 100.);

      //eta
      uheta = new TH1D("uheta", "uheta", nintEta, etaBins);
      uhetaj = new TH1D("uhetaj", "uhetaj", nintEta, etaBins);
      uhetah = new TH1D("uhetah", "uhetah", nintEta, etaBins);

      //phi
      uhphi = new TH1D("uhphi", "uhphi", nintPhi, phiBins);
      uhphij = new TH1D("uhphij", "uhphij", nintPhi, phiBins);
      uhphih = new TH1D("uhphih", "uhphih", nintPhi, phiBins);

      const int NbinNpar = 5;
      const double BinsNpart[NbinNpar + 1] = {0., 29., 90., 202., 346., 400.};

      //ptNpart
      uhNpart = new TH1D("uhNpart", "uhNpart", NbinNpar, BinsNpart);
      uhNpartj = new TH1D("uhNpartj", "uhNpartj", NbinNpar, BinsNpart);
      uhNparth = new TH1D("uhNparth", "uhNparth", NbinNpar, BinsNpart);

      //ptNpart_db
      uhNpart_db = new TH1D("uhNpart_db", "uhNpart_db", 400, 0., 400.);
      uhNpartj_db = new TH1D("uhNpartj_db", "uhNpartj_db", 400, 0., 400.);
      uhNparth_db = new TH1D("uhNparth_db", "uhNparth_db", 400, 0., 400.);

      //ptNpart
      uhPtNpart = new TH1D("uhptNpart", "uhptNpart", NbinNpar, BinsNpart);
      uhPtNpartj = new TH1D("uhptNpartj", "uhptNpartj", NbinNpar, BinsNpart);
      uhPtNparth = new TH1D("uhptNparth", "uhptNparth", NbinNpar, BinsNpart);

      //ptNpart_db
      uhPtNpart_db = new TH1D("uhptNpart_db", "uhptNpart_db", 400, 0., 400.);
      uhPtNpartj_db = new TH1D("uhptNpartj_db", "uhptNpartj_db", 400, 0., 400.);
      uhPtNparth_db = new TH1D("uhptNparth_db", "uhptNparth_db", 400, 0., 400.);

      //v2Npart
      uhv2Npart = new TH1D("uhv2Npart", "uhv2Npart", NbinNpar, BinsNpart);
      uhv2Npartj = new TH1D("uhv2Npartj", "uhv2Npartj", NbinNpar, BinsNpart);
      uhv2Nparth = new TH1D("uhv2Nparth", "uhv2Nparth", NbinNpar, BinsNpart);

      //v2Npart_db
      uhv2Npart_db = new TH1D("uhv2Npart_db", "uhv2Npart_db", 400, 0., 400.);
      uhv2Npartj_db = new TH1D("uhv2Npartj_db", "uhv2Npartj_db", 400, 0., 400.);
      uhv2Nparth_db = new TH1D("uhv2Nparth_db", "uhv2Nparth_db", 400, 0., 400.);

      //v0pt
      uhv0pt = new TH1D("uhv0pt", "uhv0pt", nintV2pt, v2ptBins);
      uhv0ptj = new TH1D("uhv0ptj", "uhv0ptj", nintV2pt, v2ptBins);
      uhv0pth = new TH1D("uhv0pth", "uhv0pth", nintV2pt, v2ptBins);

      //v2pt
      uhv2pt = new TH1D("uhv2pt", "uhv2pt", nintV2pt, v2ptBins);
      uhv2ptj = new TH1D("uhv2ptj", "uhv2ptj", nintV2pt, v2ptBins);
      uhv2pth = new TH1D("uhv2pth", "uhv2pth", nintV2pt, v2ptBins);

      uhv3pt = new TH1D("uhv3pt", "uhv3pt", nintV2pt, v2ptBins);
      uhv4pt = new TH1D("uhv4pt", "uhv4pt", nintV2pt, v2ptBins);
      uhv5pt = new TH1D("uhv5pt", "uhv5pt", nintV2pt, v2ptBins);
      uhv6pt = new TH1D("uhv6pt", "uhv6pt", nintV2pt, v2ptBins);

      //v0pt
      uhv0pt_db = new TH1D("uhv0pt_db", "uhv0pt_db", 200, 0.0, 10.);
      uhv0ptj_db = new TH1D("uhv0ptj_db", "uhv0ptj_db", 200, 0.0, 10.);
      uhv0pth_db = new TH1D("uhv0pth_db", "uhv0pth_db", 200, 0.0, 10.);

      //v2pt_db
      uhv2pt_db = new TH1D("uhv2pt_db", "uhv2pt_db", 200, 0.0, 10.);
      uhv2ptj_db = new TH1D("uhv2ptj_db", "uhv2ptj_db", 200, 0.0, 10.);
      uhv2pth_db = new TH1D("uhv2pth_db", "uhv2pth_db", 200, 0.0, 10.);

      uhv3pt_db = new TH1D("uhv3pt_db", "uhv3pt_db", 200, 0.0, 10.);
      uhv4pt_db = new TH1D("uhv4pt_db", "uhv4pt_db", 200, 0.0, 10.);
      uhv5pt_db = new TH1D("uhv5pt_db", "uhv5pt_db", 200, 0.0, 10.);
      uhv6pt_db = new TH1D("uhv6pt_db", "uhv6pt_db", 200, 0.0, 10.);

      //v0eta
      uhv0eta = new TH1D("uhv0eta", "uhv0eta", nintV2eta, v2etaBins);
      uhv0etaj = new TH1D("uhv0etaj", "uhv0etaj", nintV2eta, v2etaBins);
      uhv0etah = new TH1D("uhv0etah", "uhv0etah", nintV2eta, v2etaBins);

      //v2eta
      uhv2eta = new TH1D("uhv2eta", "uhv2eta", nintV2eta, v2etaBins);
      uhv2etaj = new TH1D("uhv2etaj", "uhv2etaj", nintV2eta, v2etaBins);
      uhv2etah = new TH1D("uhv2etah", "uhv2etah", nintV2eta, v2etaBins);

      uhv3eta = new TH1D("uhv3eta", "uhv3eta", nintV2eta, v2etaBins);
      uhv4eta = new TH1D("uhv4eta", "uhv4eta", nintV2eta, v2etaBins);
      uhv5eta = new TH1D("uhv5eta", "uhv5eta", nintV2eta, v2etaBins);
      uhv6eta = new TH1D("uhv6eta", "uhv6eta", nintV2eta, v2etaBins);

      //v0eta_db
      uhv0eta_db = new TH1D("uhv0eta_db", "uhv0eta_db", 200, -5, 5.);
      uhv0etaj_db = new TH1D("uhv0etaj_db", "uhv0etaj_db", 200, -5, 5.);
      uhv0etah_db = new TH1D("uhv0etah_db", "uhv0etah_db", 200, -5, 5.);

      //v2eta_db
      uhv2eta_db = new TH1D("uhv2eta_db", "uhv2eta_db", 200, -5, 5.);
      uhv2etaj_db = new TH1D("uhv2etaj_db", "uhv2etaj_db", 200, -5, 5.);
      uhv2etah_db = new TH1D("uhv2etah_db", "uhv2etah_db", 200, -5, 5.);

      uhv3eta_db = new TH1D("uhv3eta_db", "uhv3eta_db", 200, -5, 5.);
      uhv4eta_db = new TH1D("uhv4eta_db", "uhv4eta_db", 200, -5, 5.);
      uhv5eta_db = new TH1D("uhv5eta_db", "uhv5eta_db", 200, -5, 5.);
      uhv6eta_db = new TH1D("uhv6eta_db", "uhv6eta_db", 200, -5, 5.);
    }

    dhphi = new TH1D("dhphi", "dhphi", 1000, -3.14159265358979, 3.14159265358979);

    dhpdg = new TH1D("dhpdg", "dhpdg", 20000001, -10000000.5, 10000000.5);
    dhpdg_st = new TH2D("dhpdg_st", "dhpdg_st", 1001, -500.5, 500.5, 3, -0.5, 3.5);

    dhet_sum = new TH1D("dhet_sum", "dhet_sum", 300, 0., 100000.);
    dhet_barrel_sum = new TH1D("dhet_barrel_sum", "dhet_barrel_sum", 500, 0., 100000.);
    dhe_sum = new TH1D("dhe_sum", "dhe_sum", 800, 0., 1000000.);
    dhe_barrel_sum = new TH1D("dhe_barrel_sum", "dhe_barrel_sum", 300, 0., 100000.);
    dheta = new TH1D("dheta", "dheta", 1000, -10., 10.);
    dhpt = new TH1D("dhpt", "dhpt", 1000, 0.0000000000001, 200.);

    //charged
    dhphi_cha = new TH1D("dhphi_cha", "dhphi_cha", 1000, -3.14159265358979, 3.14159265358979);
    dhet_cha_sum = new TH1D("dhet_cha_sum", "dhet_cha_sum", 200, 0., 20000.);
    dhet_cha_barrel_sum = new TH1D("dhet_cha_barrel_sum", "dhet_cha_barrel_sum", 300, 0., 10000.);
    dheta_cha = new TH1D("dheta_cha", "dheta_cha", 1000, -10., 10.);
    dhpt_cha = new TH1D("dhpt_cha", "dhpt_cha", 1000, 0.0000000000001, 100.);

    dhv2pt_cha = new TH1D("dhv2pt_cha", "dhv2pt_cha", 200, 0.0, 10.);
    dhv0pt_cha = new TH1D("dhv0pt_cha", "dhv0pt_cha", 200, 0.0, 10.);
    dhv2eta_cha = new TH1D("dhv2eta_cha", "dhv2eta_cha", 200, -5, 5.);
    dhv0eta_cha = new TH1D("dhv0eta_cha", "dhv0eta_cha", 200, -5, 5.);

    //charged hadrons
    dhphi_ch = new TH1D("dhphi_ch", "dhphi_ch", 1000, -3.14159265358979, 3.14159265358979);
    dhet_ch_sum = new TH1D("dhet_ch_sum", "dhet_ch_sum", 200, 0., 20000.);
    dhet_ch_barrel_sum = new TH1D("dhet_ch_barrel_sum", "dhet_ch_barrel_sum", 300, 0., 10000.);
    dhe_ch_sum = new TH1D("dhe_ch_sum", "dhe_ch_sum", 400, 0., 500000.);
    dhe_ch_barrel_sum = new TH1D("dhe_ch_barrel_sum", "dhe_ch_barrel_sum", 150, 0., 10000.);
    dheta_ch = new TH1D("dheta_ch", "dheta_ch", 1000, -10., 10.);
    dhpt_ch = new TH1D("dhpt_ch", "dhpt_ch", 1000, 0.0000000000001, 100.);

    dhv2pt_ch = new TH1D("dhv2pt_ch", "dhv2pt_ch", 200, 0.0, 10.);
    dhv0pt_ch = new TH1D("dhv0pt_ch", "dhv0pt_ch", 200, 0.0, 10.);
    dhv2eta_ch = new TH1D("dhv2eta_ch", "dhv2eta_ch", 200, -5, 5.);
    dhv0eta_ch = new TH1D("dhv0eta_ch", "dhv0eta_ch", 200, -5, 5.);

    //ph
    dhet_ph_sum = new TH1D("dhet_ph_sum", "dhet_ph_sum", 150, 0., 8000.);
    dhet_ph_barrel_sum = new TH1D("dhet_ph_barrel_sum", "dhet_ph_barrel_sum", 100, 0., 5000.);
    dhe_ph_sum = new TH1D("dhe_ph_sum", "dhe_ph_sum", 1000, 0., 200000.);
    dhe_ph_barrel_sum = new TH1D("dhe_ph_barrel_sum", "dhe_ph_barrel_sum", 100, 0., 5000.);
    dheta_ph = new TH1D("dheta_ph", "dheta_ph", 1000, -10., 10.);
    dhpt_ph = new TH1D("dhpt_ph", "dhpt_ph", 1000, 0.0000000000001, 100.);

    //n
    dhet_n_sum = new TH1D("dhet_n_sum", "dhet_n_sum", 150, 0., 3000.);
    dhet_n_barrel_sum = new TH1D("dhet_n_barrel_sum", "dhet_n_barrel_sum", 100, 0., 1100.);
    dhe_n_sum = new TH1D("dhe_n_sum", "dhe_n_sum", 600, 0., 200000.);
    dhe_n_barrel_sum = new TH1D("dhe_n_barrel_sum", "dhe_n_barrel_sum", 100, 0., 1100.);
    dheta_n = new TH1D("dheta_n", "dheta_n", 1000, -10., 10.);
    dhpt_n = new TH1D("dhpt_n", "dhpt_n", 1000, 0.0000000000001, 100.);

    //p
    dhet_p_sum = new TH1D("dhet_p_sum", "dhet_p_sum", 150, 0., 3000.);
    dhet_p_barrel_sum = new TH1D("dhet_p_barrel_sum", "dhet_p_barrel_sum", 100, 0., 1100.);
    dheta_p = new TH1D("dheta_p", "dheta_p", 1000, -10., 10.);
    dhpt_p = new TH1D("dhpt_p", "dhpt_p", 1000, 0.0000000000001, 100.);

    //pi
    dhet_pi_sum = new TH1D("dhet_pi_sum", "dhet_pi_sum", 300, 6000., 18000.);
    dhet_pi_barrel_sum = new TH1D("dhet_pi_barrel_sum", "dhet_pi_barrel_sum", 300, 1000., 7000.);
    dheta_pi = new TH1D("dheta_pi", "dheta_pi", 1000, -10., 10.);
    dhpt_pi = new TH1D("dhpt_pi", "dhpt_pi", 1000, 0.0000000000001, 100.);

    //K
    dhet_K_sum = new TH1D("dhet_K_sum", "dhet_K_sum", 150, 1500., 4500.);
    dhet_K_barrel_sum = new TH1D("dhet_K_barrel_sum", "dhet_K_barrel_sum", 100, 500., 1600.);
    dheta_K = new TH1D("dheta_K", "dheta_K", 1000, -10., 10.);
    dhpt_K = new TH1D("dhpt_K", "dhpt_K", 1000, 0.0000000000001, 100.);

    hNev = new TH1D("hNev", "hNev", 1, 0., 2.);
  }

  if (doAnalysis_) {
    usesResource(TFileService::kSharedResource);
    edm::Service<TFileService> f;
    nt = f->make<TNtuple>("nt", "Mixing Analysis", "mix:np:src:sig");
    hydjetTree_ = f->make<TTree>("hi", "Tree of Hydjet2 Events");
    hydjetTree_->Branch("event", &hev_.event, "event/I");
    hydjetTree_->Branch("b", &hev_.b, "b/F");
    hydjetTree_->Branch("npart", &hev_.npart, "npart/F");
    hydjetTree_->Branch("ncoll", &hev_.ncoll, "ncoll/F");
    hydjetTree_->Branch("nhard", &hev_.nhard, "nhard/F");
    hydjetTree_->Branch("phi0", &hev_.phi0, "phi0/F");
    hydjetTree_->Branch("scale", &hev_.scale, "scale/F");
    hydjetTree_->Branch("n", hev_.n, "n[3]/I");
    hydjetTree_->Branch("ptav", hev_.ptav, "ptav[3]/F");
    if (doParticles_) {
      hydjetTree_->Branch("mult", &hev_.mult, "mult/I");
      hydjetTree_->Branch("px", hev_.px, "px[mult]/F");
      hydjetTree_->Branch("py", hev_.py, "py[mult]/F");
      hydjetTree_->Branch("pz", hev_.pz, "pz[mult]/F");
      hydjetTree_->Branch("e", hev_.e, "e[mult]/F");
      hydjetTree_->Branch("pseudoRapidity", hev_.pseudoRapidity, "pseudoRapidity[mult]/F");
      hydjetTree_->Branch("pt", hev_.pt, "pt[mult]/F");
      hydjetTree_->Branch("eta", hev_.eta, "eta[mult]/F");
      hydjetTree_->Branch("phi", hev_.phi, "phi[mult]/F");
      hydjetTree_->Branch("pdg", hev_.pdg, "pdg[mult]/I");
      hydjetTree_->Branch("chg", hev_.chg, "chg[mult]/I");

      hydjetTree_->Branch("vx", &hev_.vx, "vx/F");
      hydjetTree_->Branch("vy", &hev_.vy, "vy/F");
      hydjetTree_->Branch("vz", &hev_.vz, "vz/F");
      hydjetTree_->Branch("vr", &hev_.vr, "vr/F");
    }
  }
}
// ------------ method called once each job just after ending the event loop  ------------
void Hydjet2Analyzer::endJob() {
  if (doHistos_) {
    dhphi->Write();
    dhpdg->Write();
    dhpdg_st->Write();

    dhet_sum->Write();
    dhet_barrel_sum->Write();
    dhe_sum->Write();
    dhe_barrel_sum->Write();
    dhpt->Write();
    dheta->Write();

    dhphi_ch->Write();
    dhet_ch_sum->Write();
    dhet_ch_barrel_sum->Write();
    dhe_ch_sum->Write();
    dhe_ch_barrel_sum->Write();
    dhpt_ch->Write();
    dheta_ch->Write();
    dhv0pt_ch->Write();
    dhv2pt_ch->Write();
    dhv0eta_ch->Write();
    dhv2eta_ch->Write();

    dhphi_cha->Write();
    dhet_cha_sum->Write();
    dhet_cha_barrel_sum->Write();
    dhpt_cha->Write();
    dheta_cha->Write();
    dhv0pt_cha->Write();
    dhv2pt_cha->Write();
    dhv0eta_cha->Write();
    dhv2eta_cha->Write();

    dhet_ph_sum->Write();
    dhet_ph_barrel_sum->Write();
    dhe_ph_sum->Write();
    dhe_ph_barrel_sum->Write();
    dhpt_ph->Write();
    dheta_ph->Write();

    dhet_n_sum->Write();
    dhet_n_barrel_sum->Write();
    dhe_n_sum->Write();
    dhe_n_barrel_sum->Write();
    dhpt_n->Write();
    dheta_n->Write();

    dhet_p_sum->Write();
    dhet_p_barrel_sum->Write();
    dhpt_p->Write();
    dheta_p->Write();

    dhet_pi_sum->Write();
    dhet_pi_barrel_sum->Write();
    dhpt_pi->Write();
    dheta_pi->Write();

    dhet_K_sum->Write();
    dhet_K_barrel_sum->Write();
    dhpt_K->Write();
    dheta_K->Write();

    hNev->Write();
    if (userHistos_) {
      uhpt->Write();
      uhpth->Write();
      uhptj->Write();

      uhpt_db->Write();
      uhpth_db->Write();
      uhptj_db->Write();

      uhNpart->Write();
      uhNparth->Write();
      uhNpartj->Write();

      uhNpart_db->Write();
      uhNparth_db->Write();
      uhNpartj_db->Write();

      uhPtNpart->Write();
      uhPtNparth->Write();
      uhPtNpartj->Write();

      uhPtNpart_db->Write();
      uhPtNparth_db->Write();
      uhPtNpartj_db->Write();

      uhv2Npart->Write();
      uhv2Nparth->Write();
      uhv2Npartj->Write();

      uhv2Npart_db->Write();
      uhv2Nparth_db->Write();
      uhv2Npartj_db->Write();

      uheta->Write();
      uhetah->Write();
      uhetaj->Write();
      uhphi->Write();
      uhphih->Write();
      uhphij->Write();

      uhv0eta->Write();
      uhv0etah->Write();
      uhv0etaj->Write();

      uhv0eta_db->Write();
      uhv0etah_db->Write();
      uhv0etaj_db->Write();

      uhv0pt->Write();
      uhv0pth->Write();
      uhv0ptj->Write();

      uhv0pt_db->Write();
      uhv0pth_db->Write();
      uhv0ptj_db->Write();

      uhv2eta->Write();
      uhv2etah->Write();
      uhv2etaj->Write();

      uhv2eta_db->Write();
      uhv2etah_db->Write();
      uhv2etaj_db->Write();

      uhv2pt->Write();
      uhv2pth->Write();
      uhv2ptj->Write();

      uhv2pt_db->Write();
      uhv2pth_db->Write();
      uhv2ptj_db->Write();

      uhv3eta->Write();
      uhv4eta->Write();
      uhv5eta->Write();
      uhv6eta->Write();

      uhv3eta_db->Write();
      uhv4eta_db->Write();
      uhv5eta_db->Write();
      uhv6eta_db->Write();

      uhv3pt->Write();
      uhv4pt->Write();
      uhv5pt->Write();
      uhv6pt->Write();

      uhv3pt_db->Write();
      uhv4pt_db->Write();
      uhv5pt_db->Write();
      uhv6pt_db->Write();
    }
  }
}
//define this as a plug-in
DEFINE_FWK_MODULE(Hydjet2Analyzer);
