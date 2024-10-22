// -*- C++ -*-
//
// Package:    Lukas/TestPythiaDecays
// Class:      TestPythiaDecays
//
/**\class TestPythiaDecays TestPythiaDecays.cc Lukas/TestPythiaDecays/plugins/TestPythiaDecays.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Lukas Vanelderen
//         Created:  Tue, 13 May 2014 09:50:05 GMT
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "SimDataFormats/Track/interface/SimTrack.h"
#include "SimDataFormats/Vertex/interface/SimVertex.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "CommonTools/Utils/interface/TFileDirectory.h"

// pythia
#include <Pythia8/Pythia.h>

// root
#include "TH1D.h"
#include "TLorentzVector.h"

#if PYTHIA_VERSION_INTEGER < 8304
typedef Pythia8::ParticleDataEntry* ParticleDataEntryPtr;
#else
typedef Pythia8::ParticleDataEntryPtr ParticleDataEntryPtr;
#endif
//
// class declaration
//

class TestPythiaDecays : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit TestPythiaDecays(const edm::ParameterSet&);
  ~TestPythiaDecays() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void analyze(const edm::Event&, const edm::EventSetup&) override;

  // ----------member data ---------------------------
  const edm::EDGetTokenT<std::vector<SimTrack> > tokTrack_;
  const edm::EDGetTokenT<std::vector<SimVertex> > tokVertex_;
  std::vector<int> pids;
  std::map<int, TH1D*> h_mass;
  std::map<int, TH1D*> h_p;
  std::map<int, TH1D*> h_v;
  std::map<int, TH1D*> h_mass_ref;
  std::map<int, TH1D*> h_plt;  // plt: proper life time
  std::map<int, TH1D*> h_originVertexRho;
  std::map<int, TH1D*> h_originVertexZ;
  std::map<int, TH1D*> h_decayVertexRho;
  std::map<int, TH1D*> h_decayVertexZ;
  std::map<int, TH1D*> h_plt_ref;  // plt: proper life time
  std::map<int, TH1D*> h_br;
  std::map<int, TH1D*> h_br_ref;

  std::map<int, std::vector<std::string> > knownDecayModes;

  Pythia8::Pythia* pythia;
  std::string outputFile;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
TestPythiaDecays::TestPythiaDecays(const edm::ParameterSet& iConfig)
    : tokTrack_(consumes<std::vector<SimTrack> >(edm::InputTag("famosSimHits"))),
      tokVertex_(consumes<std::vector<SimVertex> >(edm::InputTag("famosSimHits"))) {
  usesResource(TFileService::kSharedResource);
  // output file
  outputFile = iConfig.getParameter<std::string>("outputFile");

  // create pythia8 instance to access particle data
  pythia = new Pythia8::Pythia();
  pythia->init();
  Pythia8::ParticleData pdt = pythia->particleData;

  // which particles will we study?
  pids.push_back(15);   // tau
  pids.push_back(211);  // pi+
  pids.push_back(111);  // pi0
  pids.push_back(130);  // K0L
  pids.push_back(321);  // K+
  pids.push_back(323);  // K*(392)
  pids.push_back(411);  // D+
  pids.push_back(521);  // B+

  // define histograms
  edm::Service<TFileService> file;
  TFileDirectory observe = file->mkdir("observed");
  TFileDirectory predict = file->mkdir("prediction");
  for (size_t i = 0; i < pids.size(); ++i) {
    int pid = std::abs(pids[i]);

    // get particle data
    if (!pdt.isParticle(pid)) {
      edm::LogError("TestPythiaDecays") << "ERROR: BAD PARTICLE, pythia is not aware of pid " << pid;
      throw cms::Exception("Unknown", "FastSim") << "Bad Particle Type " << pid << "\n";
      std::exit(1);
    }
    ParticleDataEntryPtr pd = pdt.particleDataEntryPtr(pid);

    // mass histograms
    double m0 = pd->m0();
    double w = pd->mWidth();
    double mmin, mmax;
    if (w == 0) {
      mmin = m0 - m0 / 1000.;
      mmax = m0 + m0 / 1000.;
    } else {
      mmin = m0 - 2 * w;
      mmax = m0 + 2 * w;
    }
    std::stringstream strstr;
    strstr << "mass_" << pid;
    h_mass[pid] = observe.make<TH1D>(strstr.str().c_str(), strstr.str().c_str(), 100, mmin, mmax);
    h_mass_ref[pid] = predict.make<TH1D>(strstr.str().c_str(), strstr.str().c_str(), 100, mmin, mmax);
    if (w == 0)
      h_mass_ref[pid]->Fill(m0);
    else {
      for (int b = 1; b <= h_mass_ref[pid]->GetNbinsX(); ++b) {
        double _val = h_mass_ref[pid]->GetBinCenter(b);
        h_mass_ref[pid]->SetBinContent(b, TMath::BreitWigner(_val, m0, w));
      }
    }

    // p histogram
    strstr.str("");
    strstr << "p_" << pid;
    h_p[pid] = observe.make<TH1D>(strstr.str().c_str(), strstr.str().c_str(), 100, 0, 20);

    // v histogram
    strstr.str("");
    strstr << "v_" << pid;
    h_v[pid] = observe.make<TH1D>(strstr.str().c_str(), strstr.str().c_str(), 100, 0, 1.);

    // ctau histograms
    double ctau0 = pd->tau0() / 10.;
    strstr.str("");
    strstr << "plt_" << pid;
    h_plt[pid] = observe.make<TH1D>(strstr.str().c_str(), strstr.str().c_str(), 100, 0, std::min(5. * ctau0, 500.));
    h_plt_ref[pid] = predict.make<TH1D>(strstr.str().c_str(), strstr.str().c_str(), 100, 0, std::min(5. * ctau0, 500.));
    h_plt_ref[pid]->SetTitle(h_plt_ref[pid]->GetName());
    for (int b = 1; b <= h_plt_ref[pid]->GetNbinsX(); ++b) {
      double _val = h_plt_ref[pid]->GetBinCenter(b);
      h_plt_ref[pid]->SetBinContent(b, TMath::Exp(-_val / ctau0));  //convert mm to cm
    }

    // br histograms
    strstr.str("");
    strstr << "br_" << pid;
    h_br[pid] = observe.make<TH1D>(strstr.str().c_str(), strstr.str().c_str(), 0, 0, 0);
    h_br[pid]->SetCanExtend(TH1::kAllAxes);
    h_br_ref[pid] = predict.make<TH1D>(strstr.str().c_str(), strstr.str().c_str(), 0, 0, 0);
    h_br_ref[pid]->SetCanExtend(TH1::kAllAxes);
    h_br_ref[pid]->SetTitle(h_br_ref[pid]->GetName());
    knownDecayModes[pid] = std::vector<std::string>();
    for (int d = 0; d < pd->sizeChannels(); ++d) {
      Pythia8::DecayChannel& channel = pd->channel(d);
      std::vector<int> prod;
      for (int p = 0; p < channel.multiplicity(); ++p) {
        int pId = abs(channel.product(p));
        // from FastSimulation/Event/src/KineParticleFilter.cc
        bool particleCut =
            (pId > 10 && pId != 12 && pId != 14 && pId != 16 && pId != 18 && pId != 21 && (pId < 23 || pId > 40) &&
             (pId < 81 || pId > 100) && pId != 2101 && pId != 3101 && pId != 3201 && pId != 1103 && pId != 2103 &&
             pId != 2203 && pId != 3103 && pId != 3203 && pId != 3303);
        if (particleCut)
          prod.push_back(abs(channel.product(p)));
      }
      std::sort(prod.begin(), prod.end());
      strstr.str("");
      for (size_t p = 0; p < prod.size(); ++p) {
        strstr << "_" << prod[p];
      }
      std::string label = strstr.str();
      h_br[pid]->Fill(label.c_str(), 0.);
      h_br_ref[pid]->Fill(label.c_str(), channel.bRatio());
      h_br[pid]->SetEntries(0);
      knownDecayModes[pid].push_back(label);
    }

    // vertex plots
    strstr.str("");
    strstr << "originVertexRho_" << pid;
    h_originVertexRho[pid] = observe.make<TH1D>(strstr.str().c_str(), strstr.str().c_str(), 100, 0, 200);
    strstr.str("");
    strstr << "originVertexZ_" << pid;
    h_originVertexZ[pid] = observe.make<TH1D>(strstr.str().c_str(), strstr.str().c_str(), 100, 0, 400);
    strstr.str("");
    strstr << "decayVertexRho_" << pid;
    h_decayVertexRho[pid] = observe.make<TH1D>(strstr.str().c_str(), strstr.str().c_str(), 100, 0, 200);
    strstr.str("");
    strstr << "decayVertexZ_" << pid;
    h_decayVertexZ[pid] = observe.make<TH1D>(strstr.str().c_str(), strstr.str().c_str(), 100, 0, 400);
  }
}

//
// member functions
//

// ------------ method called for each event  ------------
void TestPythiaDecays::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  const edm::Handle<std::vector<SimTrack> >& simtracks = iEvent.getHandle(tokTrack_);
  const edm::Handle<std::vector<SimVertex> > simvertices = iEvent.getHandle(tokVertex_);

  // create maps

  // initialize
  std::map<size_t, std::vector<size_t> > childMap;  // child indices vs parent index
  std::map<size_t, int> parentMap;                  // parent index vs child index
  for (size_t j = 0; j < simtracks->size(); j++) {
    childMap[j] = std::vector<size_t>();
    parentMap[j] = -1;
  }

  // do the mapping
  for (size_t j = 0; j < simtracks->size(); j++) {
    size_t childIndex = j;
    const SimTrack& child = simtracks->at(childIndex);
    if (child.noVertex())
      continue;
    const SimVertex& vertex = simvertices->at(child.vertIndex());
    if (vertex.noParent())
      continue;
    size_t parentIndex = vertex.parentIndex();
    childMap[parentIndex].push_back(childIndex);
    parentMap[childIndex] = int(parentIndex);
  }

  for (size_t j = 0; j < simtracks->size(); j++) {
    const SimTrack& parent = simtracks->at(j);
    int pid = abs(parent.type());
    if (std::find(pids.begin(), pids.end(), pid) == pids.end())
      continue;

    // fill mass hist
    double mass = parent.momentum().M();
    h_mass[pid]->Fill(mass);

    // fill p hist
    h_p[pid]->Fill(parent.momentum().P());

    // fill vertex position hist
    if (!parent.noVertex()) {
      const SimVertex& originVertex = simvertices->at(parent.vertIndex());
      h_originVertexRho[pid]->Fill(originVertex.position().Rho());
      h_originVertexZ[pid]->Fill(std::fabs(originVertex.position().Z()));
    }
    if (!childMap[j].empty()) {
      const SimTrack& child = simtracks->at(childMap[j][0]);
      const SimVertex& decayVertex = simvertices->at(child.vertIndex());
      h_decayVertexRho[pid]->Fill(decayVertex.position().Rho());
      h_decayVertexZ[pid]->Fill(std::fabs(decayVertex.position().Z()));
    }
  }

  for (std::map<size_t, std::vector<size_t> >::iterator it = childMap.begin(); it != childMap.end(); ++it) {
    // fill ctau hist
    size_t parentIndex = it->first;
    const SimTrack& parent = simtracks->at(parentIndex);
    int pid = abs(parent.type());
    std::vector<size_t>& childIndices = it->second;
    if (childIndices.empty())
      continue;

    if (std::find(pids.begin(), pids.end(), pid) == pids.end())
      continue;

    const SimVertex& origin_vertex = simvertices->at(parent.vertIndex());
    const SimTrack& child0 = simtracks->at(childIndices[0]);
    const SimVertex& decay_vertex = simvertices->at(child0.vertIndex());

    TLorentzVector lv_origin_vertex(origin_vertex.position().X(),
                                    origin_vertex.position().Y(),
                                    origin_vertex.position().Z(),
                                    origin_vertex.position().T());
    TLorentzVector lv_decay_vertex(decay_vertex.position().X(),
                                   decay_vertex.position().Y(),
                                   decay_vertex.position().Z(),
                                   decay_vertex.position().T());
    TLorentzVector lv_dist = lv_decay_vertex - lv_origin_vertex;
    TLorentzVector lv_parent(
        parent.momentum().Px(), parent.momentum().Py(), parent.momentum().Pz(), parent.momentum().E());
    TVector3 boost = lv_parent.BoostVector();
    lv_dist.Boost(-boost);
    h_v[pid]->Fill(boost.Mag());
    double plt = lv_dist.T();
    h_plt[pid]->Fill(plt);

    // fill br hist
    std::vector<int> prod;
    for (size_t d = 0; d < childIndices.size(); ++d) {
      prod.push_back(abs(simtracks->at(childIndices[d]).type()));
    }
    std::sort(prod.begin(), prod.end());
    std::stringstream strstr;
    for (size_t p = 0; p < prod.size(); ++p) {
      strstr << "_" << prod[p];
    }
    std::string label = strstr.str();
    if (std::find(knownDecayModes[pid].begin(), knownDecayModes[pid].end(), label) == knownDecayModes[pid].end())
      label = "u" + label;
    h_br[pid]->Fill(label.c_str(), 1.);
    h_br_ref[pid]->Fill(label.c_str(), 0.);  // keep h_br and h_br_ref in sync
  }
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void TestPythiaDecays::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(TestPythiaDecays);
