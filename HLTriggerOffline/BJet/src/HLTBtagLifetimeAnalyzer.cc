#include <memory>
#include <string>
#include <iostream>
#include <sstream>
#include <iomanip>

#include <TFile.h>
#include <TH1.h>
#include <TH3.h>
#include <Math/GenVector/VectorUtil.h>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/JetReco/interface/JetTracksAssociation.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/BTauReco/interface/JetTag.h"
#include "SimDataFormats/JetMatching/interface/JetFlavour.h"
#include "SimDataFormats/JetMatching/interface/JetFlavourMatching.h"

static const unsigned int jetEnergyBins   =  100;
static const unsigned int jetGeometryBins =  100;
static const unsigned int vertex1DBins    = 1000;
static const unsigned int vertex3DBins    =  100;

// find the index of the object key of an association vector closest to a given jet, within a given distance
template <typename T, typename V>
int closestJet(const reco::Jet & jet, const edm::AssociationVector<T, V> & association, double distance) {
  int closest = -1;
  for (unsigned int i = 0; i < association.size(); ++i) {
    double d = ROOT::Math::VectorUtil::DeltaR(jet.momentum(), association[i].first->momentum());
    if (d < distance) {
      distance = d;
      closest  = i;
    }
  }
  return closest;
}

// jet plots (for each step)
struct JetPlots {
  JetPlots()
  { }
  
  // let ROOT handle deleting the histograms
  ~JetPlots()
  { }
  
  void init(const std::string & name, const std::string & title, unsigned int energyBins, double minEnergy, double maxEnergy, unsigned int geometryBins, double maxEta, bool hasTracks = false)
  {
    m_overall       = 0.0;
    m_name          = name;
    m_title         = title;
    m_energyBins    = energyBins;
    m_minEnergy     = minEnergy;
    m_maxEnergy     = maxEnergy;
    m_geometryBins  = geometryBins;
    m_maxEta        = maxEta;
    m_hasTracks     = hasTracks;
    
    bool sumw2 = TH1::GetDefaultSumw2();
    TH1::SetDefaultSumw2(true);
    m_jetEnergy = new TH1F((m_name + "_jets_energy").c_str(), (m_title + " jets energy").c_str(), m_energyBins,    m_minEnergy, m_maxEnergy);
    m_jetET     = new TH1F((m_name + "_jets_ET").c_str(),     (m_title + " jets ET").c_str(),     m_energyBins,    m_minEnergy, m_maxEnergy);
    m_jetEta    = new TH1F((m_name + "_jets_eta").c_str(),    (m_title + " jets eta").c_str(),    m_geometryBins, -m_maxEta,    m_maxEta);
    m_jetPhi    = new TH1F((m_name + "_jets_phi").c_str(),    (m_title + " jets phi").c_str(),    m_geometryBins, -M_PI,        M_PI);
    if (m_hasTracks) {
      m_tracksEnergy = new TH1F((m_name + "_tracks_energy").c_str(), ("Tracks in " + m_title + " jets vs. jet energy").c_str(), m_energyBins,    m_minEnergy, m_maxEnergy);
      m_tracksET     = new TH1F((m_name + "_tracks_ET").c_str(),     ("Tracks in " + m_title + " jets vs. jet ET").c_str(),     m_energyBins,    m_minEnergy, m_maxEnergy);
      m_tracksEta    = new TH1F((m_name + "_tracks_eta").c_str(),    ("Tracks in " + m_title + " jets vs. jet eta").c_str(),    m_geometryBins, -m_maxEta,    m_maxEta);
      m_tracksPhi    = new TH1F((m_name + "_tracks_phi").c_str(),    ("Tracks in " + m_title + " jets vs. jet phi").c_str(),    m_geometryBins, -M_PI,        M_PI);
    }
    TH1::SetDefaultSumw2(sumw2);
  }

  void fill(const reco::Jet & jet) {
    ++m_overall;
    m_jetEnergy->Fill(jet.energy());
    m_jetET->Fill(jet.et());
    m_jetEta->Fill(jet.eta());
    m_jetPhi->Fill(jet.phi());
  }

  void fill(const reco::Jet & jet, const reco::TrackRefVector & tracks) {
    fill(jet);
    if (m_hasTracks) {
      m_tracksEnergy->Fill(jet.energy(), tracks.size());
      m_tracksET->Fill(jet.et(), tracks.size());
      m_tracksEta->Fill(jet.eta(), tracks.size());
      m_tracksPhi->Fill(jet.phi(), tracks.size());
    }
  }

  void save(TDirectory & file) {
    m_jetEnergy->SetDirectory(&file);
    m_jetET->SetDirectory(&file);
    m_jetEta->SetDirectory(&file);
    m_jetPhi->SetDirectory(&file);
    if (m_hasTracks) {
      // normalize the number of tracks to the number of jets (i.e. compute average number of tracks per jet)
      m_tracksEnergy->Divide(m_jetEnergy);
      m_tracksEnergy->SetDirectory(&file);
      m_tracksET->Divide(m_jetET);
      m_tracksET->SetDirectory(&file);
      m_tracksEta->Divide(m_jetEta);
      m_tracksEta->SetDirectory(&file);
      m_tracksPhi->Divide(m_jetPhi);
      m_tracksPhi->SetDirectory(&file);
    }
  }

  JetPlots efficiency(const JetPlots & denominator)
  {
    std::stringstream out;

    JetPlots efficiency;
    efficiency.init(m_name + "_vs_" + denominator.m_name, m_title + " vs. " + denominator.m_title, m_energyBins, m_minEnergy, m_maxEnergy, m_geometryBins, m_maxEta, false);
    if (denominator.m_overall != 0.0) {
      efficiency.m_overall = m_overall / denominator.m_overall;
      out << std::setw(80) << std::left << efficiency.m_title << "efficiency: " << std::right << std::setw(6) << std::fixed << std::setprecision(2) << efficiency.m_overall * 100. << "%";
    } else {
      efficiency.m_overall = NAN;
      out << std::setw(80) << std::left << efficiency.m_title << "efficiency:     NaN";
    }
    std::cout << out.str() << std::endl;

    efficiency.m_jetEnergy->Divide(m_jetEnergy, denominator.m_jetEnergy, 1., 1., "B");
    efficiency.m_jetEnergy->SetMinimum(0.0);
    efficiency.m_jetEnergy->SetMaximum(1.0);
    efficiency.m_jetET->Divide(m_jetET, denominator.m_jetET, 1., 1., "B");
    efficiency.m_jetET->SetMinimum(0.0);
    efficiency.m_jetET->SetMaximum(1.0);
    efficiency.m_jetEta->Divide(m_jetEta, denominator.m_jetEta, 1., 1., "B");
    efficiency.m_jetEta->SetMinimum(0.0);
    efficiency.m_jetEta->SetMaximum(1.0);
    efficiency.m_jetPhi->Divide(m_jetPhi, denominator.m_jetPhi, 1., 1., "B");
    efficiency.m_jetPhi->SetMinimum(0.0);
    efficiency.m_jetPhi->SetMaximum(1.0);
    return efficiency;
  }
 
  TH1 * m_jetEnergy;
  TH1 * m_jetET;
  TH1 * m_jetEta;
  TH1 * m_jetPhi;
  TH1 * m_tracksEnergy;
  TH1 * m_tracksET;
  TH1 * m_tracksEta;
  TH1 * m_tracksPhi;
  double        m_overall;
  std::string   m_name;
  std::string   m_title;
  unsigned int  m_geometryBins;
  unsigned int  m_energyBins;
  double        m_minEnergy;
  double        m_maxEnergy;
  double        m_maxEta;

  bool          m_hasTracks;
};


typedef unsigned int            flavour_t;
typedef std::vector<flavour_t>  flavours_t;
struct FlavouredJetPlots {

  std::vector<flavours_t>   m_flavours;
  std::vector<std::string>  m_labels;
  std::vector<JetPlots>     m_plots;
  
  void init(
      const std::string & name, 
      const std::string & title, 
      const std::vector<flavours_t> & flavours, 
      const std::vector<std::string> & labels,
      unsigned int energyBins, 
      double minEnergy, 
      double maxEnergy, 
      unsigned int geometryBins, 
      double maxEta, 
      bool hasTracks = false
  ) {
    m_flavours = flavours;
    m_labels = labels; m_labels.push_back("other");
    m_plots.resize( m_flavours.size() + 1);
    for (unsigned int i = 0; i <= m_flavours.size(); ++i)
      m_plots[i].init(name + "_" + m_labels[i], title + " (" + m_labels[i] + ")", energyBins, minEnergy, maxEnergy, geometryBins, maxEta, hasTracks);
  }

  void fill(const reco::Jet & jet, flavour_t flavour) {
    bool match = false;
    for (unsigned int i = 0; i < m_flavours.size(); ++i) {
      if (std::find(m_flavours[i].begin(), m_flavours[i].end(), flavour) != m_flavours[i].end()) {
        m_plots[i].fill(jet);
        match = true;
      }
    }
    if (not match)
      m_plots[m_flavours.size()].fill(jet);
  }
  
  void fill(const reco::Jet & jet, const reco::TrackRefVector & tracks, flavour_t flavour) {
    bool match = false;
    for (unsigned int i = 0; i < m_flavours.size(); ++i) {
      if (std::find(m_flavours[i].begin(), m_flavours[i].end(), flavour) != m_flavours[i].end()) {
        m_plots[i].fill(jet, tracks);
        match = true;
      }
    }
    if (not match)
      m_plots[m_flavours.size()].fill(jet);
  }
  
  void save(TDirectory & file) {
    for (unsigned int i = 0; i <= m_flavours.size(); ++i)
      m_plots[i].save(file);
  }

  FlavouredJetPlots efficiency(const FlavouredJetPlots & denominator) {
    FlavouredJetPlots efficiency;
    efficiency.m_flavours = m_flavours;
    efficiency.m_labels = m_labels;
    efficiency.m_plots.resize(m_flavours.size() + 1);
    for (unsigned int i = 0; i <= m_flavours.size(); ++i)
      efficiency.m_plots[i] = m_plots[i].efficiency( denominator.m_plots[i] );
    return efficiency;
  }
};

struct OfflineJetPlots {

  std::vector<double>       m_cuts;
  std::vector<std::string>  m_labels;
  std::vector<JetPlots>     m_plots;

  void init(
      const std::string & name, 
      const std::string & title, 
      const std::vector<double> & cuts, 
      const std::vector<std::string> & labels, 
      unsigned int energyBins, 
      double minEnergy, 
      double maxEnergy, 
      unsigned int geometryBins, 
      double maxEta, 
      bool hasTracks = false
  ) {
    m_cuts = cuts;
    m_labels = labels;
    m_plots.resize( m_cuts.size() );
    for (unsigned int i = 0; i < m_cuts.size(); ++i)
      m_plots[i].init(name + "_" + m_labels[i], title + " (" + m_labels[i] + ")", energyBins, minEnergy, maxEnergy, geometryBins, maxEta, hasTracks);
  }
  
  void fill(const reco::Jet & jet, double discriminant) {
    for (unsigned int i = 0; i < m_cuts.size(); ++i)
      if (discriminant >= m_cuts[i])
        m_plots[i].fill(jet);
  }
  
  void fill(const reco::Jet & jet, const reco::TrackRefVector & tracks, double discriminant) {
    for (unsigned int i = 0; i < m_cuts.size(); ++i)
      if (discriminant >= m_cuts[i])
        m_plots[i].fill(jet, tracks);
  }

  void save(TDirectory & file) {
    for (unsigned int i = 0; i < m_cuts.size(); ++i)
      m_plots[i].save(file);
  }

  OfflineJetPlots efficiency(const OfflineJetPlots & denominator) {
    OfflineJetPlots efficiency;
    efficiency.m_cuts = m_cuts;
    efficiency.m_labels = m_labels;
    efficiency.m_plots.resize(m_cuts.size());
    for (unsigned int i = 0; i < m_cuts.size(); ++i)
      efficiency.m_plots[i] = m_plots[i].efficiency( denominator.m_plots[i] );
    return efficiency;
  }
};


struct VertexPlots {
  VertexPlots() :
    m_r(0),
    m_z(0)
  { }

  void init(const std::string & name, const std::string & title, unsigned int bins, double zRange, double rRange)
  {
    m_r  = new TH1F((name + "_R").c_str(),  (title + " R position").c_str(), bins, -rRange, rRange);
    m_z  = new TH1F((name + "_Z").c_str(),  (title + " Z position").c_str(), bins, -zRange, zRange);
  }
    
  void fill(const reco::Vertex & vertex)
  {
    m_r->Fill(vertex.position().rho());
    m_z->Fill(vertex.z());
  }

  void save(TDirectory & file)
  {
    m_r->SetDirectory(&file);
    m_z->SetDirectory(&file);
  }
  
  TH1 * m_r;
  TH1 * m_z;
};


class HLTBtagLifetimeAnalyzer : public edm::EDAnalyzer {
public:
  explicit HLTBtagLifetimeAnalyzer(const edm::ParameterSet& config);
  virtual ~HLTBtagLifetimeAnalyzer();
    
  virtual void beginJob(const edm::EventSetup & setup);
  virtual void analyze(const edm::Event & event, const edm::EventSetup & setup);
  virtual void endJob();

private:
  struct InputData {
    std::string     m_name;
    std::string     m_title;
    edm::InputTag   m_label;
    edm::InputTag   m_tracks;
  };
  
  // input collections
  edm::InputTag             m_trigger;          // HLT event
  edm::InputTag             m_vertex;           // primary vertex
  std::vector<InputData>    m_levels;

  // counters for per-event efficiencies
  std::vector<unsigned int> m_events;           // number of jets passing each level

  // match to MC truth
  edm::InputTag             m_mcPartons;        // MC truth match - jet association to partons
  std::vector<std::string>  m_mcLabels;         // MC truth match - labels
  std::vector<flavours_t>   m_mcFlavours;       // MC truth match - flavours selection
  double                    m_mcRadius;         // MC truth match - deltaR association radius

  // match to Offline reco
  edm::InputTag             m_offlineBJets;     // Offline match - jet association to discriminator
  std::vector<std::string>  m_offlineLabels;    // Offline match - labels
  std::vector<double>       m_offlineCuts;      // Offline match - discriminator cuts
  double                    m_offlineRadius;    // Offline match - deltaR association radius

  // plot configuration
  double m_jetMinEnergy;
  double m_jetMaxEnergy;
  double m_jetMaxEta;

  double m_vertexMaxR;
  double m_vertexMaxZ;

  // plot data
  VertexPlots                       m_vertexPlots;
  std::vector<JetPlots>             m_jetPlots;
  std::vector<FlavouredJetPlots>    m_mcPlots;
  std::vector<OfflineJetPlots>      m_offlinePlots;

  // output configuration
  std::string m_outputFile;
};


HLTBtagLifetimeAnalyzer::HLTBtagLifetimeAnalyzer(const edm::ParameterSet & config) :
  m_vertex( config.getParameter<edm::InputTag>("vertex") ),
  m_levels(),
  m_events(),
  m_mcPartons( config.getParameter<edm::InputTag>("mcPartons") ),
  m_mcLabels(),
  m_mcFlavours(),
  m_mcRadius( config.getParameter<double>("mcRadius") ),
  m_offlineBJets( config.getParameter<edm::InputTag>("offlineBJets") ),
  m_offlineLabels(),
  m_offlineCuts(),
  m_offlineRadius( config.getParameter<double>("offlineRadius") ),
  m_jetMinEnergy(  0. ),    //   0 GeV
  m_jetMaxEnergy( 300. ),   // 300 GeV
  m_jetMaxEta( 5. ),        //  ±5 pseudorapidity units
  m_vertexMaxR( 0.1 ),      //   1 mm
  m_vertexMaxZ( 15. ),      //  15 cm
  m_vertexPlots(),
  m_jetPlots(),
  m_mcPlots(),
  m_offlinePlots(),
  m_outputFile( config.getParameter<std::string>("outputFile") )
{
  const std::vector<edm::ParameterSet> levels = config.getParameter<std::vector<edm::ParameterSet> >("levels");
  for (unsigned int i = 0; i < levels.size(); ++i) {
    InputData level;
    level.m_label  = levels[i].getParameter<edm::InputTag>("jets");
    level.m_name   = levels[i].exists("name")   ? levels[i].getParameter<std::string>("name")  : level.m_label.encode();
    level.m_title  = levels[i].exists("title")  ? levels[i].getParameter<std::string>("title") : level.m_name;
    level.m_tracks = levels[i].exists("tracks") ? levels[i].getParameter<edm::InputTag>("tracks") : edm::InputTag("none");
    m_levels.push_back(level);
  }
    
  const edm::ParameterSet & jetConfig = config.getParameter<edm::ParameterSet>("jetConfiguration");
  m_jetMaxEnergy = jetConfig.getParameter<double>("maxEnergy");
  m_jetMaxEta    = jetConfig.getParameter<double>("maxEta");
  const edm::ParameterSet & vertexConfig = config.getParameter<edm::ParameterSet>("vertexConfiguration");
  m_vertexMaxR = vertexConfig.getParameter<double>("maxR");
  m_vertexMaxZ = vertexConfig.getParameter<double>("maxZ");

  edm::ParameterSet mc = config.getParameter<edm::ParameterSet>("mcFlavours");
  m_mcLabels = mc.getParameterNamesForType<std::vector<unsigned int> >();
  for (unsigned int i = 0; i < m_mcLabels.size(); ++i)
    m_mcFlavours.push_back( mc.getParameter<std::vector<unsigned int> >(m_mcLabels[i]) );

  edm::ParameterSet offline = config.getParameter<edm::ParameterSet>("offlineCuts");
  m_offlineLabels = offline.getParameterNamesForType<double>();
  for (unsigned int i = 0; i < m_offlineLabels.size(); ++i)
    m_offlineCuts.push_back( offline.getParameter<double>(m_offlineLabels[i]) );

}

HLTBtagLifetimeAnalyzer::~HLTBtagLifetimeAnalyzer() 
{
}

void HLTBtagLifetimeAnalyzer::beginJob(const edm::EventSetup & setup) 
{
  m_events.resize( m_levels.size(), 0 );
  m_jetPlots.resize( m_levels.size() );
  m_mcPlots.resize( m_levels.size() );
  m_offlinePlots.resize( m_levels.size() );
  
  for (unsigned int i = 0; i < m_levels.size(); ++i) {
    m_jetPlots[i].init(     m_levels[i].m_name, m_levels[i].m_title,                                 jetEnergyBins, m_jetMinEnergy, m_jetMaxEnergy, jetGeometryBins, m_jetMaxEta, m_levels[i].m_tracks.label() != "none" );
    m_mcPlots[i].init(      m_levels[i].m_name, m_levels[i].m_title, m_mcFlavours,  m_mcLabels,      jetEnergyBins, m_jetMinEnergy, m_jetMaxEnergy, jetGeometryBins, m_jetMaxEta, m_levels[i].m_tracks.label() != "none" );
    m_offlinePlots[i].init( m_levels[i].m_name, m_levels[i].m_title, m_offlineCuts, m_offlineLabels, jetEnergyBins, m_jetMinEnergy, m_jetMaxEnergy, jetGeometryBins, m_jetMaxEta, m_levels[i].m_tracks.label() != "none" );
  }
  
  m_vertexPlots.init( "PrimaryVertex", "Primary vertex", vertex1DBins, m_vertexMaxZ, m_vertexMaxR );
}

void HLTBtagLifetimeAnalyzer::analyze(const edm::Event & event, const edm::EventSetup & setup) 
{
  edm::Handle<reco::VertexCollection> h_vertex;
  event.getByLabel(m_vertex, h_vertex);
  if (h_vertex.isValid() and not h_vertex->empty())
    m_vertexPlots.fill(h_vertex->front());

  edm::Handle<reco::JetFlavourMatchingCollection> h_mcPartons;
  event.getByLabel(m_mcPartons, h_mcPartons);
  const reco::JetFlavourMatchingCollection & mcPartons = * h_mcPartons;
  
  edm::Handle<reco::JetTagCollection> h_offlineBJets;
  event.getByLabel(m_offlineBJets, h_offlineBJets);
  const reco::JetTagCollection & offlineBJets = * h_offlineBJets;

  for (unsigned int l = 0; l < m_levels.size(); ++l) {
    edm::Handle<edm::View<reco::Jet> >                  h_jets;
    edm::Handle<reco::JetTracksAssociation::Container>  h_tracks;
    
    event.getByLabel(m_levels[l].m_label, h_jets);
    if (m_levels[l].m_tracks.label() != "none")
      event.getByLabel(m_levels[l].m_tracks, h_tracks);
    
    if (h_jets.isValid()) {
      const edm::View<reco::Jet> & jets = * h_jets;
      if (jets.size() > 0)
        ++m_events[l];

      for (unsigned int j = 0; j < jets.size(); ++j) {
        const reco::Jet & jet = jets[j];
        
        // match to MC parton
        int m = closestJet(jet, mcPartons, m_mcRadius);
        unsigned int flavour = (m != -1) ? abs(mcPartons[m].second.getFlavour()) : 0;

        // match to offline reconstruted b jets
        int o = closestJet(jet, offlineBJets, m_offlineRadius);
        double discriminator = (o != -1) ? offlineBJets[o].second : -INFINITY;

        if (not h_tracks.isValid()) {
          // no tracks, fill only the jets
          m_jetPlots[l].fill( jet );
          m_mcPlots[l].fill( jet, flavour);
          m_offlinePlots[l].fill( jet, discriminator );
        } else {
          // fill jets and tracks
          const reco::TrackRefVector & tracks = (*h_tracks)[jets.refAt(j)];
          m_jetPlots[l].fill( jet, tracks );
          m_mcPlots[l].fill( jet, tracks, flavour);
          m_offlinePlots[l].fill( jet, tracks, discriminator );
        }
        
      }
    }

  }
}

void HLTBtagLifetimeAnalyzer::endJob()
{
  // compute and print overall per-event efficiencies
  for (unsigned int i = 0; i < m_levels.size(); ++i) {
    std::stringstream out;
    out << std::setw(64) << std::left << ("events passing " + m_levels[i].m_title) << std::right << std::setw(12) << m_events[i];
    std::cout << out.str() << std::endl;
  }
  for (unsigned int i = 1; i < m_levels.size(); ++i) {
    std::stringstream out;
    out << std::setw(64) << std::left << ("step efficiency at " + m_levels[i].m_title);
    if (m_events[i-1] > 0) {
      double eff = (double) m_events[i] / (double) m_events[i-1];
      out << std::right << std::setw(11) << std::fixed << std::setprecision(2) << eff * 100. << "%";
    } else {
      out << std::right << std::setw(12) << "NaN";
    }
    std::cout << out.str() << std::endl;
  }
  for (unsigned int i = 1; i < m_levels.size(); ++i) {
    std::stringstream out;
    out << std::setw(64) << std::left << ("cumulative efficiency at " + m_levels[i].m_title);
    if (m_events[0] > 0) {
      double eff = (double) m_events[i] / (double) m_events[0];
      out << std::right << std::setw(11) << std::fixed << std::setprecision(2) << eff * 100. << "%";
    } else {
      out << std::right << std::setw(12) << "NaN";
    }
    std::cout << out.str() << std::endl;
  }
  std::cout << std::endl;
  
  TFile * file = new TFile(m_outputFile.c_str(), "RECREATE");
  
  for (unsigned int i = 0; i < m_levels.size(); ++i) {
    m_jetPlots[i].save(*file);
    m_mcPlots[i].save(*file);
    m_offlinePlots[i].save(*file);
  }
  for (unsigned int i = 1; i < m_levels.size(); ++i) {
    // make step-by-step efficiency plots
    m_jetPlots[i].efficiency( m_jetPlots[i-1] ).save(*file);
    m_mcPlots[i].efficiency( m_mcPlots[i-1] ).save(*file);
    m_offlinePlots[i].efficiency( m_offlinePlots[i-1] ).save(*file);
  }
  for (unsigned int i = 2; i < m_levels.size(); ++i) {
    // make overall plots
    m_jetPlots[i].efficiency( m_jetPlots[0] ).save(*file);
    m_mcPlots[i].efficiency( m_mcPlots[0] ).save(*file);
    m_offlinePlots[i].efficiency( m_offlinePlots[0] ).save(*file);
  }

  m_vertexPlots.save(*file);

  file->Write();
  file->Close();
}


#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HLTBtagLifetimeAnalyzer);
