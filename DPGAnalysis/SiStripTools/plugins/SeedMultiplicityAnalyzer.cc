// -*- C++ -*-
//
// Package:    TrackRecoMonitoring
// Class:      SeedMultiplicityAnalyzer
//
/**\class SeedMultiplicityAnalyzer SeedMultiplicityAnalyzer.cc myTKAnalyses/TrackRecoMonitoring/src/SeedMultiplicityAnalyzer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Andrea Venturi
//         Created:  Mon Oct 27 17:37:53 CET 2008
// $Id: SeedMultiplicityAnalyzer.cc,v 1.8 2012/04/21 14:03:26 venturia Exp $
//
//

// system include files
#include <memory>

// user include files

#include <vector>
#include <string>
#include <numeric>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "TH1F.h"
#include "TH2F.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/PatternTools/interface/TSCBLBuilderNoMaterial.h"

//
// class decleration
//

class SeedMultiplicityAnalyzer : public edm::EDAnalyzer {
public:
  explicit SeedMultiplicityAnalyzer(const edm::ParameterSet&);
  ~SeedMultiplicityAnalyzer() override;

  class FromTrackRefSeedFilter {
  public:
    FromTrackRefSeedFilter();
    FromTrackRefSeedFilter(edm::ConsumesCollector&& iC, const edm::ParameterSet& iConfig);
    const std::string& suffix() const;
    void prepareEvent(const edm::Event& iEvent);
    bool isSelected(const unsigned int iseed) const;

  private:
    std::string m_suffix;
    bool m_passthrough;
    edm::EDGetTokenT<reco::TrackCollection> m_trackcollToken;
    edm::EDGetTokenT<reco::TrackRefVector> m_seltrackrefcollToken;
    edm::Handle<reco::TrackCollection> m_tracks;
    edm::Handle<reco::TrackRefVector> m_seltrackrefs;
  };

private:
  void analyze(const edm::Event&, const edm::EventSetup&) override;

  // ----------member data ---------------------------

  edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> _magFieldToken;
  edm::ESGetToken<TransientTrackingRecHitBuilder, TransientRecHitRecord> _TTRHBuilderToken;
  std::vector<edm::EDGetTokenT<TrajectorySeedCollection>> _seedcollTokens;
  std::vector<unsigned int> _seedbins;
  std::vector<double> _seedmax;
  std::vector<FromTrackRefSeedFilter> _seedfilters;
  std::vector<edm::EDGetTokenT<std::map<unsigned int, int>>> _multiplicityMapTokens;
  std::vector<std::string> _labels;
  std::vector<unsigned int> _selections;
  std::vector<unsigned int> _binsmult;
  std::vector<unsigned int> _binseta;
  std::vector<double> _maxs;
  std::vector<TH1F*> _hseedmult;
  std::vector<TH1F*> _hseedeta;
  std::vector<TH2F*> _hseedphieta;
  std::vector<TH1F*> _hpixelrhmult;
  std::vector<TH2F*> _hbpixclusleneta;
  std::vector<TH2F*> _hfpixclusleneta;
  std::vector<TH2F*> _hbpixcluslenangle;
  std::vector<TH2F*> _hfpixcluslenangle;
  std::vector<std::vector<TH2F*>> _hseedmult2D;
  std::vector<std::vector<TH2F*>> _hseedeta2D;
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
SeedMultiplicityAnalyzer::SeedMultiplicityAnalyzer(const edm::ParameterSet& iConfig)
    : _magFieldToken(esConsumes(edm::ESInputTag{"", iConfig.getParameter<std::string>("TTRHBuilder")})),
      _TTRHBuilderToken(esConsumes()),
      _seedcollTokens(),
      _seedbins(),
      _seedmax(),
      _seedfilters(),
      _multiplicityMapTokens(),
      _labels(),
      _selections(),
      _binsmult(),
      _binseta(),
      _maxs(),
      _hseedmult2D(),
      _hseedeta2D() {
  //now do what ever initialization is needed

  //

  std::vector<edm::ParameterSet> seedCollectionConfigs =
      iConfig.getParameter<std::vector<edm::ParameterSet>>("seedCollections");

  for (std::vector<edm::ParameterSet>::const_iterator scps = seedCollectionConfigs.begin();
       scps != seedCollectionConfigs.end();
       ++scps) {
    _seedcollTokens.push_back(consumes<TrajectorySeedCollection>(scps->getParameter<edm::InputTag>("src")));
    _seedbins.push_back(scps->getUntrackedParameter<unsigned int>("nBins", 1000));
    _seedmax.push_back(scps->getUntrackedParameter<double>("maxValue", 100000.));

    if (scps->exists("trackFilter")) {
      _seedfilters.push_back(
          FromTrackRefSeedFilter(consumesCollector(), scps->getParameter<edm::ParameterSet>("trackFilter")));
    } else {
      _seedfilters.push_back(FromTrackRefSeedFilter());
    }
  }

  std::vector<edm::ParameterSet> correlationConfigs =
      iConfig.getParameter<std::vector<edm::ParameterSet>>("multiplicityCorrelations");

  for (std::vector<edm::ParameterSet>::const_iterator ps = correlationConfigs.begin(); ps != correlationConfigs.end();
       ++ps) {
    _multiplicityMapTokens.push_back(
        consumes<std::map<unsigned int, int>>(ps->getParameter<edm::InputTag>("multiplicityMap")));
    _labels.push_back(ps->getParameter<std::string>("detLabel"));
    _selections.push_back(ps->getParameter<unsigned int>("detSelection"));
    _binsmult.push_back(ps->getParameter<unsigned int>("nBins"));
    _binseta.push_back(ps->getParameter<unsigned int>("nBinsEta"));
    _maxs.push_back(ps->getParameter<double>("maxValue"));
  }

  edm::Service<TFileService> tfserv;

  std::vector<unsigned int>::const_iterator nseedbins = _seedbins.begin();
  std::vector<double>::const_iterator seedmax = _seedmax.begin();
  std::vector<FromTrackRefSeedFilter>::const_iterator filter = _seedfilters.begin();

  for (std::vector<edm::ParameterSet>::const_iterator scps = seedCollectionConfigs.begin();
       scps != seedCollectionConfigs.end();
       ++scps, ++nseedbins, ++seedmax, ++filter) {
    std::string extendedlabel = std::string(scps->getParameter<edm::InputTag>("src").encode()) + filter->suffix();

    std::string hname = extendedlabel + std::string("_mult");
    std::string htitle = extendedlabel + std::string(" seed multiplicity");
    _hseedmult.push_back(tfserv->make<TH1F>(
        hname.c_str(), htitle.c_str(), *nseedbins + 1, 0.5 - *seedmax / (*nseedbins), *seedmax + 0.5));
    _hseedmult[_hseedmult.size() - 1]->GetXaxis()->SetTitle("seeds");
    _hseedmult[_hseedmult.size() - 1]->GetYaxis()->SetTitle("events");

    hname = extendedlabel + std::string("_eta");
    htitle = extendedlabel + std::string(" seed pseudorapidity");
    _hseedeta.push_back(tfserv->make<TH1F>(hname.c_str(), htitle.c_str(), 80, -4., 4.));
    _hseedeta[_hseedeta.size() - 1]->GetXaxis()->SetTitle("#eta");
    _hseedeta[_hseedeta.size() - 1]->GetYaxis()->SetTitle("seeds");

    hname = extendedlabel + std::string("_phieta");
    htitle = extendedlabel + std::string(" seed phi vs pseudorapidity");
    _hseedphieta.push_back(tfserv->make<TH2F>(hname.c_str(), htitle.c_str(), 80, -4., 4., 80, -M_PI, M_PI));
    _hseedphieta[_hseedphieta.size() - 1]->GetXaxis()->SetTitle("#eta");
    _hseedphieta[_hseedphieta.size() - 1]->GetYaxis()->SetTitle("#phi");

    _hseedmult2D.push_back(std::vector<TH2F*>());
    _hseedeta2D.push_back(std::vector<TH2F*>());

    hname = extendedlabel + std::string("_npixelrh");
    htitle = extendedlabel + std::string(" seed SiPixelRecHit multiplicity");
    _hpixelrhmult.push_back(tfserv->make<TH1F>(hname.c_str(), htitle.c_str(), 5, -.5, 4.5));
    _hpixelrhmult[_hpixelrhmult.size() - 1]->GetXaxis()->SetTitle("NRecHits");
    _hpixelrhmult[_hpixelrhmult.size() - 1]->GetYaxis()->SetTitle("seeds");

    hname = extendedlabel + std::string("_bpixleneta");
    htitle = extendedlabel + std::string(" seed BPIX cluster length vs pseudorapidity");
    _hbpixclusleneta.push_back(tfserv->make<TH2F>(hname.c_str(), htitle.c_str(), 80, -4., 4., 40, -0.5, 39.5));
    _hbpixclusleneta[_hbpixclusleneta.size() - 1]->GetXaxis()->SetTitle("#eta");
    _hbpixclusleneta[_hbpixclusleneta.size() - 1]->GetYaxis()->SetTitle("length");

    hname = extendedlabel + std::string("_fpixleneta");
    htitle = extendedlabel + std::string(" seed FPIX cluster length vs pseudorapidity");
    _hfpixclusleneta.push_back(tfserv->make<TH2F>(hname.c_str(), htitle.c_str(), 80, -4., 4., 40, -0.5, 39.5));
    _hfpixclusleneta[_hfpixclusleneta.size() - 1]->GetXaxis()->SetTitle("#eta");
    _hfpixclusleneta[_hfpixclusleneta.size() - 1]->GetYaxis()->SetTitle("length");

    hname = extendedlabel + std::string("_bpixlenangle");
    htitle = extendedlabel + std::string(" seed BPIX cluster length vs track projection");
    _hbpixcluslenangle.push_back(tfserv->make<TH2F>(hname.c_str(), htitle.c_str(), 200, -1., 1., 40, -0.5, 39.5));
    _hbpixcluslenangle[_hbpixcluslenangle.size() - 1]->GetXaxis()->SetTitle("projection");
    _hbpixcluslenangle[_hbpixcluslenangle.size() - 1]->GetYaxis()->SetTitle("length");

    hname = extendedlabel + std::string("_fpixlenangle");
    htitle = extendedlabel + std::string(" seed FPIX cluster length vs track projection");
    _hfpixcluslenangle.push_back(tfserv->make<TH2F>(hname.c_str(), htitle.c_str(), 200, -1., 1., 40, -0.5, 39.5));
    _hfpixcluslenangle[_hfpixcluslenangle.size() - 1]->GetXaxis()->SetTitle("projection");
    _hfpixcluslenangle[_hfpixcluslenangle.size() - 1]->GetYaxis()->SetTitle("length");

    for (unsigned int i = 0; i < _multiplicityMapTokens.size(); ++i) {
      std::string hname2D = extendedlabel + _labels[i];
      hname2D += "_mult";
      std::string htitle2D = extendedlabel + " seeds multiplicity";
      htitle2D += " vs ";
      htitle2D += _labels[i];
      htitle2D += " hits";
      _hseedmult2D[_hseedmult2D.size() - 1].push_back(tfserv->make<TH2F>(hname2D.c_str(),
                                                                         htitle2D.c_str(),
                                                                         _binsmult[i],
                                                                         0.,
                                                                         _maxs[i],
                                                                         *nseedbins + 1,
                                                                         0.5 - *seedmax / (*nseedbins),
                                                                         *seedmax + 0.5));
      _hseedmult2D[_hseedmult2D.size() - 1][_hseedmult2D[_hseedmult2D.size() - 1].size() - 1]->GetXaxis()->SetTitle(
          "hits");
      _hseedmult2D[_hseedmult2D.size() - 1][_hseedmult2D[_hseedmult2D.size() - 1].size() - 1]->GetYaxis()->SetTitle(
          "seeds");

      hname2D = extendedlabel + _labels[i];
      hname2D += "_eta";
      htitle2D = extendedlabel + " seeds pseudorapidity";
      htitle2D += " vs ";
      htitle2D += _labels[i];
      htitle2D += " hits";
      _hseedeta2D[_hseedeta2D.size() - 1].push_back(
          tfserv->make<TH2F>(hname2D.c_str(), htitle2D.c_str(), _binseta[i], 0., _maxs[i], 80, -4., 4.));
      _hseedeta2D[_hseedeta2D.size() - 1][_hseedeta2D[_hseedeta2D.size() - 1].size() - 1]->GetXaxis()->SetTitle("hits");
      _hseedeta2D[_hseedeta2D.size() - 1][_hseedeta2D[_hseedeta2D.size() - 1].size() - 1]->GetYaxis()->SetTitle("#eta");
    }
  }
}

SeedMultiplicityAnalyzer::~SeedMultiplicityAnalyzer() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

// ------------ method called to for each event  ------------
void SeedMultiplicityAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  // compute cluster multiplicities

  std::vector<int> tmpmult(_multiplicityMapTokens.size(), -1);
  for (unsigned int i = 0; i < _multiplicityMapTokens.size(); ++i) {
    Handle<std::map<unsigned int, int>> mults;
    iEvent.getByToken(_multiplicityMapTokens[i], mults);

    // check if the selection exists

    std::map<unsigned int, int>::const_iterator mult = mults->find(_selections[i]);

    if (mult != mults->end()) {
      tmpmult[i] = mult->second;
    } else {
      edm::LogWarning("DetSelectionNotFound") << " DetSelection " << _selections[i] << " not found";
    }
  }

  // preparation for loop on seeds

  //  TrajectoryStateTransform tsTransform;
  TSCBLBuilderNoMaterial tscblBuilder;  // I could have used TSCBLBuilderWithPropagator

  const auto theMF = &iSetup.getData(_magFieldToken);
  const auto& theTTRHBuilder = iSetup.getData(_TTRHBuilderToken);

  // I need:
  // - beamspot bs POSTPONED

  std::vector<TH1F*>::iterator histomult = _hseedmult.begin();
  std::vector<std::vector<TH2F*>>::iterator histomult2D = _hseedmult2D.begin();
  std::vector<TH1F*>::iterator histoeta = _hseedeta.begin();
  std::vector<TH2F*>::iterator histophieta = _hseedphieta.begin();
  std::vector<std::vector<TH2F*>>::iterator histoeta2D = _hseedeta2D.begin();
  std::vector<TH1F*>::iterator hpixelrhmult = _hpixelrhmult.begin();
  std::vector<TH2F*>::iterator histobpixleneta = _hbpixclusleneta.begin();
  std::vector<TH2F*>::iterator histofpixleneta = _hfpixclusleneta.begin();
  std::vector<TH2F*>::iterator histobpixlenangle = _hbpixcluslenangle.begin();
  std::vector<TH2F*>::iterator histofpixlenangle = _hfpixcluslenangle.begin();
  std::vector<FromTrackRefSeedFilter>::iterator filter = _seedfilters.begin();

  // loop on seed collections

  for (std::vector<edm::EDGetTokenT<TrajectorySeedCollection>>::const_iterator coll = _seedcollTokens.begin();
       coll != _seedcollTokens.end() && histomult != _hseedmult.end() && histomult2D != _hseedmult2D.end() &&
       histoeta != _hseedeta.end() && histoeta2D != _hseedeta2D.end() && histophieta != _hseedphieta.end() &&
       hpixelrhmult != _hpixelrhmult.end() && histobpixleneta != _hbpixclusleneta.end() &&
       histofpixleneta != _hfpixclusleneta.end() && histobpixlenangle != _hbpixcluslenangle.end() &&
       histofpixlenangle != _hfpixcluslenangle.end();
       ++coll,
                                                                               ++histomult,
                                                                               ++histomult2D,
                                                                               ++histoeta,
                                                                               ++histophieta,
                                                                               ++histoeta2D,
                                                                               ++hpixelrhmult,
                                                                               ++histobpixleneta,
                                                                               ++histofpixleneta,
                                                                               ++histobpixlenangle,
                                                                               ++histofpixlenangle,
                                                                               ++filter) {
    filter->prepareEvent(iEvent);

    Handle<TrajectorySeedCollection> seeds;
    iEvent.getByToken(*coll, seeds);

    /*
    (*histomult)->Fill(seeds->size());
    
    for(unsigned int i=0;i<_multiplicityMaps.size();++i) {
      if(tmpmult[i]>=0)	(*histomult2D)[i]->Fill(tmpmult[i],seeds->size());
    }
    */

    // loop on seeds

    unsigned int nseeds = 0;
    unsigned int iseed = 0;
    for (TrajectorySeedCollection::const_iterator seed = seeds->begin(); seed != seeds->end(); ++seed, ++iseed) {
      if (filter->isSelected(iseed)) {
        ++nseeds;

        TransientTrackingRecHit::RecHitPointer recHit = theTTRHBuilder.build(&*(seed->recHits().end() - 1));
        TrajectoryStateOnSurface state =
            trajectoryStateTransform::transientState(seed->startingState(), recHit->surface(), theMF);
        //      TrajectoryStateClosestToBeamLine tsAtClosestApproachSeed = tscblBuilder(*state.freeState(),bs); // here I need them BS

        double eta = state.globalMomentum().eta();
        double phi = state.globalMomentum().phi();

        (*histoeta)->Fill(eta);
        (*histophieta)->Fill(eta, phi);

        for (unsigned int i = 0; i < _multiplicityMapTokens.size(); ++i) {
          if (tmpmult[i] >= 0)
            (*histoeta2D)[i]->Fill(tmpmult[i], eta);
        }

        int npixelrh = 0;
        for (auto const& hit : seed->recHits()) {
          const SiPixelRecHit* sphit = dynamic_cast<const SiPixelRecHit*>(&hit);
          if (sphit) {
            ++npixelrh;
            // compute state on recHit surface
            TransientTrackingRecHit::RecHitPointer ttrhit = theTTRHBuilder.build(&hit);
            TrajectoryStateOnSurface tsos =
                trajectoryStateTransform::transientState(seed->startingState(), ttrhit->surface(), theMF);

            if (sphit->geographicalId().det() == DetId::Tracker &&
                sphit->geographicalId().subdetId() == PixelSubdetector::PixelBarrel) {
              (*histobpixleneta)->Fill(eta, sphit->cluster()->sizeY());
              if (tsos.isValid()) {
                //		double normdx = sin(atan2(tsos.localMomentum().x(),tsos.localMomentum().z()));
                double normdx = tsos.localMomentum().x() / sqrt(tsos.localMomentum().x() * tsos.localMomentum().x() +
                                                                tsos.localMomentum().z() * tsos.localMomentum().z());
                (*histobpixlenangle)->Fill(normdx, sphit->cluster()->sizeY());
              }
            } else if (sphit->geographicalId().det() == DetId::Tracker &&
                       sphit->geographicalId().subdetId() == PixelSubdetector::PixelEndcap) {
              (*histofpixleneta)->Fill(eta, sphit->cluster()->sizeX());
              if (tsos.isValid()) {
                //		double normdy = sin(atan2(tsos.localMomentum().y(),tsos.localMomentum().z()));
                double normdy = tsos.localMomentum().y() / sqrt(tsos.localMomentum().y() * tsos.localMomentum().y() +
                                                                tsos.localMomentum().z() * tsos.localMomentum().z());
                (*histofpixlenangle)->Fill(normdy, sphit->cluster()->sizeX());
              }
            } else {
              edm::LogError("InconsistentSiPixelRecHit")
                  << "SiPixelRecHit with a non-pixel DetId " << sphit->geographicalId().rawId();
            }
          }
        }
        (*hpixelrhmult)->Fill(npixelrh);
      }
    }
    (*histomult)->Fill(nseeds);

    for (unsigned int i = 0; i < _multiplicityMapTokens.size(); ++i) {
      if (tmpmult[i] >= 0)
        (*histomult2D)[i]->Fill(tmpmult[i], nseeds);
    }
  }
}

SeedMultiplicityAnalyzer::FromTrackRefSeedFilter::FromTrackRefSeedFilter()
    : m_suffix(""), m_passthrough(true), m_trackcollToken(), m_seltrackrefcollToken(), m_tracks(), m_seltrackrefs() {}

SeedMultiplicityAnalyzer::FromTrackRefSeedFilter::FromTrackRefSeedFilter(edm::ConsumesCollector&& iC,
                                                                         const edm::ParameterSet& iConfig)
    : m_suffix(iConfig.getParameter<std::string>("suffix")),
      m_passthrough(false),
      m_trackcollToken(iC.consumes<reco::TrackCollection>(iConfig.getParameter<edm::InputTag>("trkCollection"))),
      m_seltrackrefcollToken(
          iC.consumes<reco::TrackRefVector>(iConfig.getParameter<edm::InputTag>("selRefTrkCollection"))),
      m_tracks(),
      m_seltrackrefs() {}

const std::string& SeedMultiplicityAnalyzer::FromTrackRefSeedFilter::suffix() const { return m_suffix; }

void SeedMultiplicityAnalyzer::FromTrackRefSeedFilter::prepareEvent(const edm::Event& iEvent) {
  if (!m_passthrough) {
    iEvent.getByToken(m_trackcollToken, m_tracks);
    iEvent.getByToken(m_seltrackrefcollToken, m_seltrackrefs);
  }

  return;
}

bool SeedMultiplicityAnalyzer::FromTrackRefSeedFilter::isSelected(const unsigned int iseed) const {
  if (m_passthrough) {
    return true;
  } else {
    // create a reference for the element iseed of the track collection
    const reco::TrackRef trkref(m_tracks, iseed);

    // loop on the selected trackref to check if there is the same track
    for (reco::TrackRefVector::const_iterator seltrkref = m_seltrackrefs->begin(); seltrkref != m_seltrackrefs->end();
         ++seltrkref) {
      if (trkref == *seltrkref)
        return true;
    }
  }
  return false;
}

//define this as a plug-in
DEFINE_FWK_MODULE(SeedMultiplicityAnalyzer);
