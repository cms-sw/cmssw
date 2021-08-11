#include "Alignment/MuonAlignmentAlgorithms/plugins/CSCOverlapsAlignmentAlgorithm.h"

CSCOverlapsAlignmentAlgorithm::CSCOverlapsAlignmentAlgorithm(const edm::ParameterSet& iConfig,
                                                             edm::ConsumesCollector& iC)
    : AlignmentAlgorithmBase(iConfig, iC),
      m_minHitsPerChamber(iConfig.getParameter<int>("minHitsPerChamber")),
      m_maxdrdz(iConfig.getParameter<double>("maxdrdz")),
      m_fiducial(iConfig.getParameter<bool>("fiducial")),
      m_useHitWeights(iConfig.getParameter<bool>("useHitWeights")),
      m_slopeFromTrackRefit(iConfig.getParameter<bool>("slopeFromTrackRefit")),
      m_minStationsInTrackRefits(iConfig.getParameter<int>("minStationsInTrackRefits")),
      m_truncateSlopeResid(iConfig.getParameter<double>("truncateSlopeResid")),
      m_truncateOffsetResid(iConfig.getParameter<double>("truncateOffsetResid")),
      m_combineME11(iConfig.getParameter<bool>("combineME11")),
      m_useTrackWeights(iConfig.getParameter<bool>("useTrackWeights")),
      m_errorFromRMS(iConfig.getParameter<bool>("errorFromRMS")),
      m_minTracksPerOverlap(iConfig.getParameter<int>("minTracksPerOverlap")),
      m_makeHistograms(iConfig.getParameter<bool>("makeHistograms")),
      m_cscGeometryToken(iC.esConsumes<edm::Transition::BeginRun>()),
      m_propToken(iC.esConsumes(edm::ESInputTag(
          "",
          m_slopeFromTrackRefit
              ? iConfig.getParameter<edm::ParameterSet>("TrackTransformer").getParameter<std::string>("Propagator")
              : std::string("")))),
      m_tthbToken(iC.esConsumes(edm::ESInputTag("", "TransientTrackBuilder"))),
      m_mode_string(iConfig.getParameter<std::string>("mode")),
      m_reportFileName(iConfig.getParameter<std::string>("reportFileName")),
      m_minP(iConfig.getParameter<double>("minP")),
      m_maxRedChi2(iConfig.getParameter<double>("maxRedChi2")),
      m_writeTemporaryFile(iConfig.getParameter<std::string>("writeTemporaryFile")),
      m_readTemporaryFiles(iConfig.getParameter<std::vector<std::string> >("readTemporaryFiles")),
      m_doAlignment(iConfig.getParameter<bool>("doAlignment")) {
  if (m_mode_string == std::string("phiy"))
    m_mode = CSCPairResidualsConstraint::kModePhiy;
  else if (m_mode_string == std::string("phipos"))
    m_mode = CSCPairResidualsConstraint::kModePhiPos;
  else if (m_mode_string == std::string("phiz"))
    m_mode = CSCPairResidualsConstraint::kModePhiz;
  else if (m_mode_string == std::string("radius"))
    m_mode = CSCPairResidualsConstraint::kModeRadius;
  else
    throw cms::Exception("BadConfig") << "mode must be one of \"phiy\", \"phipos\", \"phiz\", \"radius\"" << std::endl;

  std::vector<edm::ParameterSet> fitters = iConfig.getParameter<std::vector<edm::ParameterSet> >("fitters");
  for (std::vector<edm::ParameterSet>::const_iterator fitter = fitters.begin(); fitter != fitters.end(); ++fitter) {
    m_fitters.push_back(CSCChamberFitter(*fitter, m_residualsConstraints));
  }

  for (std::vector<CSCPairResidualsConstraint*>::const_iterator residualsConstraint = m_residualsConstraints.begin();
       residualsConstraint != m_residualsConstraints.end();
       ++residualsConstraint) {
    (*residualsConstraint)->configure(this);
    m_quickChamberLookup[std::pair<CSCDetId, CSCDetId>((*residualsConstraint)->id_i(), (*residualsConstraint)->id_j())] =
        *residualsConstraint;
  }

  if (m_slopeFromTrackRefit) {
    m_trackTransformer = new TrackTransformer(iConfig.getParameter<edm::ParameterSet>("TrackTransformer"));
  } else {
    m_trackTransformer = nullptr;
  }

  m_propagatorPointer = nullptr;

  if (m_makeHistograms) {
    edm::Service<TFileService> tFileService;
    m_histP10 = tFileService->make<TH1F>("P10", "", 100, 0, 10);
    m_histP100 = tFileService->make<TH1F>("P100", "", 100, 0, 100);
    m_histP1000 = tFileService->make<TH1F>("P1000", "", 100, 0, 1000);

    m_hitsPerChamber = tFileService->make<TH1F>("hitsPerChamber", "", 10, -0.5, 9.5);

    m_fiducial_ME11 = tFileService->make<TProfile>("fiducial_ME11", "", 100, 0.075, 0.100);
    m_fiducial_ME12 = tFileService->make<TProfile>("fiducial_ME12", "", 100, 0.080, 0.105);
    m_fiducial_MEx1 = tFileService->make<TProfile>("fiducial_MEx1", "", 100, 0.160, 0.210);
    m_fiducial_MEx2 = tFileService->make<TProfile>("fiducial_MEx2", "", 100, 0.080, 0.105);

    m_slope = tFileService->make<TH1F>("slope", "", 100, -0.5, 0.5);
    m_slope_MEp4 = tFileService->make<TH1F>("slope_MEp4", "", 100, -0.5, 0.5);
    m_slope_MEp3 = tFileService->make<TH1F>("slope_MEp3", "", 100, -0.5, 0.5);
    m_slope_MEp2 = tFileService->make<TH1F>("slope_MEp2", "", 100, -0.5, 0.5);
    m_slope_MEp1 = tFileService->make<TH1F>("slope_MEp1", "", 100, -0.5, 0.5);
    m_slope_MEm1 = tFileService->make<TH1F>("slope_MEm1", "", 100, -0.5, 0.5);
    m_slope_MEm2 = tFileService->make<TH1F>("slope_MEm2", "", 100, -0.5, 0.5);
    m_slope_MEm3 = tFileService->make<TH1F>("slope_MEm3", "", 100, -0.5, 0.5);
    m_slope_MEm4 = tFileService->make<TH1F>("slope_MEm4", "", 100, -0.5, 0.5);

    m_slopeResiduals = tFileService->make<TH1F>("slopeResiduals", "mrad", 300, -30., 30.);
    m_slopeResiduals_weighted = tFileService->make<TH1F>("slopeResiduals_weighted", "mrad", 300, -30., 30.);
    m_slopeResiduals_normalized = tFileService->make<TH1F>("slopeResiduals_normalized", "", 200, -20., 20.);
    m_offsetResiduals = tFileService->make<TH1F>("offsetResiduals", "mm", 300, -30., 30.);
    m_offsetResiduals_weighted = tFileService->make<TH1F>("offsetResiduals_weighted", "mm", 300, -30., 30.);
    m_offsetResiduals_normalized = tFileService->make<TH1F>("offsetResiduals_normalized", "", 200, -20., 20.);

    m_drdz = tFileService->make<TH1F>("drdz", "", 100, -0.5, 0.5);

    m_occupancy = tFileService->make<TH2F>("occupancy", "", 36, 1, 37, 20, 1, 21);
    for (int i = 1; i <= 36; i++) {
      std::stringstream pairname;
      pairname << i << "-";
      if (i + 1 == 37)
        pairname << 1;
      else
        pairname << (i + 1);
      m_occupancy->GetXaxis()->SetBinLabel(i, pairname.str().c_str());
    }
    m_occupancy->GetYaxis()->SetBinLabel(1, "ME-4/2");
    m_occupancy->GetYaxis()->SetBinLabel(2, "ME-4/1");
    m_occupancy->GetYaxis()->SetBinLabel(3, "ME-3/2");
    m_occupancy->GetYaxis()->SetBinLabel(4, "ME-3/1");
    m_occupancy->GetYaxis()->SetBinLabel(5, "ME-2/2");
    m_occupancy->GetYaxis()->SetBinLabel(6, "ME-2/1");
    m_occupancy->GetYaxis()->SetBinLabel(7, "ME-1/3");
    m_occupancy->GetYaxis()->SetBinLabel(8, "ME-1/2");
    if (!m_combineME11) {
      m_occupancy->GetYaxis()->SetBinLabel(9, "ME-1/1b");
      m_occupancy->GetYaxis()->SetBinLabel(10, "ME-1/1a");
      m_occupancy->GetYaxis()->SetBinLabel(11, "ME+1/1a");
      m_occupancy->GetYaxis()->SetBinLabel(12, "ME+1/1b");
    } else {
      m_occupancy->GetYaxis()->SetBinLabel(9, "ME-1/1");
      m_occupancy->GetYaxis()->SetBinLabel(10, "");
      m_occupancy->GetYaxis()->SetBinLabel(11, "");
      m_occupancy->GetYaxis()->SetBinLabel(12, "ME+1/1");
    }
    m_occupancy->GetYaxis()->SetBinLabel(13, "ME+1/2");
    m_occupancy->GetYaxis()->SetBinLabel(14, "ME+1/3");
    m_occupancy->GetYaxis()->SetBinLabel(15, "ME+2/1");
    m_occupancy->GetYaxis()->SetBinLabel(16, "ME+2/2");
    m_occupancy->GetYaxis()->SetBinLabel(17, "ME+3/1");
    m_occupancy->GetYaxis()->SetBinLabel(18, "ME+3/2");
    m_occupancy->GetYaxis()->SetBinLabel(19, "ME+4/1");
    m_occupancy->GetYaxis()->SetBinLabel(20, "ME+4/2");

    m_XYpos_mep1 = tFileService->make<TH2F>("XYpos_mep1", "Positions: ME+1", 140, -700., 700., 140, -700., 700.);
    m_XYpos_mep2 = tFileService->make<TH2F>("XYpos_mep2", "Positions: ME+2", 140, -700., 700., 140, -700., 700.);
    m_XYpos_mep3 = tFileService->make<TH2F>("XYpos_mep3", "Positions: ME+3", 140, -700., 700., 140, -700., 700.);
    m_XYpos_mep4 = tFileService->make<TH2F>("XYpos_mep4", "Positions: ME+4", 140, -700., 700., 140, -700., 700.);
    m_XYpos_mem1 = tFileService->make<TH2F>("XYpos_mem1", "Positions: ME-1", 140, -700., 700., 140, -700., 700.);
    m_XYpos_mem2 = tFileService->make<TH2F>("XYpos_mem2", "Positions: ME-2", 140, -700., 700., 140, -700., 700.);
    m_XYpos_mem3 = tFileService->make<TH2F>("XYpos_mem3", "Positions: ME-3", 140, -700., 700., 140, -700., 700.);
    m_XYpos_mem4 = tFileService->make<TH2F>("XYpos_mem4", "Positions: ME-4", 140, -700., 700., 140, -700., 700.);
    m_RPhipos_mep1 = tFileService->make<TH2F>("RPhipos_mep1", "Positions: ME+1", 144, -M_PI, M_PI, 21, 0., 700.);
    m_RPhipos_mep2 = tFileService->make<TH2F>("RPhipos_mep2", "Positions: ME+2", 144, -M_PI, M_PI, 21, 0., 700.);
    m_RPhipos_mep3 = tFileService->make<TH2F>("RPhipos_mep3", "Positions: ME+3", 144, -M_PI, M_PI, 21, 0., 700.);
    m_RPhipos_mep4 = tFileService->make<TH2F>("RPhipos_mep4", "Positions: ME+4", 144, -M_PI, M_PI, 21, 0., 700.);
    m_RPhipos_mem1 = tFileService->make<TH2F>("RPhipos_mem1", "Positions: ME-1", 144, -M_PI, M_PI, 21, 0., 700.);
    m_RPhipos_mem2 = tFileService->make<TH2F>("RPhipos_mem2", "Positions: ME-2", 144, -M_PI, M_PI, 21, 0., 700.);
    m_RPhipos_mem3 = tFileService->make<TH2F>("RPhipos_mem3", "Positions: ME-3", 144, -M_PI, M_PI, 21, 0., 700.);
    m_RPhipos_mem4 = tFileService->make<TH2F>("RPhipos_mem4", "Positions: ME-4", 144, -M_PI, M_PI, 21, 0., 700.);
  } else {
    m_histP10 = nullptr;
    m_histP100 = nullptr;
    m_histP1000 = nullptr;
    m_hitsPerChamber = nullptr;
    m_fiducial_ME11 = nullptr;
    m_fiducial_ME12 = nullptr;
    m_fiducial_MEx1 = nullptr;
    m_fiducial_MEx2 = nullptr;
    m_slope = nullptr;
    m_slope_MEp4 = nullptr;
    m_slope_MEp3 = nullptr;
    m_slope_MEp2 = nullptr;
    m_slope_MEp1 = nullptr;
    m_slope_MEm1 = nullptr;
    m_slope_MEm2 = nullptr;
    m_slope_MEm3 = nullptr;
    m_slope_MEm4 = nullptr;
    m_slopeResiduals = nullptr;
    m_slopeResiduals_weighted = nullptr;
    m_slopeResiduals_normalized = nullptr;
    m_offsetResiduals = nullptr;
    m_offsetResiduals_weighted = nullptr;
    m_offsetResiduals_normalized = nullptr;
    m_drdz = nullptr;
    m_occupancy = nullptr;
    m_XYpos_mep1 = nullptr;
    m_XYpos_mep2 = nullptr;
    m_XYpos_mep3 = nullptr;
    m_XYpos_mep4 = nullptr;
    m_XYpos_mem1 = nullptr;
    m_XYpos_mem2 = nullptr;
    m_XYpos_mem3 = nullptr;
    m_XYpos_mem4 = nullptr;
    m_RPhipos_mep1 = nullptr;
    m_RPhipos_mep2 = nullptr;
    m_RPhipos_mep3 = nullptr;
    m_RPhipos_mep4 = nullptr;
    m_RPhipos_mem1 = nullptr;
    m_RPhipos_mem2 = nullptr;
    m_RPhipos_mem3 = nullptr;
    m_RPhipos_mem4 = nullptr;
  }
}

CSCOverlapsAlignmentAlgorithm::~CSCOverlapsAlignmentAlgorithm() {}

void CSCOverlapsAlignmentAlgorithm::initialize(const edm::EventSetup& iSetup,
                                               AlignableTracker* alignableTracker,
                                               AlignableMuon* alignableMuon,
                                               AlignableExtras* alignableExtras,
                                               AlignmentParameterStore* alignmentParameterStore) {
  m_alignmentParameterStore = alignmentParameterStore;
  m_alignables = m_alignmentParameterStore->alignables();

  if (alignableTracker == nullptr)
    m_alignableNavigator = new AlignableNavigator(alignableMuon);
  else
    m_alignableNavigator = new AlignableNavigator(alignableTracker, alignableMuon);

  for (const auto& alignable : m_alignables) {
    DetId id = alignable->geomDetId();
    if (id.det() != DetId::Muon || id.subdetId() != MuonSubdetId::CSC || CSCDetId(id.rawId()).layer() != 0) {
      throw cms::Exception("BadConfig") << "Only CSC chambers may be alignable" << std::endl;
    }

    std::vector<bool> selector = alignable->alignmentParameters()->selector();
    for (std::vector<bool>::const_iterator i = selector.begin(); i != selector.end(); ++i) {
      if (!(*i))
        throw cms::Exception("BadConfig") << "All selector strings should be \"111111\"" << std::endl;
    }
  }

  const CSCGeometry* cscGeometry = &iSetup.getData(m_cscGeometryToken);

  for (std::vector<CSCPairResidualsConstraint*>::const_iterator residualsConstraint = m_residualsConstraints.begin();
       residualsConstraint != m_residualsConstraints.end();
       ++residualsConstraint) {
    (*residualsConstraint)->setZplane(cscGeometry);
  }

  if (!m_readTemporaryFiles.empty()) {
    std::vector<std::ifstream*> input;
    for (std::vector<std::string>::const_iterator fileName = m_readTemporaryFiles.begin();
         fileName != m_readTemporaryFiles.end();
         ++fileName) {
      input.push_back(new std::ifstream(fileName->c_str()));
    }

    for (std::vector<CSCPairResidualsConstraint*>::const_iterator residualsConstraint = m_residualsConstraints.begin();
         residualsConstraint != m_residualsConstraints.end();
         ++residualsConstraint) {
      (*residualsConstraint)->read(input, m_readTemporaryFiles);
    }

    for (std::vector<std::ifstream*>::const_iterator file = input.begin(); file != input.end(); ++file) {
      delete (*file);
    }
  }
}

void CSCOverlapsAlignmentAlgorithm::run(const edm::EventSetup& iSetup, const EventInfo& eventInfo) {
  edm::ESHandle<Propagator> propagator;
  if (m_slopeFromTrackRefit) {
    iSetup.getHandle(m_propToken);
    if (m_propagatorPointer != &*propagator) {
      m_propagatorPointer = &*propagator;

      for (std::vector<CSCPairResidualsConstraint*>::const_iterator residualsConstraint =
               m_residualsConstraints.begin();
           residualsConstraint != m_residualsConstraints.end();
           ++residualsConstraint) {
        (*residualsConstraint)->setPropagator(m_propagatorPointer);
      }
    }
  }

  const TransientTrackBuilder* transientTrackBuilder = &iSetup.getData(m_tthbToken);

  if (m_trackTransformer != nullptr)
    m_trackTransformer->setServices(iSetup);

  const ConstTrajTrackPairCollection& trajtracks = eventInfo.trajTrackPairs();
  for (ConstTrajTrackPairCollection::const_iterator trajtrack = trajtracks.begin(); trajtrack != trajtracks.end();
       ++trajtrack) {
    const Trajectory* traj = (*trajtrack).first;
    const reco::Track* track = (*trajtrack).second;

    if (m_makeHistograms) {
      m_histP10->Fill(track->p());
      m_histP100->Fill(track->p());
      m_histP1000->Fill(track->p());
    }
    if (track->p() >= m_minP) {
      std::vector<TrajectoryMeasurement> measurements = traj->measurements();
      reco::TransientTrack transientTrack = transientTrackBuilder->build(track);

      std::map<int, std::map<CSCDetId, bool> > stationsToChambers;
      for (std::vector<TrajectoryMeasurement>::const_iterator measurement = measurements.begin();
           measurement != measurements.end();
           ++measurement) {
        DetId id = measurement->recHit()->geographicalId();
        if (id.det() == DetId::Muon && id.subdetId() == MuonSubdetId::CSC) {
          CSCDetId cscid(id.rawId());
          CSCDetId chamberId(cscid.endcap(), cscid.station(), cscid.ring(), cscid.chamber(), 0);
          if (m_combineME11 && cscid.station() == 1 && cscid.ring() == 4)
            chamberId = CSCDetId(cscid.endcap(), 1, 1, cscid.chamber(), 0);
          int station = (cscid.endcap() == 1 ? 1 : -1) * cscid.station();

          if (stationsToChambers.find(station) == stationsToChambers.end())
            stationsToChambers[station] = std::map<CSCDetId, bool>();
          stationsToChambers[station][chamberId] = true;

          if (m_makeHistograms) {
            GlobalPoint pos = measurement->recHit()->globalPosition();
            if (cscid.endcap() == 1 && cscid.station() == 1) {
              m_XYpos_mep1->Fill(pos.x(), pos.y());
              m_RPhipos_mep1->Fill(pos.phi(), pos.perp());
            }
            if (cscid.endcap() == 1 && cscid.station() == 2) {
              m_XYpos_mep2->Fill(pos.x(), pos.y());
              m_RPhipos_mep2->Fill(pos.phi(), pos.perp());
            }
            if (cscid.endcap() == 1 && cscid.station() == 3) {
              m_XYpos_mep3->Fill(pos.x(), pos.y());
              m_RPhipos_mep3->Fill(pos.phi(), pos.perp());
            }
            if (cscid.endcap() == 1 && cscid.station() == 4) {
              m_XYpos_mep4->Fill(pos.x(), pos.y());
              m_RPhipos_mep4->Fill(pos.phi(), pos.perp());
            }
            if (cscid.endcap() == 2 && cscid.station() == 1) {
              m_XYpos_mem1->Fill(pos.x(), pos.y());
              m_RPhipos_mem1->Fill(pos.phi(), pos.perp());
            }
            if (cscid.endcap() == 2 && cscid.station() == 2) {
              m_XYpos_mem2->Fill(pos.x(), pos.y());
              m_RPhipos_mem2->Fill(pos.phi(), pos.perp());
            }
            if (cscid.endcap() == 2 && cscid.station() == 3) {
              m_XYpos_mem3->Fill(pos.x(), pos.y());
              m_RPhipos_mem3->Fill(pos.phi(), pos.perp());
            }
            if (cscid.endcap() == 2 && cscid.station() == 4) {
              m_XYpos_mem4->Fill(pos.x(), pos.y());
              m_RPhipos_mem4->Fill(pos.phi(), pos.perp());
            }
          }
        }
      }

      std::map<CSCPairResidualsConstraint*, bool> residualsConstraints;
      for (std::map<int, std::map<CSCDetId, bool> >::const_iterator iter = stationsToChambers.begin();
           iter != stationsToChambers.end();
           ++iter) {
        for (std::map<CSCDetId, bool>::const_iterator one = iter->second.begin(); one != iter->second.end(); ++one) {
          for (std::map<CSCDetId, bool>::const_iterator two = one; two != iter->second.end(); ++two) {
            if (one != two) {
              std::map<std::pair<CSCDetId, CSCDetId>, CSCPairResidualsConstraint*>::const_iterator quick;

              quick = m_quickChamberLookup.find(std::pair<CSCDetId, CSCDetId>(one->first, two->first));
              if (quick != m_quickChamberLookup.end())
                residualsConstraints[quick->second] = true;

              quick = m_quickChamberLookup.find(std::pair<CSCDetId, CSCDetId>(two->first, one->first));
              if (quick != m_quickChamberLookup.end())
                residualsConstraints[quick->second] = true;
            }
          }
        }
      }

      for (std::map<CSCPairResidualsConstraint*, bool>::const_iterator residualsConstraint =
               residualsConstraints.begin();
           residualsConstraint != residualsConstraints.end();
           ++residualsConstraint) {
        residualsConstraint->first->addTrack(measurements, transientTrack, m_trackTransformer);
      }
    }
  }
}

void CSCOverlapsAlignmentAlgorithm::terminate(const edm::EventSetup& iSetup) {
  // write residuals partial fits to temporary files for collection
  if (m_writeTemporaryFile != std::string("")) {
    std::ofstream output(m_writeTemporaryFile.c_str());
    for (std::vector<CSCPairResidualsConstraint*>::const_iterator residualsConstraint = m_residualsConstraints.begin();
         residualsConstraint != m_residualsConstraints.end();
         ++residualsConstraint) {
      (*residualsConstraint)->write(output);
    }
  }

  // write report for alignment results
  if (m_doAlignment) {
    std::ofstream report;
    bool writeReport = (m_reportFileName != std::string(""));
    if (writeReport) {
      report.open(m_reportFileName.c_str());
      report << "cscReports = []" << std::endl
             << std::endl
             << "class CSCChamberCorrection:" << std::endl
             << "    def __init__(self, name, detid, value):" << std::endl
             << "        self.name, self.detid, self.value = name, detid, value" << std::endl
             << std::endl
             << "class CSCErrorMode:" << std::endl
             << "    def __init__(self, error):" << std::endl
             << "        self.error = error" << std::endl
             << "        self.terms = {}" << std::endl
             << "        self.detids = {}" << std::endl
             << "    def addTerm(self, name, detid, coefficient):" << std::endl
             << "        self.terms[name] = coefficient" << std::endl
             << "        self.detids[name] = detid" << std::endl
             << std::endl
             << "class CSCConstraintResidual:" << std::endl
             << "    def __init__(self, i, j, before, uncert, residual, pull):" << std::endl
             << "        self.i, self.j, self.before, self.error, self.residual, self.pull = i, j, before, uncert, "
                "residual, pull"
             << std::endl
             << std::endl
             << "class CSCFitterReport:" << std::endl
             << "    def __init__(self, name, oldchi2, newchi2):" << std::endl
             << "        self.name, self.oldchi2, self.newchi2 = name, oldchi2, newchi2" << std::endl
             << "        self.chamberCorrections = []" << std::endl
             << "        self.errorModes = []" << std::endl
             << "        self.constraintResiduals = []" << std::endl
             << std::endl
             << "    def addChamberCorrection(self, name, detid, value):" << std::endl
             << "        self.chamberCorrections.append(CSCChamberCorrection(name, detid, value))" << std::endl
             << std::endl
             << "    def addErrorMode(self, error):" << std::endl
             << "        self.errorModes.append(CSCErrorMode(error))" << std::endl
             << std::endl
             << "    def addErrorModeTerm(self, name, detid, coefficient):" << std::endl
             << "        self.errorModes[-1].addTerm(name, detid, coefficient)" << std::endl
             << std::endl
             << "    def addCSCConstraintResidual(self, i, j, before, uncert, residual, pull):" << std::endl
             << "        self.constraintResiduals.append(CSCConstraintResidual(i, j, before, uncert, residual, pull))"
             << std::endl
             << std::endl
             << "import re" << std::endl
             << "def nameToKey(name):" << std::endl
             << "    match = re.match(\"ME([\\+\\-])([1-4])/([1-4])/([0-9]{2})\", name)" << std::endl
             << "    if match is None: return None" << std::endl
             << "    endcap, station, ring, chamber = match.groups()" << std::endl
             << "    if endcap == \"+\": endcap = 1" << std::endl
             << "    else: endcap = 2" << std::endl
             << "    station = int(station)" << std::endl
             << "    ring = int(ring)" << std::endl
             << "    chamber = int(chamber)" << std::endl
             << "    return endcap, station, ring, chamber" << std::endl
             << std::endl;
    }

    for (std::vector<CSCChamberFitter>::const_iterator fitter = m_fitters.begin(); fitter != m_fitters.end();
         ++fitter) {
      if (m_mode == CSCPairResidualsConstraint::kModeRadius) {
        fitter->radiusCorrection(m_alignableNavigator, m_alignmentParameterStore, m_combineME11);

      } else {
        std::vector<CSCAlignmentCorrections*> corrections;
        fitter->fit(corrections);

        // corrections only exist if the fit was successful
        for (std::vector<CSCAlignmentCorrections*>::iterator correction = corrections.begin();
             correction != corrections.end();
             ++correction) {
          (*correction)->applyAlignment(m_alignableNavigator, m_alignmentParameterStore, m_mode, m_combineME11);
          if (m_makeHistograms)
            (*correction)->plot();
          if (writeReport)
            (*correction)->report(report);
        }
      }
    }
  }
}

#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentAlgorithmPluginFactory.h"
DEFINE_EDM_PLUGIN(AlignmentAlgorithmPluginFactory, CSCOverlapsAlignmentAlgorithm, "CSCOverlapsAlignmentAlgorithm");
