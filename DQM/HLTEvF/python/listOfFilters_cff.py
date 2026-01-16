import FWCore.ParameterSet.Config as cms

filters = cms.VPSet(
  cms.PSet(
        name = cms.string('hltQuadJet15'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltTauJet5'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltQuadJet20'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltPFQuadJetLooseID15'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltL1sQuadJetOrHTTOrMuonHTTOrDiTauOrDiTauJet'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltEGL1SingleAndDoubleEGAndDoubleEGEBEBOrFilter'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltPFTauTrack'),
        type = cms.int32(-15),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltPFTauTrack'),
        type = cms.int32(15),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltL1sTauVeryBigOR'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltEGL1SingleAndDoubleEGOrPairFilter'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltEGL1SingleAndDoubleEGNonIsoForDisplacedTrig'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltEGL1SingleAndDoubleEGNonIsoOrWithJetAndTauFilter'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltEGL1SingleAndDoubleEGNonIsoOrWithEG26WithJetAndTauFilter'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltEGL1SingleAndDoubleEGEta1p5OrFilter'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltEGL1SingleAndDoubleEGOrFilter'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltL1sAlCaEcalPi0Eta'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltEG8EtFilter'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltSingleCentralCaloJetpt40'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltDoubleCentralCaloJetpt40'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltHpsPFTauTrackDispl'),
        type = cms.int32(-15),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltHpsPFTauTrackDispl'),
        type = cms.int32(15),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltL1sSingleAndDoubleEGAndDoubleEGEBEBor'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltHtEcal800'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltL1sSingleAndDoubleEGNonIsoForDisplacedTrig'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hlt4PFCentralJetPt25'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltL1sSingleAndDoubleEGWithJetAndTau'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltOverlapFilterPhoton32IsoTau32WPTightGsfCaloJet5'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltL1sSingleAndDoubleEG'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltPFJetForBtagSelector'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltL1sSingleAndDoubleEGNonIsoOrWithJetAndTau'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltL1sSingleAndDoubleEGNonIsoOrWithEG26WithJetAndTau'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltL1sDoubleEGIsoeta1p5'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltTripleJet35'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltEGL1SingleEGNonIsoOrWithJetAndTauFilter'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltL1sSingleAndDoubleEGor'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltEGL1SingleEGOrFilter'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltPFQuadJetLooseID30'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltEGL1EGAndTauOrEGOnlyFilter'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltL1sDoubleTauBigOR'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltSingleCaloJet10'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltDoubleL2Tau26eta2p2'),
        type = cms.int32(15),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltL1sEG40To45IorJet170To200IorHTT300To500IorETM70ToETM150'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hlt4PFCentralJetTightIDPt25'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltEGL1SingleIsoEG28to45Filter'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hlt4PFCentralJetPt20'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltEG10EtFilter'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltEGL1SingleEGNonIsoOrWithJetAndTauNoPSFilter'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltL1sTripleJet957565VBFIorHTTIorDoubleJetCIorSingleJetorQuadJet95756520'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltL1sBigORDoubleTauJet'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hlt4PFCentralJetPt30'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hlt4PFCentralJetTightIDPt30'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltTripleJet50'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hlt6PFCentralJetTightIDPt32'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hlt4PFCentralJetTightIDPt35'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltDiEG5TightIDMWEtUnseededFilter'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hlt6PFCentralJetTightIDPt36'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hlt4PFCentralJetTightIDPt40'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltDoubleJet50'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hlt4PFCentralJetLooseID40'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltL1sSingleEGNonIsoOrWithJetAndTau'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltL1sSingleJetOrDoubleJetOrTripleJetOrHTTOrMuHTT'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltL1sHTT280to500erIorHTT250to340erQuadJetTripleJet'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltL1sBigORLooseIsoEGXXerIsoTauYYerdRMin0p3ORSingleEG36er'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltL1sSingleEGor'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltL1sDoubleEG6to8HTT250to300IorL1sHTT'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltEG12TightIDMWL1SingleAndDoubleEGAndDoubleEGEBEBOrEtFilter'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltEG10HEFilter'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltL1sTripleJetVBFIorHTTIorSingleJet'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltL1sSingleIsoEG28to45'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltL1sDSTRun3JetHTPFScoutingPixelTracking'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltL1fL1sL1WTau3MuSeedsL1Filtered0'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltL1sDoubleMu157IorDoubleMu4p5SQOSdR1p2IorSingleMu25IorSingleMu22erIorTripleMuMassMax9'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltL1fL1sL1DoubleMuonScouting'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltDiEG14p25EtEta2p55UnseededFilter'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltEle23Ele12CaloIdLTrackIdLIsoVLEtLeg2Filter'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltEG10CaloIdLClusterShapeFilter'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltL2TauTightIsoFilterL1TauSeededGlob'),
        type = cms.int32(15),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltL1sSingleEGNonIsoOrWithJetAndTauNoPS'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltEGL1SingleEG40Filter'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltL1sDSTRun3DoubleMuonPFScoutingPixelTracking'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltDiEG5TightIDMWClusterShapeUnseededFilter'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltEG25EtFilter'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltL1sTripleJet1008572VBFIorHTTIorDoubleJetCIorSingleJet'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltHT100Jet30'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltHT100Jet30'),
        type = cms.int32(89),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),    
  cms.PSet(
        name = cms.string('hltL1sQuadJetOrHTTOrMuonHTT'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltL1sIsoEG14erHTT200'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltEG12TightIDMWClusterShapeFilter'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltEG15EBL1SingleAndDoubleEGAndDoubleEGEBEBOrEtFilter'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltEG27L1SingleAndDoubleEGEtFilter'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltL1sQuadJetC50to60IorHTT280to500IorHTT250to340QuadJet'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltL1sHTT280to500erIorHTT250to340erQuadJet'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltL1sAllHTTSeeds'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltHT900Jet30'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltHT650'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltHT430'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltHT390Eta2p0'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltHT390'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltHT360'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltHT350'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltHT350'),
        type = cms.int32(89),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltDiCaloJet20MJJ250'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltL1sTripleJet1058576VBFIorHTTIorDoubleJetCIorSingleJet'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltEG27EtFilter'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltL2DoubleTauTagNNFilter'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltL1sMu5EG23IorMu7EG23IorMu20EG17IorMu23EG10'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltDoubleL2GlobIsoTau30eta2p2'),
        type = cms.int32(15),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltL2fL1sL1DoubleMuonScoutingNoVtx'),
        type = cms.int32(-13),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltL2fL1sL1DoubleMuonScoutingNoVtx'),
        type = cms.int32(13),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltL2fL1sL1DoubleMuonScouting'),
        type = cms.int32(-13),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltL2fL1sL1DoubleMuonScouting'),
        type = cms.int32(13),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltEle28HighEtaSC20EtFilter'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltL1sDoubleMuORTripleMuORDoubleMu0upt'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltL1fL1sDoubleMuORTripleMuORDoubleMuUptL1Filtered0'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltSingleJet80'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltSingleCaloJet10SerialSync'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltDiEG5TightIDMWHEUnseededFilter'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltL1sHTT200erFromObjectMap'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltEG12TightIDMWHEFilter'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltEGL1LooseIsoEGTau'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltEGL1EGAndTauFilter'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltEGL1EGerAndTauFilter'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltDoubleL2GlobIsoTau36eta2p2'),
        type = cms.int32(15),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltEle23Ele12CaloIdLTrackIdLIsoVLEtLeg1Filter'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltEG30L1SingleAndDoubleEGOrEtFilter'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltDoubleJet65'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltPFTripleJetLooseID75'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltEG30L1SingleEGOrEtFilter'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltL1fL1sL1DoubleMuorTripleMuL1Filtered0'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltL1sHTTForBeamSpot'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltL1sDoubleMu0er2p0SQOSdEtaMax1p6orTripleMu21p50'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltL1fForIterL3L1fL1sL1DoubleMuorTripleMuL1Filtered0'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltDoubleMuonL3PreFilteredScoutingVtx'),
        type = cms.int32(-13),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltDoubleMuonL3PreFilteredScoutingVtx'),
        type = cms.int32(13),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltL1sSingleJetOrHTTOrMuHTT'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltEG22EBL1SingleAndDoubleEGOrEtFilter'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltDoubleMuonL3PreFilteredScoutingNoVtx'),
        type = cms.int32(-13),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltDoubleMuonL3PreFilteredScoutingNoVtx'),
        type = cms.int32(13),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltEG30LR9Id50b80eR9IdFilter'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltPFTripleJetLooseID80'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltEG32L1SingleAndDoubleEGEtFilter'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltEG33EtFilter'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltL1sSingleJet180OrHTT'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltL1sSingleEG40to50'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltL1sSingleEG34to50'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltL1sSingleEG34to45'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltEG32L1SingleEGOrEtFilter'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltEG30R9Id50b90eR9IdFilter'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltEG22R9Id50b80eR9IdFilter'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltEle28HighEtaSC20ClusterShapeFilter'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltDiEG5TightIDMWEcalIsoDr0p2to0p4UnseededFilter'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltL1fL1sMu5EG20orMu20EG15L1Filtered5'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltEG24EBL1SingleAndDoubleEGOrEtFilter'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltEGL1SingleEG5OpenFilter'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltEG12TightIDMWEcalIsoDr0p2to0p4Filter'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltEG25HEFilter'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltPFTripleJetLooseID76'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltL1sMu23EG10IorMu20EG17'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltEle27L1DoubleEGWPTightClusterShapeFilter'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltEG24R9Id50b80eR9IdFilter'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltL1fL1sL1DoubleMuL1Filtered0LowMassInclusive'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltL1fL1sL1DoubleMuL1Filtered0BMMG'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltL1fL1sL1DoubleMu0er1p5SQOSdR1p4L1Filtered0'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltL1fL1sDoubleMuL1Filtered0ForLowMassDisplaced'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltHT200Jet30'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltHT200Jet30'),
        type = cms.int32(89),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltPFTripleJetLooseID70'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltL1sDoubleMuForLowMassInclusive'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltL1sDoubleMuForLowMassDisplaced'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltL1sDoubleMuForBsToMMG'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltL1sDoubleMuForBs'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltL1fForIterL3L1fL1sL1DoubleMuL1Filtered0LowMassInclusive'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltL1fForIterL3L1fL1sL1DoubleMuL1Filtered0BMMG'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltL1fForIterL3L1fL1sL1DoubleMu0er1p5SQOSdR1p4L1Filtered0'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltL1fForIterL3L1fL1sDoubleMuL1Filtered0ForLowMassDisplaced'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltL1fL1sSingleMuORUpt15OR20L1Filtered0'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltEG27HEFilter'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltL2fL1sL1DoubleMuorTripleMuL1f0L2PreFiltered0'),
        type = cms.int32(-13),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltL2fL1sL1DoubleMuorTripleMuL1f0L2PreFiltered0'),
        type = cms.int32(13),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltL1sHTTMultiJet'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltL1sBigORLooseIsoEGXXerIsoTauYYerdRMin0p3'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltL1sSingleMuORUpt15OR20'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltEG35L1SingleEGOrEtFilter'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltEGL1IsoEG14erHTT200Filter'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltL2TauTagFilterForQuadJet'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltL1sMu12HTT150er'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltL1fL1sL1DoubleMuL1Filtered0LowPtJpsi'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltL1sDoubleMuForLowPtJpsi'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltL1fForIterL3L1fL1sL1DoubleMuL1Filtered0LowPtJpsi'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltDiEG22EtEta2p55UnseededFilter'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltEG30LHE12R9Id50b80eHEFilter'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltEG25CaloIdLClusterShapeFilter'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltPFJetFilterTwo120er3p0'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltEle28HighEtaSC20HEFilter'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltMu23TrkIsoVVLEle12CaloIdLTrackIdLIsoVLMuonlegL1Filtered0'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltL1fL1sMuORL1Filtered0'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltEG15EBTightIdClusterShapeFilter'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltSingleCaloJet10AK8'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltL1fForIterL3Mu23TrkIsoVVLEle12CaloIdLTrackIdLIsoVLMuonlegL1Filtered0'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltL1fL1sMu22orMu25L1Filtered0'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltL1fL1sMu22orMu20erorMu25L1Filtered0'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltL1fL1sMu22or25L1Filtered0'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltL1fL1sMu22L1Filtered0'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltL1fL1sMu16orMu18erorMu20erL1Filtered0'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltEG30HE10b9eR9Id50b90eHEFilter'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltL2DoubleTauJetTagNNFilter'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltL1sSingleMuOR'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltL1sSingleMu5IorSingleMu14erIorSingleMu16er'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltL1fForIterL3L1fL1sMu22orMu20erorMu25L1Filtered0'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
  cms.PSet(
        name = cms.string('hltL1fForIterL3L1fL1sMu22or25L1Filtered0'),
        type = cms.int32(0),
        ptMin = cms.untracked.double(0),
        ptMax = cms.untracked.double(200)
      ),
)
