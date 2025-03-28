import FWCore.ParameterSet.Config as cms

DisplacedVertexProducer = cms.EDProducer('DisplacedVertexProducer',
  l1TracksInputTag = cms.InputTag("l1tTTTracksFromExtendedTrackletEmulation", "Level1TTTracks"),
  l1TracksGTTInputTag = cms.InputTag("l1tGTTInputProducerExtended","Level1TTTracksExtendedConverted"),
  l1TrackVertexCollectionName = cms.string("dispVertices"),
  mcTruthTrackInputTag = cms.InputTag("TTTrackAssociatorFromPixelDigisExtended", "Level1TTTracks"),
  model = cms.FileInPath("L1Trigger/L1TTrackMatch/data/dispVertTaggerEmulationConiferShifted13p8.json"),
  runEmulation = cms.bool(True),                                  
  cutSet = cms.PSet(
      chi2rzMax = cms.double(3.0), # chi2rz value for all tracks must be less than this
      promptMVAMin = cms.double(0.2), # prompt track quality MVA score for all tracks must be greater than this
      ptMin = cms.double(3.0), # pt value for all tracks must be greater than this [GeV]
      etaMax = cms.double(2.4), # eta value for all tracks must be less than this
      dispD0Min = cms.double(1.0), # d0 value for tracks to be considered for displaced track cuts must be greater than this [cm]
      promptMVADispTrackMin = cms.double(0.5), # prompt track quality MVA score for tracks with |d0|>dispD0Min must be greater than this
      overlapEtaMin = cms.double(1.1), # eta value for tracks to be considered for overlap track cuts must be greater than this
      overlapEtaMax = cms.double(1.7), # eta value for tracks to be considered for overlap track cuts must be less than this
      overlapNStubsMin = cms.int32(4), # number of stubs for tracks with overlapEtaMin<|eta|<overlapEtaMax must be greater than this
      diskEtaMin = cms.double(0.95), # eta value for tracks to be considered for disk track cuts must be greater than this
      diskD0Min = cms.double(0.08), # abs d0 value for tracks with |eta|>diskEtaMin must be greater than this [cm]
      barrelD0Min = cms.double(0.06), # abs d0 value for tracks with |eta|<=diskEtaMin must be greater than this [cm]
      RTMin = cms.double(0.02152), # R_T value for all vertices must be greater than this
      RTMax = cms.double(20.0) # R_T value for all vertices must be less than this
  ) 
)

'''
Features for displaced vertex BDT: ['trkExt_pt_firstTrk', 'trkExt_pt', 'trkExt_eta_firstTrk', 'trkExt_eta', 'trkExt_phi_firstTrk', 'trkExt_phi', 'trkExt_d0_firstTrk', 'trkExt_d0', 'trkExt_z0_firstTrk', 'trkExt_z0', 'trkExt_chi2rz_firstTrk', 'trkExt_chi2rz', 'trkExt_bendchi2_firstTrk', 'trkExt_bendchi2', 'trkExt_MVA_firstTrk', 'trkExt_MVA', 'dv_d_T', 'dv_R_T', 'dv_cos_T', 'dv_del_Z'])

dv inputs are vertex quantities and trkExt is a displaced track property. The firstTrk suffix means the track quantity comes from the higher pt track associated to a vertex. If there's no firstTrk suffix, then the track property is from the lower pt track associated to a vertex.

Note: TrackQuality parameter in L1Trigger/TrackFindingTracklet/python/l1tTTTracksFromTrackletEmulation_cfi.py needs to be set to True to get MVA values needed for BDT

'''
