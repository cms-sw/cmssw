autoDQM = {'DQMMessageLogger': ['DQMMessageLoggerSeq',
                                'PostDQMOffline',
                                'DQMMessageLoggerClientSeq'],

           'ExtraHLT': ['DQMOfflineExtraHLT', 'PostDQMOffline', 'dqmHarvestingExtraHLT'],

           'HLTMon': ['HLTMonitoring', 'PostDQMOffline', 'HLTMonitoringClient'],

           'HLTMonPA': ['HLTMonitoringPA', 'PostDQMOffline', 'HLTMonitoringClientPA'],

           'L1TEgamma': ['DQMOfflineL1TEgamma', 'PostDQMOffline', 'DQMHarvestL1TEgamma'],

           'L1TMon': ['DQMOfflineL1T', 'PostDQMOffline', 'DQMHarvestL1T'],

           'L1TMonPhase2': ['DQMOfflineL1TPhase2',
                            'PostDQMOffline',
                            'DQMHarvestL1TPhase2'],

           'L1TMuon': ['DQMOfflineL1TMuon', 'PostDQMOffline', 'DQMHarvestL1TMuon'],

           'allForPrompt': ['@common+@muon+@L1TMon+@hcal+@jetmet+@ecal+@egamma',
                            'PostDQMOffline',
                            '@common+@muon+@L1TMon+@hcal+@jetmet+@ecal+@egamma'],

           'beam': ['DQMOfflineBeam', 'PostDQMOffline', 'DQMHarvestBeam'],

           'btag': ['DQMOfflineBTag', 'PostDQMOffline', 'DQMHarvestBTag'],

           'castor': ['DQMOfflineCASTOR', 'PostDQMOffline', 'DQMNone'],

           'common': ['@dcs+@DQMMessageLogger+@stripCommon+@pixel+@tracking+@hlt+@beam+@castor+@physics',
                      'PostDQMOffline',
                      '@dcs+@DQMMessageLogger+@stripCommon+@pixel+@tracking+@hlt+@beam+@fed+dqmFastTimerServiceClient'],

           'commonFakeHLT': ['@dcs+@DQMMessageLogger+@stripCommon+@pixel+@tracking+@beam+@castor+@physics',
                             'PostDQMOffline',
                             '@dcs+@DQMMessageLogger+@stripCommon+@pixel+@tracking+@beam+@fed+dqmFastTimerServiceClient'],

           'commonReduced': ['@dcs+@DQMMessageLogger+@hlt+@beam+@castor+@physics',
                             'PostDQMOffline',
                             '@dcs+@DQMMessageLogger+@hlt+@beam+@fed+dqmFastTimerServiceClient'],

           'commonSiStripZeroBias': ['@dcs+@DQMMessageLogger+@stripZeroBias+@pixelOnlyDQM+@trackingZeroBias+@L1TMon+@hlt+@beam+@castor+@physics',
                                     'PostDQMOffline',
                                     '@dcs+@DQMMessageLogger+@stripZeroBias+@pixelOnlyDQM+@trackingZeroBias+@L1TMon+@hlt+@beam+@fed+dqmFastTimerServiceClient'],

           'commonSiStripZeroBiasFakeHLT': ['@dcs+@DQMMessageLogger+@stripZeroBias+@pixelOnlyDQM+@trackingZeroBias+@L1TMon+@beam+@castor+@physics',
                                            'PostDQMOffline',
                                            '@dcs+@DQMMessageLogger+@stripZeroBias+@pixelOnlyDQM+@trackingZeroBias+@L1TMon+@beam+@fed+dqmFastTimerServiceClient'],

           'commonWithScouting': ['@dcs+@DQMMessageLogger+@stripCommon+@pixel+@tracking+@hlt+@beam+@castor+@physics+@hltScouting',
                                  'PostDQMOffline',
                                  '@dcs+@DQMMessageLogger+@stripCommon+@pixel+@tracking+@hlt+@beam+@fed+dqmFastTimerServiceClient'],

           'cosmics': ['DQMOfflineCosmics', 'PostDQMOffline', 'DQMOfflineCosmics'],

           'ctpps': ['DQMOfflineCTPPS', 'PostDQMOffline', 'DQMHarvestCTPPS'],

           'dcs': ['DQMOfflineDCS', 'PostDQMOffline', 'DQMNone'],

           'ecal': ['DQMOfflineEcal', 'PostDQMOffline', 'DQMHarvestEcal+DQMCertEcal'],

           'ecalOnly': ['DQMOfflineEcalOnly',
                        'PostDQMOffline',
                        'DQMHarvestEcal+DQMCertEcal'],

           'egamma': ['DQMOfflineEGamma',
                      'PostDQMOffline',
                      'DQMHarvestEGamma+DQMCertEGamma'],

           'express': ['@commonSiStripZeroBias+@muon+@hcal+@jetmet+@ecal',
                       'PostDQMOffline',
                       '@commonSiStripZeroBias+@muon+@hcal+@jetmet+@ecal'],

           'fed': ['DQMNone', 'PostDQMOffline', 'DQMHarvestFED'],

           'hcal': ['DQMOfflineHcal', 'PostDQMOffline', 'DQMHarvestHcal'],

           'hcal2': ['DQMOfflineHcal2', 'PostDQMOffline', 'DQMHarvestHcal2'],

           'hcal2Only': ['DQMOfflineHcal2Only', 'PostDQMOffline', 'DQMHarvestHcal2'],

           'hcalOnly': ['DQMOfflineHcalOnly', 'PostDQMOffline', 'DQMHarvestHcal'],

           'heavyFlavor': ['DQMOfflineHeavyFlavor', 'PostDQMOffline', 'DQMNone'],

           'hlt': ['DQMOfflineTrigger',
                   'PostDQMOffline',
                   'DQMHarvestTrigger+DQMCertTrigger'],

           'hltGPUvsCPU': ['DQMOfflineHLTGPUvsCPU',
                           'PostDQMOffline',
                           'DQMHarvestHLTGPUvsCPU'],

           'hltScouting': ['DQMOfflineScouting',
                           'PostDQMOffline',
                           'DQMHarvestHLTScouting'],

           'jetmet': ['DQMOfflineJetMET',
                      'PostDQMOffline',
                      'DQMHarvestJetMET+DQMCertJetMET'],

           'liteDQMHI': ['liteDQMOfflineHeavyIons', 'PostDQMOfflineHI', 'dqmHarvesting'],

           'lumi': ['DQMOfflineLumi', 'PostDQMOffline', 'DQMNone'],

           'miniAODDQM': ['DQMOfflineMiniAOD',
                          'PostDQMOfflineMiniAOD',
                          'DQMHarvestMiniAOD'],

           'miniAODDQMBTagOnly': ['DQMOfflineMiniAODBTagOnly',
                                  'PostDQMOfflineMiniAOD',
                                  'DQMHarvestMiniAODBTagOnly'],

           'muon': ['DQMOfflineMuon', 'PostDQMOffline', 'DQMHarvestMuon+DQMCertMuon'],

           'nanoAODDQM': ['DQMOfflineNanoAOD', 'PostDQMOffline', 'DQMHarvestNanoAOD'],

           'nanogenDQM': ['DQMOfflineNanoGen', 'PostDQMOffline', 'DQMHarvestNanoAOD'],

           'nanohltDQM': ['DQMOfflineNanoHLT', 'PostDQMOffline', 'DQMHarvestNanoAOD'],

           'nanojmeDQM': ['DQMOfflineNanoJME', 'PostDQMOffline', 'DQMHarvestNanoAOD'],

           'none': ['DQMNone', 'PostDQMOffline', 'DQMNone'],

           'outerTracker': ['DQMOuterTracker',
                            'PostDQMOffline',
                            'DQMHarvestOuterTracker'],

           'pfDQM': ['DQMOfflinePF+DQMOfflinePFExtended',
                     'PostDQMOffline',
                     'DQMHarvestPF'],

           'physics': ['DQMOfflinePhysics', 'PostDQMOffline', 'DQMNone'],

           'pixel': ['DQMOfflineTrackerPixel',
                     'PostDQMOffline',
                     'DQMHarvestTrackerPixel+DQMCertTrackerPixel'],

           'pixelOnlyDQM': ['DQMOfflineTrackerPixel',
                            'PostDQMOffline',
                            'DQMHarvestTrackerPixel'],

           'pixelTrackingOnlyDQM': ['DQMOfflinePixelTracking',
                                    'PostDQMOffline',
                                    'DQMHarvestPixelTracking'],

           'rerecoCommon': ['@common+@muon+@L1TMon+@hcal+@hcal2+@jetmet+@ecal+@egamma+@L1TMuon+@L1TEgamma+@ctpps',
                            'PostDQMOffline',
                            '@common+@muon+@L1TMon+@hcal+@hcal2+@jetmet+@ecal+@egamma+@L1TMuon+@L1TEgamma+@ctpps'],

           'rerecoSingleMuon': ['@common+@muon+@hcal+@jetmet+@ecal+@egamma+@lumi+@L1TMuon+@L1TEgamma+@ctpps',
                                'PostDQMOffline',
                                '@common+@muon+@hcal+@jetmet+@ecal+@egamma+@lumi+@L1TMuon+@L1TEgamma+@ctpps'],

           'rerecoZeroBias': ['@commonSiStripZeroBias+@muon+@hcal+@hcal2+@jetmet+@ecal+@egamma+@L1TMuon+@L1TEgamma+@ctpps',
                              'PostDQMOffline',
                              '@commonSiStripZeroBias+@muon+@hcal+@hcal2+@jetmet+@ecal+@egamma+@L1TMuon+@L1TEgamma+@ctpps'],

           'rerecoZeroBiasFakeHLT': ['@commonSiStripZeroBiasFakeHLT+@muon+@hcal+@hcal2+@jetmet+@ecal+@egamma+@L1TMuon+@L1TEgamma+@ctpps',
                                     'PostDQMOffline',
                                     '@commonSiStripZeroBiasFakeHLT+@muon+@hcal+@hcal2+@jetmet+@ecal+@egamma+@L1TMuon+@L1TEgamma+@ctpps'],

           # 'standardDQM': ['@dcs+@DQMMessageLogger+@ecal+@hcal+@hcal2+@strip+@pixel+@castor+@ctpps+@muon+@tracking+@jetmet+@egamma+@L1TMon+@hlt+@btag+@beam+@physics+@HLTMon',
           'standardDQM': ['DQMOffline', 'PostDQMOffline', 'dqmHarvesting'],

           'standardDQMExpress': ['DQMOfflineExpress',
                                  'PostDQMOffline',
                                  'dqmHarvestingExpress'],

           'standardDQMFS': ['DQMOfflineFS', 'PostDQMOffline', 'dqmHarvesting'],

           # standardDQMFakeHLT': ['@dcs+@DQMMessageLogger+@ecal+@hcal+@hcal2+@strip+@pixel+@castor+@ctpps+@muon+@tracking+@jetmet+@egamma+@L1TMon+@btag+@beam+@physics',
           'standardDQMFakeHLT': ['DQMOfflineFakeHLT',
                                  'PostDQMOffline',
                                  'dqmHarvestingFakeHLT'],

           'standardDQMHIFakeHLT': ['DQMOfflineHeavyIonsFakeHLT',
                                    'PostDQMOfflineHI',
                                    'dqmHarvestingFakeHLT'],

           'strip': ['DQMOfflineTrackerStrip',
                     'PostDQMOffline',
                     'DQMHarvestTrackerStrip+DQMCertTrackerStrip'],

           'stripCommon': ['DQMOfflineTrackerStripCommon',
                           'PostDQMOffline',
                           'DQMHarvestTrackerStrip+DQMCertTrackerStrip'],

           'stripZeroBias': ['DQMOfflineTrackerStripMinBias',
                             'PostDQMOffline',
                             'DQMHarvestTrackerStrip+DQMCertTrackerStrip'],

           'tau': ['DQMOfflineTAU', 'PostDQMOffline', 'DQMHarvestTAU'],

           'trackerPhase2': ['DQMOfflineTrackerPhase2',
                             'PostDQMOffline',
                             'DQMHarvestTrackerPhase2'],

           'tracking': ['DQMOfflineTracking',
                        'PostDQMOffline',
                        'DQMHarvestTracking+DQMCertTracking'],

           'trackingOnlyDQM': ['DQMOfflineTracking',
                               'PostDQMOffline',
                               'DQMHarvestTracking'],

           'trackingZeroBias': ['DQMOfflineTrackingMinBias',
                                'PostDQMOffline',
                                'DQMHarvestTrackingZeroBias+DQMCertTracking']}

_phase2_allowed = [
    'beam',
    'trackingOnlyDQM',
    'outerTracker',
    'trackerPhase2',
    'muon',
    'hcal',
    'hcal2',
    'egamma',
    'L1TMonPhase2',
    'HLTMon']
autoDQM['phase2'] = ['', '', '']
for i in [0, 2]:
    autoDQM['phase2'][i] = '+'.join([autoDQM[m][i] for m in _phase2_allowed])
autoDQM['phase2'][1] = 'PostDQMOffline'

# Creating autoDQM['phase2FakeHLT'] excluding elements containing 'HLTMon'
autoDQM['phase2FakeHLT'] = []
for val in autoDQM['phase2']:
    if any('HLTMon' in s for s in val.split('+')):
        filtered_val = '+'.join(filter(lambda x: 'HLTMon' not in x,
                                val.split('+')))
        autoDQM['phase2FakeHLT'].append(filtered_val)
    else:
        autoDQM['phase2FakeHLT'].append(val)
