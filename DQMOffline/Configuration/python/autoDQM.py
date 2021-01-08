autoDQM = { 'common': ['DQMOfflineCommon+@L1TMon',
                        'PostDQMOffline',
                        'DQMHarvestCommon+DQMCertCommon+@L1TMon'],

            'commonSiStripZeroBias': ['DQMOfflineCommonSiStripZeroBias',
                                      'PostDQMOffline',
                                      'DQMHarvestCommonSiStripZeroBias+DQMCertCommon'],

            'trackingOnlyDQM': ['DQMOfflineTracking',
                                'PostDQMOffline',
                                'DQMHarvestTracking'],

            'pixelTrackingOnlyDQM': ['DQMOfflinePixelTracking',
                                     'PostDQMOffline',
                                     'DQMHarvestPixelTracking'],

            'outerTracker': ['DQMOuterTracker',
                             'PostDQMOffline',
                             'DQMHarvestOuterTracker'],

            'lumi': ['DQMOfflineLumi',
                     'PostDQMOffline',
                     'DQMHarvestLumi'],

            'muon': ['DQMOfflineMuon',
                     'PostDQMOffline',
                     'DQMHarvestMuon+DQMCertMuon'],

            'hcal': ['DQMOfflineHcal',
                     'PostDQMOffline',
                     'DQMHarvestHcal'],

            'hcal2': ['HcalDQMOfflineSequence',
                      'PostDQMOffline',
                      'HcalDQMOfflinePostProcessor'],

            'jetmet': ['DQMOfflineJetMET',
                       'PostDQMOffline',
                       'DQMHarvestJetMET+DQMCertJetMET'],

            'ecal': ['DQMOfflineEcal',
                     'PostDQMOffline',
                     'DQMHarvestEcal+DQMCertEcal'],

            'egamma': ['DQMOfflineEGamma',
                       'PostDQMOffline',
                       'DQMHarvestEGamma'],

            'ctpps': ['DQMOfflineCTPPS',
                      'PostDQMOffline',
                      'DQMHarvestCTPPS'],
            
            'btag': ['DQMOfflineBTag',
                     'PostDQMOffline',
                     'DQMHarvestBTag'],

            'L1TMon': ['DQMOfflineL1TMonitoring',
                       'PostDQMOffline',
                       'DQMHarvestL1TMonitoring'],

            'L1TEgamma': ['DQMOfflineL1TEgamma',
                          'PostDQMOffline',
                          'DQMHarvestL1TEgamma'],

            'L1TMuon': ['DQMOfflineL1TMuon',
                        'PostDQMOffline',
                        'DQMHarvestL1TMuon'],

            'HLTMon': ['HLTMonitoring',
                       'PostDQMOffline',
                       'HLTMonitoringClient'],

            'HLTMonPA': ['HLTMonitoringPA', 'PostDQMOffline', 'HLTMonitoringClientPA'],

            'express': ['@commonSiStripZeroBias+@muon+@hcal+@jetmet+@ecal',
                        'PostDQMOffline',
                        '@commonSiStripZeroBias+@muon+@hcal+@jetmet+@ecal'],

            'allForPrompt': ['@common+@muon+@hcal+@jetmet+@ecal+@egamma',
                             'PostDQMOffline',
                             '@common+@muon+@hcal+@jetmet+@ecal+@egamma'],

            'rerecoCommon': ['@common+@muon+@hcal+@jetmet+@ecal+@egamma+@L1TMuon+@L1TEgamma+@ctpps',
                             'PostDQMOffline',
                             '@common+@muon+@hcal+@jetmet+@ecal+@egamma+@L1TMuon+@L1TEgamma+@ctpps'],

            'rerecoSingleMuon': ['@common+@muon+@hcal+@jetmet+@ecal+@egamma+@lumi+@L1TMuon+@L1TEgamma+@ctpps',
                                 'PostDQMOffline',
                                 '@common+@muon+@hcal+@jetmet+@ecal+@egamma+@lumi+@L1TMuon+@L1TEgamma+@ctpps'],

            'rerecoZeroBias' : ['@commonSiStripZeroBias+@muon+@hcal+@jetmet+@ecal+@egamma+@L1TMuon+@L1TEgamma+@ctpps',
                                'PostDQMOffline',
                                '@commonSiStripZeroBias+@muon+@hcal+@jetmet+@ecal+@egamma+@L1TMuon+@L1TEgamma+@ctpps'],

            'miniAODDQM': ['DQMOfflineMiniAOD',
                           'PostDQMOfflineMiniAOD',
                           'DQMHarvestMiniAOD'],

            'nanoAODDQM': ['DQMOfflineNanoAOD',
                           'PostDQMOffline',
                           'DQMHarvestNanoAOD'],

            'nanogenDQM': ['DQMOfflineNanoGen',
                           'PostDQMOffline',
                           'DQMHarvestNanoAOD'],

            'standardDQM': ['DQMOffline',
                            'PostDQMOffline',
                            'dqmHarvesting'],

            'ExtraHLT': ['DQMOfflineExtraHLT',
                         'PostDQMOffline',
                         'dqmHarvestingExtraHLT'],

            'standardDQMFakeHLT': ['DQMOfflineFakeHLT',
                                   'PostDQMOffline',
                                   'dqmHarvestingFakeHLT'],

            'liteDQMHI': ['liteDQMOfflineHeavyIons',
                          'PostDQMOffline',
                          'dqmHarvesting'],

            'none': ['DQMNone',
                     'PostDQMOffline',
                     'DQMNone'],
            }

_phase2_allowed = ['trackingOnlyDQM','outerTracker','muon','hcal','hcal2','egamma']
autoDQM['phase2'] = ['','','']
for i in [0,2]:
    autoDQM['phase2'][i] = '+'.join([autoDQM[m][i] for m in _phase2_allowed])
autoDQM['phase2'][1] = 'PostDQMOffline'

