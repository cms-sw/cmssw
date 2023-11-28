autoDQM = { 'DQMMessageLogger': ['DQMMessageLoggerSeq',
                              'PostDQMOffline',
                              'DQMMessageLoggerClientSeq'],

           'commonReduced': ['@dcs+@DQMMessageLogger+@hlt+@beam+@castor+@physics',
                        'PostDQMOffline',
                        '@dcs+@DQMMessageLogger+@hlt+@beam+@fed+dqmFastTimerServiceClient'],

	   'common': ['@dcs+@DQMMessageLogger+@stripCommon+@pixel+@tracking+@hlt+@beam+@castor+@physics',
                        'PostDQMOffline',
                        '@dcs+@DQMMessageLogger+@stripCommon+@pixel+@tracking+@hlt+@beam+@fed+dqmFastTimerServiceClient'],

            'commonFakeHLT': ['@dcs+@DQMMessageLogger+@stripCommon+@pixel+@tracking+@beam+@castor+@physics',
                        'PostDQMOffline',
                        '@dcs+@DQMMessageLogger+@stripCommon+@pixel+@tracking+@beam+@fed+dqmFastTimerServiceClient'],
                        
            'commonSiStripZeroBias': ['@dcs+@DQMMessageLogger+@stripZeroBias+@pixelOnlyDQM+@trackingZeroBias+@L1TMon+@hlt+@beam+@castor+@physics',
                                      'PostDQMOffline',
                                      '@dcs+@DQMMessageLogger+@stripZeroBias+@pixelOnlyDQM+@trackingZeroBias+@L1TMon+@hlt+@beam+@fed+dqmFastTimerServiceClient'],

            'commonSiStripZeroBiasFakeHLT': ['@dcs+@DQMMessageLogger+@stripZeroBias+@pixelOnlyDQM+@trackingZeroBias+@L1TMon+@beam+@castor+@physics',
                                      'PostDQMOffline',
                                      '@dcs+@DQMMessageLogger+@stripZeroBias+@pixelOnlyDQM+@trackingZeroBias+@L1TMon+@beam+@fed+dqmFastTimerServiceClient'],

            'trackingOnlyDQM': ['DQMOfflineTracking',
                                'PostDQMOffline',
                                'DQMHarvestTracking'],

            'pixelTrackingOnlyDQM': ['DQMOfflinePixelTracking',
                                     'PostDQMOffline',
                                     'DQMHarvestPixelTracking'],

            'outerTracker': ['DQMOuterTracker',
                             'PostDQMOffline',
                             'DQMHarvestOuterTracker'],

            'trackerPhase2': ['DQMOfflineTrackerPhase2',
                              'PostDQMOffline',
                              'DQMHarvestTrackerPhase2'],
	    'dcs': ['DQMOfflineDCS',
		    'PostDQMOffline',
		    'DQMNone'],

	    'strip': ['DQMOfflineTrackerStrip',
		      'PostDQMOffline',
		      'DQMHarvestTrackerStrip+DQMCertTrackerStrip'],

            'stripCommon': ['DQMOfflineTrackerStripCommon',
                      'PostDQMOffline',
                      'DQMHarvestTrackerStrip+DQMCertTrackerStrip'],

	    'stripZeroBias': ['DQMOfflineTrackerStripMinBias',
			      'PostDQMOffline',
                              'DQMHarvestTrackerStrip+DQMCertTrackerStrip'],

	    'pixel': ['DQMOfflineTrackerPixel',
		      'PostDQMOffline',
                      'DQMHarvestTrackerPixel+DQMCertTrackerPixel'],

            'pixelOnlyDQM': ['DQMOfflineTrackerPixel',
                      'PostDQMOffline',
                      'DQMHarvestTrackerPixel'],

	    'castor': ['DQMOfflineCASTOR',
		       'PostDQMOffline',
		       'DQMNone'],

	    'tracking': ['DQMOfflineTracking',
			 'PostDQMOffline',
                         'DQMHarvestTracking+DQMCertTracking'],

            'trackingZeroBias': ['DQMOfflineTrackingMinBias',
                       'PostDQMOffline',
                       'DQMHarvestTrackingZeroBias+DQMCertTracking'],

	    'hlt': ['DQMOfflineTrigger',
		    'PostDQMOffline',
                    'DQMHarvestTrigger+DQMCertTrigger'],

	    'fed': ['DQMNone',
		    'PostDQMOffline',
		    'DQMHarvestFED'],

	    'tau': ['DQMOfflineTAU',
		    'PostDQMOffline',
		    'DQMHarvestTAU'],

	    'beam': ['DQMOfflineBeam',
		     'PostDQMOffline',
		     'DQMHarvestBeam'],

            'lumi': ['DQMOfflineLumi',
                     'PostDQMOffline',
                     'DQMNone'],

            'muon': ['DQMOfflineMuon',
                     'PostDQMOffline',
                     'DQMHarvestMuon+DQMCertMuon'],

            'hcal': ['DQMOfflineHcal',
                     'PostDQMOffline',
                     'DQMHarvestHcal'],

            'hcal2': ['DQMOfflineHcal2',
                      'PostDQMOffline',
                      'DQMHarvestHcal2'],

            'hcalOnly': ['DQMOfflineHcalOnly',
                         'PostDQMOffline',
                         'DQMHarvestHcal'],

            'hcal2Only': ['DQMOfflineHcal2Only',
                          'PostDQMOffline',
                          'DQMHarvestHcal2'],


            'jetmet': ['DQMOfflineJetMET',
                       'PostDQMOffline',
                       'DQMHarvestJetMET+DQMCertJetMET'],

            'ecal': ['DQMOfflineEcal',
                     'PostDQMOffline',
                     'DQMHarvestEcal+DQMCertEcal'],

            'ecalOnly': ['DQMOfflineEcalOnly',
                         'PostDQMOffline',
                         'DQMHarvestEcal+DQMCertEcal'],

            'egamma': ['DQMOfflineEGamma',
                       'PostDQMOffline',
                       'DQMHarvestEGamma+DQMCertEGamma'],

            'ctpps': ['DQMOfflineCTPPS',
                      'PostDQMOffline',
                      'DQMHarvestCTPPS'],

            'btag': ['DQMOfflineBTag',
                     'PostDQMOffline',
                     'DQMHarvestBTag'],

	    'physics': ['DQMOfflinePhysics',
			'PostDQMOffline',
			'DQMNone'],

            'heavyFlavor': ['DQMOfflineHeavyFlavor',
                            'PostDQMOffline',
                            'DQMNone'],

            'L1TMon': ['DQMOfflineL1T',
                       'PostDQMOffline',
                       'DQMHarvestL1T'],

            'L1TEgamma': ['DQMOfflineL1TEgamma',
                          'PostDQMOffline',
                          'DQMHarvestL1TEgamma'],

            'L1TMuon': ['DQMOfflineL1TMuon',
                        'PostDQMOffline',
                        'DQMHarvestL1TMuon'],

            'L1TMonPhase2': ['DQMOfflineL1TPhase2',
                       'PostDQMOffline',
                       'DQMHarvestL1TPhase2'],

            'HLTMon': ['HLTMonitoring',
                       'PostDQMOffline',
                       'HLTMonitoringClient'],

            'HLTMonPA': ['HLTMonitoringPA',
			 'PostDQMOffline',
			 'HLTMonitoringClientPA'],

            'express': ['@commonSiStripZeroBias+@muon+@hcal+@jetmet+@ecal',
                        'PostDQMOffline',
                        '@commonSiStripZeroBias+@muon+@hcal+@jetmet+@ecal'],

            'allForPrompt': ['@common+@muon+@L1TMon+@hcal+@jetmet+@ecal+@egamma',
                             'PostDQMOffline',
                             '@common+@muon+@L1TMon+@hcal+@jetmet+@ecal+@egamma'],

            'rerecoCommon': ['@common+@muon+@L1TMon+@hcal+@hcal2+@jetmet+@ecal+@egamma+@L1TMuon+@L1TEgamma+@ctpps',
                             'PostDQMOffline',
                             '@common+@muon+@L1TMon+@hcal+@hcal2+@jetmet+@ecal+@egamma+@L1TMuon+@L1TEgamma+@ctpps'],

            'rerecoSingleMuon': ['@common+@muon+@hcal+@jetmet+@ecal+@egamma+@lumi+@L1TMuon+@L1TEgamma+@ctpps',
                                 'PostDQMOffline',
                                 '@common+@muon+@hcal+@jetmet+@ecal+@egamma+@lumi+@L1TMuon+@L1TEgamma+@ctpps'],

            'rerecoZeroBias' : ['@commonSiStripZeroBias+@muon+@hcal+@hcal2+@jetmet+@ecal+@egamma+@L1TMuon+@L1TEgamma+@ctpps',
                                'PostDQMOffline',
                                '@commonSiStripZeroBias+@muon+@hcal+@hcal2+@jetmet+@ecal+@egamma+@L1TMuon+@L1TEgamma+@ctpps'],

            'rerecoZeroBiasFakeHLT' : ['@commonSiStripZeroBiasFakeHLT+@muon+@hcal+@hcal2+@jetmet+@ecal+@egamma+@L1TMuon+@L1TEgamma+@ctpps',
                                       'PostDQMOffline',
                                       '@commonSiStripZeroBiasFakeHLT+@muon+@hcal+@hcal2+@jetmet+@ecal+@egamma+@L1TMuon+@L1TEgamma+@ctpps'],

            'miniAODDQM': ['DQMOfflineMiniAOD',
                           'PostDQMOfflineMiniAOD',
                           'DQMHarvestMiniAOD'],

            'nanoAODDQM': ['DQMOfflineNanoAOD',
                           'PostDQMOffline',
                           'DQMHarvestNanoAOD'],

            'nanogenDQM': ['DQMOfflineNanoGen',
                           'PostDQMOffline',
                           'DQMHarvestNanoAOD'],

            'nanojmeDQM': ['DQMOfflineNanoJME',
                             'PostDQMOffline',
                             'DQMHarvestNanoAOD'],

            'pfDQM': ['DQMOfflinePF+DQMOfflinePFExtended',
                      'PostDQMOffline',
                      'DQMHarvestPF'],

#           'standardDQM': ['@dcs+@DQMMessageLogger+@ecal+@hcal+@hcal2+@strip+@pixel+@castor+@ctpps+@muon+@tracking+@jetmet+@egamma+@L1TMon+@hlt+@btag+@beam+@physics+@HLTMon',
            'standardDQM': ['DQMOffline',
                            'PostDQMOffline',
                            'dqmHarvesting'],

            'standardDQMExpress': ['DQMOfflineExpress',
                                   'PostDQMOffline',
                                   'dqmHarvestingExpress'],

            'standardDQMFS': ['DQMOfflineFS',
                            'PostDQMOffline',
                            'dqmHarvesting'],

            'ExtraHLT': ['DQMOfflineExtraHLT',
                         'PostDQMOffline',
                         'dqmHarvestingExtraHLT'],

#           'standardDQMFakeHLT': ['@dcs+@DQMMessageLogger+@ecal+@hcal+@hcal2+@strip+@pixel+@castor+@ctpps+@muon+@tracking+@jetmet+@egamma+@L1TMon+@btag+@beam+@physics',
            'standardDQMFakeHLT': ['DQMOfflineFakeHLT',
                                   'PostDQMOffline',
                                   'dqmHarvestingFakeHLT'],

            'standardDQMHIFakeHLT': ['DQMOfflineHeavyIonsFakeHLT',
                                   'PostDQMOfflineHI',
                                   'dqmHarvestingFakeHLT'],

            'liteDQMHI': ['liteDQMOfflineHeavyIons',
                          'PostDQMOfflineHI',
                          'dqmHarvesting'],

            'none': ['DQMNone',
                     'PostDQMOffline',
                     'DQMNone'],
            }

_phase2_allowed = ['beam','trackingOnlyDQM','outerTracker', 'trackerPhase2', 'muon','hcal','hcal2','egamma','L1TMonPhase2','HLTMon']
autoDQM['phase2'] = ['','','']
for i in [0,2]:
    autoDQM['phase2'][i] = '+'.join([autoDQM[m][i] for m in _phase2_allowed])
autoDQM['phase2'][1] = 'PostDQMOffline'
