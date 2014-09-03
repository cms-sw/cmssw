def shiftegammalayout(i, p, *rows): i["00 Shift/Egamma/" + p] = DQMItem(layout=rows)

shiftegammalayout(dqmitems, "1-Good Photon Candidates", [{ 'path': "Egamma/PhotonAnalyzer/GoodCandidatePhotons/Et above 0 GeV/nPhoAllEcal", 'description': "Number of good candidate photons per event - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftOfflineEgamma>Shift Instructions</a>" }])

#shiftegammalayout(dqmitems, "2-Good Photon Candidates: Occupancy", [{ 'path': "Egamma/PhotonAnalyzer/GoodCandidatePhotons/Et above 0 GeV/DistributionAllEcal", 'description': "Distribution of good candidate photons over the entire ECAL - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftOfflineEgamma>Shift Instructions</a>" },
 #                                                               { 'path': "Egamma/PhotonAnalyzer/GoodCandidatePhotons/Et above 0 GeV/DistributionBarrel", 'description': "Distribution of good candidate photons in ECAL barrel  - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftOfflineEgamma>Shift Instructions</a>" }],
  #                [{ 'path': "Egamma/PhotonAnalyzer/GoodCandidatePhotons/Et above 0 GeV/DistributionEndcapMinus", 'description': " Distribution of good candidate photons in ECAL minus endcap- <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftOfflineEgamma>Shift Instructions</a>" },
   #                                                              { 'path': "Egamma/PhotonAnalyzer/GoodCandidatePhotons/Et above 0 GeV/DistributionEndcapPlus", 'description': "Distribution of good candidate photons in ECAL plus endcap - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftOfflineEgamma>Shift Instructions</a>" }])

shiftegammalayout(dqmitems, "3-Good Photon Candidates: Et Spectra", [{ 'path': "Egamma/PhotonAnalyzer/GoodCandidatePhotons/Et above 0 GeV/phoEtAllEcal", 'description': "Transverse energy of good candidate photons over the entire ECAL - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftOfflineEgamma>Shift Instructions</a>" }],
[{ 'path': "Egamma/PhotonAnalyzer/GoodCandidatePhotons/Et above 0 GeV/phoEtBarrel", 'description': "Transverse energy of good candidate photons in ECAL barrel - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftOfflineEgamma>Shift Instructions</a>" },
                                                                 { 'path': "Egamma/PhotonAnalyzer/GoodCandidatePhotons/Et above 0 GeV/phoEtEndcaps", 'description': "Transverse energy of good candidate photons in ECAL endcaps - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftOfflineEgamma>Shift Instructions</a>" }])

shiftegammalayout(dqmitems, "4-Good Photon Candidates", [{ 'path': "Egamma/PhotonAnalyzer/GoodCandidatePhotons/Et above 0 GeV/r9VsEt", 'description': "R9 parameter versus transverse energy for good candidate photons - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftOfflineEgamma>Shift Instructions</a>" }])

shiftegammalayout(dqmitems, "5-Good Photon Candidates", [{ 'path': "Egamma/PhotonAnalyzer/GoodCandidatePhotons/Et above 0 GeV/r9AllEcal", 'description': "R9 parameter for good candidate photons - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftOfflineEgamma>Shift Instructions</a>" }])

shiftegammalayout(dqmitems, "6-Efficiencies", [{ 'path': "Egamma/PhotonAnalyzer/Efficiencies/EfficiencyVsEtHLT", 'description': "Number of photons per event - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftOfflineEgamma>Shift Instructions</a>" }],
                  [{ 'path': "Egamma/PhotonAnalyzer/Efficiencies/EfficiencyVsEtaHLT", 'description': "Number of photons per event - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftOfflineEgamma>Shift Instructions</a>" }])

shiftegammalayout(dqmitems, "7-All Photons", [{ 'path': "Egamma/PhotonAnalyzer/AllPhotons/Et above 0 GeV/nPhoAllEcal", 'description': "Number of total photons per event - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftOfflineEgamma>Shift Instructions</a>" }])

shiftegammalayout(dqmitems, "8-Background Photons", [{ 'path': "Egamma/PhotonAnalyzer/BackgroundPhotons/Et above 0 GeV/nPhoAllEcal", 'description': "Number of background photons per event - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftOfflineEgamma>Shift Instructions</a>" }])

shiftegammalayout(dqmitems, "9-Converted Photons", [{ 'path': "Egamma/PhotonAnalyzer/AllPhotons/Et above 0 GeV/Conversions/phoConvEta", 'description': "Eta distribution of conversions - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftOfflineEgamma>Shift Instructions</a>" }])


shiftegammalayout(dqmitems, "10-PiZeros", [{ 'path': "Egamma/PiZeroAnalyzer/Pi0InvmassEB", 'description': "Reconstructed mass of the PiZero particle - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftOfflineEgamma>Shift Instructions</a>" }])

