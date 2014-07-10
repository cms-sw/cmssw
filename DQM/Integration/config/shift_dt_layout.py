def shiftdtlayout(i, p, *rows): i["00 Shift/DT/" + p] = DQMItem(layout=rows)

# shiftdtlayout(dqmitems, "00-DataIntegritySummary",
#   [{ 'path': "DT/00-DataIntegrity/DataIntegritySummary", 'description': "<a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftDT>Description and Instructions</a>" }])

shiftdtlayout(dqmitems, "00-ROChannelSummary",
  [{ 'path': "DT/00-ROChannels/ROChannelSummary", 'description': "<a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftDT>Description and Instructions</a>" }])

shiftdtlayout(dqmitems, "01-OccupancySummary",
  [{ 'path': "DT/01-Digi/OccupancyGlbSummary", 'description': "<a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftDT>Description and Instructions</a>" }])

shiftdtlayout(dqmitems, "02-SegmentSummary",
  [{ 'path': "DT/02-Segments/SegmentGlbSummary", 'description': "<a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftDT>Description and Instructions</a>" }])

shiftdtlayout(dqmitems, "03-DDU_TriggerCorrFactionSummary",
              [{ 'path': "DT/04-LocalTrigger-DDU/DDU_CorrFractionSummary", 'description': "<a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftDT>Description and Instructions</a>" }])

# shiftdtlayout(dqmitems, "04-DCC-TriggerCorrFactionSummary",
#               [{ 'path': "DT/03-LocalTrigger-DCC/DCC_CorrFractionSummary", 'description': "<a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftDT>Description and Instructions</a>" }])

# shiftdtlayout(dqmitems, "05-DDU_Trigger2ndFactionSummary",
#   [{ 'path': "DT/03-LocalTrigger/DDU_2ndFractionSummary", 'description': "<a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftDT>Description and Instructions</a>" }])

# shiftdtlayout(dqmitems, "06-DCC-Trigger2ndFactionSummary",
#   [{ 'path': "DT/03-LocalTrigger/DCC/DCC_2ndFractionSummary", 'description': "<a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftDT>Description and Instructions</a>" }])

shiftdtlayout(dqmitems, "04-NoiseChannelsSummary",
              [{ 'path': "DT/05-Noise/NoiseSummary", 'description': "<a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftDT>Description and Instructions</a>" }])

shiftdtlayout(dqmitems, "05-SynchNoiseSummary",
              [{ 'path': "DT/05-Noise/SynchNoise/SynchNoiseSummary", 'description': "<a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftDT>Description and Instructions</a>" }])

