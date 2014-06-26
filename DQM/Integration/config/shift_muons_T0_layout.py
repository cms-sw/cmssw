def shiftmuonlayout(i, p, *rows): i["00 Shift/Muons/" + p] = DQMItem(layout=rows)

shiftmuonlayout(dqmitems, "00-reportSummary",
                [{ 'path': "Muons/EventInfo/reportSummaryMap", 'description': "<a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftOfflineMuon>Description</a>" }])


shiftmuonlayout(dqmitems, "01-kinematicsSummary",
                [{ 'path': "Muons/TestSummary/kinematicsSummaryMap", 'description': "<a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftOfflineMuon>Description</a>" }])


shiftmuonlayout(dqmitems, "02-residualsSummary",
                [{ 'path': "Muons/TestSummary/residualsSummaryMap", 'description': "<a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftOfflineMuon>Description</a>" }])


shiftmuonlayout(dqmitems, "03-energySummary",
                [{ 'path': "Muons/TestSummary/energySummaryMap", 'description': "<a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftOfflineMuon>Description</a>" }])


shiftmuonlayout(dqmitems, "04-muonIdSummary",
                [{ 'path': "Muons/TestSummary/muonIdSummaryMap", 'description': "<a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftOfflineMuon>Description</a>" }])


shiftmuonlayout(dqmitems, "05-molteplicitySummary",
                [{ 'path': "Muons/TestSummary/molteplicitySummaryMap", 'description': "<a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftOfflineMuon>Description</a>" }])




