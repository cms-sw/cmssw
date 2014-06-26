def hltoverviewlayout(i, p, *rows): i["Collisions/HLTFeedBack/" + p] = DQMItem(layout=rows)

hltoverviewlayout(dqmitems,"00 HLT_Egamma_Pass_Any",
  	[{'path': "HLT/FourVector/PathsSummary/HLT_Egamma_Pass_Any", 'description': "Shows total number of HLT Egamma trigger accepts and the total number of any HLT accepts. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/TriggerShiftHLTGuide\">here</a>."}])

hltoverviewlayout(dqmitems,"01 HLT_JetMet_Pass_Any",
  	[{'path': "HLT/FourVector/PathsSummary/HLT_JetMet_Pass_Any", 'description': "Shows total number of HLT JetMET trigger accepts and the total number of any HLT accepts. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/TriggerShiftHLTGuide\">here</a>."}])

hltoverviewlayout(dqmitems,"02 HLT_Muon_Pass_Any",
  	[{'path': "HLT/FourVector/PathsSummary/HLT_Muon_Pass_Any", 'description': "Shows total number of HLT Muon trigger accepts and the total number of any HLT accepts. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/TriggerShiftHLTGuide\">here</a>."}])

hltoverviewlayout(dqmitems,"03 HLT_Rest_Pass_Any",
  	[{'path': "HLT/FourVector/PathsSummary/HLT_Rest_Pass_Any", 'description': "Shows total number of HLT Rest trigger accepts and the total number of any HLT accepts. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/TriggerShiftHLTGuide\">here</a>."}])

hltoverviewlayout(dqmitems,"04 HLT_Special_Pass_Any",
  	[{'path': "HLT/FourVector/PathsSummary/HLT_Special_Pass_Any", 'description': "Shows total number of HLT Special trigger accepts and the total number of any HLT accepts. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/TriggerShiftHLTGuide\">here</a>."}])

hltoverviewlayout(dqmitems,"05 All_count_LS",
  	[{'path': "HLT/FourVector/PathsSummary/HLT LS/All_count_LS", 'description': "Show the number of events passing all HLT paths vs. LS (2D histogram) . For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/TriggerShiftHLTGuide\">here</a>."}])

hltoverviewlayout(dqmitems,"06 Group_0_paths_count_LS",
  	[{'path': "HLT/FourVector/PathsSummary/HLT LS/Group_0_paths_count_LS", 'description': "Show the number of events passing HLT paths which are subdivided in groups of 20 vs. LS (2D histogram) . For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/TriggerShiftHLTGuide\">here</a>."}])

hltoverviewlayout(dqmitems,"07 Group_1_paths_count_LS",
  	[{'path': "HLT/FourVector/PathsSummary/HLT LS/Group_1_paths_count_LS", 'description': "Show the number of events passing HLT paths which are subdivided in groups of 20 vs. LS (2D histogram) . For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/TriggerShiftHLTGuide\">here</a>."}])

hltoverviewlayout(dqmitems,"08 Group_2_paths_count_LS",
  	[{'path': "HLT/FourVector/PathsSummary/HLT LS/Group_2_paths_count_LS", 'description': "Show the number of events passing HLT paths which are subdivided in groups of 20 vs. LS (2D histogram) . For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/TriggerShiftHLTGuide\">here</a>."}])

hltoverviewlayout(dqmitems,"09 Group_3_paths_count_LS",
  	[{'path': "HLT/FourVector/PathsSummary/HLT LS/Group_3_paths_count_LS", 'description': "Show the number of events passing HLT paths which are subdivided in groups of 20 vs. LS (2D histogram) . For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/TriggerShiftHLTGuide\">here</a>."}])

hltoverviewlayout(dqmitems,"10 Group_4_paths_count_LS",
  	[{'path': "HLT/FourVector/PathsSummary/HLT LS/Group_4_paths_count_LS", 'description': "Show the number of events passing HLT paths which are subdivided in groups of 20 vs. LS (2D histogram) . For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/TriggerShiftHLTGuide\">here</a>."}])

hltoverviewlayout(dqmitems,"11 Group_-1_paths_count_LS",
  	[{'path': "HLT/FourVector/PathsSummary/HLT LS/Group_-1_paths_count_LS", 'description': "Show the number of events passing HLT paths which are subdivided in groups of 20 vs. LS (2D histogram) . For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/TriggerShiftHLTGuide\">here</a>."}])
