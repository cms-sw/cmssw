from .adapt_to_new_backend import *
dqmitems={}

def dtlayout(i, p, *rows): i["DT/Layouts/" + p] = rows

dtlayout(dqmitems, "00-Summary/00-DataIntegritySummary",
  [{ 'path': "DT/00-DataIntegrithdjfhjdhy/DataIntegritySummary", 'description': "<a href=https://twiki.cern.ch/twiki/bin/view/CMS/DTDQMPlots>Description and Instructions</a>" }])

dtlayout(dqmitems, "00-Summary/00-ROChannelSummary",
  [{ 'path': "DT/00-ROChannels/ROChannelSummary", 'description': "<a href=https://twiki.cern.ch/twiki/bin/view/CMS/DTDQMPlots>Description and Instructions</a>" }])

dtlayout(dqmitems, "00-Summary/01-OccupancySummary",
  [{ 'path': "DT/01-Digi/OccupancySummary", 'description': "<a href=https://twiki.cern.ch/twiki/bin/view/CMS/DTDQMPlots>Description and Instructions</a>" }])

dtlayout(dqmitems, "00-Summary/02-SegmentSummary",
  [{ 'path': "DT/02-Segments/segmentSummary", 'description': "<a href=https://twiki.cern.ch/twiki/bin/view/CMS/DTDQMPlots>Description and Instructions</a>" }])

dtlayout(dqmitems, "00-Summary/03-TM_TriggerCorrFractionSummaryIn",
  [{ 'path': "DT/03-LocalTrigger-TM/TM_CorrFractionSummaryIn", 'description': "<a href=https://twiki.cern.ch/twiki/bin/view/CMS/DTDQMPlots>Description and Instructions</a>" }])

dtlayout(dqmitems, "00-Summary/04-TM_Trigger2ndFractionSummaryIn",
  [{ 'path': "DT/03-LocalTrigger-TM/TM_2ndFractionSummaryIn", 'description': "<a href=https://twiki.cern.ch/twiki/bin/view/CMS/DTDQMPlots>Description and Instructions</a>" }])

dtlayout(dqmitems, "00-Summary/05-NoiseChannelsSummary",
  [{ 'path': "DT/05-Noise/NoiseSummary", 'description': "<a href=https://twiki.cern.ch/twiki/bin/view/CMS/DTDQMPlots>Description and Instructions</a>" }])

dtlayout(dqmitems, "00-Summary/06-SynchNoiseSummary",
         [{ 'path': "DT/05-Noise/SynchNoise/SynchNoiseSummary", 'description': "<a href=https://twiki.cern.ch/twiki/bin/view/CMS/DTDQMPlots>Description and Instructions</a>" }])

#dtlayout(dqmitems, "00-Summary/09-TestPulseOccupancy",
#         [{ 'path': "DT/10-TestPulses/OccupancySummary", 'description': "<a href=https://twiki.cern.ch/twiki/bin/view/CMS/DTDQMPlots>Description and Instructions</a>" }])

#### OCCUPANCIES #################################################################################

for wheel in range(-2, 3):
    for station in range (1, 5):
        for sector in range (1, 15):
            if station != 4 and (sector == 13 or sector == 14):
                continue
            name = "01-Occupancy/Wheel" + str(wheel) + "/St" + str(station) + "_Sec" + str(sector)
            histoname = "DT/01-Digi/Wheel" + str(wheel) + "/Sector" + str(sector) + "/Station" + str(station) +  "/OccupancyAllHits_perCh_W" + str(wheel) + "_St" + str(station) + "_Sec" +  str(sector)
            dtlayout(dqmitems, name,[{ 'path': histoname}])

#### TIME BOXES #################################################################################

for wheel in range(-2, 3):
    for sector in range (1, 15):
        for station in range (1, 5):
            if station != 4 and (sector == 13 or sector == 14):
                continue
            name = "02-TimeBoxes/Wheel" + str(wheel) + "/St" + str(station) + "_Sec" + str(sector)
            histoname = "DT/01-Digi/Wheel" + str(wheel) + "/Sector" + str(sector) + "/Station" + str(station) +  "/TimeBox_W" + str(wheel) + "_St" + str(station) + "_Sec" +  str(sector)
            histoname_SL1 = histoname + "_SL1"
            histoname_SL2 = histoname + "_SL2"
            histoname_SL3 = histoname + "_SL3"
            if station != 4:
                dtlayout(dqmitems, name,[{ 'path': histoname_SL1}],
                         [{ 'path': histoname_SL2}],
                         [{ 'path': histoname_SL3}])
            else:
                dtlayout(dqmitems, name,[{ 'path': histoname_SL1}],
                         [{ 'path': histoname_SL3}])

#### EVENT SIZE #################################################################################
for fed in range(770, 775):
   name = name = "03-FEDEventSize/FED" + str(fed)
   histoname = "DT/00-DataIntegrity/FED" + str(fed) + "/FED" + str(fed) + "_EventLenght"
   dtlayout(dqmitems, name,[{ 'path': histoname}])
   for rosid in range(1, 13):
       ros = rosid
       name = "04-ROSEventSize/FED" + str(fed) + "_ROS" + str(ros)
       histoname = "DT/00-DataIntegrity/FED" + str(fed) + "/ROS" + str(ros) + "/FED" + str(fed) + "_ROS" + str(ros) + "_ROSEventLenght"
       dtlayout(dqmitems, name,[{ 'path': histoname}])

#### TRIGGER SYNCH ##############################################################################

for wheel in range(-2, 3):
    name = "05-TriggerSynch/01-CorrectBX_In_Wh" + str(wheel) + "_TM"
    histoname = "DT/03-LocalTrigger-TM/Wheel" + str(wheel) + "/TM_CorrectBXPhiIn_W" + str(wheel)
    dtlayout(dqmitems, name,[{ 'path': histoname}])
    name = "05-TriggerSynch/Peak-Mean/01-Peak-Mean_In_Wh" + str(wheel) + "_TM"
    histoname = "DT/03-LocalTrigger-TM/Wheel" + str(wheel) + "/TM_ResidualBXPhiIn_W" + str(wheel)
    dtlayout(dqmitems, name,[{ 'path': histoname}])
    name = "05-TriggerSynch/01-CorrectBX_Out_Wh" + str(wheel) + "_TM"
    histoname = "DT/03-LocalTrigger-TM/Wheel" + str(wheel) + "/TM_CorrectBXPhiOut_W" + str(wheel)
    dtlayout(dqmitems, name,[{ 'path': histoname}])
    name = "05-TriggerSynch/Peak-Mean/01-Peak-Mean_Out_Wh" + str(wheel) + "_TM"
    histoname = "DT/03-LocalTrigger-TM/Wheel" + str(wheel) + "/TM_ResidualBXPhiOut_W" + str(wheel)
    dtlayout(dqmitems, name,[{ 'path': histoname}])

#### TRIGGER BASICS ##############################################################################

for wheel in range(-2, 3):
    name = "06-TriggerBasics/01-CorrFraction_In_Wh" + str(wheel) + "_TM"
    histoname = "DT/03-LocalTrigger-TM/Wheel" + str(wheel) + "/TM_CorrFractionPhiIn_W" + str(wheel)
    dtlayout(dqmitems, name,[{ 'path': histoname}])
    name = "06-TriggerBasics/03-2ndFractionPhi_In_Wh" + str(wheel)
    histoname = "DT/03-LocalTrigger-TM/Wheel" + str(wheel) + "/TM_2ndFractionPhiIn_W" + str(wheel)
    dtlayout(dqmitems, name,[{ 'path': histoname}])
    name = "06-TriggerBasics/01-CorrFraction_Out_Wh" + str(wheel) + "_TM"
    histoname = "DT/03-LocalTrigger-TM/Wheel" + str(wheel) + "/TM_CorrFractionPhiOut_W" + str(wheel)
    dtlayout(dqmitems, name,[{ 'path': histoname}])
    name = "06-TriggerBasics/03-2ndFractionPhi_Out_Wh" + str(wheel)
    histoname = "DT/03-LocalTrigger-TM/Wheel" + str(wheel) + "/TM_2ndFractionPhiOut_W" + str(wheel)
    dtlayout(dqmitems, name,[{ 'path': histoname}])

#### TRIGGER POS LUTs ###########################################################################
for wheel in range(-2, 3):
    for sector in range (1, 13):
        name = "07-TriggerPosLUTs/Wheel" + str(wheel) + "/Sec" + str(sector) + " In"
        histoname1 = "DT/03-LocalTrigger-TM/Wheel" + str(wheel) + "/Sector" + str(sector) + "/Station1/Segment/TM_PhiResidualIn_W" + str(wheel) + "_Sec" +  str(sector) + "_St1"
        histoname2 = "DT/03-LocalTrigger-TM/Wheel" + str(wheel) + "/Sector" + str(sector) + "/Station2/Segment/TM_PhiResidualIn_W" + str(wheel) + "_Sec" +  str(sector) + "_St2"
        histoname3 = "DT/03-LocalTrigger-TM/Wheel" + str(wheel) + "/Sector" + str(sector) + "/Station3/Segment/TM_PhiResidualIn_W" + str(wheel) + "_Sec" +  str(sector) + "_St3"
        histoname4 = "DT/03-LocalTrigger-TM/Wheel" + str(wheel) + "/Sector" + str(sector) + "/Station4/Segment/TM_PhiResidualIn_W" + str(wheel) + "_Sec" +  str(sector) + "_St4"
        dtlayout(dqmitems, name,[{ 'path': histoname1},{ 'path': histoname2}], [{ 'path': histoname3},{ 'path': histoname4}])
        name = "07-TriggerPosLUTs/Wheel" + str(wheel) + "/Sec" + str(sector) + " Out"
        histoname5 = "DT/03-LocalTrigger-TM/Wheel" + str(wheel) + "/Sector" + str(sector) + "/Station1/Segment/TM_PhiResidualOut_W" + str(wheel) + "_Sec" +  str(sector) + "_St1"
        histoname6 = "DT/03-LocalTrigger-TM/Wheel" + str(wheel) + "/Sector" + str(sector) + "/Station2/Segment/TM_PhiResidualOut_W" + str(wheel) + "_Sec" +  str(sector) + "_St2"
        histoname7 = "DT/03-LocalTrigger-TM/Wheel" + str(wheel) + "/Sector" + str(sector) + "/Station3/Segment/TM_PhiResidualOut_W" + str(wheel) + "_Sec" +  str(sector) + "_St3"
        histoname8 = "DT/03-LocalTrigger-TM/Wheel" + str(wheel) + "/Sector" + str(sector) + "/Station4/Segment/TM_PhiResidualOut_W" + str(wheel) + "_Sec" +  str(sector) + "_St4"
        dtlayout(dqmitems, name,[{ 'path': histoname5},{ 'path': histoname6}],[{ 'path': histoname7},{ 'path': histoname8}])

#### TRIGGER DIR LUTs ###########################################################################
for wheel in range(-2, 3):
    for sector in range (1, 13):
        name = "08-TriggerDirLUTs/Wheel" + str(wheel) + "/Sec" + str(sector) + " In"
        histoname1 = "DT/03-LocalTrigger-TM/Wheel" + str(wheel) + "/Sector" + str(sector) + "/Station1/Segment/TM_PhibResidualIn_W" + str(wheel) + "_Sec" +  str(sector) + "_St1"
        histoname2 = "DT/03-LocalTrigger-TM/Wheel" + str(wheel) + "/Sector" + str(sector) + "/Station2/Segment/TM_PhibResidualIn_W" + str(wheel) + "_Sec" +  str(sector) + "_St2"
        histoname3 = "DT/03-LocalTrigger-TM/Wheel" + str(wheel) + "/Sector" + str(sector) + "/Station3/Segment/TM_PhibResidualIn_W" + str(wheel) + "_Sec" +  str(sector) + "_St3"
        histoname4 = "DT/03-LocalTrigger-TM/Wheel" + str(wheel) + "/Sector" + str(sector) + "/Station4/Segment/TM_PhibResidualIn_W" + str(wheel) + "_Sec" +  str(sector) + "_St4"
        dtlayout(dqmitems, name,[{ 'path': histoname1},{ 'path': histoname2}], [{ 'path': histoname3},{ 'path': histoname4}])
        name = "08-TriggerDirLUTs/Wheel" + str(wheel) + "/Sec" + str(sector) + " Out"
        histoname5 = "DT/03-LocalTrigger-TM/Wheel" + str(wheel) + "/Sector" + str(sector) + "/Station1/Segment/TM_PhibResidualOut_W" + str(wheel) + "_Sec" +  str(sector) + "_St1"
        histoname6 = "DT/03-LocalTrigger-TM/Wheel" + str(wheel) + "/Sector" + str(sector) + "/Station2/Segment/TM_PhibResidualOut_W" + str(wheel) + "_Sec" +  str(sector) + "_St2"
        histoname7 = "DT/03-LocalTrigger-TM/Wheel" + str(wheel) + "/Sector" + str(sector) + "/Station3/Segment/TM_PhibResidualOut_W" + str(wheel) + "_Sec" +  str(sector) + "_St3"
        histoname8 = "DT/03-LocalTrigger-TM/Wheel" + str(wheel) + "/Sector" + str(sector) + "/Station4/Segment/TM_PhibResidualOut_W" + str(wheel) + "_Sec" +  str(sector) + "_St4"
        dtlayout(dqmitems, name,[{ 'path': histoname5},{ 'path': histoname6}], [{ 'path': histoname7},{ 'path': histoname8}])


#### HITS RESIDUALS ###########################################################################
for wheel in range(-2, 3):
    for station in range (1, 5):
        for sector in range (1, 15):
            if station != 4 and (sector == 13 or sector == 14):
               continue
            name = "09-Nhits/Wheel" + str(wheel) + "/Sector" + str(sector) + "_Station" + str(station)
            histoname = "DT/02-Segments/Wheel" + str(wheel)+ "/Sector" + str(sector) + "/Station" + str(station) + "/h4DSegmNHits_W" + str(wheel) + "_St" + str(station) + "_Sec" + str(sector)
            dtlayout(dqmitems, name,[{ 'path': histoname}])

#### SEGMENTS RESIDUALS #########################################################################
for wheel in range(-2, 3):
    for station in range (1, 5):
        for sector in range (1, 15):
            if station != 4 and (sector == 13 or sector == 14):
               continue
            for sl in range (1, 4):
                if sl == 2 and (sector == 13 or sector == 14 or station == 4):
                   continue
                name = "10-Segments/Wheel" + str(wheel) + "/Sector" + str(sector) + "_Station" + str(station) + "_SL" + str(sl)
                histoname = "DT/02-Segments/Wheel" + str(wheel)+ "/Sector" + str(sector) + "/Station" + str(station) + "/hResDist_W" + str(wheel) + "_St" + str(station) + "_Sec" + str(sector) + "_SL" + str(sl)
                dtlayout(dqmitems, name,[{ 'path': histoname}])

#### QUALITY vs PHI RAD ############################################################################
for wheel in range(-2, 3):
    for station in range (1, 5):
        for sector in range (1, 13):
            name = "11-QualityPhi/Wheel" + str(wheel) + "/Sector" + str(sector) + "_Station" + str(station)
            histoname = "DT/03-LocalTrigger-TM/Wheel" + str(wheel)+ "/Sector" + str(sector) + "/Station" + str(station) + "/LocalTriggerPhiIn/TM_QualvsPhirad_In_W" + str(wheel) + "_Sec" + str(sector) + "_St" + str(station)
            dtlayout(dqmitems, name,[{ 'path': histoname}])

#
#
#
#
#

apply_dqm_items_to_new_back_end(dqmitems, __file__)
