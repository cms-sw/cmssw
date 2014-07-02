# Dummy check of python syntax within file when run stand-alone
if __name__=="__main__":
    class DQMItem:
        def __init__(self,layout):
            print layout
            print

    dqmitems={}

def hcaloverviewlayout(i, p, *rows): i["Collisions/HcalFeedBack/"+p] = DQMItem(layout=rows)

#  HF+/HF- coincidence triggers, requiring minbias
hcaloverviewlayout(dqmitems, "01 - HF+,HF- distributions for MinBias",
           [{'path':"Hcal/RecHitMonitor_Hcal/Distributions_PassedMinBias/HFweightedtimeDifference",
             'description':"Difference in weighted times between HF+ and HF-, for events passing MinBias HLT trigger, with HT_HFP>1 GeV, HT_HFM>1 GeV, and at least one hit above threshold in both HF+ and HF-.  Weighted times are calculated from all HF cells above threshold in such events."},
            {'path':"Hcal/RecHitMonitor_Hcal/Distributions_PassedMinBias/HFenergyDifference",
             'description':"Sum(E_HFplus - E_HFminus)/Sum(E_HFplus+E_HFminus) for events passing MinBias HLT trigger, with HT_HFP>1 GeV, HT_HFM>1 GeV, and at least one hit above threshold in both HF+ and HF-.  Energies are summed from all HF channels above threshold in such events."},
            ])

# HF+/HF- coincidence triggers, also requiring !BPTX
hcaloverviewlayout(dqmitems, "02 - HF+,HF- distributions for Hcal HLT",
           [{'path':"Hcal/RecHitMonitor_Hcal/Distributions_PassedHcalHLTriggers/HF_HcalHLT_weightedtimeDifference",
             'description':"Difference in weighted times between HF+ and HF-, for events passing Hcal HLT trigger, with HT_HFP>1 GeV, HT_HFM>1 GeV, and at least one hit above threshold in both HF+ and HF-.  Weighted times are calculated from all HF cells above threshold in such events."},
            {'path':"Hcal/RecHitMonitor_Hcal/Distributions_PassedHcalHLTriggers/HF_HcalHLT_energyDifference",
             'description':"Sum(E_HFplus - E_HFminus)/Sum(E_HFplus+E_HFminus) for events passing Hcal HLT trigger, with HT_HFP>1 GeV, HT_HFM>1 GeV, and at least one hit above threshold in both HF+ and HF-.  Energies are summed from all HF channels above threshold in such events."},
            ])

# HE+/HE- distributions, requiring BPTX
hcaloverviewlayout(dqmitems, "03 - HE+,HE- distributions for MinBias",
                   [{'path':"Hcal/RecHitMonitor_Hcal/Distributions_PassedMinBias/HEweightedtimeDifference",
                     'description':"Difference in weighted times between HE+ and HE-, in events passing Hcal HLT trigger with HT_HFP>1 GeV and HT_HFM>1 GeV, and at least one hit above threshold in both HE+ and HE-.  Weighted times are calculated from all HE channels above threshold."},
                    {'path':"Hcal/RecHitMonitor_Hcal/Distributions_PassedMinBias/HEenergyDifference",
                     'description':"Sum(E_HEplus - E_HEminus)/Sum(E_HEplus+E_HEminus), in events passing Hcal HLT trigger with HT_HFP>1 GeV and HT_HFM>1 GeV, and at least one hit above threshold in both HE+ and HE-."}
                     ])


# Digi Shape plots for digis with ADC sum > threshold N
hcaloverviewlayout(dqmitems, "04 - Digi Shapes for Total Digi Signals > N counts",
                   [{'path':"Hcal/DigiMonitor_Hcal/digi_info/HB/HB Digi Shape - over thresh",
                     'description':"Digi shape for all HB digis with sum ADC count > 20 counts above pedestal, for events passing Minbias HLT and with HT_HFP> 1 GeV and HT_HFM > 1 GeV"},
                    {'path':"Hcal/DigiMonitor_Hcal/digi_info/HE/HE Digi Shape - over thresh",
                     'description':"Digi shape for all HE digis with sum ADC count > 20 counts above pedestal, for events passing Minbias HLT and with HT_HFP> 1 GeV and HT_HFM > 1 GeV"}
                    ],
                   [{'path':"Hcal/DigiMonitor_Hcal/digi_info/HF/HF Digi Shape - over thresh",
                     'description':"Digi shape for all HF digis with sum ADC count > 20 counts above pedestal, for events passing Minbias HLT and with HT_HFP> 1 GeV and HT_HFM > 1 GeV"},
                    {'path':"Hcal/DigiMonitor_Hcal/digi_info/HO/HO Digi Shape - over thresh",
                     'description':"Digi shape for all HO digis with sum ADC count > 20 counts above pedestal, for events passing Minbias HLT and with HT_HFP> 1 GeV and HT_HFM > 1 GeV"}
                    ]
                   )

# Lumi plots
hcaloverviewlayout(dqmitems,"05 - Lumi Bunch Crossing Checks",
                   [{'path':"Hcal/RecHitMonitor_Hcal/Distributions_AllRecHits/BX_allevents",
                     'description':"Bunch Crossing # for all processed events"}],
                   [{'path':"Hcal/RecHitMonitor_Hcal/Distributions_PassedMinBias/BX_MinBias_Events_notimecut",
                     'description':"BC # for all events with HT_HF+ > 1 GeV, HT_HF- > 1 GeV, passing MinBias HLT trigger"}],
                   [{'path':"Hcal/RecHitMonitor_Hcal/Distributions_PassedHcalHLTriggers/BX_HcalHLT_Events_notimecut",
                     'description':"BC # for all events with HT_HF+ > 1 GeV, HT_HF- > 1 GeV, Hcal HLT trigger passed"}]
                   )


hcaloverviewlayout(dqmitems,"06 - Events Per Lumi Section",
                   [{'path':"Hcal/RecHitMonitor_Hcal/Distributions_AllRecHits/AllEventsPerLS",
                     'description':"LS # for all processed events"}],
                   [{'path':"Hcal/RecHitMonitor_Hcal/Distributions_PassedMinBias/MinBiasEventsPerLS_notimecut",
                     'description':"LS# for all events with HT_HF+ > 1 GeV, HT_HF- > 1 GeV, passed MinBias HLT  trigger"}],
                   [{'path':"Hcal/RecHitMonitor_Hcal/Distributions_PassedHcalHLTriggers/HcalHLTEventsPerLS_notimecut",
                     'description':"LS# for all events with HT_HF+ > 1 GeV, HT_HF- > 1 GeV, passed Hcal HLT trigger"}]
                   )


hcaloverviewlayout(dqmitems,"07 - Lumi Distributions",
                   [
    {'path':"Hcal/RecHitMonitor_Hcal/Distributions_AllRecHits/MinTime_vs_MinSumET",
     'description':"Min (HF+,HF-) time vs energy, filled for ALL events"},
    {'path':"Hcal/RecHitMonitor_Hcal/Distributions_AllRecHits/HFM_Time_vs_SumET",
     'description':"average HF- time vs energy, filled for ALL events"},
     {'path':"Hcal/RecHitMonitor_Hcal/Distributions_AllRecHits/HFP_Time_vs_SumET",
     'description':"average HF+ time vs energy, filled for ALL events"},
                   ],
                   [
    {'path':"Hcal/RecHitMonitor_Hcal/Distributions_PassedMinBias/timeHFplus_vs_timeHFminus",
     'description':"HF+ vs HF- energy-weighted average time, for events passing MinBias HLT, with HT_HFP>1 GeV and HT_HFM>1 GeV "},
    {'path':"Hcal/RecHitMonitor_Hcal/Distributions_PassedMinBias/SumHT_plus_minus",
     'description':"HF+ sum HT vs HF- sum HT, for all events passing Minbias HLT (but no HT_HFP or HT_HFM cut required)"},
    {'path':"Hcal/RecHitMonitor_Hcal/Distributions_PassedMinBias/SumEnergy_plus_minus",
     'description':"HF+ vs HF- total energy, for events passing Minbias HLT, with HT_HFP>1 GeV and HT_HFM>1 GeV"}
                   ],
                   )


hcaloverviewlayout(dqmitems,"08 - RecHit Average Occupancy",
                   [
                   {'path':"Hcal/RecHitMonitor_Hcal/Distributions_PassedMinBias/HB HE HF Depth 1 Above Threshold RecHit Occupancy",
                   'description':"occupancy for rechits > threshold in MinBias HLT events"},
                   {'path':"Hcal/RecHitMonitor_Hcal/Distributions_PassedMinBias/HB HE HF Depth 2 Above Threshold RecHit Occupancy",
                   'description':"occupancy for rechits > threshold in MinBias HLT events"},
                   ],
                   [
                   {'path':"Hcal/RecHitMonitor_Hcal/Distributions_PassedMinBias/HE Depth 3 Above Threshold RecHit Occupancy",
                   'description':"occupancy for rechits > threshold in MinBias HLT events"},
                   {'path':"Hcal/RecHitMonitor_Hcal/Distributions_PassedMinBias/HO Depth 4 Above Threshold RecHit Occupancy",
                   'description':"occupancy for rechits > threshold in MinBias HLT events"},
                   ],
                   )

hcaloverviewlayout(dqmitems,"09 - RecHit Average Energy",
                   [
                   {'path':"Hcal/RecHitMonitor_Hcal/Distributions_PassedMinBias/HB HE HF Depth 1 Above Threshold RecHit Average Energy GeV",
                   'description':"occupancy for rechits > threshold in MinBias HLT events"},
                   {'path':"Hcal/RecHitMonitor_Hcal/Distributions_PassedMinBias/HB HE HF Depth 2 Above Threshold RecHit Average Energy GeV",
                   'description':"occupancy for rechits > threshold in MinBias HLT events"},
                   ],
                   [
                   {'path':"Hcal/RecHitMonitor_Hcal/Distributions_PassedMinBias/HE Depth 3 Above Threshold RecHit Average Energy GeV",
                   'description':"occupancy for rechits > threshold in MinBias HLT events"},
                   {'path':"Hcal/RecHitMonitor_Hcal/Distributions_PassedMinBias/HO Depth 4 Above Threshold RecHit Average Energy GeV",
                   'description':"occupancy for rechits > threshold in MinBias HLT events"},
                   ],
                   )

hcaloverviewlayout(dqmitems,"10 - RecHit Average Time",
                   [
                   {'path':"Hcal/RecHitMonitor_Hcal/Distributions_PassedMinBias/HB HE HF Depth 1 Above Threshold RecHit Average Time nS",
                   'description':"occupancy for rechits > threshold in MinBias HLT events"},
                   {'path':"Hcal/RecHitMonitor_Hcal/Distributions_PassedMinBias/HB HE HF Depth 2 Above Threshold RecHit Average Time nS",
                   'description':"occupancy for rechits > threshold in MinBias HLT events"},
                   ],
                   [
                   {'path':"Hcal/RecHitMonitor_Hcal/Distributions_PassedMinBias/HE Depth 3 Above Threshold RecHit Average Time nS",
                   'description':"occupancy for rechits > threshold in MinBias HLT events"},
                   {'path':"Hcal/RecHitMonitor_Hcal/Distributions_PassedMinBias/HO Depth 4 Above Threshold RecHit Average Time nS",
                   'description':"occupancy for rechits > threshold in MinBias HLT events"},
                   ],
                   )

hcaloverviewlayout(dqmitems,"11 - Coarse Pedestal Monitor",
                   [
                   {'path':"Hcal/CoarsePedestalMonitor_Hcal/HB HE HF Depth 1 Coarse Pedestal Map",
                   'description':""},
                   {'path':"Hcal/CoarsePedestalMonitor_Hcal/HB HE HF Depth 2 Coarse Pedestal Map",
                   'description':""},
                   ],
                   [
                   {'path':"Hcal/CoarsePedestalMonitor_Hcal/HE Depth 3 Coarse Pedestal Map",
                   'description':""},
                   {'path':"Hcal/CoarsePedestalMonitor_Hcal/HO Depth 4 Coarse Pedestal Map",
                   'description':""},
                   ],
                   )

hcaloverviewlayout(dqmitems,"12 - HFPlus VS HFMinus Energy",
                   [{'path':"Hcal/RecHitMonitor_Hcal/Distributions_PassedMinBias/HFP_HFM_Energy",
                    'description':"Correlation between total energy in HFPlus versus HFMinus. Should be strongly correlated for HI collisions"}
                    ]
                   )

hcaloverviewlayout(dqmitems,"1729 - Temporary HF Timing Study Plots",
                   [{'path':"Hcal/DigiMonitor_Hcal/HFTimingStudy/HFTimingStudy_Average_Time",
                    'description':"HF Timing study average time plot, for events passing HLT MinBias trigger, 20<maxenergy<100, 2<=maxtime<=5"},
                    {'path':"Hcal/DigiMonitor_Hcal/HFTimingStudy/HFTiming_etaProfile",
                     'description':"HF Timing Average time vs. eta, for events passing HLT MinBias trigger, 20<maxenergy<100, 2<=maxtime<=5"},
                    ],
                   [{'path':"Hcal/DigiMonitor_Hcal/HFTimingStudy/HFP_signal_shape",
                     'description':"Normalized HFP Signal Shape passing timing study requirements (pass minbias trigger, 20<maxenergy<100, 2<=maxtime<=5)"},
                    {'path':"Hcal/DigiMonitor_Hcal/HFTimingStudy/HFM_signal_shape",
                     'description':"Normalized HFM Signal Shape passing timing study requirements (pass minbias trigger, 20<maxenergy<100, 2<=maxtime<=5)"}],
                   )
    
    
