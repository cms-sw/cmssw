# Dummy check of python syntax within file when run stand-alone
if __name__=="__main__":
    class DQMItem:
        def __init__(self,layout):
            print layout

    dqmitems={}
        
def hcalcaliblayout(i, p, *rows): i["HcalCalib/Layouts/" + p] = DQMItem(layout=rows)

hcalcaliblayout(dqmitems, "01 HcalCalib Summary",
           [{ 'path':"HcalCalib/EventInfo/reportSummaryMap",
             'description':"This shows the fraction of bad cells in each subdetector.  All subdetectors should appear green.  Values should all be above 98%."}]
           )

hcalcaliblayout(dqmitems, "02 HcalCalib Problem Pedestals",
                [{ 'path':"HcalCalib/DetDiagPedestalMonitor_Hcal/ ProblemDetDiagPedestal",
                   'description': "Channels with pedestals that have some kind of problem (bad mean or RMS, unstable, or missing)"}]
                )
                
hcalcaliblayout(dqmitems, "03 HcalCalib Problem Laser",
                [{ 'path':"HcalCalib/DetDiagLaserMonitor_Hcal/ ProblemDetDiagLaser",
                   'description': "Channels that are outside reference bounds in either time or energy"}]
)

hcalcaliblayout(dqmitems, "04 HcalCalib Pedestal Check",
                [{ 'path': "HcalCalib/DetDiagPedestalMonitor_Hcal/Summary Plots/HBHEHF pedestal mean map",
                   'description': "Pedestal mean values calculated from orbit gap events for HB, HE, and HF"},
                { 'path': "HcalCalib/DetDiagPedestalMonitor_Hcal/Summary Plots/HBHEHF pedestal rms map",
                   'description': "Pedestal RMS values calculated from orbit gap events for HB, HE, and HF"}],  
                [{ 'path': "HcalCalib/DetDiagPedestalMonitor_Hcal/Summary Plots/HO pedestal mean map",
                   'description': "Pedestal mean values calculated from orbit gap events for HO"},
                { 'path': "HcalCalib/DetDiagPedestalMonitor_Hcal/Summary Plots/HO pedestal rms map",
                  'description': "Pedestal RMS values calculated from orbit gap events for HO"}]  
                )

hcalcaliblayout(dqmitems, "05 HcalCalib Pedestal Reference Comparison",
                [{'path':"HcalCalib/DetDiagPedestalMonitor_Hcal/Summary Plots/HB Pedestal-Reference Distribution (average over 4 caps)"},
                 {'path':"HcalCalib/DetDiagPedestalMonitor_Hcal/Summary Plots/HE Pedestal-Reference Distribution (average over 4 caps)"}],
                [{'path':"HcalCalib/DetDiagPedestalMonitor_Hcal/Summary Plots/HF Pedestal-Reference Distribution (average over 4 caps)"},
                 {'path':"HcalCalib/DetDiagPedestalMonitor_Hcal/Summary Plots/HO Pedestal-Reference Distribution (average over 4 caps)"}]
                )

hcalcaliblayout(dqmitems, "06 HcalCalib Laser Reference Comparison",
                [{ 'path':"HcalCalib/DetDiagLaserMonitor_Hcal/Summary Plots/HBHEHF Laser Energy_div_Ref",
                   'description':"Laser Average energy - reference in HBHEHF"},
                 { 'path':"HcalCalib/DetDiagLaserMonitor_Hcal/Summary Plots/HO Laser Energy_div_Ref",
                   'description':"Laser Average energy - reference in HO"},],
                
                [{ 'path':"HcalCalib/DetDiagLaserMonitor_Hcal/Summary Plots/HBHEHF Laser (Timing-Ref)+1",
                   'description':"Laser Average timing - reference in HBHEHF"},
                 { 'path':"HcalCalib/DetDiagLaserMonitor_Hcal/Summary Plots/HO Laser (Timing-Ref)+1",
                   'description':"Laser Average timing - reference in HO"},]
                )

hcalcaliblayout(dqmitems, "07 HcalCalib Laser RBX Plots",
                [{ 'path':"HcalCalib/DetDiagLaserMonitor_Hcal/Summary Plots/HB RBX average Time-Ref",
                   'description':"1D Laser RBX energy - reference in HB"},
                 { 'path':"HcalCalib/DetDiagLaserMonitor_Hcal/Summary Plots/HE RBX average Time-Ref",
                   'description':"1D Laser RBX energy - reference in HE"}],
                [{ 'path':"HcalCalib/DetDiagLaserMonitor_Hcal/Summary Plots/HF RoBox average Time-Ref",
                   'description':"1D Laser RBX energy - reference in HF"},
                 { 'path':"HcalCalib/DetDiagLaserMonitor_Hcal/Summary Plots/HO RBX average Time-Ref",
                   'description':"1D Laser RBX energy - reference in HO"}]

                )

hcalcaliblayout(dqmitems, "08 HcalCalib RecHit Energies",
                [{'path': "HcalCalib/RecHitMonitor_Hcal/Distributions_AllRecHits/rechit_1D_plots/HB_energy_1D",
                  'description':"Average energy/hit for each of the 2592 channels in HB"},
                {'path': "HcalCalib/RecHitMonitor_Hcal/Distributions_AllRecHits/rechit_1D_plots/HE_energy_1D",
                  'description':"Average energy/hit for each of the 2592 channels in HE"}],
                [{'path': "HcalCalib/RecHitMonitor_Hcal/Distributions_AllRecHits/rechit_1D_plots/HF_energy_1D",
                  'description':"Average energy/hit for each of the 1728 channels in HF"},
                 {'path': "HcalCalib/RecHitMonitor_Hcal/Distributions_AllRecHits/rechit_1D_plots/HO_energy_1D",
                  'description':"Average energy/hit for each of the 2160 channels in HO.  (You may see fewer entries than 2160 because of ~30 known dead cells in HO.)"}],
                )

hcalcaliblayout(dqmitems, "09 HCAL Calibration Type",
                [{'path':"HcalCalib/HcalInfo/CalibrationType",
                  'description':"This shows the distribution of HCAL event types received by DQM.  Calibration events (pedestal, laser, etc.) are used for additional monitoring and diagnostics."}])
