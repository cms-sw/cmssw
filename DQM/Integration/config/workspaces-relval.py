server.workspace('DQMQuality', 0, 'Summaries', 'Summary')
server.workspace('DQMSummary', 1, 'Summaries', 'Reports')
server.workspace('DQMShift',   2, 'Summaries', 'Shift')
server.workspace('DQMContent', 3, 'Summaries', 'Everything', '^')

server.workspace('DQMContent', 10, 'Data', 'Tk' , '^Tk/')
server.workspace('DQMContent', 11, 'Data', 'Ecal' , '^Ecal.*/',
        'EcalBarrel/EBOccupancyTask/EBOT rec hit spectrum',
        'EcalEndcap/EEOccupancyTask/EEOT rec hit spectrum EE +',
        'EcalEndcap/EEOccupancyTask/EEOT rec hit spectrum EE -',
        'EcalBarrel/EcalInfo/EBMM hit number',
        'EcalEndcap/EcalInfo/EEMM hit number',
        'EcalBarrel/EBSummaryClient/EBTMT timing mean 1D summary',
        'EcalEndcap/EESummaryClient/EETMT EE - timing mean 1D summary',
        'EcalEndcap/EESummaryClient/EETMT EE + timing mean 1D summary'
)
server.workspace('DQMContent', 12, 'Data', 'Hcal' , '^Hcal/')
server.workspace('DQMContent', 13, 'Data', 'DT' , '^DT/')
server.workspace('DQMContent', 14, 'Data', 'CSC' , '^CSC/')
server.workspace('DQMContent', 15, 'Data', 'RPC' , '^RPC/')
server.workspace('DQMContent', 16, 'Data', 'Tracking' , '^Tracking/')
server.workspace('DQMContent', 17, 'Data', 'Electron' , '^Electron/')
server.workspace('DQMContent', 18, 'Data', 'Photon' , '^Photon/')
server.workspace('DQMContent', 19, 'Data', 'Muon' , '^Muon/')
server.workspace('DQMContent', 20, 'Data', 'Jet' , '^Jet/')
server.workspace('DQMContent', 21, 'Data', 'MET' , '^MET/')
server.workspace('DQMContent', 22, 'Data', 'BTag' , '^BTag/')
server.workspace('DQMContent', 23, 'Data', 'Tau' , '^Tau/')

server.workspace('DQMContent', 30, 'Monte Carlo', 'MC Tk' , '^Tk/')
server.workspace('DQMContent', 31, 'Monte Carlo', 'MC Ecal' , '^Ecal.*/')
server.workspace('DQMContent', 32, 'Monte Carlo', 'MC Hcal' , '^Hcal/')
server.workspace('DQMContent', 33, 'Monte Carlo', 'MC DT' , '^DT/')
server.workspace('DQMContent', 34, 'Monte Carlo', 'MC CSC' , '^CSC/')
server.workspace('DQMContent', 35, 'Monte Carlo', 'MC RPC' , '^RPC/')
server.workspace('DQMContent', 36, 'Monte Carlo', 'MC Tracking' , '^Tracking/')
server.workspace('DQMContent', 37, 'Monte Carlo', 'MC Electron' , '^Electron/')
server.workspace('DQMContent', 38, 'Monte Carlo', 'MC Photon' , '^Photon/')
server.workspace('DQMContent', 39, 'Monte Carlo', 'MC Muon' , '^Muon/')
server.workspace('DQMContent', 40, 'Monte Carlo', 'MC Jet' , '^Jet/')
server.workspace('DQMContent', 41, 'Monte Carlo', 'MC MET' , '^MET/')
server.workspace('DQMContent', 42, 'Monte Carlo', 'MC BTag' , '^BTag/')
server.workspace('DQMContent', 43, 'Monte Carlo', 'MC Tau' , '^Tau/')
