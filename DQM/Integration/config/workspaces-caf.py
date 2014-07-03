server.workspace('DQMQuality', 0, 'Summaries', 'Summary')
server.workspace('DQMSummary', 1, 'Summaries', 'Reports')
server.workspace('DQMShift',   2, 'Summaries', 'Shift')
server.workspace('DQMContent', 3, 'Summaries', 'Everything', '^')

server.workspace('DQMContent', 10, 'Tracker/Muons', 'Pixel', '^Pixel/')
server.workspace('DQMContent', 11, 'Tracker/Muons', 'SiStrip', '^SiStrip/')               
server.workspace('DQMContent', 12, 'Tracker/Muons', 'CSC', '^CSC/')
server.workspace('DQMContent', 13, 'Tracker/Muons', 'DT', '^DT/')
server.workspace('DQMContent', 14, 'Tracker/Muons', 'RPC', '^RPC/') 

server.workspace('DQMContent', 21, 'Calorimeter', 'EcalBarrel', '^EcalBarrel/')
server.workspace('DQMContent', 22, 'Calorimeter', 'EcalEndcap', '^EcalEndcap/')
server.workspace('DQMContent', 23, 'Calorimeter', 'EcalPreshower', '^EcalPreshower')
server.workspace('DQMContent', 24, 'Calorimeter', 'HCAL', '^Hcal/')
server.workspace('DQMContent', 25, 'Calorimeter', 'CASTOR', '^Castor/')

server.workspace('DQMContent', 31, 'Trigger/Lumi', 'L1T', '^L1T/')
server.workspace('DQMContent', 32, 'Trigger/Lumi', 'L1TEMU', '^L1TEMU/')
server.workspace('DQMContent', 33, 'Trigger/Lumi', 'HLT', '^HLT/')
server.workspace('DQMContent', 34, 'Trigger/Lumi', 'HLX', '^HLX')

server.workspace('DQMContent', 41, 'POG', 'Muons', '^Muons/')
server.workspace('DQMContent', 42, 'POG', 'JetMet', '^JetMET/')
server.workspace('DQMContent', 43, 'POG', 'EGamma', '^Egamma/')

server.workspace('DQMContent', 51, 'Alignment', 'TrackerAl', '^Alignment/Tracker/',
                 'Alignment/Tracker/Layouts/00 - PIXEL absolute Residuals',
		 'Alignment/Tracker/Layouts/01 - STRIP absolute Residuals',
		 'Alignment/Tracker/Layouts/02 - PIXEL normalized Residuals',
		 'Alignment/Tracker/Layouts/03 - STRIP normalized Residuals',
		 'Alignment/Tracker/Layouts/10 - PIXEL DMRs',
		 'Alignment/Tracker/Layouts/11 - STRIP DMRs',
)
