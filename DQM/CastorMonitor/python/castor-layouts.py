def castorlayout(i, p, *rows): i["Castor/Layouts/" + p] = DQMItem(layout=rows)


castorlayout(dqmitems, "CASTOR Pedestals",
          [{ 'path': "Castor/CastorPedestalMonitor/Castor All Pedestal Values",
                   'description': "test <a href=https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMHistograms>HcalDQMHistograms</a>" },

          { 'path': "Castor/CastorPedestalMonitor/Castor Pedestal Mean Reference Values",
                   'description': "test <a href=https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMHistograms>HcalDQMHistograms</a>"},

          { 'path': "Castor/CastorPedestalMonitor/Castor Pedestal RMS Reference Values",
                   'description': "test <a href=https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMHistograms>HcalDQMHistograms</a>" }]      
          )

castorlayout(dqmitems, "CASTOR LED Distributions",
          [{ 'path': "Castor/CastorLEDMonitor/Castor Average Pulse Energy",
                   'description': "test <a href=https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMHistograms>HcalDQMHistograms</a>" },

           { 'path': "Castor/CastorLEDMonitor/Castor Average Pulse Time",
                   'description': "test <a href=https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMHistograms>HcalDQMHistograms</a>"},

          { 'path': "Castor/CastorLEDMonitor/Castor Average Pulse Shape",
                   'description': "test <a href=https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMHistograms>HcalDQMHistograms</a>" }]      
          )

castorlayout(dqmitems, "CASTOR RecHit Energy Distributions",
          [{ 'path': "Castor/CastorRecHitMonitor/RecHit Channel Energy Map",
                   'description': "test <a href=https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMHistograms>HcalDQMHistograms</a>" },
             
          { 'path': "Castor/CastorRecHitMonitor/Castor RecHit Energies",
                   'description': "test <a href=https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMHistograms>HcalDQMHistograms</a>" },
           
          { 'path': "Castor/CastorRecHitMonitor/Castor RecHit Times",
                   'description': "test <a href=https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMHistograms>HcalDQMHistograms</a>" }]
           )


