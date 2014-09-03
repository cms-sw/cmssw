def l1temulayout(i, p, *rows): i["00 Shift/L1TEMU/" + p] = DQMItem(layout=rows)

l1temulayout(dqmitems,"00-Global Summary",
             [{'path': "L1TEMU/common/sysrates", 'description': "Data|Emulator disagreement rates per subsystem. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftTrigger\">here</a>."},
              {'path': "L1TEMU/common/errorflag", 'description': "Data|Emulator overall disagreement flags. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftTrigger\">here</a>."}],
             [{'path': "L1TEMU/common/sysncandData", 'description': "Number of trigger objects per subsystem observed in data. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftTrigger\">here</a>."},
              {'path': "L1TEMU/common/sysncandEmul", 'description': "Number of trigger objects per subsystem expected by emulator. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftTrigger\">here</a>."}])

l1temulayout(dqmitems,"01-ECAL (Dis)Agreement Flag",
             [{'path': "L1TEMU/ECAL/ETPErrorFlag", 'description': "Data|Emulator disagreement type for ECAL trigger primitives. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftTriggerEmulator\">here</a>."}])

l1temulayout(dqmitems,"02-HCAL (Dis)Agreement Flag",
             [{'path': "L1TEMU/HCAL/HTPErrorFlag", 'description': "Data|Emulator disagreement type for HCAL trigger primitives. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftTriggerEmulator\">here</a>."}])

l1temulayout(dqmitems,"03-RCT (Dis)Agreement Flag",
             [{'path': "L1TEMU/RCT/RCTErrorFlag", 'description': "Data|Emulator disagreement type for RCT. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftTriggerEmulator\">here</a>."}])

l1temulayout(dqmitems,"04-GCT (Dis)Agreement Flag",
             [{'path': "L1TEMU/GCT/GCTErrorFlag", 'description': "Data|Emulator disagreement type for GCT. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftTriggerEmulator\">here</a>."}])

l1temulayout(dqmitems,"05-DTTPG (Dis)Agreement Flag",
             [{'path': "L1TEMU/DTTPG/DTPErrorFlag", 'description': "Data|Emulator disagreement type for DT trigger primitives. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftTriggerEmulator\">here</a>."}])

l1temulayout(dqmitems,"06-DTTF (Dis)Agreement Flag",
             [{'path': "L1TEMU/DTTF/DTFErrorFlag", 'description': "Data|Emulator disagreement type for DT track finder. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftTriggerEmulator\">here</a>."}])

l1temulayout(dqmitems,"07-CSCTPG (Dis)Agreement Flag",
             [{'path': "L1TEMU/CSCTPG/CTPErrorFlag", 'description': "Data|Emulator disagreement type for CSC trigger primitives. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftTriggerEmulator\">here</a>."}])

l1temulayout(dqmitems,"08-CSCTF (Dis)Agreement Flag",
             [{'path': "L1TEMU/CSCTF/CTFErrorFlag", 'description': "Data|Emulator disagreement type for CSC track finder. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftTriggerEmulator\">here</a>."}])
 
l1temulayout(dqmitems,"09-RPC (Dis)Agreement Flag",
             [{'path': "L1TEMU/RPC/RPCErrorFlag", 'description': "Data|Emulator disagreement type for RPC. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftTriggerEmulator\">here</a>."}])

l1temulayout(dqmitems,"10-GMT (Dis)Agreement Flag",
             [{'path': "L1TEMU/GMT/GMTErrorFlag", 'description': "Data|Emulator disagreement type for GMT. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftTriggerEmulator\">here</a>."}])

l1temulayout(dqmitems,"11-GT (Dis)Agreement Flag",
             [{'path': "L1TEMU/GT/GLTErrorFlag", 'description': "Data|Emulator disagreement type for GT. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftTriggerEmulator\">here</a>."}])

