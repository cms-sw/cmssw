def shifteslayout(i, p, *rows): i["00 Shift/EcalPreshower/" + p] = DQMItem(layout=rows)

shifteslayout(dqmitems, "01-IntegritySummary",
  [{ 'path': "EcalPreshower/ESIntegrityClient/ES Integrity Summary 1 Z 1 P 1", 'description': "ES+ Front Integrity Summary 1 - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftPreshower>DQMShiftPreshower</a> <br/> <table width=100%> <tr><td>1 - not used<td>2 - fiber problem<td>3 - OK<td>4 - FED problem<td><tr>5 - KCHIP problem<td>6 - ES counters are not synced with GT counters (see ESRawDataTask) <td> 7 - more than one problem<td>8 - SLink CRC error</table> " },
   { 'path': "EcalPreshower/ESIntegrityClient/ES Integrity Summary 1 Z -1 P 1", 'description': "ES- Front Integrity Summary 1 - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftPreshower>DQMShiftPreshower</a> <br/> <table width=100%> <tr><td>1 - not used<td>2 - fiber problem<td>3 - OK<td>4 - FED problem<td><tr>5 - KCHIP problem<td>6 - ES counters are not synced with GT counters (see ESRawDataTask) <td> 7 - more than one problem<td>8 - SLink CRC error</table> " }],
  [{ 'path': "EcalPreshower/ESIntegrityClient/ES Integrity Summary 1 Z 1 P 2", 'description': "ES+ Rear Integrity Summary 1 - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftPreshower>DQMShiftPreshower</a> <br/> <table width=100%> <tr><td>1 - not used<td>2 - fiber problem<td>3 - OK<td>4 - FED problem<td><tr>5 - KCHIP problem<td>6 - ES counters are not synced with GT counters (see ESRawDataTask) <td> 7 - more than one problem<td>8 - SLink CRC error</table> " },
   { 'path': "EcalPreshower/ESIntegrityClient/ES Integrity Summary 1 Z -1 P 2", 'description': "ES- Rear Integrity Summary 1 - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftPreshower>DQMShiftPreshower</a> <br/> <table width=100%> <tr><td>1 - not used<td>2 - fiber problem<td>3 - OK<td>4 - FED problem<td><tr>5 - KCHIP problem<td>6 - ES counters are not synced with GT counters (see ESRawDataTask) <td> 7 - more than one problem<td>8 - SLink CRC error</table> " }])

shifteslayout(dqmitems, "02-OccupancySummary",
  [{ 'path': "EcalPreshower/ESOccupancyTask/ES RecHit 2D Occupancy Z 1 P 1", 'description': "ES RecHit 2D Occupancy Z 1 P 1 - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftPreshower>DQMShiftPreshower</a> " },
   { 'path': "EcalPreshower/ESOccupancyTask/ES RecHit 2D Occupancy Z -1 P 1", 'description': "ES RecHit 2D Occupancy Z -1 P 1 - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftPreshower>DQMShiftPreshower</a> " }],
  [{ 'path': "EcalPreshower/ESOccupancyTask/ES RecHit 2D Occupancy Z 1 P 2", 'description': "ES RecHit 2D Occupancy Z 1 P 2 - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftPreshower>DQMShiftPreshower</a> " },
   { 'path': "EcalPreshower/ESOccupancyTask/ES RecHit 2D Occupancy Z -1 P 2", 'description': "ES RecHit 2D Occupancy Z -1 P 2 - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftPreshower>DQMShiftPreshower</a> " }])

shifteslayout(dqmitems, "03-RechitEnergySummary",
  [{ 'path': "EcalPreshower/ESOccupancyTask/ES RecHit Energy Z 1 P 1", 'description': "ES RecHit Energy Z 1 P 1 - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftPreshower>DQMShiftPreshower</a> " },
   { 'path': "EcalPreshower/ESOccupancyTask/ES RecHit Energy Z -1 P 1", 'description': "ES RecHit Energy Z -1 P 1 - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftPreshower>DQMShiftPreshower</a> " }],
  [{ 'path': "EcalPreshower/ESOccupancyTask/ES RecHit Energy Z 1 P 2", 'description': "ES RecHit Energy Z 1 P 2 - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftPreshower>DQMShiftPreshower</a> " },
   { 'path': "EcalPreshower/ESOccupancyTask/ES RecHit Energy Z -1 P 2", 'description': "ES RecHit Energy Z -1 P 2 - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftPreshower>DQMShiftPreshower</a> " }])

shifteslayout(dqmitems, "04-ESTimingTaskSummary-EcalPreshower",
  [{ 'path': "EcalPreshower/ESTimingTask/ES Timing Z 1 P 1", 'description': "ES Timing Z 1 P 1 - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftPreshower>DQMShiftPreshower</a> " },
   { 'path': "EcalPreshower/ESTimingTask/ES Timing Z -1 P 1", 'description': "ES Timing Z -1 P 1 - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftPreshower>DQMShiftPreshower</a> " }],
  [{ 'path': "EcalPreshower/ESTimingTask/ES Timing Z 1 P 2", 'description': "ES Timing Z 1 P 2 - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftPreshower>DQMShiftPreshower</a> " },
   { 'path': "EcalPreshower/ESTimingTask/ES Timing Z -1 P 2", 'description': "ES Timing Z -1 P 2 - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftPreshower>DQMShiftPreshower</a> " }])
