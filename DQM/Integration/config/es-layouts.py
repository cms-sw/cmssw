def ecalpreshowerlayout(i, p, *rows): i["EcalPreshower/Layouts/" + p] = DQMItem(layout=rows)
def ecalpreshowershiftlayout(i, p, *rows): i["EcalPreshower/Layouts/00 Shift/" + p] = DQMItem(layout=rows)
def ecalpreshowerintegritylayout(i, p, *rows): i["EcalPreshower/Layouts/01 Preshower Shift/01 Integrity/" + p] = DQMItem(layout=rows)
def ecalpreshoweroccupancylayout(i, p, *rows): i["EcalPreshower/Layouts/01 Preshower Shift/02 Occupancy/" + p] = DQMItem(layout=rows)
def ecalpreshowerintegrityexpertlayout(i, p, *rows): i["EcalPreshower/Layouts/01 Preshower Expert/01 Integrity/" + p] = DQMItem(layout=rows)
def ecalpreshoweroccupancyexpertlayout(i, p, *rows): i["EcalPreshower/Layouts/01 Preshower Expert/02 Occupancy/" + p] = DQMItem(layout=rows)

# Quick Collections
ecalpreshowerlayout(dqmitems, "01-IntegritySummary-EcalPreshower",
  [{ 'path': "EcalPreshower/ESIntegrityClient/ES Integrity Summary 1 Z 1 P 1", 'description': "ES+ Front Integrity Summary 1 - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftPreshower>DQMShiftPreshower</a> <br/> <table width=100%> <tr><td>1 - not used<td>2 - fiber problem<td>3 - OK<td>4 - FED problem<td><tr>5 - KCHIP problem<td>6 - ES counters are not synced with GT counters (see ESRawDataTask) <td> 7 - more than one problem<td>8 - SLink CRC error</table> " },
   { 'path': "EcalPreshower/ESIntegrityClient/ES Integrity Summary 1 Z -1 P 1", 'description': "ES- Front Integrity Summary 1 - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftPreshower>DQMShiftPreshower</a> <br/> <table width=100%> <tr><td>1 - not used<td>2 - fiber problem<td>3 - OK<td>4 - FED problem<td><tr>5 - KCHIP problem<td>6 - ES counters are not synced with GT counters (see ESRawDataTask) <td> 7 - more than one problem<td>8 - SLink CRC error</table> " }],
  [{ 'path': "EcalPreshower/ESIntegrityClient/ES Integrity Summary 1 Z 1 P 2", 'description': "ES+ Rear Integrity Summary 1 - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftPreshower>DQMShiftPreshower</a> <br/> <table width=100%> <tr><td>1 - not used<td>2 - fiber problem<td>3 - OK<td>4 - FED problem<td><tr>5 - KCHIP problem<td>6 - ES counters are not synced with GT counters (see ESRawDataTask) <td> 7 - more than one problem<td>8 - SLink CRC error</table> " },
   { 'path': "EcalPreshower/ESIntegrityClient/ES Integrity Summary 1 Z -1 P 2", 'description': "ES- Rear Integrity Summary 1 - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftPreshower>DQMShiftPreshower</a> <br/> <table width=100%> <tr><td>1 - not used<td>2 - fiber problem<td>3 - OK<td>4 - FED problem<td><tr>5 - KCHIP problem<td>6 - ES counters are not synced with GT counters (see ESRawDataTask) <td> 7 - more than one problem<td>8 - SLink CRC error</table> " }])

ecalpreshowerlayout(dqmitems, "02-OccupancySummary-EcalPreshower",
  [{ 'path': "EcalPreshower/ESOccupancyTask/ES RecHit 2D Occupancy Z 1 P 1", 'description': "ES RecHit 2D Occupancy Z 1 P 1 - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftPreshower>DQMShiftPreshower</a> " },
   { 'path': "EcalPreshower/ESOccupancyTask/ES RecHit 2D Occupancy Z -1 P 1", 'description': "ES RecHit 2D Occupancy Z -1 P 1 - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftPreshower>DQMShiftPreshower</a> " }],
  [{ 'path': "EcalPreshower/ESOccupancyTask/ES RecHit 2D Occupancy Z 1 P 2", 'description': "ES RecHit 2D Occupancy Z 1 P 2 - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftPreshower>DQMShiftPreshower</a> " },
   { 'path': "EcalPreshower/ESOccupancyTask/ES RecHit 2D Occupancy Z -1 P 2", 'description': "ES RecHit 2D Occupancy Z -1 P 2 - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftPreshower>DQMShiftPreshower</a> " }])

ecalpreshowerlayout(dqmitems, "03-RechitEnergySummary-EcalPreshower",
  [{ 'path': "EcalPreshower/ESOccupancyTask/ES RecHit Energy Z 1 P 1", 'description': "ES RecHit Energy Z 1 P 1 - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftPreshower>DQMShiftPreshower</a> " },
   { 'path': "EcalPreshower/ESOccupancyTask/ES RecHit Energy Z -1 P 1", 'description': "ES RecHit Energy Z -1 P 1 - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftPreshower>DQMShiftPreshower</a> " }],
  [{ 'path': "EcalPreshower/ESOccupancyTask/ES RecHit Energy Z 1 P 2", 'description': "ES RecHit Energy Z 1 P 2 - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftPreshower>DQMShiftPreshower</a> " },
   { 'path': "EcalPreshower/ESOccupancyTask/ES RecHit Energy Z -1 P 2", 'description': "ES RecHit Energy Z -1 P 2 - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftPreshower>DQMShiftPreshower</a> " }])

ecalpreshowerlayout(dqmitems, "04-ESTimingTaskSummary-EcalPreshower",
  [{ 'path': "EcalPreshower/ESTimingTask/ES Timing Z 1 P 1", 'description': "ES Timing Z 1 P 1 - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftPreshower>DQMShiftPreshower</a> " },
   { 'path': "EcalPreshower/ESTimingTask/ES Timing Z -1 P 1", 'description': "ES Timing Z -1 P 1 - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftPreshower>DQMShiftPreshower</a> " }],
  [{ 'path': "EcalPreshower/ESTimingTask/ES Timing Z 1 P 2", 'description': "ES Timing Z 1 P 2 - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftPreshower>DQMShiftPreshower</a> " },
   { 'path': "EcalPreshower/ESTimingTask/ES Timing Z -1 P 2", 'description': "ES Timing Z -1 P 2 - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftPreshower>DQMShiftPreshower</a> " }])

# Layouts
ecalpreshowershiftlayout(dqmitems, "01-IntegritySummary-EcalPreshower",
  [{ 'path': "EcalPreshower/ESIntegrityClient/ES Integrity Summary 1 Z 1 P 1", 'description': "ES+ Front Integrity Summary 1 - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftPreshower>DQMShiftPreshower</a> <br/> <table width=100%> <tr><td>1 - not used<td>2 - fiber problem<td>3 - OK<td>4 - FED problem<td><tr>5 - KCHIP problem<td>6 - ES counters are not synced with GT counters (see ESRawDataTask) <td> 7 - more than one problem<td>8 - SLink CRC error</table> " },
   { 'path': "EcalPreshower/ESIntegrityClient/ES Integrity Summary 1 Z -1 P 1", 'description': "ES- Front Integrity Summary 1 - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftPreshower>DQMShiftPreshower</a> <br/> <table width=100%> <tr><td>1 - not used<td>2 - fiber problem<td>3 - OK<td>4 - FED problem<td><tr>5 - KCHIP problem<td>6 - ES counters are not synced with GT counters (see ESRawDataTask) <td> 7 - more than one problem<td>8 - SLink CRC error</table> " }],
  [{ 'path': "EcalPreshower/ESIntegrityClient/ES Integrity Summary 1 Z 1 P 2", 'description': "ES+ Rear Integrity Summary 1 - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftPreshower>DQMShiftPreshower</a> <br/> <table width=100%> <tr><td>1 - not used<td>2 - fiber problem<td>3 - OK<td>4 - FED problem<td><tr>5 - KCHIP problem<td>6 - ES counters are not synced with GT counters (see ESRawDataTask) <td> 7 - more than one problem<td>8 - SLink CRC error</table> " },
   { 'path': "EcalPreshower/ESIntegrityClient/ES Integrity Summary 1 Z -1 P 2", 'description': "ES- Rear Integrity Summary 1 - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftPreshower>DQMShiftPreshower</a> <br/> <table width=100%> <tr><td>1 - not used<td>2 - fiber problem<td>3 - OK<td>4 - FED problem<td><tr>5 - KCHIP problem<td>6 - ES counters are not synced with GT counters (see ESRawDataTask) <td> 7 - more than one problem<td>8 - SLink CRC error</table> " }])

ecalpreshowershiftlayout(dqmitems, "02-OccupancySummary-EcalPreshower",
  [{ 'path': "EcalPreshower/ESOccupancyTask/ES RecHit 2D Occupancy Z 1 P 1", 'description': "ES RecHit 2D Occupancy Z 1 P 1 - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftPreshower>DQMShiftPreshower</a> " },
   { 'path': "EcalPreshower/ESOccupancyTask/ES RecHit 2D Occupancy Z -1 P 1", 'description': "ES RecHit 2D Occupancy Z -1 P 1 - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftPreshower>DQMShiftPreshower</a> " }],
  [{ 'path': "EcalPreshower/ESOccupancyTask/ES RecHit 2D Occupancy Z 1 P 2", 'description': "ES RecHit 2D Occupancy Z 1 P 2 - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftPreshower>DQMShiftPreshower</a> " },
   { 'path': "EcalPreshower/ESOccupancyTask/ES RecHit 2D Occupancy Z -1 P 2", 'description': "ES RecHit 2D Occupancy Z -1 P 2 - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftPreshower>DQMShiftPreshower</a> " }])

ecalpreshowershiftlayout(dqmitems, "03-RechitEnergySummary-EcalPreshower",
  [{ 'path': "EcalPreshower/ESOccupancyTask/ES RecHit Energy Z 1 P 1", 'description': "ES RecHit Energy Z 1 P 1 - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftPreshower>DQMShiftPreshower</a> " },
   { 'path': "EcalPreshower/ESOccupancyTask/ES RecHit Energy Z -1 P 1", 'description': "ES RecHit Energy Z -1 P 1 - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftPreshower>DQMShiftPreshower</a> " }],
  [{ 'path': "EcalPreshower/ESOccupancyTask/ES RecHit Energy Z 1 P 2", 'description': "ES RecHit Energy Z 1 P 2 - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftPreshower>DQMShiftPreshower</a> " },
   { 'path': "EcalPreshower/ESOccupancyTask/ES RecHit Energy Z -1 P 2", 'description': "ES RecHit Energy Z -1 P 2 - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftPreshower>DQMShiftPreshower</a> " }])

ecalpreshowershiftlayout(dqmitems, "04-ESTimingTaskSummary-EcalPreshower",
  [{ 'path': "EcalPreshower/ESTimingTask/ES Timing Z 1 P 1", 'description': "ES Timing Z 1 P 1 - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftPreshower>DQMShiftPreshower</a> " },
   { 'path': "EcalPreshower/ESTimingTask/ES Timing Z -1 P 1", 'description': "ES Timing Z -1 P 1 - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftPreshower>DQMShiftPreshower</a> " }],
  [{ 'path': "EcalPreshower/ESTimingTask/ES Timing Z 1 P 2", 'description': "ES Timing Z 1 P 2 - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftPreshower>DQMShiftPreshower</a> " },
   { 'path': "EcalPreshower/ESTimingTask/ES Timing Z -1 P 2", 'description': "ES Timing Z -1 P 2 - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftPreshower>DQMShiftPreshower</a> " }])

ecalpreshowerintegritylayout(dqmitems, "01 Integrity Summary 1 Z 1 P 1",
   [{'path': "EcalPreshower/ESIntegrityClient/ES Integrity Summary 1 Z 1 P 1", 'description': "ES+ Front Integrity Summary 1 - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftPreshower>DQMShiftPreshower</a> <br/> <table width=100%> <tr><td>1 - not used<td>2 - fiber problem<td>3 - OK<td>4 - FED problem<td><tr>5 - KCHIP problem<td>6 - ES counters are not synced with GT counters (see ESRawDataTask) <td> 7 - more than one problem<td>8 - SLink CRC error</table> " }])
ecalpreshowerintegritylayout(dqmitems, "02 Integrity Summary 1 Z -1 P 1",
   [{'path': "EcalPreshower/ESIntegrityClient/ES Integrity Summary 1 Z -1 P 1", 'description': "ES- Front Integrity Summary 1 - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftPreshower>DQMShiftPreshower</a> <br/> <table width=100%> <tr><td>1 - not used<td>2 - fiber problem<td>3 - OK<td>4 - FED problem<td><tr>5 - KCHIP problem<td>6 - ES counters are not synced with GT counters (see ESRawDataTask) <td> 7 - more than one problem<td>8 - SLink CRC error</table> " }])
ecalpreshowerintegritylayout(dqmitems, "03 Integrity Summary 1 Z 1 P 2",
   [{'path': "EcalPreshower/ESIntegrityClient/ES Integrity Summary 1 Z 1 P 2", 'description': "ES+ Rear Integrity Summary 1 - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftPreshower>DQMShiftPreshower</a> <br/> <table width=100%> <tr><td>1 - not used<td>2 - fiber problem<td>3 - OK<td>4 - FED problem<td><tr>5 - KCHIP problem<td>6 - ES counters are not synced with GT counters (see ESRawDataTask) <td> 7 - more than one problem<td>8 - SLink CRC error</table> " }])
ecalpreshowerintegritylayout(dqmitems, "04 Integrity Summary 1 Z -1 P 2",
   [{'path': "EcalPreshower/ESIntegrityClient/ES Integrity Summary 1 Z -1 P 2", 'description': "ES- Rear Integrity Summary 1 - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftPreshower>DQMShiftPreshower</a> <br/> <table width=100%> <tr><td>1 - not used<td>2 - fiber problem<td>3 - OK<td>4 - FED problem<td><tr>5 - KCHIP problem<td>6 - ES counters are not synced with GT counters (see ESRawDataTask) <td> 7 - more than one problem<td>8 - SLink CRC error</table> " }])
 
ecalpreshoweroccupancylayout(dqmitems, "01 ES RecHit 2D Occupancy Z 1 P 1",
  [{ 'path': "EcalPreshower/ESOccupancyTask/ES RecHit 2D Occupancy Z 1 P 1", 'description': "ES RecHit 2D Occupancy Z 1 P 1 - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftPreshower>DQMShiftPreshower</a> " }])
ecalpreshoweroccupancylayout(dqmitems, "02 ES RecHit 2D Occupancy Z -1 P 1",
  [{ 'path': "EcalPreshower/ESOccupancyTask/ES RecHit 2D Occupancy Z -1 P 1", 'description': "ES RecHit 2D Occupancy Z -1 P 1 - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftPreshower>DQMShiftPreshower</a> " }])
ecalpreshoweroccupancylayout(dqmitems, "03 ES RecHit 2D Occupancy Z 1 P 2",
  [{ 'path': "EcalPreshower/ESOccupancyTask/ES RecHit 2D Occupancy Z 1 P 2", 'description': "ES RecHit 2D Occupancy Z 1 P 2 - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftPreshower>DQMShiftPreshower</a> " }])
ecalpreshoweroccupancylayout(dqmitems, "04 ES RecHit 2D Occupancy Z -1 P 2",
  [{ 'path': "EcalPreshower/ESOccupancyTask/ES RecHit 2D Occupancy Z -1 P 2", 'description': "ES RecHit 2D Occupancy Z -1 P 2 - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftPreshower>DQMShiftPreshower</a> " }])
ecalpreshoweroccupancylayout(dqmitems, "05 ES RecHit Energy Z 1 P 1",
  [{ 'path': "EcalPreshower/ESOccupancyTask/ES RecHit Energy Z 1 P 1", 'description': "ES RecHit Energy Z 1 P 1 - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftPreshower>DQMShiftPreshower</a> " }])
ecalpreshoweroccupancylayout(dqmitems, "06 ES RecHit Energy Z -1 P 1",
  [{ 'path': "EcalPreshower/ESOccupancyTask/ES RecHit Energy Z -1 P 1", 'description': "ES RecHit Energy Z -1 P 1 - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftPreshower>DQMShiftPreshower</a> " }])
ecalpreshoweroccupancylayout(dqmitems, "07 ES RecHit Energy Z 1 P 2",
  [{ 'path': "EcalPreshower/ESOccupancyTask/ES RecHit Energy Z 1 P 2", 'description': "ES RecHit Energy Z 1 P 2 - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftPreshower>DQMShiftPreshower</a> " }])
ecalpreshoweroccupancylayout(dqmitems, "08 ES RecHit Energy Z -1 P 2",
  [{ 'path': "EcalPreshower/ESOccupancyTask/ES RecHit Energy Z -1 P 2", 'description': "ES RecHit Energy Z -1 P 2 - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftPreshower>DQMShiftPreshower</a> " }])
  
ecalpreshowerintegrityexpertlayout(dqmitems, "01 Integrity Summary 1 Z 1 P 1",
   [{'path': "EcalPreshower/ESIntegrityClient/ES Integrity Summary 1 Z 1 P 1", 'description': "ES+ Front Integrity Summary 1 - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftPreshower>DQMShiftPreshower</a> <br/> <table width=100%> <tr><td>1 - not used<td>2 - fiber problem<td>3 - OK<td>4 - FED problem<td><tr>5 - KCHIP problem<td>6 - ES counters are not synced with GT counters (see ESRawDataTask) <td> 7 - more than one problem<td>8 - SLink CRC error</table> " }])
ecalpreshowerintegrityexpertlayout(dqmitems, "02 Integrity Summary 1 Z -1 P 1",
   [{'path': "EcalPreshower/ESIntegrityClient/ES Integrity Summary 1 Z -1 P 1", 'description': "ES- Front Integrity Summary 1 - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftPreshower>DQMShiftPreshower</a> <br/> <table width=100%> <tr><td>1 - not used<td>2 - fiber problem<td>3 - OK<td>4 - FED problem<td><tr>5 - KCHIP problem<td>6 - ES counters are not synced with GT counters (see ESRawDataTask) <td> 7 - more than one problem<td>8 - SLink CRC error</table> " }])
ecalpreshowerintegrityexpertlayout(dqmitems, "03 Integrity Summary 1 Z 1 P 2",
   [{'path': "EcalPreshower/ESIntegrityClient/ES Integrity Summary 1 Z 1 P 2", 'description': "ES+ Rear Integrity Summary 1 - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftPreshower>DQMShiftPreshower</a> <br/> <table width=100%> <tr><td>1 - not used<td>2 - fiber problem<td>3 - OK<td>4 - FED problem<td><tr>5 - KCHIP problem<td>6 - ES counters are not synced with GT counters (see ESRawDataTask) <td> 7 - more than one problem<td>8 - SLink CRC error</table> " }])
ecalpreshowerintegrityexpertlayout(dqmitems, "04 Integrity Summary 1 Z -1 P 2",
   [{'path': "EcalPreshower/ESIntegrityClient/ES Integrity Summary 1 Z -1 P 2", 'description': "ES- Rear Integrity Summary 1 - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftPreshower>DQMShiftPreshower</a> <br/> <table width=100%> <tr><td>1 - not used<td>2 - fiber problem<td>3 - OK<td>4 - FED problem<td><tr>5 - KCHIP problem<td>6 - ES counters are not synced with GT counters (see ESRawDataTask) <td> 7 - more than one problem<td>8 - SLink CRC error</table> " }])
 
ecalpreshoweroccupancyexpertlayout(dqmitems, "01 ES RecHit 2D Occupancy Z 1 P 1",
  [{ 'path': "EcalPreshower/ESOccupancyTask/ES RecHit 2D Occupancy Z 1 P 1", 'description': "ES RecHit 2D Occupancy Z 1 P 1 - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftPreshower>DQMShiftPreshower</a> " }])
ecalpreshoweroccupancyexpertlayout(dqmitems, "02 ES RecHit 2D Occupancy Z -1 P 1",
  [{ 'path': "EcalPreshower/ESOccupancyTask/ES RecHit 2D Occupancy Z -1 P 1", 'description': "ES RecHit 2D Occupancy Z -1 P 1 - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftPreshower>DQMShiftPreshower</a> " }])
ecalpreshoweroccupancyexpertlayout(dqmitems, "03 ES RecHit 2D Occupancy Z 1 P 2",
  [{ 'path': "EcalPreshower/ESOccupancyTask/ES RecHit 2D Occupancy Z 1 P 2", 'description': "ES RecHit 2D Occupancy Z 1 P 2 - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftPreshower>DQMShiftPreshower</a> " }])
ecalpreshoweroccupancyexpertlayout(dqmitems, "04 ES RecHit 2D Occupancy Z -1 P 2",
  [{ 'path': "EcalPreshower/ESOccupancyTask/ES RecHit 2D Occupancy Z -1 P 2", 'description': "ES RecHit 2D Occupancy Z -1 P 2 - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftPreshower>DQMShiftPreshower</a> " }])
ecalpreshoweroccupancyexpertlayout(dqmitems, "05 ES RecHit Energy Z 1 P 1",
  [{ 'path': "EcalPreshower/ESOccupancyTask/ES RecHit Energy Z 1 P 1", 'description': "ES RecHit Energy Z 1 P 1 - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftPreshower>DQMShiftPreshower</a> " }])
ecalpreshoweroccupancyexpertlayout(dqmitems, "06 ES RecHit Energy Z -1 P 1",
  [{ 'path': "EcalPreshower/ESOccupancyTask/ES RecHit Energy Z -1 P 1", 'description': "ES RecHit Energy Z -1 P 1 - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftPreshower>DQMShiftPreshower</a> " }])
ecalpreshoweroccupancyexpertlayout(dqmitems, "07 ES RecHit Energy Z 1 P 2",
  [{ 'path': "EcalPreshower/ESOccupancyTask/ES RecHit Energy Z 1 P 2", 'description': "ES RecHit Energy Z 1 P 2 - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftPreshower>DQMShiftPreshower</a> " }])
ecalpreshoweroccupancyexpertlayout(dqmitems, "08 ES RecHit Energy Z -1 P 2",
  [{ 'path': "EcalPreshower/ESOccupancyTask/ES RecHit Energy Z -1 P 2", 'description': "ES RecHit Energy Z -1 P 2 - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftPreshower>DQMShiftPreshower</a> " }])
  
