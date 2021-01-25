from .adapt_to_new_backend import *
dqmitems={}

def shiftcastorlayout(i, p, *rows):
    i["00 Shift/Castor/" + p] = rows

shiftcastorlayout(dqmitems, "01 - Map of frontend and readout errors",
           [{ 'path': "Castor/CastorDigiMonitor/CASTOR QIE_capID+er+dv",
             'description':"Frontend and readout errors"}]
           )

shiftcastorlayout(dqmitems, "02 - Channel-wise timing",
           [{ 'path': "Castor/CastorDigiMonitor/Castor cells avr digi(fC) per event Map TS vs Channel",
             'description':"Channel-wise timing"}]
           )
shiftcastorlayout(dqmitems, "02b - Channel-wise timing (rms)",
           [{ 'path': "Castor/CastorDigiMonitor/Castor cells avr digiRMS(fC) per event Map TS vs Channel",
             'description':"Channel-wise timing (rms)"}]
           )

shiftcastorlayout(dqmitems, "03 - CASTOR DeadChannelsMap",
           [{ 'path': "Castor/CastorDigiMonitor/CASTOR BadChannelsMap",
             'description':"CASTOR DeadChannelsMap"}]
           )

shiftcastorlayout(dqmitems, "04 - DigiSize",
           [{ 'path': "Castor/CastorDigiMonitor/DigiSize",
             'description':"CASTOR DigiSize",
		'draw': { 'ytype':'log' } }]
           )

shiftcastorlayout(dqmitems, "05 - CASTOR Tower Depth",
           [{ 'path': "Castor/CastorRecHitMonitor/CASTORTowerDepth",
             'description':"CASTORTowerDepth"}]
           )

shiftcastorlayout(dqmitems, "06 - Tower EM vs HAD",
           [{ 'path': "Castor/CastorRecHitMonitor/CASTORTowerEMvsEhad",
             'description':"CASTOR Tower EM vs Ehad",
 'draw': { 'xtype': 'log', 'ytype':'log', 'drawopts': "COLZ" } }]
           )


apply_dqm_items_to_new_back_end(dqmitems, __file__)
