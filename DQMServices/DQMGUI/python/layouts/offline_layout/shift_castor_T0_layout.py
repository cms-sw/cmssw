from ..layouts.layout_manager import register_layout

register_layout(source='Castor/CastorDigiMonitor/CASTOR QIE_capID+er+dv', destination='00 Shift/Castor/CASTOR QIE_capID+er+dv', name='01 - Map of frontend and readout errors', overlay='')
register_layout(source='Castor/CastorDigiMonitor/QfC=f(x=Tile y=TS) (cumulative)', destination='00 Shift/Castor/QfC=f(x=Tile y=TS) (cumulative)', name='02 - Channel-wise timing', overlay='')
register_layout(source='Castor/CastorDigiMonitor/QrmsfC=f(Tile TS)', destination='00 Shift/Castor/QrmsfC=f(Tile TS)', name='02b - Channel-wise timing (rms)', overlay='')
register_layout(source='Castor/CastorDigiMonitor/CASTOR DeadChannelsMap', destination='00 Shift/Castor/CASTOR DeadChannelsMap', name='03 - CASTOR DeadChannelsMap', overlay='')
register_layout(source='Castor/CastorDigiMonitor/DigiSize', destination='00 Shift/Castor/DigiSize', name='04 - DigiSize', overlay='')
register_layout(source='Castor/CastorRecHitMonitor/CASTORTowerDepth', destination='00 Shift/Castor/CASTORTowerDepth', name='05 - CASTOR Tower Depth', overlay='')
register_layout(source='Castor/CastorRecHitMonitor/CASTORTowerEMvsEhad', destination='00 Shift/Castor/CASTORTowerEMvsEhad', name='06 - Tower EM vs HAD', overlay='')
