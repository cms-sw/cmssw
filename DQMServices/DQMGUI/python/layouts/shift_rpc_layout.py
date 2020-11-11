from ..layouts.layout_manager import register_layout

register_layout(source='RPC/EventInfo/reportSummaryMap', destination='00 Shift/RPC/reportSummaryMap', name='00-Summary_Map', overlay='')
register_layout(source='RPC/FEDIntegrity_EvF/FEDFatal', destination='00 Shift/RPC/FEDFatal', name='01-Fatal_FED_Errors', overlay='')
register_layout(source='RPC/AllHits/RPCEvents', destination='00 Shift/RPC/RPCEvents', name='02-RPC_Events', overlay='')
register_layout(source='RPC/AllHits/SummaryHistograms/RPC_System_Quality_Overview', destination='00 Shift/RPC/RPC_System_Quality_Overview', name='03-Quality_State_Overview', overlay='')
register_layout(source='RPC/AllHits/SummaryHistograms/Occupancy_for_Barrel', destination='00 Shift/RPC/Occupancy_for_Barrel', name='04-RPC_Occupancy', overlay='')
register_layout(source='RPC/AllHits/SummaryHistograms/Occupancy_for_Endcap', destination='00 Shift/RPC/Occupancy_for_Endcap', name='04-RPC_Occupancy', overlay='')
