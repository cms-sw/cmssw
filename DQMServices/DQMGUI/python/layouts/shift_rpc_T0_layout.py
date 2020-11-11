from ..layouts.layout_manager import register_layout

register_layout(source='RPC/EventInfo/reportSummaryMap', destination='00 Shift/RPC/reportSummaryMap', name='00-Summary_Map', overlay='')
register_layout(source='RPC/FEDIntegrity/FEDFatal', destination='00 Shift/RPC/FEDFatal', name='01-FED_Fatal_Errors', overlay='')
register_layout(source='RPC/DCSInfo/rpcHVStatus', destination='00 Shift/RPC/rpcHVStatus', name='02-RPC_HV_Status', overlay='')
register_layout(source='RPC/AllHits/RPCEvents', destination='00 Shift/RPC/RPCEvents', name='03-RPC_Events', overlay='')
register_layout(source='RPC/AllHits/SummaryHistograms/RPC_System_Quality_Overview', destination='00 Shift/RPC/RPC_System_Quality_Overview', name='04-Quality_State_Overview', overlay='')
register_layout(source='RPC/AllHits/SummaryHistograms/Occupancy_for_Barrel', destination='00 Shift/RPC/Occupancy_for_Barrel', name='05-RPC_Occupancy', overlay='')
register_layout(source='RPC/AllHits/SummaryHistograms/Occupancy_for_Endcap', destination='00 Shift/RPC/Occupancy_for_Endcap', name='05-RPC_Occupancy', overlay='')
register_layout(source='RPC/RPCEfficiency/Statistics', destination='00 Shift/RPC/Statistics', name='06-Statistics', overlay='')
register_layout(source='RPC/RPCEfficiency/EffBarrelRoll', destination='00 Shift/RPC/EffBarrelRoll', name='07-Efficiency_Distribution', overlay='')
register_layout(source='RPC/RPCEfficiency/EffEndcapPlusRoll', destination='00 Shift/RPC/EffEndcapPlusRoll', name='07-Efficiency_Distribution', overlay='')
register_layout(source='RPC/RPCEfficiency/EffEndcapMinusRoll', destination='00 Shift/RPC/EffEndcapMinusRoll', name='07-Efficiency_Distribution', overlay='')
