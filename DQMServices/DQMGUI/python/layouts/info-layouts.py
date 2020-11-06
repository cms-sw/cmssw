from ..layouts.layout_manager import register_layout

register_layout(source='Info/EventInfo/reportSummaryMap', destination='Info/Layouts/1 - HV and Beam Status per LumiSection')
register_layout(source='Info/ProvInfo/Run Type', destination='Info/Layouts/2 - Run key set for DQM')
register_layout(source='Info/ProvInfo/hltKey', destination='Info/Layouts/3 - HLT menu used')
register_layout(source='Info/ProvInfo/CMSSW', destination='Info/Layouts/4 - Version of CMSSW used')
register_layout(source='Info/ProvInfo/Globaltag', destination='Info/Layouts/5 - Global Tag used')
