from ..layouts.layout_manager import register_layout

register_layout(source='Info/EventInfo/reportSummaryMap', destination='Info/Layouts/1 - High Voltage (HV) per LumiSection')
register_layout(source='Info/EventInfo/ProcessedLS', destination='Info/Layouts/2 - Processed LumiSections')
register_layout(source='Info/ProvInfo/runIsComplete', destination='Info/Layouts/3 - Run is completely processed')
register_layout(source='Info/ProvInfo/CMSSW', destination='Info/Layouts/4 - Version of CMSSW used')
register_layout(source='Info/CMSSWInfo/globalTag_Step1', destination='Info/Layouts/5 - Global Tag used for filling')
register_layout(source='Info/CMSSWInfo/globalTag_Harvesting', destination='Info/Layouts/6 - Global Tag used for harvesting')
