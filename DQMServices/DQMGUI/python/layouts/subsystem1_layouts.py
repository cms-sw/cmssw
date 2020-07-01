from ..layouts.layout_manager import register_layout

register_layout(source='Hcal/TPTask/EtEmul/TTSubdet/HBHE', destination='Hcal/Layouts/EtEmul/TP/TTSubdet/HBHE_changed_name', name='layout1')
register_layout(source='Hcal/EventInfo/reportSummaryMap', destination='Hcal/Layouts/reportSummaryMap_renamed')
register_layout(source='SiStrip/IsolatedBunches/MechanicalView/TEC/MINUS/wheel_1/Summary_ClusterStoNCorr__OnTrack__TEC__MINUS__wheel__1', destination='Hcal/Layouts/should_have_quality_test')
