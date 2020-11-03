from ..layouts.layout_manager import register_layout

register_layout(source='SiStripLAS/EventInfo/reportSummaryMap', destination='SiStripLAS/Layouts/reportSummaryMap', name='00 - SiStripLAS ReportSummary', description='NumberOfSignals_AlignmentTubes ', overlay='')
register_layout(source='SiStripLAS/NumberOfSignals_AlignmentTubes', destination='SiStripLAS/Layouts/NumberOfSignals_AlignmentTubes', name='01 - SiStripLAS TIB&TOB', description='NumberOfSignals_AlignmentTubes ', overlay='')
register_layout(source='SiStripLAS/NumberOfSignals_TEC+R4', destination='SiStripLAS/Layouts/NumberOfSignals_TEC+R4', name='02 - SiStripLAS TEC+', description='NumberOfSignals_TEC+R4 ', overlay='')
register_layout(source='SiStripLAS/NumberOfSignals_TEC+R6', destination='SiStripLAS/Layouts/NumberOfSignals_TEC+R6', name='02 - SiStripLAS TEC+', description='NumberOfSignals_TEC+R6 ', overlay='')
register_layout(source='SiStripLAS/NumberOfSignals_TEC-R4', destination='SiStripLAS/Layouts/NumberOfSignals_TEC-R4', name='03 - SiStripLAS TEC-', description='NumberOfSignals_TEC-R4 ', overlay='')
register_layout(source='SiStripLAS/NumberOfSignals_TEC-R6', destination='SiStripLAS/Layouts/NumberOfSignals_TEC-R6', name='03 - SiStripLAS TEC-', description='NumberOfSignals_TEC-R6 ', overlay='')
