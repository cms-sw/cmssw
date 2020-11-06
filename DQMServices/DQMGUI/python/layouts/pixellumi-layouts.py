from ..layouts.layout_manager import register_layout

register_layout(source='PixelLumi/PixelLumiDqmZeroBias/totalPixelLumiByLS', destination='PixelLumi/Layouts/totalPixelLumiByLS', name='01_PXLumi_AlcaLumiZeroBias', description='', overlay='')
register_layout(source='PixelLumi/PixelLumiDqmZeroBias/PXLumiByBXsum', destination='PixelLumi/Layouts/PixelLumi/Layouts/totalPixelLumiByLSPXLumiByBXsum', name='01_PXLumi_AlcaLumiZeroBias', description='', overlay='')
register_layout(source='PixelLumi/HF/totalHFDeliveredLumiByLS', destination='PixelLumi/Layouts/totalHFDeliveredLumiByLS', name='02_HFLumi_RunTimeLogger', description='', overlay='')
register_layout(source='PixelLumi/HF/totalHFRecordedLumiByLS', destination='PixelLumi/Layouts/PixelLumi/Layouts/totalHFDeliveredLumiByLStotalHFRecordedLumiByLS', name='02_HFLumi_RunTimeLogger', description='', overlay='')
register_layout(source='PixelLumi/HF/HFRecordedToDeliveredRatioByLS', destination='PixelLumi/Layouts/PixelLumi/Layouts/PixelLumi/Layouts/totalHFDeliveredLumiByLStotalHFRecordedLumiByLSHFRecordedToDeliveredRatioByLS', name='02_HFLumi_RunTimeLogger', description='', overlay='')
register_layout(source='PixelLumi/PixelLumiDqmZeroBias/HFDeliveredToPXByLS', destination='PixelLumi/Layouts/HFDeliveredToPXByLS', name='03_Ratio_HFtoPX', description='', overlay='')
register_layout(source='PixelLumi/PixelLumiDqmZeroBias/HFRecordedToPXByLS', destination='PixelLumi/Layouts/PixelLumi/Layouts/HFDeliveredToPXByLSHFRecordedToPXByLS', name='03_Ratio_HFtoPX', description='', overlay='')
