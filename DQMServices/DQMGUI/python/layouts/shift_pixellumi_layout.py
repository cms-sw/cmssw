from ..layouts.layout_manager import register_layout

register_layout(source='PixelLumi/PixelLumiDqmZeroBias/HFDeliveredToPXByLS', destination='00 Shift/PixelLumi/HFDeliveredToPXByLS', name='00 - HFDeliveredToPXByLS', overlay='')
register_layout(source='PixelLumi/PixelLumiDqmZeroBias/HFRecordedToPXByLS', destination='00 Shift/PixelLumi/HFRecordedToPXByLS', name='01 - HFRecordedToPXByLS', overlay='')
register_layout(source='PixelLumi/PixelLumiDqmZeroBias/totalPixelLumiByLS', destination='00 Shift/PixelLumi/totalPixelLumiByLS', name='02 - totalPixelLumiByLS', overlay='')
register_layout(source='PixelLumi/PixelLumiDqmZeroBias/PXLumiByBXsum', destination='00 Shift/PixelLumi/PXLumiByBXsum', name='03 - PXLumiByBXsum', overlay='')
register_layout(source='PixelLumi/HF/totalHFDeliveredLumiByLS', destination='00 Shift/PixelLumi/totalHFDeliveredLumiByLS', name='04 - totalHFDeliveredLumiByLS', overlay='')
register_layout(source='PixelLumi/HF/totalHFRecordedLumiByLS', destination='00 Shift/PixelLumi/totalHFRecordedLumiByLS', name='05 - totalHFRecordedLumiByLS', overlay='')
register_layout(source='PixelLumi/HF/HFRecordedToDeliveredRatioByLS', destination='00 Shift/PixelLumi/HFRecordedToDeliveredRatioByLS', name='06 - HFRecordedToDeliveredRatioByLS', overlay='')
