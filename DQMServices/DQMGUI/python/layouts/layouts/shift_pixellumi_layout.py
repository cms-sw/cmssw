from .adapt_to_new_backend import *
dqmitems={}

def shiftpixellumilayout(i, p, *rows): i["00 Shift/PixelLumi/" + p] = rows

shiftpixellumilayout(dqmitems, "00 - HFDeliveredToPXByLS",
                     [{ 'path': "PixelLumi/PixelLumiDqmZeroBias/HFDeliveredToPXByLS",
                        'description':""}]
                     )

shiftpixellumilayout(dqmitems, "01 - HFRecordedToPXByLS",
                     [{ 'path': "PixelLumi/PixelLumiDqmZeroBias/HFRecordedToPXByLS",
                        'description':""}]
                     )

shiftpixellumilayout(dqmitems, "02 - totalPixelLumiByLS",
                     [{ 'path': "PixelLumi/PixelLumiDqmZeroBias/totalPixelLumiByLS",
                        'description':""}]
                     )

shiftpixellumilayout(dqmitems, "03 - PXLumiByBXsum",
                     [{ 'path': "PixelLumi/PixelLumiDqmZeroBias/PXLumiByBXsum",
                        'description':""}]
                     )

shiftpixellumilayout(dqmitems, "04 - totalHFDeliveredLumiByLS",
                     [{ 'path': "PixelLumi/HF/totalHFDeliveredLumiByLS",
                        'description':""}]
                     )

shiftpixellumilayout(dqmitems, "05 - totalHFRecordedLumiByLS",
                     [{ 'path': "PixelLumi/HF/totalHFRecordedLumiByLS",
                        'description':""}]
                     )

shiftpixellumilayout(dqmitems, "06 - HFRecordedToDeliveredRatioByLS",
                     [{ 'path': "PixelLumi/HF/HFRecordedToDeliveredRatioByLS",
                        'description':""}]
                     )

apply_dqm_items_to_new_back_end(dqmitems, __file__)
