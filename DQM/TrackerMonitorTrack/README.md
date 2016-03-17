### Purpose

This code generates the `HitResiduals` and `NormalizedHitResiduals` histograms.

### Data displayed

The data shown in the histograms is computed by the Alignment/OfflineValidation code. It is computed for SiPixel and SiStrip in x- and y-direction (resXprime, resYprime values), however for the strips only the x-residual is relevant.

### Options

* `MonitorTrackResiduals.Mod_On` setting this to `True` will generate plots for every module. By default, only one plot per layer and wheel/disk is generated.

### Bugs

* The plots for pixel disks appear hidden in the shell and half cylinder folders, even though they cover the full layer/disk.

### Currently generated histograms

* `Pixel/Barrel/Shell_mI/Layer_1/HitResiduals_BPIX__Layer__1`
* `Pixel/Barrel/Shell_mI/Layer_1/HitResiduals_BPIX__Layer__1_Y`
* `Pixel/Barrel/Shell_mI/Layer_2/HitResiduals_BPIX__Layer__2`
* `Pixel/Barrel/Shell_mI/Layer_2/HitResiduals_BPIX__Layer__2_Y`
* `Pixel/Barrel/Shell_mI/Layer_3/HitResiduals_BPIX__Layer__3`
* `Pixel/Barrel/Shell_mI/Layer_3/HitResiduals_BPIX__Layer__3_Y`
* `Pixel/Endcap/HalfCylinder_pI/Disk_1/HitResiduals_FPIX__wheel__1`
* `Pixel/Endcap/HalfCylinder_pI/Disk_1/HitResiduals_FPIX__wheel__1_Y`
* `Pixel/Endcap/HalfCylinder_pI/Disk_2/HitResiduals_FPIX__wheel__2`
* `Pixel/Endcap/HalfCylinder_pI/Disk_2/HitResiduals_FPIX__wheel__2_Y`
* `Pixel/Endcap/HalfCylinder_mI/Disk_1/HitResiduals_FPIX__wheel__1`
* `Pixel/Endcap/HalfCylinder_mI/Disk_1/HitResiduals_FPIX__wheel__1_Y`
* `Pixel/Endcap/HalfCylinder_mI/Disk_2/HitResiduals_FPIX__wheel__2`
* `Pixel/Endcap/HalfCylinder_mI/Disk_2/HitResiduals_FPIX__wheel__2_Y`
* `SiStrip/MechanicalView/TIB/layer_1/HitResiduals_TIB__Layer__1`
* `SiStrip/MechanicalView/TIB/layer_1/HitResiduals_TIB__Layer__1_Y`
* `SiStrip/MechanicalView/TIB/layer_2/HitResiduals_TIB__Layer__2`
* `SiStrip/MechanicalView/TIB/layer_2/HitResiduals_TIB__Layer__2_Y`
* `SiStrip/MechanicalView/TIB/layer_3/HitResiduals_TIB__Layer__3`
* `SiStrip/MechanicalView/TIB/layer_3/HitResiduals_TIB__Layer__3_Y`
* `SiStrip/MechanicalView/TIB/layer_4/HitResiduals_TIB__Layer__4`
* `SiStrip/MechanicalView/TIB/layer_4/HitResiduals_TIB__Layer__4_Y`
* `SiStrip/MechanicalView/TID/PLUS/wheel_1/HitResiduals_TID__wheel__1`
* `SiStrip/MechanicalView/TID/PLUS/wheel_1/HitResiduals_TID__wheel__1_Y`
* `SiStrip/MechanicalView/TID/PLUS/wheel_2/HitResiduals_TID__wheel__2`
* `SiStrip/MechanicalView/TID/PLUS/wheel_2/HitResiduals_TID__wheel__2_Y`
* `SiStrip/MechanicalView/TID/PLUS/wheel_3/HitResiduals_TID__wheel__3`
* `SiStrip/MechanicalView/TID/PLUS/wheel_3/HitResiduals_TID__wheel__3_Y`
* `SiStrip/MechanicalView/TOB/layer_1/HitResiduals_TOB__Layer__1`
* `SiStrip/MechanicalView/TOB/layer_1/HitResiduals_TOB__Layer__1_Y`
* `SiStrip/MechanicalView/TOB/layer_2/HitResiduals_TOB__Layer__2`
* `SiStrip/MechanicalView/TOB/layer_2/HitResiduals_TOB__Layer__2_Y`
* `SiStrip/MechanicalView/TOB/layer_3/HitResiduals_TOB__Layer__3`
* `SiStrip/MechanicalView/TOB/layer_3/HitResiduals_TOB__Layer__3_Y`
* `SiStrip/MechanicalView/TOB/layer_4/HitResiduals_TOB__Layer__4`
* `SiStrip/MechanicalView/TOB/layer_4/HitResiduals_TOB__Layer__4_Y`
* `SiStrip/MechanicalView/TOB/layer_5/HitResiduals_TOB__Layer__5`
* `SiStrip/MechanicalView/TOB/layer_5/HitResiduals_TOB__Layer__5_Y`
* `SiStrip/MechanicalView/TOB/layer_6/HitResiduals_TOB__Layer__6`
* `SiStrip/MechanicalView/TOB/layer_6/HitResiduals_TOB__Layer__6_Y`
* `SiStrip/MechanicalView/TEC/PLUS/wheel_1/HitResiduals_TEC__wheel__1`
* `SiStrip/MechanicalView/TEC/PLUS/wheel_1/HitResiduals_TEC__wheel__1_Y`
* `SiStrip/MechanicalView/TEC/PLUS/wheel_2/HitResiduals_TEC__wheel__2`
* `SiStrip/MechanicalView/TEC/PLUS/wheel_2/HitResiduals_TEC__wheel__2_Y`
* `SiStrip/MechanicalView/TEC/PLUS/wheel_3/HitResiduals_TEC__wheel__3`
* `SiStrip/MechanicalView/TEC/PLUS/wheel_3/HitResiduals_TEC__wheel__3_Y`
* `SiStrip/MechanicalView/TEC/PLUS/wheel_4/HitResiduals_TEC__wheel__4`
* `SiStrip/MechanicalView/TEC/PLUS/wheel_4/HitResiduals_TEC__wheel__4_Y`
* `SiStrip/MechanicalView/TEC/PLUS/wheel_5/HitResiduals_TEC__wheel__5`
* `SiStrip/MechanicalView/TEC/PLUS/wheel_5/HitResiduals_TEC__wheel__5_Y`
* `SiStrip/MechanicalView/TEC/PLUS/wheel_6/HitResiduals_TEC__wheel__6`
* `SiStrip/MechanicalView/TEC/PLUS/wheel_6/HitResiduals_TEC__wheel__6_Y`
* `SiStrip/MechanicalView/TEC/PLUS/wheel_7/HitResiduals_TEC__wheel__7`
* `SiStrip/MechanicalView/TEC/PLUS/wheel_7/HitResiduals_TEC__wheel__7_Y`
* `SiStrip/MechanicalView/TEC/PLUS/wheel_8/HitResiduals_TEC__wheel__8`
* `SiStrip/MechanicalView/TEC/PLUS/wheel_8/HitResiduals_TEC__wheel__8_Y`
* `SiStrip/MechanicalView/TEC/PLUS/wheel_9/HitResiduals_TEC__wheel__9`
* `SiStrip/MechanicalView/TEC/PLUS/wheel_9/HitResiduals_TEC__wheel__9_Y`
* `SiStrip/MechanicalView/TEC/MINUS/wheel_1/HitResiduals_TEC__wheel__1`
* `SiStrip/MechanicalView/TEC/MINUS/wheel_1/HitResiduals_TEC__wheel__1_Y`
* `SiStrip/MechanicalView/TEC/MINUS/wheel_2/HitResiduals_TEC__wheel__2`
* `SiStrip/MechanicalView/TEC/MINUS/wheel_2/HitResiduals_TEC__wheel__2_Y`
* `SiStrip/MechanicalView/TEC/MINUS/wheel_3/HitResiduals_TEC__wheel__3`
* `SiStrip/MechanicalView/TEC/MINUS/wheel_3/HitResiduals_TEC__wheel__3_Y`
* `SiStrip/MechanicalView/TEC/MINUS/wheel_4/HitResiduals_TEC__wheel__4`
* `SiStrip/MechanicalView/TEC/MINUS/wheel_4/HitResiduals_TEC__wheel__4_Y`
* `SiStrip/MechanicalView/TEC/MINUS/wheel_5/HitResiduals_TEC__wheel__5`
* `SiStrip/MechanicalView/TEC/MINUS/wheel_5/HitResiduals_TEC__wheel__5_Y`
* `SiStrip/MechanicalView/TEC/MINUS/wheel_6/HitResiduals_TEC__wheel__6`
* `SiStrip/MechanicalView/TEC/MINUS/wheel_6/HitResiduals_TEC__wheel__6_Y`
* `SiStrip/MechanicalView/TEC/MINUS/wheel_7/HitResiduals_TEC__wheel__7`
* `SiStrip/MechanicalView/TEC/MINUS/wheel_7/HitResiduals_TEC__wheel__7_Y`
* `SiStrip/MechanicalView/TEC/MINUS/wheel_8/HitResiduals_TEC__wheel__8`
* `SiStrip/MechanicalView/TEC/MINUS/wheel_8/HitResiduals_TEC__wheel__8_Y`
* `SiStrip/MechanicalView/TEC/MINUS/wheel_9/HitResiduals_TEC__wheel__9`
* `SiStrip/MechanicalView/TEC/MINUS/wheel_9/HitResiduals_TEC__wheel__9_Y`
