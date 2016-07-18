### Purpose

This code generates the `HitResiduals` and `NormalizedHitResiduals` histograms for SiStip and SiPixel. The source file `MonitorTrackResiduals.cc` actually implements two modules (using a template parameter) from the same code, such that the SiStrip and SiPixel configurations can be handled independently. The disadvantage of this is that every track has to be handled twice (which might hurt performance).

### Data displayed

The data shown in the histograms is computed by the Alignment/OfflineValidation code. It is computed for SiPixel and SiStrip in x- and y-direction (resXprime, resYprime values), however for the strips only the x-residual is relevant. Simple cuts on pT and the significance of dxy are applied to only select good tracks. Additionally, a trigger can be applied. 

### Options

`MonitorTrackResiduals` (for SiStrip) and `SiPixelMonitorTrackResiduals` (for SiPixel) accept the same options.

* `Mod_On` setting this to `True` will generate plots for every module. By default, only one plot per layer and wheel/disk is generated.
* `Tracks` the sort of tracks to be used. (Note that `TrackerValidationVariables.cc` uses `trajcetoryInput` as well, but this should not affect this module.)
* The usual parameters of `GenericTriggerEventFlag`.

### Currently generated histograms

* `Pixel/Barrel/HitResidualsX_L1`
* `Pixel/Barrel/HitResidualsX_L2`
* `Pixel/Barrel/HitResidualsX_L3`
* `Pixel/Barrel/HitResidualsY_L1`
* `Pixel/Barrel/HitResidualsY_L2`
* `Pixel/Barrel/HitResidualsY_L3`
* `Pixel/Endcap/HitResidualsX_Dm1`
* `Pixel/Endcap/HitResidualsX_Dm2`
* `Pixel/Endcap/HitResidualsX_Dp1`
* `Pixel/Endcap/HitResidualsX_Dp2`
* `Pixel/Endcap/HitResidualsY_Dm1`
* `Pixel/Endcap/HitResidualsY_Dm2`
* `Pixel/Endcap/HitResidualsY_Dp1`
* `Pixel/Endcap/HitResidualsY_Dp2`
* `SiStrip/MechanicalView/TEC/MINUS/wheel_1/HitResiduals_TEC__wheel__1`
* `SiStrip/MechanicalView/TEC/MINUS/wheel_2/HitResiduals_TEC__wheel__2`
* `SiStrip/MechanicalView/TEC/MINUS/wheel_3/HitResiduals_TEC__wheel__3`
* `SiStrip/MechanicalView/TEC/MINUS/wheel_4/HitResiduals_TEC__wheel__4`
* `SiStrip/MechanicalView/TEC/MINUS/wheel_5/HitResiduals_TEC__wheel__5`
* `SiStrip/MechanicalView/TEC/MINUS/wheel_6/HitResiduals_TEC__wheel__6`
* `SiStrip/MechanicalView/TEC/MINUS/wheel_7/HitResiduals_TEC__wheel__7`
* `SiStrip/MechanicalView/TEC/MINUS/wheel_8/HitResiduals_TEC__wheel__8`
* `SiStrip/MechanicalView/TEC/MINUS/wheel_9/HitResiduals_TEC__wheel__9`
* `SiStrip/MechanicalView/TEC/PLUS/wheel_1/HitResiduals_TEC__wheel__1`
* `SiStrip/MechanicalView/TEC/PLUS/wheel_2/HitResiduals_TEC__wheel__2`
* `SiStrip/MechanicalView/TEC/PLUS/wheel_3/HitResiduals_TEC__wheel__3`
* `SiStrip/MechanicalView/TEC/PLUS/wheel_4/HitResiduals_TEC__wheel__4`
* `SiStrip/MechanicalView/TEC/PLUS/wheel_5/HitResiduals_TEC__wheel__5`
* `SiStrip/MechanicalView/TEC/PLUS/wheel_6/HitResiduals_TEC__wheel__6`
* `SiStrip/MechanicalView/TEC/PLUS/wheel_7/HitResiduals_TEC__wheel__7`
* `SiStrip/MechanicalView/TEC/PLUS/wheel_8/HitResiduals_TEC__wheel__8`
* `SiStrip/MechanicalView/TEC/PLUS/wheel_9/HitResiduals_TEC__wheel__9`
* `SiStrip/MechanicalView/TIB/layer_1/HitResiduals_TIB__Layer__1`
* `SiStrip/MechanicalView/TIB/layer_2/HitResiduals_TIB__Layer__2`
* `SiStrip/MechanicalView/TIB/layer_3/HitResiduals_TIB__Layer__3`
* `SiStrip/MechanicalView/TIB/layer_4/HitResiduals_TIB__Layer__4`
* `SiStrip/MechanicalView/TID/PLUS/wheel_1/HitResiduals_TID__wheel__1`
* `SiStrip/MechanicalView/TID/PLUS/wheel_2/HitResiduals_TID__wheel__2`
* `SiStrip/MechanicalView/TID/PLUS/wheel_3/HitResiduals_TID__wheel__3`
* `SiStrip/MechanicalView/TID/MINUS/wheel_1/HitResiduals_TID__wheel__1`
* `SiStrip/MechanicalView/TID/MINUS/wheel_2/HitResiduals_TID__wheel__2`
* `SiStrip/MechanicalView/TID/MINUS/wheel_3/HitResiduals_TID__wheel__3`
* `SiStrip/MechanicalView/TOB/layer_1/HitResiduals_TOB__Layer__1`
* `SiStrip/MechanicalView/TOB/layer_2/HitResiduals_TOB__Layer__2`
* `SiStrip/MechanicalView/TOB/layer_3/HitResiduals_TOB__Layer__3`
* `SiStrip/MechanicalView/TOB/layer_4/HitResiduals_TOB__Layer__4`
* `SiStrip/MechanicalView/TOB/layer_5/HitResiduals_TOB__Layer__5`
* `SiStrip/MechanicalView/TOB/layer_6/HitResiduals_TOB__Layer__6`
