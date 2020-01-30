# Hit Resolution Project #

## Explanation for the pair method ## 
The hit resolution, $`\sigma_{hit}`$, (in centimetres) can be obtained by using the following equation:

$`\sigma_{hit} = \sqrt{\frac{\sigma^{2}_{HitDX-TrackDX} - \sigma^{2}_{TrackDXE}}{2}}`$,

where $`\sigma_{HitDX-TrackDX}`$ is the standard deviation of the gaussian-fitted distribution of the double difference between the predicted and measured hit positions, and $`\sigma_{TrackDXE}`$ is the mean of the distribution of the predicted positions. 

The following criteria are used:

* The charged particle must have traversed through two sufficiently-overlapping detector modules. 
* Tracks must have transverse momenta greater than 3 GeV ([permalink](https://gitlab.cern.ch/coldham/hitresolutionproject/-/blob/master/src/HitResol.cc#L324)).
* Track momentum is > 15 GeV for strips and > 5 GeV for pixels ([permalink](https://gitlab.cern.ch/coldham/hitresolutionproject/-/blob/master/Resolutions.cc#L243)).
* More than six hits must be left in the SST layers and the pixel detector ([permalink](https://gitlab.cern.ch/coldham/hitresolutionproject/-/blob/master/Resolutions.cc#L252)). 
* No more than four of these hits can have a large cluster width ([permalink](https://gitlab.cern.ch/coldham/hitresolutionproject/-/blob/master/Resolutions.cc#L254)).
* The track fit must yield a good $`\chi^{2}`$ probability of $`\chi^{2} \geqslant 10^{-2}`$ ([permalink](https://gitlab.cern.ch/coldham/hitresolutionproject/-/blob/master/Resolutions.cc#L251)).
* The predicted path (the distance of propagation of the charged particle from one layer to the next) must be < 7cm in the TIB and TOB ([permalink](https://gitlab.cern.ch/coldham/hitresolutionproject/-/blob/master/Resolutions.cc#L238)).
* The predicted path must be < 2cm for pixels([permalink](https://gitlab.cern.ch/coldham/hitresolutionproject/-/blob/master/Resolutions.cc#L239)).
* The error in the distance between two hit resolution measurements must be < 0.0025 ([permalink](https://gitlab.cern.ch/coldham/hitresolutionproject/-/blob/master/Resolutions.cc#L235)).
* Clusters in the silicon module under consideration must have the same width ([permalink](https://gitlab.cern.ch/coldham/hitresolutionproject/-/blob/master/Resolutions.cc#L254)).
* Clusters positioned on the edge of modules are not considered ([permalink](https://gitlab.cern.ch/coldham/hitresolutionproject/-/blob/master/src/HitResol.cc#L346))

Hit resolution measurements are obtained for each tracker region; the definitions for each region are based on those given under the "Phase 1 Upgrade Detector DetId schema" heading of the README file in the github repository linked [here](https://github.com/cms-sw/cmssw/blob/master/Geometry/TrackerNumberingBuilder/README.md). The permalinks to where these definitions are implemented are as follows:

* [TIB L1](https://gitlab.cern.ch/coldham/hitresolutionproject/-/blob/master/Resolutions.cc#L100)
* [TIB L2](https://gitlab.cern.ch/coldham/hitresolutionproject/-/blob/master/Resolutions.cc#L104)
* [TIB L3](https://gitlab.cern.ch/coldham/hitresolutionproject/-/blob/master/Resolutions.cc#L108)
* [TIB L4](https://gitlab.cern.ch/coldham/hitresolutionproject/-/blob/master/Resolutions.cc#L112)
* [TID Side](https://gitlab.cern.ch/coldham/hitresolutionproject/-/blob/master/Resolutions.cc#L117)
* [TID Wheel](https://gitlab.cern.ch/coldham/hitresolutionproject/-/blob/master/Resolutions.cc#L122)
* [TID Ring](https://gitlab.cern.ch/coldham/hitresolutionproject/-/blob/master/Resolutions.cc#L126)
* [TOB L1](https://gitlab.cern.ch/coldham/hitresolutionproject/-/blob/master/Resolutions.cc#L131)
* [TOB L2](https://gitlab.cern.ch/coldham/hitresolutionproject/-/blob/master/Resolutions.cc#L135)
* [TOB L3](https://gitlab.cern.ch/coldham/hitresolutionproject/-/blob/master/Resolutions.cc#L139)
* [TOB L4](https://gitlab.cern.ch/coldham/hitresolutionproject/-/blob/master/Resolutions.cc#L143)
* [TOB L5](https://gitlab.cern.ch/coldham/hitresolutionproject/-/blob/master/Resolutions.cc#L147)
* [TOB L6](https://gitlab.cern.ch/coldham/hitresolutionproject/-/blob/master/Resolutions.cc#L151)
* [TEC Side](https://gitlab.cern.ch/coldham/hitresolutionproject/-/blob/master/Resolutions.cc#L156)
* [TEC Wheel](https://gitlab.cern.ch/coldham/hitresolutionproject/-/blob/master/Resolutions.cc#L160)
* [TEC Ring](https://gitlab.cern.ch/coldham/hitresolutionproject/-/blob/master/Resolutions.cc#L163)
* [All TIB](https://gitlab.cern.ch/coldham/hitresolutionproject/-/blob/master/Resolutions.cc#L167)
* [All TOB](https://gitlab.cern.ch/coldham/hitresolutionproject/-/blob/master/Resolutions.cc#L171)
* [All TID](https://gitlab.cern.ch/coldham/hitresolutionproject/-/blob/master/Resolutions.cc#L175)
* [All TEC](https://gitlab.cern.ch/coldham/hitresolutionproject/-/blob/master/Resolutions.cc#L188)
* [Pixel barrel](https://gitlab.cern.ch/coldham/hitresolutionproject/-/blob/master/Resolutions.cc#L204)
* [Pixel endcap disk](https://gitlab.cern.ch/coldham/hitresolutionproject/-/blob/master/Resolutions.cc#L207)

## How to run the code ##
The following scripts are used for the hit resolution study:
* [src/HitResol.cc](https://gitlab.cern.ch/coldham/hitresolutionproject/-/blob/master/src/HitResol.cc) (original author: Denis Gele)
* [test/SiStripHitResol_testULcosmics.py](https://gitlab.cern.ch/coldham/hitresolutionproject/-/blob/master/test/SiStripHitResol_testULcosmics.py) (original author: Pieter David)
* [Resolutions.cc](https://gitlab.cern.ch/coldham/hitresolutionproject/-/blob/master/Resolutions.cc) (original author: Kathryn Coldham)
* [ResolutionPlots.cc](https://gitlab.cern.ch/coldham/hitresolutionproject/-/blob/master/ResolutionPlots.cc) (original author: Kathryn Coldham)

The scripts should be run inside CMSSW_11_1_0_pre6. Firstly, the following commands should be used:

* cd CMSSW_11_1_0_pre6/src
* cmsenv
* cd UserCode/SiStripHitResolution

Next, to specify between running over UL or non-UL samples, the integer [UL](https://gitlab.cern.ch/coldham/hitresolutionproject/-/blob/master/test/SiStripHitResol_testULcosmics.py#L11) must be set as 0 (for non-UL) or 1 (for UL). Then, compile and run the python config file using the following commands:

* scram b
* cmsRun test/SiStripHitResol_testULcosmics.py

This will produce an output root file, which will be called hitresol_ALCARECO_UL.root (for when ultra legacy samples have been used as input) or hitresol_ALCARECO.root (for ALCARECO, non-ultra legacy samples). This root file is then used as input to the macro Resolutions.cc, for which the following command should be used:

* root -l Resolutions.cc

This will produce three output directories: GaussianFits, HitResolutionValues and CutFlowReports. 

* [GaussianFits](https://gitlab.cern.ch/coldham/hitresolutionproject/-/tree/master/CutFlowReports) will contain the root files GaussianFits_Centimetres.root or GaussianFits_PitchUnits.root.
* [HitResolutionValues](https://gitlab.cern.ch/coldham/hitresolutionproject/-/tree/master/HitResolutionValues) will contain the text files HitResolutionValues_Centimetres.txt or HitResolutionValues_PitchUnits.txt. 
* [CutFlowReports](https://gitlab.cern.ch/coldham/hitresolutionproject/-/tree/master/CutFlowReports) will contain text files with the numbers of events that pass each filter, for each region.

To choose between pitch units or centimetres, [UnitInteger](https://gitlab.cern.ch/coldham/hitresolutionproject/-/blob/master/Resolutions.cc#L330) must be set as 0 (for pitch units) or 1 (for centimetres). To choose between using output file names that correspond to the UL or non-UL ALCARECO samples, the value of [ULInteger](https://gitlab.cern.ch/coldham/hitresolutionproject/-/blob/master/Resolutions.cc#L331) must be set as 0 (for non-UL samples) or 1 (for UL samples). 

The plotting script [ResolutionPlots.cc](https://gitlab.cern.ch/coldham/hitresolutionproject/-/blob/master/ResolutionPlots.cc) has been provided for comparing the hit resolution values for UL and non-UL samples. However, the elements of the arrays will need to be edited ([UL_Array](https://gitlab.cern.ch/coldham/hitresolutionproject/-/blob/master/ResolutionPlots.cc#L20) for when the unit has been specified as centimetres, [NonUL_Array](https://gitlab.cern.ch/coldham/hitresolutionproject/-/blob/master/ResolutionPlots.cc#L25) for centimetres, [UL_Array](https://gitlab.cern.ch/coldham/hitresolutionproject/-/blob/master/ResolutionPlots.cc#L40) for pitch units and [NonUL_Array](https://gitlab.cern.ch/coldham/hitresolutionproject/-/blob/master/ResolutionPlots.cc#L43) for pitch units). The following command should then be used:

* root -l ResolutionPlots.cc

This will produce two output files: ResolutionComparison_ULAndNonUL_Centimetres.pdf and ResolutionComparison_ULAndNonUL_Centimetres.root (for results in centimetres), and ResolutionComparison_ULAndNonUL_PitchUnits.pdf and ResolutionComparison_ULAndNonUL_PitchUnits.root (for results in pitch units).  


## References ##

* [Butz, E. (2017). Operation and Performance of the CMS Outer Tracker. The Compact Muon Solenoid Experiment Conference Report. 2017/428.](https://gitlab.cern.ch/coldham/hitresolutionproject/-/blob/master/Resources/CR2017_428.pdf)
* [Chiochia, V et al. (2005). Track reconstruction in the CMS tracker. CMS Analysis Note. CMS AN 2005/063.](https://gitlab.cern.ch/coldham/hitresolutionproject/-/blob/master/Resources/AN2005_063_v2.pdf)
* [Gelé, D. (2018). Status of SiStripHitResolution studies.](https://gitlab.cern.ch/coldham/hitresolutionproject/-/blob/master/Resources/PresentationTracker_17072018.pdf)
* [Ulmer, K. (2009). Tracker Hit Resolution Measurements in 2009 CRAFT Data.](https://gitlab.cern.ch/coldham/hitresolutionproject/-/blob/master/Resources/HitResLPCPhysics_10_15_09.ppt) 
* [McColl, N. (2013). RecHit Resolution Methods.](https://indico.cern.ch/event/305395/contributions/701396/attachments/580300/798934/nmccoll_3_13_RecHitRes.pdf)
