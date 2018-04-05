#!/bin/bash



dirbase="$CMSSW_BASE/src/Alignment/APEEstimation/hists/workingArea"


for folder in ${dirbase}/*;
do
  plots="${folder}/plots"
  
  if [ -d "$plots" ]; then
    
    ## FIXME: make script configurable for output types? or create automatically .png's from .eps? ROOT .png's are often incorrect
    
    #rm ${plots}/*.png
    #rm ${plots}/*.eps
    
    #for file in ${plots}/*.eps;
    #do
    #  epstopdf $file
    #done
    
    
    mkdir ${plots}/Result/
    mkdir ${plots}/Sector/
    mkdir ${plots}/Track/
    
    mv ${plots}/ape_*.* ${plots}/Result/.
    mv ${plots}/correction_*.* ${plots}/Result/.
    mv ${plots}/result_*.* ${plots}/Result/.
    
    mv ${plots}/h_NorRes*.* ${plots}/Sector/.
    mv ${plots}/h_Res*.* ${plots}/Sector/.
    mv ${plots}/h_entries*.* ${plots}/Sector/.
    mv ${plots}/h_residualWidth*.* ${plots}/Sector/.
    mv ${plots}/h_rms*.* ${plots}/Sector/.
    mv ${plots}/h_weight*.* ${plots}/Sector/.
    
    #mkdir ${plots}/Sector2/
    #mv ${plots}/h_BaryStrip*.* ${plots}/Sector2/.
    #mv ${plots}/h_ChargeOnEdges*.* ${plots}/Sector2/.
    #mv ${plots}/h_ChargePixel*.* ${plots}/Sector2/.
    #mv ${plots}/h_ChargeStrip*.* ${plots}/Sector2/.
    #mv ${plots}/h_ClusterProbXY*.* ${plots}/Sector2/.
    #mv ${plots}/h_LogClusterProb*.* ${plots}/Sector2/.
    #mv ${plots}/h_PhiSens*.* ${plots}/Sector2/.
    #mv ${plots}/h_SOverN*.* ${plots}/Sector2/.
    #mv ${plots}/h_Width*.* ${plots}/Sector2/.
    #mv ${plots}/h_sigma*.* ${plots}/Sector2/.
    #mv ${plots}/p_phiSens*.* ${plots}/Sector2/.
    #mv ${plots}/p_sigma*.* ${plots}/Sector2/.
    #mv ${plots}/p_width*.* ${plots}/Sector2/.
    
    mv ${plots}/h_charge.* ${plots}/Track/.
    mv ${plots}/h_d0Beamspot.* ${plots}/Track/.
    mv ${plots}/h_d0BeamspotErr.* ${plots}/Track/.
    mv ${plots}/h_d0BeamspotSig.* ${plots}/Track/.
    mv ${plots}/h_dz.* ${plots}/Track/.
    mv ${plots}/h_dzErr.* ${plots}/Track/.
    mv ${plots}/h_dzSig.* ${plots}/Track/.
    mv ${plots}/h_eta.* ${plots}/Track/.
    mv ${plots}/h_etaErr.* ${plots}/Track/.
    mv ${plots}/h_etaSig.* ${plots}/Track/.
    mv ${plots}/h_norChi2.* ${plots}/Track/.
    mv ${plots}/h_p.* ${plots}/Track/.
    mv ${plots}/h_prob.* ${plots}/Track/.
    mv ${plots}/h_phi.* ${plots}/Track/.
    mv ${plots}/h_phiErr.* ${plots}/Track/.
    mv ${plots}/h_phiSig.* ${plots}/Track/.
    mv ${plots}/h_pt.* ${plots}/Track/.
    mv ${plots}/h_ptErr.* ${plots}/Track/.
    mv ${plots}/h_ptSig.* ${plots}/Track/.
    mv ${plots}/h_theta.* ${plots}/Track/.
    mv ${plots}/h_trackSizeGood.* ${plots}/Track/.
    mv ${plots}/h_hitsPixel.* ${plots}/Track/.
    mv ${plots}/h_hitsStrip.* ${plots}/Track/.
    
  fi

done



