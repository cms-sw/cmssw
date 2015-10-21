#!/bin/bash

export SAMPLEDIR="/lustre/cms/store/user/piet/ME0Segment_Time/AnalysisDavid/DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola/"

# export SAMPLES="crab_DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_100ps_1p5ns_Tight_Analysis_David/151018_225436"

export SAMPLES="crab_DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_100ps_1p5ns_Loose_Analysis_David/151018_225724 
                crab_DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_50ps_1ns_Loose_Analysis_David/151018_225649 
                crab_DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_5ns_32ns_Tight_Analysis_David/151018_225630 
                crab_DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_50ps_1ns_Tight_Analysis_David/151018_225352 
                crab_DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_300ps_3ns_Tight_Analysis_David/151018_225505 
                crab_DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_300ps_3ns_Loose_Analysis_David/151018_225738 
                crab_DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_500ps_4ns_Loose_Analysis_David/151018_225751 
                crab_DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_1ns_7ns_Tight_Analysis_David/151018_225542 
                crab_DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_1ns_7ns_Loose_Analysis_David/151018_225804 
                crab_DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_500ps_4ns_Tight_Analysis_David/151018_225523 
                crab_DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_5ns_32ns_Loose_Analysis_David/151018_225818"



for i in ${SAMPLES} 
do
  export SAMPLE=`echo ${i} | awk -F "/" '{print $1}'`
  export SUBMITDATE=`echo ${i} | awk -F "/" '{print $2}'`
  echo $SAMPLE 
  echo $SUBMITDATE

  ### Make Directory on LUSTRE where to perform the merging
  echo "mkdir /lustre/home/piet/RunOnNode/ME0Analysis/ME0AnalysisDavid/Merge_${SAMPLE}/"
  mkdir /lustre/home/piet/RunOnNode/ME0Analysis/ME0AnalysisDavid/Merge_${SAMPLE}/
  lt /lustre/home/piet/RunOnNode/ME0Analysis/ME0AnalysisDavid/
  # chmod a+xw /lustre/home/piet/RunOnNode/ME0Analysis/ME0AnalysisDavid/Merge_${SAMPLE}/
  ### Make the MergeList
  touch MergeList_${SAMPLE}.txt
  lt ${SAMPLEDIR}/${SAMPLE}/${SUBMITDATE}/0000/ | grep ME0 | awk '{print $9}' &> MergeList_${SAMPLE}.txt
  ### Prepare the Submission file
  cp MergeHistos_Template.sh MergeHistos_${SAMPLE}.sh
  sed -i "s/SAMPLE/${SAMPLE}/"         MergeHistos_${SAMPLE}.sh # replace SAMPLE by SAMPLE            !!! replaces the word SAMPLE only once per line !!!
  sed -i "s/SUBMITDATE/${SUBMITDATE}/" MergeHistos_${SAMPLE}.sh # replace SUBMITDATE by SUBMITDATE
  chmod a+x MergeHistos_${SAMPLE}.sh
  ### Submit
  # qsub -l nodes=1:ppn=1 -q local -e MergeHistos_${SAMPLE}.err -o MergeHistos_${SAMPLE}.out MergeHistos_${SAMPLE}.sh"
  # qsub -l nodes=1:ppn=8 -q local -e MergeHistos_${SAMPLE}.err -o MergeHistos_${SAMPLE}.out MergeHistos_${SAMPLE}.sh"
  echo "qsub -q local -e MergeHistos_${SAMPLE}.err -o MergeHistos_${SAMPLE}.out MergeHistos_${SAMPLE}.sh"
  # qsub -q local -e MergeHistos_${SAMPLE}.err -o MergeHistos_${SAMPLE}.out MergeHistos_${SAMPLE}.sh

done
