

dbfile=JEC_PA5TeV_CMSSW538_2013.db

version=02
comment="Preliminary jet corrections for 2013 pPb run"
since=1

for r in `seq 2 6`
do

  inputtag=JetCorrectorParametersCollection_HI_PFTowers_generalTracks_PythiaZ2_5TeV_538_AK${r}PF
  outputtag=JetCorrectorParametersCollection_HI_PFTowers_generalTracks_PythiaZ2_5TeV_538_v${version}_AK${r}PF_offline

  cat template.txt | sed "s/__INPUT__/$inputtag/g" | sed "s/__OUTPUT__/$outputtag/g" | sed "s/__SINCE__/$since/g" | sed "s/__COMMENT__/$comment/g" > $outputtag.txt
  cp $dbfile $outputtag.db

  echo "./upload.py $outputtag.db"

  inputtag=JetCorrectorParametersCollection_HI_PFTowers_PythiaZ2_5TeV_538_AK${r}Calo
  outputtag=JetCorrectorParametersCollection_HI_PythiaZ2_5TeV_538_v${version}_AK${r}Calo_offline

  cat template.txt | sed "s/__INPUT__/$inputtag/g" | sed "s/__OUTPUT__/$outputtag/g" | sed "s/__SINCE__/$since/g" | sed "s/__COMMENT__/$comment/g" > $outputtag.txt
  cp $dbfile $outputtag.db

  echo "./upload.py $outputtag.db"


done

