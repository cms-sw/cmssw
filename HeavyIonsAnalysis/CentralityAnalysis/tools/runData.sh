
datadir=../data

cp $CMSSW_BASE/src/CmsHi/JulyExercise/data/CentralityTables.root $datadir/

####eff=90.2
for binning in 40 20 10 5
do
  root -b -q makeDataCentralityTable.C+\(${binning},\"hf\",\"HFhits${binning}_DataJulyExercise_Hydjet2760GeV_MC_37Y_V5_NZS_v0\",\"HFhits${binning}_MC_Hydjet2760GeV_MC_3XY_V24_v0\",1\)
done










