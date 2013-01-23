input=out/tables_Hijing_d20130119_v4.root
output=pPb_Hijing_CentralityTable_HFplus100.db
tag=CentralityTable_HFtowersPlusTrunc_Hijing_v4_mc
cmsRun makeDBFromTFile.py outputTag=$tag inputFile=$input outputFile=$output

input=out/tables_Hijing_d20130119_v4.root
output=pPb_Hijing_CentralityTable_Tracks100.db
tag=CentralityTable_Tracks_Hijing_v4_mc
cmsRun makeDBFromTFile.py outputTag=$tag inputFile=$input outputFile=$output
