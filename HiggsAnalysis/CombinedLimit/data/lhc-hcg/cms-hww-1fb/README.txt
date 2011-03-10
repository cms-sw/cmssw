To create default datacards (log-normal for multiplicative uncertainties and gammas for statistics)
   python ../../hww/errorMatrix2Lands_multiChannel.py -o -l hww1fb -L --nch 4 hww1fb-mva.txt -g
To create datacards with all log-normals:
   python ../../hww/errorMatrix2Lands_multiChannel.py -o -l hww1fb-allLogNormal -L --nch 4 hww1fb-mva.txt

To create HLF files
    P=""; for M in $(cat masses); do python ../../../python/lands2hlf.py hww1fb$P-mH$M.txt > hww1fb$P-mH$M.hlf; done
To create workspaces:
    P=""; for M in $(cat masses); do combine --saveWorkspace -m $M -n HWW$P hww1fb$P-mH$M.txt ; done

