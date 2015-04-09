9/4/2015 (Suchandra Dutta)
=================================
Bug fix in SiStripHotStripAlgorithmFromClusterOccupancy.cc. A parameter, _NEmptyBins has been initialised 
for each APV before the iterative search for 'Hot Strips' starts. _NEmptyBins counts number of 'Hot Strips' 
and gets dynamically incremented during the iterative search. It is used to calculate the  mean entries per 
bin

> long double meanVal=1.*histo._NEntries/(1.*Nbins-histo._NEmptyBins);

The problem could get traced for a very bad module with almost all strips bad for the first APV
(_NEmptyBins ~ 128). For next APVs, meanVal becomes a very large number if _NEntries is
not initialised and according no other strips are identified as 'Hot'.
 
For the PCL algorithm (SiStripBadAPVandHotStripAlgorithmFromClusterOccupancy.cc) NEmptyBins for
each APVs are initialised properly.