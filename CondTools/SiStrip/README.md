#SiStripApvGainFromFileBuilder
The SiStripApvGainFromFileBuilder class is an analyzer that reads an ASCII file
containing the APV gain scan at tickmarks and creates the corresponding payload
in the offline Database. For each APV the tickmark file is expected to have one
line with the following data:

---------------------------------------------------------------
| offline detector id | online APV id | value of the gain scan|    
---------------------------------------------------------------

An example of tickmark file can be found in the data directory of this packages.
The payload for the offline database requires to convert the online APV ids into
the offline ones. For this conversion the detector cabling - SiStripDetCabling -
and the reader of the ideal geometry - SiStripDetInfoFileReader - are used. The 
former provides the APV connectivity into the FEDs and into the detector modules;
the latter lists the full set of detector modules even those not actually cabled
in the detector. The code loops over all the possible detector modules, finds
the connected ones and associates the gain scan to the corresponding APV in the
module. The uncabled modules, the channels missing in the scan, the channels in
the scan appearing as uncabled and the channels off (giving a zero tickmark gain)
or bad (giving a negative tickmark gain) are treated in a special way. According
on the job configuration either a dummy gain value or a zero gain value is put in
the offline database for these channels. At the end of the job, it is possible to
dump the summary of the database insertion into ASCII files for both the regular
channels and the special channel. The code can also produce ASCII files with the
gain scan for the APVs to be given in input to a the tracker map. 

A special attention must be reserved to the online to offline conversion of the 
APV ids. The detector modules can have 3 APV pairs or 2 APV pair and the logic of
the conversion is different according to the module type. The conversion logic is
listed in the table below:

###Modules with 6 APVs

| online APV id | offline APV id |    
----------------------------------
| 0 | 0 |
| 1 | 1 |
| 2 | 2 |
| 3 | 3 |
| 4 | 4 |
| 5 | 5 |

###Modules with 4 APVs

| online APV id | offline APV id |
----------------------------------
| 0 | 0 |
| 1 | 1 |
| 4 | 2 |
| 5 | 3 |

