## CondTools/SiStrip

- _SiStripApvGainReader_: reads APV gains from input sqlite. Dumps TTree with following infos: detectorID, APV number, gain
                        Infos are also dumped in txt file, in format : DetId APV1 APV2 APV3 APV4 (APV5) (APV6)
  - To run: 
    - `cmsRun test/SiStripApvGainReader_cfg.py inputFiles=sqlite_input  tag=my_tag runN=run_in_IOV`
    - `readSiStripApvGain.py  inputFiles=sqlite_input  tag=my_tag runN=run_in_IOV`

- _SiStripChannelGainFromDBMiscalibrator_: reads APV gains (either G1 or G2) from DB and applies hierarchically a scale factor and/or gaussian smearing for each APV gain, down to the individual layer or disk level
  - To run: 
    - `cmsRun test/SiStripChannelGainFromDBMiscalibrator_cfg.py globalTag=<inputGT> runNumber=<inputIOV>`
  
- _SiStripNoisesFromDBMiscalibrator_: reads Noise from DB and applies hierarchically a scale factor and/or gaussian smearing for each APV gain, down to the individual layer or disk level
  - To run: 
    - `cmsRun test/SiStripNoiseFromDBMiscalibrator_cfg.py globalTag=<inputGT> runNumber=<inputIOV>`

- _SiStripApvGainFromFileBuilder_:
The SiStripApvGainFromFileBuilder class is an analyzer that reads an ASCII file
containing the APV gain scan at tickmarks and creates the corresponding payload
in the offline Database. For each APV the tickmark file is expected to have one
line with the following data:

| offline detector id | online APV id | value of the gain scan |    
| ------------------- | ------------- | ---------------------- |

An example of tickmark file can be found in the data directory of this packages.
The payload for the offline database requires to convert the online APV ids into
the offline ones. For this conversion the detector cabling `SiStripDetCabling`
and the reader of the ideal geometry `SiStripDetInfoFileReader` are used. The 
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
channels and the special channel. The code can also produces ASCII files with the
gain scan for the APVs to be given in input to a the tracker map. 

A special attention must be reserved to the online to offline conversion of the 
APV ids. The detector modules can have 3 APV pairs or 2 APV pair and the logic of
the conversion is different according to the module type. The conversion logic is
listed in the table below:

##### Modules with 6 APVs

| online APV id  | offline APV id |    
| -------------- | -------------- |
| 0 | 0 |
| 1 | 1 |
| 2 | 2 |
| 3 | 3 |
| 4 | 4 |
| 5 | 5 |

##### Modules with 4 APVs

| online APV id | offline APV id |
| ------------- | -------------- |
| 0 | 0 |
| 1 | 1 |
| 4 | 2 |
| 5 | 3 |

This logic has been implemented inside the SiStripApvGainFromFileBuilder class,
it cannot be deducted from the SiStrip cabling description code.

##### Job Configuration
The job to read the ASCII tickmark file and deploy the payload into the offline
database is configured with the `SiStripApvGainFromASCIIFile_cfg.py` fragment put
in the test directory. The `SiStripGainApvFromFileBuilder` analyzer is configured
with the following options:

| Options | Function | Default |
| ------- | -------- | ------- |
| `tickFile` | Path to the tickmark scan | `CondTools/SiStrip/data/tickheight.txt` |
| `gainThreshold` | Lower limit for good scan value | 0. |
| `dummyAPVGain` | Dummy value for the APV gain | 690./640. |
| `putDummyIntoUncabled` | Switch to put dummy gain for uncabled channels | False |
| `putDummyIntoUnscanned` | Switch to put dummy gain for unscanned channels | False |
| `putDummyIntoOffChannels` | Switch to put dummy gain for OFF channels | False |
| `putDummyIntoBadChannels` | Switch to put dummy gain for BAD channels | False |
| `outputMaps` | Switch to output the ASCII text for the tracker map | False |
| `outputSummary` | Switch to output the summary text file | False |
