# Examples

## jetHt_multiYearTrendPlot.json

This configuration shows how to plot multi-year trend plots using previously merged jetHT validation files. It uses the jetHT plotting macro standalone. Notice that the plots are saved in the output folder, and this folder has to exist for the plot saving to be successful. You can run this example using

```
jetHtPlotter jetHt_multiYearTrendPlot.json
```

## jetHt_ptHatWeightForMCPlot.json

This configuration shows how to apply ptHat weight for MC files produced with different ptHat cuts. What you need to do is to collect the file names and lower boundaries of the ptHat bins into a file, which is this case is ptHatFiles_MC2018_PFJet320.txt. For a file list like this, the ptHat weight is automatically applied by the code. The weights are correct for run2. The plotting can be done using the jetHT plotter standalone:

```
jetHtPlotter jetHt_ptHatWeightForMCPlot.json
```
