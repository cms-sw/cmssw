# DMRtrends
Tool for retrieving and plotting the trend of the distribution of medians of residuals (DMRs) and the RMS of the normalized residuals (DrmsNR) along a period of data taking.

Instructions for running the macros:

The macro can be executed in two ways:

1) From command line: "DMRtrends IOVlist Variables labels Year pathtoDMRs geometriesandcolourspairs outputdirectory pixelupdatelist showpixelupdate showlumi FORCE"
2) By editing the parameters directly in the main() function in DMRtrends.cc and executing the macro without arguments

List of arguments:
- IOVlist:                 string containing the list of IOVs separated by a ","
- Variables:               string containing the list of variables whose distribution is to be plotted i.e. median,DrmsNR
- labels:                  string containing labels that must be part of the input files separated by a ","
- Year:                    string containing the year of the studied runs (needed to retrieve the lumi-per-run file)
- pathtoDMRs:              string containing the path to the directory where the DMRs are stored
- geometrieandcolours:     string containing the list of geometries and colors in the following way name1:color1,name2:color2 etc.
- outputdirectory:         string containing the output directory for the plots
- pixelupdatelist:         string containing the list of pixelupdates separated by a ","
- showpixelupdate:         boolean, if set to true will allow to plot vertical lines in the canvas corresponding to the pixel updates
- showlumi:                boolean, if set to false the trends will be presented in function of the run (IOV) number, if set to true the integrated luminosity is used on the x axis
- FORCE:                   boolean, if set to true the plots will be made regardless of possible errors. 

In both cases the macro needs to be compiled with "scram b", a BuildFile is already provided for that.


Please note:
- If some DMRs haven't been computed correctly the macro will stop. The FORCE boolean has to be set to true to run regardless of these errors. In this case the trends will appear with "holes", as some points are not plotted. Warnings and errors are still present in the output.
- If the showlumi boolean is set to true the macro will try to retrieve the lumi-per-run txt files provided in the Alignment/OfflineValidation/data directory. The file names are currently hardcoded, and defined in the lumifileperyear() function.
- The Year string is currently used to identify the correct lumi-per-run files to read and to select the High Level Structures and the number of layers/disks per structure

For questions regarding the tool please contact: andrea.cardini@desy.de
