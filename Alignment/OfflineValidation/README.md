# Validation

We use the Boost library (Program Options, Filesystem & Property Trees) to deal with the treatment of the config file.
Basic idea:
 - a generic config file is "projected" for each validation (*e.g.* the geometry is changed, together with the plotting style);
 - for each config file, a new condor config file is produced;
 - a DAGMAN file is also produced in order to submit the whole validation at once.

In principle, the `validateAlignment` command is enough to submit everything.
However, for local testing, one may want to make a dry run: all files will be produced, but the condor jobs will not be submitted;
then one can just test locally any step, or modify any parameter before simply submitting the DAGMAN.

## TODO list 

 - improve exceptions handling (filesystem + own)
   - check inconsistencies in config file?
 - from DMR toy to real application
   - GCP (get "n-tuples" + grid, 3D, TkMaps)
   - DMRs (single + merge + trend)
   - PV (single + merge + trend)
   - Zµµ (single + merge)
   - MTS (single + merge)
   - overlap (single + merge + trend)
   - ...
 - documentation (this README)
   - tutorial
   - instructions for developers
 - details
   - copy condor config like the executable (or similar) and use soft links instead of hard copy
   - make dry and local options (i.e. just don't run any condor command)
(list from mid-January)

